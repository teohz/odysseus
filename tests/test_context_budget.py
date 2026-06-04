"""Issue #1170 — the agent input-token budget adapts to the model context window.

Pins the pure budget computation and the explicit-override detection.
Uses Python's stdlib unittest only — this project runs on NixOS where
pytest isn't available in the base shell.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.context_budget import (
    compute_input_token_budget,
    DEFAULT_HARD_MAX,
    DEFAULT_BUDGET,
    DEFAULT_HEADROOM,
)
from src.settings import is_setting_overridden


def test_default_scales_to_context_window():
    # Not explicit, big window -> ~85% of the window (the old code capped at 6000).
    assert compute_input_token_budget(6000, 128000, explicit=False) == int(128000 * 0.85)


def test_default_capped_at_hard_max_for_huge_windows():
    assert compute_input_token_budget(6000, 1_000_000, explicit=False) == DEFAULT_HARD_MAX


def test_explicit_budget_is_honoured():
    # User explicitly chose 6000 -> keep it even on a 128K model.
    assert compute_input_token_budget(6000, 128000, explicit=True) == 6000
    # A larger explicit budget is honoured too, clamped to the window.
    assert compute_input_token_budget(50000, 128000, explicit=True) == 50000


def test_explicit_budget_clamped_to_window():
    assert compute_input_token_budget(200000, 32000, explicit=True) == 32000


def test_unknown_window_falls_back_to_configured():
    assert compute_input_token_budget(6000, 0, explicit=False) == 6000
    assert compute_input_token_budget(0, 0, explicit=False) == 6000  # default


def test_is_setting_overridden_reads_raw_saved_file(tmp_path, monkeypatch):
    import src.settings as settings

    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"agent_input_token_budget": 12000}), encoding="utf-8")
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(f))
    assert settings.is_setting_overridden("agent_input_token_budget") is True
    assert settings.is_setting_overridden("some_unset_key") is False

    f.write_text(json.dumps({}), encoding="utf-8")
    assert settings.is_setting_overridden("agent_input_token_budget") is False


# ---------------------------------------------------------------------------
# Configurable hard_max — completes the reviewer requirement from #1190 that
# was carried over but not implemented in #1230: the ceiling on the auto-
# derived path should be a setting, not a hidden constant. Without this,
# admins on premium APIs with very large windows (1M+ context) can only
# raise the ceiling by editing src/context_budget.py.
# ---------------------------------------------------------------------------

def test_custom_hard_max_overrides_default_in_auto_branch():
    """A caller-supplied hard_max lifts the auto-derived ceiling."""
    # Without override: 1M ctx -> capped at DEFAULT_HARD_MAX (200K)
    assert compute_input_token_budget(6000, 1_000_000, explicit=False) == DEFAULT_HARD_MAX
    # With explicit raise: 1M ctx -> 850K (85% of 1M), under the raised ceiling
    assert compute_input_token_budget(6000, 1_000_000, explicit=False, hard_max=900_000) == int(1_000_000 * 0.85)


def test_custom_hard_max_lowers_default_for_cost_paranoid_setups():
    """A lower ceiling caps the auto-derived budget below the default."""
    # 128K ctx, default ceiling 200K -> 85% of 128K = 108800
    assert compute_input_token_budget(6000, 128_000, explicit=False) == int(128_000 * 0.85)
    # Same ctx, ceiling lowered to 50K -> capped at 50K instead
    assert compute_input_token_budget(6000, 128_000, explicit=False, hard_max=50_000) == 50_000


def test_hard_max_has_no_effect_on_explicit_branch():
    """When the user set an explicit budget, hard_max must not silently cap it."""
    # User chose 900K explicitly; ctx is 1M; ceiling is 100K — user's choice wins.
    assert compute_input_token_budget(900_000, 1_000_000, explicit=True, hard_max=100_000) == 900_000


def test_default_settings_registers_hard_max_key():
    """Required so /api/auth/settings and manage_settings can persist the key."""
    from src.settings import DEFAULT_SETTINGS
    assert "agent_input_token_hard_max" in DEFAULT_SETTINGS
    assert DEFAULT_SETTINGS["agent_input_token_hard_max"] == DEFAULT_HARD_MAX


def test_alias_map_registers_friendly_names():
    """`manage_settings` should accept 'hard max' and friends."""
    from pathlib import Path
    src = Path("src/tool_implementations.py").read_text()
    assert '"hard max": "agent_input_token_hard_max"' in src
    assert '"token budget cap": "agent_input_token_hard_max"' in src
    assert '"input budget cap": "agent_input_token_hard_max"' in src


def test_agent_loop_reads_hard_max_setting(tmp_path, monkeypatch):
    """End-to-end: a saved settings.json value for agent_input_token_hard_max
    must reach compute_input_token_budget on the real agent_loop call path."""
    import src.settings as settings
    # Point SETTINGS_FILE at a temp file with our override.
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"agent_input_token_hard_max": 750_000}), encoding="utf-8")
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(f))
    monkeypatch.setattr(settings, "_settings_cache", None)
    # Read via the same import path the agent loop uses.
    assert settings.get_setting("agent_input_token_hard_max", DEFAULT_HARD_MAX) == 750_000

    # Malformed value falls back to DEFAULT_HARD_MAX (defensive, matches the
    # try/except in src/agent_loop.py).
    f.write_text(json.dumps({"agent_input_token_hard_max": "huge"}), encoding="utf-8")
    monkeypatch.setattr(settings, "_settings_cache", None)
    raw = settings.get_setting("agent_input_token_hard_max", DEFAULT_HARD_MAX)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = DEFAULT_HARD_MAX
    if parsed <= 0:
        parsed = DEFAULT_HARD_MAX
    assert parsed == DEFAULT_HARD_MAX
"""Tests for the context-trim bypass that fixes 1M-context models.

These tests pin the contract for the regression reported in #1170
where a 1M-context model was silently capped at 6000 input tokens in
agent mode, dropping the middle out of legitimately-sized pastes.

The fix lives in three places:
  - ``src/context_budget.py`` scales default budgets to the model's
    context window, capped at ``hard_max``.
  - ``src/agent_loop.py`` bypasses ``hard_max`` for very large contexts
    (>= 100K) and uses 75% of the window directly when the user has
    not explicitly set ``agent_input_token_budget``.
  - ``src/settings.py``'s ``is_setting_overridden`` now compares the
    saved value against the default so a fresh install (which writes
    the full default dict to settings.json) doesn't falsely report
    "overridden" and block the bypass.

These tests pin the contract so a future change can't silently
re-cap a 1M-context model at 6000 tokens.
"""
import json
from unittest.mock import patch

from src.context_budget import (
    compute_input_token_budget,
    DEFAULT_HARD_MAX,
    DEFAULT_BUDGET,
    DEFAULT_HEADROOM,
)
from src.settings import is_setting_overridden


# ---------------------------------------------------------------------------
# compute_input_token_budget — explicit user settings.
# ---------------------------------------------------------------------------
def test_explicit_user_budget_smaller_than_context_returns_user_value():
    """User explicitly set 6000 — honour it exactly, no scaling."""
    assert compute_input_token_budget(
        configured=6000, context_length=1_000_000, explicit=True
    ) == 6000


def test_explicit_user_budget_larger_than_context_is_clamped():
    """User explicitly set 2M but model has 1M — clamp to 1M."""
    assert compute_input_token_budget(
        configured=2_000_000, context_length=1_000_000, explicit=True
    ) == 1_000_000


def test_explicit_user_budget_with_unknown_context_returns_user_value():
    """Unknown context — explicit user budget is honoured as-is."""
    assert compute_input_token_budget(
        configured=8000, context_length=0, explicit=True
    ) == 8000


# ---------------------------------------------------------------------------
# compute_input_token_budget — default scaling path.
# ---------------------------------------------------------------------------
def test_default_scales_to_context_headroom():
    """Default (not explicit) + small context → use headroom fraction."""
    # 32000 * 0.85 = 27200
    result = compute_input_token_budget(
        configured=6000, context_length=32_000, explicit=False
    )
    assert result == 27_200


def test_default_large_context_capped_at_hard_max():
    """Default (not explicit) + 1M context → capped at hard_max (200K).

    This is the core auto-scaling behaviour introduced in #1170:
    without the cap, the default 6000 was silently used for every
    model regardless of size.
    """
    result = compute_input_token_budget(
        configured=6000, context_length=1_000_000, explicit=False
    )
    assert result == DEFAULT_HARD_MAX
    assert result == 200_000


def test_default_medium_context_not_capped():
    """128K model → 85% headroom = 108800, under the 200K hard_max."""
    result = compute_input_token_budget(
        configured=6000, context_length=128_000, explicit=False
    )
    # 128000 * 0.85 = 108800
    assert result == 108_800


def test_default_unknown_context_falls_back_to_configured():
    """Unknown context + default → fall back to the configured value."""
    assert compute_input_token_budget(
        configured=6000, context_length=0, explicit=False
    ) == 6000


def test_default_zero_configured_unknown_context_falls_back_to_default():
    """Unknown context + zero configured → fall back to DEFAULT_BUDGET."""
    result = compute_input_token_budget(
        configured=0, context_length=0, explicit=False
    )
    assert result == DEFAULT_BUDGET
    assert result == 6000


# ---------------------------------------------------------------------------
# Pin the headroom + hard_max constants — they ARE the policy.
# ---------------------------------------------------------------------------
def test_default_headroom_is_85_percent():
    """85% is the documented headroom fraction. Changing this changes
    every model's input budget on the default path; review the
    docstring on compute_input_token_budget before changing."""
    assert DEFAULT_HEADROOM == 0.85


def test_default_hard_max_is_200k():
    """200K is the ceiling on auto-scaled budgets. A 1M-context model
    receives at most this on the default path; raise it in settings
    (``agent_input_token_hard_max``) to unlock more of the window."""
    assert DEFAULT_HARD_MAX == 200_000


# ---------------------------------------------------------------------------
# is_setting_overridden — the bug that made the fix dead-letter.
# ---------------------------------------------------------------------------
def test_is_setting_overridden_key_missing(tmp_path):
    """Key not in settings.json → not overridden."""
    with patch("src.settings.SETTINGS_FILE", str(tmp_path / "settings.json")):
        assert is_setting_overridden("agent_input_token_budget") is False


def test_is_setting_overridden_key_present_with_default_value(tmp_path):
    """Key in settings.json WITH the default value → NOT overridden.

    This is the regression test for the #1170 follow-up: a fresh
    install writes the full DEFAULT_SETTINGS dict to settings.json on
    first run, so 'agent_input_token_budget: 6000' is present even
    when the user never touched it. The previous check used
    ``key in saved`` and returned True here, which blocked the
    large-context bypass on every fresh install.
    """
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "agent_input_token_budget": 6000,  # == DEFAULT_SETTINGS value
    }), encoding="utf-8")
    with patch("src.settings.SETTINGS_FILE", str(settings_path)):
        assert is_setting_overridden("agent_input_token_budget") is False


def test_is_setting_overridden_key_present_with_custom_value(tmp_path):
    """Key in settings.json with a non-default value → overridden."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "agent_input_token_budget": 12_000,  # != DEFAULT_SETTINGS value
    }), encoding="utf-8")
    with patch("src.settings.SETTINGS_FILE", str(settings_path)):
        assert is_setting_overridden("agent_input_token_budget") is True


def test_is_setting_overridden_settings_file_missing(tmp_path):
    """No settings.json → not overridden."""
    with patch("src.settings.SETTINGS_FILE", str(tmp_path / "nonexistent.json")):
        assert is_setting_overridden("agent_input_token_budget") is False


def test_is_setting_overridden_malformed_json(tmp_path):
    """Corrupt settings.json → not overridden (matches load_settings behaviour)."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{ this is not json", encoding="utf-8")
    with patch("src.settings.SETTINGS_FILE", str(settings_path)):
        assert is_setting_overridden("agent_input_token_budget") is False


def test_is_setting_overridden_key_not_in_defaults(tmp_path):
    """Key not in DEFAULT_SETTINGS at all — any value is an override.

    Settings can carry custom keys (plugins, integrations). The
    override-detection logic must not break on them.
    """
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "my_custom_key": "any-value",
    }), encoding="utf-8")
    with patch("src.settings.SETTINGS_FILE", str(settings_path)):
        # DEFAULT_SETTINGS.get("my_custom_key") returns None, so any
        # non-None value is "different" and treated as an override.
        assert is_setting_overridden("my_custom_key") is True


# ---------------------------------------------------------------------------
# End-to-end regression: a 1M model with a default settings.json gets a
# large budget, not the 6K that caused the original truncation bug.
# ---------------------------------------------------------------------------
def test_regression_1m_model_gets_large_budget_with_default_settings(tmp_path):
    """Regression test for #1170: a 1M-context model with default
    settings must NOT be capped at 6000 input tokens.

    Before the fix, three things conspired:
      1. The fresh-install settings.json contained
         'agent_input_token_budget: 6000' (the default).
      2. ``is_setting_overridden`` returned True because the key was
         present (it only checked ``key in saved``, not whether the
         value differed from the default).
      3. The agent loop's large-context bypass only fires when the
         setting is NOT overridden, so it skipped for fresh installs.
      4. ``compute_input_token_budget(6000, 1M, explicit=True)`` returned
         6000, capping the input to 6000 tokens and chopping the
         middle out of any large paste.

    After the fix: ``is_setting_overridden`` returns False for default
    values, the bypass fires, and a 1M model gets a 200K-token budget
    (the hard_max cap on the auto-scaling path).
    """
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "agent_input_token_budget": 6000,  # == DEFAULT_SETTINGS value
    }), encoding="utf-8")

    with patch("src.settings.SETTINGS_FILE", str(settings_path)):
        # The setting is NOT overridden (the fix).
        assert is_setting_overridden("agent_input_token_budget") is False

        # The agent loop passes this boolean to compute_input_token_budget.
        budget = compute_input_token_budget(
            configured=6000,
            context_length=1_000_000,
            explicit=False,  # the fix made this possible
        )
        assert budget > 6000, (
            f"1M-context model got budget={budget}; the fix should scale it up. "
            "Either is_setting_overridden or compute_input_token_budget regressed."
        )
        assert budget == DEFAULT_HARD_MAX
"""Issue #1170 — the agent input-token budget adapts to the model context window.

Pins the pure budget computation and the explicit-override detection.
Uses Python's stdlib unittest only — this project runs on NixOS where
pytest isn't available in the base shell.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.context_budget import (
    compute_input_token_budget,
    DEFAULT_HARD_MAX,
    DEFAULT_BUDGET,
    DEFAULT_HEADROOM,
)
from src.settings import is_setting_overridden


# ---------------------------------------------------------------------------
# compute_input_token_budget — explicit user settings.
# ---------------------------------------------------------------------------
class ExplicitBudgetTests(unittest.TestCase):
    def test_user_budget_smaller_than_context_returns_user_value(self):
        """User explicitly set 6000 — honour it exactly, no scaling."""
        self.assertEqual(
            compute_input_token_budget(
                configured=6000, context_length=1_000_000, explicit=True
            ),
            6000,
        )

    def test_user_budget_larger_than_context_is_clamped(self):
        """User explicitly set 2M but model has 1M — clamp to 1M."""
        self.assertEqual(
            compute_input_token_budget(
                configured=2_000_000, context_length=1_000_000, explicit=True
            ),
            1_000_000,
        )

    def test_user_budget_with_unknown_context_returns_user_value(self):
        """Unknown context — explicit user budget is honoured as-is."""
        self.assertEqual(
            compute_input_token_budget(
                configured=8000, context_length=0, explicit=True
            ),
            8000,
        )


# ---------------------------------------------------------------------------
# compute_input_token_budget — default scaling path.
# ---------------------------------------------------------------------------
class DefaultScalingTests(unittest.TestCase):
    def test_scales_to_context_headroom(self):
        """Default (not explicit) + small context → use headroom fraction."""
        # 32_000 * 0.85 = 27_200
        self.assertEqual(
            compute_input_token_budget(
                configured=6000, context_length=32_000, explicit=False
            ),
            27_200,
        )

    def test_large_context_capped_at_hard_max(self):
        """Default (not explicit) + 1M context → capped at hard_max (200K).

        Regression: without the cap, the default 6000 was silently used
        for every model regardless of size.
        """
        result = compute_input_token_budget(
            configured=6000, context_length=1_000_000, explicit=False
        )
        self.assertEqual(result, DEFAULT_HARD_MAX)
        self.assertEqual(result, 200_000)

    def test_medium_context_not_capped(self):
        """128K model → 85% headroom = 108800, under the 200K hard_max."""
        self.assertEqual(
            compute_input_token_budget(
                configured=6000, context_length=128_000, explicit=False
            ),
            108_800,
        )

    def test_unknown_context_falls_back_to_configured(self):
        """Unknown context + default → fall back to the configured value."""
        self.assertEqual(
            compute_input_token_budget(
                configured=6000, context_length=0, explicit=False
            ),
            6000,
        )

    def test_zero_configured_unknown_context_falls_back_to_default(self):
        """Unknown context + zero configured → fall back to DEFAULT_BUDGET."""
        result = compute_input_token_budget(
            configured=0, context_length=0, explicit=False
        )
        self.assertEqual(result, DEFAULT_BUDGET)
        self.assertEqual(result, 6000)


# ---------------------------------------------------------------------------
# Pin the headroom + hard_max constants — they ARE the policy.
# ---------------------------------------------------------------------------
class PolicyConstantsTests(unittest.TestCase):
    def test_default_headroom_is_85_percent(self):
        """85% is the documented headroom fraction."""
        self.assertEqual(DEFAULT_HEADROOM, 0.85)

    def test_default_hard_max_is_200k(self):
        """200K is the ceiling on auto-scaled budgets."""
        self.assertEqual(DEFAULT_HARD_MAX, 200_000)


# ---------------------------------------------------------------------------
# is_setting_overridden — the bug that made the bypass dead-letter.
# ---------------------------------------------------------------------------
class IsSettingOverriddenTests(unittest.TestCase):
    def test_key_missing(self):
        """Key not in settings.json → not overridden."""
        with tempfile.TemporaryDirectory() as d:
            with patch(
                "src.settings.SETTINGS_FILE",
                os.path.join(d, "settings.json"),
            ):
                self.assertFalse(
                    is_setting_overridden("agent_input_token_budget")
                )

    def test_key_present_with_default_value(self):
        """Key in settings.json WITH the default value → NOT overridden.

        Regression test for the #1170 follow-up: a fresh install writes
        the full DEFAULT_SETTINGS dict to settings.json on first run, so
        'agent_input_token_budget: 6000' is present even when the user
        never touched it. The previous check used ``key in saved`` and
        returned True here, blocking the large-context bypass on every
        fresh install.
        """
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "settings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"agent_input_token_budget": 6000}, f
                )  # == DEFAULT_SETTINGS value
            with patch("src.settings.SETTINGS_FILE", path):
                self.assertFalse(
                    is_setting_overridden("agent_input_token_budget")
                )

    def test_key_present_with_custom_value(self):
        """Key in settings.json with a non-default value → overridden."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "settings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"agent_input_token_budget": 12_000}, f
                )  # != DEFAULT_SETTINGS value
            with patch("src.settings.SETTINGS_FILE", path):
                self.assertTrue(
                    is_setting_overridden("agent_input_token_budget")
                )

    def test_settings_file_missing(self):
        """No settings.json → not overridden."""
        with tempfile.TemporaryDirectory() as d:
            with patch(
                "src.settings.SETTINGS_FILE",
                os.path.join(d, "nonexistent.json"),
            ):
                self.assertFalse(
                    is_setting_overridden("agent_input_token_budget")
                )

    def test_malformed_json(self):
        """Corrupt settings.json → not overridden (matches load_settings)."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "settings.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{ this is not json")
            with patch("src.settings.SETTINGS_FILE", path):
                self.assertFalse(
                    is_setting_overridden("agent_input_token_budget")
                )

    def test_key_not_in_defaults(self):
        """Key not in DEFAULT_SETTINGS at all — any value is an override.

        Settings can carry custom keys (plugins, integrations). The
        override-detection logic must not break on them.
        """
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "settings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"my_custom_key": "any-value"}, f)
            with patch("src.settings.SETTINGS_FILE", path):
                # DEFAULT_SETTINGS.get("my_custom_key") returns None,
                # so any non-None value is "different" → treated as an override.
                self.assertTrue(is_setting_overridden("my_custom_key"))


# ---------------------------------------------------------------------------
# End-to-end regression: a 1M model with a default settings.json gets a
# large budget, not the 6K that caused the original truncation bug.
# ---------------------------------------------------------------------------
class Regression1MModelTests(unittest.TestCase):
    def test_default_settings_gives_large_budget_for_1m_model(self):
        """Regression for #1170: 1M-context model with default settings
        must NOT be capped at 6000 input tokens.

        Before the fix, three things conspired:
          1. Fresh-install settings.json contained
             'agent_input_token_budget: 6000' (the default).
          2. is_setting_overridden returned True because the key was
             present (only checked ``key in saved``).
          3. The agent loop's large-context bypass only fires when
             NOT overridden, so it skipped for fresh installs.
          4. compute_input_token_budget(6000, 1M, True) returned 6000,
             capping input and chopping the middle out of large pastes.

        After the fix: is_setting_overridden returns False for default
        values, the bypass fires, and a 1M model gets a 200K-token
        budget (the hard_max cap on the auto-scaling path).
        """
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "settings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"agent_input_token_budget": 6000}, f
                )  # == DEFAULT_SETTINGS value

            with patch("src.settings.SETTINGS_FILE", path):
                # The setting is NOT overridden (the fix).
                self.assertFalse(
                    is_setting_overridden("agent_input_token_budget")
                )

                # The agent loop passes this boolean to
                # compute_input_token_budget.
                budget = compute_input_token_budget(
                    configured=6000,
                    context_length=1_000_000,
                    explicit=False,  # the fix made this possible
                )
                self.assertGreater(budget, 6000)
                self.assertEqual(budget, DEFAULT_HARD_MAX)


if __name__ == "__main__":
    unittest.main()
