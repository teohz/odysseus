import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const markdownPath = path.join(__dirname, '..', 'static', 'js', 'markdown.js');
let src = fs.readFileSync(markdownPath, 'utf8');

src = src.replace(
  /import uiModule from '\.\/ui\.js';/,
  'const uiModule = { esc: (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\\"/g, "&quot;") };'
);
src = src.replace(
  /import \{ splitTableRow \} from '\.\/markdown\/tableRow\.js';/,
  'const splitTableRow = (row) => row.split("|").filter((cell) => cell.trim() !== "");'
);
src = src.replace(/export function /g, 'function ');
src = src.replace(/export const /g, 'const ');
src = src.replace(/export default markdownModule;?/g, '');
src += '\nthis.__mdToHtml = mdToHtml;';

class MutationObserver {
  observe() {}
  disconnect() {}
}

const sandbox = {
  console,
  URL,
  MutationObserver,
  localStorage: { getItem() { return '[]'; }, setItem() {} },
  document: {
    body: { classList: { contains() { return true; } } },
    addEventListener() {},
    querySelectorAll() { return []; },
    getElementById() { return null; },
    contains() { return true; },
  },
  window: {
    location: { origin: 'http://localhost' },
    katex: null,
    mermaid: null,
  },
};

vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: markdownPath });

const input = [
  '> ```html',
  '> <script>',
  '>   newWindow.addEventListener(\'click\', () => {',
  '>     desktop.appendChild(newWindow);',
  '>   });',
  '> </script>',
  '> ```',
].join('\n');

const html = sandbox.__mdToHtml(input);
// No placeholder from any of the four families should remain in the
// output — if any does, the corresponding restoration pass is broken.
assert.equal(html.includes('___ALLOWED_HTML_'), false, html);
assert.equal(html.includes('___CODE_BLOCK_'), false, html);
assert.equal(html.includes('___MATH_BLOCK_'), false, html);
assert.equal(html.includes('___MERMAID_BLOCK_'), false, html);

// The original code content must be preserved verbatim.
assert.equal(html.includes('appendChild'), true, html);

// The <script> tag inside the code sample must be HTML-escaped (not
// emitted as a live script element) and the blockquote "> " prefix
// must be stripped from each line of the code body.
assert.equal(html.includes('&lt;script&gt;'), true, html);
assert.equal(html.includes('<script>'), false, html);
assert.equal(html.includes('&gt; &lt;'), false, html);

// The code block should be wrapped in <pre><code> with the language
// class applied.
assert.match(html, /<pre><code[^>]*language-html/);

console.log('ok');
