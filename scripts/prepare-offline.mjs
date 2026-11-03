import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { mkdir, readFile, writeFile, copyFile, readdir, stat, rename } from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { dirname, join, resolve, extname } from 'node:path';
import { parseArgs } from 'node:util';
import { buildBrowser } from './build-browser.mjs';
import { parseHTML } from 'linkedom';
import { pageUrls } from './kubernetes-chunks.mjs';
import { GEMMA4_MODEL } from '../src/OfflineAnswerGenerator.js';

const { values } = parseArgs({ options: {
  'assets-only': { type: 'boolean' },
  out: { type: 'string', default: 'offline' },
  html: { type: 'string' },
  sitemap: { type: 'string' }
} });
const root = resolve(values.out);
const source = 'https://kubernetes.io/docs/_print/';
const modelId = 'onnx-community/embeddinggemma-300m-ONNX';
const sourceData = 'data/kubernetes';
let manifest;
try {
  manifest = JSON.parse(await readFile(join(sourceData, 'manifest.json'), 'utf8'));
} catch (error) {
  if (!values['assets-only']) throw error;
}
if (!values['assets-only'] && (manifest?.model.runtime !== 'embeddinggemma' || manifest.partial)) {
  throw new Error('Finish the complete EmbeddingGemma corpus build before preparing the exam dataset.');
}
const revision = manifest?.model.runtime === 'embeddinggemma' ? manifest.model.revision :
  '5090578d9565bb06545b4552f76e6bc2c93e4a66';
await mkdir(root, { recursive: true });

async function fetchChecked(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(180000) });
  if (!response.ok) throw new Error(`Offline preparation download failed (${response.status}): ${url}`);
  return response;
}

async function download(url, destination, expected = {}) {
  try {
    const size = (await stat(destination)).size;
    if (size > 0 && (!expected.bytes || size === expected.bytes)) return;
  } catch {}
  await mkdir(dirname(destination), { recursive: true });
  const response = await fetchChecked(url);
  const temporary = destination + '.download';
  const checksum = expected.sha256 ? createHash('sha256') : undefined;
  await pipeline(Readable.fromWeb(response.body), new Transform({
    transform(chunk, encoding, callback) {
      checksum?.update(chunk);
      callback(null, chunk);
    }
  }), createWriteStream(temporary));
  if (expected.bytes && (await stat(temporary)).size !== expected.bytes) throw new Error(`Incomplete download: ${url}`);
  if (checksum && checksum.digest('hex') !== expected.sha256) throw new Error(`Model checksum mismatch: ${url}`);
  await rename(temporary, destination);
}

await Promise.all(['js', 'css', 'vendor/onnx', 'licenses'].map(path => mkdir(join(root, path), { recursive: true })));
const require = createRequire(import.meta.resolve('@huggingface/transformers'));
const ortDirectory = dirname(require.resolve('onnxruntime-web'));
for (const file of await readdir(ortDirectory)) {
  if (/^ort-wasm-simd-threaded.*\.(mjs|wasm)$/.test(file)) {
    await copyFile(join(ortDirectory, file), join(root, 'vendor/onnx', file));
  }
}
await Promise.all([
  copyFile('LICENSE', join(root, 'licenses/vectorsearch.txt')),
  copyFile('node_modules/@huggingface/transformers/LICENSE', join(root, 'licenses/transformers.txt'))
]);
await Promise.all([
  download('https://raw.githubusercontent.com/tensorflow/tfjs/master/LICENSE', join(root, 'licenses/tensorflow.txt')),
  download('https://raw.githubusercontent.com/microsoft/onnxruntime/main/LICENSE', join(root, 'licenses/onnxruntime.txt')),
  download('https://creativecommons.org/licenses/by/4.0/legalcode.txt', join(root, 'licenses/kubernetes.txt'))
]);

console.log('Preparing local EmbeddingGemma weights and tokenizer...');
const modelFiles = [
  'README.md', 'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
  'special_tokens_map.json', 'added_tokens.json', 'tokenizer.model',
  'onnx/model_q4.onnx', 'onnx/model_q4.onnx_data'
];
for (const file of modelFiles) {
  await download(`https://huggingface.co/${modelId}/resolve/${revision}/${file}`,
    join(root, 'models', revision, modelId, file));
}
const termsPage = await fetchChecked('https://ai.google.dev/gemma/terms').then(response => response.text());
const termsDocument = parseHTML(termsPage).document;
const terms = termsDocument.querySelector('article') || termsDocument.querySelector('main');
if (!terms) throw new Error('Could not extract the official Gemma terms.');
await writeFile(join(root, 'licenses/gemma.txt'),
  `Source: https://ai.google.dev/gemma/terms\n\n${terms.textContent.trim()}\n`);

console.log('Preparing the local Gemma 4 E2B answer model (2.01 GB)...');
const generationDirectory = join(root, 'models', GEMMA4_MODEL.revision, GEMMA4_MODEL.id);
await download(`https://huggingface.co/${GEMMA4_MODEL.id}/resolve/${GEMMA4_MODEL.revision}/${GEMMA4_MODEL.file}`,
  join(generationDirectory, GEMMA4_MODEL.file), GEMMA4_MODEL);
await download(`https://huggingface.co/${GEMMA4_MODEL.id}/resolve/${GEMMA4_MODEL.revision}/README.md`,
  join(generationDirectory, 'README.md'));
const lmDirectory = dirname(createRequire(import.meta.url).resolve('@litert-lm/core/package.json'));
const wasmDirectory = join(root, 'vendor/litert-lm');
await mkdir(wasmDirectory, { recursive: true });
for (const file of await readdir(join(lmDirectory, 'wasm'))) {
  if (/\.(js|wasm)$/.test(file)) await copyFile(join(lmDirectory, 'wasm', file), join(wasmDirectory, file));
}
await download('https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/main/LICENSE',
  join(root, 'licenses/litert-lm.txt'));
await copyFile(join(root, 'licenses/litert-lm.txt'), join(root, 'licenses/gemma4.txt'));

console.log('Preparing readable local Kubernetes source pages...');
const [html, sitemap] = await Promise.all([
  values.html ? readFile(values.html, 'utf8') : manifest?.sourceSha256 ?
    readFile(join(sourceData, 'source.html'), 'utf8') : fetchChecked(source).then(response => response.text()),
  values.sitemap ? readFile(values.sitemap, 'utf8') : manifest?.sourceSha256 ?
    readFile(join(sourceData, 'sitemap.xml'), 'utf8') :
    fetchChecked('https://kubernetes.io/en/sitemap.xml').then(response => response.text())
]);
if (manifest?.sourceSha256 && createHash('sha256').update(html).digest('hex') !== manifest.sourceSha256) {
  throw new Error('The documentation snapshot differs from the indexed source.');
}
const { document } = parseHTML(html);
const urls = pageUrls(sitemap, source);
const pages = [...document.querySelectorAll('main .td-content')]
  .filter(page => page.querySelector('h1[id^="pg-"]'));
if (!pages.length) throw new Error('No printable documentation pages found.');
const assets = new Map();
const byPrintId = new Map();
for (const page of pages) {
  const id = page.querySelector('h1').id;
  byPrintId.set(id, urls.get(id) ? new URL(urls.get(id)).pathname : `/docs/offline/${id}/`);
}
const knownPaths = new Set(['/docs', '/docs/_print', ...[...byPrintId.values()].map(path => path.replace(/\/$/, ''))]);
async function localAsset(url) {
  if (!assets.has(url.href)) {
    const extension = extname(url.pathname).replace(/[^.a-z0-9]/gi, '').slice(0, 12);
    const filename = createHash('sha256').update(url.href).digest('hex').slice(0, 20) + extension;
    const local = `/docs/assets/${filename}`;
    assets.set(url.href, download(url.href, join(root, local.slice(1))).then(() => local));
  }
  return assets.get(url.href);
}
function escapeHtml(text) {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}
const policy = "default-src 'self'; script-src 'none'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; base-uri 'self'";
function pageHtml(title, content, originalUrl = source) {
  return `<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="${policy}"><title>${escapeHtml(title)}</title><style>body{max-width:1000px;margin:2em auto;padding:0 1em;font:17px/1.6 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f3;padding:1em}code{font-family:monospace}table{border-collapse:collapse;display:block;overflow-x:auto}td,th{border:1px solid #ccc;padding:.5em}img{max-width:100%}a{color:#165bb7}.offline-unavailable{color:#666}h1,h2,h3,h4{scroll-margin-top:1em}</style></head><body><nav><a href="/kubernetes.html">Search documentation</a> | <a href="/docs/">Documentation index</a></nav>${content}<footer><hr>Kubernetes Authors. Documentation licensed under <a href="/licenses/kubernetes.txt">CC BY 4.0</a>. Original source: ${escapeHtml(originalUrl)}. This is a local snapshot; links outside the snapshot are disabled.</footer></body></html>`;
}
async function rewritePage(page) {
  const pageId = page.querySelector('h1').id;
  const originalPage = urls.get(pageId) || source;
  for (const element of page.querySelectorAll('script,style,link,iframe,object,embed,form,button,source,.pageinfo,.feedback')) {
    element.remove();
  }
  for (const element of page.querySelectorAll('*')) {
    for (const attribute of [...element.attributes]) {
      if (/^on/i.test(attribute.name) || ['style', 'srcset'].includes(attribute.name)) {
        element.removeAttribute(attribute.name);
      }
    }
  }
  for (const image of page.querySelectorAll('img')) {
    const src = image.getAttribute('src');
    if (!src || src.startsWith('data:')) continue;
    image.setAttribute('src', await localAsset(new URL(src, originalPage)));
  }
  for (const link of page.querySelectorAll('a[href]')) {
    const href = link.getAttribute('href');
    if (href.startsWith('#')) {
      const path = byPrintId.get(href.slice(1));
      if (path) link.setAttribute('href', path + href);
      continue;
    }
    let target;
    try { target = new URL(href, originalPage); } catch { target = null; }
    if (target?.origin === 'https://kubernetes.io' &&
        knownPaths.has(target.pathname.replace(/\/$/, ''))) {
      link.setAttribute('href', target.pathname + target.hash);
    } else if (target?.hostname === 'raw.githubusercontent.com' &&
        target.pathname.startsWith('/kubernetes/website/') && link.hasAttribute('download')) {
      link.setAttribute('href', await localAsset(target));
    } else {
      link.removeAttribute('href');
      link.classList.add('offline-unavailable');
      link.setAttribute('title', 'Outside the offline documentation snapshot');
    }
  }
}
const index = [];
const printContents = [];
for (let offset = 0; offset < pages.length; offset += 4) {
  const prepared = await Promise.all(pages.slice(offset, offset + 4).map(async page => {
    await rewritePage(page);
    const heading = page.querySelector('h1');
    const title = heading.textContent.replace(/^[\d.]+\s*-\s*/, '').trim() || 'Documentation';
    const path = byPrintId.get(heading.id);
    const directory = join(root, path.slice(1));
    await mkdir(directory, { recursive: true });
    await writeFile(join(directory, 'index.html'), pageHtml(title, page.outerHTML, urls.get(heading.id) || source));
    return { title, path, html: page.outerHTML };
  }));
  for (const page of prepared) {
    index.push({ title: page.title, path: page.path });
    printContents.push(page.html);
  }
}
await mkdir(join(root, 'docs/_print'), { recursive: true });
await writeFile(join(root, 'docs/_print/index.html'), pageHtml('Kubernetes printable documentation', printContents.join('\n')));
await writeFile(join(root, 'docs/index.html'), pageHtml('Kubernetes documentation',
  `<h1>Offline Kubernetes Documentation</h1><ul>${index.map(page =>
    `<li><a href="${page.path}">${escapeHtml(page.title)}</a></li>`).join('')}</ul>`));
await writeFile(join(root, 'docs/snapshot.json'), JSON.stringify({
  source, pages: pages.length, assets: assets.size,
  sourceSha256: createHash('sha256').update(html).digest('hex'),
  modelId, modelRevision: revision
}, null, 2) + '\n');

if (!values['assets-only']) {
  const destination = join(root, 'data/kubernetes');
  await mkdir(destination, { recursive: true });
  for (const file of [manifest.chunks, manifest.vectors, manifest.keywords]) {
    await copyFile(join(sourceData, file), join(destination, file));
  }
  await copyFile(join(sourceData, 'manifest.json'), join(destination, 'manifest.json.download'));
  await rename(join(destination, 'manifest.json.download'), join(destination, 'manifest.json'));
}
await buildBrowser(root);
if (manifest && !values['assets-only']) {
  const project = JSON.parse(await readFile('package.json', 'utf8'));
  await writeFile(join(root, 'release.json'), JSON.stringify({
    application: project.name, version: project.version, datasetVersion: manifest.version,
    generationModel: GEMMA4_MODEL,
    generationModelPath: `models/${GEMMA4_MODEL.revision}/${GEMMA4_MODEL.id}/${GEMMA4_MODEL.file}`,
    embeddingModelPath: `models/${revision}/${modelId}`
  }, null, 2) + '\n');
}
console.log(`Prepared ${pages.length} source pages, ${assets.size} local assets, model weights, and runtimes in ${root}.`);
if (values['assets-only']) console.log('Assets only: rerun npm run prepare:offline after the complete corpus build to include the dataset.');
else console.log('Serve this directory on localhost. No internet is needed during the student test.');
