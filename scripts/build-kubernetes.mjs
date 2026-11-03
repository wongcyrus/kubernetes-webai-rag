import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, rename, unlink } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { join } from 'node:path';
import { AutoModel, AutoTokenizer, env } from '@huggingface/transformers';
import { chunkSections, extractSections, documentInput, buildKeywordIndex, DOCUMENT_PREFIX, QUERY_PREFIX } from './kubernetes-chunks.mjs';

const { values } = parseArgs({
  options: {
    html: { type: 'string' },
    sitemap: { type: 'string' },
    out: { type: 'string', default: 'data/kubernetes' },
    revision: { type: 'string', default: 'main' },
    'max-tokens': { type: 'string', default: '512' },
    'overlap-tokens': { type: 'string', default: '64' },
    'batch-size': { type: 'string', default: '4' },
    limit: { type: 'string' },
    help: { type: 'boolean' }
  }
});

if (values.help) {
  console.log('Usage: npm run build:kubernetes -- [--html file] [--sitemap file] [--out directory]');
  console.log('Options: --revision SHA, --max-tokens 512, --overlap-tokens 64, --batch-size 4');
  console.log('--limit N builds only the first N chunks for a smoke test, not the full corpus.');
  process.exit(0);
}

const sourceUrl = 'https://kubernetes.io/docs/_print/';
const modelUrl = 'onnx-community/embeddinggemma-300m-ONNX';
const batchSize = Number(values['batch-size']);
const maxTokens = Number(values['max-tokens']);
const overlapTokens = Number(values['overlap-tokens']);
const limit = values.limit === undefined ? undefined : Number(values.limit);
if (!Number.isInteger(batchSize) || batchSize < 1 ||
    (limit !== undefined && (!Number.isInteger(limit) || limit < 1)) ||
    !Number.isInteger(maxTokens) || maxTokens < 32 || maxTokens > 2048 ||
    !Number.isInteger(overlapTokens) || overlapTokens < 0 || overlapTokens >= maxTokens / 2) {
  throw new Error('Use a positive batch size/limit, 32-2048 tokens, and overlap below half the chunk size.');
}

async function fetchChecked(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(120000) });
  if (!response.ok) throw new Error(`Download failed (${response.status}): ${url}`);
  return response;
}

await mkdir(values.out, { recursive: true });
const progressPath = join(values.out, '.build-progress.json');
const progressVectorsPath = join(values.out, '.build-vectors.f32');
let progress;
try {
  progress = JSON.parse(await readFile(progressPath, 'utf8'));
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}
console.log('Reading Kubernetes documentation and its English sitemap...');
const [html, sitemap, modelInfo] = await Promise.all([
  values.html ? readFile(values.html, 'utf8') : progress ?
    readFile(join(values.out, 'source.html'), 'utf8') : fetchChecked(sourceUrl).then(r => r.text()),
  values.sitemap ? readFile(values.sitemap, 'utf8') : progress ?
    readFile(join(values.out, 'sitemap.xml'), 'utf8') :
    fetchChecked('https://kubernetes.io/en/sitemap.xml').then(r => r.text()),
  fetchChecked(`https://huggingface.co/api/models/${modelUrl}/revision/${encodeURIComponent(
    values.revision === 'main' && progress ? progress.model.revision : values.revision)}`)
    .then(r => r.json())
]);
if (!/^[a-f0-9]{40}$/.test(modelInfo.sha)) throw new Error('Could not resolve an immutable model revision.');

const { sections, unresolvedPages } = extractSections(html, sitemap, sourceUrl);
console.log(`Extracted ${sections.length} sections; loading the offline embedding model...`);
env.allowLocalModels = false;
const tokenizer = await AutoTokenizer.from_pretrained(modelUrl, { revision: modelInfo.sha });
const extractor = await AutoModel.from_pretrained(modelUrl, {
  revision: modelInfo.sha, dtype: 'q4', device: 'cpu',
  session_options: { intraOpNumThreads: 4, interOpNumThreads: 1 }
});
const allChunks = chunkSections(sections, tokenizer, maxTokens, overlapTokens);
const chunks = limit === undefined ? allChunks : allChunks.slice(0, limit);
if (!chunks.length) throw new Error('No documentation chunks were extracted.');
if (limit !== undefined) console.warn(`SMOKE TEST: building ${chunks.length} of ${allChunks.length} chunks.`);
if (unresolvedPages.length) {
  console.warn(`${unresolvedPages.length} pages use their print-page anchor because their canonical URL could not be resolved.`);
}

const model = {
  runtime: 'embeddinggemma',
  url: modelUrl,
  revision: modelInfo.sha,
  dtype: 'q4',
  sequenceLength: maxTokens
};
let vectors;
let dimensions;
const chunksJson = JSON.stringify(chunks);
const fingerprint = createHash('sha256')
  .update(JSON.stringify({ model, batchSize })).update(chunksJson).digest('hex');
const ordered = chunks.map((chunk, row) => ({
  chunk, row, tokens: tokenizer(documentInput(chunk)).input_ids.data.length
})).sort((left, right) => left.tokens - right.tokens || left.row - right.row);
let completed = 0;
if (progress) {
  if (progress.fingerprint !== fingerprint || progress.dimensions !== 768 ||
      !Number.isInteger(progress.completed) || progress.completed < 0 ||
      progress.completed > chunks.length ||
      (progress.completed !== chunks.length && progress.completed % batchSize !== 0)) {
    throw new Error('Saved build progress does not match these chunks/model/batch size. Use a different --out directory.');
  }
  const saved = await readFile(progressVectorsPath);
  if (saved.length !== chunks.length * progress.dimensions * 4) throw new Error('Truncated saved build vectors.');
  dimensions = progress.dimensions;
  vectors = new Float32Array(saved.buffer, saved.byteOffset, saved.length / 4).slice();
  completed = progress.completed;
  console.log(`Resuming ${completed} / ${chunks.length} saved embeddings.`);
}
await Promise.all([
  writeFile(join(values.out, 'source.html'), html),
  writeFile(join(values.out, 'sitemap.xml'), sitemap)
]);
for (let offset = completed; offset < ordered.length; offset += batchSize) {
  const entries = ordered.slice(offset, offset + batchSize);
  const batch = entries.map(entry => entry.chunk);
  const inputs = await tokenizer(batch.map(documentInput), {
    padding: true, truncation: true, max_length: maxTokens
  });
  const tensors = await extractor(inputs);
  const output = tensors.sentence_embedding;
  if (!vectors) {
    dimensions = output.dims[1];
    vectors = new Float32Array(chunks.length * dimensions);
  }
  if (output.data.length !== batch.length * dimensions) throw new Error('Unexpected embedding dimensions.');
  for (let index = 0; index < entries.length; index++) {
    vectors.set(output.data.subarray(index * dimensions, (index + 1) * dimensions), entries[index].row * dimensions);
  }
  for (const tensor of Object.values(tensors)) tensor.dispose?.();
  completed = Math.min(offset + batch.length, chunks.length);
  console.log(`Embedded ${completed} / ${chunks.length} chunks`);
  if (completed % (batchSize * 16) === 0 || completed === chunks.length) {
    await writeFile(progressVectorsPath + '.download', Buffer.from(vectors.buffer));
    await rename(progressVectorsPath + '.download', progressVectorsPath);
    await writeFile(progressPath + '.download', JSON.stringify({ fingerprint, model, dimensions, completed }));
    await rename(progressPath + '.download', progressPath);
    console.log(`Saved progress: ${completed} / ${chunks.length} embeddings.`);
  }
}
await extractor.dispose();

for (let row = 0; row < chunks.length; row++) {
  let norm = 0;
  for (let column = 0; column < dimensions; column++) {
    const value = vectors[row * dimensions + column];
    norm += value * value;
  }
  if (!Number.isFinite(norm) || norm === 0) throw new Error('Model produced an invalid embedding norm.');
  const length = Math.sqrt(norm);
  for (let column = 0; column < dimensions; column++) vectors[row * dimensions + column] /= length;
}
const vectorBuffer = Buffer.alloc(vectors.byteLength);
for (let index = 0; index < vectors.length; index++) {
  if (!Number.isFinite(vectors[index])) throw new Error('Model produced a non-finite embedding.');
  vectorBuffer.writeFloatLE(vectors[index], index * 4);
}
const keywordJson = JSON.stringify(buildKeywordIndex(chunks));
const version = createHash('sha256')
  .update(JSON.stringify({ model, maxTokens, overlapTokens }))
  .update(chunksJson).update(keywordJson).update(vectorBuffer).digest('hex').slice(0, 16);
const manifest = {
  format: 'vectorsearch-prebuilt-v1',
  version,
  sourceUrl,
  sourceSha256: createHash('sha256').update(html).digest('hex'),
  model,
  documentPrefix: DOCUMENT_PREFIX,
  queryPrefix: QUERY_PREFIX,
  normalized: true,
  count: chunks.length,
  dimensions,
  maxTokens,
  overlapTokens,
  partial: limit !== undefined,
  unresolvedPages,
  chunks: `chunks-${version}.json`,
  vectors: `vectors-${version}.f32`,
  keywords: `keywords-${version}.json`,
  encoding: 'float32-le',
  license: { name: 'CC BY 4.0', url: 'https://creativecommons.org/licenses/by/4.0/', attribution: 'Kubernetes Authors' }
};
await mkdir(values.out, { recursive: true });
await Promise.all([
  writeFile(join(values.out, manifest.chunks), chunksJson),
  writeFile(join(values.out, manifest.keywords), keywordJson),
  writeFile(join(values.out, manifest.vectors), vectorBuffer)
]);
await writeFile(join(values.out, 'manifest.json.download'), JSON.stringify(manifest, null, 2) + '\n');
await rename(join(values.out, 'manifest.json.download'), join(values.out, 'manifest.json'));
await Promise.all([unlink(progressPath), unlink(progressVectorsPath)]);
console.log(`Prebuilt ${chunks.length} vectors (${dimensions} dimensions) into ${values.out}/manifest.json`);
