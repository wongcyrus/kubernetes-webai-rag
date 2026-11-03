import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { AutoModel, AutoTokenizer, env } from '@huggingface/transformers';
import { searchKeywords, rankHybridResults } from '../src/KeywordSearch.js';
import { documentInput } from './kubernetes-chunks.mjs';

const root = process.argv[2] || 'data/kubernetes';
const manifest = JSON.parse(await readFile(join(root, 'manifest.json'), 'utf8'));
assert.equal(manifest.model.runtime, 'embeddinggemma', 'Build the EmbeddingGemma corpus before evaluating it.');
assert.equal(manifest.partial, false, 'Relevance evaluation requires the complete corpus.');
const [chunks, buffer, keywordIndex] = await Promise.all([
  readFile(join(root, manifest.chunks), 'utf8').then(JSON.parse),
  readFile(join(root, manifest.vectors)),
  readFile(join(root, manifest.keywords), 'utf8').then(JSON.parse)
]);
const vectors = new Float32Array(manifest.count * manifest.dimensions);
for (let index = 0; index < vectors.length; index++) vectors[index] = buffer.readFloatLE(index * 4);
const data = { manifest, chunks, vectors, keywordIndex };
env.allowLocalModels = false;
const tokenizer = await AutoTokenizer.from_pretrained(manifest.model.url, { revision: manifest.model.revision });
const model = await AutoModel.from_pretrained(manifest.model.url, {
  revision: manifest.model.revision, dtype: manifest.model.dtype, device: 'cpu',
  session_options: { intraOpNumThreads: 4, interOpNumThreads: 1 }
});
const cases = [
  {
    query: 'How do I create a egress?',
    expected: /\/docs\/concepts\/services-networking\/network-policies\/#.*egress/
  },
  {
    query: 'How do I create a Deployment?',
    expected: /\/docs\/concepts\/workloads\/controllers\/deployment\/#creating-a-deployment/
  },
  {
    query: 'How do I expose a service outside the cluster?',
    expected: /\/docs\/(?:concepts\/services-networking\/service|tasks\/access-application-cluster\/service-access-application-cluster)/
  }
];
try {
  for (const row of [0, Math.floor(manifest.count / 2), manifest.count - 1]) {
    const inputs = await tokenizer(documentInput(chunks[row]), {
      padding: true, truncation: true, max_length: manifest.model.sequenceLength
    });
    const output = await model(inputs);
    const embedding = output.sentence_embedding.data;
    let dot = 0;
    let norm = 0;
    for (let column = 0; column < manifest.dimensions; column++) {
      dot += embedding[column] * vectors[row * manifest.dimensions + column];
      norm += embedding[column] * embedding[column];
    }
    const similarity = dot / Math.sqrt(norm);
    for (const tensor of Object.values(output)) tensor.dispose?.();
    assert.ok(similarity > 0.99, `Stored vector does not match its source text at chunk ${row + 1}.`);
    console.log(`Verified stored chunk ${row + 1}: embedding cosine ${similarity.toFixed(6)}`);
  }
  for (const { query, expected } of cases) {
    const inputs = await tokenizer(manifest.queryPrefix + query, {
      padding: true, truncation: true, max_length: manifest.model.sequenceLength
    });
    const output = await model(inputs);
    const vector = Array.from(output.sentence_embedding.data);
    for (const tensor of Object.values(output)) tensor.dispose?.();
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    const semantic = [];
    for (let row = 0; row < manifest.count; row++) {
      let dot = 0;
      for (let column = 0; column < manifest.dimensions; column++) {
        dot += vector[column] * vectors[row * manifest.dimensions + column];
      }
      semantic.push({ row, score: dot / norm });
    }
    semantic.sort((left, right) => right.score - left.score || left.row - right.row);
    const keyword = searchKeywords(keywordIndex, query);
    const matches = rankHybridResults(semantic.slice(0, 50), keyword, data, vector, 0.25, 5);
    const sources = matches.map(match => chunks[match.row].sourceUrl);
    console.log(JSON.stringify({ query, sources }, null, 2));
    assert.ok(sources.slice(0, 3).some(source => expected.test(source)),
      `Expected documentation missing from top three sources for: ${query}`);
    if (query.includes('egress')) {
      assert.ok(!sources.some(source => /SuccessPolicyRule|fine-parallel-processing-work-queue/.test(source)),
        'The reported unrelated Job/Redis retrieval must not recur.');
    }
  }
} finally {
  await model.dispose();
}
console.log(`PASS: ${cases.length} Kubernetes relevance checks on ${manifest.count} prebuilt chunks.`);
