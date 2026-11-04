const assert = require('node:assert/strict');
const { readFile } = require('node:fs/promises');
const test = require('node:test');
const vm = require('node:vm');
const MarkdownIt = require('markdown-it');
const { dirname, resolve } = require('node:path');

async function loadClass(path, name, dependencies = {}, globals = {}) {
  dependencies = { 'markdown-it': { default: MarkdownIt }, ...dependencies };
  const context = vm.createContext({ URL, TextEncoder, ...globals });
  const module = new vm.SourceTextModule(await readFile(path, 'utf8'), {
    context,
    importModuleDynamically: async specifier => {
      const exports = dependencies[specifier] || {};
      const dependency = new vm.SyntheticModule(Object.keys(exports), function() {
        for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
      }, { context });
      await dependency.link(() => {});
      await dependency.evaluate();
      return dependency;
    }
  });
  await module.link(async specifier => {
    if (specifier.startsWith('.') && !(specifier in dependencies)) {
      return new vm.SourceTextModule(await readFile(resolve(dirname(path), specifier), 'utf8'), { context });
    }
    const exports = dependencies[specifier] || {};
    return new vm.SyntheticModule(Object.keys(exports), function() {
      for (const [key, value] of Object.entries(exports)) {
        this.setExport(key, value);
      }
    }, { context });
  });
  await module.evaluate();
  return name ? module.namespace[name] : module.namespace;
}

const sourceUrl = 'https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#creating-a-deployment';
const document = { text: 'Create a Deployment.', sourceUrl, title: 'Creating a Deployment' };

test('storage preserves anchor citations and supports text-only and missing records', async () => {
  const VectorStore = await loadClass('src/VectorStore.js', 'VectorStore');
  const records = new Map();
  const db = {
    transaction() {
      const transaction = {
        objectStore() {
          return {
            add(item) {
              records.set(records.size + 1, structuredClone(item));
            },
            get(id) {
              const request = { result: records.get(id) };
              queueMicrotask(() => request.onsuccess());
              return request;
            }
          };
        }
      };
      queueMicrotask(() => transaction.oncomplete?.());
      return transaction;
    }
  };
  const store = new VectorStore();
  store.initDb = async () => db;
  await store.storeBatch([
    { ...document, embedding: [1, 0] },
    { text: 'Legacy text', embedding: [0, 1] }
  ]);

  assert.deepEqual(structuredClone(await store.getDocumentByID(1)), document);
  assert.equal(await store.getTextByID(1), document.text);
  assert.deepEqual(structuredClone(await store.getDocumentByID(2)), { text: 'Legacy text' });
  assert.equal(await store.getTextByID(2), 'Legacy text');
  assert.equal(await store.getDocumentByID(99), null);
  assert.equal(await store.getTextByID(99), null);
});

test('search returns citations and reloads cached records after importing another batch', async () => {
  class Stub {}
  const VectorSearch = await loadClass('src/VectorSearch.js', 'VectorSearch', {
    './VectorStore.js': { VectorStore: Stub },
    './CosineSimilarity.js': { CosineSimilarity: Stub },
    './EmbeddingModel.js': { EmbeddingModel: Stub }
  });
  const search = new VectorSearch({});
  const items = [{ text: 'Legacy text', embedding: [0, 1] }];
  let reads = 0;
  let cacheClears = 0;
  search.vectorStore = {
    async getAllVectors() {
      reads++;
      return items.map((item, index) => ({ id: index + 1, embedding: item.embedding }));
    },
    async getDocumentByID(id) {
      const { embedding, ...storedDocument } = items[id - 1];
      return storedDocument;
    },
    async storeBatch(batch) {
      items.push(...batch);
    }
  };
  search.cosineSimilarity = {
    async deleteGPUVectorCache() {
      cacheClears++;
    },
    async cosineSimilarityTFJSGPUMatrix(matrix) {
      return {
        values: matrix.map(() => 1),
        indices: matrix.map((_, index) => index)
      };
    }
  };

  let response = await search.search([1, 0], 0.25, 'kubernetes');
  assert.equal(response.results[0].text, 'Legacy text');
  assert.equal('sourceUrl' in response.results[0], false);
  await search.storeBatch([{ ...document, embedding: [1, 0] }]);
  response = await search.search([1, 0], 0.25, 'kubernetes');
  assert.equal(response.results[1].sourceUrl, sourceUrl);
  assert.equal(response.results[1].title, document.title);
  assert.equal(response.results[1].text, document.text);
  assert.equal(reads, 2);
  assert.ok(cacheClears >= 2);
});

function staticFixture(runtime = 'transformersjs') {
  const manifest = {
    format: 'vectorsearch-prebuilt-v1',
    encoding: 'float32-le',
    normalized: true,
    count: 2,
    dimensions: runtime === 'embeddinggemma' ? 768 : 2,
    model: {
      runtime,
      url: runtime === 'embeddinggemma' ? 'onnx-community/embeddinggemma-300m-ONNX' : 'Xenova/all-MiniLM-L6-v2',
      revision: 'a'.repeat(40), dtype: 'q4', sequenceLength: 512
    },
    documentPrefix: 'title: none | text: ',
    queryPrefix: 'task: search result | query: ',
    chunks: 'chunks.json',
    vectors: 'vectors.f32'
  };
  const chunks = [
    { id: 1, ...document },
    { id: 2, text: 'Other passage', title: 'Other section', sourceUrl: sourceUrl.replace(/#.*/, '#updating-a-deployment') }
  ];
  const vectors = new Float32Array(manifest.count * manifest.dimensions);
  vectors[0] = 1;
  vectors[manifest.dimensions + 1] = 1;
  const keywordIndex = {
    format: 'vectorsearch-keywords-v1', count: 2, averageLength: 3,
    lengths: [3, 3], terms: { query: [1, 3] }
  };
  const fetch = async url => {
    const path = new URL(url).pathname;
    return {
      ok: true,
      url: new URL(url).href,
      json: async () => path.endsWith('manifest.json') ? manifest :
        path.endsWith('keywords.json') ? keywordIndex : chunks,
      arrayBuffer: async () => vectors.buffer
    };
  };
  return { manifest, chunks, vectors, fetch, keywordIndex };
}

test('retained text import preserves vectors in a final partial batch', async () => {
  class Stub {}
  const VectorSearch = await loadClass('src/VectorSearch.js', 'VectorSearch', {
    './VectorStore.js': { VectorStore: Stub },
    './CosineSimilarity.js': { CosineSimilarity: Stub },
    './EmbeddingModel.js': { EmbeddingModel: Stub }
  });
  const search = new VectorSearch({});
  const records = [];
  search.setDb = () => {};
  search.deleteGPUVectorCache = async () => {};
  search.embeddingModel.getEmbeddingTransformers = async texts => ({
    embedding: texts.flatMap(text => [text.length, 1])
  });
  search.vectorStore.storeBatch = async batch => records.push(...structuredClone(batch));
  await search.storeTexts(['one', 'longer', 'last'], 'test', undefined, 2);
  assert.deepEqual(records, [
    { text: 'one', embedding: [3, 1] }, { text: 'longer', embedding: [6, 1] },
    { text: 'last', embedding: [4, 1] }
  ]);
});

async function staticSearch(fixture, backend = { setBackend: async () => true, ready: async () => {} }, options = {}) {
  class NoDatabase {
    setDb() { throw new Error('IndexedDB must not be used'); }
    storeBatch() { throw new Error('IndexedDB must not be used'); }
    getAllVectors() { throw new Error('IndexedDB must not be used'); }
  }
  class Model {
    constructor(runtime, config) { this.config = config; }
    getTaskPrefix(task) {
      return task === 'document' ? 'title: none | text: ' : 'task: search result | query: ';
    }
    async load(url, revision) {
      assert.equal(revision, fixture.manifest.model.revision);
    }
    async getEmbeddingTransformers(text, task) {
      assert.equal(text, 'My query');
      assert.equal(task, 'query');
      return { embedding: [1, 0] };
    }
  }
  class Similarity {
    async loadPrebuiltMatrix(vectors, count, dimensions) {
      assert.deepEqual(Array.from(vectors), Array.from(fixture.vectors));
      assert.equal(count, fixture.manifest.count);
      assert.equal(dimensions, fixture.manifest.dimensions);
      this.cachedMatrix = {};
    }
    async cosineSimilarityPrebuilt(vector, topK) {
      assert.equal(topK, 2);
      return { values: [0.9, 0.1], indices: [0, 1] };
    }
  }
  const VectorSearch = await loadClass('src/VectorSearch.js', 'VectorSearch', {
    '@tensorflow/tfjs': backend,
    './VectorStore.js': { VectorStore: NoDatabase },
    './CosineSimilarity.js': { CosineSimilarity: Similarity },
    './EmbeddingModel.js': { EmbeddingModel: Model }
  }, { fetch: fixture.fetch });
  return VectorSearch.fromPrebuiltDataset('https://example.test/data/manifest.json', undefined, options);
}

test('offline datasets retain original citations but navigate to localhost heading anchors', async () => {
  const options = { offline: true, assetBaseUrl: 'http://localhost:8080/' };
  await assert.rejects(staticSearch(staticFixture(), undefined, options), /requires the prebuilt EmbeddingGemma/);
  const search = await staticSearch(staticFixture('embeddinggemma'), undefined, options);
  assert.equal(search.embeddingModel.config.offline, true);
  assert.equal(search.prebuiltData.chunks[0].originalSourceUrl, sourceUrl);
  assert.equal(search.prebuiltData.chunks[0].sourceUrl,
    'http://localhost:8080/docs/concepts/workloads/controllers/deployment/#creating-a-deployment');
});

test('static dataset loading skips IndexedDB and document embedding, returning source anchors', async () => {
  const search = await staticSearch(staticFixture());
  const { embedding } = await search.getEmbedding('My query', 'query');
  const response = await search.search(embedding, 0.25);
  assert.equal(response.results.length, 1);
  assert.equal(response.results[0].sourceUrl, sourceUrl);
  assert.equal(response.results[0].title, document.title);
  assert.deepEqual(Array.from(response.results[0].vector), [1, 0]);
  assert.equal(response.bestScore, 0.9);
  assert.throws(() => search.setDb('another'), /read-only/);
  await assert.rejects(search.storeBatch([]), /read-only/);
  await assert.rejects(search.storeTexts(['text'], 'db'), /read-only/);
  await assert.rejects(search.search(embedding, 0.25, 'db'), /Do not select/);
  await assert.rejects(search.search([1], 0.25), /Invalid query/);
  await assert.rejects(search.search([NaN, 0], 0.25), /Invalid query/);
  const empty = await search.search(embedding, 1);
  assert.equal(empty.results.length, 0);
  assert.equal(empty.bestScore, 0);
});

test('static dataset loading rejects truncated vectors, mismatched preprocessing and unsafe source URLs', async () => {
  let fixture = staticFixture();
  fixture.manifest.count = 3;
  await assert.rejects(staticSearch(fixture), /dimensions\/count/);
  fixture = staticFixture();
  fixture.manifest.queryPrefix = 'different query prefix';
  await assert.rejects(staticSearch(fixture), /preprocessing/);
  fixture = staticFixture();
  fixture.chunks[0].sourceUrl = 'javascript:alert(1)';
  await assert.rejects(staticSearch(fixture), /source citation/);
  fixture = staticFixture();
  fixture.manifest.model.revision = 'main';
  await assert.rejects(staticSearch(fixture), /manifest/);
});

test('static retrieval waits for a working backend and falls back when WebGPU initialization fails', async () => {
  const attempts = [];
  let ready = false;
  const search = await staticSearch(staticFixture(), {
    async setBackend(name) {
      attempts.push(name);
      if (name === 'webgpu') throw new Error('No GPU adapter');
      if (name === 'webgl') return false;
      return true;
    },
    async ready() {
      ready = true;
    }
  });
  assert.deepEqual(attempts, ['webgpu', 'webgl', 'cpu']);
  assert.equal(ready, true);
  assert.equal(search.searchBackend, 'cpu');
  const response = await search.search([1, 0], 0.25);
  assert.equal(response.results[0].sourceUrl, sourceUrl);
  await assert.rejects(staticSearch(staticFixture(), {
    setBackend: async () => false,
    ready: async () => { throw new Error('Must not use an uninitialized backend'); }
  }), /No TensorFlow.js search backend/);
});

test('searchText loads offline keyword postings and combines them with vector retrieval', async () => {
  const fixture = staticFixture();
  fixture.manifest.keywords = 'keywords.json';
  const search = await staticSearch(fixture);
  const response = await search.searchText('My query', 0.25);
  assert.equal(response.results[0].sourceUrl, fixture.chunks[1].sourceUrl);
  assert.ok(response.results[0].keywordScore > 0);
  assert.ok(response.results[0].hybridScore > 0);
  assert.equal(response.bestIndex, 1);
  await assert.rejects(search.searchText(' '), /must not be empty/);
  fixture.keywordIndex.terms.query = [99, 1];
  await assert.rejects(staticSearch(fixture), /keyword postings/);
});

test('EmbeddingGemma queries use its task prefix and sentence_embedding output rather than generic mean pooling', async () => {
  const EmbeddingModel = await loadClass('src/EmbeddingModel.js', 'EmbeddingModel');
  const model = new EmbeddingModel('embeddinggemma', { sequenceLength: 512 });
  let disposed = 0;
  model.gemmaTokenizer = async (text, options) => {
    assert.equal(text, 'task: search result | query: How do I create a egress?');
    assert.equal(options.max_length, 512);
    return { input_ids: [1, 2] };
  };
  model.model = async () => ({
    sentence_embedding: { data: new Float32Array([0.5, 0.25]), dispose() { disposed++; } },
    last_hidden_state: { dispose() { disposed++; } }
  });
  const result = await model.getEmbeddingTransformers('How do I create a egress?', 'query');
  assert.deepEqual(Array.from(result.embedding), [0.5, 0.25]);
  assert.equal(disposed, 2);
});

test('offline Gemma loads use same-origin metadata paths and disable remote model access', async () => {
  const env = { allowRemoteModels: true, backends: { onnx: { wasm: {} } } };
  const revision = 'a'.repeat(40);
  const tokenizer = {};
  const loadedModel = {};
  const EmbeddingModel = await loadClass('src/EmbeddingModel.js', 'EmbeddingModel', {
    '@huggingface/transformers': {
      env,
      AutoTokenizer: {
        async from_pretrained(id, options) {
          assert.equal(env.allowRemoteModels, false);
          assert.equal(env.allowLocalModels, true);
          assert.equal(env.localModelPath, `/models/${revision}/`);
          assert.equal(options.local_files_only, true);
          return tokenizer;
        }
      },
      AutoModel: {
        async from_pretrained(id, options) {
          assert.equal(options.local_files_only, true);
          assert.equal(options.device, 'webgpu');
          return loadedModel;
        }
      }
    }
  });
  const model = new EmbeddingModel('embeddinggemma', { offline: true, assetBaseUrl: 'http://localhost:8080/' });
  await model.load('onnx-community/embeddinggemma-300m-ONNX', revision);
  assert.equal(model.gemmaTokenizer, tokenizer);
  assert.equal(model.model, loadedModel);
  assert.equal(env.backends.onnx.wasm.wasmPaths, 'http://localhost:8080/vendor/onnx/');
  assert.equal(env.backends.onnx.wasm.numThreads, 1);
});

test('Gemma 4 streams grounded answers from local assets using a fresh conversation per question', async () => {
  const prompts = [];
  let created = 0;
  let deleted = 0;
  let runtime;
  const engine = {
    async createConversation(config) {
      assert.equal(config.preface.extra_context.enable_thinking, false);
      assert.match(config.preface.messages[0].content, /only the supplied documentation/);
      return {
        async *sendMessageStreaming(prompt) {
          prompts.push(prompt);
          yield { content: [{ text: 'Use the documentation ' }] };
          yield { content: [{ type: 'text', text: '[1].' }], channels: { thought: 'Not answer text' } };
        },
        async delete() { deleted++; }
      };
    }
  };
  const Generator = await loadClass('src/OfflineAnswerGenerator.js', 'OfflineAnswerGenerator', {
    '@litert-lm/core': {
      Engine: {
        async create(settings) {
          created++;
          assert.deepEqual(Array.from(new Uint8Array(await new Response(settings.model).arrayBuffer())), [1, 2]);
          return engine;
        }
      },
      loadLiteRtLm: async path => { runtime = path; },
      SamplerType: { GREEDY: 3 }
    }
  }, {
    TransformStream,
    fetch: async url => {
      assert.equal(url.origin, 'http://localhost:8080');
      assert.match(url.pathname, /gemma-4-E2B-it-web\.litertlm$/);
      return new Response(new Uint8Array([1, 2]));
    }
  });
  const generator = new Generator('http://localhost:8080/');
  const updates = [];
  assert.equal(await generator.generate('First question', [document], text => updates.push(text)),
    'Use the documentation [1].');
  await generator.generate('Second question', [{ title: 'Egress', text: 'Allow outgoing traffic.' }]);
  assert.equal(runtime, 'http://localhost:8080/vendor/litert-lm/');
  assert.equal(created, 1);
  assert.equal(deleted, 2);
  assert.match(prompts[0], /\[1\] Creating a Deployment\nCreate a Deployment\./);
  assert.doesNotMatch(prompts[1], /First question|Create a Deployment/);
  assert.deepEqual(updates, ['Use the documentation ', 'Use the documentation [1].']);
  await generator.generate('Make it three replicas', [document], undefined, undefined, [
    { role: 'user', content: 'First question' },
    { role: 'assistant', content: 'Use the documentation [1].' }
  ]);
  assert.match(prompts[2], /user: First question/);
  assert.match(prompts[2], /assistant: Use the documentation \(historical source 1\)/);
  assert.equal(created, 1);
  assert.equal(deleted, 3);
  await assert.rejects(generator.generate('Question', []), /No documentation passages/);
});

test('follow-up retrieval uses recent questions rather than generated answers', async () => {
  const { buildRetrievalQuery } = await loadClass('src/OfflineAnswerGenerator.js', undefined, {
    '@litert-lm/core': { Engine: {}, loadLiteRtLm() {}, SamplerType: {} }
  });
  const history = [
    { role: 'user', content: 'Old unrelated question' },
    { role: 'user', content: 'Create a Deployment with nginx' },
    { role: 'assistant', content: 'An unsupported generated claim' },
    { role: 'user', content: 'Expose its HTTP port' }
  ];
  const query = buildRetrievalQuery('Change it to three replicas', history);
  assert.match(query, /^Change it to three replicas/);
  assert.match(query, /Create a Deployment with nginx/);
  assert.match(query, /Expose its HTTP port/);
  assert.doesNotMatch(query, /Old unrelated|unsupported generated/);
  assert.equal(buildRetrievalQuery('New problem'), 'New problem');
});

test('chat prompts bound multilingual history and passages and separate historical citations', async () => {
  const { buildAnswerPrompt } = await loadClass('src/OfflineAnswerGenerator.js', undefined, {
    '@litert-lm/core': { Engine: {}, loadLiteRtLm() {}, SamplerType: {} }
  });
  const history = [
    { role: 'user', content: 'Forgotten old question' },
    { role: 'assistant', content: 'Old answer' },
    { role: 'user', content: 'Create nginx' },
    { role: 'assistant', content: 'Apply this [1].\n```yaml\nvalues: [1]\n```' },
    { role: 'user', content: 'Expose nginx' },
    { role: 'assistant', content: 'Use a Service [2].' }
  ];
  const sources = Array.from({ length: 5 }, (_, index) => ({
    title: `Current source ${index + 1}`, text: 'Kubernetes documentation '.repeat(200) + '中文'.repeat(200)
  }));
  const prompt = buildAnswerPrompt('Make it three replicas', sources, history);
  assert.ok(new TextEncoder().encode(prompt).length <= 6000);
  assert.doesNotMatch(prompt, /Forgotten old question|Old answer/);
  assert.match(prompt, /historical source 1/);
  assert.match(prompt, /```yaml\nvalues: \[1\]\n```/);
  assert.match(prompt, /\[5\] Current source 5/);
  const unicode = buildAnswerPrompt('請解釋 Deployment', [{ title: '中文'.repeat(100), text: '中文'.repeat(4000) }], [
    { role: 'user', content: '中文'.repeat(2000) }
  ]);
  assert.ok(new TextEncoder().encode(unicode).length <= 6000);
  assert.throws(() => buildAnswerPrompt('中文'.repeat(1000), sources), /shorten/);
});

test('Gemma 4 cleans up failed streams and can answer a later question', async () => {
  const Generator = await loadClass('src/OfflineAnswerGenerator.js', 'OfflineAnswerGenerator', {
    '@litert-lm/core': { Engine: {}, loadLiteRtLm() {}, SamplerType: { GREEDY: 3 } }
  });
  const generator = new Generator('http://localhost:8080/');
  let deleted = 0;
  let fail = true;
  generator.engine = {
    async createConversation() {
      return {
        async *sendMessageStreaming() {
          if (fail) throw new Error('GPU stopped');
          yield { content: 'Answer [1].' };
        },
        async delete() { deleted++; }
      };
    }
  };
  await assert.rejects(generator.generate('Question', [document]), /GPU stopped/);
  assert.equal(generator.busy, false);
  assert.equal(deleted, 1);
  fail = false;
  assert.equal(await generator.generate('Question', [document]), 'Answer [1].');
  assert.equal(deleted, 2);
});

function modelStorage() {
  const files = new Map();
  const state = { files, quota: 100, failWrites: false, aborts: 0 };
  state.storage = {
    async estimate() { return { quota: state.quota, usage: 0 }; },
    async persist() { return false; },
    async getDirectory() {
      return {
        async getFileHandle(name) {
          if (!files.has(name)) files.set(name, new Uint8Array());
          return {
            async getFile() { return new Blob([files.get(name)]); },
            async createWritable() {
              const bytes = [];
              return new WritableStream({
                write(chunk) {
                  if (state.failWrites) throw new Error('Storage quota exceeded');
                  bytes.push(...chunk);
                },
                close() { files.set(name, Uint8Array.from(bytes)); },
                abort() { state.aborts++; }
              });
            }
          };
        }
      };
    }
  };
  return state;
}

async function cachedGenerator(storage, fetch) {
  const module = await loadClass('src/OfflineAnswerGenerator.js', undefined, {
    '@litert-lm/core': {
      Engine: {
        async create({ model }) {
          assert.deepEqual(Array.from(new Uint8Array(await new Response(model).arrayBuffer())), [1, 2]);
          return {};
        }
      },
      loadLiteRtLm: async () => {},
      SamplerType: { GREEDY: 3 }
    }
  }, { navigator: { storage }, TransformStream, fetch });
  module.GEMMA4_MODEL.bytes = 2;
  return { generator: new module.OfflineAnswerGenerator('http://localhost:8080/'), model: module.GEMMA4_MODEL };
}

test('Gemma 4 reuses a complete disk cache in a fresh page context without fetching the model', async () => {
  const disk = modelStorage();
  let fetched = 0;
  const first = await cachedGenerator(disk.storage, async () => {
    fetched++;
    return new Response(new Uint8Array([1, 2]));
  });
  const statuses = [];
  await first.generator.load(status => statuses.push(status));
  assert.equal(first.generator.modelCached, true);
  assert.ok(statuses.some(status => status.startsWith('Saving Gemma 4')));
  const refreshed = await cachedGenerator(disk.storage, () => {
    assert.fail('Refreshing must not fetch a cached model.');
  });
  await refreshed.generator.load(status => statuses.push(status));
  assert.equal(refreshed.generator.modelCached, true);
  assert.equal(fetched, 1);
  assert.ok(statuses.some(status => status.startsWith('Loading cached Gemma 4')));
});

test('Gemma 4 ignores stale and incomplete cached models and atomically replaces them', async () => {
  const disk = modelStorage();
  const { generator, model } = await cachedGenerator(disk.storage, async () =>
    new Response(new Uint8Array([1, 2])));
  disk.files.set('gemma4-previous-model.litertlm', new Uint8Array([8, 8]));
  disk.files.set(`gemma4-${model.sha256}.litertlm`, new Uint8Array([9]));
  await generator.load();
  assert.deepEqual(Array.from(disk.files.get(`gemma4-${model.sha256}.litertlm`)), [1, 2]);
  assert.equal(generator.modelCached, true);
});

test('Gemma 4 does not commit a truncated download and can cache successfully on a later load', async () => {
  const disk = modelStorage();
  let fetched = 0;
  const fetch = async () => new Response(new Uint8Array(++fetched === 1 ? [1] : [1, 2]));
  const { generator, model } = await cachedGenerator(disk.storage, fetch);
  await generator.load();
  assert.equal(generator.modelCached, false);
  assert.equal(disk.aborts, 1);
  assert.equal(disk.files.get(`gemma4-${model.sha256}.litertlm`).length, 0);
  const refreshed = await cachedGenerator(disk.storage, fetch);
  await refreshed.generator.load();
  assert.equal(refreshed.generator.modelCached, true);
  assert.equal(fetched, 3);
});

test('Gemma 4 remains usable and reports unavailable, full or failed browser storage', async t => {
  for (const reason of ['unavailable', 'full', 'write failure']) {
    await t.test(reason, async () => {
      const disk = modelStorage();
      if (reason === 'full') disk.quota = 1;
      if (reason === 'write failure') disk.failWrites = true;
      const storage = reason === 'unavailable' ? {
        async getDirectory() { throw new Error('Storage denied'); }
      } : disk.storage;
      const { generator } = await cachedGenerator(storage, async () => new Response(new Uint8Array([1, 2])));
      generator.modelCached = true;
      const statuses = [];
      await generator.load(status => statuses.push(status));
      assert.equal(generator.modelCached, false);
      assert.ok(statuses.some(status => status.includes('without caching')));
    });
  }
});

test('answer rendering creates only supplied citation links and keeps model HTML and code inert', async () => {
  const { parseHTML } = await import('linkedom');
  const dom = parseHTML('<html><body><div id="answer"></div></body></html>').document;
  const render = await loadClass('web/js/answer-citations.js', 'renderAnswer', {}, { document: dom });
  const container = dom.getElementById('answer');
  const text = '<img src="https://outside.test/image" onerror="bad()"> Answer [1]. Unknown [99].\n```yaml\nvalues: [1]\n```';
  const cited = render(container, text, [document]);
  assert.deepEqual(Array.from(cited), [1]);
  assert.equal(container.querySelectorAll('a').length, 1);
  assert.equal(container.querySelector('a').getAttribute('href'), sourceUrl);
  assert.equal(container.querySelectorAll('img,script').length, 0);
  assert.ok(container.textContent.startsWith('<img src="https://outside.test/image" onerror="bad()"> Answer [1]. Unknown [99].'));
  assert.equal(container.querySelector('code').textContent, 'values: [1]\n');
  assert.equal(container.querySelector('.copy-code').textContent, 'Copy code');
});

test('code buttons copy exact YAML and shell blocks without fences or linked citations', async () => {
  const { parseHTML } = await import('linkedom');
  const window = parseHTML('<html><body><div id="answer"></div></body></html>');
  const copied = [];
  const render = await loadClass('web/js/answer-citations.js', 'renderAnswer', {}, {
    document: window.document, navigator: { clipboard: { async writeText(text) { copied.push(text); } } }
  });
  const container = window.document.getElementById('answer');
  const yaml = 'apiVersion: v1\nkind: Service\nspec:\n  selector:\n    app: my-app\n  values: [1]\n';
  const shell = 'kubectl apply -f service.yaml\r\n';
  const cited = render(container, `Example [1].\n\`\`\`yaml\n${yaml}\`\`\`\nThen:\n\`\`\`bash\r\n${shell}\`\`\`\nVerify [1].`, [document]);
  assert.deepEqual(Array.from(cited), [1]);
  const buttons = container.querySelectorAll('.copy-code');
  assert.equal(buttons.length, 2);
  for (const button of buttons) button.dispatchEvent(new window.Event('click'));
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(copied, [yaml, shell.replace(/\r\n/g, '\n')]);
  assert.equal(buttons[0].textContent, 'Copied');
  assert.equal(container.querySelectorAll('.answer-code a').length, 0);
  assert.equal(container.querySelectorAll('a').length, 2);
});

test('streaming code stays inert and becomes a complete copyable block on the next render', async () => {
  const { parseHTML } = await import('linkedom');
  const window = parseHTML('<html><body><div id="answer"></div></body></html>');
  const copied = [];
  const render = await loadClass('web/js/answer-citations.js', 'renderAnswer', {}, {
    document: window.document, navigator: { clipboard: { async writeText(text) { copied.push(text); } } }
  });
  const container = window.document.getElementById('answer');
  const partial = '```html\n<script>bad()</script>\nvalues: [1]';
  assert.equal(render(container, partial, [document]).size, 0);
  assert.equal(container.querySelector('.copy-code').textContent, 'Copy code');
  assert.equal(container.querySelectorAll('script,a').length, 0);
  render(container, partial + '\n```\nReference [1].', [document]);
  container.querySelector('.copy-code').dispatchEvent(new window.Event('click'));
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(copied, ['<script>bad()</script>\nvalues: [1]\n']);
  assert.equal(container.querySelectorAll('script').length, 0);
  assert.equal(container.querySelectorAll('.answer-code').length, 1);
});

test('denied clipboard access selects code for manual copying without claiming success', async () => {
  const { parseHTML } = await import('linkedom');
  const window = parseHTML('<html><body><div id="answer"></div></body></html>');
  let selected;
  let added = 0;
  window.document.createRange = () => ({ selectNodeContents(node) { selected = node; } });
  window.document.getSelection = () => ({ removeAllRanges() {}, addRange() { added++; } });
  const render = await loadClass('web/js/answer-citations.js', 'renderAnswer', {}, {
    document: window.document, navigator: { clipboard: { async writeText() { throw new Error('Denied'); } } }
  });
  const container = window.document.getElementById('answer');
  render(container, '```yaml\n  app: nginx\n```', []);
  container.querySelector('.copy-code').dispatchEvent(new window.Event('click'));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(selected.textContent, '  app: nginx\n');
  assert.equal(added, 1);
  assert.equal(container.querySelector('.copy-code').textContent, 'Selected; press Ctrl+C');
});

test('Markdown formatting works while raw HTML, arbitrary links and images stay inert', async () => {
  const { parseHTML } = await import('linkedom');
  const window = parseHTML('<html><body><div id="answer"></div></body></html>');
  const render = await loadClass('web/js/answer-citations.js', 'renderAnswer', {}, { document: window.document });
  const container = window.document.getElementById('answer');
  render(container, '**Example of creating a Service:**\n\n- Run `kubectl apply`.\n- Check [1].\n\n' +
    '[outside](https://outside.test/) ![image](https://outside.test/image)\n' +
    '<https://outside.test/>\n<script>bad()</script>\n\nInline `values: [1]`.', [document]);
  assert.equal(container.querySelector('strong').textContent, 'Example of creating a Service:');
  assert.equal(container.querySelectorAll('li').length, 2);
  assert.equal(container.querySelectorAll('a').length, 1);
  assert.equal(container.querySelectorAll('img,script').length, 0);
  assert.equal(container.querySelectorAll('code a').length, 0);
});
test('older answer citations retain their own sources when opening the document panel', async () => {
  const { parseHTML } = await import('linkedom');
  const window = parseHTML('<html><body><pre id="first"></pre><pre id="second"></pre></body></html>');
  const render = await loadClass('web/js/answer-citations.js', 'renderAnswer', {}, { document: window.document });
  const first = window.document.getElementById('first');
  const second = window.document.getElementById('second');
  const other = { title: 'Egress', sourceUrl: 'http://localhost:8080/docs/concepts/services-networking/network-policies/#egress' };
  const opened = [];
  render(first, 'First answer [1].', [document], (source, index) => opened.push([source, index]));
  render(second, 'Second answer [1].', [other], (source, index) => opened.push([source, index]));
  second.querySelector('a').dispatchEvent(new window.Event('click', { cancelable: true }));
  first.querySelector('a').dispatchEvent(new window.Event('click', { cancelable: true }));
  assert.equal(opened[0][0], other);
  assert.equal(opened[1][0], document);
  assert.equal(opened[1][1], 1);
});
