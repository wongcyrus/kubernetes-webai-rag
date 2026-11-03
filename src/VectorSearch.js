import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

import { VectorStore } from './VectorStore.js';
import { CosineSimilarity } from './CosineSimilarity.js';
import { EmbeddingModel } from './EmbeddingModel.js';
import { fetchPrebuiltManifest, fetchPrebuiltData } from './PrebuiltDataset.js';
import { searchKeywords, rankHybridResults } from './KeywordSearch.js';

/**
 * VectorSearch - A master orchestrator class for the RAG system.
 * Coded by Jason Mayes 2026.
 * Modified for kubernetes-webai-rag: prebuilt offline retrieval and hybrid ranking.
 */
export class VectorSearch {
  /**
   * @param {Object} modelConfig Config object for VectorSearch setup.
   */
  constructor(modelConfig) {
    this.modelUrl = modelConfig.url;
    this.modelRuntime = modelConfig.runtime;
    this.modelRevision = modelConfig.revision || 'main';

    this.vectorStore = new VectorStore();
    this.cosineSimilarity = new CosineSimilarity();
    this.embeddingModel = new EmbeddingModel(this.modelRuntime, modelConfig);

    this.allStoredData = undefined;
    this.lastDBName = '';
    this.prebuiltData = undefined;
  }

  /**
   * Loads an offline-built static dataset and its query model. No IndexedDB writes
   * or document embedding are performed.
   */
  static async fromPrebuiltDataset(manifestUrl, statusElement, options = {}) {
    const { manifest, baseUrl } = await fetchPrebuiltManifest(manifestUrl);
    if (options.offline && manifest.model.runtime !== 'embeddinggemma') {
      throw new Error('The offline exam bundle requires the prebuilt EmbeddingGemma dataset.');
    }
    const search = new VectorSearch({ ...manifest.model, ...options });
    if (manifest.documentPrefix !== search.embeddingModel.getTaskPrefix('document') ||
        manifest.queryPrefix !== search.embeddingModel.getTaskPrefix('query')) {
      throw new Error('Prebuilt dataset preprocessing does not match the query model.');
    }
    const [data] = await Promise.all([
      fetchPrebuiltData(manifest, baseUrl),
      search.load(statusElement)
    ]);
    if (options.offline) {
      data.chunks = data.chunks.map(chunk => {
        const original = new URL(chunk.sourceUrl);
        return {
          ...chunk,
          originalSourceUrl: chunk.sourceUrl,
          sourceUrl: new URL(original.pathname.slice(1) + original.hash, options.assetBaseUrl).href
        };
      });
    }
    if (statusElement) statusElement.innerText = `Loading prebuilt vectors for ${search.searchBackend} search...`;
    await search.cosineSimilarity.loadPrebuiltMatrix(data.vectors, manifest.count, manifest.dimensions);
    search.prebuiltData = data;
    return search;
  }

  /**
   * Initializes the embedding model and tokenizer.
   * @return {Promise<void>}
   */
  async load(STATUS_EL) {
    const backends = this.modelRuntime === 'embeddinggemma' ? ['webgpu'] : ['webgpu', 'webgl', 'cpu'];
    let initialized = false;
    for (const backend of backends) {
      if (STATUS_EL) STATUS_EL.innerText = `Initializing ${backend} search backend...`;
      try {
        if (!await tf.setBackend(backend)) continue;
        await tf.ready();
        this.searchBackend = backend;
        initialized = true;
        break;
      } catch {
        // Transformers.js embeddings can still be searched without WebGPU.
      }
    }
    if (!initialized) {
      throw new Error(this.modelRuntime === 'embeddinggemma' ?
        'WebGPU could not be initialized. This embedding model requires WebGPU.' :
        'No TensorFlow.js search backend could be initialized.');
    }
    
    if (STATUS_EL) {
      STATUS_EL.innerText = 'Initializing Model Runtime...';
    }

    if (STATUS_EL) {
      STATUS_EL.innerText = 'Loading Tokenizer & Embedding Model...';
    }
    await this.embeddingModel.load(this.modelUrl, this.modelRevision);
    
  }

  /**
   * Sets the current database name in the vector store.
   * @param {string} dbName
   */
  setDb(dbName) {
    if (this.prebuiltData) throw new Error('Prebuilt datasets are read-only. Rebuild the dataset offline to update it.');
    this.vectorStore.setDb(dbName);
  }

  /**
   * Encodes text and generates an embedding.
   * @param {string} text
   * @param {string|Object} [taskType='query'] The task type or task options.
   * @return {Promise<{embedding: Array<number>}>}
   */
  async getEmbedding(text, taskType = 'query') {
    return this.embeddingModel.getEmbeddingTransformers(text, taskType);
  }

  /**
   * Deletes the GPU vector cache.
   */
  async deleteGPUVectorCache() {
    await this.cosineSimilarity.deleteGPUVectorCache();
  }

  /**
   * Performs a vector search, returning text and any stored source citations.
   * @param {Array<number>} queryVector
   * @param {number} threshold
   * @param {string} selectedDB
   * @param {number} maxMatches
   * @return {Promise<{results: Array<Object>, bestScore: number, bestIndex: number}>}
   */
  async search(queryVector, threshold, selectedDB, maxMatches = 5) {
    if (this.prebuiltData) {
      if (selectedDB !== undefined) throw new Error('Do not select an IndexedDB database for a prebuilt dataset.');
      return this.searchPrebuiltDataset(queryVector, threshold, maxMatches);
    }
    let matrixData = undefined;

    if (this.lastDBName !== selectedDB || !this.allStoredData) {
      await this.deleteGPUVectorCache();
      this.lastDBName = selectedDB;
      this.allStoredData = await this.vectorStore.getAllVectors();
      matrixData = this.allStoredData.map(item => item.embedding);
    } else {
      matrixData = this.allStoredData.map(item => item.embedding);
    }

    if (matrixData.length === 0)  {
      console.warn('No data in chosen vector store. Store some data first before searching');
      return { results: [], bestScore: 0, bestIndex: 0 };
    }
    const { values, indices } = await this.cosineSimilarity.cosineSimilarityTFJSGPUMatrix(matrixData, queryVector, maxMatches);
    
    let topMatches = [];
    let bestIndex = 0;
    let bestScore = 0;

    for (let i = 0; i < values.length; i++) {
      if (values[i] >= threshold) {
        if (topMatches.length < maxMatches) {
          topMatches.push({
            id: this.allStoredData[indices[i]].id,
            score: values[i],
            vector: this.allStoredData[indices[i]].embedding
          });
          if (values[i] > bestScore) {
            bestIndex = topMatches.length - 1;
            bestScore = values[i];
          }
        }
      }
    }

    const results = [];
    for (const match of topMatches) {
      const document = await this.vectorStore.getDocumentByID(match.id);
      results.push({ ...match, ...(document || { text: null }) });
    }

    return { results, bestScore, bestIndex };
  }

  async searchPrebuiltDataset(queryVector, threshold = 0.25, maxMatches = 5, queryText) {
    const { manifest, chunks, vectors } = this.prebuiltData;
    if (queryVector.length !== manifest.dimensions ||
        !Array.from(queryVector).every(Number.isFinite) ||
        !Number.isFinite(threshold) || !Number.isInteger(maxMatches) || maxMatches < 1) {
      throw new Error('Invalid query vector, threshold, or match count for the prebuilt dataset.');
    }
    if (!this.cosineSimilarity.cachedMatrix) {
      await this.cosineSimilarity.loadPrebuiltMatrix(vectors, manifest.count, manifest.dimensions);
    }
    const hybrid = this.prebuiltData.keywordIndex && queryText !== undefined;
    const { values, indices } = await this.cosineSimilarity.cosineSimilarityPrebuilt(
      queryVector, Math.min(hybrid ? Math.max(50, maxMatches) : maxMatches, manifest.count)
    );
    if (hybrid) {
      const semantic = Array.from(values, (score, index) => ({ row: indices[index], score }));
      const keyword = searchKeywords(this.prebuiltData.keywordIndex, queryText, Math.max(50, maxMatches));
      const matches = rankHybridResults(semantic, keyword, this.prebuiltData, queryVector, threshold, maxMatches);
      const results = matches.map(({ row, ...scores }) => ({
        ...chunks[row], ...scores,
        vector: vectors.subarray(row * manifest.dimensions, (row + 1) * manifest.dimensions)
      }));
      const bestScore = results.length ? Math.max(...results.map(result => result.score)) : 0;
      const bestIndex = results.findIndex(result => result.score === bestScore);
      return { results, bestScore, bestIndex: Math.max(0, bestIndex) };
    }
    const results = [];
    for (let index = 0; index < values.length; index++) {
      if (values[index] < threshold) continue;
      const row = indices[index];
      results.push({
        ...chunks[row],
        score: values[index],
        vector: vectors.subarray(row * manifest.dimensions, (row + 1) * manifest.dimensions)
      });
    }
    return { results, bestScore: results[0]?.score || 0, bestIndex: 0 };
  }

  /**
   * Embeds a query and combines vector retrieval with the offline-built keyword index.
   */
  async searchText(queryText, threshold = 0.25, maxMatches = 5) {
    if (!this.prebuiltData) throw new Error('Load a prebuilt dataset before searching text.');
    if (typeof queryText !== 'string' || !queryText.trim()) throw new Error('Search query must not be empty.');
    const { embedding } = await this.getEmbedding(queryText, 'query');
    return this.searchPrebuiltDataset(embedding, threshold, maxMatches, queryText);
  }

  /**
   * Stores multiple items in the vector store.
   * @param {Array<{embedding: Array<number>, text: string, sourceUrl: string=, title: string=}>} storagePayload
   */
  async storeBatch(storagePayload) {
    if (this.prebuiltData) throw new Error('Prebuilt datasets are read-only. Rebuild the dataset offline to update it.');
    await this.vectorStore.storeBatch(storagePayload);
    this.allStoredData = undefined;
    await this.deleteGPUVectorCache();
  }

  /**
   * Embeds and stores multiple texts.
   * @param {Array<string>} texts
   * @param {string} dbName
   * @param {HTMLElement} statusElement
   * @param {number} batchSize
   * @param {string|Object} [taskType='document'] The task type or task options.
   */
  async storeTexts(texts, dbName, statusElement, batchSize = 2, taskType = 'document') {
    this.setDb(dbName);
    let textBatch = [];
    for (let i = 0; i < texts.length; i++) {
      if (statusElement) {
        statusElement.innerText = `Embedding paragraph ${i + 1} of ${texts.length}...`;
      }
      textBatch.push(texts[i]);
      if (textBatch.length >= batchSize || i === texts.length - 1) {
        const { embedding } = await this.embeddingModel.getEmbeddingTransformers(textBatch, taskType);
        const dimensions = embedding.length / textBatch.length;
        await this.vectorStore.storeBatch(textBatch.map((text, index) => ({
          text, embedding: embedding.slice(index * dimensions, (index + 1) * dimensions)
        })));
        textBatch = [];
      }
    }
    this.allStoredData = undefined;
    await this.deleteGPUVectorCache();
  }
}
