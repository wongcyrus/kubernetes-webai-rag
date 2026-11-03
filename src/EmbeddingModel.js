/**
 * Loads query embedding models with Transformers.js.
 * Coded by Jason Mayes 2026.
 * Modified for kubernetes-webai-rag: pinned offline ONNX EmbeddingGemma support.
 */
export class EmbeddingModel {
  constructor(modelRuntime, modelConfig = {}) {
    this.model = undefined;
    this.runtime = modelRuntime;
    this.modelConfig = modelConfig;
  }

  /**
   * Loads a Transformers.js embedding model.
   * @param {string} modelUrl Model repository identifier.
   * @param {string} [revision='main'] Transformers.js model revision.
   * @return {Promise<void>}
   */
  async load(modelUrl, revision = 'main') {
    const { AutoModel, AutoTokenizer, env, pipeline } = await import('@huggingface/transformers');
    if (this.runtime === 'embeddinggemma') {
      if (this.modelConfig.offline) {
        env.allowRemoteModels = false;
        env.allowLocalModels = true;
        // Metadata probing requires a same-origin path rather than an absolute HTTP URL.
        env.localModelPath = new URL(`models/${revision}/`, this.modelConfig.assetBaseUrl).pathname;
        env.backends.onnx.wasm.wasmPaths = new URL('vendor/onnx/', this.modelConfig.assetBaseUrl).href;
        env.backends.onnx.wasm.numThreads = 1;
      }
      const local = this.modelConfig.offline ? { local_files_only: true } : {};
      this.gemmaTokenizer = await AutoTokenizer.from_pretrained(modelUrl, { revision, ...local });
      this.model = await AutoModel.from_pretrained(modelUrl, {
        revision,
        dtype: this.modelConfig.dtype || 'q4',
        ...local,
        device: 'webgpu'
      });
    } else {
      // Transformers.js model.
      // Load the feature-extraction pipeline (Transformers.js does its own caching).
      this.model = await pipeline('feature-extraction', modelUrl, { revision });
    }
  }

  /**
   * Returns the instruction prefix for the given task type.
   * @param {string|Object} [taskType='query'] The task type or task options.
   * @param {string} [title='none'] The document title (only used for 'document' task type).
   * @return {string} The prefix to prepend.
   */
  getTaskPrefix(taskType = 'query', title = 'none') {
    let task = 'query';
    let docTitle = title;
    if (typeof taskType === 'object' && taskType !== null) {
      task = taskType.type || 'query';
      docTitle = taskType.title || 'none';
    } else if (typeof taskType === 'string') {
      task = taskType;
    }
    task = task.toLowerCase();

    if (task === 'document' || task === 'search_document') {
      return `title: ${docTitle} | text: `;
    }

    const taskDescriptions = {
      query: 'search result',
      retrieval: 'search result',
      search_query: 'search result',
      classification: 'classification',
      clustering: 'clustering',
      similarity: 'sentence similarity',
      code_retrieval: 'code retrieval',
    };

    const desc = taskDescriptions[task] || 'search result';
    return `task: ${desc} | query: `;
  }

  /**
   * Generates an embedding for the given tokens.
   * @param {string|Array<string>} textBatch Text or array of text to embed.
   * @param {string|Object} [taskType='query'] The task type or task options.
   * @return {Promise<{embedding: Array<number>}>} The generated embedding.
   */
  async getEmbeddingTransformers(textBatch, taskType = 'query') {
    if (this.runtime === 'embeddinggemma') {
      const prefix = this.getTaskPrefix(taskType);
      const texts = Array.isArray(textBatch) ? textBatch.map(text => prefix + text) : prefix + textBatch;
      const inputs = await this.gemmaTokenizer(texts, {
        padding: true, truncation: true, max_length: this.modelConfig.sequenceLength || 512
      });
      const output = await this.model(inputs);
      try {
        return { embedding: Array.from(output.sentence_embedding.data) };
      } finally {
        for (const tensor of Object.values(output)) tensor.dispose?.();
      }
    }
    if (this.runtime === 'transformersjs') {
      const prefix = this.getTaskPrefix(taskType);
      let formattedBatch;
      if (Array.isArray(textBatch)) {
        formattedBatch = textBatch.map(text => prefix + text);
      } else {
        formattedBatch = prefix + textBatch;
      }
      const queryResult = await this.model(formattedBatch, { pooling: 'mean', normalize: true });
      const queryVector = Array.from(queryResult.data);
      return {
        embedding: queryVector
      };
    }
  }
}
