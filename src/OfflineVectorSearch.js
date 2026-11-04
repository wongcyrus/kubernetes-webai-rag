import { VectorSearch } from './VectorSearch.js';
export { OfflineAnswerGenerator, buildRetrievalQuery } from './OfflineAnswerGenerator.js';

export class OfflineVectorSearch extends VectorSearch {
  static async fromPrebuiltDataset(manifestUrl, statusElement) {
    return VectorSearch.fromPrebuiltDataset(manifestUrl, statusElement, {
      offline: true,
      assetBaseUrl: new URL('./', import.meta.url).href
    });
  }
}
