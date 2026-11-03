/********************************************************************* 
 * A simple vector store using IndexedDB for persistence.
 * Coded by Jason Mayes 2026. 
 * Modified for kubernetes-webai-rag: preserve source citations in stored records.
 *--------------------------------------------------------------------
 * Connect with me on social if aquestions or comments:
 *
 * LinkedIn: https://www.linkedin.com/in/webai/
 * Twitter / X: https://x.com/jason_mayes
 * Github: https://github.com/jasonmayes
 * CodePen: https://codepen.io/jasonmayes
 *********************************************************************/
export class VectorStore {
  /**
   * @param {string} dbName Name of the IndexedDB database.
   */
  constructor(dbName = 'unnamed') {
    this.dbName = dbName;
    this.db = null;
  }

  /**
   * Sets the database name to use for storage and retrieval.
   * @param {string} name The name of the database to use.
   */
  setDb(name) {
    if (this.dbName !== name) {
      this.dbName = name;
      if (this.db) {
        this.db.close();
        this.db = null;
      }
    }
  }

  /**
   * Initializes the IndexedDB database.
   * @return {!Promise<!IDBDatabase>}
   * @private
   */
  async initDb() {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      // Increment version to 2 to trigger upgrade if you already had version 1
      const request = indexedDB.open(this.dbName, 1);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        // Single store for both text and vectors
        if (!db.objectStoreNames.contains('embeddings')) {
          db.createObjectStore('embeddings', { keyPath: 'id', autoIncrement: true });
        }
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        resolve(this.db);
      };

      request.onerror = (e) => reject(e.target.error);
    });
  }


  /**
   * Stores embeddings, text, and optional source citations.
   * @param {!Array<{embedding: Array<number>, text: string, sourceUrl: string=, title: string=}>} items
   * @return {!Promise<void>}
   */
  async storeBatch(items) {
    const db = await this.initDb();
    const transaction = db.transaction(['embeddings'], 'readwrite');
    const store = transaction.objectStore('embeddings');

    return new Promise((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = (e) => reject(e.target.error);

      for (const item of items) {
        // item.embedding should be a Float32Array for max speed
        store.add({
          text: item.text,
          embedding: item.embedding,
          ...(item.sourceUrl !== undefined ? { sourceUrl: item.sourceUrl } : {}),
          ...(item.title !== undefined ? { title: item.title } : {})
        });
      }
    });
  }


  /**
   * Fetches all vectors from the database.
   * @return {!Promise<!Array<{id: number, embedding: !Array<number>}>>}
   */
  async getAllVectors() {
    const db = await this.initDb();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['embeddings'], 'readonly');
      const store = transaction.objectStore('embeddings');
      const vectors = [];

      // A cursor lets us step through records without loading 
      // the whole store into memory at once.
      const request = store.openCursor();

      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          // We only grab the ID and the embedding
          vectors.push({
            id: cursor.value.id,
            embedding: cursor.value.embedding // Float32Array
          });
          cursor.continue();
        } else {
          // No more results
          resolve(vectors);
        }
      };

      request.onerror = (e) => reject(e.target.error);
    });
  }


  /**
   * Fetches the text for a given ID.
   * @param {number} id The ID of the text to fetch.
   * @return {!Promise<string>}
   */
  async getTextByID(id) {
    const document = await this.getDocumentByID(id);
    return document ? document.text : null;
  }


  /**
   * Fetches text and optional source citations without returning the embedding.
   * @param {number} id
   * @return {!Promise<?{text: string, sourceUrl: string=, title: string=}>}
   */
  async getDocumentByID(id) {
    const db = await this.initDb();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['embeddings'], 'readonly');
      const store = transaction.objectStore('embeddings');
      const request = store.get(id);

      request.onsuccess = () => {
        const item = request.result;
        resolve(item ? {
          text: item.text,
          ...(item.sourceUrl !== undefined ? { sourceUrl: item.sourceUrl } : {}),
          ...(item.title !== undefined ? { title: item.title } : {})
        } : null);
      };
      request.onerror = (e) => reject(e.target.error);
    });
  }
}
