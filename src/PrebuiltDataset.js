async function fetchChecked(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Prebuilt dataset download failed (${response.status}): ${url}`);
  return response;
}

export async function fetchPrebuiltManifest(url) {
  const response = await fetchChecked(url);
  const manifest = await response.json();
  if (manifest.format !== 'vectorsearch-prebuilt-v1' || manifest.encoding !== 'float32-le' ||
      manifest.normalized !== true || !Number.isSafeInteger(manifest.count) || manifest.count < 1 ||
      !Number.isSafeInteger(manifest.dimensions) || manifest.dimensions < 1 ||
      !Number.isSafeInteger(manifest.count * manifest.dimensions * 4) ||
      typeof manifest.chunks !== 'string' || typeof manifest.vectors !== 'string' ||
      !['transformersjs', 'embeddinggemma'].includes(manifest.model?.runtime) || typeof manifest.model.url !== 'string' ||
      (manifest.model.runtime === 'embeddinggemma' &&
        (manifest.dimensions !== 768 || manifest.model.url !== 'onnx-community/embeddinggemma-300m-ONNX' ||
          !['q4', 'q8', 'fp32'].includes(manifest.model.dtype) ||
          !Number.isInteger(manifest.model.sequenceLength) || manifest.model.sequenceLength < 32 ||
          manifest.model.sequenceLength > 2048)) ||
      (manifest.keywords !== undefined && typeof manifest.keywords !== 'string') ||
      !/^[a-f0-9]{40}$/.test(manifest.model.revision) ||
      typeof manifest.documentPrefix !== 'string' || typeof manifest.queryPrefix !== 'string') {
    throw new Error('Invalid or unsupported prebuilt dataset manifest.');
  }
  return { manifest, baseUrl: response.url || new URL(url, globalThis.location?.href).href };
}

export async function fetchPrebuiltData(manifest, baseUrl) {
  const [chunks, buffer, keywordIndex] = await Promise.all([
    fetchChecked(new URL(manifest.chunks, baseUrl)).then(response => response.json()),
    fetchChecked(new URL(manifest.vectors, baseUrl)).then(response => response.arrayBuffer()),
    manifest.keywords ? fetchChecked(new URL(manifest.keywords, baseUrl)).then(response => response.json()) : null
  ]);
  if (!Array.isArray(chunks) || chunks.length !== manifest.count ||
      buffer.byteLength !== manifest.count * manifest.dimensions * 4) {
    throw new Error('Prebuilt dataset files do not match the manifest dimensions/count.');
  }
  const ids = new Set();
  for (const chunk of chunks) {
    if (!Number.isSafeInteger(chunk.id) || ids.has(chunk.id) ||
        typeof chunk.text !== 'string' || typeof chunk.title !== 'string' ||
        typeof chunk.sourceUrl !== 'string' || !/^https?:\/\//.test(chunk.sourceUrl)) {
      throw new Error('Invalid prebuilt dataset chunk or source citation.');
    }
    ids.add(chunk.id);
  }
  const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
  const vectors = littleEndian ? new Float32Array(buffer) : new Float32Array(manifest.count * manifest.dimensions);
  const view = littleEndian ? null : new DataView(buffer);
  for (let index = 0; index < vectors.length; index++) {
    if (view) vectors[index] = view.getFloat32(index * 4, true);
    if (!Number.isFinite(vectors[index])) throw new Error('Prebuilt dataset contains invalid vector values.');
  }
  if (keywordIndex) {
    if (keywordIndex.format !== 'vectorsearch-keywords-v1' || keywordIndex.count !== manifest.count ||
        !Array.isArray(keywordIndex.lengths) || keywordIndex.lengths.length !== manifest.count ||
        !keywordIndex.lengths.every(length => Number.isFinite(length) && length > 0) ||
        !Number.isFinite(keywordIndex.averageLength) || keywordIndex.averageLength <= 0 ||
        !keywordIndex.terms || typeof keywordIndex.terms !== 'object') {
      throw new Error('Invalid offline keyword index.');
    }
    for (const postings of Object.values(keywordIndex.terms)) {
      if (!Array.isArray(postings) || postings.length % 2) throw new Error('Invalid keyword postings.');
      let previous = -1;
      for (let offset = 0; offset < postings.length; offset += 2) {
        const row = postings[offset];
        if (!Number.isInteger(row) || row <= previous || row >= manifest.count ||
            !Number.isFinite(postings[offset + 1]) || postings[offset + 1] <= 0) {
          throw new Error('Invalid keyword postings.');
        }
        previous = row;
      }
    }
  }
  return { manifest, chunks, vectors, keywordIndex };
}
