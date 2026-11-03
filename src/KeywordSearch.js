const STOP_WORDS = new Set(
  'a an the how do does i we you my to is are was were what can of in on for and or with it this that why'.split(' ')
);

export function keywordTerms(text) {
  return (text.toLowerCase().match(/[a-z0-9]+/g) || [])
    .filter(term => !STOP_WORDS.has(term))
    .map(term => {
      if (term.endsWith('ies') && term.length > 4) return term.slice(0, -3) + 'y';
      if (term.endsWith('sses')) return term.slice(0, -2);
      if (term.length > 3 && term.endsWith('s') && !term.endsWith('ss') && !term.endsWith('is')) {
        return term.slice(0, -1);
      }
      return term;
    });
}

export function searchKeywords(index, query, limit = 50) {
  const scores = new Map();
  for (const term of new Set(keywordTerms(query))) {
    if (!Object.hasOwn(index.terms, term)) continue;
    const postings = index.terms[term];
    const frequency = postings.length / 2;
    const idf = Math.log(1 + (index.count - frequency + 0.5) / (frequency + 0.5));
    for (let offset = 0; offset < postings.length; offset += 2) {
      const row = postings[offset];
      const count = postings[offset + 1];
      const normalization = 1.2 * (0.25 + 0.75 * index.lengths[row] / index.averageLength);
      const score = idf * count * 2.2 / (count + normalization);
      scores.set(row, (scores.get(row) || 0) + score);
    }
  }
  return [...scores].map(([row, score]) => ({ row, score }))
    .sort((left, right) => right.score - left.score || left.row - right.row)
    .slice(0, limit);
}

export function rankHybridResults(semantic, keyword, data, queryVector, threshold, maxMatches) {
  const candidates = new Map();
  const pageScores = new Map();
  const pageOf = source => source.includes('/_print/') ? source : source.split('#')[0];
  for (const [list, field] of [[semantic, 'score'], [keyword, 'keywordScore']]) {
    const seen = new Set();
    const seenPages = new Set();
    let rank = 0;
    for (const match of list) {
      const source = data.chunks[match.row].sourceUrl;
      const page = pageOf(source);
      if (!seenPages.has(page)) {
        seenPages.add(page);
        pageScores.set(page, (pageScores.get(page) || 0) + 1 / (60 + seenPages.size));
      }
      if (seen.has(source)) continue;
      seen.add(source);
      // Fuse heading evidence even when each retriever prefers a different chunk.
      const candidate = candidates.get(source) || { row: match.row, hybridScore: 0, keywordScore: 0 };
      candidate[field] = match.score;
      candidate.hybridScore += 1 / (60 + ++rank);
      candidates.set(source, candidate);
    }
  }
  const norm = Math.sqrt(Array.from(queryVector).reduce((sum, value) => sum + value * value, 0));
  for (const match of candidates.values()) {
    if (match.score !== undefined) continue;
    let dot = 0;
    for (let column = 0; column < data.manifest.dimensions; column++) {
      dot += queryVector[column] * data.vectors[match.row * data.manifest.dimensions + column];
    }
    match.score = norm ? dot / norm : 0;
  }
  const ranked = [...candidates.values()]
    .filter(match => match.score >= threshold || match.keywordScore > 0)
    .sort((left, right) => right.hybridScore - left.hybridScore ||
      right.keywordScore - left.keywordScore || right.score - left.score || left.row - right.row);
  const pages = new Map();
  for (const match of ranked) {
    const page = pageOf(data.chunks[match.row].sourceUrl);
    if (!pages.has(page)) pages.set(page, match);
  }
  const diverse = [...pages].sort(([leftPage, left], [rightPage, right]) =>
    pageScores.get(rightPage) - pageScores.get(leftPage) ||
    right.hybridScore - left.hybridScore || right.keywordScore - left.keywordScore || right.score - left.score
  ).map(([, match]) => match).slice(0, maxMatches);
  const selected = new Set(diverse.map(match => match.row));
  for (const match of ranked) {
    if (diverse.length === maxMatches) break;
    if (!selected.has(match.row)) diverse.push(match);
  }
  return diverse;
}
