import { OfflineVectorSearch as VectorSearch, OfflineAnswerGenerator } from '../OfflineVectorSearch-min.js';
import { renderAnswer } from './answer-citations.js';

const FORM = document.getElementById('search-form');
const QUERY = document.getElementById('query');
const BUTTON = document.getElementById('search-button');
const STATUS = document.getElementById('status');
const RESULTS = document.getElementById('results');
const ANSWER = document.getElementById('answer');
const ANSWER_TEXT = document.getElementById('answer-text');
const STOP = document.getElementById('stop-button');
const MANIFEST_URL = new URL('../data/kubernetes/manifest.json', import.meta.url);
let vectorSearch;
const generator = new OfflineAnswerGenerator();
STOP.addEventListener('click', () => {
  generator.cancel();
  STOP.disabled = true;
  STATUS.innerText = 'Stopping answer generation...';
});

FORM.addEventListener('submit', async event => {
  event.preventDefault();
  const text = QUERY.value.trim();
  if (!vectorSearch || !text || BUTTON.disabled) return;
  BUTTON.disabled = true;
  STATUS.innerText = 'Embedding your query and searching the prebuilt vectors...';
  RESULTS.replaceChildren();
  ANSWER.hidden = true;
  ANSWER_TEXT.replaceChildren();
  try {
    const { results } = await vectorSearch.searchText(text, 0.25);
    for (const result of results) {
      const article = document.createElement('article');
      article.className = 'visual-sub-section';
      const title = document.createElement('h3');
      const source = document.createElement('a');
      source.href = result.sourceUrl;
      source.textContent = result.title || result.sourceUrl;
      source.target = '_blank';
      source.rel = 'noopener noreferrer';
      title.append(source);
      const score = document.createElement('p');
      score.className = 'subtitle';
      score.textContent = `Semantic similarity: ${result.score.toFixed(3)}${result.keywordScore !== undefined ?
        ` | Keyword relevance: ${result.keywordScore.toFixed(2)}` : ''} | Source: ${result.sourceUrl}`;
      const passage = document.createElement('pre');
      passage.style.whiteSpace = 'pre-wrap';
      passage.textContent = result.text;
      article.append(title, score, passage);
      RESULTS.append(article);
    }
    if (!results.length) {
      STATUS.innerText = 'No matching passages; there is not enough local documentation to generate an answer.';
      return;
    }
    ANSWER.hidden = false;
    let cited = new Set();
    const answer = await generator.generate(text, results, output => {
      cited = renderAnswer(ANSWER_TEXT, output, results);
    }, status => {
      STATUS.innerText = status;
      const generating = status.startsWith('Gemma 4 E2B is generating');
      STOP.hidden = !generating;
      STOP.disabled = !generating;
    });
    cited = renderAnswer(ANSWER_TEXT, answer, results);
    STATUS.innerText = generator.cancelled ? 'Generation stopped. The partial answer and retrieved passages are shown.' :
      cited.size ? `Answer generated locally with ${cited.size} source citation(s). Verify against the documentation.` :
      'Answer generated without source citations. Verify it against the retrieved passages below.';
    if (!generator.modelCached) STATUS.innerText += ' Browser caching is unavailable; refreshing will reload the model from the local server.';
  } catch (error) {
    STATUS.innerText = `Answer failed: ${error.message}. Retrieved passages remain available below.`;
  } finally {
    BUTTON.disabled = false;
    STOP.hidden = true;
  }
});

try {
  vectorSearch = await VectorSearch.fromPrebuiltDataset(MANIFEST_URL, STATUS);
  const { count, partial } = vectorSearch.prebuiltData.manifest;
  STATUS.innerText = `Ready: ${count.toLocaleString()} prebuilt chunks (${vectorSearch.searchBackend})${partial ? ' (partial smoke-test dataset)' : ''}. Gemma 4 E2B loads on the first question.`;
  BUTTON.disabled = false;
} catch (error) {
  STATUS.innerText = `Load failed: ${error.message}`;
}
