import { OfflineVectorSearch as VectorSearch, OfflineAnswerGenerator, buildRetrievalQuery } from '../OfflineVectorSearch-min.js';
import { renderAnswer } from './answer-citations.js';

const FORM = document.getElementById('search-form');
const QUERY = document.getElementById('query');
const BUTTON = document.getElementById('search-button');
const STATUS = document.getElementById('status');
const RESULTS = document.getElementById('results');
const CHAT = document.getElementById('chat-history');
const EMPTY = document.getElementById('chat-empty');
const STOP = document.getElementById('stop-button');
const NEW = document.getElementById('new-chat');
const READER = document.getElementById('document-reader');
const DOCUMENT_EMPTY = document.getElementById('document-empty');
const SOURCE_CONTEXT = document.getElementById('source-context');
const OPEN_DOCUMENT = document.getElementById('open-document');
const WORKSPACE = document.getElementById('workspace');
const CHAT_TAB = document.getElementById('chat-tab');
const DOCUMENTS_TAB = document.getElementById('documents-tab');
const MANIFEST_URL = new URL('../data/kubernetes/manifest.json', import.meta.url);
const generator = new OfflineAnswerGenerator();
let vectorSearch;
let turns = [];
let selectedTurn;
let activeTurn;

function showPanel(panel) {
  WORKSPACE.dataset.panel = panel;
  CHAT_TAB.setAttribute('aria-pressed', String(panel === 'chat'));
  DOCUMENTS_TAB.setAttribute('aria-pressed', String(panel === 'documents'));
}

CHAT_TAB.addEventListener('click', () => showPanel('chat'));
DOCUMENTS_TAB.addEventListener('click', () => showPanel('documents'));

function openDocument(source) {
  const url = new URL(source.sourceUrl, location.href);
  if (url.origin !== location.origin || !url.pathname.startsWith('/docs/')) {
    STATUS.innerText = 'Only bundled local documentation can be opened in this panel.';
    return;
  }
  READER.src = url.href;
  READER.hidden = false;
  DOCUMENT_EMPTY.hidden = true;
  OPEN_DOCUMENT.href = url.href;
  for (const article of RESULTS.children) {
    article.classList.toggle('source-active', article.dataset.source === url.href);
  }
}

function selectTurn(turn) {
  selectedTurn = turn;
  for (const item of turns) item.article.classList.toggle('selected', item === turn);
  SOURCE_CONTEXT.textContent = `Sources for: ${turn.question}`;
  RESULTS.replaceChildren();
  READER.hidden = true;
  READER.removeAttribute('src');
  DOCUMENT_EMPTY.hidden = false;
  OPEN_DOCUMENT.href = new URL('../docs/', import.meta.url).href;
  for (const [index, result] of turn.sources.entries()) {
    const article = document.createElement('article');
    article.className = 'source-card';
    article.dataset.source = result.sourceUrl;
    const title = document.createElement('h3');
    const source = document.createElement('a');
    source.href = result.sourceUrl;
    source.textContent = `[${index + 1}] ${result.title || result.sourceUrl}`;
    source.addEventListener('click', event => {
      event.preventDefault();
      openDocument(result);
      showPanel('documents');
    });
    title.append(source);
    const score = document.createElement('p');
    score.className = 'subtitle';
    score.textContent = `Similarity: ${result.score.toFixed(3)}${result.keywordScore !== undefined ?
      ` | Keyword relevance: ${result.keywordScore.toFixed(2)}` : ''}`;
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = 'Retrieved excerpt';
    const passage = document.createElement('pre');
    passage.textContent = result.text;
    details.append(summary, passage);
    article.append(title, score, details);
    RESULTS.append(article);
  }
  if (turn.sources.length) openDocument(turn.sources[0]);
  else DOCUMENT_EMPTY.textContent = 'No matching documentation for this question.';
}

function renderTurn(turn, text) {
  turn.answer = text;
  return renderAnswer(turn.output, text, turn.sources, source => {
    selectTurn(turn);
    openDocument(source);
    showPanel('documents');
  });
}

function scrollChat() {
  CHAT.scrollTop = CHAT.scrollHeight;
}

STOP.addEventListener('click', () => {
  if (activeTurn) activeTurn.stopped = true;
  generator.cancel();
  STOP.disabled = true;
  STATUS.innerText = 'Stopping answer generation...';
});

NEW.addEventListener('click', () => {
  turns = [];
  selectedTurn = undefined;
  CHAT.replaceChildren(EMPTY);
  EMPTY.hidden = false;
  RESULTS.replaceChildren();
  READER.hidden = true;
  READER.removeAttribute('src');
  DOCUMENT_EMPTY.hidden = false;
  DOCUMENT_EMPTY.textContent = 'Select a source or an answer citation to read its local documentation here.';
  SOURCE_CONTEXT.textContent = 'Sources will appear after your first question.';
  OPEN_DOCUMENT.href = new URL('../docs/', import.meta.url).href;
  QUERY.value = '';
  showPanel('chat');
  QUERY.focus();
  STATUS.innerText = 'New chat ready. Loaded models and browser model cache are kept.';
});

QUERY.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!BUTTON.disabled) FORM.requestSubmit();
  }
});

FORM.addEventListener('submit', async event => {
  event.preventDefault();
  const text = QUERY.value.trim();
  if (!vectorSearch || !text || BUTTON.disabled) return;
  const history = turns.filter(turn => turn.complete).flatMap(turn => [
    { role: 'user', content: turn.question }, { role: 'assistant', content: turn.answer }
  ]);
  BUTTON.disabled = true;
  NEW.disabled = true;
  QUERY.disabled = true;
  QUERY.value = '';
  EMPTY.hidden = true;
  showPanel('chat');
  const article = document.createElement('article');
  article.className = 'chat-turn';
  const question = document.createElement('div');
  question.className = 'user-message';
  question.textContent = text;
  const heading = document.createElement('div');
  heading.className = 'answer-heading';
  const label = document.createElement('strong');
  label.textContent = 'Gemma 4';
  const sourcesButton = document.createElement('button');
  sourcesButton.type = 'button';
  sourcesButton.className = 'secondary';
  sourcesButton.textContent = 'View sources';
  const output = document.createElement('div');
  output.className = 'answer-text';
  output.setAttribute('aria-live', 'polite');
  output.textContent = 'Searching local documentation...';
  const outcome = document.createElement('p');
  outcome.className = 'subtitle turn-status';
  heading.append(label, sourcesButton);
  article.append(question, heading, output, outcome);
  CHAT.append(article);
  const turn = { question: text, answer: '', sources: [], complete: false, article, output };
  activeTurn = turn;
  turns.push(turn);
  sourcesButton.addEventListener('click', () => {
    selectTurn(turn);
    showPanel('documents');
  });
  selectTurn(turn);
  scrollChat();
  STATUS.innerText = 'Embedding your question and searching the prebuilt documentation...';
  try {
    const { results } = await vectorSearch.searchText(buildRetrievalQuery(text, history), 0.25);
    turn.sources = results;
    if (selectedTurn === turn) selectTurn(turn);
    if (!results.length) {
      output.textContent = 'There is not enough matching local documentation to generate an answer.';
      outcome.textContent = 'No matching passages. Try a more specific question.';
      STATUS.innerText = outcome.textContent;
      return;
    }
    output.textContent = 'Preparing Gemma 4...';
    const answer = await generator.generate(text, results, text => {
      const nearBottom = CHAT.scrollHeight - CHAT.scrollTop - CHAT.clientHeight < 100;
      renderTurn(turn, text);
      if (nearBottom) scrollChat();
    }, status => {
      STATUS.innerText = status;
      const generating = status.startsWith('Gemma 4 E2B is generating');
      STOP.hidden = !generating;
      STOP.disabled = !generating;
    }, history);
    const cited = renderTurn(turn, answer);
    turn.complete = !generator.cancelled && Boolean(answer.trim());
    outcome.textContent = generator.cancelled ? 'Stopped. This partial answer is not used as follow-up context.' :
      cited.size ? `${cited.size} source citation(s). Click a citation to read its documentation.` :
      'No source citations. Verify this answer against the retrieved documentation.';
    STATUS.innerText = generator.cancelled ? 'Generation stopped. You can send another message.' : 'Ready for a follow-up question.';
    if (!generator.modelCached) STATUS.innerText += ' Model caching is unavailable; refreshing reloads it from the local server.';
  } catch (error) {
    if (!turn.answer) output.textContent = turn.stopped ? 'Generation stopped before an answer was returned.' :
      'Could not generate an answer. Retrieved sources remain available.';
    outcome.textContent = turn.stopped ? 'Stopped. This partial answer is not used as follow-up context.' :
      `Answer failed: ${error.message}`;
    STATUS.innerText = outcome.textContent;
  } finally {
    activeTurn = undefined;
    BUTTON.disabled = false;
    NEW.disabled = false;
    QUERY.disabled = false;
    STOP.hidden = true;
    QUERY.focus();
  }
});

try {
  vectorSearch = await VectorSearch.fromPrebuiltDataset(MANIFEST_URL, STATUS);
  const { count, partial } = vectorSearch.prebuiltData.manifest;
  STATUS.innerText = `Ready: ${count.toLocaleString()} prebuilt chunks (${vectorSearch.searchBackend})${partial ? ' (partial smoke-test dataset)' : ''}. Gemma 4 loads on the first message.`;
  BUTTON.disabled = false;
  NEW.disabled = false;
} catch (error) {
  STATUS.innerText = `Load failed: ${error.message}`;
}
