import { createHash } from 'node:crypto';
import { parseHTML } from 'linkedom';
import { keywordTerms } from '../src/KeywordSearch.js';

export const DOCUMENT_PREFIX = 'title: none | text: ';
export const QUERY_PREFIX = 'task: search result | query: ';

export function documentInput(chunk) {
  return `title: ${(chunk.embeddingTitle ?? chunk.title) || 'none'} | text: ${chunk.text}`;
}

export function buildKeywordIndex(chunks) {
  const terms = Object.create(null);
  const lengths = [];
  for (let row = 0; row < chunks.length; row++) {
    const frequencies = new Map();
    for (const [text, weight] of [[chunks[row].text, 1], [chunks[row].title, 3]]) {
      for (const term of keywordTerms(text)) frequencies.set(term, (frequencies.get(term) || 0) + weight);
    }
    lengths.push([...frequencies.values()].reduce((sum, count) => sum + count, 0) || 1);
    for (const [term, count] of frequencies) {
      (terms[term] ||= []).push(row, count);
    }
  }
  return {
    format: 'vectorsearch-keywords-v1',
    count: chunks.length,
    averageLength: lengths.reduce((sum, length) => sum + length, 0) / chunks.length,
    lengths,
    terms
  };
}

export function pageUrls(sitemap, sourceUrl) {
  const urls = new Map();
  for (const match of sitemap.matchAll(/<loc>(.*?)<\/loc>/g)) {
    const url = new URL(match[1].replaceAll('&amp;', '&'));
    if (url.origin !== new URL(sourceUrl).origin || !url.pathname.startsWith('/docs/')) continue;
    const path = decodeURIComponent(url.pathname).replace(/^\//, '').replace(/\/$/, '');
    // Docsy's print page uses Hugo File.UniqueID: MD5 of the content-relative filename.
    for (const file of [`${path}.md`, `${path}/_index.md`, `${path}/index.md`]) {
      urls.set(`pg-${createHash('md5').update(file).digest('hex')}`, url.href);
    }
  }
  return urls;
}

function plainText(node) {
  if (node.nodeType === 3) return node.textContent;
  if (node.nodeType !== 1) return '';
  if (node.tagName === 'PRE') return `\n${node.textContent}\n`;
  if (node.tagName === 'BR') return '\n';
  const text = Array.from(node.childNodes, plainText).join('');
  return /^(P|DIV|LI|UL|OL|TABLE|TR|DL|DT|DD)$/.test(node.tagName) ? `${text}\n` : text;
}

export function extractSections(html, sitemap, sourceUrl = 'https://kubernetes.io/docs/_print/') {
  const { document } = parseHTML(html);
  const urls = pageUrls(sitemap, sourceUrl);
  const sections = [];
  const unresolvedPages = [];
  const pages = [...document.querySelectorAll('main .td-content')]
    .filter(page => page.querySelector('h1[id^="pg-"]'));
  if (!pages.length) throw new Error('No Kubernetes print-page documentation blocks found.');

  for (const page of pages) {
    const heading = page.querySelector('h1[id^="pg-"]');
    const pageTitle = heading.textContent.replace(/^[\d.]+\s*-\s*/, '').trim();
    const pageUrl = urls.get(heading.id);
    if (!pageUrl) unresolvedPages.push({ title: pageTitle, sourceUrl: `${sourceUrl}#${heading.id}` });
    for (const element of page.querySelectorAll(
      'script, style, nav, button, .pageinfo, .feedback, .td-heading-self-link, .copy-code-icon'
    )) element.remove();

    let title = pageTitle;
    let citation = pageUrl || `${sourceUrl}#${heading.id}`;
    let paragraphs = [];
    const flush = () => {
      const text = paragraphs.join('\n\n').trim();
      if (text) sections.push({ title, sourceUrl: citation, text });
      paragraphs = [];
    };
    const visit = node => {
      if (node.nodeType === 3) {
        if (node.textContent.trim()) paragraphs.push(node.textContent.trim());
        return;
      }
      if (node.nodeType !== 1) return;
      if (/^H[1-6]$/.test(node.tagName)) {
        if (node === heading) return;
        flush();
        title = `${pageTitle} > ${node.textContent.trim()}`;
        if (pageUrl && node.id) {
          const url = new URL(pageUrl);
          url.hash = node.id;
          citation = url.href;
        }
        return;
      }
      if (/^(P|PRE|UL|OL|TABLE|DL)$/.test(node.tagName)) {
        const text = plainText(node).trim();
        if (text) paragraphs.push(text);
        return;
      }
      for (const child of node.childNodes) visit(child);
    };
    for (const node of page.childNodes) visit(node);
    flush();
  }
  return { sections, unresolvedPages };
}

export function chunkSections(sections, tokenizer, maxTokens = 128, overlapTokens = 16) {
  if (!Number.isInteger(maxTokens) || maxTokens < 32 ||
      !Number.isInteger(overlapTokens) || overlapTokens < 0 || overlapTokens >= maxTokens / 2) {
    throw new Error('Use maxTokens >= 32 and overlapTokens < half of maxTokens.');
  }
  const chunks = [];
  for (const section of sections) {
    const titleIds = tokenizer(section.title, { add_special_tokens: false }).input_ids.data;
    const titleLimit = Math.min(32, Math.floor(maxTokens / 3));
    let embeddingTitle = section.title;
    if (titleIds.length > titleLimit) {
      const half = Math.floor(titleLimit / 2);
      const selected = [...titleIds.slice(0, half), ...titleIds.slice(-(titleLimit - half))];
      embeddingTitle = tokenizer.decode(selected.map(Number), { skip_special_tokens: true });
    }
    const tokenCount = text => tokenizer(documentInput({ ...section, embeddingTitle, text })).input_ids.data.length;
    // Keep the original text (including code whitespace), rather than decoding BERT tokens.
    const words = [...section.text.matchAll(/\S+\s*/g)].map(match => match[0]);
    let start = 0;
    while (start < words.length) {
      let low = start + 1;
      let high = Math.min(words.length, start + maxTokens);
      let end = start;
      while (low <= high) {
        const middle = Math.floor((low + high) / 2);
        if (tokenCount(words.slice(start, middle).join('').trim()) <= maxTokens) {
          end = middle;
          low = middle + 1;
        } else {
          high = middle - 1;
        }
      }
      if (end === start) {
        const characters = Array.from(words[start]);
        let left = 1;
        let right = characters.length;
        let cut = 0;
        while (left <= right) {
          const middle = Math.floor((left + right) / 2);
          if (tokenCount(characters.slice(0, middle).join('')) <= maxTokens) {
            cut = middle;
            left = middle + 1;
          } else {
            right = middle - 1;
          }
        }
        if (!cut) throw new Error(`Cannot fit a character within the token limit in ${section.sourceUrl}.`);
        words.splice(start, 1, characters.slice(0, cut).join(''), characters.slice(cut).join(''));
        continue;
      }
      chunks.push({
        id: chunks.length + 1,
        text: words.slice(start, end).join('').trim(),
        embeddingTitle,
        sourceUrl: section.sourceUrl,
        title: section.title
      });
      if (end === words.length) break;
      let next = end;
      const prefixTokens = tokenCount('');
      while (next > start + 1 &&
             tokenCount(words.slice(next - 1, end).join('').trim()) - prefixTokens <= overlapTokens) {
        next--;
      }
      start = next;
    }
  }
  return chunks;
}
