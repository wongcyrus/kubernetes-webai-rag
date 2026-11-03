import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { chunkSections, extractSections, documentInput, buildKeywordIndex } from '../scripts/kubernetes-chunks.mjs';
import { searchKeywords, rankHybridResults } from '../src/KeywordSearch.js';

const pageUrl = 'https://kubernetes.io/docs/concepts/workloads/controllers/deployment/';
const id = `pg-${createHash('md5').update('docs/concepts/workloads/controllers/deployment.md').digest('hex')}`;
const sitemap = `<urlset><url><loc>${pageUrl}</loc></url></urlset>`;

test('print-page extraction retains canonical heading anchors and code, excluding the print navigation', () => {
  const html = `<main>
    <div class="td-content"><h1>Documentation</h1><p>Print navigation</p></div>
    <div class="td-content">
      <h1 id="${id}">3.4.2.1 - Deployments</h1>
      <p>Manage replicated workloads.</p>
      <h2 id="creating-a-deployment">Creating a Deployment<a class="td-heading-self-link">noise</a></h2>
      <p>Apply the YAML &amp; check the rollout.</p>
      <div class="highlight"><div class="copy-code-icon">Copy noise</div>
        <pre><code>kind: Deployment\nspec:\n  replicas: 3</code></pre></div>
      <h2 id="updating-a-deployment">Updating a Deployment</h2><p>Update the image.</p>
    </div>
  </main>`;
  const { sections, unresolvedPages } = extractSections(html, sitemap);
  assert.equal(sections.length, 3);
  assert.equal(sections[0].sourceUrl, pageUrl);
  assert.equal(sections[1].sourceUrl, `${pageUrl}#creating-a-deployment`);
  assert.equal(sections[1].title, 'Deployments > Creating a Deployment');
  assert.match(sections[1].text, /YAML & check/);
  assert.match(sections[1].text, /spec:\n  replicas: 3/);
  assert.doesNotMatch(sections[1].text, /noise|Print navigation/);
  assert.equal(sections[2].sourceUrl, `${pageUrl}#updating-a-deployment`);
  assert.deepEqual(unresolvedPages, []);
});

test('unresolved canonical pages cite their unique print-page anchor rather than an invented URL', () => {
  const { sections, unresolvedPages } = extractSections(
    '<main><div class="td-content"><h1 id="pg-unknown">1 - Unlisted</h1><p>Content.</p></div></main>', ''
  );
  assert.equal(sections[0].sourceUrl, 'https://kubernetes.io/docs/_print/#pg-unknown');
  assert.equal(unresolvedPages.length, 1);
  assert.throws(() => extractSections('<main>No docs</main>', sitemap), /No Kubernetes/);
});

test('chunking handles long unbroken code and overlap without exceeding the model token budget', () => {
  const tokenizer = text => ({ input_ids: { data: new Array(text.length + 2) } });
  const sections = [{
    text: `first second third fourth fifth sixth seventh\n${'x'.repeat(75)}\nlast words`,
    title: 'Example',
    sourceUrl: `${pageUrl}#creating-a-deployment`
  }];
  const chunks = chunkSections(sections, tokenizer, 48, 4);
  assert.ok(chunks.length > 3);
  for (const chunk of chunks) {
    assert.ok(tokenizer(documentInput(chunk)).input_ids.data.length <= 48);
    assert.equal(chunk.sourceUrl, sections[0].sourceUrl);
    assert.equal(chunk.title, sections[0].title);
  }
  assert.match(chunks.at(-1).text, /last words$/);
  const overlapping = chunkSections([{
    ...sections[0], text: 'one two three four five six seven eight nine ten eleven twelve'
  }], tokenizer, 48, 8);
  assert.equal(overlapping[0].text.split(/\s+/).at(-1), overlapping[1].text.split(/\s+/)[0]);
  assert.throws(() => chunkSections(sections, tokenizer, 48, 24), /overlapTokens/);
});

test('offline keyword retrieval ranks egress documentation above unrelated Job and Redis chunks', () => {
  const chunks = [
    { title: 'Job > SuccessPolicyRule', text: 'Create a job using indexes.', sourceUrl: 'https://example.test/job#success' },
    { title: 'Network Policies > Egress', text: 'Create a NetworkPolicy to allow outgoing traffic.', sourceUrl: 'https://example.test/network#egress' },
    { title: 'Work Queue > Redis', text: 'Create a Redis list.', sourceUrl: 'https://example.test/redis#list' },
    { title: 'Network Policies > Egress', text: 'Configure outbound traffic.', sourceUrl: 'https://example.test/network#egress' }
  ];
  const index = buildKeywordIndex(chunks);
  const keyword = searchKeywords(index, 'How do I create a egress?');
  assert.equal(keyword[0].row, 1);
  assert.ok(keyword.slice(0, 2).every(match => [1, 3].includes(match.row)));
  const data = { chunks, vectors: new Float32Array([1, 0, 0.7, 0.7, 1, 0, 0.7, 0.7]), manifest: { dimensions: 2 } };
  const ranked = rankHybridResults([
    { row: 0, score: 0.99 }, { row: 1, score: 0.8 }, { row: 2, score: 0.75 }, { row: 3, score: 0.7 }
  ], keyword, data, [1, 0], 0.25, 3);
  assert.equal(ranked[0].row, 1);
  assert.equal(ranked.filter(match => [1, 3].includes(match.row)).length, 1);
});

test('hybrid fusion combines different chunks of one heading without duplicate rank penalties', () => {
  const chunks = [
    { sourceUrl: 'https://example.test/deployment#creating' },
    { sourceUrl: 'https://example.test/deployment#creating' },
    { sourceUrl: 'https://example.test/other#section' }
  ];
  const data = { chunks, vectors: new Float32Array([1, 0, 1, 0, 1, 0]), manifest: { dimensions: 2 } };
  const semantic = [{ row: 0, score: 0.9 }, { row: 1, score: 0.85 }, { row: 2, score: 0.8 }];
  const keyword = [{ row: 1, score: 10 }, { row: 2, score: 8 }, { row: 0, score: 7 }];
  const ranked = rankHybridResults(semantic, keyword, data, [1, 0], 0.25, 3);
  assert.equal(ranked.length, 2);
  assert.equal(ranked[0].row, 0);
  assert.equal(ranked[0].score, 0.9);
  assert.equal(ranked[0].keywordScore, 10);
  assert.ok(Math.abs(ranked[0].hybridScore - 2 / 61) < 1e-12);
  assert.ok(Math.abs(ranked[1].hybridScore - 2 / 62) < 1e-12);
});

test('multiple headings on one page cannot crowd other relevant pages out of hybrid results', () => {
  const chunks = [
    { sourceUrl: 'https://example.test/kubectl#synopsis' },
    { sourceUrl: 'https://example.test/kubectl#examples' },
    { sourceUrl: 'https://example.test/kubectl' },
    { sourceUrl: 'https://example.test/deployment#creating' },
    { sourceUrl: 'https://example.test/tutorial#deployment' }
  ];
  const data = { chunks, vectors: new Float32Array([1, 1, 1, 1, 1]), manifest: { dimensions: 1 } };
  const semantic = chunks.map((_, row) => ({ row, score: 0.9 - row * 0.01 }));
  const keyword = [1, 2, 0, 3, 4].map((row, rank) => ({ row, score: 10 - rank }));
  const ranked = rankHybridResults(semantic, keyword, data, [1], 0.25, 3);
  assert.equal(ranked.length, 3);
  assert.ok(ranked.some(match => match.row === 3));
  assert.equal(ranked.filter(match => match.row < 3).length, 1);
  assert.equal(rankHybridResults(semantic, keyword, data, [1], 0.25, 5).length, 5);
});
