export function renderAnswer(container, text, sources) {
  const nodes = [];
  const cited = new Set();
  let offset = 0;
  let inCode = false;
  for (const match of text.matchAll(/```|\[(\d+)\]/g)) {
    nodes.push(document.createTextNode(text.slice(offset, match.index)));
    const source = sources[Number(match[1]) - 1];
    if (match[0] === '```') {
      inCode = !inCode;
      nodes.push(document.createTextNode(match[0]));
    } else if (!inCode && source) {
      const link = document.createElement('a');
      link.href = source.sourceUrl;
      link.textContent = match[0];
      link.title = source.title || source.sourceUrl;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      nodes.push(link);
      cited.add(Number(match[1]));
    } else {
      nodes.push(document.createTextNode(match[0]));
    }
    offset = match.index + match[0].length;
  }
  nodes.push(document.createTextNode(text.slice(offset)));
  container.replaceChildren(...nodes);
  return cited;
}
