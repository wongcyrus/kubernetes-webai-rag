import MarkdownIt from 'markdown-it';

const markdown = new MarkdownIt({ html: false, breaks: true }).disable(['link', 'image', 'autolink']);

export function renderAnswer(container, text, sources, onCitation) {
  container.innerHTML = markdown.render(text);
  const cited = new Set();
  const walker = document.createTreeWalker(container, 4);
  const textNodes = [];
  while (walker.nextNode()) {
    const protectedContainer = walker.currentNode.parentElement.closest('pre, code, a');
    if (!protectedContainer || protectedContainer === container) textNodes.push(walker.currentNode);
  }
  for (const node of textNodes) {
    const nodes = [];
    let offset = 0;
    for (const match of node.textContent.matchAll(/\[(\d+)\]/g)) {
      nodes.push(document.createTextNode(node.textContent.slice(offset, match.index)));
      const source = sources[Number(match[1]) - 1];
      if (source) {
        const link = document.createElement('a');
        link.href = source.sourceUrl;
        link.textContent = match[0];
        link.title = source.title || source.sourceUrl;
        if (onCitation) {
          link.addEventListener('click', event => {
            event.preventDefault();
            onCitation(source, Number(match[1]));
          });
        } else {
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
        }
        nodes.push(link);
        cited.add(Number(match[1]));
      } else {
        nodes.push(document.createTextNode(match[0]));
      }
      offset = match.index + match[0].length;
    }
    nodes.push(document.createTextNode(node.textContent.slice(offset)));
    node.replaceWith(...nodes);
  }

  for (const pre of container.querySelectorAll('pre')) {
    const code = pre.querySelector('code');
    const block = document.createElement('div');
    block.className = 'answer-code';
    const toolbar = document.createElement('div');
    toolbar.className = 'code-toolbar';
    const language = document.createElement('span');
    language.textContent = code.className.replace(/^language-/, '') || 'Code';
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'secondary copy-code';
    button.textContent = 'Copy code';
    button.setAttribute('aria-live', 'polite');
    pre.classList.add('code-block');
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = 'Copied';
      } catch {
        const selection = document.getSelection?.();
        if (selection) {
          const range = document.createRange();
          range.selectNodeContents(code);
          selection.removeAllRanges();
          selection.addRange(range);
          button.textContent = 'Selected; press Ctrl+C';
        } else {
          button.textContent = 'Copy unavailable; select manually';
        }
      }
    });
    toolbar.append(language, button);
    pre.replaceWith(block);
    block.append(toolbar, pre);
  }
  return cited;
}
