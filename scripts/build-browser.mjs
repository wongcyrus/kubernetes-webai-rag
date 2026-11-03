import { mkdir, copyFile, rename } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

export async function buildBrowser(directory = 'offline') {
  const root = resolve(directory);
  await Promise.all(['js', 'css', 'licenses'].map(path => mkdir(join(root, path), { recursive: true })));
  await build({
    entryPoints: ['src/OfflineVectorSearch.js'],
    outfile: join(root, 'OfflineVectorSearch-min.js.download'),
    bundle: true, minify: true, format: 'esm', platform: 'browser',
    plugins: [{
      name: 'local-only-imports',
      setup(api) {
        api.onResolve({ filter: /^https?:\/\// }, args => ({
          errors: [{ text: `External module import is forbidden in the offline app: ${args.path}` }]
        }));
      }
    }]
  });
  await rename(join(root, 'OfflineVectorSearch-min.js.download'), join(root, 'OfflineVectorSearch-min.js'));
  await Promise.all([
    copyFile('web/kubernetes.html', join(root, 'kubernetes.html')),
    copyFile('web/kubernetes.html', join(root, 'index.html')),
    copyFile('web/js/kubernetes.js', join(root, 'js/kubernetes.js')),
    copyFile('web/js/answer-citations.js', join(root, 'js/answer-citations.js')),
    copyFile('web/css/style.css', join(root, 'css/style.css')),
    copyFile('LICENSE', join(root, 'licenses/project.txt')),
    copyFile('NOTICE', join(root, 'NOTICE'))
  ]);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await buildBrowser();
}
