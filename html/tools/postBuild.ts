import type { Plugin } from 'vite';
import type { OutputAsset, OutputChunk } from 'rollup';

const warnNotInlined = (filename: string) => console.warn(`WARNING: asset not inlined: ${filename}`);

export function vitePostBuildFix(): Plugin {
  return {
    name: 'vite:postBuildFix',
    enforce: 'post',
    generateBundle: (_, bundle) => {
      // Check if bundle has index.html
      if (!bundle['index.html']) {
        return;
      }

      // Check if the source of index.html is a string and available
      if ('source' in bundle['index.html'] && typeof bundle['index.html'].source === 'string') {
        const indexHtml: OutputAsset = bundle['index.html'] as OutputAsset;

        if (typeof indexHtml.source !== 'string') {
          return;
        }

        // Now replace the placeholder with the correct quotes
        const source = indexHtml.source.replace(/"\/\/ JS_TEMPLATE"|'\/\/ JS_TEMPLATE'/g, '`// JS_TEMPLATE`');

        // Log it to console that we replaced the incorrect quotes
        console.log('Post Build Fix: Replaced incorrect quotes in index.html');

        indexHtml.source = source;
      }
    }
  };
}
