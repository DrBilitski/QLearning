import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteSingleFile } from 'vite-plugin-singlefile';

// The build is a single self-contained dist/index.html (JS and CSS inlined),
// so it runs from any static host or by simply opening the file in a browser.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  base: './',
});
