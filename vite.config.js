import { defineConfig } from 'vite';

// Cross-origin isolation lets ONNX Runtime use multi-threaded WebAssembly when served locally.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp'
};

export default defineConfig({
  // Relative base so the same build works at a domain root or under a GitHub Pages sub-path.
  base: './',
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    target: 'es2022',
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 2048
  }
});
