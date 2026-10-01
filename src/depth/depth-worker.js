// Background depth worker: owns one ONNX Runtime session and turns model-input pixels into raw depth plus the
// frame-independent half of the stabiliser (luma + robust range), so the page only does the sequential part.
import { DepthEngine, detectWebGPU, setAssetBase, setThreadBudget } from './engine.js';
import { prepareFrame } from './stabilizer.js';

const engine = new DepthEngine();

/** A typed array that owns its whole buffer (safe to transfer). */
const owned = (a) => (a.byteOffset === 0 && a.byteLength === a.buffer.byteLength ? a : a.slice());

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'init') {
    try {
      setAssetBase(m.base);
      if (m.threads) setThreadBudget(m.threads);
      const gpu = m.backend === 'wasm' ? null : await detectWebGPU();
      const info = await engine.load(m.model, { preferBackend: m.backend, gpu, fixed: m.fixed });
      self.postMessage({ type: 'ready', backend: info.backend, precision: info.precision, mode: info.mode });
    } catch (err) {
      self.postMessage({ type: 'init-error', message: String(err?.message ?? err) });
    }
  } else if (m.type === 'infer') {
    try {
      const res = await engine.infer(m.rgba, m.w, m.h);
      const data = owned(res.data);
      const pre = prepareFrame(data, res.w, res.h, m.rgba, m.w, m.h);
      self.postMessage(
        { type: 'result', id: m.id, data, w: res.w, h: res.h, luma: pre.luma, pLo: pre.pLo, pHi: pre.pHi, ms: res.ms, mode: engine.mode },
        [data.buffer, pre.luma.buffer]
      );
    } catch (err) {
      self.postMessage({ type: 'error', id: m.id, message: String(err?.message ?? err) });
    }
  } else if (m.type === 'dispose') {
    try {
      await engine.session?.release();
    } catch {
      /* ignore */
    }
    self.postMessage({ type: 'disposed' });
    self.close();
  }
};
