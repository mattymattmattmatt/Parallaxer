import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import { getModel, DETAIL_LEVELS } from './models.js';

const CACHE_NAME = 'parallaxer-models-v1';
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let ortPromise = null;

async function loadOrt() {
  ortPromise ??= import('onnxruntime-web/webgpu').then((ort) => {
    ort.env.wasm.wasmPaths = { wasm: new URL(ortWasmUrl, import.meta.url).href };
    ort.env.logLevel = 'error';
    if (self.crossOriginIsolated) {
      ort.env.wasm.numThreads = Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 4) - 1));
    } else {
      ort.env.wasm.numThreads = 1;
    }
    return ort;
  });
  return ortPromise;
}

export async function detectWebGPU() {
  if (!navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    const info = adapter.info ?? {};
    return {
      vendor: info.vendor || 'GPU',
      architecture: info.architecture || '',
      description: info.description || '',
      f16: adapter.features.has('shader-f16')
    };
  } catch {
    return null;
  }
}

function resolveUrl(url) {
  return new URL(url, document.baseURI).href;
}

/** Fetch with streaming progress, backed by the Cache Storage API so large models download once. */
export async function fetchCached(url, { onProgress, signal, expectedSize = 0 } = {}) {
  const href = resolveUrl(url);
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
  } catch {
    cache = null;
  }
  const hit = cache ? await cache.match(href).catch(() => null) : null;
  if (hit) {
    const buf = await hit.arrayBuffer();
    onProgress?.({ loaded: buf.byteLength, total: buf.byteLength, cached: true });
    return new Uint8Array(buf);
  }

  const res = await fetch(href, { signal, mode: 'cors' });
  if (!res.ok) throw new Error(`Download failed (HTTP ${res.status}) for ${href}`);
  const total = Number(res.headers.get('content-length')) || expectedSize;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  let last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    const now = performance.now();
    if (now - last > 80) {
      onProgress?.({ loaded, total, cached: false });
      last = now;
    }
  }
  onProgress?.({ loaded, total: loaded, cached: false });
  const blob = new Blob(chunks, { type: 'application/octet-stream' });
  if (cache) {
    try {
      await navigator.storage?.persist?.();
    } catch {
      /* best effort */
    }
    cache.put(href, new Response(blob, { headers: { 'content-type': 'application/octet-stream' } })).catch(() => {});
  }
  return new Uint8Array(await blob.arrayBuffer());
}

export async function isModelCached(model) {
  try {
    const cache = await caches.open(CACHE_NAME);
    const urls = [model.url, model.urlFp16].filter(Boolean).map(resolveUrl);
    for (const u of urls) if (await cache.match(u)) return true;
  } catch {
    /* no cache storage */
  }
  return false;
}

export async function clearModelCache() {
  try {
    await caches.delete(CACHE_NAME);
  } catch {
    /* ignore */
  }
}

// ---------- float16 helpers ----------

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

function toHalf(v) {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x800000) >> (1 - exp);
    return sign | ((mant + 0x1000) >> 13);
  }
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | ((mant + 0x1000) >> 13);
}

function fromHalf(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

function toFloat32(data) {
  if (data instanceof Float32Array) return data;
  if (data instanceof Uint16Array) {
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) out[i] = fromHalf(data[i]);
    return out;
  }
  return Float32Array.from(data);
}

// ---------- Engine ----------

export class DepthEngine {
  constructor() {
    this.session = null;
    this.model = null;
    this.backend = null;
    this.precision = 'fp32';
    this.inputType = 'float32';
    this.fixedSize = null;
    this.queue = Promise.resolve();
    this.lastMs = 0;
    this.loading = null;
  }

  get ready() {
    return !!this.session;
  }

  /**
   * Load (or switch to) a model.
   * @param {string} id
   * @param {{ preferBackend?: 'auto'|'webgpu'|'wasm', gpu?: object|null, onProgress?: Function, onStatus?: Function, signal?: AbortSignal }} opts
   */
  async load(id, opts = {}) {
    const model = typeof id === 'object' ? id : getModel(id);
    const task = this.queue.then(() => this.#load(model, opts));
    this.queue = task.catch(() => {});
    return task;
  }

  async #load(model, { preferBackend = 'auto', gpu = null, onProgress, onStatus, signal } = {}) {
    onStatus?.('Starting ONNX Runtime…');
    const ort = await loadOrt();
    if (this.session) {
      try {
        await this.session.release();
      } catch {
        /* ignore */
      }
      this.session = null;
    }

    const wantGpu = preferBackend !== 'wasm' && !!gpu;
    const useFp16 = wantGpu && gpu.f16 && !!model.urlFp16 && !model.bytes;
    const attempts = [];
    if (wantGpu) attempts.push({ backend: 'webgpu', fp16: useFp16 });
    if (wantGpu && useFp16) attempts.push({ backend: 'webgpu', fp16: false });
    if (preferBackend !== 'webgpu' || !wantGpu) attempts.push({ backend: 'wasm', fp16: false });

    let lastErr = null;
    let bytes = null;
    let bytesFp16 = null;
    for (const a of attempts) {
      try {
        const url = a.fp16 ? model.urlFp16 : model.url;
        const size = a.fp16 ? model.sizeFp16 : model.size;
        let data = model.bytes ?? (a.fp16 ? bytesFp16 : bytes);
        if (!data) {
          onStatus?.(`Fetching ${model.name}${a.fp16 ? ' (fp16)' : ''}…`);
          data = await fetchCached(url, { onProgress, signal, expectedSize: size });
          if (a.fp16) bytesFp16 = data;
          else bytes = data;
        }
        onStatus?.(`Compiling for ${a.backend === 'webgpu' ? 'WebGPU' : 'WebAssembly'}…`);
        const session = await ort.InferenceSession.create(data, {
          executionProviders: [a.backend],
          graphOptimizationLevel: 'all',
          logSeverityLevel: 3
        });
        this.session = session;
        this.model = model;
        this.backend = a.backend;
        this.precision = a.fp16 ? 'fp16' : 'fp32';
        const meta = session.inputMetadata?.[0];
        this.inputType = meta?.type === 'float16' ? 'float16' : 'float32';
        const shape = meta?.shape ?? [];
        const h = shape[2];
        const w = shape[3];
        this.fixedSize = typeof w === 'number' && typeof h === 'number' && w > 0 && h > 0 ? [w, h] : model.input.fixed ?? null;
        this.ort = ort;
        return { backend: this.backend, precision: this.precision, model };
      } catch (err) {
        if (signal?.aborted) throw err;
        lastErr = err;
        console.warn(`[depth] ${model.id} on ${a.backend}${a.fp16 ? '/fp16' : ''} failed`, err);
      }
    }
    throw lastErr ?? new Error('Unable to initialise the depth model.');
  }

  /** Network input size for a source of the given aspect ratio. */
  inputSize(srcW, srcH, detailId = 'high') {
    if (this.fixedSize) return { w: this.fixedSize[0], h: this.fixedSize[1] };
    const mult = this.model?.input?.multiple ?? 14;
    const short = (DETAIL_LEVELS.find((d) => d.id === detailId) ?? DETAIL_LEVELS[2]).short;
    const aspect = srcW / Math.max(1, srcH);
    let w;
    let h;
    if (aspect >= 1) {
      h = short;
      w = Math.min(short * 2.5, short * aspect);
    } else {
      w = short;
      h = Math.min(short * 2.5, short / aspect);
    }
    const snap = (v) => Math.max(mult, Math.round(v / mult) * mult);
    return { w: snap(w), h: snap(h) };
  }

  /**
   * Run the network on an RGBA8 buffer (image row order).
   * Returns { data: Float32Array (disparity-like), w, h, ms }.
   */
  infer(rgba, w, h) {
    const task = this.queue.then(() => this.#infer(rgba, w, h));
    this.queue = task.catch(() => {});
    return task;
  }

  async #infer(rgba, w, h) {
    if (!this.session) throw new Error('Depth model not loaded.');
    const ort = this.ort;
    const t0 = performance.now();
    const n = w * h;
    const planar = new Float32Array(3 * n);
    const imagenet = this.model.norm === 'imagenet';
    const s0 = imagenet ? 1 / (255 * STD[0]) : 1 / 255;
    const s1 = imagenet ? 1 / (255 * STD[1]) : 1 / 255;
    const s2 = imagenet ? 1 / (255 * STD[2]) : 1 / 255;
    const b0 = imagenet ? -MEAN[0] / STD[0] : 0;
    const b1 = imagenet ? -MEAN[1] / STD[1] : 0;
    const b2 = imagenet ? -MEAN[2] / STD[2] : 0;
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      planar[i] = rgba[j] * s0 + b0;
      planar[n + i] = rgba[j + 1] * s1 + b1;
      planar[2 * n + i] = rgba[j + 2] * s2 + b2;
    }
    let tensor;
    if (this.inputType === 'float16') {
      const half = new Uint16Array(planar.length);
      for (let i = 0; i < planar.length; i++) half[i] = toHalf(planar[i]);
      tensor = new ort.Tensor('float16', half, [1, 3, h, w]);
    } else {
      tensor = new ort.Tensor('float32', planar, [1, 3, h, w]);
    }
    const feeds = { [this.session.inputNames[0]]: tensor };
    const results = await this.session.run(feeds);
    const out = results[this.session.outputNames[0]];
    const dims = out.dims;
    const oh = dims[dims.length - 2];
    const ow = dims[dims.length - 1];
    const data = toFloat32(out.data);
    tensor.dispose?.();
    out.dispose?.();
    this.lastMs = performance.now() - t0;
    return { data, w: ow, h: oh, ms: this.lastMs };
  }
}
