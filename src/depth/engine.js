import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import { getModel, DETAIL_LEVELS } from './models.js';

const CACHE_NAME = 'parallaxer-models-v1';
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let ortPromise = null;
let threadBudget = null;
let assetBase = null;

/** Cap WebAssembly threads for this context (parallel workers share the CPU). Call before the first load. */
export function setThreadBudget(n) {
  threadBudget = Math.max(1, Math.floor(n));
}

/** Base URL for relative model paths when there is no document (e.g. inside a worker). */
export function setAssetBase(base) {
  assetBase = base;
}

async function loadOrt() {
  ortPromise ??= import('onnxruntime-web/webgpu').then((ort) => {
    ort.env.wasm.wasmPaths = { wasm: new URL(ortWasmUrl, import.meta.url).href };
    ort.env.logLevel = 'error';
    if (self.crossOriginIsolated) {
      const auto = Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 4) - 1));
      ort.env.wasm.numThreads = threadBudget ? Math.min(auto, threadBudget) : auto;
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
  return new URL(url, assetBase ?? globalThis.document?.baseURI ?? self.location.href).href;
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
  // Add (not OR) the rounded mantissa: rounding up from 0x3ff carries into the exponent (1.9999 → 2, not 1).
  return sign | Math.min(0x7c00, (exp << 10) + ((mant + 0x1000) >> 13));
}

function fromHalf(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// Native Float16Array (Chrome 135+, Firefox 129+, Safari 18.2+) converts whole arrays far faster than JS bit-twiddling.
const HAS_F16 = typeof Float16Array === 'function';

/** Float32 values → IEEE half bits. */
function toHalfArray(src) {
  if (HAS_F16) return new Uint16Array(new Float16Array(src).buffer);
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = toHalf(src[i]);
  return out;
}

function toFloat32(data) {
  if (data instanceof Float32Array) return data;
  if (data instanceof Uint16Array) {
    if (HAS_F16) return new Float32Array(new Float16Array(data.buffer, data.byteOffset, data.length));
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) out[i] = fromHalf(data[i]);
    return out;
  }
  return Float32Array.from(data);
}

// ---------- Fixed-size sessions (export workers) ----------

/**
 * freeDimensionOverrides for an NCHW image input whose batch / height / width are symbolic, or null when a
 * symbolic dimension can't be pinned (then the model must run with dynamic shapes). {} = already static.
 */
function freeDims(dims, w, h) {
  if (!Array.isArray(dims) || dims.length !== 4) return null;
  const want = [1, 3, h, w];
  const out = {};
  for (let i = 0; i < 4; i++) {
    const d = dims[i];
    if (typeof d === 'string' && d) {
      if (i === 1) return null;
      if (d in out && out[d] !== want[i]) return null;
      out[d] = want[i];
    } else if (typeof d === 'number' && d > 0 && d !== want[i]) {
      return null;
    }
  }
  return out;
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
    // 'dynamic' (any input size) or 'static' (pinned to one input size, see load({ fixed })).
    this.mode = 'dynamic';
    this.spec = null;
  }

  get ready() {
    return !!this.session;
  }

  /**
   * Load (or switch to) a model.
   * @param {string} id
   * @param {{ preferBackend?: 'auto'|'webgpu'|'wasm', gpu?: object|null, onProgress?: Function, onStatus?: Function, signal?: AbortSignal,
   *           fixed?: { w: number, h: number, dims: Array<string|number> } }} opts
   *   fixed  pin the session to one input size (dims = the model's input shape with symbolic names, from a dynamic
   *          session's inputMetadata). ONNX Runtime can then precompute every shape and fuse more. Inputs of another
   *          size re-pin the session.
   *
   * (WebGPU graph capture was evaluated for this and rejected: in ONNX Runtime 1.30 captured graphs can silently
   * skip replays or return wrong results depending on other sessions' activity.)
   */
  async load(id, opts = {}) {
    const model = typeof id === 'object' ? id : getModel(id);
    const task = this.queue.then(() => this.#load(model, opts));
    this.queue = task.catch(() => {});
    return task;
  }

  async #load(model, { preferBackend = 'auto', gpu = null, onProgress, onStatus, signal, fixed = null } = {}) {
    onStatus?.('Starting ONNX Runtime…');
    const ort = await loadOrt();
    await this.#release();

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
        this.ort = ort;
        const { session, mode } = await this.#createSession(data, a.backend, fixed);
        this.session = session;
        this.mode = mode;
        this.spec = fixed && mode !== 'dynamic' ? { model, fp16: a.fp16, backend: a.backend, dims: fixed.dims, w: fixed.w, h: fixed.h } : null;
        this.model = model;
        this.backend = a.backend;
        this.precision = a.fp16 ? 'fp16' : 'fp32';
        const meta = session.inputMetadata?.[0];
        this.inputType = meta?.type === 'float16' ? 'float16' : 'float32';
        const shape = meta?.shape ?? [];
        const h = shape[2];
        const w = shape[3];
        // Only the model's own fixed input counts here, not a session pinned for an export.
        this.fixedSize = !fixed && typeof w === 'number' && typeof h === 'number' && w > 0 && h > 0 ? [w, h] : model.input.fixed ?? null;
        return { backend: this.backend, precision: this.precision, model, mode: this.mode };
      } catch (err) {
        if (signal?.aborted) throw err;
        lastErr = err;
        console.warn(`[depth] ${model.id} on ${a.backend}${a.fp16 ? '/fp16' : ''} failed`, err);
      }
    }
    throw lastErr ?? new Error('Unable to initialise the depth model.');
  }

  /** Create a pinned (static-shape) session when possible, otherwise a dynamic one. */
  async #createSession(data, backend, fixed) {
    const ort = this.ort;
    const base = { executionProviders: [backend], graphOptimizationLevel: 'all', logSeverityLevel: 3 };
    const overrides = fixed ? freeDims(fixed.dims, fixed.w, fixed.h) : null;
    if (overrides && Object.keys(overrides).length) {
      try {
        return { session: await ort.InferenceSession.create(data, { ...base, freeDimensionOverrides: overrides }), mode: 'static' };
      } catch (err) {
        console.warn('[depth] fixed-size session unavailable, using dynamic shapes', err?.message ?? err);
      }
    }
    // A model without symbolic dimensions ({} overrides) is pinned already.
    return { session: await ort.InferenceSession.create(data, base), mode: overrides && !Object.keys(overrides).length ? 'static' : 'dynamic' };
  }

  async #release() {
    if (this.session) {
      try {
        await this.session.release();
      } catch {
        /* ignore */
      }
      this.session = null;
    }
  }

  /** Rebuild the pinned session for a new input size (e.g. the next video of a batch). */
  async #respecialize(w, h) {
    const sp = this.spec;
    const model = sp.model;
    const data = model.bytes ?? (await fetchCached(sp.fp16 ? model.urlFp16 : model.url, { expectedSize: sp.fp16 ? model.sizeFp16 : model.size }));
    await this.#release();
    const { session, mode } = await this.#createSession(data, sp.backend, { dims: sp.dims, w, h });
    this.session = session;
    this.mode = mode;
    this.spec = mode === 'dynamic' ? null : { ...sp, w, h };
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
    const input = this.inputType === 'float16' ? toHalfArray(planar) : planar;
    if (this.spec && (w !== this.spec.w || h !== this.spec.h)) await this.#respecialize(w, h);
    const tensor = new ort.Tensor(this.inputType, input, [1, 3, h, w]);
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
