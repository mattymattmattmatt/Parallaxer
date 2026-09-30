// Parallel depth inference: a small pool of background workers, each with its own ONNX Runtime session.
//
// Safety rules (the pool must never be the reason an export fails):
//  - Workers start one at a time; work begins as soon as the first is ready.
//  - A worker that can't use the GPU when the page can is rejected (it would only slow things down).
//  - A failed or crashed worker is removed and its frames are retried on another worker.
//  - When no worker is left, callers get PoolExhausted and fall back to in-page inference.

const INIT_TIMEOUT_MS = 180_000;
const MAX_ATTEMPTS = 3;
export const MAX_WORKERS = 4;

export class PoolExhausted extends Error {
  constructor(reason) {
    super(reason || 'No depth workers are available.');
    this.name = 'PoolExhausted';
  }
}

export function poolSupported() {
  return typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined';
}

/**
 * Upper bound for parallel workers on this machine, with the reason that limited it. Leaves CPU cores for the
 * page and the OS, respects low-memory devices and heavy models. Auto mode only ramps up to this while each
 * extra worker actually speeds things up.
 */
export function workerCeiling({ model, gpu }) {
  const cores = navigator.hardwareConcurrency || 4;
  const memGB = navigator.deviceMemory || 8; // browsers report at most 8
  let cap = MAX_WORKERS;
  let why = 'the pool maximum';
  const byCores = Math.max(1, Math.floor((cores - 2) / (gpu ? 2 : 1)));
  if (byCores < cap) {
    cap = byCores;
    why = `${cores} CPU threads`;
  }
  const byMem = memGB <= 4 ? 1 : memGB < 8 ? 2 : MAX_WORKERS;
  if (byMem < cap) {
    cap = byMem;
    why = `${memGB} GB of memory`;
  }
  const heavy = model?.id === 'da2-base' || (model?.bytes && model.bytes.byteLength > 150e6);
  if (heavy && cap > 2) {
    cap = 2;
    why = 'the large model';
  }
  return { cap: Math.max(1, cap), why };
}

/** Rough per-worker memory footprint in MB, for the UI. */
export function workerFootprintMB(model, detail) {
  const base = model?.id === 'midas-small' ? 250 : model?.id === 'da2-base' ? 1400 : model?.id === 'custom' ? 600 : 450;
  const scale = model?.input?.fixed ? 1 : { fast: 0.6, balanced: 0.8, high: 1, ultra: 1.5 }[detail] ?? 1;
  return Math.round((base * scale) / 50) * 50;
}

export class DepthWorkerPool {
  /**
   * @param {object} o
   *   model          registry id, or a model object (custom models carry `bytes`)
   *   backend        'auto' | 'webgpu' | 'wasm' (same preference as the page engine)
   *   expectBackend  backend the page engine actually got; workers must match it
   *   threads        WebAssembly threads per worker (only matters when cross-origin isolated)
   *   onChange       called whenever worker state or stats change
   */
  constructor(o) {
    this.o = o;
    this.workers = [];
    this.jobs = new Map();
    this.waiting = [];
    this.seq = 0;
    this.count = 0;
    this.disposed = false;
  }

  get ready() {
    return this.workers.filter((w) => w.state === 'ready');
  }

  get alive() {
    return this.workers.filter((w) => w.state === 'ready' || w.state === 'starting');
  }

  snapshot() {
    return this.workers.map((w) => ({
      label: `W${w.idx}`,
      state: w.state,
      backend: w.backend,
      frames: w.frames,
      ms: w.ms,
      inflight: w.inflight,
      error: w.error
    }));
  }

  #changed() {
    this.o.onChange?.(this.snapshot());
  }

  /** Start one more worker. Resolves with the worker once it is ready, or null if it could not start. */
  addWorker() {
    if (this.disposed) return Promise.resolve(null);
    const idx = ++this.count;
    const worker = new Worker(new URL('./depth-worker.js', import.meta.url), { type: 'module', name: `parallaxer-depth-${idx}` });
    const w = { idx, worker, state: 'starting', backend: null, inflight: 0, frames: 0, ms: 0, error: null };
    this.workers.push(w);
    this.#changed();
    return new Promise((resolve) => {
      w.onInit = resolve;
      w.timer = setTimeout(() => this.#fail(w, 'took too long to load the model'), INIT_TIMEOUT_MS);
      worker.onmessage = (e) => this.#message(w, e.data);
      worker.onerror = (ev) => {
        ev.preventDefault?.();
        this.#fail(w, (ev.message || '').replace(/^Uncaught (\w*Error: )?/, '') || 'the worker crashed');
      };
      worker.onmessageerror = () => this.#fail(w, 'could not exchange data with the worker');
      const bytes = typeof this.o.model === 'object' ? this.o.model.bytes?.slice() : null;
      const model = typeof this.o.model === 'object' ? { ...this.o.model, bytes } : this.o.model;
      worker.postMessage({ type: 'init', model, backend: this.o.backend, base: document.baseURI, threads: this.o.threads }, bytes ? [bytes.buffer] : []);
    });
  }

  #message(w, m) {
    if (m.type === 'ready') {
      clearTimeout(w.timer);
      if (this.o.expectBackend === 'webgpu' && m.backend !== 'webgpu') {
        this.#fail(w, "this browser can't use the GPU from background workers");
        return;
      }
      w.state = 'ready';
      w.backend = m.backend;
      w.onInit?.(w);
      w.onInit = null;
      this.#flushWaiting();
      this.#changed();
    } else if (m.type === 'init-error') {
      this.#fail(w, m.message);
    } else if (m.type === 'result') {
      const job = this.jobs.get(m.id);
      w.inflight = Math.max(0, w.inflight - 1);
      w.frames++;
      w.ms = w.ms ? w.ms * 0.8 + m.ms * 0.2 : m.ms;
      if (job) {
        this.jobs.delete(m.id);
        job.resolve({ data: m.data, w: m.w, h: m.h, luma: m.luma, pLo: m.pLo, pHi: m.pHi, ms: m.ms, worker: w.idx });
      }
      if (w.state === 'retiring' && !w.inflight) this.#close(w, 'retired');
      this.#changed();
    } else if (m.type === 'error') {
      // A runtime error means this session is unhealthy (e.g. GPU device lost): drop the worker, retry elsewhere.
      this.#fail(w, m.message);
    }
  }

  #fail(w, reason) {
    if (w.state === 'failed' || w.state === 'retired') return;
    clearTimeout(w.timer);
    w.state = 'failed';
    w.error = reason;
    this.lastError = reason;
    try {
      w.worker.terminate();
    } catch {
      /* already gone */
    }
    w.onInit?.(null);
    w.onInit = null;
    // Re-dispatch this worker's frames.
    for (const job of [...this.jobs.values()]) {
      if (job.worker === w) {
        job.worker = null;
        this.#dispatch(job);
      }
    }
    if (!this.alive.length) this.#rejectWaiting(new PoolExhausted(`The background workers stopped (${reason}).`));
    this.#changed();
  }

  #close(w, state) {
    clearTimeout(w.timer);
    w.state = state;
    try {
      w.worker.postMessage({ type: 'dispose' });
    } catch {
      /* ignore */
    }
    setTimeout(() => {
      try {
        w.worker.terminate();
      } catch {
        /* ignore */
      }
    }, 3000);
    this.#changed();
  }

  /** Stop sending work to the most recently added ready worker and release it once idle. */
  retireNewest() {
    const w = [...this.ready].sort((a, b) => b.idx - a.idx)[0];
    if (!w || this.ready.length <= 1) return false;
    w.state = 'retiring';
    if (!w.inflight) this.#close(w, 'retired');
    this.#changed();
    return true;
  }

  /** Infer depth for model-input pixels. Resolves with raw depth plus the prepared stabiliser inputs. */
  infer(rgba, w, h) {
    return new Promise((resolve, reject) => {
      const job = { id: ++this.seq, rgba, w, h, attempts: 0, worker: null, resolve, reject };
      this.#dispatch(job);
    });
  }

  #dispatch(job) {
    if (this.disposed) {
      job.reject(new PoolExhausted('The pool was shut down.'));
      return;
    }
    if (++job.attempts > MAX_ATTEMPTS) {
      this.jobs.delete(job.id);
      job.reject(new PoolExhausted('A frame failed on several workers.'));
      return;
    }
    const ready = this.ready;
    if (!ready.length) {
      if (this.alive.length) {
        job.attempts--;
        this.waiting.push(job);
      } else {
        this.jobs.delete(job.id);
        job.reject(new PoolExhausted(this.lastError ? `The background workers stopped (${this.lastError}).` : undefined));
      }
      return;
    }
    const target = ready.reduce((a, b) => (b.inflight < a.inflight ? b : a));
    target.inflight++;
    job.worker = target;
    this.jobs.set(job.id, job);
    const copy = job.rgba.slice(); // keep the original for retries
    target.worker.postMessage({ type: 'infer', id: job.id, rgba: copy, w: job.w, h: job.h }, [copy.buffer]);
  }

  #flushWaiting() {
    const q = this.waiting;
    this.waiting = [];
    for (const job of q) this.#dispatch(job);
  }

  #rejectWaiting(err) {
    const q = this.waiting;
    this.waiting = [];
    for (const job of q) job.reject(err);
  }

  dispose() {
    this.disposed = true;
    for (const w of this.workers) {
      if (w.state === 'ready' || w.state === 'starting' || w.state === 'retiring') this.#close(w, 'retired');
      w.onInit?.(null);
      w.onInit = null;
    }
    for (const job of this.jobs.values()) job.reject(new PoolExhausted('The pool was shut down.'));
    this.jobs.clear();
    this.#rejectWaiting(new PoolExhausted('The pool was shut down.'));
  }
}

/**
 * Auto mode: measure throughput, add a worker, keep it only if frames/s improved meaningfully. Stops at the
 * machine ceiling or as soon as the GPU is the bottleneck, so it never uses more resources than help.
 */
export class AutoTuner {
  constructor(pool, { cap, onMessage }) {
    this.pool = pool;
    this.cap = cap;
    this.onMessage = onMessage;
    this.state = 'warmup';
    this.best = null;
    this.stamp = [];
    this.sinceChange = 0;
    this.settled = false;
  }

  /** The export paused or moved to the next file: throw away the current measurement, it would read low. */
  interrupted() {
    if (this.settled || this.state === 'adding') return;
    this.state = 'warmup';
    this.sinceChange = 0;
    this.stamp = [];
  }

  /** Call once per emitted frame. */
  frame() {
    if (this.settled || this.state === 'adding') return;
    const now = performance.now();
    this.sinceChange++;
    if (this.state === 'warmup') {
      if (this.sinceChange >= 10) {
        this.state = 'measuring';
        this.stamp = [now];
      }
      return;
    }
    this.stamp.push(now);
    const span = now - this.stamp[0];
    if (span < 3000 || this.stamp.length < 24) return;
    const fps = ((this.stamp.length - 1) * 1000) / span;
    const n = this.pool.ready.length;
    const workers = (k) => `${k} worker${k > 1 ? 's' : ''}`;
    if (this.best === null || fps > this.best.fps * 1.1) {
      const gain = this.best ? ` (+${Math.round((fps / this.best.fps - 1) * 100)}%)` : '';
      this.best = { fps, workers: n };
      if (n >= this.cap) {
        this.#settle(`Auto: ${workers(n)} → ${fps.toFixed(1)} fps${gain}. That's the safe limit for this machine.`, 'ok');
        return;
      }
      this.onMessage?.(`Auto: ${workers(n)} → ${fps.toFixed(1)} fps${gain}. Trying ${n + 1} to see if that's faster…`, 'info');
      this.state = 'adding';
      this.pool.addWorker().then((w) => {
        if (this.pool.disposed) return;
        if (!w) {
          this.#settle(`Auto: staying at ${workers(n)} (${fps.toFixed(1)} fps) — another one couldn't start.`, 'warn');
          return;
        }
        this.state = 'warmup';
        this.sinceChange = 0;
      });
    } else {
      // No meaningful gain: the GPU (or memory bandwidth) is the limit. Give the extra worker back.
      this.pool.retireNewest();
      const k = this.best.workers;
      this.#settle(`Auto: settled on ${workers(k)} (${this.best.fps.toFixed(1)} fps). Worker ${k + 1} didn't make it faster, so it was released.`, 'ok');
    }
  }

  #settle(msg, tone) {
    this.settled = true;
    this.state = 'settled';
    this.onMessage?.(msg, tone);
  }
}
