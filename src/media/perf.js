// Export profiling for the Advanced stats panel.
//
// The page thread drives the whole export, so its wall time is split into buckets that add up exactly:
// time spent *waiting* on another part of the system (decoder, depth model, encoder) and time spent *working*
// (preparing frames, stabilising depth, issuing GPU rendering). The biggest bucket is the bottleneck.

export const STAGES = [
  { key: 'decode', label: 'Decode', wait: true, desc: 'Waiting for the video decoder' },
  { key: 'prepare', label: 'Prepare', desc: 'Copying each decoded frame to the GPU and starting its readback' },
  { key: 'depth', label: 'Depth', wait: true, desc: 'Waiting for the depth model' },
  { key: 'stabilise', label: 'Stabilise', desc: 'Temporal depth smoothing on the CPU' },
  { key: 'render', label: 'Render', desc: 'Issuing the stereo rendering and handing the frame to the encoder' },
  { key: 'encode', label: 'Encode', wait: true, desc: 'Waiting for the encoder (and the file writer)' },
  { key: 'ui', label: 'Preview & UI', desc: 'Live preview thumbnail and progress updates' },
  { key: 'other', label: 'Other', desc: 'Audio, garbage collection, giving the browser a turn' }
];

const WINDOW_MS = 3000;

class Avg {
  constructor() {
    this.sum = 0;
    this.n = 0;
    this.recent = null;
  }
  add(v) {
    if (!Number.isFinite(v)) return;
    this.sum += v;
    this.n++;
    this.recent = this.recent === null ? v : this.recent * 0.9 + v * 0.1;
  }
  get mean() {
    return this.n ? this.sum / this.n : null;
  }
}

export class ExportProfiler {
  constructor() {
    this.buckets = Object.fromEntries(STAGES.map((s) => [s.key, 0]));
    this.frames = 0;
    this.started = null;
    this.pausedMs = 0;
    this.marks = [];
    this.avg = { depthMs: new Avg(), readbackMs: new Avg(), gpuRenderMs: new Avg(), inflight: new Avg() };
    this.info = {};
    this.waitingFor = null;
    this.stolen = 0;
    this.pageDepth = 0;
  }

  start() {
    this.started = performance.now();
    this.mark();
  }

  add(key, ms) {
    if (key in this.buckets && ms > 0) this.buckets[key] += ms;
  }

  /** Measure an async wait into a bucket. */
  async time(key, promise) {
    const t0 = performance.now();
    this.waitingFor = key;
    this.stolen = 0;
    try {
      return await promise;
    } finally {
      this.add(key, performance.now() - t0 - this.stolen);
      this.waitingFor = null;
      this.stolen = 0;
    }
  }

  /**
   * Depth computed on the page thread itself (Standard mode on the CPU). It blocks the page, so it is time the
   * loop could not spend on anything else: count it as depth, and take it out of whichever wait it interrupted.
   */
  pageDepthBlock(ms) {
    if (!(ms > 0)) return;
    this.buckets.depth += ms;
    this.pageDepth += ms;
    if (this.waitingFor) this.stolen += ms;
  }

  sample(name, v) {
    this.avg[name]?.add(v);
  }

  paused(ms) {
    this.pausedMs += ms;
  }

  frameDone() {
    this.frames++;
    const now = performance.now();
    if (now - (this.marks.at(-1)?.t ?? 0) >= 250) this.mark(now);
  }

  mark(t = performance.now()) {
    this.marks.push({ t, frames: this.frames, paused: this.pausedMs, pageDepth: this.pageDepth, buckets: { ...this.buckets } });
    while (this.marks.length > 2 && t - this.marks[1].t > WINDOW_MS) this.marks.shift();
  }

  /**
   * Per-frame breakdown. `recent` = over the last few seconds (live view); otherwise over the whole export.
   * Returns null until there is something to report.
   */
  snapshot({ recent = true } = {}) {
    if (this.started === null) return null;
    const now = performance.now();
    const base = recent ? this.marks[0] : { t: this.started, frames: 0, paused: 0, pageDepth: 0, buckets: Object.fromEntries(STAGES.map((s) => [s.key, 0])) };
    const frames = this.frames - base.frames;
    const wall = now - base.t - (this.pausedMs - base.paused);
    if (frames < 1 || wall <= 0) return null;
    const ms = {};
    let measured = 0;
    for (const s of STAGES) {
      if (s.key === 'other') continue;
      ms[s.key] = Math.max(0, this.buckets[s.key] - base.buckets[s.key]);
      measured += ms[s.key];
    }
    ms.other = Math.max(0, wall - measured);
    const total = measured + ms.other;
    const stages = STAGES.map((s) => ({ ...s, ms: ms[s.key] / frames, share: ms[s.key] / total }));
    if (this.pageDepth - base.pageDepth > 0) {
      // Depth runs on the page thread: this row is mostly computing, not waiting.
      Object.assign(stages.find((s) => s.key === 'depth'), { wait: false, desc: 'Running the depth model on the page thread (CPU)' });
    }
    const pick = (a) => (recent ? a.recent : a.mean);
    return {
      frames,
      msPerFrame: wall / frames,
      fps: (frames * 1000) / wall,
      stages,
      depthMs: pick(this.avg.depthMs),
      readbackMs: pick(this.avg.readbackMs),
      gpuRenderMs: pick(this.avg.gpuRenderMs),
      inflight: pick(this.avg.inflight),
      info: this.info
    };
  }
}

const pct = (x) => `${Math.round(x * 100)}%`;

/**
 * Plain-language bottleneck and what would help. `snap` from ExportProfiler.snapshot(); the info block carries
 * codec names, hardware support and the processing setup.
 */
export function bottleneck(snap) {
  if (!snap) return null;
  const i = snap.info;
  const top = [...snap.stages].sort((a, b) => b.share - a.share)[0];
  if (top.share < 0.3) {
    return { key: 'balanced', title: 'Balanced', text: 'No single stage dominates — the export is using its pipeline evenly.' };
  }
  const share = pct(top.share);
  switch (top.key) {
    case 'decode':
      return {
        key: 'decode',
        title: 'Video decoding',
        text: `${share} of the time is spent waiting for decoded frames.${i.decoderHw === false ? ` ${i.sourceCodec ?? 'This codec'} is decoded on the CPU here; H.264 or HEVC sources decode on the GPU.` : ''}`
      };
    case 'prepare':
      return { key: 'prepare', title: 'Frame preparation', text: `${share} goes into handing each decoded frame to the GPU. A lower export Resolution makes this cheaper.` };
    case 'depth': {
      if (!top.wait) {
        return {
          key: 'depth',
          title: 'Depth model (on the page thread)',
          text: `${share} goes into running the depth model on the page thread, which blocks everything else meanwhile. Processing speed Auto or 2× moves it to background workers; a lower Detail setting or a smaller model also helps.`
        };
      }
      const more = i.workersMax && (i.workers ?? 0) < i.workersMax ? 'More workers (Processing speed), a ' : 'A ';
      return { key: 'depth', title: 'Depth model', text: `${share} of the time is spent waiting for depth. ${more}lower Detail setting or a smaller model will speed this up.` };
    }
    case 'stabilise':
      return { key: 'stabilise', title: 'Depth smoothing (CPU)', text: `${share} goes into temporal smoothing on the CPU. A lower Detail setting reduces it.` };
    case 'render': {
      const gpu = snap.gpuRenderMs ? ` The GPU itself needs ${snap.gpuRenderMs.toFixed(1)} ms per frame; the rest is the browser passing work to it.` : '';
      return {
        key: 'render',
        title: 'Stereo rendering',
        text: `${share} goes into rendering the two views.${gpu} A lower Resolution or Ray-march quality helps; extra depth workers won't.`
      };
    }
    case 'encode': {
      const codec = i.codecName ?? 'The codec';
      const hw =
        i.encoderHw === false
          ? ` ${codec} is encoded on the CPU on this machine — MP4 with H.264 or HEVC uses the GPU's hardware encoder.`
          : i.encoderHw
            ? ` ${codec} has a hardware encoder here; a lower Quality or Resolution lightens its load.`
            : '';
      const disk = i.streaming ? ' Writing straight to a slow disk can also hold it up.' : '';
      return { key: 'encode', title: 'Video encoder', text: `${share} of the time is spent waiting for the encoder.${hw}${disk}` };
    }
    case 'ui':
      return { key: 'ui', title: 'Preview & UI', text: `${share} goes into the live preview and progress updates. That is unusually high for this step — copy these stats and share them so it can be looked into.` };
    default:
      return { key: 'other', title: 'Browser overhead', text: `${share} goes to browser housekeeping. Keep this tab in the foreground and close other heavy tabs.` };
  }
}

/** Text report for the clipboard. */
export function statsReport(snap, verdict) {
  if (!snap) return '';
  const i = snap.info;
  const f = (v, d = 1) => (v === null || v === undefined ? '—' : v.toFixed(d));
  const lines = [
    `Parallaxer export stats`,
    `Source: ${i.source ?? '?'} (${i.sourceCodec ?? '?'}, decode ${hwWord(i.decoderHw)})`,
    `Output: ${i.output ?? '?'} ${i.codecName ?? ''} (encode ${hwWord(i.encoderHw)})${i.streaming ? ', streamed to disk' : ''}`,
    `Depth: ${i.model ?? '?'} at ${i.modelInput ?? '?'} on ${i.backend ?? '?'} · ${i.processing ?? '?'}`,
    `Throughput: ${f(snap.fps)} fps (${f(snap.msPerFrame)} ms/frame over ${snap.frames} frames)`,
    `Where the time goes (per frame):`,
    ...snap.stages.map((s) => `  ${s.label.padEnd(13)} ${f(s.ms).padStart(7)} ms  ${pct(s.share).padStart(4)}${s.wait ? '  (waiting)' : ''}`),
    `Depth inference ${f(snap.depthMs)} ms/frame per ${i.workers ? 'worker' : 'run'} · readback latency ${f(snap.readbackMs)} ms · GPU render ${f(snap.gpuRenderMs)} ms · frames in flight ${f(snap.inflight)}`,
    verdict ? `Bottleneck: ${verdict.title} — ${verdict.text}` : '',
    `Browser: ${i.browser ?? navigator.userAgent} · ${navigator.hardwareConcurrency ?? '?'} threads${navigator.deviceMemory ? ` · ${navigator.deviceMemory}+ GB` : ''} · isolated ${globalThis.crossOriginIsolated ? 'yes' : 'no'}`
  ];
  return lines.filter(Boolean).join('\n');
}

export function hwWord(v) {
  return v === true ? 'GPU' : v === false ? 'CPU' : 'unknown';
}

/** Short browser name/version for reports. */
export function browserName() {
  const brands = navigator.userAgentData?.brands?.filter((b) => !/not.?a.?brand|chromium/i.test(b.brand));
  if (brands?.length) return `${brands[0].brand} ${brands[0].version}`;
  const m = navigator.userAgent.match(/(Firefox|Edg|Chrome|Version)\/(\d+)/);
  return m ? `${m[1] === 'Edg' ? 'Edge' : m[1] === 'Version' ? 'Safari' : m[1]} ${m[2]}` : navigator.userAgent;
}
