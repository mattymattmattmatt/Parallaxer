import {
  Input,
  Output,
  Conversion,
  BlobSource,
  BufferTarget,
  StreamTarget,
  ALL_FORMATS,
  Mp4OutputFormat,
  WebMOutputFormat,
  MkvOutputFormat,
  MovOutputFormat,
  CanvasSource,
  Quality,
  VideoSample,
  VideoSampleSink,
  VideoSampleSource,
  canEncodeVideo
} from 'mediabunny';
import { StereoRenderer, LAYOUT } from '../gl/renderer.js';
import { FrameProcessor } from '../core/pipeline.js';
import { LAYOUTS, layoutGeometry } from '../core/settings.js';
import { prepareFrame } from '../depth/stabilizer.js';
import { PoolExhausted } from '../depth/pool.js';
import { ExportProfiler, browserName } from './perf.js';

export const CONTAINERS = {
  mp4: { label: 'MP4', ext: 'mp4', mime: 'video/mp4', codecs: ['avc', 'hevc', 'av1', 'vp9'] },
  mov: { label: 'MOV', ext: 'mov', mime: 'video/quicktime', codecs: ['avc', 'hevc'] },
  webm: { label: 'WebM', ext: 'webm', mime: 'video/webm', codecs: ['vp9', 'av1', 'vp8'] },
  mkv: { label: 'MKV', ext: 'mkv', mime: 'video/x-matroska', codecs: ['avc', 'hevc', 'av1', 'vp9'] }
};

export const VIDEO_CODECS = {
  avc: 'H.264',
  hevc: 'H.265 / HEVC',
  av1: 'AV1',
  vp9: 'VP9',
  vp8: 'VP8'
};

export const QUALITY_LEVELS = [
  { id: 'medium', label: 'Standard' },
  { id: 'high', label: 'High' },
  { id: 'very-high', label: 'Master' }
];

function makeFormat(container, streaming) {
  switch (container) {
    case 'webm':
      return new WebMOutputFormat();
    case 'mkv':
      return new MkvOutputFormat();
    case 'mov':
      return new MovOutputFormat({ fastStart: streaming ? false : 'in-memory' });
    default:
      return new Mp4OutputFormat({ fastStart: streaming ? false : 'in-memory' });
  }
}

// Bits per pixel per frame for each quality tier (H.264 reference) and relative codec efficiency.
const BPP = { medium: 0.07, high: 0.12, 'very-high': 0.2 };
const CODEC_EFFICIENCY = { avc: 1, hevc: 0.6, vp9: 0.65, av1: 0.5, vp8: 1.15 };

/** Target bitrate (bits/s) for a quality tier; explicit so results are predictable across encoders. */
export function targetBitrate(q, width, height, fps, codec) {
  if (q.mode === 'bitrate') return Math.round(q.mbps * 1e6);
  const bpp = BPP[q.level] ?? BPP.high;
  const raw = width * height * (fps || 30) * bpp * (CODEC_EFFICIENCY[codec] ?? 1);
  return Math.round(Math.min(250e6, Math.max(1.5e6, raw)));
}

function makeQuality(q, width, height, fps, codec) {
  return new Quality({ bitrate: targetBitrate(q, width, height, fps, codec), bitrateMode: 'variable' });
}

/** Which codecs of a container can encode the given frame size in this browser. */
export async function probeEncoders(container, width, height) {
  const list = CONTAINERS[container]?.codecs ?? [];
  const res = await Promise.all(
    list.map(async (codec) => {
      try {
        return { codec, ok: await canEncodeVideo(codec, { width, height }) };
      } catch {
        return { codec, ok: false };
      }
    })
  );
  return res;
}

export function baseName(fileName) {
  return (
    fileName
      .replace(/\.[^.]+$/, '')
      .replace(/[\u2012-\u2015]/g, '-')
      .replace(/[\u2018\u2019]/g, "'")
      .replace(/[\u201c\u201d]/g, '"')
      .replace(/[\\/:*?"<>|]+/g, '_')
      .trim() || 'parallaxer'
  );
}

export function outputName(fileName, layoutId, ext) {
  const base = baseName(fileName);
  const tag = LAYOUTS[layoutId]?.tag ?? '3D';
  return `${base}.${tag}.${ext}`;
}

export function supportsDiskStreaming() {
  return typeof window.showSaveFilePicker === 'function';
}

/** Ask for a destination file up front (must run inside a user gesture). */
export async function pickSaveFile(suggestedName, container) {
  const c = CONTAINERS[container] ?? CONTAINERS.mp4;
  return window.showSaveFilePicker({
    suggestedName,
    types: [{ description: `${c.label} video`, accept: { [c.mime]: [`.${c.ext}`] } }]
  });
}

/**
 * Give the browser a turn between frames. With CPU (WASM) inference and pre-decoded frames the export can
 * otherwise run back-to-back without ever returning to the event loop, starving input and rendering so
 * Pause / Stop / Cancel stop responding. A timeout task queues behind pending input and paint work (unlike
 * scheduler.yield(), whose continuation jumps the queue), and only kicks in after ~30 ms of busy time.
 */
let lastYield = 0;
async function yieldToUI() {
  const now = performance.now();
  if (now - lastYield < 30) return;
  await new Promise((resolve) => setTimeout(resolve, 0));
  lastYield = performance.now();
}

export class ExportCancelled extends Error {
  constructor() {
    super('Export cancelled');
    this.name = 'ExportCancelled';
  }
}

/** Frame-centre sample times for constant-rate output (k-th output frame shows the source at (k + ½)/fps). */
function* frameTimes(start, end, fps) {
  const n = Math.max(1, Math.round((end - start) * fps));
  for (let k = 0; k < n; k++) yield start + (k + 0.5) / fps;
}

// Decoded frames waiting for depth are kept as GPU bitmaps; cap their total size.
const IN_FLIGHT_BUDGET_BYTES = 512 * 2 ** 20;
const MAX_IN_FLIGHT = 12;

/**
 * Converts a video file to a stereoscopic video.
 *
 * Pipeline: hardware decode → GPU downscale + async readback → depth inference (a pool of background workers,
 * or the page's own engine) → in-order stabilise + render + hardware encode. Several frames are in flight at
 * once so the CPU and GPU work side by side, but frames are always finished strictly in order, so the result
 * is the same as a one-frame-at-a-time export. Original audio is copied in lockstep behind the video.
 *
 * Options: { file, engine, settings, eyeW, eyeH, container, codec, quality, frameRate, sourceFps, audio, trim,
 *            fileHandle, pool?, tuner?, workerCap?, modelName?, onFrame, onProgress, onState, onNotice, onStats }
 * onStats receives a profiler snapshot about once a second (see perf.js); the result carries the final one.
 */
export class VideoExportJob {
  constructor(opts) {
    this.o = opts;
    this.state = 'idle';
    this.frames = 0;
    this.cancelled = false;
    this.paused = false;
    this.stopRequested = false;
    this.lastTs = null;
    this.renderedEnd = 0;
    this.warnings = [];
    this.poolDown = false;
  }

  pause() {
    if (this.state !== 'running') return;
    this.paused = true;
  }

  resume() {
    if (!this.paused) return;
    this.paused = false;
    this.o.tuner?.interrupted();
    this.#wake();
  }

  /** End the export at the last rendered frame and finalise a playable file with everything done so far. */
  stopAndSave() {
    if (this.state !== 'running' && this.state !== 'paused') return;
    this.stopRequested = true;
    this.#wake();
  }

  async cancel() {
    this.cancelled = true;
    this.#wake();
    try {
      await this.audioConversion?.cancel();
    } catch {
      /* ignore */
    }
  }

  #wake() {
    const r = this.resumeResolve;
    this.resumeResolve = null;
    r?.();
  }

  /** Pause / stop / cancel checkpoint between frames. Returns false when the export should stop dispatching. */
  async #gate() {
    if (this.cancelled) throw new ExportCancelled();
    if (this.stopRequested) return false;
    if (this.paused) {
      this.state = 'paused';
      this.o.onState?.('paused');
      const t0 = performance.now();
      await new Promise((r) => (this.resumeResolve = r));
      this.pausedMs += performance.now() - t0;
      this.prof?.paused(performance.now() - t0);
      if (this.cancelled) throw new ExportCancelled();
      if (this.stopRequested) return false;
      this.state = 'running';
      this.o.onState?.('running');
    }
    return true;
  }

  /** Depth for one frame: the worker pool when available, otherwise (or after it gives up) the page engine. */
  async #depth(rgba, size) {
    const o = this.o;
    if (o.pool && !this.poolDown) {
      try {
        return await o.pool.infer(rgba, size.w, size.h);
      } catch (err) {
        if (!(err instanceof PoolExhausted)) throw err;
        this.poolDown = true;
        o.onNotice?.(`${err.message} Continuing in the page — slower, but the export carries on.`, 'warn');
      }
    }
    const res = await o.engine.infer(rgba, size.w, size.h);
    const t0 = performance.now();
    const prepared = prepareFrame(res.data, res.w, res.h, rgba, size.w, size.h);
    // On the CPU the model (and this prep) run on the page thread and block everything else meanwhile.
    if (o.engine.backend === 'wasm') this.prof?.pageDepthBlock(res.ms + performance.now() - t0);
    return { ...res, ...prepared };
  }

  async run() {
    const o = this.o;
    const s = o.settings;
    const geo = layoutGeometry(s.layout, o.eyeW, o.eyeH);
    const canvas = new OffscreenCanvas(geo.canvasW, geo.canvasH);
    const renderer = new StereoRenderer(canvas, { preserveDrawingBuffer: true });
    const fp = new FrameProcessor(renderer, o.engine);
    const input = new Input({ source: new BlobSource(o.file), formats: ALL_FORMATS });
    const streaming = !!o.fileHandle;
    let writable = null;
    let target;
    if (streaming) {
      writable = await o.fileHandle.createWritable();
      target = new StreamTarget(writable, { chunked: true, chunkSize: 8 * 2 ** 20 });
    } else {
      target = new BufferTarget();
    }
    const output = new Output({ format: makeFormat(o.container, streaming), target });
    const queue = [];
    const startWall = performance.now();
    this.pausedMs = 0;
    const layoutLabel = LAYOUTS[s.layout]?.label ?? s.layout;

    const tidy = () => {
      for (const item of queue.splice(0)) item.bitmap.close();
      input.dispose();
      renderer.dispose();
    };

    let audioConv = null;
    let audioRun = null;
    let audioUntil = -Infinity;
    let audioError = null;
    const runAudio = (until) => {
      audioUntil = until;
      audioRun = audioConv
        .execute({ until })
        .catch((e) => {
          audioError ??= e;
        })
        .finally(() => {
          audioRun = null;
        });
      return audioRun;
    };
    const advanceAudio = (t) => {
      if (audioConv && !audioRun && t > audioUntil) runAudio(t);
    };
    const settleAudio = async (until) => {
      if (!audioConv) return;
      while (audioRun) await audioRun;
      if (until > audioUntil) await runAudio(until);
      if (audioError && !this.cancelled) throw audioError;
    };

    try {
      const videoTrack = await input.getPrimaryVideoTrack();
      if (!videoTrack) throw new Error('This file has no video track.');
      if (!(await videoTrack.canDecode())) throw new Error("This browser can't decode the video in this file.");
      const audioTrack = o.audio === 'none' ? null : await input.getPrimaryAudioTrack();
      const tracks = [videoTrack, audioTrack].filter(Boolean);
      const first = Math.max(0, await input.getFirstTimestamp(tracks).catch(() => 0));
      const start = o.trim?.start ?? first;
      const end = o.trim?.end ?? (await input.computeDuration(tracks));
      const span = Math.max(1e-3, end - start);

      const fpsOut = o.frameRate || null;
      const videoSource = new VideoSampleSource({
        codec: o.codec,
        quality: makeQuality(o.quality, geo.canvasW, geo.canvasH, fpsOut || o.sourceFps, o.codec),
        keyFrameInterval: o.keyFrameInterval ?? 2,
        hardwareAcceleration: o.hardwareAcceleration ?? 'no-preference'
      });
      output.addVideoTrack(videoSource, fpsOut || o.sourceFps ? { frameRate: fpsOut || o.sourceFps } : {});

      // Audio is copied (or transcoded when the container needs it) by a composable conversion into the same
      // output, advanced in lockstep behind the video so a partial export never carries audio past its last frame.
      if (audioTrack) {
        audioConv = await Conversion.init({
          input,
          output,
          tracks: 'primary',
          trim: { start, end },
          showWarnings: false,
          video: { discard: true },
          audio: {},
          composable: true
        });
        for (const d of audioConv.discardedTracks) {
          if (d.reason !== 'discarded_by_user') this.warnings.push(`Audio track dropped (${d.reason.replaceAll('_', ' ')}).`);
        }
        if (!audioConv.utilizedTracks.length) audioConv = null;
      }
      this.audioConversion = audioConv;

      const inputTags = await input.getMetadataTags().catch(() => ({}));
      const tags = { ...inputTags, comment: `Stereoscopic 3D (${layoutLabel}) converted with Parallaxer` };
      delete tags.raw;
      output.setMetadataTags(tags);
      await output.start();

      const prof = (this.prof = new ExportProfiler());
      const codecName = VIDEO_CODECS[o.codec] ?? o.codec;
      Object.assign(prof.info, {
        sourceCodec: VIDEO_CODECS[videoTrack.codec] ?? videoTrack.codec ?? null,
        output: `${geo.canvasW}×${geo.canvasH}`,
        codecName,
        streaming,
        model: o.modelName ?? null,
        workersMax: o.workerCap ?? null,
        browser: browserName()
      });
      // Whether the GPU's video engines can take this job; answered in the background, shown when known.
      canEncodeVideo(o.codec, { width: geo.canvasW, height: geo.canvasH, hardwareAcceleration: 'prefer-hardware' })
        .then((ok) => (prof.info.encoderHw = !!ok))
        .catch(() => {});
      videoTrack
        .getDecoderConfig()
        .then((cfg) => cfg && VideoDecoder.isConfigSupported({ ...cfg, hardwareAcceleration: 'prefer-hardware' }))
        .then((r) => r && (prof.info.decoderHw = !!r.supported))
        .catch(() => {});
      let lastStats = 0;
      const describeProcessing = () => {
        const n = o.pool && !this.poolDown ? o.pool.ready.length : 0;
        prof.info.workers = n;
        prof.info.processing = n ? `${n} background worker${n > 1 ? 's' : ''}` : 'in page';
        const backends = n ? [...new Set(o.pool.ready.map((w) => w.backend))] : [o.engine.backend];
        prof.info.backend = backends.map((b) => (b === 'webgpu' ? 'WebGPU' : b === 'wasm' ? 'CPU (WASM)' : b)).join(' + ');
      };

      const sink = new VideoSampleSink(videoTrack);
      const samples = fpsOut ? sink.samplesAtTimestamps(frameTimes(start, end, fpsOut)) : sink.samples(start, end);
      const frameCanvas = new OffscreenCanvas(2, 2);
      const frameCtx = frameCanvas.getContext('2d', { alpha: false });
      let memCap = MAX_IN_FLIGHT;
      const capacity = () => {
        const workers = o.pool && !this.poolDown ? o.pool.ready.length : 0;
        return Math.max(2, Math.min(workers ? workers * 2 + 2 : 3, memCap));
      };

      const emit = async (item) => {
        prof.sample('inflight', queue.length + 1);
        let res;
        try {
          res = await prof.time('depth', item.depthP);
        } finally {
          if (this.cancelled) item.bitmap.close();
        }
        if (this.cancelled) throw new ExportCancelled();
        prof.sample('depthMs', res.ms);
        const t0 = performance.now();
        renderer.stage(item.bitmap, item.fw, item.fh);
        const continuous = this.lastTs !== null && item.ts > this.lastTs && item.ts - this.lastTs < 1;
        const stats = fp.finish(res, item.size, s, { continuous });
        item.bitmap.close();
        const timed = renderer.beginGpuTimer();
        fp.renderOutput(s, geo);
        if (timed) renderer.endGpuTimer();
        const sample = new VideoSample(
          new VideoFrame(canvas, { timestamp: Math.round(item.ts * 1e6), duration: Math.max(1, Math.round(item.dur * 1e6)) })
        );
        prof.add('stabilise', fp.stabiliseMs);
        prof.add('render', performance.now() - t0 - fp.stabiliseMs);
        for (const ms of renderer.takeGpuTimings?.() ?? []) prof.sample('gpuRenderMs', ms);
        try {
          await prof.time('encode', videoSource.add(sample));
        } finally {
          sample.close();
        }
        prof.frameDone();
        this.frames++;
        this.lastTs = item.ts;
        this.renderedEnd = item.ts + item.dur;
        advanceAudio(item.ts);
        const tUi = performance.now();
        o.onFrame?.({ canvas, frames: this.frames, stats, timestamp: item.ts });
        o.tuner?.frame();
        const elapsed = (performance.now() - startWall) / 1000;
        const active = Math.max(0.001, elapsed - this.pausedMs / 1000);
        const progress = Math.min(1, this.renderedEnd / span);
        o.onProgress?.({
          progress,
          time: this.renderedEnd,
          frames: this.frames,
          elapsed,
          fps: this.frames / active,
          eta: progress > 0.01 ? (active / progress) * (1 - progress) : null
        });
        const now = performance.now();
        if (o.onStats && now - lastStats > 1000) {
          lastStats = now;
          describeProcessing();
          o.onStats(prof.snapshot());
        }
        prof.add('ui', performance.now() - tUi);
      };

      this.state = 'running';
      let k = 0;
      describeProcessing();
      prof.start();
      let tWait = performance.now();
      for await (const sample of samples) {
        prof.add('decode', performance.now() - tWait);
        try {
          if (!sample) {
            k++;
            continue;
          }
          let ts;
          let dur;
          if (fpsOut) {
            ts = k / fpsOut;
            dur = 1 / fpsOut;
            k++;
          } else {
            const a = Math.max(start, sample.timestamp);
            const b = Math.min(end, sample.timestamp + sample.duration);
            if (b <= a) {
              sample.close();
              continue;
            }
            ts = a - start;
            dur = b - a;
          }
          let go;
          try {
            go = await this.#gate();
          } catch (err) {
            sample.close();
            throw err;
          }
          if (!go) {
            sample.close();
            break;
          }
          // drawWithFit applies rotation, flip and pixel aspect ratio and crops coded padding in one GPU draw.
          const tPrep = performance.now();
          const fw = sample.displayWidth;
          const fh = sample.displayHeight;
          if (frameCanvas.width !== fw || frameCanvas.height !== fh) {
            frameCanvas.width = fw;
            frameCanvas.height = fh;
          }
          sample.drawWithFit(frameCtx, { fit: 'fill' });
          sample.close();
          const bitmap = frameCanvas.transferToImageBitmap();
          const size = o.engine.inputSize(fw, fh, s.detail);
          if (!this.frames && !queue.length) {
            const perFrame = fw * fh * 4 + size.w * size.h * 16;
            memCap = Math.max(2, Math.min(MAX_IN_FLIGHT, Math.floor(IN_FLIGHT_BUDGET_BYTES / perFrame)));
          }
          if (!prof.info.source) {
            prof.info.source = `${fw}×${fh}${o.sourceFps ? ` @ ${+o.sourceFps.toFixed(3)} fps` : ''}`;
            prof.info.modelInput = `${size.w}×${size.h}`;
          }
          renderer.stage(bitmap, fw, fh);
          const tRead = performance.now();
          const depthP = renderer.readModelInputAsync(size.w, size.h).then((rgba) => {
            prof.sample('readbackMs', performance.now() - tRead);
            return this.#depth(rgba, size);
          });
          depthP.catch(() => {}); // surfaced when the frame is emitted
          queue.push({ bitmap, fw, fh, ts, dur, size, depthP });
          prof.add('prepare', performance.now() - tPrep);
          while (queue.length >= capacity()) await emit(queue.shift());
          await yieldToUI();
        } finally {
          tWait = performance.now();
        }
      }
      // Finish every frame that is already in flight (also on Stop & save: that work is already paid for).
      while (queue.length) {
        if (this.cancelled) throw new ExportCancelled();
        await emit(queue.shift());
      }
      describeProcessing();
      const finalStats = prof.snapshot({ recent: false });

      const partial = this.stopRequested;
      if (partial) {
        if (!this.frames) throw new ExportCancelled();
        this.state = 'finishing';
        o.onState?.('finishing');
        // Bring audio exactly up to the end of the last rendered frame so both streams end together.
        await settleAudio(this.renderedEnd);
      } else {
        await settleAudio(Infinity);
      }
      if (this.cancelled) throw new ExportCancelled();
      await output.finalize();

      this.state = 'done';
      tidy();
      const elapsed = (performance.now() - startWall) / 1000;
      const summary = { frames: this.frames, elapsed, warnings: this.warnings, partial, rendered: this.renderedEnd, stats: finalStats };
      if (streaming) {
        try {
          await writable.close();
        } catch {
          /* already closed by the target */
        }
        const f = await o.fileHandle.getFile().catch(() => null);
        return { ...summary, streamed: true, name: o.fileHandle.name, bytes: f?.size ?? 0 };
      }
      const mime = CONTAINERS[o.container]?.mime ?? 'video/mp4';
      const blob = new Blob([output.target.buffer], { type: mime });
      return { ...summary, blob, bytes: blob.size };
    } catch (err) {
      this.state = 'failed';
      try {
        await audioConv?.cancel();
      } catch {
        /* ignore */
      }
      await output.cancel().catch(() => {});
      tidy();
      if (writable) {
        await writable.abort?.().catch(() => {});
        await o.fileHandle.remove?.().catch(() => {});
      }
      if (this.cancelled || err?.name === 'ConversionCanceledError' || err instanceof ExportCancelled) throw new ExportCancelled();
      throw err;
    }
  }
}

/**
 * Render a still (image source or a paused video frame) at full resolution.
 * format: 'png' | 'jpeg' | 'webp' | 'jps'
 */
export async function renderStill({ source, width, height, settings, engine, format = 'png', quality = 0.95, scale = 1 }) {
  const s = { ...settings };
  let layoutId = s.layout;
  if (format === 'jps') {
    layoutId = 'sbs-full';
    s.swap = !s.swap; // JPS stores the right view first (cross-eyed order)
  }
  s.layout = layoutId;
  const geo = layoutGeometry(layoutId, width * scale, height * scale);
  const canvas = new OffscreenCanvas(geo.canvasW, geo.canvasH);
  const renderer = new StereoRenderer(canvas, { preserveDrawingBuffer: true });
  try {
    const fp = new FrameProcessor(renderer, engine);
    await fp.ingest(source, width, height, s, { continuous: false });
    fp.renderOutput(s, geo);
    const type = format === 'png' ? 'image/png' : format === 'webp' ? 'image/webp' : 'image/jpeg';
    return await canvas.convertToBlob({ type, quality });
  } finally {
    renderer.dispose();
  }
}

export const MOTION_PATHS = {
  sway: { label: 'Sway', hint: 'Side-to-side camera drift' },
  orbit: { label: 'Orbit', hint: 'Circular camera move (3D photo)' },
  dolly: { label: 'Dolly zoom', hint: 'Push in with depth-aware parallax' },
  swing: { label: 'Swing + push', hint: 'Orbit combined with a gentle push-in' },
  wiggle: { label: 'Wigglegram', hint: 'Snapping left/right views' }
};

/** Camera for a motion path at phase t in [0,1). Returns eye params for the warp shader. */
export function motionEye(path, t, amount, s) {
  const S = s.strength / 100;
  const c = s.convergence;
  const tau = Math.PI * 2 * t;
  // Directional parallax at full amount spans `amount` × the stereo budget.
  const overscanFor = (e) => 1 / Math.max(0.5, 1 - 2 * Math.abs(e) * S * Math.max(c, 1 - c) - 0.01);
  if (path === 'wiggle') {
    const e = (Math.sin(tau) >= 0 ? 0.5 : -0.5) * amount;
    return { e, dir: [1, 0], zoom: overscanFor(0.5 * amount) };
  }
  if (path === 'sway') {
    const e = Math.sin(tau) * amount;
    return { e, dir: [1, 0], zoom: overscanFor(amount) };
  }
  if (path === 'dolly') {
    const k = 0.18 * amount * (0.5 - 0.5 * Math.cos(tau));
    const mMin = Math.max(0.05, 1 - k * c);
    return { e: k, radial: true, center: [0.5, 0.5], zoom: 1 / mMin + 0.005 };
  }
  const vx = Math.cos(tau);
  const vy = Math.sin(tau) * 0.6;
  const len = Math.hypot(vx, vy);
  const eye = { e: amount * len, dir: [vx / len, vy / len], zoom: overscanFor(amount) };
  if (path === 'swing') eye.zoom *= 1 + 0.06 * (0.5 - 0.5 * Math.cos(tau));
  return eye;
}

/** Render an animated "3D photo" from a still image. */
export async function exportMotion({ source, width, height, settings, engine, path, seconds, fps, amount, container, codec, quality, maxSize, onProgress, isCancelled }) {
  const scale = Math.min(1, (maxSize ?? 1920) / Math.max(width, height));
  const w = Math.max(2, Math.round((width * scale) / 2) * 2);
  const h = Math.max(2, Math.round((height * scale) / 2) * 2);
  const canvas = new OffscreenCanvas(w, h);
  const renderer = new StereoRenderer(canvas, { preserveDrawingBuffer: true });
  try {
    const fp = new FrameProcessor(renderer, engine);
    await fp.ingest(source, width, height, settings, { continuous: false });
    const target = new BufferTarget();
    const output = new Output({ format: makeFormat(container, false), target });
    const videoSource = new CanvasSource(canvas, { codec, quality: makeQuality(quality, w, h, fps, codec), keyFrameInterval: 2 });
    output.addVideoTrack(videoSource, { frameRate: fps });
    await output.start();
    const total = Math.max(2, Math.round(seconds * fps));
    const geo = { canvasW: w, canvasH: h, eyeW: w, eyeH: h, logicalW: w, logicalH: h };
    for (let i = 0; i < total; i++) {
      if (isCancelled?.()) {
        await output.cancel();
        throw new ExportCancelled();
      }
      if (i % 4 === 0) await yieldToUI();
      const eye = motionEye(path, i / total, amount, settings);
      renderer.render(fp.params(settings, geo, { layout: LAYOUT.MONO_L, eyes: [eye] }));
      await videoSource.add(i / fps, 1 / fps);
      onProgress?.((i + 1) / total);
    }
    await output.finalize();
    return new Blob([target.buffer], { type: CONTAINERS[container]?.mime ?? 'video/mp4' });
  } finally {
    renderer.dispose();
  }
}
