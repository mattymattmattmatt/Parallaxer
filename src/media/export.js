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
  canEncodeVideo
} from 'mediabunny';
import { StereoRenderer, LAYOUT } from '../gl/renderer.js';
import { FrameProcessor } from '../core/pipeline.js';
import { LAYOUTS, layoutGeometry } from '../core/settings.js';

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

/**
 * Converts a video file to a stereoscopic video. Hardware-decoded, depth-processed on the GPU,
 * hardware-encoded, with the original audio copied bit-exact when the container allows it.
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
  }

  /** End the export at the last rendered frame and finalise a playable file with everything done so far. */
  stopAndSave() {
    if (this.state !== 'running' && this.state !== 'paused') return;
    this.stopRequested = true;
    this.pauseCtl?.abort();
    const r = this.resumeResolve;
    this.resumeResolve = null;
    r?.();
  }

  pause() {
    if (this.state !== 'running') return;
    this.paused = true;
    this.pauseCtl?.abort();
  }

  resume() {
    if (!this.paused) return;
    this.paused = false;
    const r = this.resumeResolve;
    this.resumeResolve = null;
    r?.();
  }

  async cancel() {
    this.cancelled = true;
    this.pauseCtl?.abort();
    this.resumeResolve?.();
    for (const c of [this.conversion, this.audioConversion]) {
      try {
        await c?.cancel();
      } catch {
        /* ignore */
      }
    }
  }

  #tidy(input, renderer) {
    input.dispose();
    renderer.dispose();
  }

  async run() {
    const o = this.o;
    const geo = layoutGeometry(o.settings.layout, o.eyeW, o.eyeH);
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
    const s = o.settings;
    let lastTs = -Infinity;
    const startWall = performance.now();
    let procTime = 0;

    const frameCanvas = new OffscreenCanvas(2, 2);
    const frameCtx = frameCanvas.getContext('2d', { alpha: false });
    const layoutLabel = LAYOUTS[s.layout]?.label ?? s.layout;
    const conversion = await Conversion.init({
      input,
      output,
      tracks: 'primary',
      trim: o.trim ?? undefined,
      showWarnings: false,
      video: {
        codec: o.codec,
        quality: makeQuality(o.quality, geo.canvasW, geo.canvasH, o.frameRate || o.sourceFps, o.codec),
        keyFrameInterval: o.keyFrameInterval ?? 2,
        frameRate: o.frameRate || undefined,
        allowTransformationMetadata: false,
        forceTranscode: true,
        hardwareAcceleration: o.hardwareAcceleration ?? 'no-preference',
        processedWidth: geo.canvasW,
        processedHeight: geo.canvasH,
        process: async (sample) => {
          await yieldToUI();
          if (this.cancelled) throw new ExportCancelled();
          const t0 = performance.now();
          // drawWithFit applies rotation, flip and pixel aspect ratio and crops coded padding in one GPU draw.
          const fw = sample.displayWidth;
          const fh = sample.displayHeight;
          if (frameCanvas.width !== fw || frameCanvas.height !== fh) {
            frameCanvas.width = fw;
            frameCanvas.height = fh;
          }
          sample.drawWithFit(frameCtx, { fit: 'fill' });
          const continuous = sample.timestamp > lastTs && sample.timestamp - lastTs < 1;
          lastTs = sample.timestamp;
          const stats = await fp.ingest(frameCanvas, fw, fh, s, { continuous });
          fp.renderOutput(s, geo);
          const out = new VideoFrame(canvas, {
            timestamp: Math.round(sample.timestamp * 1e6),
            duration: Math.max(1, Math.round(sample.duration * 1e6))
          });
          this.frames++;
          this.lastTs = sample.timestamp;
          this.renderedEnd = sample.timestamp + sample.duration;
          advanceAudio(sample.timestamp);
          procTime += performance.now() - t0;
          o.onFrame?.({ canvas, frames: this.frames, stats, timestamp: sample.timestamp });
          return new VideoSample(out);
        }
      },
      // Audio runs in its own conversion (below) so it can be held back to the last rendered video frame.
      audio: { discard: true },
      // Composable: this job owns the output's lifecycle, which is what lets "Stop & save" finalise early.
      composable: true
    });
    this.conversion = conversion;

    // Audio is copied (or transcoded when the container needs it) by a second conversion into the same output,
    // advanced in lockstep behind the video so a partial export never carries audio past its last frame.
    let audioConv = null;
    if (o.audio !== 'none') {
      audioConv = await Conversion.init({
        input,
        output,
        tracks: 'primary',
        trim: o.trim ?? undefined,
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

    for (const d of conversion.discardedTracks) {
      if (d.reason === 'discarded_by_user') continue;
      this.warnings.push(`${d.track.type === 'audio' ? 'Audio' : 'Video'} track dropped (${d.reason.replaceAll('_', ' ')}).`);
    }
    // Composable conversions are always "valid"; require a usable video track ourselves.
    if (!conversion.utilizedTracks.some((t) => t.isVideoTrack())) {
      this.#tidy(input, renderer);
      await writable?.abort?.().catch(() => {});
      const why = conversion.discardedTracks.map((d) => `${d.track.type}: ${d.reason.replaceAll('_', ' ')}`).join('; ');
      throw new Error(`This file can't be converted in this browser (${why || 'no usable tracks'}).`);
    }

    // Composable conversions leave metadata to the output's owner: carry the source tags over.
    const inputTags = await input.getMetadataTags().catch(() => ({}));
    const tags = { ...inputTags, comment: `Stereoscopic 3D (${layoutLabel}) converted with Parallaxer` };
    delete tags.raw;
    output.setMetadataTags(tags);
    await output.start();

    conversion.onProgress = (progress, time) => {
      const elapsed = (performance.now() - startWall) / 1000;
      o.onProgress?.({
        progress,
        time,
        frames: this.frames,
        elapsed,
        fps: this.frames / Math.max(0.001, procTime / 1000),
        eta: progress > 0.01 ? (elapsed / progress) * (1 - progress) : null
      });
    };

    this.state = 'running';
    let partial = false;
    try {
      for (;;) {
        this.pauseCtl = new AbortController();
        await conversion.execute({ pauseSignal: this.pauseCtl.signal });
        if (conversion.state === 'done') break;
        if (this.cancelled) throw new ExportCancelled();
        if (this.stopRequested) {
          partial = true;
          break;
        }
        if (this.paused) {
          this.state = 'paused';
          o.onState?.('paused');
          await new Promise((r) => (this.resumeResolve = r));
          if (this.cancelled) throw new ExportCancelled();
          if (this.stopRequested) {
            partial = true;
            break;
          }
          this.state = 'running';
          o.onState?.('running');
        }
      }
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
    } catch (err) {
      this.state = 'failed';
      await output.cancel().catch(() => {});
      this.#tidy(input, renderer);
      if (writable) {
        await writable.abort?.().catch(() => {});
        await o.fileHandle.remove?.().catch(() => {});
      }
      if (this.cancelled || err?.name === 'ConversionCanceledError') throw new ExportCancelled();
      throw err;
    }

    this.state = 'done';
    this.#tidy(input, renderer);
    const elapsed = (performance.now() - startWall) / 1000;
    const summary = { frames: this.frames, elapsed, warnings: this.warnings, partial, rendered: this.renderedEnd };
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
