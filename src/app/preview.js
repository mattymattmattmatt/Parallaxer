import { StereoRenderer, LAYOUT } from '../gl/renderer.js';
import { FrameProcessor, eyeFactors } from '../core/pipeline.js';
import { LAYOUTS, layoutGeometry } from '../core/settings.js';
import { motionEye } from '../media/export.js';

const MAX_PREVIEW_PIXELS = 4.2e6;

/**
 * Live preview: pulls frames from a <video> (file, camera or screen) or a still image, keeps colour and
 * depth in lock-step, and re-renders instantly when only view parameters change.
 */
export class Preview {
  constructor({ canvas, video, viewer, engine, store, onStats, onError, onRendered }) {
    this.canvas = canvas;
    this.video = video;
    this.viewer = viewer;
    this.engine = engine;
    this.store = store;
    this.onStats = onStats;
    this.onError = onError;
    this.onRendered = onRendered;
    this.renderer = new StereoRenderer(canvas);
    this.fp = new FrameProcessor(this.renderer, engine);
    this.source = null;
    this.busy = false;
    this.again = null;
    this.raf = 0;
    this.pointer = null;
    this.compare = false;
    this.lastMediaTime = -1;
    this.processedTime = null;
    this.frameTimes = [];
    this.stats = null;
    this.animStart = performance.now();

    video.addEventListener('seeked', () => this.#videoFrame(false));
    video.addEventListener('loadeddata', () => this.#videoFrame(false));

    viewer.addEventListener('pointermove', (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.pointer = [((e.clientX - r.left) / r.width) * 2 - 1, ((e.clientY - r.top) / r.height) * 2 - 1];
      if (this.store.state.view === 'look') this.requestDraw();
    });
    viewer.addEventListener('pointerleave', () => {
      this.pointer = null;
    });
    new ResizeObserver(() => this.requestDraw()).observe(viewer);
  }

  get hasSource() {
    return !!this.source;
  }

  get sourceSize() {
    return this.source ? { w: this.source.w, h: this.source.h } : null;
  }

  clear() {
    this.gen = (this.gen ?? 0) + 1;
    this.lastMediaTime = -1;
    this.processedTime = null;
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.srcObject = null;
    this.video.load();
    this.source = null;
    this.fp.reset();
    this.renderer.hasSource = false;
    this.requestDraw();
  }

  setImage(bitmap) {
    this.clear();
    this.source = { kind: 'image', el: bitmap, w: bitmap.width, h: bitmap.height };
    this.refresh();
  }

  setVideo(url, meta) {
    this.clear();
    this.source = { kind: 'video', el: this.video, w: meta.width, h: meta.height, fps: meta.fps };
    this.video.srcObject = null;
    this.video.src = url;
    this.video.muted = false;
    this.#armVfc();
  }

  setStream(stream) {
    this.clear();
    this.source = { kind: 'live', el: this.video, w: 0, h: 0, fps: 30 };
    this.video.srcObject = stream;
    this.video.muted = true;
    this.video.play().catch(() => {});
    this.#armVfc();
  }

  // Each source gets its own callback chain; the generation counter retires chains from earlier sources.
  #armVfc() {
    const gen = (this.gen = (this.gen ?? 0) + 1);
    const live = () => gen === this.gen && this.source?.el === this.video;
    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      const cb = (_now, md) => {
        if (!live()) return;
        this.vfc(md.mediaTime);
        this.video.requestVideoFrameCallback(cb);
      };
      this.video.requestVideoFrameCallback(cb);
    } else {
      const poll = () => {
        if (!live()) return;
        if (!this.video.paused) this.#videoFrame(true);
        requestAnimationFrame(poll);
      };
      requestAnimationFrame(poll);
    }
  }

  vfc(t) {
    const continuous = this.source.kind === 'live' || (t > this.lastMediaTime && t - this.lastMediaTime < 0.3);
    this.lastMediaTime = t;
    this.#videoFrame(continuous, t);
  }

  #videoFrame(continuous, mediaTime = this.video.currentTime) {
    const v = this.video;
    if (!this.source || v.readyState < 2 || !v.videoWidth) return;
    this.source.w = v.videoWidth;
    this.source.h = v.videoHeight;
    if (!continuous && this.processedTime === mediaTime && !this.busy) return;
    this.ingest(continuous, mediaTime);
  }

  /** Poll the video for a new frame (used when the page's own frame callbacks are throttled, e.g. in WebXR). */
  pump() {
    const src = this.source;
    if (!src || src.el !== this.video || this.busy) return;
    const t = this.video.currentTime;
    if (t === this.pumpTime) return;
    this.pumpTime = t;
    this.#videoFrame(!this.video.paused, t);
  }

  /** Re-run depth on the current frame (after model / detail changes). */
  refresh() {
    if (!this.source) return;
    if (this.source.kind === 'image') this.ingest(false);
    else this.#videoFrame(false);
  }

  async ingest(continuous, mediaTime = null) {
    const src = this.source;
    if (!src) return;
    if (this.busy) {
      this.again = { continuous: this.again ? this.again.continuous && continuous : continuous };
      // Smooth playback: show every decoded frame with the most recent depth while inference catches up.
      if (src.kind !== 'image' && this.store.state.smoothPlayback && continuous && src.w) {
        this.renderer.uploadSource(src.el, src.w, src.h);
        this.requestDraw();
      }
      return;
    }
    if (!this.engine.ready) {
      this.renderer.uploadSource(src.el, src.w, src.h);
      this.requestDraw();
      return;
    }
    this.busy = true;
    const t0 = performance.now();
    try {
      const stats = await this.fp.ingest(src.el, src.w, src.h, this.store.state, { continuous });
      this.processedTime = mediaTime;
      const now = performance.now();
      this.frameTimes.push(now);
      while (this.frameTimes.length && now - this.frameTimes[0] > 1000) this.frameTimes.shift();
      this.stats = { ...stats, total: now - t0, fps: this.frameTimes.length };
      this.onStats?.(this.stats, this.fp.convergence(this.store.state));
    } catch (err) {
      this.onError?.(err);
    } finally {
      this.busy = false;
    }
    if (this.source !== src) return;
    this.requestDraw();
    if (this.again) {
      const next = this.again;
      this.again = null;
      if (src.kind === 'image') this.ingest(false);
      else this.#videoFrame(next.continuous || !this.video.paused);
    }
  }

  requestDraw() {
    if (this.raf) return;
    this.raf = requestAnimationFrame((t) => {
      this.raf = 0;
      this.draw(t);
    });
  }

  #viewSpec(s, t) {
    const view = this.compare ? 'original' : s.view;
    const [eL, eR] = eyeFactors(s.eyes);
    switch (view) {
      case 'output':
        return { layoutId: s.layout, layout: LAYOUTS[s.layout]?.code ?? LAYOUT.SBS };
      case 'holes':
        return { layoutId: 'sbs-full', layout: LAYOUT.HOLES };
      case 'anaglyph':
        return { layoutId: 'mono', layout: LAYOUT.ANAGLYPH };
      case 'wiggle': {
        const phase = Math.floor((t - this.animStart) / 130) % 2;
        return { layoutId: 'mono', layout: LAYOUT.MONO_L, eyes: [{ e: phase ? eR : eL }], animate: true };
      }
      case 'look': {
        let eye;
        if (this.pointer) {
          const [px, py] = this.pointer;
          const vx = Math.max(-1.2, Math.min(1.2, px)) * 1.6;
          const vy = Math.max(-1.2, Math.min(1.2, py)) * 1.0;
          const len = Math.hypot(vx, vy);
          eye = len < 1e-4 ? { e: 0, dir: [1, 0] } : { e: len, dir: [vx / len, vy / len] };
          eye.zoom = motionEye('sway', 0.25, 1.9, s).zoom;
        } else {
          eye = motionEye('orbit', ((t - this.animStart) / 5000) % 1, 1.4, s);
        }
        return { layoutId: 'mono', layout: LAYOUT.MONO_L, eyes: [eye], animate: !this.pointer };
      }
      case 'depth':
        return { layoutId: 'mono', layout: LAYOUT.DEPTH };
      case 'parallax':
        return { layoutId: 'mono', layout: LAYOUT.PARALLAX };
      default:
        return { layoutId: 'mono', layout: LAYOUT.ORIGINAL };
    }
  }

  draw(t = performance.now()) {
    const src = this.source;
    if (!src || !src.w || !this.renderer.hasSource) {
      this.renderer.render({ canvasW: 2, canvasH: 2 });
      return;
    }
    const s = this.store.state;
    const spec = this.#viewSpec(s, t);
    const full = layoutGeometry(spec.layoutId, src.w, src.h);
    const box = this.viewer.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const pad = document.fullscreenElement === this.viewer ? 0 : 24;
    const bw = Math.max(64, box.width - pad);
    const bh = Math.max(64, box.height - pad);
    const cssScale = Math.min(bw / full.canvasW, bh / full.canvasH);
    const cssW = full.canvasW * cssScale;
    const cssH = full.canvasH * cssScale;
    const k = Math.min(1, (cssW * dpr) / full.canvasW, Math.sqrt(MAX_PREVIEW_PIXELS / (full.canvasW * full.canvasH)));
    const geo = layoutGeometry(spec.layoutId, src.w * k, src.h * k);
    this.canvas.style.width = `${Math.round(cssW)}px`;
    this.canvas.style.height = `${Math.round(cssH)}px`;
    const t0 = performance.now();
    this.renderer.render(this.fp.params(s, geo, { layout: spec.layout, eyes: spec.eyes }));
    this.renderMs = performance.now() - t0;
    this.geo = geo;
    this.onRendered?.(geo, spec);
    if (spec.animate) this.requestDraw();
  }
}
