import { h, clamp, timecode } from './dom.js';
import { curveJs } from '../core/settings.js';

function fitCanvas(canvas) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr));
  const hgt = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== hgt) {
    canvas.width = w;
    canvas.height = hgt;
  }
  return { w, h: hgt, dpr };
}

/**
 * Depth distribution in perceived (post-curve) space with a draggable screen plane.
 * Left = far, right = near. Blue = behind the screen, orange = in front.
 */
export class DepthHistogram {
  constructor(store) {
    this.store = store;
    this.hist = null;
    this.canvas = h('canvas', { class: 'histo-canvas', 'aria-label': 'Depth histogram. Drag to set the screen plane.' });
    this.el = h('div', { class: 'histo' }, this.canvas, h('div', { class: 'histo-axis' }, h('span', {}, 'Far'), h('span', {}, 'Screen plane'), h('span', {}, 'Near')));
    let dragging = false;
    const setFromEvent = (e) => {
      const r = this.canvas.getBoundingClientRect();
      const x = clamp((e.clientX - r.left) / r.width, 0.02, 0.98);
      this.store.set({ convergence: Math.round(x * 1000) / 1000, autoConvergence: false }, { undoable: false });
    };
    this.canvas.addEventListener('pointerdown', (e) => {
      dragging = true;
      this.canvas.setPointerCapture(e.pointerId);
      this.store.begin();
      setFromEvent(e);
    });
    this.canvas.addEventListener('pointermove', (e) => dragging && setFromEvent(e));
    const end = () => {
      if (!dragging) return;
      dragging = false;
      this.store.commit();
    };
    this.canvas.addEventListener('pointerup', end);
    this.canvas.addEventListener('pointercancel', end);
    new ResizeObserver(() => this.draw()).observe(this.canvas);
  }

  update(hist, convergence) {
    this.hist = hist;
    this.conv = convergence;
    this.draw();
  }

  draw() {
    const { w, h: H, dpr } = fitCanvas(this.canvas);
    const ctx = this.canvas.getContext('2d');
    ctx.clearRect(0, 0, w, H);
    const s = this.store.state;
    const conv = this.conv ?? s.convergence;
    const cx = conv * w;

    const bg = ctx.createLinearGradient(0, 0, w, 0);
    bg.addColorStop(0, 'rgba(64,140,255,0.10)');
    bg.addColorStop(Math.max(0, conv - 0.001), 'rgba(64,140,255,0.03)');
    bg.addColorStop(Math.min(1, conv + 0.001), 'rgba(255,120,60,0.03)');
    bg.addColorStop(1, 'rgba(255,120,60,0.12)');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, H);

    if (this.hist) {
      const bins = 96;
      const acc = new Float32Array(bins);
      const n = this.hist.length;
      for (let i = 0; i < n; i++) {
        const v = curveJs((i + 0.5) / n, s);
        acc[Math.min(bins - 1, Math.floor(v * bins))] += this.hist[i];
      }
      let mx = 0;
      for (const a of acc) mx = Math.max(mx, a);
      const bw = w / bins;
      for (let i = 0; i < bins; i++) {
        const v = acc[i] / (mx || 1);
        const bh = Math.pow(v, 0.6) * (H - 6 * dpr);
        const x = i * bw;
        const inFront = (i + 0.5) / bins > conv;
        ctx.fillStyle = inFront ? 'rgba(255,140,90,0.85)' : 'rgba(110,160,255,0.85)';
        ctx.fillRect(x + 0.5 * dpr, H - bh, Math.max(1, bw - 1 * dpr), bh);
      }
    }

    ctx.strokeStyle = '#f5e663';
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, H);
    ctx.stroke();
    ctx.fillStyle = '#f5e663';
    ctx.beginPath();
    ctx.moveTo(cx - 5 * dpr, 0);
    ctx.lineTo(cx + 5 * dpr, 0);
    ctx.lineTo(cx, 6 * dpr);
    ctx.fill();
  }
}

/**
 * Timeline with thumbnail strip, playhead, hover scrub preview and draggable in / out points.
 */
export class Timeline {
  constructor({ onSeek, onRange, onScrubEnd }) {
    this.duration = 0;
    this.time = 0;
    this.inPoint = null;
    this.outPoint = null;
    this.fps = 30;
    this.onSeek = onSeek;
    this.onRange = onRange;
    this.onScrubEnd = onScrubEnd;
    this.thumbs = [];

    this.strip = h('canvas', { class: 'tl-strip' });
    this.range = h('div', { class: 'tl-range' }, h('div', { class: 'tl-handle in', title: 'Drag to set In point (I)' }), h('div', { class: 'tl-handle out', title: 'Drag to set Out point (O)' }));
    this.head = h('div', { class: 'tl-head' }, h('div', { class: 'tl-head-cap' }));
    this.hover = h('div', { class: 'tl-hover' }, h('span', { class: 'tl-hover-tc' }));
    this.marks = h('div', { class: 'tl-marks' });
    this.el = h('div', { class: 'timeline', tabindex: '-1' }, this.strip, this.marks, this.range, this.hover, this.head);

    let mode = null;
    const tAt = (e) => {
      const r = this.el.getBoundingClientRect();
      return clamp((e.clientX - r.left) / r.width, 0, 1) * this.duration;
    };
    this.el.addEventListener('pointerdown', (e) => {
      if (!this.duration) return;
      this.el.setPointerCapture(e.pointerId);
      if (e.target.classList.contains('in')) mode = 'in';
      else if (e.target.classList.contains('out')) mode = 'out';
      else {
        mode = 'seek';
        this.onSeek?.(tAt(e), true);
      }
    });
    this.el.addEventListener('pointermove', (e) => {
      if (!this.duration) return;
      const t = tAt(e);
      const r = this.el.getBoundingClientRect();
      this.hover.style.transform = `translateX(${clamp(e.clientX - r.left, 0, r.width)}px)`;
      this.hover.firstChild.textContent = timecode(t, this.fps);
      if (mode === 'seek') this.onSeek?.(t, true);
      else if (mode === 'in') this.setRange(Math.min(t, (this.outPoint ?? this.duration) - 1 / this.fps), this.outPoint, true);
      else if (mode === 'out') this.setRange(this.inPoint, Math.max(t, (this.inPoint ?? 0) + 1 / this.fps), true);
    });
    const end = () => {
      if (mode === 'seek') this.onScrubEnd?.();
      mode = null;
    };
    this.el.addEventListener('pointerup', end);
    this.el.addEventListener('pointercancel', end);
    new ResizeObserver(() => this.#drawStrip()).observe(this.el);
  }

  setMedia(duration, fps) {
    this.duration = duration || 0;
    this.fps = fps || 30;
    this.inPoint = null;
    this.outPoint = null;
    this.thumbs = [];
    this.marks.replaceChildren();
    this.#layout();
    this.#drawStrip();
  }

  setThumbnails(list) {
    this.thumbs = list;
    this.#drawStrip();
  }

  setTime(t) {
    this.time = t;
    const p = this.duration ? t / this.duration : 0;
    this.head.style.left = `${p * 100}%`;
  }

  setRange(inPoint, outPoint, notify = false) {
    this.inPoint = inPoint;
    this.outPoint = outPoint;
    this.#layout();
    if (notify) this.onRange?.(this.inPoint, this.outPoint);
  }

  #layout() {
    const d = this.duration || 1;
    const a = (this.inPoint ?? 0) / d;
    const b = (this.outPoint ?? d) / d;
    this.range.style.left = `${a * 100}%`;
    this.range.style.width = `${Math.max(0, b - a) * 100}%`;
    this.range.classList.toggle('active', this.inPoint !== null || this.outPoint !== null);
    this.el.classList.toggle('has-media', !!this.duration);
  }

  #drawStrip() {
    const { w, h: H } = fitCanvas(this.strip);
    const ctx = this.strip.getContext('2d');
    ctx.fillStyle = '#0d0f13';
    ctx.fillRect(0, 0, w, H);
    if (!this.thumbs.length) return;
    const n = this.thumbs.length;
    const tw = w / n;
    for (let i = 0; i < n; i++) {
      const t = this.thumbs[i];
      if (!t) continue;
      const ar = t.width / t.height;
      const dw = Math.max(tw, H * ar);
      const sx = (dw - tw) / 2;
      ctx.save();
      ctx.beginPath();
      ctx.rect(i * tw, 0, tw, H);
      ctx.clip();
      ctx.drawImage(t, i * tw - sx, 0, dw, H);
      ctx.restore();
    }
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    for (let i = 1; i < n; i++) ctx.fillRect(Math.round(i * tw), 0, 1, H);
  }
}
