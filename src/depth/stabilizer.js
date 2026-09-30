const HIST_BINS = 64;
const RANGE_BINS = 1024;

/**
 * Turns raw, affine-invariant network output into a stable nearness map (1 = closest).
 *
 *  - Robust percentile normalisation (ignores specular outliers / sky noise).
 *  - Temporal smoothing of the normalisation range, so global depth doesn't "breathe".
 *  - Motion-gated per-pixel temporal filter: static regions are denoised, moving regions follow the new
 *    estimate immediately (gated on both depth change and colour change).
 *  - Scene-cut detection resets all temporal state.
 *  - Exposes statistics for the UI (histogram, subject depth, border nearness for floating windows).
 */
export class DepthStabilizer {
  constructor() {
    this.reset();
  }

  reset() {
    this.prev = null;
    this.prevLuma = null;
    this.lo = null;
    this.hi = null;
    this.w = 0;
    this.h = 0;
    this.frames = 0;
  }

  /**
   * @param {Float32Array} raw  network output (disparity-like), w*h
   * @param {Uint8Array} rgba   network input pixels (iw*ih*4), used for motion / cut detection
   * @param {{ temporal?: number, cutSensitivity?: number, continuous?: boolean }} opts
   */
  process(raw, w, h, rgba, iw, ih, opts = {}) {
    const temporal = opts.temporal ?? 0.5;
    const continuous = opts.continuous !== false;
    const n = w * h;

    // Luma at depth resolution (nearest sample from the network input).
    const luma = new Float32Array(n);
    const sx = iw / w;
    const sy = ih / h;
    for (let y = 0; y < h; y++) {
      const ry = Math.min(ih - 1, Math.floor((y + 0.5) * sy)) * iw;
      for (let x = 0; x < w; x++) {
        const j = (ry + Math.min(iw - 1, Math.floor((x + 0.5) * sx))) * 4;
        luma[y * w + x] = (rgba[j] * 0.2126 + rgba[j + 1] * 0.7152 + rgba[j + 2] * 0.0722) / 255;
      }
    }

    const sizeChanged = w !== this.w || h !== this.h;
    let cut = !continuous || sizeChanged || !this.prev;
    let motion = 0;
    if (!cut && this.prevLuma) {
      let acc = 0;
      let cnt = 0;
      for (let i = 0; i < n; i += 3) {
        acc += Math.abs(luma[i] - this.prevLuma[i]);
        cnt++;
      }
      motion = acc / Math.max(1, cnt);
      const sens = opts.cutSensitivity ?? 0.5;
      const threshold = 0.32 - 0.24 * sens;
      if (motion > threshold) cut = true;
    }

    // Robust range via histogram percentiles.
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = raw[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (!Number.isFinite(mn) || !Number.isFinite(mx)) {
      mn = 0;
      mx = 1;
    }
    const span = mx - mn || 1;
    const bins = new Uint32Array(RANGE_BINS);
    const k = (RANGE_BINS - 1) / span;
    for (let i = 0; i < n; i++) bins[((raw[i] - mn) * k) | 0]++;
    const pick = (q) => {
      const target = q * n;
      let acc = 0;
      for (let b = 0; b < RANGE_BINS; b++) {
        acc += bins[b];
        if (acc >= target) return mn + (b / (RANGE_BINS - 1)) * span;
      }
      return mx;
    };
    const pLo = pick(0.01);
    const pHi = pick(0.995);

    if (cut || this.lo === null) {
      this.lo = pLo;
      this.hi = pHi;
    } else {
      const a = 1 - 0.92 * temporal;
      this.lo += (pLo - this.lo) * a;
      this.hi += (pHi - this.hi) * a;
    }
    const lo = this.lo;
    const range = Math.max(1e-6, this.hi - lo);

    const out = new Float32Array(n);
    const prev = this.prev;
    if (!cut && prev && temporal > 0) {
      const keep = 0.9 * temporal;
      const invD = 1 / (2 * 0.05 * 0.05);
      const invC = 1 / (2 * 0.045 * 0.045);
      const prevLuma = this.prevLuma;
      for (let i = 0; i < n; i++) {
        let v = (raw[i] - lo) / range;
        v = v < 0 ? 0 : v > 1 ? 1 : v;
        const dd = v - prev[i];
        const dc = luma[i] - prevLuma[i];
        const wgt = keep * Math.exp(-dd * dd * invD - dc * dc * invC);
        out[i] = v + wgt * (prev[i] - v);
      }
    } else {
      for (let i = 0; i < n; i++) {
        const v = (raw[i] - lo) / range;
        out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
    }

    // Statistics for the UI and automatic controls.
    const hist = new Float32Array(HIST_BINS);
    let subjectAcc = 0;
    let subjectW = 0;
    const cx = (w - 1) / 2;
    const cy = (h - 1) * 0.45;
    const sig2x = 2 * (w * 0.22) ** 2;
    const sig2y = 2 * (h * 0.25) ** 2;
    const border = Math.max(1, Math.round(w * 0.05));
    let borderL = 0;
    let borderR = 0;
    for (let y = 0; y < h; y++) {
      const wy = Math.exp(-((y - cy) ** 2) / sig2y);
      for (let x = 0; x < w; x++) {
        const v = out[y * w + x];
        hist[Math.min(HIST_BINS - 1, (v * HIST_BINS) | 0)]++;
        const wgt = wy * Math.exp(-((x - cx) ** 2) / sig2x) * (0.35 + v);
        subjectAcc += v * wgt;
        subjectW += wgt;
        if (x < border && v > borderL) borderL = v;
        if (x >= w - border && v > borderR) borderR = v;
      }
    }
    let histMax = 0;
    for (let b = 0; b < HIST_BINS; b++) histMax = Math.max(histMax, hist[b]);
    for (let b = 0; b < HIST_BINS; b++) hist[b] /= histMax || 1;

    this.prev = out;
    this.prevLuma = luma;
    this.w = w;
    this.h = h;
    this.frames++;

    return {
      depth: out,
      w,
      h,
      cut: cut && this.frames > 1,
      motion,
      hist,
      subject: subjectAcc / Math.max(1e-6, subjectW),
      borderL,
      borderR
    };
  }
}

/** Temporal smoother for scalar controls (auto convergence, floating window). */
export class Smoothed {
  constructor(rate = 0.15) {
    this.rate = rate;
    this.value = null;
  }
  push(v, reset = false) {
    if (this.value === null || reset) this.value = v;
    else this.value += (v - this.value) * this.rate;
    return this.value;
  }
  reset() {
    this.value = null;
  }
}
