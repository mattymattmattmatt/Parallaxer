const HIST_BINS = 64;
const RANGE_BINS = 1024;

// Gaussian gates are evaluated through lookup tables: ~1M Math.exp calls per frame at 924×518 otherwise
// dominate the stabiliser's cost. Differences are indexed by |d| in [0, 1] at 1/4096 resolution.
const LUT_STEPS = 4096;
const SIGMA_COLOUR = 0.045;
const SIGMA_DEPTH = 0.05;

function gaussLut(sigma) {
  const t = new Float32Array(LUT_STEPS + 1);
  const inv = 1 / (2 * sigma * sigma);
  for (let i = 0; i <= LUT_STEPS; i++) {
    const d = i / LUT_STEPS;
    t[i] = Math.exp(-d * d * inv);
  }
  return t;
}
const COLOUR_GATE = gaussLut(SIGMA_COLOUR);
const DEPTH_GATE = gaussLut(SIGMA_DEPTH);

/** exp(-d² / 2σ²) via the table; |d| ≥ 1 maps to the (≈0) last entry. */
function gate(lut, d) {
  const i = (d < 0 ? -d : d) * LUT_STEPS;
  return lut[i < LUT_STEPS ? i | 0 : LUT_STEPS];
}

const colMaps = new Map();

/**
 * Frame-independent half of the stabiliser: luma at depth resolution and the robust percentile range of the
 * raw network output. Depends on nothing but this frame, so parallel depth workers compute it off the main
 * thread; process() computes it itself when it isn't supplied.
 */
export function prepareFrame(raw, w, h, rgba, iw, ih, into = null) {
  const n = w * h;
  const luma = into && into.length === n ? into : new Float32Array(n);
  const sx = iw / w;
  const sy = ih / h;
  const key = `${w}:${iw}`;
  let cols = colMaps.get(key);
  if (!cols) {
    cols = new Int32Array(w);
    for (let x = 0; x < w; x++) cols[x] = Math.min(iw - 1, Math.floor((x + 0.5) * sx)) * 4;
    if (colMaps.size > 8) colMaps.clear();
    colMaps.set(key, cols);
  }
  const kr = 0.2126 / 255;
  const kg = 0.7152 / 255;
  const kb = 0.0722 / 255;
  let mn = Infinity;
  let mx = -Infinity;
  for (let y = 0; y < h; y++) {
    const ry = Math.min(ih - 1, Math.floor((y + 0.5) * sy)) * iw * 4;
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const j = ry + cols[x];
      luma[row + x] = rgba[j] * kr + rgba[j + 1] * kg + rgba[j + 2] * kb;
      const v = raw[row + x];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx)) {
    mn = 0;
    mx = 1;
  }
  const span = mx - mn || 1;
  const bins = new Uint32Array(RANGE_BINS);
  const k = (RANGE_BINS - 1) / span;
  for (let i = 0; i < n; i++) {
    const b = ((raw[i] - mn) * k) | 0;
    bins[b >= 0 && b < RANGE_BINS ? b : 0]++;
  }
  const pick = (q) => {
    const target = q * n;
    let acc = 0;
    for (let b = 0; b < RANGE_BINS; b++) {
      acc += bins[b];
      if (acc >= target) return mn + (b / (RANGE_BINS - 1)) * span;
    }
    return mx;
  };
  return { luma, pLo: pick(0.01), pHi: pick(0.995) };
}

/**
 * Turns raw, affine-invariant network output into a stable nearness map (1 = closest).
 *
 *  - Robust per-frame percentile normalisation, which cancels the network's per-frame scale/shift.
 *  - Scale/shift alignment to the previous frame, fitted on static pixels only, so global depth doesn't
 *    "breathe" when content changes.
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
    this.w = 0;
    this.h = 0;
    this.frames = 0;
    // Ping-pong buffers so a steady stream of same-sized frames allocates nothing per frame.
    this.spare = null;
    this.outRing = null;
    this.weights = null;
  }

  /**
   * @param {Float32Array} raw  network output (disparity-like), w*h
   * @param {Uint8Array} rgba   network input pixels (iw*ih*4), used for motion / cut detection
   * @param {{ temporal?: number, cutSensitivity?: number, continuous?: boolean }} opts
   */
  process(raw, w, h, rgba, iw, ih, opts = {}, prepared = null) {
    const temporal = opts.temporal ?? 0.5;
    const continuous = opts.continuous !== false;
    const n = w * h;

    const sizeChanged = w !== this.w || h !== this.h;
    let spare = this.spare;
    if (!spare || spare.luma.length !== n) spare = { luma: new Float32Array(n), state: new Float32Array(n) };
    this.spare = null;

    const pre = prepared ?? prepareFrame(raw, w, h, rgba, iw, ih, spare.luma);
    const luma = pre.luma;
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

    const pLo = pre.pLo;
    const invRange = 1 / Math.max(1e-6, pre.pHi - pLo);
    const out = spare.state;
    // Output buffers alternate, so the returned depth stays valid until the next-but-one call.
    this.outRing = this.outRing?.[0].length === n ? this.outRing : [new Float32Array(n), new Float32Array(n)];
    this.outRing.reverse();
    const depth = this.outRing[0];
    const prev = this.prev;
    const temporalPath = !cut && prev && temporal > 0;

    // 1) Per-frame robust normalisation. Depth networks output affine-invariant depth (arbitrary scale and
    //    shift every frame); normalising each frame by its own percentiles cancels that jitter exactly.
    //    On the temporal path the least-squares sums for step 2 are gathered in the same pass.
    let sw = 0;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    if (temporalPath) {
      const prevLuma = this.prevLuma;
      for (let i = 0; i < n; i++) {
        let v = (raw[i] - pLo) * invRange;
        v = v > 0 ? (v < 1 ? v : 1) : 0; // also maps NaN to 0
        out[i] = v;
        if ((i & 1) === 0) {
          const g = gate(COLOUR_GATE, luma[i] - prevLuma[i]);
          const y = prev[i];
          sw += g;
          sx += g * v;
          sy += g * y;
          sxx += g * v * v;
          sxy += g * v * y;
        }
      }
    } else {
      for (let i = 0; i < n; i++) {
        const v = (raw[i] - pLo) * invRange;
        out[i] = v > 0 ? (v < 1 ? v : 1) : 0;
      }
    }

    if (temporalPath) {
      const prevLuma = this.prevLuma;
      // 2) Align this frame to the previous stabilised frame with a scale/shift fitted by weighted least
      //    squares on static pixels (unchanged colour). Removes the global "breathing" that happens when
      //    content entering the frame changes the percentiles; a small leak lets any drift decay.
      let a = 1;
      let b = 0;
      const det = sw * sxx - sx * sx;
      if (sw > n * 0.05 && det > 1e-9) {
        let scale = (sw * sxy - sx * sy) / det;
        scale = Math.min(3, Math.max(0.33, scale));
        const shift = (sy - scale * sx) / sw;
        // Leak back toward per-frame normalisation so errors can't accumulate: ~0.3 s time constant at low
        // stabilisation, ~2 s at high.
        const leak = 0.02 + 0.3 * (1 - temporal) ** 2;
        const k = 1 - leak;
        a = 1 + (scale - 1) * k;
        b = shift * k;
      }

      // 3) Motion-gated per-pixel filter: static pixels are denoised, moving ones follow the new estimate.
      //    The temporal state stays unclamped (keeps the alignment linear); the renderer gets [0, 1].
      const keep = 0.9 * temporal;
      for (let i = 0; i < n; i++) {
        const v = out[i] * a + b;
        const p = prev[i];
        const wgt = keep * gate(DEPTH_GATE, v - p) * gate(COLOUR_GATE, luma[i] - prevLuma[i]);
        const f = v + wgt * (p - v);
        out[i] = f;
        depth[i] = f > 0 ? (f < 1 ? f : 1) : 0;
      }
    } else {
      depth.set(out);
    }

    // Statistics for the UI and automatic controls, sampled on every second pixel in each direction.
    const state = out;
    const hist = new Float32Array(HIST_BINS);
    let subjectAcc = 0;
    let subjectW = 0;
    if (!this.weights || this.weights.key !== `${w}x${h}`) {
      // Centre-weighted subject window, separable in x and y.
      const cx = (w - 1) / 2;
      const cy = (h - 1) * 0.45;
      const sig2x = 2 * (w * 0.22) ** 2;
      const sig2y = 2 * (h * 0.25) ** 2;
      const wx = new Float32Array(w);
      const wy = new Float32Array(h);
      for (let x = 0; x < w; x++) wx[x] = Math.exp(-((x - cx) ** 2) / sig2x);
      for (let y = 0; y < h; y++) wy[y] = Math.exp(-((y - cy) ** 2) / sig2y);
      this.weights = { key: `${w}x${h}`, wx, wy };
    }
    const { wx, wy } = this.weights;
    const border = Math.max(1, Math.round(w * 0.05));
    let borderL = 0;
    let borderR = 0;
    const step = n > 40000 ? 2 : 1;
    for (let y = 0; y < h; y += step) {
      const row = y * w;
      const ry = wy[y];
      for (let x = 0; x < w; x += step) {
        const v = depth[row + x];
        const hb = (v * HIST_BINS) | 0;
        hist[hb < HIST_BINS ? hb : HIST_BINS - 1]++;
        const wgt = ry * wx[x] * (0.35 + v);
        subjectAcc += v * wgt;
        subjectW += wgt;
      }
      for (let x = 0; x < border; x++) if (depth[row + x] > borderL) borderL = depth[row + x];
      for (let x = w - border; x < w; x++) if (depth[row + x] > borderR) borderR = depth[row + x];
    }
    let histMax = 0;
    for (let b = 0; b < HIST_BINS; b++) histMax = Math.max(histMax, hist[b]);
    for (let b = 0; b < HIST_BINS; b++) hist[b] /= histMax || 1;

    // Recycle the buffers that just became stale.
    if (this.prev && this.prev.length === n && this.prevLuma?.length === n) this.spare = { luma: this.prevLuma, state: this.prev };
    this.prev = state;
    this.prevLuma = luma;
    this.w = w;
    this.h = h;
    this.frames++;

    return {
      depth,
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
