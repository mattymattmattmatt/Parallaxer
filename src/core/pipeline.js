import { LAYOUT } from '../gl/renderer.js';
import { DepthStabilizer, Smoothed } from '../depth/stabilizer.js';
import { LAYOUTS, curveJs } from './settings.js';

const STEP = { draft: 2.0, standard: 1.0, high: 0.5 };

export function eyeFactors(mode) {
  if (mode === 'left') return [0, 1];
  if (mode === 'right') return [-1, 0];
  return [-0.5, 0.5];
}

/**
 * Glue between a renderer, the depth engine and temporal state. One instance per output stream
 * (the live preview has one; every export job creates its own so they never share temporal state).
 */
export class FrameProcessor {
  constructor(renderer, engine) {
    this.renderer = renderer;
    this.engine = engine;
    this.stabilizer = new DepthStabilizer();
    this.autoConv = new Smoothed(0.06);
    this.winL = new Smoothed(0.12);
    this.winR = new Smoothed(0.12);
    this.stats = null;
  }

  reset() {
    this.stabilizer.reset();
    this.autoConv.reset();
    this.winL.reset();
    this.winR.reset();
  }

  /**
   * Stage a frame, estimate its depth and commit colour + depth together.
   * `continuous` = this frame directly follows the previous one (enables temporal filtering).
   */
  async ingest(source, w, h, s, { continuous = false } = {}) {
    const r = this.renderer;
    r.stage(source, w, h);
    const size = this.engine.inputSize(w, h, s.detail);
    const rgba = r.readModelInput(size.w, size.h);
    const res = await this.engine.infer(rgba, size.w, size.h);
    return this.finish({ ...res, rgba }, size, s, { continuous });
  }

  /**
   * Second half of ingest(), for depth computed elsewhere (e.g. a worker pool): stabilise, upload and commit.
   * The frame's colour must already be staged. `res` may carry prepared stabiliser inputs (luma, pLo, pHi).
   */
  finish(res, size, s, { continuous = false } = {}) {
    const r = this.renderer;
    const prepared = res.luma ? { luma: res.luma, pLo: res.pLo, pHi: res.pHi } : null;
    const t0 = performance.now();
    const st = this.stabilizer.process(
      res.data,
      res.w,
      res.h,
      res.rgba ?? null,
      size.w,
      size.h,
      { temporal: s.temporal, cutSensitivity: s.cutSensitivity, continuous },
      prepared
    );
    this.stabiliseMs = performance.now() - t0;
    r.uploadDepth(st.depth, st.w, st.h);
    r.commit();
    this.autoConv.push(st.subject, st.cut || !continuous);
    this.winL.push(st.borderL, st.cut || !continuous);
    this.winR.push(st.borderR, st.cut || !continuous);
    this.stats = { ...st, ms: res.ms, inputW: size.w, inputH: size.h };
    return this.stats;
  }

  convergence(s) {
    if (s.autoConvergence && this.autoConv.value !== null) {
      return Math.min(0.97, Math.max(0.03, curveJs(this.autoConv.value, s)));
    }
    return s.convergence;
  }

  /** Floating-window masks (uv units) that keep in-front objects from being cut by the frame edge. */
  floatingWindow(s, conv) {
    if (!s.floatingWindow || this.winL.value === null) return { left: 0, right: 0, feather: 0 };
    const S = s.strength / 100;
    const need = (near) => Math.max(0, -S * (conv - curveJs(near, s)));
    const left = need(this.winL.value);
    const right = need(this.winR.value);
    const pad = (v) => (v > 0.0005 ? v + 0.002 : 0);
    return { left: pad(left), right: pad(right), feather: 0.004 };
  }

  /**
   * Build renderer parameters.
   * @param s        settings
   * @param geo      layoutGeometry() result (canvas / eye / logical sizes)
   * @param view     { layout: code, eyes?: [...], swap?: bool }
   */
  params(s, geo, view) {
    const conv = this.convergence(s);
    const [eL, eR] = eyeFactors(s.eyes);
    const eyes = view.eyes ?? [{ e: eL }, { e: eR }];
    const isStereoLayout = [LAYOUT.SBS, LAYOUT.TB, LAYOUT.ANAGLYPH, LAYOUT.ROWS, LAYOUT.COLUMNS, LAYOUT.CHECKER].includes(view.layout);
    return {
      ...geo,
      layout: view.layout,
      swap: view.swap ?? s.swap,
      anaglyph: s.anaglyph,
      colormap: s.colormap === 'turbo' ? 1 : s.colormap === 'magma' ? 2 : 0,
      eyes,
      stereo: {
        strength: s.strength / 100,
        convergence: conv,
        curve: [s.near, s.far, s.gamma, s.invert ? 1 : 0],
        step: STEP[s.quality] ?? 1,
        maxStretch: s.tear,
        fill: s.fill,
        soften: s.soften
      },
      refine: {
        edge: s.edgeSnap,
        sigmaC: s.edgeSigma,
        snap: s.snapRadius / 1000,
        snapIterations: 2,
        dilate: s.dilate / 1000,
        blur: s.smooth / 1000,
        preserve: s.preserve
      },
      window: isStereoLayout ? this.floatingWindow(s, conv) : null,
      comfort: comfortLimitPercent(s)
    };
  }

  /** Render the export layout for the given settings. */
  renderOutput(s, geo) {
    const layout = LAYOUTS[s.layout] ?? LAYOUTS['sbs-full'];
    this.renderer.render(this.params(s, geo, { layout: layout.code }));
  }
}

/**
 * Largest comfortable positive parallax as a percentage of screen width: on-screen separation must not
 * exceed the human interocular distance (~63 mm) or the eyes are forced to diverge.
 */
export function comfortLimitPercent(s) {
  const widthMm = s.screenInches * 25.4 * (16 / Math.hypot(16, 9));
  return (63 / widthMm) * 100;
}

export function screenWidthMm(s) {
  return s.screenInches * 25.4 * (16 / Math.hypot(16, 9));
}
