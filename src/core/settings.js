import { LAYOUT } from '../gl/renderer.js';

export const LAYOUTS = {
  'sbs-full': { label: 'Side-by-side · Full', short: 'Full SBS', code: LAYOUT.SBS, tag: '3D.SBS', hint: 'VR headsets, 3D players, YouTube VR' },
  'sbs-half': { label: 'Side-by-side · Half', short: 'Half SBS', code: LAYOUT.SBS, tag: '3D.HSBS', hint: '3D TVs and projectors (frame-compatible)' },
  'tb-full': { label: 'Top-bottom · Full', short: 'Full TB', code: LAYOUT.TB, tag: '3D.TAB', hint: 'Over-under players' },
  'tb-half': { label: 'Top-bottom · Half', short: 'Half TB', code: LAYOUT.TB, tag: '3D.HTAB', hint: '3D TVs (frame-compatible)' },
  anaglyph: { label: 'Anaglyph', short: 'Anaglyph', code: LAYOUT.ANAGLYPH, tag: 'anaglyph', hint: 'Coloured glasses, any screen' },
  rows: { label: 'Row interleaved', short: 'Rows', code: LAYOUT.ROWS, tag: '3D.interleaved', hint: 'Passive polarised monitors' },
  columns: { label: 'Column interleaved', short: 'Columns', code: LAYOUT.COLUMNS, tag: '3D.columns', hint: 'Lenticular / autostereo panels' },
  checker: { label: 'Checkerboard', short: 'Checker', code: LAYOUT.CHECKER, tag: '3D.checker', hint: 'DLP 3D projectors' },
  rgbd: { label: '2D + Depth (RGB-D)', short: 'RGB-D', code: LAYOUT.RGBD, tag: 'RGBD', hint: 'Looking Glass, Leia, depth players' },
  depth: { label: 'Depth map', short: 'Depth', code: LAYOUT.DEPTH, tag: 'depth', hint: 'Grayscale depth for compositing' }
};

/** Output geometry for a layout given the per-eye source size (scaled). All dims are even. */
export function layoutGeometry(layoutId, w, h) {
  const even = (v) => Math.max(2, Math.round(v / 2) * 2);
  w = even(w);
  h = even(h);
  switch (layoutId) {
    case 'sbs-full':
      return { canvasW: w * 2, canvasH: h, eyeW: w, eyeH: h, logicalW: w, logicalH: h };
    case 'sbs-half':
      return { canvasW: w, canvasH: h, eyeW: w / 2, eyeH: h, logicalW: w, logicalH: h };
    case 'tb-full':
      return { canvasW: w, canvasH: h * 2, eyeW: w, eyeH: h, logicalW: w, logicalH: h };
    case 'tb-half':
      return { canvasW: w, canvasH: h, eyeW: w, eyeH: h / 2, logicalW: w, logicalH: h };
    case 'rgbd':
      return { canvasW: w * 2, canvasH: h, eyeW: w, eyeH: h, logicalW: w, logicalH: h };
    default:
      return { canvasW: w, canvasH: h, eyeW: w, eyeH: h, logicalW: w, logicalH: h };
  }
}

export const VIEWS = [
  { id: 'output', label: 'Output', key: '1', hint: 'Exactly what will be exported' },
  { id: 'anaglyph', label: 'Anaglyph', key: '2', hint: 'View in 3D with red/cyan glasses' },
  { id: 'wiggle', label: 'Wiggle', key: '3', hint: 'Alternating eyes — see depth without glasses' },
  { id: 'look', label: 'Look-around', key: '4', hint: 'Move the pointer to look around the scene' },
  { id: 'depth', label: 'Depth', key: '5', hint: 'Refined depth map' },
  { id: 'parallax', label: 'Parallax', key: '6', hint: 'Blue = behind screen, orange = in front, magenta = beyond comfort' },
  { id: 'holes', label: 'Occlusion', key: '7', hint: 'Disoccluded pixels synthesised by the hole filler' },
  { id: 'original', label: 'Original', key: '8', hint: 'Unprocessed source' }
];

export const DEFAULTS = {
  model: 'midas-small',
  detail: 'high',
  backend: 'auto',

  strength: 2.4,
  convergence: 0.55,
  autoConvergence: false,
  eyes: 'symmetric',
  gamma: 1.0,
  near: 1.0,
  far: 0.0,
  invert: false,

  edgeSnap: 0.85,
  edgeSigma: 0.08,
  snapRadius: 6,
  dilate: 1.2,
  smooth: 1.2,
  preserve: 0.4,

  fill: 'mirror',
  tear: 3.0,
  soften: 0.6,
  quality: 'high',

  temporal: 0.55,
  cutSensitivity: 0.5,

  layout: 'sbs-full',
  anaglyph: 'dubois-rc',
  swap: false,
  floatingWindow: true,

  screenInches: 65,
  colormap: 'turbo',
  view: 'output'
};

export const PRESETS = [
  { id: 'gentle', label: 'Gentle', hint: 'Subtle, all-day comfortable depth', values: { strength: 1.3, convergence: 0.6, gamma: 1.0, dilate: 1.0, smooth: 1.6 } },
  { id: 'standard', label: 'Standard', hint: 'Balanced depth for most footage', values: { strength: 2.4, convergence: 0.55, gamma: 1.0, dilate: 1.2, smooth: 1.2 } },
  { id: 'cinema', label: 'Cinema', hint: 'Mostly behind the screen, like a theatrical 3D grade', values: { strength: 2.2, convergence: 0.85, gamma: 0.9, dilate: 1.2, smooth: 1.4 } },
  { id: 'popout', label: 'Pop-out', hint: 'Subjects float in front of the screen', values: { strength: 3.2, convergence: 0.3, gamma: 1.15, dilate: 1.6, smooth: 1.2 } },
  { id: 'vr', label: 'VR', hint: 'Deep, immersive parallax for headsets', values: { strength: 4.0, convergence: 0.5, gamma: 1.0, dilate: 1.5, smooth: 1.0 } },
  { id: 'extreme', label: 'Extreme', hint: 'Maximum effect — expect artefacts', values: { strength: 6.0, convergence: 0.45, gamma: 1.2, dilate: 2.0, smooth: 1.8 } }
];

const STORAGE_KEY = 'parallaxer.settings.v2';

export function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    /* storage unavailable */
  }
  return { ...DEFAULTS };
}

export function saveSettings(s) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    /* storage unavailable */
  }
}

export function curveJs(d, s) {
  if (s.invert) d = 1 - d;
  d = Math.min(1, Math.max(0, (d - s.far) / Math.max(s.near - s.far, 1e-4)));
  return Math.pow(d, s.gamma);
}
