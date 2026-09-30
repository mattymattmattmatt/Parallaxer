import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DepthStabilizer } from '../src/depth/stabilizer.js';

const W = 128;
const H = 96;
const N = W * H;

function rng(seed) {
  return () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
}

/** Static ramp scene with a near box moving 2 px/frame; network output has per-frame affine jitter + noise. */
function makeFlickerScene(seed = 7) {
  const r = rng(seed);
  return (t) => {
    const truth = new Float32Array(N);
    const rgba = new Uint8Array(N * 4);
    const bx = 20 + 2 * t;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const inBox = x >= bx && x < bx + 20 && y >= 30 && y < 60;
        truth[i] = inBox ? 0.9 : 0.1 + 0.5 * (y / H);
        const c = inBox ? 220 : 60 + (y / H) * 80;
        rgba.fill(c, i * 4, i * 4 + 3);
        rgba[i * 4 + 3] = 255;
      }
    }
    const a = 3 + (r() - 0.5) * 0.9;
    const b = 5 + (r() - 0.5) * 1.2;
    const raw = new Float32Array(N);
    for (let i = 0; i < N; i++) raw[i] = a * truth[i] + b + (r() - 0.5) * 0.12;
    return { raw, rgba, bx };
  };
}

function runFlicker(temporal) {
  const scene = makeFlickerScene();
  const st = new DepthStabilizer();
  let prev = null;
  let flicker = 0;
  let count = 0;
  let boxErr = 0;
  const frames = 40;
  for (let t = 0; t < frames; t++) {
    const f = scene(t);
    const out = st.process(f.raw, W, H, f.rgba, W, H, { temporal, continuous: t > 0 }).depth;
    if (prev) {
      for (let i = 0; i < 20 * W; i++) {
        flicker += Math.abs(out[i] - prev[i]);
        count++;
      }
    }
    let s = 0;
    let n = 0;
    for (let y = 30; y < 60; y++) for (let x = f.bx; x < f.bx + 20; x++, n++) s += out[y * W + x];
    boxErr += Math.abs(s / n - 1);
    prev = out;
  }
  return { flicker: flicker / count, lag: boxErr / frames };
}

test('temporal stabilisation reduces background flicker without smearing motion', () => {
  const off = runFlicker(0);
  const on = runFlicker(0.55);
  const strong = runFlicker(0.9);
  assert.ok(on.flicker < off.flicker * 0.6, `default should cut flicker (off ${off.flicker}, on ${on.flicker})`);
  assert.ok(strong.flicker < off.flicker * 0.3, `strong should cut flicker further (${strong.flicker})`);
  assert.ok(on.lag < 0.06 && strong.lag < 0.06, `moving object must stay near (lag ${on.lag}, ${strong.lag})`);
});

test('scale/shift alignment prevents global depth jumps when an object enters', () => {
  const run = (temporal) => {
    const st = new DepthStabilizer();
    let prev = null;
    let worst = 0;
    for (let t = 0; t < 30; t++) {
      const truth = new Float32Array(N);
      const rgba = new Uint8Array(N * 4);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = y * W + x;
          const obj = t >= 20 && x < 40 && y > 60;
          truth[i] = obj ? 1.6 : 0.1 + 0.5 * (y / H);
          rgba.fill(obj ? 250 : 60 + (y / H) * 80, i * 4, i * 4 + 3);
          rgba[i * 4 + 3] = 255;
        }
      }
      const out = st.process(truth, W, H, rgba, W, H, { temporal, continuous: t > 0 }).depth;
      if (prev && t >= 20 && t < 25) {
        let d = 0;
        let c = 0;
        for (let y = 0; y < 20; y++) for (let x = 60; x < W; x++, c++) d += Math.abs(out[y * W + x] - prev[y * W + x]);
        worst = Math.max(worst, d / c);
      }
      prev = out;
    }
    return worst;
  };
  const off = run(0);
  const on = run(0.55);
  assert.ok(off > 0.03, 'per-frame normalisation alone jumps when the depth distribution changes');
  assert.ok(on < off / 5, `stabilised output should stay steady (off ${off}, on ${on})`);
});

test('scene cuts reset temporal state', () => {
  const st = new DepthStabilizer();
  const rgbaA = new Uint8Array(N * 4).fill(30);
  const rgbaB = new Uint8Array(N * 4).fill(230);
  const rawA = new Float32Array(N).map((_, i) => (i % W) / W);
  const rawB = new Float32Array(N).map((_, i) => 1 - (i % W) / W);
  st.process(rawA, W, H, rgbaA, W, H, { temporal: 0.9, continuous: false });
  const r = st.process(rawB, W, H, rgbaB, W, H, { temporal: 0.9, continuous: true });
  assert.equal(r.cut, true);
  assert.ok(r.depth[0] > 0.95 && r.depth[W - 1] < 0.05, 'after a cut the new frame is used as-is');
});

test('non-finite network output never reaches the renderer', () => {
  const st = new DepthStabilizer();
  const raw = new Float32Array(N).map((_, i) => (i % 97 === 0 ? NaN : i / N));
  const rgba = new Uint8Array(N * 4).fill(100);
  const { depth, hist } = st.process(raw, W, H, rgba, W, H, { temporal: 0.5 });
  assert.ok(depth.every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
  assert.ok(hist.every(Number.isFinite));
});
