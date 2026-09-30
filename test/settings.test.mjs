import { test } from 'node:test';
import assert from 'node:assert/strict';

// These modules are DOM-free at import time, so they run under plain Node.
const { layoutGeometry, curveJs, DEFAULTS, PRESETS } = await import('../src/core/settings.js');
const { targetBitrate, baseName, outputName, motionEye } = await import('../src/media/export.js').catch(() => ({}));

test('layout geometry keeps parallax relative to the unsqueezed frame', () => {
  const full = layoutGeometry('sbs-full', 1920, 1080);
  assert.deepEqual([full.canvasW, full.canvasH, full.eyeW, full.logicalW], [3840, 1080, 1920, 1920]);
  const half = layoutGeometry('sbs-half', 1920, 1080);
  assert.deepEqual([half.canvasW, half.eyeW, half.logicalW], [1920, 960, 1920]);
  const tb = layoutGeometry('tb-half', 1920, 1080);
  assert.deepEqual([tb.canvasH, tb.eyeH, tb.logicalH], [1080, 540, 1080]);
  const odd = layoutGeometry('sbs-full', 1001, 563);
  assert.ok(odd.canvasW % 2 === 0 && odd.canvasH % 2 === 0, 'encoder-friendly even dimensions');
});

test('depth curve maps clip planes and gamma', () => {
  const s = { ...DEFAULTS, near: 0.8, far: 0.2, gamma: 1, invert: false };
  assert.equal(curveJs(0.1, s), 0);
  assert.equal(curveJs(0.9, s), 1);
  assert.ok(Math.abs(curveJs(0.5, s) - 0.5) < 1e-9);
  assert.ok(curveJs(0.5, { ...s, gamma: 2 }) < 0.5);
  assert.ok(Math.abs(curveJs(0.3, { ...s, invert: true }) - curveJs(0.7, s)) < 1e-9);
});

test('presets only reference known settings', () => {
  for (const p of PRESETS) for (const k of Object.keys(p.values)) assert.ok(k in DEFAULTS, `${p.id}.${k}`);
});

test('export helpers', { skip: !targetBitrate && 'export module needs browser globals' }, () => {
  const avc = targetBitrate({ level: 'high' }, 3840, 1080, 30, 'avc');
  const av1 = targetBitrate({ level: 'high' }, 3840, 1080, 30, 'av1');
  assert.ok(avc > 10e6 && avc < 25e6, `1080p full-SBS H.264 high ≈ 15 Mbps, got ${avc}`);
  assert.ok(av1 < avc, 'more efficient codecs get less bitrate');
  assert.equal(targetBitrate({ mode: 'bitrate', mbps: 12 }, 1, 1, 1, 'avc'), 12e6);
  assert.equal(outputName('My clip — final.mov', 'sbs-half', 'mp4'), 'My clip - final.3D.HSBS.mp4');
  assert.equal(baseName('a/b:c.mp4'), 'a_b_c');
  const e = motionEye('sway', 0.25, 1, DEFAULTS);
  assert.ok(e.e > 0.99 && e.zoom > 1, 'sway peaks at a quarter cycle and overscans');
});
