import test from 'node:test';
import assert from 'node:assert/strict';

// Deterministic clock for the profiler.
let clock = 0;
Object.defineProperty(globalThis, 'performance', { value: { now: () => clock }, configurable: true, writable: true });
const { ExportProfiler, bottleneck, statsReport } = await import('../src/media/perf.js');

const settle = () => new Promise((r) => setImmediate(r));

/** Simulate one frame: decode wait, prepare, depth wait, stabilise, render, encode wait, then idle. */
async function frame(p, ms) {
  clock += ms.decode ?? 0;
  p.add('decode', ms.decode ?? 0);
  clock += ms.prepare ?? 0;
  p.add('prepare', ms.prepare ?? 0);
  let release;
  const waiting = p.time('depth', new Promise((r) => (release = r)));
  clock += ms.depth ?? 0;
  if (ms.pageDepth) p.pageDepthBlock(ms.pageDepth);
  release();
  await waiting;
  clock += ms.stabilise ?? 0;
  p.add('stabilise', ms.stabilise ?? 0);
  clock += ms.render ?? 0;
  p.add('render', ms.render ?? 0);
  const enc = p.time('encode', settle());
  clock += ms.encode ?? 0;
  await enc;
  clock += ms.idle ?? 0;
  p.frameDone();
}

test('stage times add up to wall time, pauses excluded', async () => {
  clock = 1000;
  const p = new ExportProfiler();
  p.start();
  for (let i = 0; i < 10; i++) await frame(p, { decode: 1, prepare: 2, depth: 5, stabilise: 3, render: 20, encode: 4, idle: 5 });
  clock += 500;
  p.paused(500);
  const s = p.snapshot({ recent: false });
  assert.equal(s.frames, 10);
  assert.equal(Math.round(s.msPerFrame), 40);
  const by = Object.fromEntries(s.stages.map((x) => [x.key, x.ms]));
  assert.deepEqual([by.decode, by.prepare, by.depth, by.stabilise, by.render, by.encode, by.other].map(Math.round), [1, 2, 5, 3, 20, 4, 5]);
  assert.ok(Math.abs(s.stages.reduce((a, x) => a + x.share, 0) - 1) < 1e-9);
  assert.equal(bottleneck(s).key, 'render');
});

test('depth computed on the page thread is moved out of the wait it interrupted', async () => {
  clock = 0;
  const p = new ExportProfiler();
  p.start();
  // 30 ms depth "wait", of which 25 ms was really the page computing depth; another 40 ms blocked during idle.
  for (let i = 0; i < 5; i++) await frame(p, { prepare: 2, depth: 30, pageDepth: 25, render: 10, idle: 40 });
  for (let i = 0; i < 5; i++) {
    clock += 40;
    p.pageDepthBlock(40);
    p.frameDone();
  }
  const s = p.snapshot({ recent: false });
  const depth = s.stages.find((x) => x.key === 'depth');
  assert.equal(depth.wait, false, 'shown as computing, not waiting');
  assert.ok(Math.abs(s.stages.reduce((a, x) => a + x.ms, 0) - s.msPerFrame) < 1e-6, 'still sums to wall time');
  assert.equal(bottleneck(s).key, 'depth');
  assert.match(bottleneck(s).text, /background workers/);
});

test('encoder verdict names software encoding and the report is complete', async () => {
  clock = 0;
  const p = new ExportProfiler();
  Object.assign(p.info, { codecName: 'VP9', encoderHw: false, sourceCodec: 'H.264', decoderHw: true, output: '3840×1080' });
  p.start();
  for (let i = 0; i < 8; i++) await frame(p, { prepare: 1, depth: 2, render: 5, encode: 30 });
  const s = p.snapshot();
  const v = bottleneck(s);
  assert.equal(v.key, 'encode');
  assert.match(v.text, /encoded on the CPU/);
  const report = statsReport(s, v);
  for (const needle of ['Output: 3840×1080 VP9 (encode CPU)', 'Encode', 'Bottleneck: Video encoder']) assert.ok(report.includes(needle), needle);
});

test('no single dominant stage reads as balanced', async () => {
  clock = 0;
  const p = new ExportProfiler();
  p.start();
  for (let i = 0; i < 6; i++) await frame(p, { decode: 5, prepare: 5, depth: 6, stabilise: 5, render: 6, encode: 5, idle: 5 });
  assert.equal(bottleneck(p.snapshot({ recent: false })).key, 'balanced');
});
