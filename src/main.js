import './style.css';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);

const fileEl = $('file');
const fileLabel = $('fileLabel');
const loadBtn = $('loadBtn');
const runBtn = $('runBtn');
const fpsEl = $('fps');
const maxWEl = $('maxW');
const strengthEl = $('strength');
const maxFramesEl = $('maxFrames');

const barEl = $('bar');
const statusEl = $('status');
const logEl = $('log');

const downloadEl = $('download');
const previewEl = $('preview');

// Vite base URL (handles GitHub Pages subpaths)
const BASE = import.meta.env.BASE_URL;

// ---------- Helpers ----------
function log(line) {
  logEl.textContent += line + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function setProgress(p01) {
  const clamped = Math.max(0, Math.min(1, p01));
  barEl.style.width = `${Math.round(clamped * 100)}%`;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

async function blobToU8(blob) {
  return new Uint8Array(await blob.arrayBuffer());
}

// ---------- FFmpeg (WASM) ----------
const ffmpeg = new FFmpeg();
let ffmpegLoaded = false;

async function loadFFmpeg() {
  if (ffmpegLoaded) return;

  await ffmpeg.load({
    coreURL: `${BASE}ffmpeg/ffmpeg-core.js`,
    wasmURL: `${BASE}ffmpeg/ffmpeg-core.wasm`,
    workerURL: `${BASE}ffmpeg/ffmpeg-core.worker.js`
  });

  ffmpeg.on('log', ({ message }) => log(`[ffmpeg] ${message}`));
  ffmpeg.on('progress', ({ progress }) => setProgress(progress));

  ffmpegLoaded = true;
  log('FFmpeg loaded.');
}

// ---------- ONNX Runtime (MiDaS) ----------
let ort = null;
let session = null;
let ortBackend = 'wasm';

async function loadORT() {
  try {
    ort = await import('onnxruntime-web/webgpu');
    ortBackend = 'webgpu';
  } catch {
    ort = await import('onnxruntime-web');
    ortBackend = 'wasm';
  }

  const modelUrl = `${BASE}models/midas-small.onnx`;
  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ortBackend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
    graphOptimizationLevel: 'all'
  });

  log(`ONNX Runtime loaded (${ortBackend}).`);
  log(`Model inputs: ${session.inputNames.join(', ')}`);
  log(`Model outputs: ${session.outputNames.join(', ')}`);
}

const NET_W = 256;
const NET_H = 256;

function imageDataToMidasTensor(imageData) {
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = imageData.width;
  srcCanvas.height = imageData.height;
  const sctx = srcCanvas.getContext('2d', { willReadFrequently: true });
  sctx.putImageData(imageData, 0, 0);

  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = NET_W;
  dstCanvas.height = NET_H;
  const dctx = dstCanvas.getContext('2d', { willReadFrequently: true });
  dctx.drawImage(srcCanvas, 0, 0, NET_W, NET_H);
  const { data } = dctx.getImageData(0, 0, NET_W, NET_H);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  const out = new Float32Array(1 * 3 * NET_W * NET_H);
  const planeSize = NET_W * NET_H;

  for (let y = 0; y < NET_H; y++) {
    for (let x = 0; x < NET_W; x++) {
      const i = (y * NET_W + x) * 4;
      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;

      const idx = y * NET_W + x;
      out[idx] = (r - mean[0]) / std[0];
      out[planeSize + idx] = (g - mean[1]) / std[1];
      out[2 * planeSize + idx] = (b - mean[2]) / std[2];
    }
  }

  return new ort.Tensor('float32', out, [1, 3, NET_H, NET_W]);
}

function normalizeDepth(depthArr) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < depthArr.length; i++) {
    const v = depthArr[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const out = new Float32Array(depthArr.length);
  for (let i = 0; i < depthArr.length; i++) out[i] = (depthArr[i] - min) / range;
  return out;
}

function blurDepth(depth, w, h) {
  const out = new Float32Array(depth.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx >= 0 && xx < w && yy >= 0 && yy < h) {
            sum += depth[yy * w + xx];
            count++;
          }
        }
      }
      out[y * w + x] = sum / count;
    }
  }
  return out;
}

function synthesizeRightEye(leftImageData, depthNorm, strengthPx) {
  const w = leftImageData.width;
  const h = leftImageData.height;
  const src = leftImageData.data;

  const out = new ImageData(w, h);
  const dst = out.data;

  for (let y = 0; y < h; y++) {
    const v = y / (h - 1);
    const dy = Math.floor(v * (NET_H - 1));
    for (let x = 0; x < w; x++) {
      const u = x / (w - 1);
      const dx = Math.floor(u * (NET_W - 1));
      const d = depthNorm[dy * NET_W + dx];

      const disp = (1 - d) * strengthPx;
      const sx = Math.round(clamp(x + disp, 0, w - 1));

      const si = (y * w + sx) * 4;
      const di = (y * w + x) * 4;

      dst[di] = src[si];
      dst[di + 1] = src[si + 1];
      dst[di + 2] = src[si + 2];
      dst[di + 3] = 255;
    }
  }
  return out;
}

async function imageDataToPNGBytes(imgData) {
  const c = document.createElement('canvas');
  c.width = imgData.width;
  c.height = imgData.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.putImageData(imgData, 0, 0);
  const blob = await new Promise((res) => c.toBlob(res, 'image/png'));
  return blobToU8(blob);
}

// ---------- Pipeline ----------
async function generateSBS(file) {
  downloadEl.style.display = 'none';
  previewEl.style.display = 'none';
  previewEl.removeAttribute('src');

  const fps = clamp(parseInt(fpsEl.value, 10) || 8, 1, 30);
  const maxW = clamp(parseInt(maxWEl.value, 10) || 512, 240, 1920);
  const strength = clamp(parseInt(strengthEl.value, 10) || 12, 1, 40);
  const maxFrames = clamp(parseInt(maxFramesEl.value, 10) || 120, 1, 9999);

  setProgress(0);
  setStatus('Writing input…');
  log(`Parallaxer settings: fps=${fps}, maxW=${maxW}, strength=${strength}px, maxFrames=${maxFrames}`);

  try { await ffmpeg.deleteFile('/input.mp4'); } catch {}
  for (const dir of ['/frames', '/L', '/R']) {
    try {
      const nodes = await ffmpeg.listDir(dir);
      for (const n of nodes) {
        if (n.name !== '.' && n.name !== '..') {
          try { await ffmpeg.deleteFile(`${dir}/${n.name}`); } catch {}
        }
      }
      try { await ffmpeg.deleteDir(dir); } catch {}
    } catch {}
  }

  await ffmpeg.createDir('/frames');
  await ffmpeg.createDir('/L');
  await ffmpeg.createDir('/R');

  await ffmpeg.writeFile('/input.mp4', await fetchFile(file));

  setStatus('Extracting frames…');
  setProgress(0.02);

  const vf = `fps=${fps},scale=${maxW}:-2`;
  const rc1 = await ffmpeg.exec(['-i', '/input.mp4', '-vf', vf, '-vcodec', 'png', '/frames/%05d.png']);
  if (rc1 !== 0) throw new Error('FFmpeg frame extraction failed.');

  const nodes = await ffmpeg.listDir('/frames');
  const frameNames = nodes.filter(n => n.name.endsWith('.png')).map(n => n.name).sort();
  const framesToProcess = frameNames.slice(0, maxFrames);

  log(`Frames extracted: ${frameNames.length}. Processing: ${framesToProcess.length}.`);
  setStatus('Depth + right-eye synthesis…');

  for (let i = 0; i < framesToProcess.length; i++) {
    const name = framesToProcess[i];
    const bytes = await ffmpeg.readFile(`/frames/${name}`);

    const blob = new Blob([bytes], { type: 'image/png' });
    const bmp = await createImageBitmap(blob);

    const c = document.createElement('canvas');
    c.width = bmp.width;
    c.height = bmp.height;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);

    const left = ctx.getImageData(0, 0, c.width, c.height);

    const inputTensor = imageDataToMidasTensor(left);
    const feeds = { [session.inputNames[0]]: inputTensor };
    const results = await session.run(feeds);
    const outTensor = results[session.outputNames[0]];

    let depthRaw = outTensor.data;
    if (!(depthRaw instanceof Float32Array)) depthRaw = Float32Array.from(depthRaw);
    if (depthRaw.length !== NET_W * NET_H) throw new Error(`Unexpected depth size: ${depthRaw.length}. Expected ${NET_W * NET_H}.`);

    let depth = normalizeDepth(depthRaw);
    depth = blurDepth(depth, NET_W, NET_H);

    const right = synthesizeRightEye(left, depth, strength);

    await ffmpeg.writeFile(`/L/${name}`, await imageDataToPNGBytes(left));
    await ffmpeg.writeFile(`/R/${name}`, await imageDataToPNGBytes(right));

    const p = 0.10 + 0.70 * ((i + 1) / framesToProcess.length);
    setProgress(p);
    setStatus(`Frame ${i + 1}/${framesToProcess.length}`);
  }

  setStatus('Encoding SBS video…');
  setProgress(0.82);

  const rc2 = await ffmpeg.exec([
    '-framerate', String(fps),
    '-i', '/L/%05d.png',
    '-framerate', String(fps),
    '-i', '/R/%05d.png',
    '-filter_complex', '[0:v][1:v]hstack=inputs=2',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-crf', '18',
    '-preset', 'veryfast',
    '/Parallaxer_SBS.mp4'
  ]);
  if (rc2 !== 0) throw new Error('FFmpeg encode failed.');

  const out = await ffmpeg.readFile('/Parallaxer_SBS.mp4');
  const outBlob = new Blob([out], { type: 'video/mp4' });
  const url = URL.createObjectURL(outBlob);

  downloadEl.href = url;
  downloadEl.download = 'Parallaxer_SBS.mp4';
  downloadEl.style.display = 'inline-flex';

  previewEl.src = url;
  previewEl.style.display = 'block';

  setProgress(1);
  setStatus('Complete.');
  log('Done.');
}

// ---------- UI ----------
fileEl.addEventListener('change', () => {
  const f = fileEl.files?.[0];
  if (!f) return;
  fileLabel.textContent = `${f.name} (${Math.round(f.size / (1024 * 1024))} MB)`;
  runBtn.disabled = !ffmpegLoaded || !session;
});

loadBtn.addEventListener('click', async () => {
  try {
    loadBtn.disabled = true;
    setStatus('Loading engines…');
    logEl.textContent = '';
    log('Loading FFmpeg core…');
    await loadFFmpeg();
    log('Loading ONNX Runtime + MiDaS…');
    await loadORT();
    setStatus('Engines loaded. Choose a file and generate.');
    runBtn.disabled = !fileEl.files?.[0];
  } catch (e) {
    setStatus(`Error: ${e.message || e}`);
    loadBtn.disabled = false;
  }
});

runBtn.addEventListener('click', async () => {
  const f = fileEl.files?.[0];
  if (!f) return;

  try {
    runBtn.disabled = true;
    loadBtn.disabled = true;
    log('---');
    log(`Starting Parallaxer… (ORT backend: ${ortBackend})`);
    await generateSBS(f);
  } catch (e) {
    setStatus(`Error: ${e.message || e}`);
    log(`[error] ${e.stack || e}`);
  } finally {
    runBtn.disabled = false;
    loadBtn.disabled = false;
  }
});
