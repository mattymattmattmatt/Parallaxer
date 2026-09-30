import './styles.css';
import { Input, BlobSource, CanvasSink, ALL_FORMATS, canEncodeVideo } from 'mediabunny';
import { h, $, Store, clamp, timecode, formatBytes, formatDuration } from './ui/dom.js';
import { icon, LOGO_SVG } from './ui/icons.js';
import { Timeline } from './ui/widgets.js';
import { DepthEngine, detectWebGPU, isModelCached, clearModelCache } from './depth/engine.js';
import { getModel, CUSTOM_MODEL } from './depth/models.js';
import { loadSettings, saveSettings, VIEWS, LAYOUTS } from './core/settings.js';
import { classifyFile, probeVideo, codecName } from './media/probe.js';
import { Preview } from './app/preview.js';
import { buildInspector } from './app/inspector.js';
import { Exporter } from './app/exporter.js';
import { XRViewer, xrSupported } from './app/xr.js';

// ---------------------------------------------------------------------------------------------
// Core objects
// ---------------------------------------------------------------------------------------------

const appEl = $('#app');
const viewer = $('#viewer');
const video = $('#srcVideo');
const store = new Store(loadSettings(), { onPersist: saveSettings });
const engine = new DepthEngine();
let gpu = null;

const app = {
  engine,
  video,
  store,
  range: { in: null, out: null },
  gpuF16: false,
  toast,
  engineInfo,
  ensureEngine,
  isModelCached: (m) => (m.local ? Promise.resolve(true) : isModelCached(m)),
  gpu: () => gpu,
  currentModel: () => currentModel(),
  currentTime: () => video.currentTime,
  pausePlayback: () => video.pause(),
  bitmapFor: async (item) => {
    await item.ready;
    return item.bitmap;
  },
  snapshotPreview: () => {
    if (!preview.hasSource) return null;
    preview.draw();
    try {
      return $('#view').toDataURL('image/jpeg', 0.85);
    } catch {
      return null;
    }
  }
};

// ---------------------------------------------------------------------------------------------
// Toasts
// ---------------------------------------------------------------------------------------------

function toast({ kind = 'info', title, msg, actions = [], timeout = kind === 'bad' ? 9000 : 4500 }) {
  const ic = { ok: 'check', warn: 'alert', bad: 'alert', info: 'info' }[kind] ?? 'info';
  const el = h(
    'div',
    { class: `toast ${kind}`, role: kind === 'bad' ? 'alert' : 'status' },
    icon(ic),
    h(
      'div',
      { class: 'toast-body' },
      h('div', { class: 'toast-title' }, title),
      msg ? h('div', { class: 'toast-msg' }, msg) : null,
      actions.length ? h('div', { class: 'toast-actions' }, actions.map((a) => h('button', { class: 'btn', onclick: () => (a.run(), el.remove()) }, a.label))) : null
    ),
    h('button', { class: 'icon-btn sm', title: 'Dismiss', onclick: () => el.remove() }, icon('x'))
  );
  $('#toasts').appendChild(el);
  if (timeout) setTimeout(() => el.remove(), timeout);
  return el;
}

// ---------------------------------------------------------------------------------------------
// Static chrome: icons, labels
// ---------------------------------------------------------------------------------------------

$('#brandMark').innerHTML = LOGO_SVG;
const label = (id, ico, text) => $(id).replaceChildren(icon(ico), text ? h('span', {}, text) : '');
label('#openBtn', 'open', 'Open');
label('#camBtn', 'camera', 'Camera');
label('#screenBtn', 'screen', 'Screen');
label('#undoBtn', 'undo');
label('#redoBtn', 'redo');
label('#helpBtn', 'keyboard');
label('#exportBtn', 'export', 'Export');
label('#binAddBtn', 'plus');
label('#compareBtn', 'split');
label('#hudBtn', 'info');
label('#fsBtn', 'fullscreen');
label('#vrBtn', 'cube', 'View in VR');
label('#startBtn', 'toStart');
label('#backBtn', 'stepBack');
label('#playBtn', 'play');
label('#fwdBtn', 'stepFwd');
label('#endBtn', 'toEnd');
label('#clearRangeBtn', 'x');
label('#loopBtn', 'loop');
label('#muteBtn', 'volume');
label('#emptyOpen', 'open', 'Open video or photo');
label('#emptySample', 'sparkles', 'Try a sample photo');

// ---------------------------------------------------------------------------------------------
// Preview + inspector
// ---------------------------------------------------------------------------------------------

const hud = $('#hud');
let hudOn = localStorage.getItem('parallaxer.hud') !== '0';
hud.hidden = !hudOn;
$('#hudBtn').classList.toggle('on', hudOn);

const preview = new Preview({
  canvas: $('#view'),
  video,
  viewer,
  engine,
  store,
  onStats: (stats, conv) => {
    app.onDepthStats?.(stats, conv);
    app.onEngineInfo?.();
    if (stats.cut) {
      const f = $('#cutFlash');
      f.classList.add('show');
      setTimeout(() => f.classList.remove('show'), 700);
    }
    renderHud();
  },
  onError: (err) => {
    console.error(err);
    toast({ kind: 'bad', title: 'Depth inference failed', msg: err?.message ?? String(err) });
  },
  onRendered: () => renderHud()
});

const inspector = buildInspector($('#inspector'), store, app);
const exporter = new Exporter({ dlg: $('#exportDlg'), store, app });

function engineInfo() {
  const st = preview.stats;
  return {
    backend: engine.backend ? (engine.backend === 'webgpu' ? `WebGPU${gpu?.vendor ? ` · ${gpu.vendor}` : ''}` : `WebAssembly${self.crossOriginIsolated ? ' · threads' : ''}`) : 'Not loaded',
    precision: engine.ready ? engine.precision.toUpperCase() : '—',
    input: st ? `${st.inputW} × ${st.inputH}` : '—',
    ms: st ? `${st.ms.toFixed(1)} ms` : '—'
  };
}

let lastHud = 0;
function renderHud() {
  if (!hudOn || !preview.hasSource) return;
  const now = performance.now();
  if (now - lastHud < 120) return;
  lastHud = now;
  const st = preview.stats;
  const g = preview.geo;
  const s = store.state;
  const parts = [];
  if (engine.ready) {
    parts.push(h('span', { class: engine.backend === 'webgpu' ? 'gpu' : '' }, engine.backend === 'webgpu' ? 'WEBGPU' : 'CPU'));
    parts.push(h('span', {}, engine.model?.name ?? ''));
  } else {
    parts.push(h('span', {}, 'Depth engine loading…'));
  }
  if (st) {
    parts.push(h('span', {}, 'depth ', h('b', {}, `${st.ms.toFixed(1)} ms`)));
    if (preview.source?.kind !== 'image') parts.push(h('span', {}, h('b', {}, `${st.fps}`), ' fps'));
  }
  if (g) parts.push(h('span', {}, 'view ', h('b', {}, `${g.canvasW}×${g.canvasH}`), preview.renderMs ? ` · ${preview.renderMs.toFixed(1)} ms` : ''));
  parts.push(h('span', {}, 'conv ', h('b', {}, preview.fp.convergence(s).toFixed(2)), s.autoConvergence ? ' auto' : ''));
  hud.replaceChildren(...parts);
}

// ---------------------------------------------------------------------------------------------
// Depth engine lifecycle
// ---------------------------------------------------------------------------------------------

let engineKey = null;
let engineLoading = null;
let engineLoadingKey = null;
const loader = $('#loader');

function setEngineChip(state, text) {
  const b = $('#engineBtn');
  b.dataset.state = state;
  $('#engineText').textContent = text;
}

function currentModel() {
  const s = store.state;
  if (s.model === 'custom') return app.customModel ?? getModel('midas-small');
  return getModel(s.model);
}

app.loadCustomModel = async (file) => {
  const bytes = new Uint8Array(await file.arrayBuffer());
  app.customModel = {
    ...CUSTOM_MODEL,
    name: file.name.replace(/\.onnx$/i, ''),
    size: file.size,
    bytes,
    norm: store.state.customNorm,
    input: { multiple: store.state.customMultiple }
  };
  app.customModel.key = `${file.name}:${file.size}:${Date.now()}`;
  app.onCustomModel?.();
  if (store.state.model !== 'custom') store.set({ model: 'custom' });
  else engineKey = null;
  await ensureEngine();
};

function ensureEngine() {
  const s = store.state;
  if (s.model === 'custom' && !app.customModel) {
    store.set({ model: 'midas-small' }, { undoable: false });
    return Promise.resolve();
  }
  const key = `${s.model}|${s.backend}|${s.model === 'custom' ? app.customModel.key : ''}`;
  if (engine.ready && engineKey === key) return Promise.resolve();
  if (engineLoading && engineLoadingKey === key) return engineLoading;
  engineLoadingKey = key;
  const model = currentModel();
  setEngineChip('busy', `Loading ${model.name}…`);
  const showLoader = () => {
    loader.hidden = false;
    $('#loaderTitle').textContent = `Preparing ${model.name}`;
  };
  const loaderTimer = setTimeout(showLoader, 250);
  const bar = $('#loaderBar');
  bar.parentElement.classList.add('indeterminate');
  const p = engine
    .load(model.bytes ? model : model.id, {
      preferBackend: s.backend,
      gpu,
      onStatus: (msg) => ($('#loaderSub').textContent = msg),
      onProgress: ({ loaded, total, cached }) => {
        if (cached) return;
        if (total) {
          bar.parentElement.classList.remove('indeterminate');
          bar.style.width = `${Math.min(100, (loaded / total) * 100).toFixed(1)}%`;
          $('#loaderSub').textContent = `Downloading ${formatBytes(loaded)} of ${formatBytes(total)} · cached for next time`;
        } else {
          $('#loaderSub').textContent = `Downloading ${formatBytes(loaded)}…`;
        }
      }
    })
    .then((info) => {
      if (engineLoadingKey !== key) return;
      engineKey = key;
      app.customFixed = model === app.customModel && !!engine.fixedSize;
      setEngineChip('ready', `${info.backend === 'webgpu' ? 'WebGPU' : 'CPU'} · ${model.name.replace('Depth Anything V2', 'DA-V2')}`);
      preview.fp.reset();
      preview.refresh();
      inspector.refreshModels();
      inspector.refreshEngine();
      if (s.backend !== 'wasm' && info.backend === 'wasm' && gpu) {
        toast({ kind: 'warn', title: 'Running on CPU', msg: 'WebGPU initialisation failed for this model, so inference falls back to WebAssembly.' });
      }
    })
    .catch((err) => {
      console.error(err);
      setEngineChip('error', 'Engine error');
      const fallback = s.model !== 'midas-small';
      toast({
        kind: 'bad',
        title: `Couldn't load ${model.name}`,
        msg: `${err?.message ?? err}${fallback ? ' — switching back to the bundled MiDaS model.' : ''}`
      });
      if (fallback && store.state.model === model.id) store.set({ model: 'midas-small' }, { undoable: false });
    })
    .finally(() => {
      clearTimeout(loaderTimer);
      if (engineLoadingKey === key) {
        loader.hidden = true;
        bar.style.width = '0%';
        engineLoading = null;
      }
    });
  engineLoading = p;
  return p;
}

// ---------------------------------------------------------------------------------------------
// Media bin
// ---------------------------------------------------------------------------------------------

const items = [];
let current = null;
let liveStream = null;
let idSeq = 0;

function renderBin() {
  const list = $('#binList');
  if (!items.length) {
    list.replaceChildren(h('div', { class: 'bin-empty' }, 'Drop videos or photos here. Everything stays on this device.'));
  } else {
    list.replaceChildren(
      ...items.map((it) => {
        const meta = it.error
          ? h('div', { class: 'bin-status bad' }, it.error)
          : it.meta
            ? h('div', { class: 'bin-meta' }, it.kind === 'video' ? `${it.meta.width}×${it.meta.height} · ${formatDuration(it.meta.duration)}` : `${it.meta.width}×${it.meta.height} · ${formatBytes(it.file.size)}`)
            : h('div', { class: 'bin-meta' }, 'Reading…');
        const thumb = h('div', { class: 'bin-thumb', style: it.poster ? { backgroundImage: `url(${it.poster})` } : {} }, icon(it.kind === 'video' ? 'film' : 'image'));
        const rm = h('button', { class: 'icon-btn sm bin-remove', title: 'Remove from bin' }, icon('x'));
        rm.addEventListener('click', (e) => {
          e.stopPropagation();
          removeItem(it);
        });
        const row = h(
          'div',
          { class: `bin-item${it === current ? ' active' : ''}`, role: 'button', tabindex: '0', 'aria-current': it === current ? 'true' : null, onclick: () => selectItem(it) },
          thumb,
          h('div', { class: 'bin-info' }, h('div', { class: 'bin-name', title: it.name }, it.name), meta),
          rm
        );
        row.addEventListener('keydown', (e) => {
          if (e.target !== row) return;
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            e.stopPropagation();
            selectItem(it);
          } else if (e.key === 'Delete' || e.key === 'Backspace') {
            e.stopPropagation();
            removeItem(it);
          }
        });
        return row;
      })
    );
  }
  $('#batchBtn').disabled = items.filter((i) => !i.error).length < 2;
}

function removeItem(it) {
  const i = items.indexOf(it);
  if (i < 0) return;
  items.splice(i, 1);
  if (it.url) URL.revokeObjectURL(it.url);
  if (current === it) {
    current = null;
    const next = items[Math.min(i, items.length - 1)];
    if (next) selectItem(next);
    else setNoMedia();
  }
  renderBin();
}

function setNoMedia() {
  preview.clear();
  appEl.dataset.media = 'none';
  $('#docName').textContent = 'Untitled';
  $('#docMeta').textContent = '';
  $('#exportBtn').disabled = true;
}

async function makePoster(source, w, h0) {
  const k = 160 / Math.max(w, h0);
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(w * k));
  c.height = Math.max(1, Math.round(h0 * k));
  c.getContext('2d').drawImage(source, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.8);
}

function addFiles(fileList, { select = true } = {}) {
  const added = [];
  for (const file of fileList) {
    const kind = classifyFile(file);
    if (kind === 'unknown') {
      toast({ kind: 'warn', title: 'Unsupported file', msg: `${file.name} isn't a video or image.` });
      continue;
    }
    const it = { id: ++idSeq, file, kind, name: file.name, meta: null, error: null };
    it.ready = loadItem(it);
    items.push(it);
    added.push(it);
  }
  renderBin();
  if (select && added.length) selectItem(added[0]);
  return added;
}

async function loadItem(it) {
  try {
    if (it.kind === 'image') {
      it.bitmap = await createImageBitmap(it.file, { imageOrientation: 'from-image' });
      it.meta = { width: it.bitmap.width, height: it.bitmap.height };
      it.poster = await makePoster(it.bitmap, it.bitmap.width, it.bitmap.height);
    } else {
      it.meta = await probeVideo(it.file);
      if (!it.meta.canDecode) it.error = `${codecName(it.meta.codec)} can't be decoded here`;
      makeThumbs(it).catch(() => {});
    }
  } catch (err) {
    it.error = err?.message?.slice(0, 80) || 'Unreadable file';
  }
  renderBin();
}

/** Timeline filmstrip + bin poster via WebCodecs (independent of the <video> element). */
async function makeThumbs(it) {
  const input = new Input({ source: new BlobSource(it.file), formats: ALL_FORMATS });
  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track || !(await track.canDecode())) return;
    const n = 24;
    const d = it.meta.duration;
    const sink = new CanvasSink(track, { width: 192 });
    const ts = Array.from({ length: n }, (_, i) => ((i + 0.5) * d) / n);
    it.thumbs = [];
    let i = 0;
    for await (const wc of sink.canvasesAtTimestamps(ts)) {
      it.thumbs.push(wc?.canvas ?? null);
      if (i === Math.floor(n / 3) && wc) {
        it.poster = await makePoster(wc.canvas, wc.canvas.width, wc.canvas.height);
        renderBin();
      }
      if (current === it && i % 4 === 3) timeline.setThumbnails(it.thumbs);
      i++;
    }
    if (current === it) timeline.setThumbnails(it.thumbs);
  } finally {
    input.dispose();
  }
}

async function selectItem(it) {
  stopLive();
  current = it;
  renderBin();
  await it.ready;
  if (current !== it) return;
  if (it.error && it.kind === 'video') {
    toast({ kind: 'bad', title: `Can't open ${it.name}`, msg: `${it.error}. Chrome or Edge support the widest range of codecs.` });
  }
  if (!it.meta) return;
  setRange(null, null);
  if (it.kind === 'image') {
    appEl.dataset.media = 'image';
    preview.setImage(it.bitmap);
    $('#docMeta').textContent = `${it.meta.width}×${it.meta.height} · ${it.file.type.replace('image/', '').toUpperCase() || 'IMAGE'} · ${formatBytes(it.file.size)}`;
  } else {
    appEl.dataset.media = 'video';
    it.url ??= URL.createObjectURL(it.file);
    const m = it.meta;
    preview.setVideo(it.url, m);
    video.playbackRate = Number($('#rateSel').value);
    timeline.setMedia(m.duration, m.fps);
    timeline.setThumbnails(it.thumbs ?? []);
    $('#tcDur').textContent = timecode(m.duration, m.fps);
    const audio = m.audio ? `${codecName(m.audio.codec)} ${Math.round(m.audio.sampleRate / 1000)}k` : 'no audio';
    $('#docMeta').textContent = `${m.width}×${m.height} · ${m.fps.toFixed(3).replace(/\.?0+$/, '')} fps${m.vfr ? ' VFR' : ''} · ${codecName(m.codec)}${m.hdr ? ' HDR' : ''} · ${audio} · ${formatDuration(m.duration)}`;
  }
  $('#docName').textContent = it.name;
  $('#exportBtn').disabled = false;
  label('#exportBtn', 'export', 'Export');
  ensureEngine();
}

// ---------------------------------------------------------------------------------------------
// Live camera / screen + recording
// ---------------------------------------------------------------------------------------------

let recorder = null;
let recTimer = 0;

async function startLive(kind) {
  let stream;
  try {
    stream =
      kind === 'camera'
        ? await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false })
        : await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: { ideal: 30 } }, audio: false });
  } catch (err) {
    if (err?.name !== 'NotAllowedError') toast({ kind: 'bad', title: `Couldn't start ${kind}`, msg: err?.message ?? String(err) });
    return;
  }
  stopLive();
  liveStream = stream;
  current = null;
  renderBin();
  setRange(null, null);
  appEl.dataset.media = 'live';
  preview.setStream(stream);
  $('#docName').textContent = kind === 'camera' ? 'Live camera' : 'Live screen capture';
  $('#docMeta').textContent = 'real-time 2D → 3D';
  $('#exportBtn').disabled = false;
  label('#exportBtn', 'download', 'Record');
  stream.getVideoTracks()[0]?.addEventListener('ended', () => {
    if (liveStream === stream) {
      stopLive();
      setNoMedia();
    }
  });
  ensureEngine();
  toast({ kind: 'info', title: kind === 'camera' ? 'Camera is live in 3D' : 'Screen is live in 3D', msg: 'Pick a view, go fullscreen with F, or press Record to capture the output.' });
}

function stopLive() {
  if (recorder) toggleRecording();
  if (!liveStream) return;
  liveStream.getTracks().forEach((t) => t.stop());
  liveStream = null;
  label('#exportBtn', 'export', 'Export');
}

function toggleRecording() {
  if (recorder) {
    recorder.stop();
    return;
  }
  const types = ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = types.find((t) => window.MediaRecorder?.isTypeSupported?.(t));
  if (!mimeType) {
    toast({ kind: 'bad', title: 'Recording not supported in this browser' });
    return;
  }
  const stream = $('#view').captureStream(30);
  const chunks = [];
  recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 16e6 });
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const started = performance.now();
  recorder.onstop = () => {
    clearInterval(recTimer);
    recorder = null;
    const blob = new Blob(chunks, { type: mimeType.split(';')[0] });
    const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
    const view = store.state.view === 'output' ? LAYOUTS[store.state.layout].tag : store.state.view;
    const a = h('a', { href: URL.createObjectURL(blob), download: `parallaxer-live.${view}.${ext}` });
    a.click();
    label('#exportBtn', 'download', 'Record');
    $('#exportBtn').classList.remove('recording');
    toast({ kind: 'ok', title: 'Recording saved', msg: `${formatBytes(blob.size)} · ${formatDuration((performance.now() - started) / 1000)}` });
  };
  recorder.start(1000);
  $('#exportBtn').classList.add('recording');
  recTimer = setInterval(() => label('#exportBtn', 'x', `Stop · ${formatDuration((performance.now() - started) / 1000)}`), 500);
  label('#exportBtn', 'x', 'Stop · 0:00');
}

// ---------------------------------------------------------------------------------------------
// Transport + timeline
// ---------------------------------------------------------------------------------------------

const timeline = new Timeline({
  onSeek: (t) => {
    video.currentTime = t;
    timeline.setTime(t);
    updateTimecode();
  },
  onRange: (a, b) => setRange(a, b)
});
$('#timelineHost').appendChild(timeline.el);

const fps = () => current?.meta?.fps ?? 30;

function updateTimecode() {
  $('#tcNow').textContent = timecode(video.currentTime, fps());
  timeline.setTime(video.currentTime);
}

function setRange(a, b) {
  app.range.in = a;
  app.range.out = b;
  timeline.setRange(a, b);
  $('#inVal').textContent = a === null ? '—' : timecode(a, fps());
  $('#outVal').textContent = b === null ? '—' : timecode(b, fps());
  $('#inBtn').classList.toggle('on', a !== null);
  $('#outBtn').classList.toggle('on', b !== null);
}

let loop = localStorage.getItem('parallaxer.loop') === '1';
$('#loopBtn').classList.toggle('on', loop);

function clock() {
  if (video.paused) return;
  const t = video.currentTime;
  const out = app.range.out;
  if (out !== null && t >= out) {
    if (loop) video.currentTime = app.range.in ?? 0;
    else video.pause();
  }
  updateTimecode();
  requestAnimationFrame(clock);
}

video.addEventListener('play', () => {
  label('#playBtn', 'pause');
  requestAnimationFrame(clock);
});
video.addEventListener('pause', () => {
  label('#playBtn', 'play');
  updateTimecode();
});
video.addEventListener('seeked', updateTimecode);
video.addEventListener('ended', () => {
  if (loop && appEl.dataset.media === 'video') {
    video.currentTime = app.range.in ?? 0;
    video.play();
  }
});
video.addEventListener('error', () => {
  if (appEl.dataset.media !== 'video') return;
  toast({ kind: 'bad', title: 'Playback error', msg: `This browser can't play ${current?.name ?? 'this file'}. Try Chrome or Edge, or an H.264 / VP9 file.` });
});

function togglePlay() {
  if (appEl.dataset.media !== 'video') return;
  if (video.paused) {
    const t = video.currentTime;
    if ((app.range.out !== null && t >= app.range.out - 1e-3) || video.ended) video.currentTime = app.range.in ?? 0;
    video.play().catch(() => {});
  } else video.pause();
}

function stepFrames(n) {
  if (appEl.dataset.media !== 'video') return;
  video.pause();
  const f = fps();
  const idx = Math.floor(video.currentTime * f + 1e-3) + n;
  video.currentTime = clamp((idx + 0.5) / f, 0, current?.meta?.duration ?? video.duration);
}

function seekBy(sec) {
  if (appEl.dataset.media !== 'video') return;
  video.currentTime = clamp(video.currentTime + sec, 0, video.duration || 0);
}

$('#playBtn').addEventListener('click', togglePlay);
$('#backBtn').addEventListener('click', () => stepFrames(-1));
$('#fwdBtn').addEventListener('click', () => stepFrames(1));
$('#startBtn').addEventListener('click', () => (video.currentTime = app.range.in ?? 0));
$('#endBtn').addEventListener('click', () => (video.currentTime = app.range.out ?? Math.max(0, (video.duration || 0) - 1 / fps())));
$('#inBtn').addEventListener('click', () => setRange(video.currentTime, app.range.out !== null && app.range.out <= video.currentTime ? null : app.range.out));
$('#outBtn').addEventListener('click', () => setRange(app.range.in !== null && app.range.in >= video.currentTime ? null : app.range.in, video.currentTime));
$('#clearRangeBtn').addEventListener('click', () => setRange(null, null));
$('#loopBtn').addEventListener('click', () => {
  loop = !loop;
  localStorage.setItem('parallaxer.loop', loop ? '1' : '0');
  $('#loopBtn').classList.toggle('on', loop);
});
$('#rateSel').addEventListener('change', (e) => (video.playbackRate = Number(e.target.value)));

const vol = $('#volRange');
const syncVol = () => {
  vol.style.setProperty('--p', `${(video.muted ? 0 : video.volume) * 100}%`);
  label('#muteBtn', video.muted || video.volume === 0 ? 'mute' : 'volume');
};
video.volume = Number(localStorage.getItem('parallaxer.vol') ?? 0.8);
vol.value = video.volume;
vol.addEventListener('input', () => {
  video.volume = Number(vol.value);
  video.muted = false;
  localStorage.setItem('parallaxer.vol', vol.value);
  syncVol();
});
$('#muteBtn').addEventListener('click', () => {
  video.muted = !video.muted;
  syncVol();
});
video.addEventListener('volumechange', syncVol);
syncVol();

// ---------------------------------------------------------------------------------------------
// View tabs
// ---------------------------------------------------------------------------------------------

const viewsEl = $('#views');
const viewBtns = VIEWS.map((v) =>
  h(
    'button',
    { type: 'button', class: 'seg-btn', role: 'tab', title: v.hint, onclick: () => store.set({ view: v.id }, { undoable: false }) },
    v.label,
    h('kbd', {}, v.key)
  )
);
viewsEl.replaceChildren(...viewBtns);

let hintTimer = 0;
function syncView(showHint) {
  const s = store.state;
  VIEWS.forEach((v, i) => viewBtns[i].classList.toggle('on', v.id === s.view));
  viewer.classList.toggle('look', s.view === 'look');
  if (showHint) {
    const v = VIEWS.find((x) => x.id === s.view);
    const hint = $('#viewHint');
    hint.textContent = s.view === 'output' ? `${LAYOUTS[s.layout].label} — ${v.hint}` : v.hint;
    hint.classList.add('show');
    clearTimeout(hintTimer);
    hintTimer = setTimeout(() => hint.classList.remove('show'), 2600);
  }
}
syncView(false);

// ---------------------------------------------------------------------------------------------
// Settings reactions
// ---------------------------------------------------------------------------------------------

function syncUndo() {
  $('#undoBtn').disabled = !store.undoStack.length;
  $('#redoBtn').disabled = !store.redoStack.length;
}
syncUndo();

store.subscribe((s, changed) => {
  if (changed.includes('model') || changed.includes('backend')) {
    if (preview.hasSource) ensureEngine();
    else engineKey = null;
  } else if (changed.includes('detail')) {
    preview.fp.reset();
    preview.refresh();
  }
  if ((changed.includes('customNorm') || changed.includes('customMultiple')) && app.customModel) {
    app.customModel.norm = s.customNorm;
    app.customModel.input = { multiple: s.customMultiple };
    if (s.model === 'custom') {
      preview.fp.reset();
      preview.refresh();
    }
  }
  if (changed.includes('view')) syncView(true);
  if (changed.includes('layout') && s.view === 'output') syncView(true);
  preview.requestDraw();
  syncUndo();
  renderHud();
});

// ---------------------------------------------------------------------------------------------
// Buttons & dialogs
// ---------------------------------------------------------------------------------------------

const fileInput = $('#fileInput');
const openPicker = () => fileInput.click();
fileInput.addEventListener('change', () => {
  addFiles([...fileInput.files]);
  fileInput.value = '';
});
$('#openBtn').addEventListener('click', openPicker);
$('#binAddBtn').addEventListener('click', openPicker);
$('#emptyOpen').addEventListener('click', openPicker);
$('#emptySample').addEventListener('click', async () => {
  const res = await fetch(new URL('samples/coffee.jpg', document.baseURI));
  const blob = await res.blob();
  addFiles([new File([blob], 'Sample - espresso.jpg', { type: 'image/jpeg' })]);
});
$('#camBtn').addEventListener('click', () => startLive('camera'));
$('#emptyCam').addEventListener('click', () => startLive('camera'));
$('#screenBtn').addEventListener('click', () => startLive('screen'));
$('#emptyScreen').addEventListener('click', () => startLive('screen'));
$('#undoBtn').addEventListener('click', () => store.undo());
$('#redoBtn').addEventListener('click', () => store.redo());
$('#batchBtn').addEventListener('click', () => {
  const ok = items.filter((i) => !i.error && i.meta);
  if (ok.length) exporter.open(ok);
});

function openExport() {
  if (appEl.dataset.media === 'live') return toggleRecording();
  if (!current?.meta) return;
  exporter.open([current]);
}
$('#exportBtn').addEventListener('click', openExport);

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else viewer.requestFullscreen?.().catch(() => {});
}
$('#fsBtn').addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', () => preview.requestDraw());

$('#hudBtn').addEventListener('click', toggleHud);
function toggleHud() {
  hudOn = !hudOn;
  hud.hidden = !hudOn;
  $('#hudBtn').classList.toggle('on', hudOn);
  localStorage.setItem('parallaxer.hud', hudOn ? '1' : '0');
  lastHud = 0;
  renderHud();
}

const compareBtn = $('#compareBtn');
const setCompare = (on) => {
  preview.compare = on;
  compareBtn.classList.toggle('on', on);
  preview.requestDraw();
};
compareBtn.addEventListener('pointerdown', () => setCompare(true));
compareBtn.addEventListener('pointerup', () => setCompare(false));
compareBtn.addEventListener('pointerleave', () => setCompare(false));

// Help
const SHORTCUTS = [
  ['Playback', [['Play / pause', ['Space']], ['Previous / next frame', ['←', '→']], ['Back / forward 1 s', ['⇧', '← →']], ['Start / end', ['Home', 'End']], ['Set In / Out', ['I', 'O']], ['Clear In / Out', ['X']], ['Loop', ['L']], ['Mute', ['M']]]],
  ['Views', [['Output … Original', ['1', '–', '8']], ['Compare with original (hold)', ['\\']], ['Fullscreen viewer', ['F']], ['Stats overlay', ['H']], ['Toggle media bin', ['B']]]],
  ['Stereo', [['Depth budget − / +', ['[', ']']], ['Screen plane − / +', [',', '.']], ['Auto-converge', ['A']], ['Swap eyes', ['S']]]],
  ['Project', [['Open files', ['Ctrl', 'O']], ['Export', ['Ctrl', 'E']], ['Undo / redo', ['Ctrl', 'Z / ⇧Z']], ['Shortcuts', ['?']]]]
];
$('#helpBtn').addEventListener('click', openHelp);
function openHelp() {
  const dlg = $('#helpDlg');
  dlg.replaceChildren(
    h('div', { class: 'modal-head' }, h('h2', {}, 'Keyboard shortcuts'), h('button', { class: 'icon-btn', onclick: () => dlg.close() }, icon('x'))),
    h(
      'div',
      { class: 'modal-body' },
      h(
        'div',
        { class: 'kbd-grid' },
        SHORTCUTS.flatMap(([group, rows]) => [h('div', { class: 'kbd-head' }, group), ...rows.map(([what, keys]) => h('div', { class: 'kbd-row' }, h('span', {}, what), h('span', {}, keys.map((k) => h('kbd', {}, k)))))])
      )
    )
  );
  dlg.showModal();
}

// System / engine dialog
$('#engineBtn').addEventListener('click', openSystem);
async function openSystem() {
  const dlg = $('#sysDlg');
  const gl = $('#view').getContext('webgl2');
  const dbg = gl?.getExtension('WEBGL_debug_renderer_info');
  const glName = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl?.getParameter(gl.RENDERER);
  const enc = await Promise.all(['avc', 'hevc', 'av1', 'vp9'].map(async (c) => [c, await canEncodeVideo(c, { width: 3840, height: 1080 }).catch(() => false)]));
  const est = await navigator.storage?.estimate?.().catch(() => null);
  const yes = (b, t = 'Yes') => h('span', { class: b ? 'yes' : 'no' }, b ? t : 'No');
  const ei = engineInfo();
  dlg.replaceChildren(
    h('div', { class: 'modal-head' }, h('h2', {}, 'Engine & system'), h('button', { class: 'icon-btn', onclick: () => dlg.close() }, icon('x'))),
    h(
      'div',
      { class: 'modal-body' },
      h(
        'dl',
        { class: 'sys-list' },
        [
          ['Depth model', engine.ready ? `${engine.model.name} (${engine.precision})` : 'Not loaded'],
          ['Inference backend', ei.backend],
          ['Network input', ei.input],
          ['Last inference', ei.ms],
          ['WebGPU adapter', gpu ? `${gpu.vendor} ${gpu.architecture}`.trim() + (gpu.f16 ? ' · fp16' : '') : h('span', { class: 'no' }, 'Unavailable')],
          ['WebGL renderer', glName ?? '—'],
          ['WebCodecs', yes(typeof VideoEncoder !== 'undefined')],
          ['Encoders @ 3840×1080', h('span', {}, enc.map(([c, ok]) => h('span', { class: ok ? 'yes' : 'muted' }, `${c.toUpperCase()} `)))],
          ['Multi-threaded CPU', yes(self.crossOriginIsolated, `Yes (${navigator.hardwareConcurrency} cores)`)],
          ['Stream to disk', yes(typeof window.showSaveFilePicker === 'function')],
          ['Storage used', est ? `${formatBytes(est.usage)} of ${formatBytes(est.quota)}` : '—']
        ].flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)])
      )
    ),
    h(
      'div',
      { class: 'modal-foot' },
      h('span', { class: 'note' }, 'Models are cached in your browser after the first download.'),
      h(
        'div',
        { class: 'actions' },
        h(
          'button',
          {
            class: 'btn',
            onclick: async () => {
              await clearModelCache();
              inspector.refreshModels();
              toast({ kind: 'ok', title: 'Model cache cleared' });
              dlg.close();
            }
          },
          icon('trash'),
          'Clear model cache'
        ),
        installPrompt ? h('button', { class: 'btn', onclick: () => (app.install(), dlg.close()) }, icon('download'), 'Install app') : null,
        h('button', { class: 'btn primary', onclick: () => dlg.close() }, 'Close')
      )
    )
  );
  dlg.showModal();
}

for (const d of ['#helpDlg', '#sysDlg', '#exportDlg']) {
  $(d).addEventListener('click', (e) => {
    if (e.target === e.currentTarget && !(d === '#exportDlg' && exporter.running)) e.currentTarget.close();
  });
}

// ---------------------------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------------------------

function nudge(key, delta, lo, hi, digits = 3) {
  const v = clamp(Math.round((store.state[key] + delta) * 10 ** digits) / 10 ** digits, lo, hi);
  store.set({ [key]: v });
}

window.addEventListener('keydown', (e) => {
  const t = e.target;
  if (t instanceof HTMLInputElement && t.type !== 'range' && t.type !== 'checkbox') return;
  if (t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement) return;
  if (document.querySelector('dialog[open]') && e.key !== 'Escape') return;
  const mod = e.ctrlKey || e.metaKey;
  const k = e.key;
  if (mod && k.toLowerCase() === 'z') {
    e.preventDefault();
    if (e.shiftKey) store.redo();
    else store.undo();
    return;
  }
  if (mod && k.toLowerCase() === 'y') return e.preventDefault(), store.redo();
  if (mod && k.toLowerCase() === 'o') return e.preventDefault(), openPicker();
  if (mod && k.toLowerCase() === 'e') return e.preventDefault(), openExport();
  if (mod || e.altKey) return;
  if (t instanceof HTMLInputElement && t.type === 'range' && ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(k)) return;

  const view = VIEWS.find((v) => v.key === k);
  if (view) return store.set({ view: view.id }, { undoable: false });
  switch (k) {
    case ' ':
      e.preventDefault();
      togglePlay();
      break;
    case 'ArrowLeft':
      e.preventDefault();
      e.shiftKey ? seekBy(-1) : stepFrames(-1);
      break;
    case 'ArrowRight':
      e.preventDefault();
      e.shiftKey ? seekBy(1) : stepFrames(1);
      break;
    case 'Home':
      video.currentTime = app.range.in ?? 0;
      break;
    case 'End':
      video.currentTime = app.range.out ?? Math.max(0, (video.duration || 0) - 1 / fps());
      break;
    case 'i':
    case 'I':
      $('#inBtn').click();
      break;
    case 'o':
    case 'O':
      $('#outBtn').click();
      break;
    case 'x':
    case 'X':
      setRange(null, null);
      break;
    case 'l':
    case 'L':
      $('#loopBtn').click();
      break;
    case 'm':
    case 'M':
      $('#muteBtn').click();
      break;
    case 'f':
    case 'F':
      toggleFullscreen();
      break;
    case 'h':
    case 'H':
      toggleHud();
      break;
    case 'b':
    case 'B':
      appEl.classList.toggle('no-bin');
      preview.requestDraw();
      break;
    case '[':
      nudge('strength', -0.2, 0, 10, 1);
      break;
    case ']':
      nudge('strength', 0.2, 0, 10, 1);
      break;
    case ',':
      store.set({ autoConvergence: false });
      nudge('convergence', -0.02, 0, 1, 2);
      break;
    case '.':
      store.set({ autoConvergence: false });
      nudge('convergence', 0.02, 0, 1, 2);
      break;
    case 'a':
    case 'A':
      store.set({ autoConvergence: !store.state.autoConvergence });
      break;
    case 's':
    case 'S':
      store.set({ swap: !store.state.swap });
      break;
    case '\\':
      setCompare(true);
      break;
    case '?':
      openHelp();
      break;
    case 'e':
    case 'E':
      openExport();
      break;
    default:
      return;
  }
});
window.addEventListener('keyup', (e) => {
  if (e.key === '\\') setCompare(false);
});

// ---------------------------------------------------------------------------------------------
// Drag & drop, paste
// ---------------------------------------------------------------------------------------------

let dragDepth = 0;
window.addEventListener('dragenter', (e) => {
  if (![...(e.dataTransfer?.types ?? [])].includes('Files')) return;
  dragDepth++;
  appEl.classList.add('dragging');
});
window.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) appEl.classList.remove('dragging');
});
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  appEl.classList.remove('dragging');
  const files = [...(e.dataTransfer?.files ?? [])];
  if (files.length) addFiles(files);
});
window.addEventListener('paste', (e) => {
  const files = [...(e.clipboardData?.files ?? [])];
  if (files.length) {
    e.preventDefault();
    addFiles(files.map((f, i) => (f.name && f.name !== 'image.png' ? f : new File([f], `Pasted image ${i + 1}.png`, { type: f.type }))));
  }
});
window.addEventListener('beforeunload', (e) => {
  if (exporter.running) e.preventDefault();
});

// ---------------------------------------------------------------------------------------------
// WebXR
// ---------------------------------------------------------------------------------------------

const xr = new XRViewer({
  preview,
  store,
  onToggle: togglePlay,
  onEnd: () => label('#vrBtn', 'cube', 'View in VR')
});
$('#vrBtn').addEventListener('click', async () => {
  if (xr.active) return xr.stop();
  if (!preview.hasSource) {
    toast({ kind: 'info', title: 'Open a video or photo first', msg: 'The VR viewer shows the live stereo conversion on a virtual cinema screen.' });
    return;
  }
  try {
    await xr.start();
    label('#vrBtn', 'x', 'Exit VR');
  } catch (err) {
    toast({ kind: 'bad', title: 'Could not start VR', msg: err?.message ?? String(err) });
  }
});

// ---------------------------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------------------------

async function boot() {
  renderBin();
  if (store.state.model === 'custom') store.set({ model: 'midas-small' }, { undoable: false });
  gpu = await detectWebGPU();
  app.gpuF16 = !!gpu?.f16;
  inspector.refreshModels();
  const caps = [];
  caps.push(h('span', { class: `cap${gpu ? '' : ' warn'}` }, gpu ? `WebGPU · ${gpu.vendor}` : 'CPU inference (no WebGPU)'));
  const hasCodecs = typeof VideoEncoder !== 'undefined';
  caps.push(h('span', { class: `cap${hasCodecs ? '' : ' off'}` }, hasCodecs ? 'WebCodecs hardware video' : 'No WebCodecs — photos only'));
  const avc = hasCodecs && (await canEncodeVideo('avc', { width: 3840, height: 1080 }).catch(() => false));
  caps.push(h('span', { class: `cap${avc ? '' : ' warn'}` }, avc ? 'H.264 export' : 'VP9 / AV1 export'));
  caps.push(h('span', { class: 'cap' }, '100% on-device'));
  $('#caps').replaceChildren(...caps);
  setEngineChip('idle', gpu ? 'WebGPU ready' : 'CPU mode');
  if (await xrSupported()) $('#vrBtn').hidden = false;
  if (!hasCodecs) toast({ kind: 'warn', title: 'Limited browser', msg: 'This browser lacks WebCodecs, so video export is unavailable. Photos still work. Use Chrome, Edge or Safari 17+ for video.' });
}

boot();

// Offline + installable app (production builds only; dev relies on Vite's module server).
if (import.meta.env.PROD && 'serviceWorker' in navigator && window.isSecureContext) {
  window.addEventListener('load', () => navigator.serviceWorker.register(new URL('sw.js', document.baseURI)).catch(() => {}));
}

// Files opened through the installed app's file handler (e.g. "Open with Parallaxer").
if ('launchQueue' in window) {
  window.launchQueue.setConsumer(async (params) => {
    const files = await Promise.all((params.files ?? []).map((h) => h.getFile()));
    if (files.length) addFiles(files);
  });
}

let installPrompt = null;
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  installPrompt = e;
});
app.install = async () => {
  if (!installPrompt) return false;
  installPrompt.prompt();
  installPrompt = null;
  return true;
};
