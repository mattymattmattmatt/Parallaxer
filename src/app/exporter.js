import { h, formatBytes, formatDuration, timecode } from '../ui/dom.js';
import { icon } from '../ui/icons.js';
import { LAYOUTS, layoutGeometry } from '../core/settings.js';
import {
  CONTAINERS,
  VIDEO_CODECS,
  QUALITY_LEVELS,
  MOTION_PATHS,
  VideoExportJob,
  ExportCancelled,
  probeEncoders,
  targetBitrate,
  baseName,
  outputName,
  supportsDiskStreaming,
  pickSaveFile,
  renderStill,
  exportMotion
} from '../media/export.js';

const OPTS_KEY = 'parallaxer.export.v1';
const DEFAULT_OPTS = {
  tab: 'video',
  container: 'mp4',
  codec: 'auto',
  quality: 'high',
  mbps: 40,
  res: 'source',
  fps: 0,
  range: 'full',
  audio: 'copy',
  toDisk: false,
  still: 'png',
  stillScale: 1,
  motionPath: 'orbit',
  motionSeconds: 6,
  motionAmount: 1.3,
  motionFps: 30,
  motionSize: 1920
};

function loadOpts() {
  try {
    return { ...DEFAULT_OPTS, ...JSON.parse(localStorage.getItem(OPTS_KEY) || '{}') };
  } catch {
    return { ...DEFAULT_OPTS };
  }
}

function saveOpts(o) {
  try {
    localStorage.setItem(OPTS_KEY, JSON.stringify(o));
  } catch {
    /* ignore */
  }
}

function seg(options, value, onChange, { disabled = false } = {}) {
  return h(
    'div',
    { class: 'seg' },
    options.map((o) =>
      h(
        'button',
        {
          type: 'button',
          class: `seg-btn${o.value === value ? ' on' : ''}`,
          disabled: disabled || o.disabled,
          title: o.hint ?? '',
          onclick: () => onChange(o.value)
        },
        o.label
      )
    )
  );
}

function field(label, control, hint) {
  return h('div', { class: 'ctl' }, h('div', { class: 'ctl-head' }, h('span', { class: 'ctl-label' }, label)), control, hint ? h('div', { class: 'ctl-hint' }, hint) : null);
}

function selectEl(options, value, onChange) {
  const sel = h(
    'select',
    { class: 'select' },
    options.map((o) => h('option', { value: o.value, disabled: o.disabled }, o.label))
  );
  sel.value = String(value);
  sel.addEventListener('change', () => onChange(sel.value));
  sel.addEventListener('keydown', (e) => e.stopPropagation());
  return h('div', { class: 'select-wrap' }, sel, icon('chevron'));
}

/** Human duration that stays meaningful for very short spans (partial exports can be under a second). */
function formatSpan(sec) {
  if (!Number.isFinite(sec)) return '—';
  return sec < 60 ? `${sec.toFixed(1)} s` : formatDuration(sec);
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = h('a', { href: url, download: name });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
  return url;
}

function eyeSizeFor(w, h, res) {
  if (res === 'source') return { w, h };
  const cap = Number(res);
  const short = Math.min(w, h);
  if (short <= cap) return { w, h };
  const k = cap / short;
  return { w: w * k, h: h * k };
}

export class Exporter {
  constructor({ dlg, store, app }) {
    this.dlg = dlg;
    this.store = store;
    this.app = app;
    this.opts = loadOpts();
    this.job = null;
    this.running = false;
    dlg.addEventListener('cancel', (e) => {
      if (this.running) e.preventDefault();
    });
  }

  setOpt(patch) {
    Object.assign(this.opts, patch);
    saveOpts(this.opts);
    this.renderConfig();
  }

  open(items) {
    this.items = items;
    // In a mixed batch the video settings drive the dialog; photos are saved as stills alongside.
    this.item = items.find((i) => i.kind === 'video') ?? items[0];
    const kind = this.item.kind;
    if (kind === 'image' && !['still', 'motion'].includes(this.opts.tab)) this.opts.tab = 'still';
    if (kind === 'video' && !['video', 'frame'].includes(this.opts.tab)) this.opts.tab = 'video';
    if (items.length > 1) this.opts.tab = kind === 'image' ? 'still' : 'video';
    this.thumb = this.app.snapshotPreview();
    this.renderConfig();
    if (!this.dlg.open) this.dlg.showModal();
  }

  close() {
    if (this.running) return;
    this.dlg.close();
  }

  // ---------- Configuration view ----------

  outputGeometry() {
    const it = this.item;
    const s = this.store.state;
    const w = it.meta?.width ?? it.bitmap?.width ?? 1920;
    const hh = it.meta?.height ?? it.bitmap?.height ?? 1080;
    const eye = eyeSizeFor(w, hh, this.opts.res);
    return layoutGeometry(s.layout, eye.w, eye.h);
  }

  async renderConfig() {
    if (this.running) return;
    const o = this.opts;
    const s = this.store.state;
    const it = this.item;
    const batch = this.items.length > 1;
    const isVideo = it.kind === 'video';
    const tabs = isVideo
      ? [
          { value: 'video', label: 'Video' },
          { value: 'frame', label: 'Still frame', disabled: batch }
        ]
      : [
          { value: 'still', label: 'Still image' },
          { value: 'motion', label: '3D motion video', disabled: batch }
        ];

    const layoutSel = selectEl(
      Object.entries(LAYOUTS).map(([value, l]) => ({ value, label: l.label })),
      s.layout,
      (v) => {
        this.store.set({ layout: v });
        this.renderConfig();
      }
    );

    const form = h('div', { class: 'form' });
    const spec = h('dl', { class: 'spec' });
    const warn = h('div', { class: 'warnbox', hidden: true });
    const geo = this.outputGeometry();
    let startLabel = 'Export';
    const specRows = [];

    if (o.tab === 'video') {
      this.resolvedCodec = undefined;
      const token = (this.probeToken = (this.probeToken ?? 0) + 1);
      const containers = Object.entries(CONTAINERS).map(([value, c]) => ({ value, label: c.label }));
      const codecHolder = h('div');
      form.append(
        field('Layout', layoutSel, LAYOUTS[s.layout]?.hint),
        h('div', { class: 'row2' }, field('Container', seg(containers, o.container, (v) => this.setOpt({ container: v, codec: 'auto' }))), field('Video codec', codecHolder)),
        field(
          'Quality',
          seg([...QUALITY_LEVELS.map((q) => ({ value: q.id, label: q.label })), { value: 'custom', label: 'Custom' }], o.quality, (v) => this.setOpt({ quality: v }))
        )
      );
      if (o.quality === 'custom') {
        const r = h('input', { type: 'range', class: 'range', min: 2, max: 200, step: 1, value: o.mbps });
        const lbl = h('span', { class: 'num' }, `${o.mbps} Mbps`);
        r.style.setProperty('--p', `${((o.mbps - 2) / 198) * 100}%`);
        r.addEventListener('input', () => {
          lbl.textContent = `${r.value} Mbps`;
          r.style.setProperty('--p', `${((r.value - 2) / 198) * 100}%`);
          this.opts.mbps = Number(r.value);
          saveOpts(this.opts);
        });
        form.append(h('div', { class: 'ctl' }, h('div', { class: 'ctl-head' }, h('span', { class: 'ctl-label' }, 'Bitrate'), lbl), r));
      }
      const srcW = it.meta?.width ?? 0;
      const srcH = it.meta?.height ?? 0;
      const resOpts = [{ value: 'source', label: batch ? 'Source' : `Source (${srcW}×${srcH} per eye)` }];
      for (const r of [2160, 1440, 1080, 720]) if (batch || Math.min(srcW, srcH) > r) resOpts.push({ value: String(r), label: `${r}p per eye` });
      const fpsOpts = [{ value: '0', label: batch ? 'Source' : `Source (${(it.meta?.fps ?? 30).toFixed(3).replace(/\.?0+$/, '')} fps)` }, ...[23.976, 24, 25, 29.97, 30, 50, 60].map((f) => ({ value: String(f), label: `${f} fps` }))];
      const hasRange = !batch && (this.app.range.in !== null || this.app.range.out !== null);
      form.append(
        h('div', { class: 'row2' }, field('Resolution', selectEl(resOpts, o.res, (v) => this.setOpt({ res: v }))), field('Frame rate', selectEl(fpsOpts, String(o.fps), (v) => this.setOpt({ fps: Number(v) })))),
        h(
          'div',
          { class: 'row2' },
          field('Range', seg([{ value: 'full', label: 'Full clip' }, { value: 'inout', label: 'In → Out', disabled: !hasRange }], hasRange ? o.range : 'full', (v) => this.setOpt({ range: v }))),
          field('Audio', seg([{ value: 'copy', label: 'Keep original', hint: 'Copied bit-exact when the container allows it' }, { value: 'none', label: 'Remove' }], o.audio, (v) => this.setOpt({ audio: v })))
        )
      );
      if (supportsDiskStreaming()) {
        const cb = h('input', { type: 'checkbox', role: 'switch' });
        cb.checked = o.toDisk;
        cb.addEventListener('change', () => this.setOpt({ toDisk: cb.checked }));
        form.append(
          h(
            'label',
            { class: 'ctl toggle', title: 'Writes the file progressively as it encodes — no memory limit for long videos' },
            h('span', { class: 'ctl-label' }, batch ? 'Save all files into a folder' : 'Stream directly to disk'),
            h('span', { class: 'switch' }, cb, h('span', { class: 'knob' }))
          )
        );
      }

      // Codec availability depends on frame size; probe asynchronously.
      const codecs = CONTAINERS[o.container].codecs;
      codecHolder.append(selectEl([{ value: 'auto', label: 'Auto (best available)' }, ...codecs.map((c) => ({ value: c, label: VIDEO_CODECS[c] }))], o.codec, (v) => this.setOpt({ codec: v })));
      probeEncoders(o.container, geo.canvasW, geo.canvasH).then((res) => {
        if (this.running || !this.dlg.open || token !== this.probeToken) return;
        const ok = res.filter((r) => r.ok).map((r) => r.codec);
        this.resolvedCodec = o.codec === 'auto' ? ok[0] ?? null : ok.includes(o.codec) ? o.codec : null;
        codecHolder.replaceChildren(
          selectEl(
            [{ value: 'auto', label: `Auto${ok[0] ? ` (${VIDEO_CODECS[ok[0]]})` : ''}` }, ...res.map((r) => ({ value: r.codec, label: `${VIDEO_CODECS[r.codec]}${r.ok ? '' : ' — unavailable'}`, disabled: !r.ok }))],
            o.codec,
            (v) => this.setOpt({ codec: v })
          )
        );
        spec.querySelector('[data-k="codec"]')?.replaceChildren(this.resolvedCodec ? VIDEO_CODECS[this.resolvedCodec] : 'None available');
        if (this.resolvedCodec && !batch) {
          const q = o.quality === 'custom' ? { mode: 'bitrate', mbps: o.mbps } : { level: o.quality };
          const bps = targetBitrate(q, geo.canvasW, geo.canvasH, o.fps || it.meta?.fps || 30, this.resolvedCodec);
          const est = (bps / 8) * span * 1.03 + (it.meta?.audio && o.audio !== 'none' ? 20000 * span : 0);
          spec.querySelector('[data-k="size"]')?.replaceChildren(`≈ ${formatBytes(est)} · ${(bps / 1e6).toFixed(1)} Mbps`);
          if (est > 1.5e9 && !o.toDisk) {
            warn.hidden = false;
            warn.textContent = supportsDiskStreaming()
              ? `Large output (≈ ${formatBytes(est)}). Turn on “Stream directly to disk” so the file is written as it encodes instead of being held in memory.`
              : `Large output (≈ ${formatBytes(est)}). This browser has to hold the whole file in memory — consider a lower resolution, a half-width layout or a shorter range.`;
          }
        }
        if (this.startBtn) this.startBtn.disabled = !this.resolvedCodec;
        if (!this.resolvedCodec) {
          warn.hidden = false;
          warn.textContent = `This browser can't encode ${geo.canvasW}×${geo.canvasH} with ${o.codec === 'auto' ? `any ${CONTAINERS[o.container].label} codec` : VIDEO_CODECS[o.codec]}. Try a lower resolution, a half-width layout, or a different container.`;
        }
      });

      const dur = it.meta?.duration ?? 0;
      const span = hasRange && o.range === 'inout' ? (this.app.range.out ?? dur) - (this.app.range.in ?? 0) : dur;
      specRows.push(
        ['Output', `${geo.canvasW} × ${geo.canvasH}`],
        ['Per eye', `${geo.logicalW} × ${geo.logicalH}${geo.eyeW !== geo.logicalW || geo.eyeH !== geo.logicalH ? ' (squeezed)' : ''}`],
        ['Codec', '…', 'codec'],
        ['Duration', batch ? `${this.items.length} files` : formatDuration(span)],
        ['Frames', batch ? '—' : `~${Math.round(span * (o.fps || it.meta?.fps || 30))}`],
        ['Audio', o.audio === 'none' ? 'Removed' : it.meta?.audio ? 'Original (copied when possible)' : 'None in source'],
        ['Est. size', batch ? '—' : '…', 'size'],
        ['File', batch ? 'One per clip' : outputName(it.name, s.layout, CONTAINERS[o.container].ext)]
      );
      const nImages = this.items.filter((i) => i.kind === 'image').length;
      const nVideos = this.items.length - nImages;
      if (batch && nImages) {
        specRows.push(['Photos', `${nImages} × ${o.still.toUpperCase()} still${nImages > 1 ? 's' : ''}`]);
      }
      startLabel = batch ? `Export ${nVideos} video${nVideos > 1 ? 's' : ''}${nImages ? ` + ${nImages} photo${nImages > 1 ? 's' : ''}` : ''}` : 'Start export';
    } else if (o.tab === 'frame' || o.tab === 'still') {
      form.append(
        field('Layout', layoutSel, LAYOUTS[s.layout]?.hint),
        field(
          'Format',
          seg(
            [
              { value: 'png', label: 'PNG' },
              { value: 'jpeg', label: 'JPEG' },
              { value: 'webp', label: 'WebP' },
              { value: 'jps', label: 'JPS', hint: 'Stereo JPEG (cross-view order) for 3D viewers' }
            ],
            o.still,
            (v) => this.setOpt({ still: v })
          ),
          o.still === 'jps' ? 'JPS always stores a full side-by-side pair with the right view first.' : null
        ),
        field('Scale', seg([{ value: 1, label: '100%' }, { value: 0.75, label: '75%' }, { value: 0.5, label: '50%' }], o.stillScale, (v) => this.setOpt({ stillScale: v })))
      );
      const w = (it.meta?.width ?? it.bitmap?.width) * o.stillScale;
      const hh = (it.meta?.height ?? it.bitmap?.height) * o.stillScale;
      const g = layoutGeometry(o.still === 'jps' ? 'sbs-full' : s.layout, w, hh);
      specRows.push(['Output', `${g.canvasW} × ${g.canvasH}`], ['Source', o.tab === 'frame' ? `Frame at ${timecode(this.app.currentTime(), it.meta?.fps)}` : 'Photo'], ['File', batch ? 'One per photo' : this.stillName()]);
      startLabel = batch ? `Export ${this.items.length} images` : 'Save image';
    } else {
      form.append(
        field(
          'Camera move',
          seg(
            Object.entries(MOTION_PATHS).map(([value, m]) => ({ value, label: m.label, hint: m.hint })),
            o.motionPath,
            (v) => this.setOpt({ motionPath: v })
          ),
          MOTION_PATHS[o.motionPath]?.hint
        )
      );
      const mk = (label, key, min, max, step, fmt) => {
        const r = h('input', { type: 'range', class: 'range', min, max, step, value: o[key] });
        const lbl = h('span', { class: 'num' }, fmt(o[key]));
        const upd = () => r.style.setProperty('--p', `${((r.value - min) / (max - min)) * 100}%`);
        upd();
        r.addEventListener('input', () => {
          this.opts[key] = Number(r.value);
          lbl.textContent = fmt(this.opts[key]);
          upd();
          saveOpts(this.opts);
        });
        return h('div', { class: 'ctl' }, h('div', { class: 'ctl-head' }, h('span', { class: 'ctl-label' }, label), lbl), r);
      };
      form.append(
        h('div', { class: 'row2' }, mk('Duration', 'motionSeconds', 2, 20, 0.5, (v) => `${v}s`), mk('Intensity', 'motionAmount', 0.2, 3, 0.05, (v) => `${v.toFixed(2)}×`)),
        h(
          'div',
          { class: 'row2' },
          field('Frame rate', seg([{ value: 24, label: '24' }, { value: 30, label: '30' }, { value: 60, label: '60' }], o.motionFps, (v) => this.setOpt({ motionFps: v }))),
          field('Max size', seg([{ value: 1080, label: '1080' }, { value: 1920, label: '1920' }, { value: 3840, label: '4K' }], o.motionSize, (v) => this.setOpt({ motionSize: v })))
        ),
        field('Container', seg(Object.entries(CONTAINERS).map(([value, c]) => ({ value, label: c.label })), o.container, (v) => this.setOpt({ container: v })))
      );
      specRows.push(['Move', MOTION_PATHS[o.motionPath].label], ['Length', `${o.motionSeconds}s @ ${o.motionFps} fps`], ['Loop', 'Seamless'], ['File', `${baseName(it.name)}.3d-${o.motionPath}.${CONTAINERS[o.container].ext}`]);
      startLabel = 'Render motion';
    }

    spec.replaceChildren(...specRows.flatMap(([k, v, key]) => [h('dt', {}, k), h('dd', { dataset: key ? { k: key } : undefined, title: v }, v)]));

    const thumb = this.thumb ? h('img', { class: 'export-thumb', src: this.thumb, alt: 'Output preview' }) : h('div', { class: 'export-thumb' });
    this.startBtn = h('button', { class: 'btn primary', onclick: () => this.start(), disabled: o.tab === 'video' && this.resolvedCodec === undefined }, icon('export'), startLabel);

    this.dlg.replaceChildren(
      h(
        'div',
        { class: 'modal-head' },
        h('h2', {}, batch ? `Batch export · ${this.items.length} items` : `Export · ${it.name}`),
        h('button', { class: 'icon-btn', title: 'Close', onclick: () => this.close() }, icon('x'))
      ),
      h(
        'div',
        { class: 'modal-body' },
        h('div', { class: 'tabs' }, seg(tabs, o.tab, (v) => this.setOpt({ tab: v }))),
        h('div', { class: 'export-grid' }, h('div', { class: 'export-preview' }, thumb, spec), h('div', { class: 'form' }, form, warn))
      ),
      h(
        'div',
        { class: 'modal-foot' },
        h('span', { class: 'note' }, 'Rendered locally on your GPU — nothing leaves this device.'),
        h('div', { class: 'actions' }, h('button', { class: 'btn', onclick: () => this.close() }, 'Cancel'), this.startBtn)
      )
    );
  }

  stillName() {
    const o = this.opts;
    const ext = o.still === 'jpeg' ? 'jpg' : o.still;
    const layout = o.still === 'jps' ? 'sbs-full' : this.store.state.layout;
    return outputName(this.item.name, layout, ext);
  }

  // ---------- Running ----------

  async start() {
    const o = this.opts;
    if (o.tab === 'video') return this.startVideo();
    if (o.tab === 'motion') return this.startMotion();
    return this.startStill();
  }

  progressView(title) {
    const bar = h('div', { class: 'bar-fill' });
    const canvas = h('canvas', { class: 'progress-canvas' });
    const stat = (label) => {
      const v = h('div', { class: 'metric-value' }, '—');
      const sub = h('div', { class: 'metric-sub' }, '');
      return { el: h('div', { class: 'metric' }, h('div', { class: 'metric-label' }, label), v, sub), v, sub };
    };
    const sProg = stat('Progress');
    const sFrames = stat('Frames');
    const sSpeed = stat('Speed');
    const sEta = stat('Remaining');
    const subtitle = h('div', { class: 'ctl-hint' }, '');
    const pauseBtn = h('button', { class: 'btn' }, 'Pause');
    const stopBtn = h('button', { class: 'btn', title: 'End the export here and save everything rendered so far as a playable file' }, icon('save'), 'Stop & save');
    const cancelBtn = h('button', { class: 'btn danger' }, 'Cancel');
    stopBtn.hidden = true;
    const note = h('span', { class: 'note' }, 'Settings are locked in for this render. Keep this tab in the foreground for best speed.');
    const actions = h('div', { class: 'actions' }, pauseBtn, stopBtn, cancelBtn);
    const foot = h('div', { class: 'modal-foot' }, note, actions);
    this.dlg.replaceChildren(
      h('div', { class: 'modal-head' }, h('h2', {}, title), h('span')),
      h('div', { class: 'modal-body' }, h('div', { class: 'progress-view' }, subtitle, canvas, h('div', { class: 'bar' }, bar), h('div', { class: 'stats' }, sProg.el, sFrames.el, sSpeed.el, sEta.el))),
      foot
    );
    const ctx = canvas.getContext('2d');
    let lastPaint = 0;
    return {
      bar,
      subtitle,
      pauseBtn,
      stopBtn,
      cancelBtn,
      /** Swap the footer for a question with choice buttons; any choice restores the normal footer. */
      confirm(message, choices) {
        const restore = () => foot.replaceChildren(note, actions);
        foot.replaceChildren(
          h('span', { class: 'note confirm-msg' }, icon('alert'), message),
          h(
            'div',
            { class: 'actions' },
            choices.map((c) =>
              h(
                'button',
                {
                  class: `btn${c.primary ? ' primary' : ''}${c.danger ? ' danger' : ''}`,
                  onclick: () => {
                    restore();
                    c.run();
                  }
                },
                c.label
              )
            )
          )
        );
      },
      setState(st) {
        pauseBtn.textContent = st === 'paused' ? 'Resume' : 'Pause';
        const finishing = st === 'finishing';
        for (const b of [pauseBtn, stopBtn, cancelBtn]) b.disabled = finishing;
        if (finishing) foot.replaceChildren(h('span', { class: 'note' }, 'Finalising the file with the frames rendered so far…'), actions);
      },
      paint(src, force = false) {
        const now = performance.now();
        if (!force && now - lastPaint < 200) return;
        lastPaint = now;
        const k = Math.min(1, 720 / src.width);
        const w = Math.round(src.width * k);
        const hh = Math.round(src.height * k);
        if (canvas.width !== w || canvas.height !== hh) {
          canvas.width = w;
          canvas.height = hh;
        }
        ctx.drawImage(src, 0, 0, w, hh);
      },
      set(p) {
        bar.style.width = `${(p.progress * 100).toFixed(1)}%`;
        sProg.v.textContent = `${(p.progress * 100).toFixed(1)}%`;
        sProg.sub.textContent = p.time !== undefined ? timecode(p.time, 30).slice(0, 8) : '';
        sFrames.v.textContent = p.frames !== undefined ? String(p.frames) : '—';
        sFrames.sub.textContent = p.totalFrames ? `of ~${p.totalFrames}` : '';
        sSpeed.v.textContent = p.fps ? `${p.fps.toFixed(1)} fps` : '—';
        sSpeed.sub.textContent = p.realtime ? `${p.realtime.toFixed(2)}× realtime` : '';
        sEta.v.textContent = p.eta !== null && p.eta !== undefined ? formatDuration(p.eta) : '—';
        sEta.sub.textContent = p.elapsed !== undefined ? `elapsed ${formatDuration(p.elapsed)}` : '';
      }
    };
  }

  doneView({ title, lines, files = [], warnings = [] }) {
    this.dlg.replaceChildren(
      h('div', { class: 'modal-head' }, h('h2', {}, title), h('button', { class: 'icon-btn', title: 'Close', onclick: () => this.close() }, icon('x'))),
      h(
        'div',
        { class: 'modal-body' },
        h('div', { class: 'verdict ok', style: { marginBottom: '14px' } }, icon('check'), h('span', {}, lines[0])),
        h('dl', { class: 'spec' }, lines.slice(1).flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)])),
        files.length
          ? h(
              'div',
              { class: 'inline-actions', style: { marginTop: '14px' } },
              files.map((f) => h('button', { class: 'btn', onclick: () => download(f.blob, f.name) }, icon('download'), f.name))
            )
          : null,
        warnings.length ? h('div', { class: 'warnbox', style: { marginTop: '14px' } }, warnings.join(' ')) : null
      ),
      h('div', { class: 'modal-foot' }, h('span', { class: 'note' }, ''), h('div', { class: 'actions' }, h('button', { class: 'btn', onclick: () => this.renderConfig() }, 'Export again'), h('button', { class: 'btn primary', onclick: () => this.close() }, 'Done')))
    );
  }

  failView(err) {
    this.dlg.replaceChildren(
      h('div', { class: 'modal-head' }, h('h2', {}, 'Export failed'), h('button', { class: 'icon-btn', onclick: () => this.close() }, icon('x'))),
      h('div', { class: 'modal-body' }, h('div', { class: 'verdict bad' }, icon('alert'), h('span', {}, String(err?.message || err)))),
      h('div', { class: 'modal-foot' }, h('span'), h('div', { class: 'actions' }, h('button', { class: 'btn', onclick: () => this.renderConfig() }, 'Back'), h('button', { class: 'btn primary', onclick: () => this.close() }, 'Close')))
    );
  }

  async startVideo() {
    const o = this.opts;
    const s = { ...this.store.state };
    const batch = this.items.filter((i) => i.kind === 'video');
    let codec = o.codec === 'auto' ? this.resolvedCodec : o.codec;
    if (!codec) return;

    // Destination pickers must run inside the click gesture.
    let dirHandle = null;
    let fileHandle = null;
    const ext = CONTAINERS[o.container].ext;
    try {
      if (o.toDisk && supportsDiskStreaming()) {
        if (batch.length > 1 && window.showDirectoryPicker) dirHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
        else fileHandle = await pickSaveFile(outputName(batch[0].name, s.layout, ext), o.container);
      }
    } catch {
      return; // user dismissed the picker
    }

    this.running = true;
    this.app.pausePlayback();
    const ui = this.progressView(batch.length > 1 ? 'Batch export' : 'Exporting video');
    const results = [];
    const warnings = [];
    let cancelled = false;
    let stopEarly = false;
    const t0 = performance.now();
    try {
      await this.app.ensureEngine();
      for (let i = 0; i < batch.length; i++) {
        const it = batch[i];
        const meta = it.meta;
        if (!meta) {
          warnings.push(`${it.name}: unreadable, skipped.`);
          continue;
        }
        const eye = eyeSizeFor(meta.width, meta.height, o.res);
        const geo = layoutGeometry(s.layout, eye.w, eye.h);
        if (batch.length > 1) {
          const ok = (await probeEncoders(o.container, geo.canvasW, geo.canvasH)).filter((r) => r.ok).map((r) => r.codec);
          codec = o.codec === 'auto' ? ok[0] : ok.includes(o.codec) ? o.codec : null;
          if (!codec) {
            warnings.push(`${it.name}: no encoder for ${geo.canvasW}×${geo.canvasH}, skipped.`);
            continue;
          }
        }
        const name = outputName(it.name, s.layout, ext);
        let handle = fileHandle;
        if (dirHandle) handle = await dirHandle.getFileHandle(name, { create: true });
        const trim = batch.length === 1 && o.range === 'inout' && (this.app.range.in !== null || this.app.range.out !== null) ? { start: this.app.range.in ?? 0, end: this.app.range.out ?? meta.duration } : null;
        const span = trim ? trim.end - trim.start : meta.duration;
        const totalFrames = Math.round(span * (o.fps || meta.fps || 30));
        ui.subtitle.textContent = `${batch.length > 1 ? `[${i + 1}/${batch.length}] ` : ''}${it.name} → ${name} · ${geo.canvasW}×${geo.canvasH} ${VIDEO_CODECS[codec]}`;
        const job = new VideoExportJob({
          file: it.file,
          engine: this.app.engine,
          settings: s,
          eyeW: eye.w,
          eyeH: eye.h,
          container: o.container,
          codec,
          quality: o.quality === 'custom' ? { mode: 'bitrate', mbps: o.mbps } : { level: o.quality },
          frameRate: o.fps || null,
          sourceFps: meta.fps,
          audio: o.audio,
          trim,
          fileHandle: handle,
          onFrame: ({ canvas }) => ui.paint(canvas),
          onProgress: (p) => ui.set({ ...p, totalFrames, realtime: p.elapsed > 0 ? p.time / p.elapsed : 0 }),
          onState: (st) => ui.setState(st)
        });
        this.job = job;
        ui.stopBtn.hidden = false;
        ui.pauseBtn.onclick = () => (job.paused ? job.resume() : job.pause());
        ui.stopBtn.onclick = () => {
          stopEarly = true;
          job.stopAndSave();
        };
        ui.cancelBtn.onclick = () => {
          if (!job.frames) {
            cancelled = true;
            job.cancel();
            return;
          }
          // Work is at stake: pause while asking, so "save" keeps exactly what was on screen.
          const wasPaused = job.paused;
          if (!wasPaused) job.pause();
          ui.confirm(`Discard ${job.frames.toLocaleString()} rendered frame${job.frames > 1 ? 's' : ''} (${formatSpan(job.renderedEnd)} of video)?`, [
            { label: 'Keep rendering', run: () => !wasPaused && job.resume() },
            {
              label: 'Save what’s done',
              primary: true,
              run: () => {
                stopEarly = true;
                job.stopAndSave();
              }
            },
            {
              label: 'Discard',
              danger: true,
              run: () => {
                cancelled = true;
                job.cancel();
              }
            }
          ]);
        };
        const res = await job.run();
        const outName = res.partial ? outputName(`${baseName(it.name)} (partial)`, s.layout, ext) : name;
        warnings.push(...res.warnings.map((w) => `${it.name}: ${w}`));
        results.push({ ...res, name: res.name ?? outName, source: it, span });
        if (res.blob) download(res.blob, outName);
        if (stopEarly) break;
      }
      const photos = stopEarly ? [] : this.items.filter((i) => i.kind === 'image');
      ui.stopBtn.hidden = true;
      for (let i = 0; i < photos.length; i++) {
        const it = photos[i];
        ui.subtitle.textContent = `[photo ${i + 1}/${photos.length}] ${it.name}`;
        const bmp = await this.app.bitmapFor(it);
        if (!bmp) continue;
        const blob = await renderStill({ source: bmp, width: bmp.width, height: bmp.height, settings: s, engine: this.app.engine, format: o.still, scale: o.stillScale });
        const ext = o.still === 'jpeg' ? 'jpg' : o.still;
        const name = outputName(it.name, o.still === 'jps' ? 'sbs-full' : s.layout, ext);
        if (dirHandle) {
          const fh = await dirHandle.getFileHandle(name, { create: true });
          const w = await fh.createWritable();
          await w.write(blob);
          await w.close();
          results.push({ bytes: blob.size, frames: 0, name, streamed: true });
        } else {
          download(blob, name);
          results.push({ blob, bytes: blob.size, frames: 0, name });
        }
        ui.set({ progress: (i + 1) / photos.length, frames: i + 1 });
      }
      this.running = false;
      const bytes = results.reduce((a, r) => a + r.bytes, 0);
      const frames = results.reduce((a, r) => a + r.frames, 0);
      const elapsed = (performance.now() - t0) / 1000;
      const last = results[results.length - 1];
      let headline;
      if (last?.partial && results.length === 1) {
        headline = `Saved the first ${formatSpan(last.rendered)} of ${formatSpan(last.span)} as ${last.name}${last.streamed ? ' on disk' : ''}.`;
      } else if (stopEarly) {
        headline = `Stopped early: ${results.length} file${results.length > 1 ? 's' : ''} saved${last?.partial ? `, the last one partial (${formatSpan(last.rendered)})` : ''}.`;
      } else {
        headline = results.length > 1 ? `${results.length} files exported.` : `Saved ${results[0]?.name ?? ''}${results[0]?.streamed ? ' to disk' : ''}.`;
      }
      this.doneView({
        title: stopEarly ? 'Partial export saved' : 'Export complete',
        lines: [
          headline,
          ['Total size', formatBytes(bytes)],
          ['Frames', String(frames)],
          ['Time', formatDuration(elapsed)],
          ['Throughput', `${(frames / Math.max(0.001, elapsed)).toFixed(1)} fps`]
        ],
        files: results.filter((r) => r.blob).map((r) => ({ blob: r.blob, name: r.name })),
        warnings
      });
      this.app.toast({ kind: 'ok', title: stopEarly ? 'Partial export saved' : 'Export complete', msg: `${results.length} file${results.length > 1 ? 's' : ''} · ${formatBytes(bytes)}` });
    } catch (err) {
      this.running = false;
      if (cancelled || err instanceof ExportCancelled) {
        this.renderConfig();
        this.app.toast({ kind: 'warn', title: 'Export cancelled' });
      } else {
        console.error(err);
        this.failView(err);
      }
    } finally {
      this.running = false;
      this.job = null;
    }
  }

  async startStill() {
    const o = this.opts;
    const s = { ...this.store.state };
    const list = this.opts.tab === 'frame' ? [this.item] : this.items.filter((i) => i.kind === 'image');
    this.running = true;
    const ui = this.progressView('Rendering');
    ui.pauseBtn.hidden = true;
    ui.cancelBtn.hidden = true;
    const files = [];
    try {
      await this.app.ensureEngine();
      for (let i = 0; i < list.length; i++) {
        const it = list[i];
        ui.subtitle.textContent = `${it.name}`;
        let source;
        let w;
        let hh;
        if (this.opts.tab === 'frame') {
          source = this.app.video;
          w = source.videoWidth;
          hh = source.videoHeight;
        } else {
          source = await this.app.bitmapFor(it);
          w = source.width;
          hh = source.height;
        }
        const blob = await renderStill({ source, width: w, height: hh, settings: s, engine: this.app.engine, format: o.still, scale: o.stillScale });
        this.item = it;
        const name = this.stillName();
        files.push({ blob, name });
        download(blob, name);
        ui.set({ progress: (i + 1) / list.length, frames: i + 1 });
      }
      this.running = false;
      this.doneView({ title: 'Saved', lines: [`${files.length} image${files.length > 1 ? 's' : ''} saved.`, ['Size', formatBytes(files.reduce((a, f) => a + f.blob.size, 0))]], files });
    } catch (err) {
      this.running = false;
      console.error(err);
      this.failView(err);
    } finally {
      this.running = false;
    }
  }

  async startMotion() {
    const o = this.opts;
    const s = { ...this.store.state };
    const it = this.item;
    this.running = true;
    const ui = this.progressView('Rendering 3D motion');
    ui.pauseBtn.hidden = true;
    let cancelled = false;
    ui.cancelBtn.onclick = () => (cancelled = true);
    const t0 = performance.now();
    try {
      await this.app.ensureEngine();
      const bmp = await this.app.bitmapFor(it);
      const size = Math.min(o.motionSize, Math.max(bmp.width, bmp.height));
      const probe = await probeEncoders(o.container, Math.round(size / 2) * 2, Math.round(size / 2) * 2);
      const codec = probe.find((r) => r.ok)?.codec;
      if (!codec) throw new Error(`No ${CONTAINERS[o.container].label} encoder is available in this browser.`);
      const total = Math.round(o.motionSeconds * o.motionFps);
      const blob = await exportMotion({
        source: bmp,
        width: bmp.width,
        height: bmp.height,
        settings: s,
        engine: this.app.engine,
        path: o.motionPath,
        seconds: o.motionSeconds,
        fps: o.motionFps,
        amount: o.motionAmount,
        container: o.container,
        codec,
        quality: { level: 'very-high' },
        maxSize: o.motionSize,
        isCancelled: () => cancelled,
        onProgress: (p) => {
          const el = (performance.now() - t0) / 1000;
          ui.set({ progress: p, frames: Math.round(p * total), totalFrames: total, elapsed: el, eta: p > 0 ? (el / p) * (1 - p) : null, fps: (p * total) / Math.max(0.001, el) });
        }
      });
      const name = `${baseName(it.name)}.3d-${o.motionPath}.${CONTAINERS[o.container].ext}`;
      download(blob, name);
      this.running = false;
      this.doneView({ title: '3D motion ready', lines: [`Saved ${name}.`, ['Size', formatBytes(blob.size)], ['Codec', VIDEO_CODECS[codec]], ['Length', `${o.motionSeconds}s`]], files: [{ blob, name }] });
    } catch (err) {
      this.running = false;
      if (cancelled || err instanceof ExportCancelled) this.renderConfig();
      else {
        console.error(err);
        this.failView(err);
      }
    } finally {
      this.running = false;
    }
  }
}
