import { h } from '../ui/dom.js';
import { icon } from '../ui/icons.js';
import { slider, segmented, select, toggle, section } from '../ui/controls.js';
import { DepthHistogram } from '../ui/widgets.js';
import { DEFAULTS, PRESETS, LAYOUTS, curveJs } from '../core/settings.js';
import { MODELS, DETAIL_LEVELS } from '../depth/models.js';
import { ANAGLYPH } from '../gl/renderer.js';
import { comfortLimitPercent, screenWidthMm } from '../core/pipeline.js';
import { formatBytes } from '../ui/dom.js';

const PRESET_KEYS = ['strength', 'convergence', 'gamma', 'dilate', 'smooth'];
const CUSTOM_KEY = 'parallaxer.presets.v1';

function loadCustomPresets() {
  try {
    return JSON.parse(localStorage.getItem(CUSTOM_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveCustomPresets(list) {
  try {
    localStorage.setItem(CUSTOM_KEY, JSON.stringify(list));
  } catch {
    /* ignore */
  }
}

// Settings captured by a user preset (everything that shapes the look, nothing about the session).
const LOOK_KEYS = [
  'strength', 'convergence', 'autoConvergence', 'eyes', 'gamma', 'near', 'far', 'invert', 'edgeSnap', 'edgeSigma',
  'snapRadius', 'dilate', 'smooth', 'preserve', 'fill', 'tear', 'soften', 'quality', 'temporal', 'cutSensitivity'
];

export function buildInspector(root, store, app) {
  const controls = [];
  const add = (c) => {
    controls.push(c);
    return c.el;
  };
  const d = (k) => DEFAULTS[k];

  // ---------- Look / presets ----------
  const presetGrid = h('div', { class: 'presets' });
  const renderPresets = () => {
    const s = store.state;
    const all = [...PRESETS.map((p) => ({ ...p, custom: false })), ...loadCustomPresets().map((p) => ({ ...p, custom: true }))];
    presetGrid.replaceChildren(
      ...all.map((p) => {
        const on = Object.entries(p.values).every(([k, v]) => (typeof v === 'number' ? Math.abs(s[k] - v) < 1e-6 : s[k] === v));
        const btn = h(
          'button',
          {
            type: 'button',
            class: `preset${on ? ' on' : ''}${p.custom ? ' custom' : ''}`,
            title: p.hint ?? 'Custom preset',
            onclick: () => store.set({ ...p.values })
          },
          p.label
        );
        if (p.custom) {
          const del = h('span', { class: 'preset-del', title: 'Delete preset', role: 'button' }, '×');
          del.addEventListener('click', (e) => {
            e.stopPropagation();
            saveCustomPresets(loadCustomPresets().filter((c) => c.id !== p.id));
            renderPresets();
          });
          btn.appendChild(del);
        }
        return btn;
      })
    );
  };
  controls.push({ el: presetGrid, keys: [...PRESET_KEYS, ...LOOK_KEYS], sync: renderPresets });
  renderPresets();

  const savePreset = h('button', { type: 'button', class: 'sec-action', title: 'Save the current look as a preset' }, 'Save');
  savePreset.addEventListener('click', () => {
    const name = prompt('Preset name', 'My look');
    if (!name) return;
    const values = Object.fromEntries(LOOK_KEYS.map((k) => [k, store.state[k]]));
    saveCustomPresets([...loadCustomPresets(), { id: `u${Date.now()}`, label: name.slice(0, 18), values }]);
    renderPresets();
    app.toast({ kind: 'ok', title: 'Preset saved', msg: `“${name}” is now in your preset list.` });
  });
  const resetAll = h('button', { type: 'button', class: 'sec-action', title: 'Reset every look setting to its default' }, 'Reset');
  resetAll.addEventListener('click', () => store.set(Object.fromEntries(LOOK_KEYS.map((k) => [k, DEFAULTS[k]]))));

  // ---------- Depth engine ----------
  const modelList = h('div', { class: 'models' });
  const renderModels = async () => {
    const cur = store.state.model;
    const cached = await Promise.all(MODELS.map((m) => app.isModelCached(m)));
    modelList.replaceChildren(
      ...MODELS.map((m, i) =>
        h(
          'button',
          { type: 'button', class: `model${m.id === cur ? ' on' : ''}`, onclick: () => store.set({ model: m.id }) },
          h('span', { class: 'model-name' }, m.name),
          h('span', { class: `model-tag ${m.tag.toLowerCase()}` }, m.tag),
          h('span', { class: 'model-desc' }, m.description),
          h(
            'span',
            { class: 'model-meta' },
            h('span', {}, m.local ? 'Bundled' : `~${formatBytes(app.gpuF16 && m.sizeFp16 ? m.sizeFp16 : m.size)}`),
            h('span', {}, m.license),
            m.local || cached[i] ? h('span', { class: 'cached' }, m.local ? '● Local' : '● Cached') : null
          )
        )
      )
    );
  };
  controls.push({ el: modelList, keys: ['model'], sync: renderModels });
  renderModels();

  const customInput = h('input', { type: 'file', accept: '.onnx', hidden: true });
  customInput.addEventListener('change', () => {
    const f = customInput.files?.[0];
    customInput.value = '';
    if (f) app.loadCustomModel(f);
  });
  const customBox = h('div', { class: 'custom-model' });
  const renderCustom = () => {
    const cm = app.customModel;
    const on = store.state.model === 'custom';
    const children = [
      cm
        ? h(
            'button',
            { type: 'button', class: `model${on ? ' on' : ''}`, onclick: () => store.set({ model: 'custom' }) },
            h('span', { class: 'model-name' }, cm.name),
            h('span', { class: 'model-tag' }, 'Custom'),
            h('span', { class: 'model-meta' }, h('span', {}, formatBytes(cm.size)), h('span', {}, 'This session only'))
          )
        : null,
      h('button', { type: 'button', class: 'btn subtle block', onclick: () => customInput.click(), title: 'Use any ONNX depth network (NCHW RGB input, single depth output)' }, icon('plus'), cm ? 'Replace custom ONNX model…' : 'Load custom ONNX model…'),
      customInput
    ];
    customBox.replaceChildren(...children.filter(Boolean));
  };
  app.onCustomModel = renderCustom;
  controls.push({ el: customBox, keys: ['model'], sync: renderCustom });
  renderCustom();

  const engineStatus = h('dl', { class: 'engine-status' });
  const renderEngine = () => {
    const e = app.engineInfo();
    const rows = [
      ['Backend', e.backend],
      ['Precision', e.precision],
      ['Network input', e.input],
      ['Inference', e.ms]
    ];
    engineStatus.replaceChildren(...rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v ?? '—')]));
  };
  app.onEngineInfo = renderEngine;
  renderEngine();

  // Options row for custom networks, shown only while a custom model is selected.
  const customRow = (...args) => {
    const row = h(...args);
    controls.push({ el: row, keys: ['model'], sync: () => (row.hidden = store.state.model !== 'custom') });
    row.hidden = store.state.model !== 'custom';
    return row;
  };

  const depthSec = section('engine', 'Depth engine', 'cpu', [
    modelList,
    customBox,
    customRow(
      'div',
      { class: 'row2' },
      add(
        segmented(store, 'customNorm', {
          label: 'Input range',
          options: [
            { value: 'imagenet', label: 'ImageNet', hint: '(rgb − mean) / std' },
            { value: 'none', label: '0 – 1', hint: 'Plain RGB in [0, 1]' }
          ],
          disabled: (s) => s.model !== 'custom',
          deps: ['model']
        })
      ),
      add(
        segmented(store, 'customMultiple', {
          label: 'Size multiple',
          options: [
            { value: 14, label: '14', hint: 'Vision transformers (DINOv2 / Depth Anything)' },
            { value: 32, label: '32', hint: 'Convolutional encoders' }
          ],
          disabled: (s) => s.model !== 'custom',
          deps: ['model']
        })
      )
    ),
    add(
      segmented(store, 'detail', {
        label: 'Detail',
        hint: 'Network resolution for models with flexible input',
        options: DETAIL_LEVELS.map((l) => ({ value: l.id, label: l.label, hint: `${l.short}px short side` })),
        disabled: (s) => (s.model === 'custom' ? !!app.customFixed : !!MODELS.find((m) => m.id === s.model)?.input.fixed),
        deps: ['model']
      })
    ),
    add(
      segmented(store, 'backend', {
        label: 'Compute',
        options: [
          { value: 'auto', label: 'Auto', hint: 'WebGPU when available, otherwise CPU' },
          { value: 'webgpu', label: 'WebGPU', hint: 'Force GPU inference' },
          { value: 'wasm', label: 'CPU', hint: 'WebAssembly (SIMD, multi-threaded when isolated)' }
        ]
      })
    ),
    add(toggle(store, 'smoothPlayback', { label: 'Smooth playback', hint: 'While playing, show every frame with the latest depth instead of waiting for inference. Paused frames and exports are always frame-exact.' })),
    engineStatus
  ]);

  // ---------- Stereo ----------
  const stereoSec = section('stereo', 'Stereo', 'glasses', [
    add(slider(store, 'strength', { label: 'Depth budget', min: 0, max: 10, step: 0.1, unit: '%', defaultValue: d('strength'), hint: 'Total parallax range as a percentage of the frame width' })),
    add(
      slider(store, 'convergence', {
        label: 'Screen plane',
        min: 0,
        max: 1,
        step: 0.01,
        defaultValue: d('convergence'),
        hint: 'Depth that sits exactly on the screen. Nearer content pops out, farther content recedes.',
        disabled: (s) => s.autoConvergence,
        deps: ['autoConvergence']
      })
    ),
    add(toggle(store, 'autoConvergence', { label: 'Auto-converge on subject', hint: 'Continuously places the main subject on the screen plane' })),
    add(
      segmented(store, 'eyes', {
        label: 'View synthesis',
        options: [
          { value: 'symmetric', label: 'Symmetric', hint: 'Both eyes synthesised — halves artefacts per eye' },
          { value: 'left', label: 'Left = source', hint: 'Left eye is the untouched original' },
          { value: 'right', label: 'Right = source', hint: 'Right eye is the untouched original' }
        ]
      })
    ),
    add(toggle(store, 'swap', { label: 'Swap eyes (cross-view)', hint: 'Right image first — for cross-eyed free viewing' }))
  ]);

  // ---------- Depth shaping ----------
  const histo = new DepthHistogram(store);
  app.onDepthStats = (stats, conv) => {
    histo.update(stats?.hist ?? null, conv);
    renderComfort(stats, conv);
  };
  controls.push({ el: histo.el, keys: ['convergence', 'gamma', 'near', 'far', 'invert'], sync: () => histo.draw() });

  const shapeSec = section('shape', 'Depth shaping', 'layers', [
    histo.el,
    add(slider(store, 'gamma', { label: 'Depth curve', min: 0.3, max: 3, step: 0.01, defaultValue: d('gamma'), hint: '<1 expands the foreground, >1 expands the background' })),
    h(
      'div',
      { class: 'row2' },
      add(slider(store, 'far', { label: 'Far clip', min: 0, max: 0.9, step: 0.01, defaultValue: d('far'), hint: 'Everything farther is flattened to the background plane' })),
      add(slider(store, 'near', { label: 'Near clip', min: 0.1, max: 1, step: 0.01, defaultValue: d('near'), hint: 'Everything nearer is flattened to the front plane' }))
    ),
    add(toggle(store, 'invert', { label: 'Invert depth', hint: 'For models or sources that encode distance instead of nearness' }))
  ]);

  // ---------- Edges & occlusion ----------
  const edgeSec = section('edges', 'Edges & occlusion', 'wand', [
    add(slider(store, 'edgeSnap', { label: 'Edge snap', min: 0, max: 1, step: 0.01, defaultValue: d('edgeSnap'), hint: 'How strongly depth edges lock onto colour edges' })),
    add(slider(store, 'snapRadius', { label: 'Snap reach', min: 0, max: 20, step: 0.5, unit: '‰', defaultValue: d('snapRadius'), hint: 'Search radius for edge snapping, per-mille of width' })),
    h(
      'div',
      { class: 'row2' },
      add(slider(store, 'dilate', { label: 'Edge grow', min: 0, max: 8, step: 0.1, unit: '‰', defaultValue: d('dilate'), hint: 'Grows foreground depth outward to prevent halos' })),
      add(slider(store, 'smooth', { label: 'Smoothing', min: 0, max: 12, step: 0.1, unit: '‰', defaultValue: d('smooth'), hint: 'Depth blur radius — reduces warping of fine detail' }))
    ),
    add(slider(store, 'preserve', { label: 'Preserve discontinuities', min: 0, max: 1, step: 0.01, defaultValue: d('preserve'), hint: 'Keeps smoothing from bleeding across depth edges' })),
    add(
      segmented(store, 'fill', {
        label: 'Occlusion fill',
        options: [
          { value: 'mirror', label: 'Texture mirror', hint: 'Reflects background texture into revealed areas' },
          { value: 'stretch', label: 'Edge stretch', hint: 'Extends the background edge pixel' }
        ]
      })
    ),
    h(
      'div',
      { class: 'row2' },
      add(slider(store, 'tear', { label: 'Tear threshold', min: 1.2, max: 12, step: 0.1, format: (v) => `${v.toFixed(1)}×`, defaultValue: d('tear'), hint: 'Stretch factor beyond which surfaces tear into holes' })),
      add(slider(store, 'soften', { label: 'Fill soften', min: 0, max: 1, step: 0.01, defaultValue: d('soften'), hint: 'Blurs filled regions across the streak direction' }))
    ),
    add(
      segmented(store, 'quality', {
        label: 'Ray-march quality',
        options: [
          { value: 'draft', label: 'Draft', hint: '2 px steps' },
          { value: 'standard', label: 'Standard', hint: '1 px steps' },
          { value: 'high', label: 'High', hint: '½ px steps' }
        ]
      })
    )
  ]);

  // ---------- Temporal ----------
  const temporalSec = section('temporal', 'Temporal stability', 'film', [
    add(slider(store, 'temporal', { label: 'Stabilisation', min: 0, max: 1, step: 0.01, defaultValue: d('temporal'), hint: 'Motion-gated temporal filtering of depth and its range — removes flicker' })),
    add(slider(store, 'cutSensitivity', { label: 'Scene-cut sensitivity', min: 0, max: 1, step: 0.01, defaultValue: d('cutSensitivity'), hint: 'Higher resets temporal state on subtler cuts' }))
  ]);

  // ---------- Output ----------
  const layoutGroups = [
    { group: 'Stereo pairs', items: ['sbs-full', 'sbs-half', 'tb-full', 'tb-half'] },
    { group: 'Glasses & displays', items: ['anaglyph', 'rows', 'columns', 'checker'] },
    { group: 'Depth data', items: ['rgbd', 'depth'] }
  ].map((g) => ({ group: g.group, items: g.items.map((id) => ({ value: id, label: LAYOUTS[id].label })) }));

  const outputSec = section('output', 'Output format', 'export', [
    add(select(store, 'layout', { label: 'Layout', options: layoutGroups, describe: (s) => LAYOUTS[s.layout]?.hint })),
    add(
      select(store, 'anaglyph', {
        label: 'Anaglyph method',
        options: Object.entries(ANAGLYPH).map(([value, a]) => ({ value, label: a.label })),
        hidden: (s) => s.layout !== 'anaglyph' && s.view !== 'anaglyph',
        deps: ['layout', 'view']
      })
    ),
    add(toggle(store, 'floatingWindow', { label: 'Floating window', hint: 'Dynamic edge masks that stop pop-out objects being clipped by the frame (window violations)' })),
    add(
      segmented(store, 'colormap', {
        label: 'Depth colour map',
        options: [
          { value: 'gray', label: 'Gray' },
          { value: 'turbo', label: 'Turbo' },
          { value: 'magma', label: 'Magma' }
        ]
      })
    )
  ]);

  // ---------- Comfort analyzer ----------
  const comfortBox = h('div', { class: 'comfort' });
  let lastStats = null;
  let lastConv = null;
  function renderComfort(stats = lastStats, conv = lastConv) {
    lastStats = stats;
    lastConv = conv;
    const s = store.state;
    const S = s.strength / 100;
    const c = conv ?? s.convergence;
    let far = 0;
    let near = 1;
    if (stats?.hist) {
      const hist = stats.hist;
      const total = hist.reduce((a, b) => a + b, 0) || 1;
      let acc = 0;
      let lo = null;
      let hi = null;
      for (let i = 0; i < hist.length; i++) {
        acc += hist[i];
        if (lo === null && acc / total >= 0.01) lo = (i + 0.5) / hist.length;
        if (hi === null && acc / total >= 0.99) hi = (i + 0.5) / hist.length;
      }
      far = curveJs(lo ?? 0, s);
      near = curveJs(hi ?? 1, s);
    }
    const behind = Math.max(0, S * (c - far)) * 100;
    const front = Math.max(0, -S * (c - near)) * 100;
    const widthMm = screenWidthMm(s);
    const limit = comfortLimitPercent(s);
    const behindMm = (behind / 100) * widthMm;
    const frontMm = (front / 100) * widthMm;
    let verdict;
    if (behind > limit) {
      verdict = h('div', { class: 'verdict bad' }, icon('alert'), h('span', {}, `Divergence: background separation ${behindMm.toFixed(0)} mm exceeds the ~63 mm eye distance on a ${s.screenInches}" screen. Lower the depth budget or raise the screen plane.`));
    } else if (front > 3.5 || behind > limit * 0.85) {
      verdict = h('div', { class: 'verdict warn' }, icon('alert'), h('span', {}, front > 3.5 ? 'Strong pop-out. Fine for effect shots, tiring over long viewing.' : 'Close to the divergence limit for this screen size.'));
    } else {
      verdict = h('div', { class: 'verdict ok' }, icon('check'), h('span', {}, `Comfortable on a ${s.screenInches}" display.`));
    }
    comfortBox.replaceChildren(
      h('div', { class: 'metric' }, h('div', { class: 'metric-label' }, 'Behind screen'), h('div', { class: 'metric-value' }, `+${behind.toFixed(2)}%`), h('div', { class: 'metric-sub' }, `${behindMm.toFixed(1)} mm · limit ${limit.toFixed(2)}%`)),
      h('div', { class: 'metric' }, h('div', { class: 'metric-label' }, 'In front'), h('div', { class: 'metric-value' }, `−${front.toFixed(2)}%`), h('div', { class: 'metric-sub' }, `${frontMm.toFixed(1)} mm`)),
      verdict
    );
  }
  controls.push({ el: comfortBox, keys: ['strength', 'convergence', 'gamma', 'near', 'far', 'invert', 'screenInches'], sync: () => renderComfort() });
  renderComfort();

  const comfortSec = section('comfort', 'Comfort analyser', 'eye', [
    add(slider(store, 'screenInches', { label: 'Target screen', min: 5, max: 300, step: 1, format: (v) => `${v}"`, defaultValue: d('screenInches'), hint: 'Diagonal of the display you will watch on (16:9)' })),
    comfortBox
  ]);

  root.replaceChildren(
    section('look', 'Look', 'sparkles', [presetGrid], { actions: [savePreset, resetAll] }),
    depthSec,
    stereoSec,
    shapeSec,
    edgeSec,
    temporalSec,
    outputSec,
    comfortSec
  );

  store.subscribe((_, changed) => {
    for (const c of controls) if (c.keys.some((k) => changed.includes(k))) c.sync();
  });

  return {
    refreshModels: renderModels,
    refreshEngine: () => {
      renderEngine();
      for (const c of controls) if (c.keys.includes('model')) c.sync();
    }
  };
}
