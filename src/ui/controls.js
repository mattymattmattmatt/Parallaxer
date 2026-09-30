import { h, clamp } from './dom.js';
import { icon } from './icons.js';

/**
 * Store-bound inspector controls. Each factory returns { el, keys, sync } — the inspector calls sync()
 * whenever one of `keys` changes, so controls stay correct across undo/redo, presets and shortcuts.
 */

function decimalsOf(step) {
  const s = String(step);
  return s.includes('.') ? s.split('.')[1].length : 0;
}

export function slider(store, key, o) {
  const decimals = o.decimals ?? decimalsOf(o.step);
  const fmt = o.format ?? ((v) => `${Number(v).toFixed(decimals)}${o.unit ?? ''}`);
  const range = h('input', { type: 'range', class: 'range', min: o.min, max: o.max, step: o.step, 'aria-label': o.label });
  const num = h('input', { class: 'num', type: 'text', inputmode: 'decimal', spellcheck: 'false', 'aria-label': `${o.label} value` });
  const el = h(
    'div',
    { class: 'ctl slider', title: o.hint ?? '' },
    h('div', { class: 'ctl-head' }, h('span', { class: 'ctl-label' }, o.label, h('i', { class: 'mod-dot' })), num),
    range
  );

  const setValue = (v, undoable = true) => {
    v = clamp(Number(v), o.min, o.max);
    if (!Number.isFinite(v)) return;
    store.set({ [key]: v }, { undoable });
  };

  range.addEventListener('pointerdown', () => store.begin());
  range.addEventListener('keydown', () => store.begin());
  range.addEventListener('input', () => setValue(range.value, false));
  range.addEventListener('change', () => store.commit());
  range.addEventListener('dblclick', () => setValue(o.reset ?? o.defaultValue));
  num.addEventListener('focus', () => num.select());
  num.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      num.blur();
    } else if (e.key === 'Escape') {
      sync();
      num.blur();
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      const step = Number(o.step) * (e.shiftKey ? 10 : 1) * (e.key === 'ArrowUp' ? 1 : -1);
      setValue(Number(store.get(key)) + step);
    }
    e.stopPropagation();
  });
  num.addEventListener('blur', () => {
    const v = parseFloat(num.value.replace(',', '.'));
    if (Number.isFinite(v) && v !== store.get(key)) setValue(v);
    else sync();
  });

  function sync() {
    const v = store.get(key);
    const disabled = o.disabled?.(store.state) ?? false;
    range.disabled = disabled;
    num.disabled = disabled;
    el.classList.toggle('disabled', disabled);
    range.value = v;
    if (document.activeElement !== num) num.value = fmt(v);
    range.style.setProperty('--p', `${((v - o.min) / (o.max - o.min)) * 100}%`);
    el.classList.toggle('modified', o.defaultValue !== undefined && Math.abs(v - o.defaultValue) > 1e-9);
  }
  sync();
  return { el, keys: [key, ...(o.deps ?? [])], sync };
}

export function segmented(store, key, o) {
  const buttons = o.options.map((opt) =>
    h(
      'button',
      {
        type: 'button',
        class: 'seg-btn',
        title: opt.hint ?? '',
        onclick: () => store.set({ [key]: opt.value })
      },
      opt.icon ? icon(opt.icon) : null,
      opt.label
    )
  );
  const group = h('div', { class: 'seg', role: 'radiogroup', 'aria-label': o.label ?? key }, buttons);
  const el = o.label ? h('div', { class: 'ctl', title: o.hint ?? '' }, h('div', { class: 'ctl-head' }, h('span', { class: 'ctl-label' }, o.label)), group) : group;
  function sync() {
    const v = store.get(key);
    const disabled = o.disabled?.(store.state) ?? false;
    el.classList.toggle('disabled', disabled);
    buttons.forEach((b, i) => {
      const on = o.options[i].value === v;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', on);
      b.setAttribute('role', 'radio');
      b.disabled = disabled;
    });
  }
  sync();
  return { el, keys: [key, ...(o.deps ?? [])], sync };
}

export function select(store, key, o) {
  const sel = h('select', { class: 'select', 'aria-label': o.label });
  const fill = () => {
    sel.replaceChildren(
      ...(typeof o.options === 'function' ? o.options(store.state) : o.options).map((opt) =>
        opt.group
          ? h('optgroup', { label: opt.group }, opt.items.map((it) => h('option', { value: it.value }, it.label)))
          : h('option', { value: opt.value }, opt.label)
      )
    );
  };
  fill();
  sel.addEventListener('change', () => store.set({ [key]: o.parse ? o.parse(sel.value) : sel.value }));
  sel.addEventListener('keydown', (e) => e.stopPropagation());
  const hint = h('div', { class: 'ctl-hint' });
  const el = h('div', { class: 'ctl' }, h('div', { class: 'ctl-head' }, h('span', { class: 'ctl-label' }, o.label)), h('div', { class: 'select-wrap' }, sel, icon('chevron')), o.describe ? hint : null);
  function sync() {
    if (typeof o.options === 'function') fill();
    sel.value = String(store.get(key));
    if (o.describe) hint.textContent = o.describe(store.state) ?? '';
    const hidden = o.hidden?.(store.state) ?? false;
    el.hidden = hidden;
  }
  sync();
  return { el, keys: [key, ...(o.deps ?? [])], sync };
}

export function toggle(store, key, o) {
  const input = h('input', { type: 'checkbox', role: 'switch' });
  input.addEventListener('change', () => store.set({ [key]: input.checked }));
  const el = h(
    'label',
    { class: 'ctl toggle', title: o.hint ?? '' },
    h('span', { class: 'ctl-label' }, o.label),
    h('span', { class: 'switch' }, input, h('span', { class: 'knob' }))
  );
  function sync() {
    input.checked = !!store.get(key);
    const hidden = o.hidden?.(store.state) ?? false;
    el.hidden = hidden;
  }
  sync();
  return { el, keys: [key, ...(o.deps ?? [])], sync };
}

const SECTION_KEY = 'parallaxer.sections';
function sectionState() {
  try {
    return JSON.parse(localStorage.getItem(SECTION_KEY) || '{}');
  } catch {
    return {};
  }
}

export function section(id, title, iconName, children, { open = true, actions = null } = {}) {
  const saved = sectionState();
  const isOpen = saved[id] ?? open;
  const body = h('div', { class: 'sec-body' }, children);
  const btn = h(
    'button',
    { type: 'button', class: 'sec-toggle', 'aria-expanded': String(isOpen) },
    icon('chevron', 'sec-chev'),
    icon(iconName, 'sec-ico'),
    h('span', { class: 'sec-title' }, title)
  );
  const head = h('div', { class: 'sec-head' }, btn, actions ? h('div', { class: 'sec-actions' }, actions) : null);
  const el = h('section', { class: `sec${isOpen ? ' open' : ''}`, dataset: { id } }, head, body);
  btn.addEventListener('click', () => {
    const now = !el.classList.contains('open');
    el.classList.toggle('open', now);
    btn.setAttribute('aria-expanded', String(now));
    const st = sectionState();
    st[id] = now;
    try {
      localStorage.setItem(SECTION_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  });
  return el;
}
