/** Tiny hyperscript helper. */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === 'html') el.innerHTML = v;
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, v);
    }
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

export function formatDuration(sec) {
  if (!Number.isFinite(sec)) return '—';
  sec = Math.max(0, sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

/** SMPTE-style timecode HH:MM:SS:FF. */
export function timecode(sec, fps) {
  if (!Number.isFinite(sec)) return '00:00:00:00';
  const f = Math.max(1, Math.round(fps || 30));
  const total = Math.max(0, Math.round(sec * f));
  const ff = total % f;
  const s = Math.floor(total / f);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}:${pad(ff)}`;
}

/** Observable settings store with coalesced undo / redo. */
export class Store {
  constructor(initial, { onPersist } = {}) {
    this.state = { ...initial };
    this.subs = new Set();
    this.undoStack = [];
    this.redoStack = [];
    this.pending = null;
    this.onPersist = onPersist;
    this.persistTimer = 0;
  }

  get(key) {
    return this.state[key];
  }

  subscribe(fn) {
    this.subs.add(fn);
    return () => this.subs.delete(fn);
  }

  /** Start a gesture (e.g. slider drag): the snapshot is pushed once when the gesture commits. */
  begin() {
    this.pending ??= { ...this.state };
  }

  commit() {
    if (!this.pending) return;
    if (JSON.stringify(this.pending) !== JSON.stringify(this.state)) {
      this.undoStack.push(this.pending);
      if (this.undoStack.length > 200) this.undoStack.shift();
      this.redoStack.length = 0;
    }
    this.pending = null;
  }

  set(patch, { undoable = true } = {}) {
    const changed = [];
    for (const [k, v] of Object.entries(patch)) {
      if (this.state[k] !== v) changed.push(k);
    }
    if (!changed.length) return;
    if (undoable && !this.pending) {
      this.undoStack.push({ ...this.state });
      if (this.undoStack.length > 200) this.undoStack.shift();
      this.redoStack.length = 0;
    }
    Object.assign(this.state, patch);
    this.#emit(changed);
  }

  undo() {
    const prev = this.undoStack.pop();
    if (!prev) return false;
    this.redoStack.push({ ...this.state });
    this.#replace(prev);
    return true;
  }

  redo() {
    const next = this.redoStack.pop();
    if (!next) return false;
    this.undoStack.push({ ...this.state });
    this.#replace(next);
    return true;
  }

  #replace(next) {
    const changed = Object.keys(next).filter((k) => next[k] !== this.state[k]);
    this.state = { ...next };
    this.#emit(changed);
  }

  #emit(changed) {
    for (const fn of this.subs) fn(this.state, changed);
    clearTimeout(this.persistTimer);
    this.persistTimer = setTimeout(() => this.onPersist?.(this.state), 250);
  }
}
