import { VERT, COPY, JBU, SNAP, DILATE, BLUR, WARP, SOFTEN, COMPOSE } from './shaders.js';

// Row-major 3x3 matrices, out = L * left + R * right.
export const ANAGLYPH = {
  'dubois-rc': {
    label: 'Red / Cyan — Dubois',
    linear: true,
    L: [0.456, 0.5, 0.176, -0.04, -0.038, -0.016, -0.015, -0.021, -0.005],
    R: [-0.043, -0.088, -0.002, 0.378, 0.734, -0.018, -0.072, -0.113, 1.226]
  },
  'halfcolor-rc': {
    label: 'Red / Cyan — Half colour',
    linear: false,
    L: [0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0],
    R: [0, 0, 0, 0, 1, 0, 0, 0, 1]
  },
  'color-rc': {
    label: 'Red / Cyan — Full colour',
    linear: false,
    L: [1, 0, 0, 0, 0, 0, 0, 0, 0],
    R: [0, 0, 0, 0, 1, 0, 0, 0, 1]
  },
  'gray-rc': {
    label: 'Red / Cyan — Monochrome',
    linear: false,
    L: [0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0],
    R: [0, 0, 0, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114]
  },
  'dubois-gm': {
    label: 'Green / Magenta — Dubois',
    linear: true,
    L: [-0.062, -0.158, -0.039, 0.284, 0.668, 0.143, -0.015, -0.027, 0.021],
    R: [0.529, 0.705, 0.024, -0.016, -0.015, -0.065, 0.009, 0.075, 0.937]
  },
  'dubois-ab': {
    label: 'Amber / Blue — Dubois',
    linear: true,
    L: [1.062, -0.205, 0.299, -0.026, 0.908, 0.068, -0.038, -0.173, 0.022],
    R: [-0.016, -0.123, -0.017, 0.006, 0.062, -0.017, 0.094, 0.185, 0.911]
  }
};

// Compose-shader layout codes.
export const LAYOUT = {
  SBS: 0,
  TB: 1,
  ANAGLYPH: 2,
  ROWS: 3,
  COLUMNS: 4,
  CHECKER: 5,
  MONO_L: 6,
  MONO_R: 7,
  DEPTH: 8,
  RGBD: 9,
  PARALLAX: 10,
  HOLES: 11,
  ORIGINAL: 12
};

const EYE_LAYOUTS = new Set([LAYOUT.SBS, LAYOUT.TB, LAYOUT.ANAGLYPH, LAYOUT.ROWS, LAYOUT.COLUMNS, LAYOUT.CHECKER, LAYOUT.MONO_L, LAYOUT.MONO_R, LAYOUT.HOLES]);

export class StereoRenderer {
  constructor(canvas, { preserveDrawingBuffer = false } = {}) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer,
      powerPreference: 'high-performance'
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    const floatRT = gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float');
    this.depthRT = floatRT ? { internal: gl.R16F, format: gl.RED, type: gl.HALF_FLOAT } : { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE };
    this.vao = gl.createVertexArray();
    this.programs = {};
    for (const [name, src] of Object.entries({ copy: COPY, jbu: JBU, snap: SNAP, dilate: DILATE, blur: BLUR, warp: WARP, soften: SOFTEN, compose: COMPOSE })) {
      this.programs[name] = this.#program(src, name);
    }
    this.targets = new Map();
    this.color = { tex: this.#texture(), w: 0, h: 0 };
    this.depth = { tex: this.#texture(), w: 0, h: 0 };
    this.hasSource = false;
    this.hasDepth = false;
    gl.bindTexture(gl.TEXTURE_2D, this.color.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
    this.scratch = null;
  }

  // ---------- GL plumbing ----------

  #shader(type, src, name) {
    const gl = this.gl;
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS) && !gl.isContextLost()) {
      throw new Error(`Shader "${name}" failed to compile: ${gl.getShaderInfoLog(s)}`);
    }
    return s;
  }

  #program(fsSrc, name) {
    const gl = this.gl;
    const p = gl.createProgram();
    gl.attachShader(p, this.#shader(gl.VERTEX_SHADER, VERT, name));
    gl.attachShader(p, this.#shader(gl.FRAGMENT_SHADER, fsSrc, name));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS) && !gl.isContextLost()) {
      throw new Error(`Program "${name}" failed to link: ${gl.getProgramInfoLog(p)}`);
    }
    const uniforms = {};
    const count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(p, i);
      uniforms[info.name] = gl.getUniformLocation(p, info.name);
    }
    return { p, u: uniforms };
  }

  #texture(filter = this.gl.LINEAR) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  #target(name, w, h, spec) {
    const gl = this.gl;
    let t = this.targets.get(name);
    if (t && t.w === w && t.h === h && t.spec === spec) return t;
    if (t) {
      gl.deleteTexture(t.tex);
      gl.deleteFramebuffer(t.fbo);
    }
    const tex = this.#texture();
    gl.texImage2D(gl.TEXTURE_2D, 0, spec.internal, w, h, 0, spec.format, spec.type, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    t = { tex, fbo, w, h, spec };
    this.targets.set(name, t);
    return t;
  }

  get #rgba() {
    const gl = this.gl;
    return (this._rgba ??= { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE });
  }

  #use(name) {
    const prog = this.programs[name];
    this.gl.useProgram(prog.p);
    return prog.u;
  }

  #bind(unit, tex, loc) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (loc) gl.uniform1i(loc, unit);
  }

  #draw(target, w, h) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
    gl.viewport(0, 0, w ?? target.w, h ?? target.h);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // ---------- Inputs ----------

  /**
   * Upload a frame (VideoFrame, <video>, ImageBitmap, canvas, image) into the back colour buffer.
   * The frame only becomes visible after commit(), so colour and depth always stay in lock-step.
   */
  stage(source, width, height) {
    const gl = this.gl;
    this.back ??= { tex: this.#texture(), w: 0, h: 0 };
    gl.bindTexture(gl.TEXTURE_2D, this.back.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    // Decoded video goes through a 2D canvas: Chromium's direct WebGL upload of a VideoFrame whose coded size
    // exceeds its visible rect bleeds padding chroma into the last visible column (a green edge line).
    const isVideo = (typeof VideoFrame !== 'undefined' && source instanceof VideoFrame) || (typeof HTMLVideoElement !== 'undefined' && source instanceof HTMLVideoElement);
    let uploaded = false;
    if (!isVideo) {
      try {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, source);
        uploaded = true;
      } catch {
        uploaded = false;
      }
    }
    if (!uploaded) {
      this.scratch ??= new OffscreenCanvas(width, height);
      if (this.scratch.width !== width || this.scratch.height !== height) {
        this.scratch.width = width;
        this.scratch.height = height;
      }
      this.scratchCtx ??= this.scratch.getContext('2d', { alpha: false });
      this.scratchCtx.drawImage(source, 0, 0, width, height);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.scratch);
    }
    gl.generateMipmap(gl.TEXTURE_2D);
    this.back.w = width;
    this.back.h = height;
    this.staged = true;
  }

  /** Make the staged frame current. */
  commit() {
    if (!this.staged) return;
    const t = this.color;
    this.color = this.back;
    this.back = t;
    this.staged = false;
    this.hasSource = true;
  }

  uploadSource(source, width, height) {
    this.stage(source, width, height);
    this.commit();
  }

  /** Downscale the current source on the GPU and read it back as RGBA8 (image row order). */
  readModelInput(w, h) {
    const gl = this.gl;
    const t = this.#target('modelIn', w, h, this.#rgba);
    const u = this.#use('copy');
    this.#bind(0, (this.staged ? this.back : this.color).tex, u.uTex);
    this.#draw(t);
    const out = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out);
    return out;
  }

  /** Upload a normalised nearness map (1 = closest) produced by the depth stage. */
  uploadDepth(data, w, h) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.depth.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, w, h, 0, gl.RED, gl.FLOAT, data);
    this.depth.w = w;
    this.depth.h = h;
    this.hasDepth = true;
  }

  // ---------- Pipeline ----------

  #refine(eyeW, eyeH, logicalW, logicalH, r) {
    const gl = this.gl;
    const A = this.#target('dA', eyeW, eyeH, this.depthRT);
    const B = this.#target('dB', eyeW, eyeH, this.depthRT);

    let u = this.#use('jbu');
    this.#bind(0, this.color.tex, u.uColor);
    this.#bind(1, this.depth.tex, u.uDepth);
    gl.uniform2f(u.uDepthSize, this.depth.w, this.depth.h);
    gl.uniform1f(u.uGuideLod, Math.max(0, Math.log2(this.color.w / eyeW)));
    gl.uniform1f(u.uTapLod, Math.max(0, Math.log2(this.color.w / this.depth.w)));
    gl.uniform1f(u.uSigmaC, r.sigmaC);
    gl.uniform1f(u.uEdge, r.edge);
    this.#draw(A);

    // Radii arrive as a fraction of the logical width and are converted to texels per axis.
    const px = (frac) => frac * logicalW;
    const sx = eyeW / logicalW;
    const sy = eyeH / logicalH;

    // Edge snapping: alternating horizontal / vertical cross-bilateral passes.
    const snapPx = px(r.snap ?? 0);
    if (snapPx >= 1 && r.edge > 0) {
      u = this.#use('snap');
      this.#bind(1, this.color.tex, u.uColor);
      gl.uniform1f(u.uLod, Math.max(0, Math.log2(this.color.w / eyeW)));
      const sigmaC = r.sigmaC * (1.4 - 0.8 * r.edge);
      gl.uniform1f(u.uInvC, 1 / (2 * sigmaC * sigmaC));
      const iters = r.snapIterations ?? 2;
      for (let k = 0; k < iters; k++) {
        for (const [src, dst, step, radius] of [
          [A, B, [1 / eyeW, 0], snapPx * sx],
          [B, A, [0, 1 / eyeH], snapPx * sy]
        ]) {
          const taps = Math.min(48, Math.max(2, Math.round(radius)));
          const stride = radius / taps;
          this.#bind(0, src.tex, u.uSrc);
          gl.uniform2f(u.uStep, step[0] * stride, step[1] * stride);
          gl.uniform1i(u.uRadius, taps);
          gl.uniform1f(u.uInvS, 1 / (2 * (taps * 0.6) ** 2));
          this.#draw(dst);
        }
      }
    }

    const dil = px(r.dilate);
    if (dil >= 0.5) {
      u = this.#use('dilate');
      this.#bind(0, A.tex, u.uSrc);
      gl.uniform2f(u.uStep, 1 / eyeW, 0);
      gl.uniform1i(u.uRadius, Math.min(64, Math.round(dil * sx)));
      this.#draw(B);
      this.#bind(0, B.tex, u.uSrc);
      gl.uniform2f(u.uStep, 0, 1 / eyeH);
      gl.uniform1i(u.uRadius, Math.min(64, Math.round(dil * sy)));
      this.#draw(A);
    }

    const blur = px(r.blur);
    if (blur >= 0.5) {
      u = this.#use('blur');
      const rangeInv = r.preserve > 0 ? 1 / (2 * Math.pow(0.5 * (1 - r.preserve) + 0.02, 2)) : 0;
      gl.uniform1f(u.uRangeInv, rangeInv);
      const pass = (src, dst, step, radiusPx) => {
        const sigma = Math.max(0.5, radiusPx / 2.5);
        this.#bind(0, src.tex, u.uSrc);
        gl.uniform2f(u.uStep, step[0], step[1]);
        gl.uniform1i(u.uRadius, Math.min(96, Math.ceil(radiusPx)));
        gl.uniform1f(u.uSigma, sigma);
        this.#draw(dst);
      };
      pass(A, B, [1 / eyeW, 0], blur * sx);
      pass(B, A, [0, 1 / eyeH], blur * sy);
    }
    return A;
  }

  #warp(name, eye, depthTarget, eyeW, eyeH, logicalW, logicalH, p) {
    const gl = this.gl;
    const raw = this.#target(name + 'raw', eyeW, eyeH, this.#rgba);
    let u = this.#use('warp');
    this.#bind(0, this.color.tex, u.uColor);
    this.#bind(1, depthTarget.tex, u.uDepth);
    gl.uniform2f(u.uLogical, logicalW, logicalH);
    gl.uniform1f(u.uColorLod, Math.max(0, Math.log2(this.color.w / logicalW)));
    gl.uniform1f(u.uE, eye.e);
    gl.uniform1f(u.uS, p.strength);
    gl.uniform1f(u.uConv, p.convergence);
    const dir = eye.dir ?? [1, 0];
    gl.uniform2f(u.uDir, dir[0], dir[1]);
    gl.uniform1i(u.uRadial, eye.radial ? 1 : 0);
    const c = eye.center ?? [0.5, 0.5];
    gl.uniform2f(u.uCenter, c[0], c[1]);
    gl.uniform1f(u.uStep, p.step);
    gl.uniform1f(u.uMaxStretch, p.maxStretch);
    gl.uniform1i(u.uFill, p.fill === 'mirror' ? 1 : 0);
    gl.uniform1f(u.uMirrorMax, Math.max(2, 0.08 * logicalW));
    gl.uniform1f(u.uZoom, eye.zoom ?? 1);
    gl.uniform4f(u.uCurve, ...p.curve);
    this.#draw(raw);

    if (!(p.soften > 0) || (!eye.radial && Math.abs(eye.e) < 1e-5)) return raw;
    const out = this.#target(name, eyeW, eyeH, this.#rgba);
    u = this.#use('soften');
    this.#bind(0, raw.tex, u.uEye);
    // Perpendicular to the streak direction, one texel per step.
    const perp = eye.radial ? [0, 1] : [-dir[1], dir[0]];
    gl.uniform2f(u.uPerp, perp[0] / eyeW, perp[1] / eyeH);
    gl.uniform1f(u.uAmount, p.soften);
    this.#draw(out);
    return out;
  }

  /**
   * Render a frame.
   * @param {object} o
   *   canvasW/H   drawing-buffer size
   *   eyeW/H      per-eye texture size (already squeezed for half layouts)
   *   logicalW/H  unsqueezed geometry the parallax budget refers to
   *   layout      LAYOUT code; swap; anaglyph key; colormap
   *   eyes        [{e, dir?, radial?, center?}, {…}] (second optional)
   *   stereo      {strength, convergence, curve, step, maxStretch, fill, soften}
   *   refine      {edge, sigmaC, dilate, blur, preserve}
   *   window      {left, right, feather}; comfort (percent)
   */
  render(o) {
    const gl = this.gl;
    if (gl.isContextLost()) return;
    if (this.canvas.width !== o.canvasW) this.canvas.width = o.canvasW;
    if (this.canvas.height !== o.canvasH) this.canvas.height = o.canvasH;
    if (!this.hasSource) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, o.canvasW, o.canvasH);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }
    if (!this.hasDepth) this.uploadDepth(new Float32Array([0.5]), 1, 1);

    const { eyeW, eyeH, logicalW, logicalH, layout } = o;
    const needsDepth = layout !== LAYOUT.ORIGINAL;
    const depthT = needsDepth ? this.#refine(eyeW, eyeH, logicalW, logicalH, o.refine) : null;

    let L = null;
    let R = null;
    if (EYE_LAYOUTS.has(layout)) {
      const [eL, eR] = o.eyes;
      if (layout === LAYOUT.MONO_R) {
        R = this.#warp('eyeR', eR ?? eL, depthT, eyeW, eyeH, logicalW, logicalH, o.stereo);
      } else {
        L = this.#warp('eyeL', eL, depthT, eyeW, eyeH, logicalW, logicalH, o.stereo);
        if (layout !== LAYOUT.MONO_L) R = this.#warp('eyeR', eR, depthT, eyeW, eyeH, logicalW, logicalH, o.stereo);
      }
    }

    const u = this.#use('compose');
    this.#bind(0, (L ?? R ?? this.color).tex, u.uL);
    this.#bind(1, (R ?? L ?? this.color).tex, u.uR);
    this.#bind(2, this.color.tex, u.uColor);
    this.#bind(3, (depthT ?? this.depth).tex, u.uDepth);
    gl.uniform1i(u.uLayout, layout);
    gl.uniform1i(u.uSwap, o.swap ? 1 : 0);
    const ana = ANAGLYPH[o.anaglyph] ?? ANAGLYPH['dubois-rc'];
    gl.uniformMatrix3fv(u.uAnaL, true, ana.L);
    gl.uniformMatrix3fv(u.uAnaR, true, ana.R);
    gl.uniform1i(u.uAnaLinear, ana.linear ? 1 : 0);
    const w = o.window ?? { left: 0, right: 0, feather: 0 };
    gl.uniform4f(u.uWindow, w.left, w.right, Math.max(w.feather, 1e-4), 0);
    gl.uniform1f(u.uS, o.stereo.strength);
    gl.uniform1f(u.uConv, o.stereo.convergence);
    gl.uniform4f(u.uCurve, ...o.stereo.curve);
    gl.uniform1f(u.uComfort, o.comfort ?? 100);
    gl.uniform1i(u.uColormap, o.colormap ?? 0);
    gl.uniform1f(u.uColorLod, Math.max(0, Math.log2(this.color.w / Math.max(1, o.canvasW))));
    this.#draw(null, o.canvasW, o.canvasH);
  }

  /** Render both eye views (no compose). Returns the eye targets for callers that present them themselves (WebXR). */
  renderEyes(o) {
    if (!this.hasSource || this.gl.isContextLost()) return null;
    if (!this.hasDepth) this.uploadDepth(new Float32Array([0.5]), 1, 1);
    const { eyeW, eyeH, logicalW, logicalH } = o;
    const depthT = this.#refine(eyeW, eyeH, logicalW, logicalH, o.refine);
    const [eL, eR] = o.eyes;
    const L = this.#warp('eyeL', eL, depthT, eyeW, eyeH, logicalW, logicalH, o.stereo);
    const R = this.#warp('eyeR', eR, depthT, eyeW, eyeH, logicalW, logicalH, o.stereo);
    return { L, R };
  }

  dispose() {
    const ext = this.gl.getExtension('WEBGL_lose_context');
    ext?.loseContext();
  }
}
