import { layoutGeometry } from '../core/settings.js';

const VS = `#version 300 es
uniform mat4 uProj;
uniform mat4 uView;
uniform vec3 uCenter;
uniform vec2 uHalf;
out vec2 vUv;
void main() {
  vec2 c = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
  vUv = vec2(c.x, 1.0 - c.y);
  vec3 p = uCenter + vec3((c * 2.0 - 1.0) * uHalf, 0.0);
  gl_Position = uProj * uView * vec4(p, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform float uFade;
in vec2 vUv;
out vec4 o;
void main() { o = vec4(texture(uTex, vUv).rgb * uFade, 1.0); }`;

export async function xrSupported() {
  try {
    return !!(await navigator.xr?.isSessionSupported?.('immersive-vr'));
  } catch {
    return false;
  }
}

/**
 * Presents the live stereo pair on a virtual cinema screen in a WebXR headset: each eye sees its own
 * synthesised view, so the depth is real — no export needed. Controller "select" toggles playback.
 */
export class XRViewer {
  constructor({ preview, store, onToggle, onEnd }) {
    this.preview = preview;
    this.store = store;
    this.onToggle = onToggle;
    this.onEnd = onEnd;
    this.session = null;
  }

  get active() {
    return !!this.session;
  }

  #program(gl) {
    if (this.prog) return this.prog;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (const n of ['uProj', 'uView', 'uCenter', 'uHalf', 'uTex', 'uFade']) u[n] = gl.getUniformLocation(p, n);
    this.vao = gl.createVertexArray();
    this.prog = { p, u };
    return this.prog;
  }

  async start() {
    const gl = this.preview.renderer.gl;
    await gl.makeXRCompatible();
    const session = await navigator.xr.requestSession('immersive-vr', { optionalFeatures: ['local-floor'] });
    this.session = session;
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl, { antialias: false, depth: false }) });
    this.space = await session.requestReferenceSpace('local');
    this.#program(gl);
    this.started = performance.now();
    session.addEventListener('select', () => this.onToggle?.());
    session.addEventListener('end', () => {
      this.session = null;
      this.onEnd?.();
      this.preview.requestDraw();
    });
    session.requestAnimationFrame((t, f) => this.#frame(t, f));
  }

  async stop() {
    await this.session?.end();
  }

  #frame(t, frame) {
    const session = frame.session;
    session.requestAnimationFrame((tt, ff) => this.#frame(tt, ff));
    const pv = this.preview;
    pv.pump();
    const pose = frame.getViewerPose(this.space);
    const layer = session.renderState.baseLayer;
    const gl = pv.renderer.gl;
    if (!pose || !layer) return;

    const src = pv.source;
    let eyes = null;
    let aspect = 16 / 9;
    if (src?.w && pv.renderer.hasSource) {
      aspect = src.w / src.h;
      const k = Math.min(1, 1920 / Math.max(src.w, src.h));
      const geo = layoutGeometry('mono', src.w * k, src.h * k);
      const params = pv.fp.params(this.store.state, geo, { layout: 0 });
      eyes = pv.renderer.renderEyes(params);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.clearColor(0.015, 0.016, 0.02, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!eyes) return;

    // A 3.2 m wide screen, 2.8 m ahead of the starting head position.
    const width = 3.2;
    const half = [width / 2, width / 2 / aspect];
    const fade = Math.min(1, (performance.now() - this.started) / 600);
    const { p, u } = this.prog;
    gl.useProgram(p);
    gl.bindVertexArray(this.vao);
    gl.uniform3f(u.uCenter, 0, 0, -2.8);
    gl.uniform2f(u.uHalf, half[0], half[1]);
    gl.uniform1f(u.uFade, fade);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(u.uTex, 0);
    for (const view of pose.views) {
      const vp = layer.getViewport(view);
      gl.viewport(vp.x, vp.y, vp.width, vp.height);
      gl.uniformMatrix4fv(u.uProj, false, view.projectionMatrix);
      gl.uniformMatrix4fv(u.uView, false, view.transform.inverse.matrix);
      gl.bindTexture(gl.TEXTURE_2D, (view.eye === 'right' ? eyes.R : eyes.L).tex);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  }
}
