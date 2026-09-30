// GLSL ES 3.00 programs for the stereo pipeline.
//
// Coordinate convention: every internal texture is stored in image order (v = 0 is the top row).
// Only the final compose pass flips Y, because the default framebuffer is presented bottom-up.

export const VERT = /* glsl */ `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const HEADER = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
in vec2 vUv;
out vec4 o;
`;

// Straight copy with hardware trilinear minification (used to build the depth-model input).
export const COPY = HEADER + /* glsl */ `
uniform sampler2D uTex;
void main() { o = vec4(texture(uTex, vUv).rgb, 1.0); }`;

// Joint bilateral upsampling: low-res network depth -> work resolution, snapped to colour edges.
export const JBU = HEADER + /* glsl */ `
uniform sampler2D uColor;
uniform sampler2D uDepth;
uniform vec2 uDepthSize;
uniform float uGuideLod;
uniform float uTapLod;
uniform float uSigmaC;
uniform float uEdge;
void main() {
  vec2 texel = 1.0 / uDepthSize;
  vec2 pos = vUv * uDepthSize - 0.5;
  vec2 base = floor(pos);
  vec3 c0 = textureLod(uColor, vUv, uGuideLod).rgb;
  float inv2sc = 1.0 / (2.0 * uSigmaC * uSigmaC);
  float wsum = 0.0;
  float dsum = 0.0;
  for (int j = -1; j <= 2; j++) {
    for (int i = -1; i <= 2; i++) {
      vec2 cell = base + vec2(float(i), float(j));
      vec2 uvS = (cell + 0.5) * texel;
      vec2 dd = cell - pos;
      float ws = exp(-dot(dd, dd) * 0.6);
      float d = texture(uDepth, uvS).r;
      vec3 dc = textureLod(uColor, uvS, uTapLod).rgb - c0;
      float wc = exp(-dot(dc, dc) * inv2sc);
      float w = ws * mix(1.0, wc, uEdge) + 1e-6;
      wsum += w;
      dsum += w * d;
    }
  }
  o = vec4(dsum / wsum, 0.0, 0.0, 1.0);
}`;

// Separable cross-bilateral filter guided by the colour image. Averages depth only among pixels of
// similar colour, which collapses the network's soft depth ramps onto the real object silhouettes.
export const SNAP = HEADER + /* glsl */ `
uniform sampler2D uSrc;
uniform sampler2D uColor;
uniform float uLod;
uniform vec2 uStep;
uniform int uRadius;
uniform float uInvS;
uniform float uInvC;
void main() {
  vec3 c0 = textureLod(uColor, vUv, uLod).rgb;
  float acc = texture(uSrc, vUv).r;
  float ws = 1.0;
  for (int i = 1; i <= 48; i++) {
    if (i > uRadius) break;
    float fi = float(i);
    float gs = fi * fi * uInvS;
    vec2 off = uStep * fi;
    vec3 ca = textureLod(uColor, vUv + off, uLod).rgb - c0;
    vec3 cb = textureLod(uColor, vUv - off, uLod).rgb - c0;
    float wa = exp(-gs - dot(ca, ca) * uInvC);
    float wb = exp(-gs - dot(cb, cb) * uInvC);
    acc += wa * texture(uSrc, vUv + off).r + wb * texture(uSrc, vUv - off).r;
    ws += wa + wb;
  }
  o = vec4(acc / ws, 0.0, 0.0, 1.0);
}`;

// Separable max filter: grows the foreground so object edges carry foreground depth (kills halos).
export const DILATE = HEADER + /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uStep;
uniform int uRadius;
void main() {
  float m = texture(uSrc, vUv).r;
  for (int i = 1; i <= 64; i++) {
    if (i > uRadius) break;
    vec2 off = uStep * float(i);
    m = max(m, max(texture(uSrc, vUv + off).r, texture(uSrc, vUv - off).r));
  }
  o = vec4(m, 0.0, 0.0, 1.0);
}`;

// Separable Gaussian with an optional range term (edge-preserving smoothing of the depth field).
export const BLUR = HEADER + /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uStep;
uniform int uRadius;
uniform float uSigma;
uniform float uRangeInv;
void main() {
  float c = texture(uSrc, vUv).r;
  float inv = 1.0 / (2.0 * uSigma * uSigma);
  float acc = c;
  float ws = 1.0;
  for (int i = 1; i <= 96; i++) {
    if (i > uRadius) break;
    float fi = float(i);
    float g = exp(-fi * fi * inv);
    float a = texture(uSrc, vUv + uStep * fi).r;
    float b = texture(uSrc, vUv - uStep * fi).r;
    float wa = g * exp(-(a - c) * (a - c) * uRangeInv);
    float wb = g * exp(-(b - c) * (b - c) * uRangeInv);
    acc += wa * a + wb * b;
    ws += wa + wb;
  }
  o = vec4(acc / ws, 0.0, 0.0, 1.0);
}`;

const CURVE = /* glsl */ `
uniform vec4 uCurve; // x: near clip, y: far clip, z: gamma, w: invert
float curve(float d) {
  d = mix(d, 1.0 - d, uCurve.w);
  d = clamp((d - uCurve.y) / max(uCurve.x - uCurve.y, 1e-4), 0.0, 1.0);
  return pow(d, uCurve.z);
}`;

// Depth-image-based rendering by backward search.
//
// For every target pixel we march along the displacement line and look for source positions whose
// forward projection lands on the target. Among all valid intersections the nearest surface wins
// (correct occlusion). Where the projection "jumps" over the target we have a disocclusion; we locate
// the depth edge by bisection and fill from the background side, either stretched or mirrored.
//
// Directional mode shifts along uDir (stereo eyes, look-around, orbit). Radial mode scales about uCenter
// by a depth-dependent factor (dolly / zoom parallax).
export const WARP = HEADER + CURVE + /* glsl */ `
uniform sampler2D uColor;
uniform sampler2D uDepth;
uniform vec2 uLogical;
uniform float uColorLod;
uniform float uE;
uniform float uS;
uniform float uConv;
uniform vec2 uDir;
uniform int uRadial;
uniform vec2 uCenter;
uniform float uStep;
uniform float uMaxStretch;
uniform int uFill;
uniform float uMirrorMax;
uniform float uZoom;

float nearAt(vec2 px) { return curve(texture(uDepth, px / uLogical).r); }

float disp(float d) { return uE * uS * uLogical.x * (uConv - d); }
float mag(float d) { return max(1.0 + uE * (d - uConv), 0.05); }

void main() {
  vec2 uv = 0.5 + (vUv - 0.5) / uZoom;
  vec2 target = uv * uLogical;
  if (uRadial == 0 && abs(uE) < 1e-5) {
    o = vec4(textureLod(uColor, uv, uColorLod).rgb, 1.0);
    return;
  }

  vec2 u;
  float r0 = 0.0;
  float tLo;
  float tHi;
  if (uRadial == 1) {
    vec2 v = target - uCenter * uLogical;
    r0 = length(v);
    u = r0 > 1e-3 ? v / r0 : vec2(1.0, 0.0);
    float m0 = mag(0.0);
    float m1 = mag(1.0);
    tLo = r0 / max(m0, m1) - r0;
    tHi = r0 / min(m0, m1) - r0;
  } else {
    u = uDir;
    float d0 = disp(0.0);
    float d1 = disp(1.0);
    tLo = -max(d0, d1);
    tHi = -min(d0, d1);
  }
  tLo -= 1.0;
  tHi += 1.0;

  int n = int(clamp(ceil((tHi - tLo) / uStep), 2.0, 320.0)) + 1;
  float dt = (tHi - tLo) / float(n - 1);

  float tPrev = tLo;
  float dPrev = nearAt(target + u * tPrev);
  float gPrev = uRadial == 1 ? (r0 + tPrev) * mag(dPrev) - r0 : tPrev + disp(dPrev);

  float bestD = -1.0;
  float bestT = 0.0;
  float holeFg = -1.0;
  float hA = 0.0, hB = 0.0, hDa = 0.0, hDb = 0.0;

  for (int i = 1; i < 322; i++) {
    if (i >= n) break;
    float t = tLo + dt * float(i);
    float d = nearAt(target + u * t);
    float g = uRadial == 1 ? (r0 + t) * mag(d) - r0 : t + disp(d);
    if ((gPrev <= 0.0 && g > 0.0) || (gPrev >= 0.0 && g < 0.0)) {
      float slope = (g - gPrev) / dt;
      if (slope > 0.0 && slope < uMaxStretch) {
        float a = gPrev / (gPrev - g);
        float dr = mix(dPrev, d, a);
        if (dr > bestD) {
          bestD = dr;
          bestT = mix(tPrev, t, a);
        }
      } else if (slope >= uMaxStretch) {
        float fg = max(dPrev, d);
        if (fg > holeFg) {
          holeFg = fg;
          hA = tPrev; hB = t; hDa = dPrev; hDb = d;
        }
      }
    }
    tPrev = t;
    dPrev = d;
    gPrev = g;
  }

  vec2 srcPx;
  float covered = 1.0;
  if (bestD >= 0.0) {
    srcPx = target + u * bestT;
  } else if (holeFg >= 0.0) {
    // Locate the depth edge inside the bracketing interval.
    bool aIsFg = hDa > hDb;
    float mid = 0.5 * (hDa + hDb);
    float a = hA;
    float b = hB;
    for (int k = 0; k < 6; k++) {
      float m = 0.5 * (a + b);
      bool mFg = nearAt(target + u * m) > mid;
      if (mFg == aIsFg) a = m; else b = m;
    }
    float tEdge = 0.5 * (a + b);
    float dBg = min(hDa, hDb);
    float side = aIsFg ? 1.0 : -1.0;
    float tHyp = uRadial == 1 ? r0 / mag(dBg) - r0 : -disp(dBg);
    float into = min(abs(tEdge - tHyp), uMirrorMax);
    float off = uFill == 1 ? into + 1.0 : 1.0;
    srcPx = target + u * (tEdge + side * off);
    covered = 0.0;
  } else {
    float d = nearAt(target);
    srcPx = uRadial == 1 ? target + u * (r0 / mag(d) - r0) : target - u * disp(d);
  }
  o = vec4(textureLod(uColor, srcPx / uLogical, uColorLod).rgb, covered);
}`;

// Smooths filled disocclusions across the streak direction, drawing mostly from other hole pixels.
export const SOFTEN = HEADER + /* glsl */ `
uniform sampler2D uEye;
uniform vec2 uPerp;
uniform float uAmount;
void main() {
  vec4 c = texture(uEye, vUv);
  if (c.a > 0.99 || uAmount <= 0.0) { o = c; return; }
  vec3 acc = c.rgb;
  float ws = 1.0;
  for (int j = 1; j <= 8; j++) {
    float fj = float(j);
    float g = exp(-fj * fj / 18.0);
    vec4 a = texture(uEye, vUv + uPerp * fj);
    vec4 b = texture(uEye, vUv - uPerp * fj);
    float wa = g * mix(1.0, 0.08, a.a);
    float wb = g * mix(1.0, 0.08, b.a);
    acc += a.rgb * wa + b.rgb * wb;
    ws += wa + wb;
  }
  o = vec4(mix(c.rgb, acc / ws, uAmount), c.a);
}`;

// Final assembly into the canvas in the requested layout / diagnostic view.
export const COMPOSE = HEADER + CURVE + /* glsl */ `
uniform sampler2D uL;
uniform sampler2D uR;
uniform sampler2D uColor;
uniform sampler2D uDepth;
uniform int uLayout;
uniform int uSwap;
uniform mat3 uAnaL;
uniform mat3 uAnaR;
uniform int uAnaLinear;
uniform vec4 uWindow;   // x: left-eye left mask, y: right-eye right mask (uv), z: feather (uv)
uniform float uS;
uniform float uConv;
uniform float uComfort;
uniform int uColormap;
uniform float uColorLod;

vec3 turbo(float x) {
  const vec4 kr4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kg4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kb4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kr2 = vec2(-152.94239396, 59.28637943);
  const vec2 kg2 = vec2(4.27729857, 2.82956604);
  const vec2 kb2 = vec2(-89.90310912, 27.34824973);
  x = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return clamp(vec3(dot(v4, kr4) + dot(v2, kr2), dot(v4, kg4) + dot(v2, kg2), dot(v4, kb4) + dot(v2, kb2)), 0.0, 1.0);
}

vec3 colormap(float d) {
  if (uColormap == 1) return turbo(d);
  if (uColormap == 2) return mix(vec3(0.02, 0.01, 0.10), mix(vec3(0.55, 0.12, 0.55), vec3(1.0, 0.85, 0.45), smoothstep(0.45, 1.0, d)), smoothstep(0.0, 0.55, d));
  return vec3(d);
}

float windowMask(float x, float edge, bool leftEdge) {
  if (edge <= 0.0) return 1.0;
  float dist = leftEdge ? x : 1.0 - x;
  return smoothstep(edge - uWindow.z, edge, dist);
}

vec3 eyeL(vec2 uv) { return texture(uL, uv).rgb * windowMask(uv.x, uWindow.x, true); }
vec3 eyeR(vec2 uv) { return texture(uR, uv).rgb * windowMask(uv.x, uWindow.y, false); }

vec3 toLin(vec3 c) { return pow(c, vec3(2.2)); }
vec3 toGam(vec3 c) { return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2)); }

void main() {
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y);
  bool swap = uSwap == 1;
  vec3 c;
  if (uLayout == 0) {
    bool first = uv.x < 0.5;
    vec2 e = vec2(fract(uv.x * 2.0), uv.y);
    c = (first != swap) ? eyeL(e) : eyeR(e);
  } else if (uLayout == 1) {
    bool first = uv.y < 0.5;
    vec2 e = vec2(uv.x, fract(uv.y * 2.0));
    c = (first != swap) ? eyeL(e) : eyeR(e);
  } else if (uLayout == 2) {
    vec3 l = eyeL(uv);
    vec3 r = eyeR(uv);
    if (swap) { vec3 t = l; l = r; r = t; }
    if (uAnaLinear == 1) c = toGam(uAnaL * toLin(l) + uAnaR * toLin(r));
    else c = clamp(uAnaL * l + uAnaR * r, 0.0, 1.0);
  } else if (uLayout == 3 || uLayout == 4 || uLayout == 5) {
    vec2 fc = floor(gl_FragCoord.xy);
    float parity = uLayout == 3 ? mod(fc.y, 2.0) : uLayout == 4 ? mod(fc.x, 2.0) : mod(fc.x + fc.y, 2.0);
    bool left = (parity < 0.5) != swap;
    c = left ? eyeL(uv) : eyeR(uv);
  } else if (uLayout == 6) {
    c = texture(uL, uv).rgb;
  } else if (uLayout == 7) {
    c = texture(uR, uv).rgb;
  } else if (uLayout == 8) {
    c = colormap(curve(texture(uDepth, uv).r));
  } else if (uLayout == 9) {
    bool first = uv.x < 0.5;
    vec2 e = vec2(fract(uv.x * 2.0), uv.y);
    c = first ? textureLod(uColor, e, uColorLod).rgb : vec3(curve(texture(uDepth, e).r));
  } else if (uLayout == 10) {
    float d = curve(texture(uDepth, uv).r);
    float p = uS * (uConv - d);
    float pn = clamp(p / max(uS, 1e-4), -1.0, 1.0);
    vec3 base = textureLod(uColor, uv, uColorLod).rgb;
    float l = dot(base, vec3(0.2126, 0.7152, 0.0722)) * 0.75 + 0.25;
    vec3 tint = pn > 0.0 ? mix(vec3(1.0), vec3(0.25, 0.55, 1.0), pn) : mix(vec3(1.0), vec3(1.0, 0.38, 0.16), -pn);
    c = l * tint;
    if (abs(p) < uS * 0.012) c = mix(c, vec3(0.95, 0.95, 0.3), 0.55);
    if (abs(p) * 100.0 > uComfort && mod(gl_FragCoord.x + gl_FragCoord.y, 12.0) < 4.0) c = mix(c, vec3(1.0, 0.1, 0.7), 0.7);
  } else if (uLayout == 11) {
    bool first = uv.x < 0.5;
    vec2 e = vec2(fract(uv.x * 2.0), uv.y);
    vec4 s = first ? texture(uL, e) : texture(uR, e);
    float g = dot(s.rgb, vec3(0.2126, 0.7152, 0.0722)) * 0.6;
    c = mix(vec3(g), vec3(1.0, 0.1, 0.75), 1.0 - s.a);
  } else {
    c = textureLod(uColor, uv, uColorLod).rgb;
  }
  o = vec4(c, 1.0);
}`;
