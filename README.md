# Parallaxer Studio

**Professional 2D → 3D conversion for video and photos, running entirely in your browser.**
AI depth estimation, occlusion-aware stereo view synthesis and hardware video encoding — nothing is uploaded.

## What it does

| | |
|---|---|
| **Depth** | MiDaS v2.1 Small (bundled) or Depth Anything V2 Small / Base (downloaded once, cached) on **WebGPU**, with a WebAssembly fallback (multi-threaded when cross-origin isolated). |
| **Edge fidelity** | Joint-bilateral upsampling plus iterative cross-bilateral *edge snapping* lock depth discontinuities to real object silhouettes; foreground dilation removes halos. |
| **View synthesis** | Per-pixel backward ray-march with correct occlusion ordering (nearest surface wins). Disocclusions are detected by projection tearing, the depth edge is located by bisection and filled from the background side (texture mirror or edge stretch), then softened across the streak direction. |
| **Temporal stability** | Robust percentile normalisation with a temporally smoothed range, a motion- and colour-gated per-pixel filter, and automatic scene-cut detection. |
| **Stereography** | Depth budget (% of width), screen plane / convergence (manual, histogram drag, or auto-converge on subject), symmetric or single-eye synthesis, depth curve and clip planes, automatic **floating window** against window violations, and a **comfort analyser** that checks divergence for your target screen size. |
| **Outputs** | Full / half side-by-side, full / half top-bottom, anaglyph (Dubois red-cyan, green-magenta, amber-blue and more), row / column / checkerboard interleave, RGB-D, depth map. |
| **Preview** | Real-time viewer with Output, Anaglyph, Wiggle, Look-around (pointer-driven novel views), Depth, Parallax heat-map, Occlusion and Original views. Hold `\` to compare. |
| **Export** | WebCodecs hardware encoding to MP4 / MOV / WebM / MKV with H.264, HEVC, AV1 or VP9 · original audio copied bit-exact when possible · In/Out trimming · resolution and frame-rate conversion · stream straight to disk for unlimited length · batch export of the whole media bin. Filenames follow player conventions (`clip.3D.HSBS.mp4`). |
| **Photos** | PNG / JPEG / WebP / JPS stills, plus **3D motion** videos (orbit, sway, dolly-zoom, swing, wigglegram) from a single photo. |
| **Live** | Convert your camera or any screen / window / tab to 3D in real time, and record the result. |

## Workflow

1. **Open** videos or photos (or drop / paste them). They land in the media bin.
2. Pick a **look** preset, then fine-tune in the inspector — every change previews instantly.
3. Scrub the timeline, set **In / Out** with `I` / `O`, check comfort in the **Parallax** view.
4. **Export** (`Ctrl+E`).

### Keyboard

| Keys | Action |
|---|---|
| `Space`, `←` `→`, `Shift+←/→`, `Home` `End` | Play/pause, frame step, ±1 s, start/end |
| `I` `O` `X` `L` `M` | In, Out, clear range, loop, mute |
| `1`–`8` | Viewer modes |
| `[` `]` / `,` `.` | Depth budget / screen plane |
| `A` `S` | Auto-converge, swap eyes |
| `F` `H` `B` `\` | Fullscreen, stats overlay, media bin, compare |
| `Ctrl+O` `Ctrl+E` `Ctrl+Z` `Ctrl+Shift+Z` `?` | Open, export, undo, redo, shortcuts |

## Browser support

Best in current **Chrome / Edge** (WebGPU + WebCodecs + stream-to-disk). Safari 17+ and Firefox 130+ work with
WebCodecs; without WebGPU inference runs on the CPU. Codec availability depends on the browser and hardware — the
export dialog only offers encoders that can actually handle the output size.

## Development

```bash
npm install
npm run dev       # http://localhost:5173 (cross-origin isolated → multi-threaded CPU inference)
npm run build     # static build in dist/ — relative paths, deployable to any sub-path
npm run preview
```

Pushing to `main` builds and publishes `dist/` to the `gh-pages` branch (see `.github/workflows/pages.yml`).

### Source layout

```
src/
  gl/        WebGL2 stereo renderer and GLSL passes (upsample, snap, dilate, blur, warp, soften, compose)
  depth/     ONNX Runtime engine, model registry & cache, temporal stabiliser
  core/      settings / presets / layouts and the frame processor glue
  media/     probing (mediabunny) and export jobs (video, stills, 3D motion)
  app/       preview controller, inspector, export dialog
  ui/        DOM helpers, controls, timeline and histogram widgets, icons
```

## Credits

- [MiDaS](https://github.com/isl-org/MiDaS) (MIT) · [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Small: Apache-2.0, Base: CC-BY-NC-4.0) via [onnx-community](https://huggingface.co/onnx-community)
- [ONNX Runtime Web](https://onnxruntime.ai) (MIT) · [Mediabunny](https://mediabunny.dev) (MPL-2.0)
- Sample photo by Rachel Michetti (CC0), from the scikit-image data set.
