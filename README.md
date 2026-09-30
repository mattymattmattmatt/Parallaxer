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
| **Export** | WebCodecs hardware encoding to MP4 / MOV / WebM / MKV with H.264, HEVC, AV1 or VP9 · original audio copied bit-exact when possible · In/Out trimming · resolution and frame-rate conversion · stream straight to disk for unlimited length · pause / resume, and **Stop & save** to keep everything rendered so far as a playable file (audio trimmed to match) · batch export of the whole media bin · **parallel processing** (below). Filenames follow player conventions (`clip.3D.HSBS.mp4`). |
| **Photos** | PNG / JPEG / WebP / JPS stills, plus **3D motion** videos (orbit, sway, dolly-zoom, swing, wigglegram) from a single photo. |
| **Live** | Convert your camera or any screen / window / tab to 3D in real time, and record the result. |
| **VR** | *View in VR* (WebXR) puts the live stereo pair on a virtual cinema screen in your headset — each eye sees its own synthesised view, no export needed. |
| **Bring your own model** | Load any ONNX depth network (NCHW RGB in, relative depth out) for the session, with ImageNet or 0–1 input and ×14 / ×32 size snapping. |
| **App** | Installable PWA that works offline after the first visit; open video and image files straight from the OS once installed. Smooth-playback preview keeps video at full frame rate while slower models catch up. |

## Workflow

1. **Open** videos or photos (or drop / paste them). They land in the media bin.
2. Pick a **look** preset, then fine-tune in the inspector — every change previews instantly.
3. Scrub the timeline, set **In / Out** with `I` / `O`, check comfort in the **Parallax** view.
4. **Export** (`Ctrl+E`).

### Export speed

Exports keep several frames in flight: while one frame is being stabilised, rendered and encoded, the next ones
are already being decoded and run through the depth model in **background workers** (each with its own model
session). Frames are always finished strictly in order, so the file is the same as a one-frame-at-a-time export.

*Processing speed* in the export dialog:

- **Auto** (default) starts one worker, measures, and adds another only while it makes the export at least 10 %
  faster. It never goes above a safe ceiling for the machine (CPU threads, device memory, model size) and gives
  back any worker that doesn't help.
- **Standard** runs depth in the page. It is the lowest-memory option.
- **2× / 3× / 4×** start that many workers right away. Counts above the machine's safe limit are disabled.

The progress view shows each worker's state and speed, plus a plain-language status line. If a worker can't
start or crashes, its frames move to another worker. If none are left, the export carries on in the page and
still completes.

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
  depth/     ONNX Runtime engine, model registry & cache, temporal stabiliser, background worker pool
  core/      settings / presets / layouts and the frame processor glue
  media/     probing (mediabunny) and export jobs (video, stills, 3D motion)
  app/       preview controller, inspector, export dialog
  ui/        DOM helpers, controls, timeline and histogram widgets, icons
```

## Credits

- [MiDaS](https://github.com/isl-org/MiDaS) (MIT) · [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Small: Apache-2.0, Base: CC-BY-NC-4.0) via [onnx-community](https://huggingface.co/onnx-community)
- [ONNX Runtime Web](https://onnxruntime.ai) (MIT) · [Mediabunny](https://mediabunny.dev) (MPL-2.0)
- Sample photo by Rachel Michetti (CC0), from the scikit-image data set.
