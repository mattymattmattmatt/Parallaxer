const HF = 'https://huggingface.co/onnx-community';

/**
 * Depth model registry.
 *
 * norm:   'none'     – the graph normalises internally and expects RGB in [0, 1]
 *         'imagenet' – expects (rgb - mean) / std
 * input:  fixed [w, h] or { multiple, sizes } for models with dynamic spatial dims
 * output: 'disparity' – larger = closer (MiDaS / Depth Anything relative depth)
 */
export const MODELS = [
  {
    id: 'midas-small',
    name: 'MiDaS v2.1 Small',
    tag: 'Fast',
    description: 'Bundled with the app. Real-time on most GPUs, good for previews and long videos.',
    license: 'MIT',
    size: 66_764_249,
    url: 'models/model-small.onnx',
    local: true,
    norm: 'none',
    input: { fixed: [256, 256] },
    output: 'disparity'
  },
  {
    id: 'da2-small',
    name: 'Depth Anything V2 Small',
    tag: 'Best',
    description: 'State-of-the-art edges and far-field detail. Downloaded once from Hugging Face and cached.',
    license: 'Apache-2.0',
    size: 99_000_000,
    url: `${HF}/depth-anything-v2-small/resolve/main/onnx/model.onnx`,
    urlFp16: `${HF}/depth-anything-v2-small/resolve/main/onnx/model_fp16.onnx`,
    sizeFp16: 49_600_000,
    norm: 'imagenet',
    input: { multiple: 14 },
    output: 'disparity'
  },
  {
    id: 'da2-base',
    name: 'Depth Anything V2 Base',
    tag: 'Ultra',
    description: 'Larger backbone for maximum fidelity. Heavy download; needs a capable GPU. Non-commercial licence.',
    license: 'CC-BY-NC-4.0',
    size: 390_000_000,
    url: `${HF}/depth-anything-v2-base/resolve/main/onnx/model.onnx`,
    urlFp16: `${HF}/depth-anything-v2-base/resolve/main/onnx/model_fp16.onnx`,
    sizeFp16: 195_000_000,
    norm: 'imagenet',
    input: { multiple: 14 },
    output: 'disparity'
  }
];

/** Template for a user-supplied ONNX depth network (bytes are attached at runtime). */
export const CUSTOM_MODEL = {
  id: 'custom',
  name: 'Custom model',
  tag: 'Custom',
  description: 'Your own ONNX depth network. Output must be relative depth; use Invert depth if near is dark.',
  license: 'User supplied',
  local: true,
  norm: 'imagenet',
  input: { multiple: 14 },
  output: 'disparity'
};

/** Short-side resolutions offered for models with dynamic input. */
export const DETAIL_LEVELS = [
  { id: 'fast', label: 'Fast', short: 266 },
  { id: 'balanced', label: 'Balanced', short: 392 },
  { id: 'high', label: 'High', short: 518 },
  { id: 'ultra', label: 'Ultra', short: 700 }
];

export function getModel(id) {
  return MODELS.find((m) => m.id === id) ?? MODELS[0];
}
