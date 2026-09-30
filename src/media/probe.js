import { Input, BlobSource, ALL_FORMATS } from 'mediabunny';

const IMAGE_EXT = /\.(jpe?g|png|webp|avif|gif|bmp|heic|heif|jfif|tiff?)$/i;
const VIDEO_EXT = /\.(mp4|m4v|mov|webm|mkv|avi|ts|m2ts|mts|ogv|3gp)$/i;

export function classifyFile(file) {
  if (file.type.startsWith('image/') || IMAGE_EXT.test(file.name)) return 'image';
  if (file.type.startsWith('video/') || VIDEO_EXT.test(file.name)) return 'video';
  return 'unknown';
}

/** Detailed container / stream metadata for a video file (via mediabunny, no decoding). */
export async function probeVideo(file) {
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  try {
    const format = await input.getFormat();
    const video = await input.getPrimaryVideoTrack();
    if (!video) throw new Error('This file has no video track.');
    const audio = await input.getPrimaryAudioTrack();
    const [duration, codec, width, height, rotation, canDecode, hdr, bitrate] = await Promise.all([
      input.computeDuration(),
      video.getCodec(),
      video.getDisplayWidth(),
      video.getDisplayHeight(),
      video.getRotation(),
      video.canDecode(),
      video.hasHighDynamicRange().catch(() => false),
      video.getAverageBitrate().catch(() => null)
    ]);
    let fps = null;
    let vfr = false;
    try {
      const m = await video.computeFrameRateMetrics({ targetPacketCount: 240 });
      fps = m.bestGuessFrameRate;
      vfr = !m.frameRateIsConstant;
    } catch {
      /* not critical */
    }
    let audioInfo = null;
    if (audio) {
      const [aCodec, channels, sampleRate, aDecode] = await Promise.all([
        audio.getCodec(),
        audio.getNumberOfChannels(),
        audio.getSampleRate(),
        audio.canDecode()
      ]);
      audioInfo = { codec: aCodec, channels, sampleRate, canDecode: aDecode };
    }
    return {
      container: format?.name ?? 'Unknown',
      mime: await input.getMimeType().catch(() => ''),
      duration,
      width,
      height,
      rotation,
      codec,
      canDecode,
      hdr,
      bitrate,
      fps: fps || 30,
      fpsKnown: !!fps,
      vfr,
      audio: audioInfo,
      size: file.size
    };
  } finally {
    input.dispose();
  }
}

export const CODEC_NAMES = {
  avc: 'H.264',
  hevc: 'HEVC',
  vp8: 'VP8',
  vp9: 'VP9',
  av1: 'AV1',
  prores: 'ProRes',
  aac: 'AAC',
  opus: 'Opus',
  mp3: 'MP3',
  vorbis: 'Vorbis',
  flac: 'FLAC',
  ac3: 'AC-3',
  eac3: 'E-AC-3'
};

export function codecName(c) {
  if (!c) return '—';
  return CODEC_NAMES[c] ?? (c.startsWith('pcm') ? 'PCM' : c.toUpperCase());
}
