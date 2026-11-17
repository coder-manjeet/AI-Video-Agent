# ADR 0005 — Independent FFmpeg media service and explicit duration policies

- **Status:** Accepted
- **Date:** 2026-11-18

## Context

Phase 1 needs a reusable way to combine a ComfyUI-generated video with a
Chatterbox narration WAV and produce a validated MP4. FFmpeg/ffprobe 9.0.1
(essentials build) are installed on PATH. The existing `local-video-agent`
and `local-tts-agent` MCP servers, the LTX workflow, and both virtual
environments must remain unchanged. Generated clips are short (~4 s) while
narration can be longer, so a naïve `-shortest` mux would silently cut speech.

## Decision

Add a third, independent server and keep FFmpeg logic out of MCP tool bodies.

- `media_service.py` owns all `ffmpeg`/`ffprobe` subprocess calls. It depends
  only on the standard library, runs under `.venv`, and returns plain
  JSON-serializable dictionaries. It does not import FastMCP, Torch or
  Chatterbox.
- `media_server.py` (`local-media-agent`) exposes `probe_media`,
  `merge_narration` and `validate_media`. Tool bodies validate paths, move
  blocking calls to `asyncio.to_thread`, and convert expected failures to
  structured `status="error"` results.
- `merge_narration` renders to a hidden temporary MP4, validates it with
  ffprobe, and only then promotes it to the requested output path. Existing
  outputs are refused unless `overwrite=True`; a second existence check
  guards against a race during rendering.
- Duration differences are never silent. `trim_audio` matches audio to the
  video (stream-copying video where possible, padding short narration with
  silence); `extend_video` freezes the last frame with `tpad` until narration
  ends. Every trim, pad or remaining gap is returned in timing fields and
  `warnings`.
- Audio is encoded as AAC at 192 kb/s; video is stream-copied when no filter
  requires re-encoding, otherwise re-encoded with libx264 `crf=18`,
  `preset=medium`.
- `opencode.json` was backed up before registering the third server. No
  packages were installed, upgraded or downgraded.

## Consequences

- Both existing MCP servers and their business logic remain untouched; all
  three servers are connected simultaneously under `.venv`.
- Rendering is synchronous per request and bounded by configurable timeouts
  (`MEDIA_FFMPEG_TIMEOUT_S`, `MEDIA_FFPROBE_TIMEOUT_S`). Large renders block a
  worker thread, so concurrency and queueing are future work.
- `extend_video` necessarily re-encodes video; a small quality/encoding cost
  is accepted to avoid truncating narration.
- Validation checks stream presence, basic metadata and duration within a
  tolerance. It cannot judge perceptual A/V sync or audio content; manual
  review is still required for final delivery.
- Music, subtitles, transitions and multi-scene rendering remain out of scope
  for this milestone.
