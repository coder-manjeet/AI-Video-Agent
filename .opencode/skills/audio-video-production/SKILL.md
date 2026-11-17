---
name: audio-video-production
description: Use for FFmpeg/ffprobe work — audio mixing, time-stretch, subtitles, MP4 rendering, or media validation.
---

# Audio/Video Production

FFmpeg and ffprobe 9.0.1 (essentials build) are on PATH at `C:\ffmpeg\ffmpeg-9.0.1-essentials_build\bin`. Use PowerShell `&` invocation. Never overwrite a finished video without explicit permission. For reusable work, call `media_service.py` or the `local-media-agent` MCP tools instead of hand-writing one-off commands.

## Workflow
1. **Probe** inputs before processing: `& ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 input.wav`
2. **Prepare** narration/music/SFX as WAV; normalize levels and align cues to `scenes.json` timings.
3. **Mix** with explicit filter graphs, e.g. `& ffmpeg -i narration.wav -i music.wav -filter_complex "[1:a]volume=0.15[bg];[0:a][bg]amix=inputs=2:duration=first[out]" -map "[out]" mixed.wav`
4. **Render** to a temp file: `& ffmpeg -i video.mp4 -i mixed.wav -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k -movflags +faststart output.tmp.mp4`
5. **Validate** with ffprobe (codec, resolution, duration); only then move the temp file into place.

## Rules
- Subtitles: UTF-8 SRT with timings derived from measured narration duration (no speed API in Chatterbox — use `atempo` for post-processing if needed).
- Keep subtitle paths relative when using `-vf subtitles=...`; Windows drive letters and backslashes need escaping in filter graphs.
- Prefer stream copy for lossless trims (`-c copy`) and re-encode only when filters require it.
- Check A/V duration drift after every render; report mismatches instead of shipping silently.
- Store logs in ignored `logs/`; keep generated media out of Git.

## Reusable media service (Phase 1)

`media_service.py` implements the render workflow above with:

- `probe_media()` — ffprobe duration, resolution, FPS, codecs, sample rate, channels and size.
- `merge_narration()` — temp-file render, AAC audio, configurable volume, ffprobe validation before promotion, and `overwrite=False` by default.
- `validate_media()` — structured existence, readability, stream and duration verdicts.

Duration policies are explicit and always reported in `warnings`:

- `trim_audio`: stream-copy video; trim longer narration or pad shorter narration with silence to the video length.
- `extend_video`: freeze the final frame with `tpad` until narration ends (requires libx264 re-encode); never cut narration.

MCP tools with the same names are exposed by `media_server.py` (`local-media-agent`). Verified live output: `C:\AI-Video-Agent\test-output\first_narrated_video.mp4` (4.25 s, 768x512, h264 + AAC). Timeouts default to 180 s for ffmpeg and 30 s for ffprobe; override with `MEDIA_FFMPEG_TIMEOUT_S` / `MEDIA_FFPROBE_TIMEOUT_S`.
