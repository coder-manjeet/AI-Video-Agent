---
name: audio-video-production
description: Use for FFmpeg/ffprobe work — audio mixing, time-stretch, subtitles, MP4 rendering, or media validation.
---

# Audio/Video Production

FFmpeg and ffprobe 9.0.1 (essentials build) are on PATH. Use PowerShell `&` invocation. Never overwrite a finished video without explicit permission.

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
