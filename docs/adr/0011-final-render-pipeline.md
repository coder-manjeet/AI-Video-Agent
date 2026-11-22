# ADR 0011 — Final Render Pipeline and Render MCP Server

- Status: Accepted
- Date: 2026-11-22

## Context

Earlier milestones produced scene videos, narration audio and the audio
production coordinator, but a caller still had to assemble those pieces
manually. The remaining backend work needed four things: external subtitles
generated from scene narration, a persistent job model with restart recovery,
a strict final MP4 assembly and QA stage, and a thin MCP surface that never
runs FFmpeg on the event loop. Existing services, workflows, TTS internals and
MCP registrations had to stay intact, and no generated media could be
overwritten.

## Decision

- Keep every production step in MCP-independent modules:
  `subtitle_models.py`, `subtitle_renderer.py` and `subtitle_service.py` for
  SRT/VTT generation and optional libass burn-in; `render_job_models.py` and
  `render_job_store.py` for persistent jobs; `final_renderer.py` for assembly;
  `final_qa.py` and `render_report.py` for validation and reports; and
  `render_orchestrator.py` for sequencing them.
- Generate external SRT and VTT tracks by default. Sentence granularity
  distributes a scene's duration across sentences by character count; no word
  timestamps are invented. English, Hindi and Hinglish narration text is
  preserved verbatim. Burn-in is opt-in and only applied when FFmpeg libass and
  a verified font are available; otherwise the external files remain.
- Persist render jobs as atomic JSON under a dedicated store root. Creation
  refuses duplicate active jobs, duplicate job IDs and duplicate render IDs.
  A global file lock plus reentrant in-process locks serialise mutations
  across threads and processes. Jobs left active after a restart are marked
  failed (or cancelled when cancellation was requested) with a structured
  error.
- Assemble the final MP4 with H.264 video, AAC 192 kbps / 48 kHz / stereo
  audio, 1920x1080, 24 FPS, yuv420p and MP4 by default. The authoritative
  video timeline is extended with a final-frame `tpad` only when audio exceeds
  it beyond tolerance. Narration is never truncated and `-shortest` is never
  used. Rendering happens to a temporary file and is promoted only after
  metadata verification; collisions are refused.
- Separate fatal QA errors from warnings. A job can only reach `completed`
  after QA passes; validation summaries are stored on the job. Production
  reports are written as `render_report.json` and `render_report.md` and use
  project-relative paths only.
- Add `render_server.py` as the fifth MCP entry, `local-render-agent`, with six
  thin tools: `create_render_job`, `get_render_job`, `list_render_jobs`,
  `cancel_render_job`, `get_render_report` and `validate_render_output`. Job
  creation returns a persistent job ID immediately and runs the orchestrator
  with `asyncio.to_thread`; FFmpeg and QA never run on the MCP event loop.
- Orchestrator stages are reported coarsely and honestly: sub-percent FFmpeg
  progress is not invented. Cancellation is cooperative and observed only
  between stages; a running FFmpeg process is not killed and intermediates are
  preserved for diagnosis. The orchestrator starts no ComfyUI or TTS GPU work.
- Back up `opencode.json` before registration and leave the four existing MCP
  entries unchanged.

## Consequences

- A single MCP call now produces a final, validated MP4 plus external
  subtitles and a production report, while clients poll a stable job ID.
- Sentence-level subtitle timings are estimates, not word alignment. Human
  review is still needed for precise cue timing.
- Subtitle burn-in depends on the local libass filter and an installed font;
  external subtitle files are always available as the fallback.
- Restart recovery fails jobs that were active when a previous process exited;
  callers must create a new job rather than resume one in place.
- Stale cross-process lock files are removed by mtime/timeout, so a freshly
  orphaned lock can block up to the configured lock timeout.
- The render job locks coordinate render jobs only. They do not coordinate
  ComfyUI or TTS GPU work, so the one-GPU-job-at-a-time rule remains manual.
- The full suite is 633 tests; Ruff and both Mypy scopes pass, and all five MCP
  servers connect with `local-render-agent` exposing six tools.
