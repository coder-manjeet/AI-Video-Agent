# Project Status

Last updated: 2026-11-17

## Current milestone

Milestone 5 — English and Hindi GPU smoke tests — is complete.

Chatterbox is installed in `.venv-tts`, the model weights are cached, and short English and Hindi CUDA inference both passed. The next milestone is the TTS core service.

## Completed work

- Audited the existing project, Git state, MCP integration, and LTX workflow.
- Created `.gitignore` and `README.md`.
- Verified ignore rules in a temporary Git repository outside the project.
- Confirmed ComfyUI MCP status is working.
- Checked official Chatterbox documentation and dependency metadata.
- Proved that normal Chatterbox installation would downgrade Torch/TorchAudio.
- Recorded decisions in:
  - [ADR 0001](adr/0001-preserve-mcp-integration.md)
  - [ADR 0002](adr/0002-chatterbox-dependency-strategy.md)
  - [ADR 0003](adr/0003-windows-tts-dependency-pins.md)
- Pinned dev tools in `requirements-dev.txt`.
- Installed `chatterbox-tts==0.1.7` with `--no-deps`.
- Installed explicit compatible dependencies; Gradio was not installed.
- Restored `numpy==1.26.4` after an unconstrained resolution upgraded it.
- Downgraded `scikit-learn` to `1.5.2` after a Windows Application Control DLL block.
- Pinned `ml_dtypes==0.5.4`.
- Downloaded the public Chatterbox weights once into the Hugging Face cache (~3.2 GB).
- Loaded `ChatterboxMultilingualTTS.from_pretrained(torch.device("cuda"))` successfully.
- Generated one short English WAV (`language_id="en"`) and one short Hindi WAV (`language_id="hi"`) sequentially.
- Freed ComfyUI VRAM with `POST /free` before the successful retry; ComfyUI remained connected and reloads models on the next render.
- Released the TTS model and cleared the CUDA cache after generation.

## Verification results

| Check | Result |
|---|---|
| `import chatterbox.tts` | OK |
| `import chatterbox.mtl_tts` | OK |
| `torch` | `2.10.0+cu130` |
| `torchaudio` | `2.10.0+cu130` |
| `numpy` | `1.26.4` |
| CUDA available | `True` |
| GPU | RTX 5060 Ti, `sm_120` |
| CUDA kernel smoke test | Passed |
| `en` / `hi` language codes | Present |
| English GPU inference | Passed — 24 kHz, 3.52 s, peak `0.921326`, RMS `0.106031`, non-silent |
| Hindi GPU inference | Passed — 24 kHz, 4.52 s, peak `0.870392`, RMS `0.113988`, non-silent |
| WAV files re-read with `soundfile` | Both valid, matching generated metadata |
| TTS process VRAM after cleanup | `9.12 MiB` allocated; `cuda_after=True` |
| `pip check` | Only expected Chatterbox metadata conflicts |

Expected `pip check` conflicts:

```text
chatterbox-tts 0.1.7 requires gradio, which is not installed.
chatterbox-tts 0.1.7 requires torch==2.6.0, but you have torch 2.10.0+cu130.
chatterbox-tts 0.1.7 requires torchaudio==2.6.0, but you have torchaudio 2.10.0+cu130.
```

These are intentional. Do not resolve them by downgrading Torch or installing Gradio.

## Blockers and open decisions

- Hinglish remains untested.
- No TTS service, MCP tools, or WAV metadata utilities exist yet.
- No test suite exists yet.
- Git repository, remote, and commits are still deferred.

## Environment rules

- `.venv` is for MCP and dev tooling.
- `.venv-tts` is the only TTS environment.
- Never mix or recreate either environment without approval.
- Never upgrade, downgrade, or replace Torch, TorchAudio, or CUDA in `.venv-tts`.
- Never run TTS and ComfyUI GPU jobs at the same time.
- An empty ComfyUI queue is not enough: an idle ComfyUI session can retain ~11.5 GB VRAM. Check `GET /system_stats` before a TTS GPU job; if free VRAM is low, call `POST /free` with `{"unload_models":true,"free_memory":true}` and verify free VRAM before proceeding. ComfyUI reloads its models on the next render.
- `.env` files are gitignored and not auto-loaded; set variables in the PowerShell session.

## Next action

Implement the TTS core service:

1. add a project-scoped module that loads `ChatterboxMultilingualTTS` once per run
2. expose English and Hindi generation with configurable `exaggeration`, `cfg_weight`, sampling parameters, and an optional licensed `audio_prompt_path`
3. write WAV with `soundfile.write(...)` and return sample rate, duration, peak, and RMS metadata
4. check/free ComfyUI VRAM before every GPU stage, as described above
5. add unit tests with mocked model/audio I/O; keep GPU inference as a manual smoke test
6. keep all business logic out of `@mcp.tool()` bodies; add MCP tools only after the service is tested

The model weights are now cached. No further Hugging Face download is needed for subsequent runs.
