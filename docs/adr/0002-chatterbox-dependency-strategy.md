# ADR 0002 — Chatterbox dependency strategy

- Status: Accepted
- Date: 2026-11-17

## Context

Chatterbox Multilingual TTS is the selected local TTS engine for English and Hindi narration.

The project already has a working GPU stack in `.venv-tts`:

```text
Python 3.11.9
torch==2.10.0+cu130
torchaudio==2.10.0+cu130
numpy==1.26.4
CUDA 13.0
RTX 5060 Ti, sm_120
```

Chatterbox `0.1.7` declares, for Python versions below 3.14:

```text
torch==2.6.0
torchaudio==2.6.0
gradio==6.8.0
```

A read-only pip dry run confirmed that a normal install would:

- replace `torch 2.10.0+cu130` with `2.6.0`
- replace `torchaudio 2.10.0+cu130` with `2.6.0`
- downgrade `sympy 1.14.0` to `1.13.1`
- install Gradio and roughly 100 packages

A second dry run with `torch==2.10.0+cu130` constrained failed with `ResolutionImpossible`, confirming the conflict.

A controlled dry run using explicit compatible dependencies, without Torch pins and without Gradio, succeeded with 79 packages and no Torch/TorchAudio entries.

## Decision

1. Never run plain `pip install chatterbox-tts` in `.venv-tts`.
2. Preserve `torch==2.10.0+cu130` and `torchaudio==2.10.0+cu130`.
3. Install `chatterbox-tts==0.1.7` with `--no-deps`.
4. Install the remaining Chatterbox dependencies explicitly, excluding:
   - Torch
   - TorchAudio
   - Gradio
5. Use `soundfile.write(...)` for WAV output because `torchaudio.save(...)` requires TorchCodec, which is absent.
6. Default the watermark dependency to PyPI `resemble-perth==1.0.1` unless a smoke test proves Git master `1.1.0` is required.
7. Run only one GPU job at a time; never overlap TTS with ComfyUI.
8. Keep reference recordings, generated audio, and model weights out of Git.

## Consequences

- pip may report a dependency conflict because Chatterbox metadata still pins Torch 2.6.0. This is expected and must be documented rather than “fixed” by downgrading.
- The installation requires an explicit dependency list and post-install verification.
- Gradio UI features are intentionally unavailable; only programmatic TTS is needed.
- Compatibility with Torch 2.10 is not proven until an actual GPU smoke test passes.
- If `resemble-perth 1.0.1` lacks required behavior, switch deliberately to Git master and record a follow-up ADR.
