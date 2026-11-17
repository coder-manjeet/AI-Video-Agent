---
name: local-ai-tts
description: Use for Chatterbox/TTS work, PyTorch+CUDA environment questions, or WAV/audio generation tasks. Do not use for ComfyUI video work.
---

# Local AI TTS

`.venv-tts` is the **only** TTS environment: Python 3.11.9, `torch==2.10.0+cu130`, `torchaudio==2.10.0+cu130`, `numpy==1.26.4`, CUDA 13.0, RTX 5060 Ti (`sm_120`). `chatterbox-tts==0.1.7` is installed, the model weights are cached, and short English/Hindi GPU inference passed. See `docs/PROJECT_STATUS.md`.

## Environment rules
- Never run plain `pip install chatterbox-tts`: it pins torch 2.6.0 and would break this GPU stack. The completed approach was `--no-deps` plus explicit compatible dependencies; Gradio was skipped.
- Never downgrade torch, torchaudio, or CUDA. Do not install TTS packages into `.venv`.
- Write WAV with `soundfile.write(...)`; `torchaudio.save(...)` needs TorchCodec, which is absent.

## Generation workflow
1. Use official API only (`resemble-ai/chatterbox`): `audio_prompt_path` for a licensed reference voice, language `en` or `hi`. Hinglish has no official code — evaluate empirically.
2. Tune `exaggeration`, `cfg_weight`, `temperature`, `repetition_penalty`, `min_p`, `top_p`. There is no speed parameter; change speed via post-processing (see `audio-video-production` skill).
3. Before a GPU stage, check `GET http://127.0.0.1:8000/system_stats`. An empty queue can still leave ~11.5 GB VRAM held by ComfyUI. If free VRAM is low, call `POST http://127.0.0.1:8000/free` with `{"unload_models":true,"free_memory":true}`, verify free VRAM, then proceed. ComfyUI reloads models on the next render.
4. Load the model once per run with `ChatterboxMultilingualTTS.from_pretrained(torch.device("cuda"))`; the installed version has no `t3_model` argument. Release GPU memory when generation is complete; never overlap TTS with a ComfyUI render on the 16 GB GPU.
5. Save WAV to a project-scoped temp path; extract duration/sample rate metadata; do not overwrite approved narration.

## Verification
Smoke test with a short sentence: check non-silent audio, expected sample rate, plausible duration, and `torch.cuda.is_available()` before/after. Keeping reference recordings out of Git is mandatory.

Passed baseline: English 24 kHz / 3.52 s / peak `0.921326` / RMS `0.106031`; Hindi 24 kHz / 4.52 s / peak `0.870392` / RMS `0.113988`. TTS process memory after cleanup was `9.12 MiB`; CUDA remained available. Ignore harmless first-run warnings about HF symlinks, unauthenticated public downloads, missing `pkuseg` for Chinese, and deprecated `sdp_kernel`.
