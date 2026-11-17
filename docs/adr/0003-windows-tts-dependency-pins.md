# ADR 0003 — Windows TTS dependency pins

- Status: Accepted
- Date: 2026-11-17

## Context

During the controlled Chatterbox install, the dependency-resolution constraint file was not passed to the second pip command. As a result, pip upgraded `numpy` from `1.26.4` to `2.4.6` and installed `ml_dtypes 0.7.0`, which requires NumPy 2.x.

Importing Chatterbox then failed because `scikit-learn 1.9.1` could not load a compiled extension:

```text
ImportError: DLL load failed while importing _datasets_pair:
An Application Control policy has blocked this file.
```

The controlled dry run had expected `numpy 1.26.4`, `ml_dtypes 0.5.4`, and `scikit-learn 1.9.1`.

## Decision

1. Restore and retain `numpy==1.26.4`.
2. Pin `ml_dtypes==0.5.4`.
3. Pin `scikit-learn==1.5.2` instead of `1.9.1` to avoid the Windows Application Control block.
4. Verify `sklearn.metrics.roc_curve` imports successfully.
5. Accept the remaining `pip check` warnings from Chatterbox metadata about Gradio, Torch, and TorchAudio as expected consequences of the controlled install.

## Consequences

- NumPy remains compatible with the existing CUDA/Torch stack and the controlled dry run.
- `scikit-learn 1.5.2` satisfies librosa's `>=1.1.0` requirement.
- Future TTS dependency manifests must include these pins explicitly.
- The expected Chatterbox metadata conflicts must not be “fixed” by downgrading Torch or installing Gradio.
