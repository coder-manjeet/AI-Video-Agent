# ADR 0008 — Character Identity and Visual Consistency Foundation

- Status: Accepted
- Date: 2026-11-20

## Context

Milestone 2A delivered independent, versioned scene image generation, but it had
no reusable notion of a character. Consistency inputs (character references,
visual style, colour palette, character appearance) were metadata only and were
reported with warnings; callers had to compose such intent into the positive
prompt themselves. Milestone 2B required a backend foundation that lets one
character appear across independently generated scenes, while preserving the
167 passing tests and all working functionality.

A live survey of ComfyUI 0.46.0 (1,596 nodes) and the local model directories
showed that no reference-image conditioning path is usable:

- IP-Adapter core loader nodes are absent; only InspirePack helper nodes that
  require an `IPADAPTER_PIPE` are registered. There is no `ipadapter` model dir.
- `controlnet/`, `clip_vision/`, `photomaker/`, `insightface/` and `style_models/`
  are empty, and the corresponding loader combo lists are all `[]`.
- `ZImageFunControlnet` exists but needs a `MODEL_PATCH`; no Z-Image controlnet
  weights are installed.
- Generic image-to-image nodes (`LoadImage`, `VAEEncode`) exist, but the verified
  `z_image_turbo_text_to_image_api.json` has no image→sampler path at all.

Installing models or custom nodes was out of scope without approval, and neither
Z-Image nor LTX workflows could be modified.

## Decision

- Store characters in one atomic project-level manifest at
  `assets/characters/characters.json`, mirroring the `script/scenes.json`
  convention. Reference images live under `assets/characters/{character_id}/`.
  Deleting character metadata never deletes reference image files.
- Keep characters in typed `character_models.py` dataclasses with non-forced
  optional fields and `extra` preservation for unknown JSON keys. IDs are
  validated with the existing safe-identifier helper and must be unique.
- Store project-wide visual identity as an optional typed `visual_identity` on
  `project.json` (least disruptive: no new manifest, no per-scene duplication).
  Scenes hold only sparse `visual_identity_overrides`.
- Keep legacy `Scene.character_references` image-path strings untouched. Add
  typed `character_ids`, `character_overrides` (appearance/clothing/expression/
  action/pose) and `visual_identity_overrides`; old manifests remain valid.
- Add pure, deterministic `visual_prompt_builder.py` composition: project
  identity → character identities → scene description → character actions →
  original prompt, with merged negative prompts. It de-duplicates fragments,
  replaces base appearance/clothing with scene overrides, supports multiple or
  zero characters, and returns original prompts plus structured sections for
  debugging. Section labels are localised for a small language set with an
  English fallback.
- Integrate composition into `scene_image_service.py` as an explicit opt-in
  (`compose_prompt=True`). The default direct-prompt path is byte-for-byte
  unchanged. Composed prompts are resolved once and recorded in `job.options`,
  so retries reuse them; the original prompt and composition sections are kept
  in `job.metadata` for debugging.
- Extend validation with structured character/visual-identity/character-scene
  codes. Optional manifests mean projects without characters are unaffected.
- No new runtime dependencies, no model or custom-node installs, and no changes
  to the three MCP servers, `opencode.json`, either verified workflow, TTS,
  FFmpeg or generated media.

## Consequences

- The 167 existing tests still pass; 34 new mocked tests bring the total to
  201. Ruff and mypy are clean across 18 source files.
- Prompt composition expresses intent consistently but **does not guarantee**
  image-level identity preservation. Any future claim of guaranteed consistency
  requires real conditioning models/custom nodes, approval to install them, a
  new verified workflow and a GPU smoke test.
- Character data, visual identity and the prompt builder have no MCP tools or UI
  yet; callers use the Python services directly.
- Sparse overrides avoid duplicating project-wide metadata into every scene,
  and unknown keys survive round-trips, so future schema additions are additive.
