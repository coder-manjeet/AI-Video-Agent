---
name: documentation
description: Use when updating README, AGENTS.md, PROJECT_STATUS, ADRs, or changelogs, or when documenting a new feature.
---

# Documentation

Know the role of each file; do not copy the same content into several places.

- `README.md` — user-facing overview, setup, architecture, current state. Keep commands PowerShell-only.
- `AGENTS.md` — short, always-loaded project rules. Do not grow it into a manual; move detail into skills.
- `docs/PROJECT_STATUS.md` — current milestone, completed work, blockers, next action. Update in the same change that moves a milestone.
- `docs/adr/` — one short decision per file, named `NNNN-slug.md`.
- `CHANGELOG.md` — create only when releases start; newest first, `Added / Changed / Fixed`, dated sections.

## ADR format
```markdown
# ADR NNNN — Title
- Status: Proposed | Accepted | Superseded
- Date: YYYY-MM-DD
## Context
## Decision
## Consequences
```

## Rules
- Document observable behaviour and verification steps, not implementation guesses.
- No secrets, personal paths/recordings, or model weights in docs; use environment variable names.
- When changing a tool's interface, update its README/MCP section and its skill description in the same change.
