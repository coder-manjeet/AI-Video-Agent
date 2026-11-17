---
name: git-github-workflow
description: Use for Git status/diffs, commits, branches, releases, or GitHub CLI questions. Never perform remote operations without approval.
---

# Git / GitHub Workflow

The project is **not** a Git repository and has no remote. `gh` is not installed; install only when a repository URL and visibility are confirmed.

## Everyday commands (PowerShell)
```powershell
git status --short
git diff --stat
git add <specific-files>
git commit -m "feat: add narration duration helper"
```

## Rules
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`. One logical change per commit.
- Branches (after init): `feature/<name>`, `fix/<name>`, `docs/<name>` from the default branch.
- Review `git status --short` and `git diff --stat` before staging; stage explicit files, never `git add -A` blindly.
- Never commit secrets, `.env`, tokens, personal audio, weights (`*.safetensors`, `*.pt`, ...), or generated media — `.gitignore` covers these.
- No `git init`, remote add, push, pull, force push, tag, or hook changes without explicit user approval.
- Releases come after tests + media validation; use annotated tags and a short changelog entry (see `documentation` skill).
