# HTCS research notes

Planning / research artifacts that sit alongside the canonical project docs
in `docs/htcs/`. The split:

- **`docs/htcs/`** — current, code-aligned design and implementation docs.
  Updated alongside code changes (the `doc sync` commits).
- **`docs/htcs/research/`** (this folder) — planning, story, related work,
  paper drafts. Iterated independently of code.

## Files

| File | Purpose |
|---|---|
| `研究背景.md` | Problem statement + why now |
| `故事线.md` | Narrative arc the AAAI paper follows |
| `相关工作调研.md` | Literature survey (broader scope than the code-aligned `docs/htcs/相关工作调研-VLA历史压缩.md`) |
| `AAAI-论文初稿大纲.md` | Paper outline |
| `工作进展.md` | Running progress log |
| `refs/` | Reference papers (links only — PDFs untracked) |

## Naming overlap warning

`相关工作调研.md` exists both here and in `docs/htcs/相关工作调研-VLA历史压缩.md`.
The repo version is narrower (VLA-history-compression slice for the impl
doc); this one is the broader survey used for paper drafting. Don't merge
them unthinkingly — they serve different readers.
