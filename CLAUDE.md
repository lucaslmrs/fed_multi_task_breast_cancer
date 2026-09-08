# CLAUDE.md

The project brief lives in **`AGENTS.md`**, which is shared with Codex. It is imported here:

@AGENTS.md

> **If you cannot see the imported content** — that is, you do not have sections named
> "Commands", "Architecture Overview", "Federated training" and "Comparison experiment" in
> context — then the `@AGENTS.md` import above did not resolve. **Use the Read tool on
> `AGENTS.md` right now**, before doing anything else in this repository. Do not work without it:
> this codebase produces frozen artifacts that are easy to invalidate silently.

The invariants below are mirrored from `AGENTS.md` on purpose, so that a broken import cannot cost
you the rules whose violation is unrecoverable. `scripts/sync_agent_assets.py --check` fails the
build if the two copies ever diverge, so edit them in `AGENTS.md` and re-run the sync.

<!-- GUARD-RAILS:START -->
## Non-negotiables

Violating any of these silently invalidates frozen artifacts or already-published results.

- **`Curated_BUSI_preprocessing.py` resizes with `INTER_NEAREST`. Do NOT "fix" it to `INTER_AREA`** —
  it would change the curated images and invalidate the frozen federated partition and every
  result derived from it.
- **Never build dataset paths by hand.** Derive them from `src/dataset/paths.py`
  (`processed_dir()`, `mapping_file()`, `partition_file()`).
- **Never hand-edit `mapping.csv` or `federated_mapping.csv`.** Regenerate them — they are
  deterministic given the same config. Paths inside them are relative to the repo root, so moving
  a dataset folder invalidates both.
- **Never regenerate the federated partition without explicit intent.** Every experiment arm is
  comparable only because all of them read the same frozen file.
- **Do not apply `sigmoid` before the DICE criterion** — the criterion applies it internally.
- **`numpy` must stay `<2`** (pandas/monai ABI).
- **Do not call a p-value from this study "statistically significant".** They come from paired
  Wilcoxon over 8 non-independent client × fold pairs on a single seed, and the pipeline tags them
  `exploratory_only_non_independent_client_fold_pairs`.
<!-- GUARD-RAILS:END -->

## Claude-specific notes

- Skills have one canonical copy in `.agents/skills/`. `.claude/skills/` contains relative links
  used for Claude Code discovery; never replace their targets with local copies.
- Codex discovers the same canonical directories through links in `.codex/skills/`.
- After adding, renaming, or removing a skill, run `python -m scripts.sync_agent_assets`; use
  `python -m scripts.sync_agent_assets --check` to validate both discovery trees.
