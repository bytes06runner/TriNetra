# CLAUDE.md

## PROJECT

TriNetra, SIH26166, Chandrayaan-2 to lunar reference image correspondence.
Evaluation and validation pipeline, not yet a working registration system -
that is the honest current state.

## HARD RULES

- No training, no new model checkpoints, ever, unless explicitly asked. The
  fine-tuned matcher is fixed at `eloftr_lunar_finetuned_epoch7.pt`
  (`models/eloftr_lunar_finetuned_epoch7.pt`).
- Never overwrite an existing cache file, results file, or metrics.json entry.
  New work gets a new filename.
- Any cache file's metadata must come from the same source as its arrays.
  Violating this caused a real bug (metadata swap, Shiv Shakti vs Shackleton).
  Enforce with `assert_cache_provenance()`
  (defined in `scripts/verify_finetuned_checkpoint.py`).
- Every reported PASS or accepted result must carry a Delta_shuffle margin
  against at least the core controls (rot90, rot180, rot270, vflip, hflip,
  non-overlapping offset, uniform noise). No exceptions.
- Never claim ground truth. State the reference's own error (e.g. LOLA
  `lola_rms`) and note that no claim exceeds it.
- Never claim sub-pixel accuracy for cross-sensor registration. Sub-pixel is
  validated only for the estimator itself (synthetic Fourier-shift recovery)
  and for scale-gap handling under matched illumination. State this
  distinction every time sub-pixel comes up.
- Full test suite must pass before and after any change. Report the
  before/after count.
- If a result seems too good, run the negative controls before believing it.
  This project's worst bugs were things that looked like wins.
- Report negative results plainly. Do not soften or omit them.

## REPO MAP

- `src/trinetra/evaluate.py` - main evaluation CLI
- `src/module4_registration/` - gate and matching
- `results/metrics.json` - authoritative frozen metrics
- `results/UPGRADE_*.md` - dated stage reports, read before assuming
  something hasn't been tried
- `assets/real_cache/` - per-configuration cached arrays + metadata

## RUNNING TESTS

```bash
PYTHONDONTWRITEBYTECODE=1 ~/miniforge3/envs/trinetra/bin/python -m pytest tests/ -q -p no:cacheprovider
```

Run `tests/` only. The root-level `test_*.py` files are ad-hoc scripts with
module-level side effects, not part of the suite. Baseline on 2026-09-19 at
942f742: 144 passed.

## KNOWN HAZARDS

- `python -m trinetra.evaluate --all` rewrites `results/metrics.json`,
  `results/METRICS.md` and `results/RESULTS.md`. `--refine` and `--balance`
  rewrite `results/UPGRADE_U1_REFINE.md` and `results/UPGRADE_U2_BALANCE.md`.
  Never run these against `results/`. Use `--outdir` pointing at a new
  directory.
- `evaluate.py` never passes control ratios to `evaluate_flight_gate()`, so
  Criterion 6 (Delta_shuffle) is not applied in the CLI. Any gate status it
  prints is the superseded 5-criterion result. The U1/U2 reports say "PASSED"
  for this reason.
