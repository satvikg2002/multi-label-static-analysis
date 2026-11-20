# EMBER2024: primary analysis and repository baselines

The primary-analysis notebook remains **data understanding and feature analysis only**. A separate notebook now runs the EMBER2024 repository's binary and native behavior-tag baselines. The `run` command still performs analysis only; `run-baseline` explicitly trains the new baselines. Previous saved model bundles remain removed. New estimators exist only in memory; historical benchmark scores remain separate.

The root `pyproject.toml` and `uv.lock` configure the existing uv/Python 3.14 environment. Analysis code, data, notebooks and outputs live under `src/multi-label-static-analysis/`.

```sh
uv sync --locked
uv run --locked multi-label-static-analysis run
uv run --locked multi-label-static-analysis verify
uv run --locked multi-label-static-analysis baseline-scores
uv run --locked python -m unittest discover -s src/multi-label-static-analysis/tests -v
```

Open [the executed primary-analysis notebook](src/multi-label-static-analysis/notebooks/ember2024_primary_analysis.ipynb). Its eleven sections cover:

1. Actual source inventory and preserved reference checks.
2. Full streaming JSONL ingestion and schema/missingness audits.
3. Identity duplication, annotation disagreement, chronological partitions and exact similarity controls.
4. Class balance, annotation coverage, category support, co-occurrence and cardinality.
5. Allowlisted thrember vectorization, semantic feature names, source fingerprints, cache alignment checks and histogram normalization.
6. Exact fitting-only statistics, sparsity, tails and distributions.
7. Normalization and correlation/redundancy analysis.
8. Full-fitting ANOVA, sampled mutual information, and proposed feature subsets.
9. Fitting-only PCA, explained variance and two-dimensional visualizations.
10. Fitting-to-validation feature drift.
11. Historical baseline reference and questions for the next proposal milestone.

`run --interactive` pauses after each small code step. The 11 analysis sections now contain named, commented cells (at most 27 lines each), with shorter markdown and contained plotting helpers. The acquisition notebook has six short steps. Builders preserve this layout on regeneration. `build --output path/to/new.ipynb` generates a new unexecuted analysis notebook; replacing an existing notebook requires `--force`. The old `--baseline` training flag is deliberately unavailable. `baseline-scores` only reads the historical comparison table.

## Current data, rather than the earlier experiment

The input folder currently contains **128 files: Win64 training/test features and the challenge set**. APK/PDF fitting/test files from the earlier mixed-format experiment are no longer present. This notebook analyzes the files that exist; it does not download replacements or assume the historical population.

It ingested 1,286,315 raw rows and retained 646,079 eligible records after SHA, exact TLSH, and exact-vector controls. The fitting period contains **400,000 Win64 records**, evenly split benign/malicious. Of 200,000 fitting malware records, **101,312 lack behavior annotations**. Six mapped roles have fitting support; Botnet Agent has no matching tags. Tags remain weak, incomplete proxies.

The initial representation intentionally examines the **696 shared static inputs**, with full semantic names. PE-specific Win64 features are a separate next-step decision. There are 695 variable fitting columns and one constant column. ANOVA uses all fitting records; MI uses 6,000 deterministic fitting rows; correlation and PCA use 20,000. Normalization statistics use all fitting rows. PCA needs 308 components for 95% of sampled standardized variance; this is not a predictive-performance result.

## Repository baseline notebook

Open [the separate baseline notebook](src/multi-label-static-analysis/notebooks/ember2024_repository_baseline.ipynb). Eight sections cover provenance, re-ingestion, vector-cache alignment and labels, binary training, behavior OvR training, threshold freezing, evaluation, and visualization. All code cells have short comments and at most 35 lines.

```sh
uv run --locked multi-label-static-analysis run-baseline
uv run --locked multi-label-static-analysis verify-baseline
```

`run-baseline --interactive` pauses after each code step. `build-baseline --output path/to/new.ipynb` regenerates an unexecuted notebook; `--force` is required to replace one.

The detector uses the pinned upstream 500-round LightGBM configuration. Each native behavior tag with at least ten fitting positives gets an independent LightGBM head, up to 100 rounds with ten-round validation-AUC patience. Tags absent from validation use the full 100 rounds, explicitly recorded. Inputs retain **all 696 columns**, including the constant, with no scaling, selection or PCA. The native tag vocabulary is learned only from fitting data; it differs from the proposal's seven mapped categories. This repository reference is separate from the proposal's later logistic-regression baseline and broader pipeline search.

This is a **local configuration adaptation**, not a reproduction of the full published benchmark: fitting is Win64 only, the representation is shared 696-feature rather than PE-specific 2,568-feature, and validation is chronological rather than upstream random stratification. Only available categorical indices 2–6 are passed to LightGBM; CPU threads and logging are bounded. Challenge evaluation includes all available challenge formats and separately reports detection by format.

Results live in `src/multi-label-static-analysis/results/repository_baseline/`: binary and per-tag metrics, annotation coverage, per-head training rounds/times, configuration/provenance, validation history, threshold selection, two plots, and row-aligned held-out probabilities. Direct OvR and two detector-gated variants are compared without additional model fitting. The repository threshold 0.5 remains primary; a separate detector threshold is calibrated to at most 1% false positives on calibration benigns. Behavior thresholds stay at 0.5.

Empty behavior lists are treated as zeros to match upstream encoding. Because annotations are incomplete, metrics measure agreement with weak labels, not complete behavioral ground truth. A second scoring view excludes unannotated malware; even that view cannot establish that every absent tag is a true negative. Undefined challenge-only binary AUC/AP/FPR are left blank. Macro AUC/AP report the number of labels with usable targets. Previously inspected holdouts are not presented as a newly blind evaluation. Existing analysis and historical files are checked for unchanged contents after the baseline run.

## Files and results

- [Findings](src/multi-label-static-analysis/results/primary_analysis/findings.md), [summary](src/multi-label-static-analysis/results/primary_analysis/analysis_summary.json), and [figures](src/multi-label-static-analysis/results/primary_analysis/figures).
- `results/primary_analysis/`: source/label audits, feature dictionary/statistics, correlation pairs, rankings, candidate subsets, PCA tables and validation drift.
- `results/historical/ember2024_baseline_comparison.csv`: the earlier mixed-format comparison. Its folder disappeared after analysis execution; the table was restored from the verbatim earlier CSV report recorded in this chat and checked against the table embedded in the executed notebook. The manifest records this recovery. Older supporting score reports are no longer present. No scores were recomputed.
- `artifacts/`: input matrix/cache and notebook runtime files. There are no saved classifier weights. New baseline thresholds are confined to the baseline results directory.
- `multi_label_static_analysis/notebook_builders/build_analysis_notebook.py`: reviewable notebook source.
- `multi_label_static_analysis/data_analysis.py`: tested streaming, vectorization, semantic-name and exact-ANOVA helpers.
- `multi_label_static_analysis/repository_baseline.py`: chronological input checks, pinned configuration loading, LightGBM training, batched scoring, metrics and baseline verification.
- `multi_label_static_analysis/notebook_builders/build_repository_baseline.py`: reproducible source for the new baseline notebook.

Filter rankings and candidate top-64/128/256 subsets remain exploratory proposals. No best research pipeline or feature subset has been established. The new baseline leaves those candidates unused so later experiments can compare them against the full shared representation. A single multi-head neural network and the proposal's seven-role classifiers remain later incremental experiments.

Notebook readability checks compare the rerun with the original analysis: all 22 numerical CSV/JSON files and all nine rendered plots are byte-identical. Acquisition extraction/reuse is tested with offline fixtures; actual transfers are not repeated during a readability review.

## Minimal release

Only active command, notebook-generation, analysis/training, acquisition, feature-extraction and test modules remain. Archived experiment builders, the old serialization helper, and unused upstream model/download wrappers have been removed. The vendored feature extractor stays unchanged so existing vector-cache fingerprints remain valid.

The dependency list contains the libraries used by those components. Notebook execution and generation dependencies are installed with the project, so its commands also work without a development dependency group. Tests use Python's built-in `unittest`.

`uv build` creates a small code wheel and a source archive containing the code, tests, three notebooks, pinned configurations, lockfile and historical reference scores. Raw data, vector caches, generated analysis/baseline results, bytecode and desktop metadata are excluded from release archives. These exclusions preserve local files; acquire data and run the notebooks to regenerate outputs from a fresh source release. Running the full test suite requires the project's local data and generated reference outputs.
