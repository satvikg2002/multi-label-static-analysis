"""Build the incremental binary/native-behavior baseline notebook."""

import argparse
from pathlib import Path

import black
import nbformat as nbf

from multi_label_static_analysis.config import project_paths


def build(output=None):
    p = project_paths()
    out = (
        Path(output)
        if output
        else p.project / "notebooks/ember2024_repository_baseline.ipynb"
    )
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "name": "python3",
        "display_name": "Python (uv project)",
        "language": "python",
    }
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(stage, title, source):
        cell = nbf.v4.new_code_cell(
            black.format_str(
                "# " + title + "\n" + source, mode=black.Mode(line_length=88)
            )
        )
        cell.metadata.update(stage=stage, title=title, stage_end=False)
        cells.append(cell)

    md(
        """# EMBER2024 repository baselines: binary and multi-label

Run cells in order after the primary-analysis notebook. This experiment fits the repository's LightGBM detector and native behavior-tag OvR heads. It uses **all 696 shared inputs**, without scaling, feature selection or PCA. Estimators stay in memory; scores, diagnostics and plots are saved separately from previous results."""
    )
    md(
        """## 1. Scope and reproducibility

Sources: pinned [model implementation](https://github.com/FutureComputing4AI/EMBER2024/blob/0ef753e81d98bf209f71b03cd331dfc190b5b54d/src/thrember/model.py) and [example configurations](https://github.com/FutureComputing4AI/EMBER2024/tree/0ef753e81d98bf209f71b03cd331dfc190b5b54d/examples). This is a **local adaptation**, using Win64 fitting data, shared features and chronological validation. Upstream uses 2,568 PE inputs and random stratified validation. Results therefore do not reproduce the paper's full benchmark."""
    )
    code(
        "1_scope",
        "Import the baseline tools",
        """
import gc
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import roc_curve, precision_recall_curve
from multi_label_static_analysis.config import project_paths
from multi_label_static_analysis.data_analysis import stream_metadata
from multi_label_static_analysis.repository_baseline import (
    canonical_metadata, checked_vectors, behavior_targets, repository_configs,
    fit_binary, predict_rows, fit_behaviors, low_fpr_threshold,
    binary_metrics, multilabel_metrics, protected_hashes,
)
""",
    )
    code(
        "1_scope",
        "Create an isolated results directory",
        """
P = project_paths()
REPORT = P.project / 'results/repository_baseline'
FIGURES = REPORT / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
protected_before = protected_hashes(P)
binary_config, tag_config, provenance = repository_configs(P, threads=4)
environment = {name: version(name) for name in
               ['lightgbm', 'numpy', 'pandas', 'scikit-learn', 'nbformat']}
environment['python'] = platform.python_version()
sources = sorted(P.data.glob('*.jsonl'))
assert sources, 'No input files; complete data acquisition first.'
display(pd.DataFrame({'binary': binary_config, 'behavior': tag_config}))
print('Source files:', len(sources), '| Upstream revision:', provenance['revision'])
print('PASS: upstream configurations verified; previous reports protected.')
""",
    )
    md(
        """## 2. Ingestion and chronological partitions

Re-read every local JSONL record and hash its contents. Reconstruct the primary notebook's row order: SHA deduplication, first 40 training weeks for fitting, next six for validation, final six for calibration, then official test. Exact TLSH matches are purged across partitions. The challenge cohort is separate and contains multiple formats."""
    )
    code(
        "2_ingestion",
        "Read and validate the raw metadata",
        """
rows, profile, source_sha, schema_issues, parse_errors = stream_metadata(sources)
assert not schema_issues and not parse_errors
raw_rows = len(rows)
meta = canonical_metadata(rows)
del rows
gc.collect()
assert meta.sha256.is_unique
print('Raw rows:', raw_rows, '| Canonical rows:', len(meta))
""",
    )
    code(
        "2_ingestion",
        "Inspect the retained time windows",
        """
population = meta.groupby(['split', 'file_type']).agg(
    n=('sha256', 'size'), start=('date', 'min'), end=('date', 'max'))
display(population)
population.to_csv(REPORT / 'canonical_population.csv')
print('PASS: ingestion and chronological identity controls passed.')
""",
    )
    md(
        """## 3. Inputs and native multi-label targets

Verify the 696-column thrember cache against the new source hash and independently regenerated raw rows; purge identical vectors across partitions. LightGBM accepts raw counts, frequencies and missing values, so no normalization is fitted. Freeze behavior tags with at least ten fitting positives. These native tags differ from the proposal's seven mapped roles."""
    )
    code(
        "3_inputs",
        "Verify vector alignment and exact-vector controls",
        """
X, idx, vector_audit = checked_vectors(P, meta, source_sha)
y = meta.malicious.to_numpy(dtype=np.int8)
partition_counts = pd.Series({s: len(ii) for s, ii in idx.items()}, name='eligible_rows')
partition_counts.to_csv(REPORT / 'eligible_partition_counts.csv')
display(partition_counts)
display(pd.Series(vector_audit))
assert set(meta.iloc[idx['fit']].file_type) == {'Win64'}
assert X.shape[1] == 696
""",
    )
    code(
        "3_inputs",
        "Freeze the fitting-only behavior vocabulary",
        """
labels, Y, support = behavior_targets(meta, idx['fit'], class_min=10)
annotated = meta.tags.map(len).gt(0).to_numpy()
coverage_records = []
for split, ii in idx.items():
    support[split + '_positive'] = Y[ii].sum(axis=0)
    unknown = sum(t not in labels for tags in meta.iloc[ii].tags for t in tags)
    coverage_records.append({'split': split, 'n': len(ii),
        'malware_without_tags': int(((y[ii] == 1) & ~annotated[ii]).sum()),
        'out_of_vocabulary_tag_occurrences': unknown})
support.to_csv(REPORT / 'behavior_support.csv', index=False)
coverage = pd.DataFrame(coverage_records)
coverage.to_csv(REPORT / 'annotation_coverage.csv', index=False)
display(support)
display(coverage)
print('PASS: all inputs retained; native tag vocabulary fixed before training.')
""",
    )
    md(
        """## 4. Binary baseline

Train the pinned 500-round, 64-leaf LightGBM detector once. The validation window is diagnostic; binary training does not stop early or search configurations. Only categorical indices 2–6 exist in this representation; upstream PE-only indices 701–702 are omitted. Four CPU threads bound resource use. Calibration and test labels never enter training."""
    )
    code(
        "4_binary",
        "Fit the repository detector configuration",
        """
binary_model, binary_history, binary_seconds = fit_binary(X, y, idx, binary_config)
binary_rounds = binary_model.current_iteration()
assert binary_rounds == binary_config['num_iterations'] == 500
history = pd.DataFrame(binary_history['validation'])
history.index = np.arange(1, len(history) + 1)
history.index.name = 'iteration'
history.to_csv(REPORT / 'binary_validation_history.csv')
display(history.tail())
""",
    )
    code(
        "4_binary",
        "Score each held-out partition in batches",
        """
binary_scores = {s: predict_rows(binary_model, X, ii)
                 for s, ii in idx.items() if s != 'fit'}
assert all(len(binary_scores[s]) == len(idx[s]) for s in binary_scores)
del binary_model
gc.collect()
print(f'Binary fit time: {binary_seconds:.1f} seconds')
print('PASS: 500-round detector fitted; finite held-out scores generated.')
""",
    )
    md(
        """## 5. Native behavior OvR baseline

Fit one binary LightGBM per supported behavior tag: up to 100 rounds, 64 leaves, minimum leaf size ten, and ten-round validation-AUC patience. If a rare tag has no validation positives, use all 100 rounds and record the exception. As upstream, empty behavior lists encode zeros; these are incomplete annotations, not verified absence."""
    )
    code(
        "5_multilabel",
        "Fit and score the independent behavior heads",
        """
behavior_scores, training = fit_behaviors(X, Y, labels, idx, tag_config)
training.to_csv(REPORT / 'behavior_training.csv', index=False)
assert len(training) == len(labels)
assert training.best_iteration.between(1, 100).all()
display(training)
print('PASS: all native behavior heads trained; validation exceptions recorded.')
""",
    )
    md(
        """## 6. Freeze thresholds and inspect validation

The repository-style baseline uses probability **0.5**. Also select a binary threshold allowing at most 1% false positives on calibration benigns; tie handling is conservative. Behavior thresholds remain 0.5. Validation and calibration scores are development diagnostics. There is no feature/configuration search or refitting after inspecting held-out performance."""
    )
    code(
        "6_thresholds",
        "Select and freeze the optional detector operating point",
        """
cal = idx['calibration']
calibrated_threshold = low_fpr_threshold(y[cal], binary_scores['calibration'], .01)
thresholds = {'repository_0.5': 0.5, 'calibration_fpr_1pct': calibrated_threshold}
cal_metrics = binary_metrics(y[cal], binary_scores['calibration'], calibrated_threshold)
assert cal_metrics['fpr'] <= .01
(REPORT / 'thresholds.json').write_text(json.dumps({
    'binary': thresholds, 'behavior': .5,
    'selection_population': 'calibration benigns only'}, indent=2))
display(pd.Series(cal_metrics))
""",
    )
    code(
        "6_thresholds",
        "Inspect the fixed validation baselines",
        """
validation_binary = binary_metrics(y[idx['validation']], binary_scores['validation'], .5)
validation_behavior, _ = multilabel_metrics(
    Y[idx['validation']], behavior_scores['validation'], labels)
display(pd.DataFrame([validation_binary]))
display(pd.DataFrame([validation_behavior]))
print('PASS: thresholds frozen using development data; no test optimization.')
""",
    )
    md(
        """## 7. Evaluate the held-out cohorts

Report detector discrimination, recall, false positives and confusion counts. Multi-label reports compare direct OvR with detector-gated variants at fixed thresholds. Show all-row closed-world scoring and a sensitivity view excluding unannotated malware. Neither view resolves missing individual tags. Earlier experiments already inspected these cohorts, so they are not a newly blind external evaluation."""
    )
    code(
        "7_evaluation",
        "Evaluate binary detection without inventing single-class AUC",
        """
binary_records = []
for split, scores in binary_scores.items():
    for configuration, threshold in thresholds.items():
        metrics = binary_metrics(y[idx[split]], scores, threshold)
        binary_records.append({'split': split, 'configuration': configuration, **metrics})
binary_table = pd.DataFrame(binary_records)
binary_table.to_csv(REPORT / 'binary_metrics.csv', index=False)
display(binary_table.query("split in ['test', 'challenge']"))
# Challenge-only has no benign rows: AUC, AP and FPR are undefined.
challenge_formats = []
ci = idx['challenge']
for file_type in sorted(meta.iloc[ci].file_type.unique()):
    take = meta.iloc[ci].file_type.eq(file_type).to_numpy()
    metrics = binary_metrics(y[ci][take], binary_scores['challenge'][take], .5)
    challenge_formats.append({'file_type': file_type, **metrics})
pd.DataFrame(challenge_formats).to_csv(REPORT / 'challenge_by_format.csv', index=False)
""",
    )
    code(
        "7_evaluation",
        "Evaluate native tags and detector gates",
        """
behavior_records, per_tag_records = [], []
for split, scores in behavior_scores.items():
    ii = idx[split]
    variants = {'direct_ovr': scores}
    for name, threshold in thresholds.items():
        gate = binary_scores[split] >= threshold
        variants['gated_' + name] = scores * gate[:, None]
    views = {'closed_world_all_rows': np.ones(len(ii), dtype=bool),
             'benign_and_annotated_malware': (y[ii] == 0) | annotated[ii]}
    for configuration, probabilities in variants.items():
        for view, mask in views.items():
            metrics, per_tag = multilabel_metrics(Y[ii][mask], probabilities[mask], labels)
            key = {'split': split, 'configuration': configuration, 'view': view}
            behavior_records.append({**key, **metrics})
            per_tag_records.append(per_tag.assign(**key))
behavior_table = pd.DataFrame(behavior_records)
behavior_table.to_csv(REPORT / 'behavior_metrics.csv', index=False)
pd.concat(per_tag_records).to_csv(REPORT / 'behavior_per_tag.csv', index=False)
display(behavior_table.query("split == 'test'"))
print('PASS: detector, direct OvR and gated metrics saved with annotation limits.')
""",
    )
    md(
        """## 8. Visualize, save and verify

ROC/precision-recall plots use the official Win64 test cohort. Per-tag plots include fitting and test support to expose rare-label instability. Saved row-aligned scores permit later paired comparisons without retaining models. The historical mixed-format score table is preserved separately and should not be directly ranked against this Win64-only experiment."""
    )
    code(
        "8_outputs",
        "Plot detector discrimination and validation history",
        """
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
history[['auc']].plot(ax=axes[0], legend=False, title='Binary validation AUC')
test_y, test_p = y[idx['test']], binary_scores['test']
fpr, tpr, _ = roc_curve(test_y, test_p)
precision, recall, _ = precision_recall_curve(test_y, test_p)
axes[1].plot(fpr, tpr)
axes[1].plot([0, 1], [0, 1], '--', color='grey')
axes[1].set(title='Win64 test ROC', xlabel='False positive rate', ylabel='Recall')
axes[2].plot(recall, precision)
axes[2].axhline(test_y.mean(), ls='--', color='grey')
axes[2].set(title='Win64 test precision–recall', xlabel='Recall', ylabel='Precision')
fig.tight_layout()
fig.savefig(FIGURES / 'binary_performance.png', dpi=150)
plt.show()
""",
    )
    code(
        "8_outputs",
        "Plot native tag support and performance",
        """
per_tag = pd.concat(per_tag_records)
view = per_tag.query("split == 'test' and configuration == 'direct_ovr' "
                     "and view == 'closed_world_all_rows'").set_index('tag')
view = view.join(support.set_index('tag')[['fit_positive']]).sort_values('fit_positive')
fig, axes = plt.subplots(1, 2, figsize=(13, 12), sharey=True)
view[['fit_positive', 'support']].plot.barh(ax=axes[0], logx=True)
view[['f1', 'average_precision']].plot.barh(ax=axes[1])
axes[0].set(title='Fitting/test positive support', xlabel='Records (log scale)')
axes[1].set(title='Direct native OvR, Win64 test', xlabel='Score', xlim=(0, 1))
fig.tight_layout()
fig.savefig(FIGURES / 'behavior_performance.png', dpi=150)
plt.show()
""",
    )
    code(
        "8_outputs",
        "Save row-aligned probabilities and exact configurations",
        """
saved = {'labels': np.asarray(labels)}
for split, scores in binary_scores.items():
    saved[split + '_rows'] = idx[split]
    saved[split + '_sha256'] = meta.iloc[idx[split]].sha256.to_numpy(dtype='U64')
    saved[split + '_binary_y'] = y[idx[split]]
    saved[split + '_behavior_y'] = Y[idx[split]]
    saved[split + '_binary_p'] = scores
    saved[split + '_behavior_p'] = behavior_scores[split]
np.savez_compressed(REPORT / 'heldout_scores.npz', **saved)
configuration = {'binary': binary_config, 'behavior': tag_config,
    'categorical_features': [2, 3, 4, 5, 6], 'native_behavior_labels': labels,
    'class_min': 10, 'normalization': None, 'feature_selection': None, 'PCA': None,
    'upstream': provenance, 'environment': environment}
(REPORT / 'configuration.json').write_text(json.dumps(configuration, indent=2))
""",
    )
    code(
        "8_outputs",
        "Verify protected files and seal the completed run",
        """
assert protected_hashes(P) == protected_before
summary = {'completed': True, 'features': 696, 'fitting_rows': len(idx['fit']),
    'source_records_sha256': source_sha, 'vector_audit': vector_audit,
    'binary_rounds': binary_rounds, 'binary_fit_seconds': binary_seconds,
    'behavior_heads': len(labels), 'serialized_models': 0,
    'protocol': 'chronological 40/6/6 training weeks; official test; separate challenge',
    'limits': ['local 696-feature Win64 adaptation; not the full upstream benchmark',
              'behavior absence is incompletely annotated; scores are weak-label agreement',
              'native tags differ from seven proposal roles; holdouts previously inspected',
              'challenge is all malware and includes non-Win64 formats'],
    'protected_file_hashes': protected_before}
summary['output_file_hashes'] = {
    str(f.relative_to(REPORT)): hashlib.sha256(f.read_bytes()).hexdigest()
    for f in sorted(REPORT.rglob('*')) if f.is_file() and f.name != 'run_summary.json'}
(REPORT / 'run_summary.json').write_text(json.dumps(summary, indent=2))
display(binary_table.query("split == 'test'"))
display(behavior_table.query("split == 'test' and configuration == 'direct_ovr'"))
print('PASS: results sealed; prior analysis and scores unchanged; no models serialized.')
""",
    )
    code(
        "8_outputs",
        "Interpret the saved baseline results",
        """
from IPython.display import Markdown, display
from multi_label_static_analysis.config import project_paths
from multi_label_static_analysis.repository_baseline import baseline_readout

# This readout depends only on saved reports; it never retrains or tunes.
result_directory = project_paths().project / 'results/repository_baseline'
display(Markdown(baseline_readout(result_directory)))
print('PASS: results interpreted with annotation and comparability limits.')
""",
    )
    # Named small cells work with the existing sequential notebook runner.
    for stage in dict.fromkeys(
        c.metadata.stage for c in cells if c.cell_type == "code"
    ):
        steps = [
            c for c in cells if c.cell_type == "code" and c.metadata.stage == stage
        ]
        for step, cell in enumerate(steps, 1):
            cell.metadata.update(
                step=step, step_count=len(steps), stage_end=step == len(steps)
            )
            assert len(cell.source.splitlines()) <= 35, cell.metadata.title
    nb.cells = cells
    nbf.validate(nb)
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destination = (
        args.output
        or project_paths().project / "notebooks/ember2024_repository_baseline.ipynb"
    )
    if destination.exists() and not args.force:
        parser.error("Notebook exists; use --force to replace it.")
    build(destination)
