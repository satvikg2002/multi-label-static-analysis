"""Reproducible local adaptations of EMBER2024's LightGBM baselines.

No estimator is serialized. The analysis notebook owns the verified input cache;
this module independently reconstructs its row order before using that cache.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)

from .data_analysis import vectorize_record

SPLITS = {"fit": 0, "validation": 1, "calibration": 2, "test": 3, "challenge": 4}
IGNORE_TAGS = {"", "win32", "win64", "elf", "linux", "pdf", "apk", "android"}


def canonical_metadata(rows):
    """Apply the primary analysis's SHA, chronology and exact-TLSH policy."""
    raw = pd.DataFrame(rows)
    raw["date"] = pd.to_datetime(raw.timestamp, unit="s", utc=True, errors="coerce")
    if raw.date.isna().any() or not raw.malicious.isin([0, 1]).all():
        raise ValueError("Invalid dates or binary labels")
    conflicts = raw.groupby("sha256").agg(
        binary=("malicious", "nunique"), behavior=("tags", "nunique")
    )
    if conflicts.gt(1).any().any():
        raise ValueError("Resolve conflicting duplicate annotations before fitting")
    train_end = raw.loc[
        raw.official.eq("train"), "date"
    ].max().normalize() + pd.Timedelta(days=1)
    raw["split"] = raw.official
    train = raw.official.eq("train")
    raw.loc[train & raw.date.ge(train_end - pd.Timedelta(weeks=12)), "split"] = (
        "validation"
    )
    raw.loc[train & raw.date.ge(train_end - pd.Timedelta(weeks=6)), "split"] = (
        "calibration"
    )
    raw.loc[raw.split.eq("train"), "split"] = "fit"
    meta = (
        raw.sort_values(["date", "sha256", "source", "line"])
        .drop_duplicates("sha256")
        .copy()
    )
    meta["priority"] = meta.split.map(SPLITS)
    valid = meta.tlsh.notna() & ~meta.tlsh.isin(["", "TNULL", "NULL", "-"])
    earliest = meta.loc[valid].groupby("tlsh").priority.min()
    later = valid & meta.priority.gt(meta.tlsh.map(earliest))
    meta = meta.loc[~later].reset_index(drop=True)
    meta["row_id"] = np.arange(len(meta))
    for a, b in zip(list(SPLITS)[:3], list(SPLITS)[1:4]):
        assert (
            meta.loc[meta.split.eq(a), "date"].max()
            < meta.loc[meta.split.eq(b), "date"].min()
        )
    return meta


def checked_vectors(paths, meta, source_sha):
    """Require the exact source/extractor/cache key; check raw rows independently."""
    from .thrember import features

    extractor_sha = hashlib.sha256(Path(features.__file__).read_bytes()).hexdigest()
    key = hashlib.sha256(
        (
            source_sha + str(meta[["source", "line"]].values.tolist()) + extractor_sha
        ).encode()
    ).hexdigest()
    cache, keyfile = (
        paths.artifacts / "static_features.npy",
        paths.artifacts / "feature_cache_key.txt",
    )
    if not cache.exists() or not keyfile.exists() or keyfile.read_text() != key:
        raise ValueError(
            "Run the primary-analysis notebook first: its vector cache is missing or stale"
        )
    X = np.load(cache, mmap_mode="r")
    assert X.shape == (len(meta), 696) and X.dtype == np.float32
    checks = meta.groupby("file_type", sort=True).head(3)
    for source, group in checks.groupby("source"):
        wanted = dict(zip(group.line, group.row_id))
        with (paths.data / source).open() as stream:
            for line_no, line in enumerate(stream):
                if line_no in wanted:
                    np.testing.assert_allclose(
                        X[wanted[line_no]],
                        vectorize_record(json.loads(line)),
                        equal_nan=True,
                    )
                if line_no >= max(wanted):
                    break
    keep = np.ones(len(meta), dtype=bool)
    seen = {}
    for i in np.argsort(meta.priority.to_numpy(), kind="stable"):
        signature = hashlib.blake2b(
            np.nan_to_num(X[i], nan=-1).tobytes(), digest_size=16
        ).digest()
        if seen.setdefault(signature, meta.split.iloc[i]) != meta.split.iloc[i]:
            keep[i] = False
    indices = {s: np.flatnonzero(meta.split.eq(s).to_numpy() & keep) for s in SPLITS}
    expected = (
        pd.read_csv(paths.reports / "eligible_partition_counts.csv")
        .groupby("split")
        .n.sum()
        .to_dict()
    )
    assert {s: len(ii) for s, ii in indices.items()} == expected
    return (
        X,
        indices,
        {
            "feature_cache_key": key,
            "raw_row_checks": len(checks),
            "vector_purges": int((~keep).sum()),
        },
    )


def behavior_targets(meta, fit_indices, class_min=10):
    """Freeze native behavior vocabulary on fitting rows, never held-out labels."""
    counts = Counter(
        t for tags in meta.iloc[fit_indices].tags for t in tags if t not in IGNORE_TAGS
    )
    labels = sorted(t for t, n in counts.items() if class_min <= n < len(fit_indices))
    if not labels:
        raise ValueError("No supported fitting-period behavior tags")
    lookup = {tag: j for j, tag in enumerate(labels)}
    Y = np.zeros((len(meta), len(labels)), dtype=np.int8)
    for i, tags in enumerate(meta.tags):
        for tag in tags:
            if tag in lookup:
                Y[i, lookup[tag]] = 1
    support = pd.DataFrame({"tag": labels, "fit_positive": [counts[t] for t in labels]})
    return labels, Y, support


def repository_configs(paths, threads=4):
    """Verify pinned upstream JSONs before applying runtime-only overrides."""
    directory = paths.project / "configs/ember2024"
    provenance = json.loads((directory / "provenance.json").read_text())
    configs = []
    for name in ["lgbm_config.json", "lgbm_config_tag.json"]:
        path = directory / name
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == provenance["files"][name]["sha256"]
        )
        config = json.loads(path.read_text())
        config.update(num_threads=threads, verbosity=-1)
        configs.append(config)
    return *configs, provenance


def fit_binary(X, y, indices, config):
    """Upstream 500-round detector; chronological validation is diagnostic only."""
    start = time.perf_counter()
    train = lgb.Dataset(
        X[indices["fit"]], label=y[indices["fit"]], categorical_feature=[2, 3, 4, 5, 6]
    )
    val = lgb.Dataset(
        X[indices["validation"]], label=y[indices["validation"]], reference=train
    )
    history = {}
    model = lgb.train(
        dict(config),
        train,
        valid_sets=[val],
        valid_names=["validation"],
        callbacks=[lgb.record_evaluation(history), lgb.log_evaluation(100)],
    )
    return model, history, time.perf_counter() - start


def predict_rows(model, X, rows, batch_size=8192):
    out = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        ii = rows[start : start + batch_size]
        out[start : start + len(ii)] = model.predict(X[ii], num_threads=4)
    assert np.isfinite(out).all() and ((0 <= out) & (out <= 1)).all()
    return out


def fit_behaviors(X, Y, labels, indices, config):
    """Sequential OvR heads, shared bins, upstream AUC early stopping where defined.

    A chronological validation window can lack a rare tag. Such heads use all
    100 upstream rounds rather than early stopping on an undefined AUC.
    Only predictions and training diagnostics escape this function.
    """
    fit, val = indices["fit"], indices["validation"]
    rounds = int(config["num_iterations"])
    dataset_params = {
        k: v
        for k, v in config.items()
        if k not in {"num_iterations", "early_stopping_rounds"}
    }
    train = lgb.Dataset(
        X[fit],
        label=Y[fit, 0],
        categorical_feature=[2, 3, 4, 5, 6],
        params=dataset_params,
    )
    valid = lgb.Dataset(
        X[val],
        label=Y[val, 0],
        reference=train,
        categorical_feature=[2, 3, 4, 5, 6],
        params=dataset_params,
    )
    train.construct()
    valid.construct()
    scores = {
        s: np.empty((len(ii), len(labels)), dtype=np.float32)
        for s, ii in indices.items()
        if s != "fit"
    }
    diagnostics = []
    for j, tag in enumerate(labels):
        started = time.perf_counter()
        train.set_label(Y[fit, j])
        valid.set_label(Y[val, j])
        params = dict(config)
        auc_defined = len(np.unique(Y[val, j])) == 2
        if not auc_defined:
            params.pop("early_stopping_rounds", None)
            params["metric"] = ["binary_logloss", "binary_error"]
        model = lgb.train(
            params, train, valid_sets=[valid], callbacks=[lgb.log_evaluation(0)]
        )
        for s in scores:
            scores[s][:, j] = predict_rows(model, X, indices[s])
        record = {
            "tag": tag,
            "fit_positive": int(Y[fit, j].sum()),
            "validation_positive": int(Y[val, j].sum()),
            "best_iteration": model.best_iteration or model.current_iteration(),
            "configured_rounds": rounds,
            "auc_early_stopping": auc_defined,
            "seconds": time.perf_counter() - started,
        }
        diagnostics.append(record)
        print(
            f"[{j + 1}/{len(labels)}] {tag}: {record['best_iteration']} rounds, {record['seconds']:.1f}s",
            flush=True,
        )
        del model
    return scores, pd.DataFrame(diagnostics)


def low_fpr_threshold(y, p, target=0.01):
    """Select on calibration benigns only; strict nextafter handles score ties."""
    benign = np.asarray(p, dtype=np.float32)[np.asarray(y) == 0]
    if not len(benign):
        raise ValueError("Calibration requires benign examples")
    ordered = np.sort(benign)[::-1]
    allowed = int(np.floor(target * len(ordered)))
    if allowed >= len(ordered):
        return 0.0
    return float(np.nextafter(ordered[allowed], np.float32(np.inf)))


def binary_metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = bool(tp + fn) and bool(tn + fp)
    precision, recall, _ = precision_recall_curve(y, p) if both else (None, None, None)
    return {
        "n": len(y),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": float(fp / (tn + fp)) if tn + fp else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "pr_auc_trapezoidal": float(auc(recall, precision)) if both else None,
    }


def multilabel_metrics(Y, p, labels, threshold=0.5):
    pred = p >= threshold
    rows = []
    for j, tag in enumerate(labels):
        yy, pp = Y[:, j], p[:, j]
        both = len(np.unique(yy)) == 2
        rows.append(
            {
                "tag": tag,
                "n": len(yy),
                "support": int(yy.sum()),
                "precision": precision_score(yy, pred[:, j], zero_division=0),
                "recall": recall_score(yy, pred[:, j], zero_division=0),
                "f1": f1_score(yy, pred[:, j], zero_division=0),
                "average_precision": (
                    float(average_precision_score(yy, pp)) if yy.any() else None
                ),
                "roc_auc": float(roc_auc_score(yy, pp)) if both else None,
            }
        )
    per_tag = pd.DataFrame(rows)
    summary = {
        "n": len(Y),
        "labels": len(labels),
        "threshold": threshold,
        "micro_precision": precision_score(Y, pred, average="micro", zero_division=0),
        "micro_recall": recall_score(Y, pred, average="micro", zero_division=0),
        "micro_f1": f1_score(Y, pred, average="micro", zero_division=0),
        "macro_f1": f1_score(Y, pred, average="macro", zero_division=0),
        "macro_average_precision": per_tag.average_precision.mean(),
        "macro_roc_auc": per_tag.roc_auc.mean(),
        "ap_supported_labels": int(per_tag.average_precision.notna().sum()),
        "auc_supported_labels": int(per_tag.roc_auc.notna().sum()),
        "subset_accuracy": float(np.all(Y == pred, axis=1).mean()),
        "hamming_loss": float(hamming_loss(Y, pred)),
    }
    return summary, per_tag


def protected_hashes(paths):
    """Guard existing analysis results, notebooks and historical references."""
    files = [paths.notebook, *paths.reports.rglob("*"), *paths.historical.rglob("*")]
    return {
        str(p.relative_to(paths.root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
        if p.is_file()
    }


def baseline_readout(report):
    """Interpret saved scores without fitting or selecting any new model."""
    binary = pd.read_csv(report / "binary_metrics.csv")
    behavior = pd.read_csv(report / "behavior_metrics.csv")
    training = pd.read_csv(report / "behavior_training.csv")
    b = binary.query("split == 'test' and configuration == 'repository_0.5'").iloc[0]
    c = binary.query(
        "split == 'test' and configuration == 'calibration_fpr_1pct'"
    ).iloc[0]
    direct = behavior.query("split == 'test' and configuration == 'direct_ovr'")
    all_rows = direct.query("view == 'closed_world_all_rows'").iloc[0]
    observed = direct.query("view == 'benign_and_annotated_malware'").iloc[0]
    gated = behavior.query(
        "split == 'test' and configuration == 'gated_repository_0.5' "
        "and view == 'closed_world_all_rows'"
    ).iloc[0]
    challenge = binary.query(
        "split == 'challenge' and configuration == 'repository_0.5'"
    ).iloc[0]
    return (
        f"**Binary test.** At 0.5, F1 is {b.f1:.4f}, recall {b.recall:.2%} and FPR {b.fpr:.2%}. "
        f"The calibration-selected threshold gives recall {c.recall:.2%} and FPR {c.fpr:.2%}. "
        "Reducing false alarms costs detections; no test threshold was optimized.\n\n"
        f"**Native tags.** Direct OvR micro-F1 is {all_rows.micro_f1:.4f} "
        f"(precision {all_rows.micro_precision:.2%}, recall {all_rows.micro_recall:.2%}). "
        f"Detector gating changes micro-F1 to {gated.micro_f1:.4f} and recall to {gated.micro_recall:.2%}. "
        "The gate removes tag predictions, including some correct ones.\n\n"
        f"**Annotation sensitivity.** Excluding unannotated malware changes direct micro-F1 to "
        f"{observed.micro_f1:.4f}; this is a different evaluation population. "
        "Missing tags and incorrect predictions both affect observed precision; these data cannot distinguish all cases.\n\n"
        f"**Instability and challenge.** {int(training.best_iteration.eq(1).sum())}/{len(training)} heads retained "
        f"one round. Challenge recall is {challenge.recall:.2%} at 0.5 across all formats. "
        "Its lack of benign examples prevents estimating challenge-only FPR or ROC AUC.\n\n"
        "**Next experiments, not selected improvements.** Test per-tag calibration thresholds, "
        "compare class weighting and stopping metrics, then evaluate full PE features on the Win64 cohorts. "
        "[LightGBM documents](https://lightgbm.readthedocs.io/en/stable/Parameters.html#is_unbalance) "
        "that imbalance weighting can harm probability estimates; 0.5 need not be a useful operating point."
    )


def verify_baseline(paths):
    import nbformat
    from .data_analysis import assert_no_models, verify_score_reference

    report = paths.project / "results/repository_baseline"
    notebook = nbformat.read(
        paths.project / "notebooks/ember2024_repository_baseline.ipynb", 4
    )
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert len({c.metadata.stage for c in cells}) == 8
    for cell in cells:
        assert cell.execution_count is not None
        assert not any(o.output_type == "error" for o in cell.outputs)
        if cell.metadata.stage_end:
            assert any(
                o.output_type == "stream" and "PASS:" in o.text for o in cell.outputs
            )
    summary = json.loads((report / "run_summary.json").read_text())
    assert summary["completed"] and summary["features"] == 696
    assert summary["binary_rounds"] == 500 and summary["serialized_models"] == 0
    assert summary["behavior_heads"] == len(
        pd.read_csv(report / "behavior_training.csv")
    )
    assert protected_hashes(paths) == summary["protected_file_hashes"]
    for name, sha in summary["output_file_hashes"].items():
        assert hashlib.sha256((report / name).read_bytes()).hexdigest() == sha, name
    assert_no_models(paths.artifacts)
    assert_no_models(report)
    verify_score_reference(paths.historical)
    print(
        "PASS: all eight baseline sections executed; outputs and protected analysis verified."
    )
    print(
        "PASS: binary and native-tag OvR results saved; no classifier weights serialized."
    )
