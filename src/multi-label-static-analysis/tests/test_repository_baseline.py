"""Meaningful protocol, metric and notebook checks for the new baseline."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from multi_label_static_analysis.config import project_paths
from multi_label_static_analysis.repository_baseline import (
    behavior_targets,
    binary_metrics,
    low_fpr_threshold,
    multilabel_metrics,
    repository_configs,
    fit_behaviors,
)


class RepositoryBaselineTests(unittest.TestCase):
    def test_vocabulary_never_learns_heldout_tags(self):
        meta = pd.DataFrame({"tags": [("a", "win64"), (), ("test_only",)]})
        labels, Y, support = behavior_targets(meta, [0, 1], class_min=1)
        self.assertEqual(labels, ["a"])
        np.testing.assert_array_equal(Y[:, 0], [1, 0, 0])

    def test_calibration_ties_and_single_class_metrics(self):
        p = np.array([0.8, 0.8, 0.8, 0.1], dtype=np.float32)
        threshold = low_fpr_threshold(np.zeros(4), p, 0.25)
        self.assertEqual(int((p >= threshold).sum()), 0)
        y = np.array([0, 0, 1, 1])
        metrics = binary_metrics(y, np.array([0.1, 0.8, 0.9, 0.2]), 0.5)
        self.assertEqual([metrics[k] for k in ["tn", "fp", "fn", "tp"]], [1, 1, 1, 1])
        single = binary_metrics(np.ones(3), np.array([0.1, 0.8, 0.9]), 0.5)
        self.assertIsNone(single["roc_auc"])
        self.assertIsNone(single["fpr"])
        self.assertIsNone(single["average_precision"])

    def test_multilabel_reports_unsupported_auc_explicitly(self):
        y = np.array([[1, 0], [0, 0], [1, 0]])
        p = np.array([[0.9, 0.1], [0.1, 0.1], [0.8, 0.2]])
        metrics, per_tag = multilabel_metrics(y, p, ["present", "absent"])
        self.assertEqual(metrics["micro_f1"], 1)
        self.assertEqual(metrics["auc_supported_labels"], 1)
        self.assertTrue(pd.isna(per_tag.loc[1, "roc_auc"]))
        self.assertEqual(metrics["macro_f1"], 0.5)

    def test_ovr_scores_and_rare_validation_fallback(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(160, 8)).astype(np.float32)
        X[:, 2:7] = rng.integers(0, 3, size=(160, 5))
        Y = np.column_stack([X[:, 0] > 0, np.arange(160) % 7 == 0]).astype("int8")
        indices = {
            "fit": np.arange(100),
            "validation": np.arange(100, 130),
            "test": np.arange(130, 160),
        }
        Y[indices["validation"], 1] = 0
        config = {
            "objective": "binary",
            "num_iterations": 8,
            "early_stopping_rounds": 2,
            "metric": ["auc", "binary_logloss"],
            "first_metric_only": True,
            "num_threads": 1,
            "min_data_in_leaf": 2,
            "verbosity": -1,
            "seed": 0,
        }
        scores, training = fit_behaviors(X, Y, ["signal", "rare"], indices, config)
        self.assertEqual(scores["test"].shape, (30, 2))
        self.assertTrue(np.isfinite(scores["test"]).all())
        self.assertFalse(bool(training.loc[1, "auc_early_stopping"]))
        self.assertEqual(training.loc[1, "best_iteration"], 8)

    def test_builder_short_cells_and_pinned_configurations(self):
        paths = project_paths(Path(__file__))
        binary, tags, _ = repository_configs(paths)
        self.assertEqual(binary["num_iterations"], 500)
        self.assertEqual(tags["num_iterations"], 100)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "baseline.ipynb"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "multi_label_static_analysis",
                    "build-baseline",
                    "--output",
                    str(out),
                ],
                cwd=paths.root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            notebook = nbformat.read(out, 4)
            nbformat.validate(notebook)
            cells = [c for c in notebook.cells if c.cell_type == "code"]
            self.assertEqual(len({c.metadata.stage for c in cells}), 8)
            self.assertEqual(sum(c.metadata.stage_end for c in cells), 8)
            for cell in cells:
                compile(cell.source, cell.metadata.title, "exec")
                self.assertLessEqual(len(cell.source.splitlines()), 35)
                self.assertTrue(cell.source.startswith("# "))
            for cell in notebook.cells:
                if cell.cell_type == "markdown":
                    self.assertLessEqual(len(cell.source.split()), 85)


if __name__ == "__main__":
    unittest.main()
