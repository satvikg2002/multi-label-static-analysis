"""Contained plotting and statistics steps used by the readable notebooks.

These functions preserve the original analysis calculations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_label_quality(meta, fit, coverage, Y, mal_annotated, LABELS, savefig):
    """Keep all four panels in one function so notebook cell boundaries cannot close a half-built figure."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    meta.loc[fit].groupby(["file_type", "malicious"]).size().unstack(
        fill_value=0
    ).plot.bar(ax=axes[0, 0], rot=0, title="Fitting binary class balance")
    coverage[coverage.malicious.eq(1)].set_index("file_type").annotation_rate.plot.bar(
        ax=axes[0, 1], rot=0, ylim=(0, 1), title="Malware behavior annotation coverage"
    )
    co = Y[mal_annotated].astype(np.int64).T @ Y[mal_annotated].astype(np.int64)
    im = axes[1, 0].imshow(co, cmap="Blues")
    axes[1, 0].set_xticks(range(7), LABELS, rotation=65, ha="right")
    axes[1, 0].set_yticks(range(7), LABELS)
    axes[1, 0].set_title("Fitting role co-occurrence")
    fig.colorbar(im, ax=axes[1, 0])
    axes[1, 1].hist(Y[mal_annotated].sum(axis=1), bins=np.arange(9) - 0.5, rwidth=0.85)
    axes[1, 1].set(
        title="Mapped-role cardinality (annotated malware)",
        xlabel="Number of mapped roles",
        ylabel="Records",
    )
    savefig("02_labels_and_coverage.png")


def fitting_statistics(X, idx, D, names):
    """Calculate exact statistics in bounded column blocks; never load the whole matrix into RAM."""
    stat_records = []
    for start in range(0, D, 16):
        columns = np.arange(start, min(start + 16, D))
        a = np.asarray(X[np.ix_(idx["fit"], columns)], dtype=np.float64)
        mean = np.nanmean(a, axis=0)
        std = np.nanstd(a, axis=0)
        median = np.nanmedian(a, axis=0)
        p01, p25, p75, p99 = np.nanpercentile(a, [1, 25, 75, 99], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            skew = np.nanmean(((a - mean) / np.where(std > 0, std, 1)) ** 3, axis=0)
        for k, j in enumerate(columns):
            stat_records.append(
                {
                    "index": int(j),
                    "feature": names[j],
                    "group": names[j].split(".")[0],
                    "missing_rate": float(np.isnan(a[:, k]).mean()),
                    "zero_rate": float((a[:, k] == 0).mean()),
                    "mean": float(mean[k]),
                    "std": float(std[k]),
                    "median": float(median[k]),
                    "p01": float(p01[k]),
                    "p25": float(p25[k]),
                    "p75": float(p75[k]),
                    "p99": float(p99[k]),
                    "skewness": float(skew[k]),
                    "above_tukey_fence_rate": float(
                        (a[:, k] > p75[k] + 1.5 * (p75[k] - p25[k])).mean()
                    ),
                }
            )
        del a
    return stat_records


def plot_feature_distributions(meta, idx, X, groups, stats, savefig):
    """Build and save the entire distribution figure together."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fitmeta = meta.iloc[idx["fit"]]
    for ft in sorted(fitmeta.file_type.unique()):
        take = idx["fit"][fitmeta.file_type.eq(ft).to_numpy()]
        axes[0, 0].hist(
            np.log10(1 + X[take, 0]), bins=60, density=True, alpha=0.35, label=ft
        )
        axes[0, 1].hist(X[take, 1], bins=60, density=True, alpha=0.35, label=ft)
        axes[1, 0].hist(
            np.log10(1 + X[take, groups["strings"][0]]),
            bins=60,
            density=True,
            alpha=0.35,
            label=ft,
        )
    axes[0, 0].set(xlabel="log10(1 + bytes)", title="File size by fitting format")
    axes[0, 1].set(xlabel="Byte entropy", title="Entropy by fitting format")
    axes[1, 0].set(
        xlabel="log10(1 + string count)", title="String count by fitting format"
    )
    stats.groupby("group").zero_rate.mean().plot.bar(
        ax=axes[1, 1], rot=20, title="Mean zero prevalence by group", ylim=(0, 1)
    )
    for ax in axes.flat[:3]:
        ax.legend()
    savefig("04_feature_distributions.png")


def plot_pca_structure(explained, coordinates, meta, sample_idx, y, savefig):
    """Color the same fitted projection by format and label without fitting anything new."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(explained.component, explained.cumulative_variance)
    axes[0].axhline(0.95, color="gray", ls="--")
    axes[0].set(
        xlabel="Components",
        ylabel="Cumulative variance",
        title="Standardized fitting variance",
    )
    color_meta = meta.iloc[sample_idx[:6000]]
    for ft in sorted(color_meta.file_type.unique()):
        take = color_meta.file_type.eq(ft).to_numpy()
        axes[1].scatter(
            coordinates[take, 0],
            coordinates[take, 1],
            s=5,
            alpha=0.3,
            label=ft,
            rasterized=True,
        )
    axes[1].legend()
    axes[1].set(title="Fitting PCA by format", xlabel="PC1", ylabel="PC2")
    for value, label in [(0, "Benign"), (1, "Malicious")]:
        take = y[sample_idx[:6000]] == value
        axes[2].scatter(
            coordinates[take, 0],
            coordinates[take, 1],
            s=5,
            alpha=0.3,
            label=label,
            rasterized=True,
        )
    axes[2].legend()
    axes[2].set(title="Same projection, label overlay", xlabel="PC1", ylabel="PC2")
    savefig("08_pca_structure.png")
