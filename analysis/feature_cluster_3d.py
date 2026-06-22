#!/usr/bin/env python3
"""Analyze GG identity features in 3D feature space.

This script loads a trained Gaussian Grouping checkpoint, extracts the
N x 16 per-Gaussian identity features from `pc.get_objects`, reduces them to
3D with PCA, clusters the 3D points with KMeans and optional HDBSCAN, and
reports silhouette scores plus 3D scatter plots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    import hdbscan as external_hdbscan  # type: ignore

    HAS_HDBSCAN = True
    HDBSCAN_SOURCE = "hdbscan"
except Exception:
    external_hdbscan = None
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

        HAS_HDBSCAN = True
        HDBSCAN_SOURCE = "sklearn"
    except Exception:
        SklearnHDBSCAN = None
        HAS_HDBSCAN = False
        HDBSCAN_SOURCE = "unavailable"


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration


def load_cfg_args(model_path: Path) -> Namespace:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfg_args: {cfg_path}")
    cfg = eval(cfg_path.read_text(), {"Namespace": Namespace})
    if not isinstance(cfg, Namespace):
        raise ValueError(f"cfg_args did not evaluate to argparse.Namespace: {cfg_path}")
    return cfg


def resolve_checkpoint_path(model_path: Path, iteration: int) -> Tuple[Path, int]:
    if iteration == -1:
        point_cloud_root = model_path / "point_cloud"
        if not point_cloud_root.exists():
            raise FileNotFoundError(f"Missing point_cloud directory: {point_cloud_root}")
        iteration = searchForMaxIteration(str(point_cloud_root))
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing checkpoint ply: {ply_path}")
    return ply_path, iteration


def load_identity_features(model_path: Path, iteration: int) -> Tuple[np.ndarray, Dict]:
    cfg = load_cfg_args(model_path)
    ply_path, loaded_iter = resolve_checkpoint_path(model_path, iteration)

    gaussians = GaussianModel(getattr(cfg, "sh_degree", 3))
    gaussians.load_ply(str(ply_path))

    with torch.no_grad():
        features = gaussians.get_objects.view(gaussians.get_xyz.shape[0], -1)
        features = features.detach().float().cpu().numpy().astype(np.float32, copy=False)

    metadata = {
        "model_path": str(model_path),
        "loaded_iteration": loaded_iter,
        "checkpoint_path": str(ply_path),
        "num_points": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "sh_degree": int(getattr(cfg, "sh_degree", 3)),
    }
    if features.shape[1] != 16:
        raise ValueError(
            f"Expected N x 16 identity features from pc.get_objects, got {features.shape}"
        )
    return features, metadata


def cluster_kmeans(coords_3d: np.ndarray, n_clusters: int, random_state: int) -> Tuple[np.ndarray, Dict]:
    n_clusters = max(2, min(int(n_clusters), len(coords_3d)))
    clusterer = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = clusterer.fit_predict(coords_3d)
    return labels, {
        "algorithm": "kmeans",
        "n_clusters": n_clusters,
        "inertia": float(clusterer.inertia_),
    }


def cluster_hdbscan(
    coords_3d: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> Tuple[Optional[np.ndarray], Dict]:
    if not HAS_HDBSCAN:
        return None, {
            "algorithm": "hdbscan",
            "status": "unavailable",
            "reason": "hdbscan is not installed in this environment",
        }

    if HDBSCAN_SOURCE == "hdbscan":
        clusterer = external_hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            core_dist_n_jobs=1,
        )
    else:
        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            n_jobs=1,
        )
    labels = clusterer.fit_predict(coords_3d)
    n_clusters = len(set(labels.tolist()) - {-1})
    return labels, {
        "algorithm": "hdbscan",
        "backend": HDBSCAN_SOURCE,
        "status": "ok",
        "n_clusters": int(n_clusters),
        "noise_points": int(np.sum(labels == -1)),
    }


def stratified_sample_indices(
    labels: np.ndarray,
    max_samples: int,
    random_state: int,
    ignore_label: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)

    if ignore_label is not None:
        keep_mask = labels != ignore_label
        labels = labels[keep_mask]
        base_indices = np.flatnonzero(keep_mask)
    else:
        base_indices = np.arange(labels.shape[0])

    unique_labels = np.unique(labels)
    if unique_labels.size == 0:
        return np.array([], dtype=np.int64)

    per_cluster = max(1, max_samples // unique_labels.size)
    sampled: List[np.ndarray] = []
    for label in unique_labels:
        cluster_indices = base_indices[labels == label]
        if cluster_indices.size <= per_cluster:
            sampled.append(cluster_indices)
        else:
            sampled.append(rng.choice(cluster_indices, size=per_cluster, replace=False))

    sample = np.concatenate(sampled)
    if sample.size > max_samples:
        sample = rng.choice(sample, size=max_samples, replace=False)
    return np.sort(sample)


def compute_silhouette(
    coords_3d: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    random_state: int,
    ignore_label: Optional[int] = None,
) -> Tuple[float, Dict]:
    work_coords = coords_3d
    work_labels = labels
    if ignore_label is not None:
        keep_mask = labels != ignore_label
        work_coords = coords_3d[keep_mask]
        work_labels = labels[keep_mask]

    unique_labels = np.unique(work_labels)
    if unique_labels.size < 2:
        return float("nan"), {
            "sample_size": 0,
            "status": "insufficient_clusters",
        }

    sample_idx = stratified_sample_indices(
        work_labels, min(sample_size, len(work_labels)), random_state
    )
    if sample_idx.size < 2:
        return float("nan"), {
            "sample_size": int(sample_idx.size),
            "status": "insufficient_sample",
        }

    try:
        score = silhouette_score(
            work_coords[sample_idx],
            work_labels[sample_idx],
        )
        return float(score), {
            "sample_size": int(sample_idx.size),
            "status": "ok",
            "num_labels": int(unique_labels.size),
        }
    except Exception as exc:
        return float("nan"), {
            "sample_size": int(sample_idx.size),
            "status": "error",
            "error": str(exc),
        }


def plot_clusters_3d(
    coords_3d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    sample_size: int,
    random_state: int,
    ignore_label: Optional[int] = None,
) -> Dict:
    sample_idx = stratified_sample_indices(
        labels, min(sample_size, len(labels)), random_state, ignore_label=ignore_label
    )
    sample_coords = coords_3d[sample_idx]
    sample_labels = labels[sample_idx]

    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")

    if ignore_label is not None:
        unique_labels = [lab for lab in np.unique(sample_labels) if lab != ignore_label]
        color_lookup = {lab: cmap(i % 20) for i, lab in enumerate(sorted(unique_labels))}
        colors = [color_lookup.get(int(lab), (0.6, 0.6, 0.6, 0.25)) for lab in sample_labels]
        noise_mask = sample_labels == ignore_label
        if np.any(noise_mask):
            for idx in np.flatnonzero(noise_mask):
                colors[idx] = (0.55, 0.55, 0.55, 0.22)
    else:
        unique_labels = np.unique(sample_labels)
        color_lookup = {lab: cmap(i % 20) for i, lab in enumerate(sorted(unique_labels))}
        colors = [color_lookup[int(lab)] for lab in sample_labels]

    ax.scatter(
        sample_coords[:, 0],
        sample_coords[:, 1],
        sample_coords[:, 2],
        c=colors,
        s=2,
        alpha=0.32,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_zlabel("PCA-3")
    ax.view_init(elev=22, azim=35)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {"plot_sample_size": int(sample_idx.size)}


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def analyze_scene(
    model_path: Path,
    iteration: int,
    out_root: Path,
    kmeans_clusters: int,
    plot_sample_size: int,
    silhouette_sample_size: int,
    random_state: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    hdbscan_sample_size: int,
) -> List[Dict]:
    features, metadata = load_identity_features(model_path, iteration)
    scene_name = model_path.name

    pca = PCA(n_components=3, random_state=random_state)
    coords_3d = pca.fit_transform(features)
    pca_info = {
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
    }

    results: List[Dict] = []

    algorithms = ["kmeans", "hdbscan"]
    for algorithm in algorithms:
        if algorithm == "kmeans":
            labels, cluster_info = cluster_kmeans(coords_3d, kmeans_clusters, random_state)
            cluster_coords = coords_3d
            ignore_label = None
        else:
            hdbscan_indices = None
            hdbscan_coords = coords_3d
            if hdbscan_sample_size > 0 and coords_3d.shape[0] > hdbscan_sample_size:
                rng = np.random.default_rng(random_state)
                hdbscan_indices = np.sort(
                    rng.choice(coords_3d.shape[0], size=hdbscan_sample_size, replace=False)
                )
                hdbscan_coords = coords_3d[hdbscan_indices]
            labels, cluster_info = cluster_hdbscan(
                hdbscan_coords, hdbscan_min_cluster_size, hdbscan_min_samples
            )
            cluster_coords = hdbscan_coords
            ignore_label = -1
            if labels is None:
                cluster_info["silhouette"] = None
                cluster_info["scene"] = scene_name
                cluster_info["silhouette_score"] = float("nan")
                cluster_info["silhouette_status"] = "skipped"
                cluster_info["silhouette_sample_size"] = 0
                cluster_info["cluster_count"] = 0
                cluster_info["noise_points"] = 0
                cluster_info["cluster_sizes"] = {}
                cluster_info["plot_path"] = ""
                cluster_info["plot_sample_size"] = 0
                cluster_info.update(metadata)
                cluster_info.update(pca_info)
                cluster_info["algorithm"] = algorithm
                cluster_info["status"] = "skipped"
                results.append(cluster_info)
                continue
            cluster_info["sampled_points"] = int(cluster_coords.shape[0])
            cluster_info["sample_source_points"] = int(coords_3d.shape[0])

        silhouette, sil_info = compute_silhouette(
            cluster_coords,
            labels,
            silhouette_sample_size,
            random_state,
            ignore_label=ignore_label,
        )
        plot_dir = out_root / scene_name / f"iteration_{metadata['loaded_iteration']}" / algorithm
        plot_path = plot_dir / "scatter_3d.png"
        plot_info = plot_clusters_3d(
            cluster_coords,
            labels,
            plot_path,
            f"{scene_name} / {algorithm} / iteration {metadata['loaded_iteration']}",
            plot_sample_size,
            random_state,
            ignore_label=ignore_label,
        )

        unique_labels = np.unique(labels)
        if ignore_label is not None:
            unique_labels = unique_labels[unique_labels != ignore_label]
            noise_points = int(np.sum(labels == ignore_label))
        else:
            noise_points = 0

        cluster_sizes = {
            int(label): int(np.sum(labels == label))
            for label in np.unique(labels)
            if ignore_label is None or label != ignore_label
        }

        summary = {
            "scene": scene_name,
            "algorithm": algorithm,
            "silhouette_score": silhouette,
            "silhouette_status": sil_info.get("status"),
            "silhouette_sample_size": sil_info.get("sample_size"),
            "cluster_count": int(unique_labels.size),
            "noise_points": noise_points,
            "cluster_sizes": cluster_sizes,
            "plot_path": str(plot_path),
            "plot_sample_size": plot_info["plot_sample_size"],
        }
        summary.update(metadata)
        summary.update(pca_info)
        summary.update(cluster_info)

        metrics_path = plot_dir / "metrics.json"
        report_path = plot_dir / "report.md"
        save_text(metrics_path, json.dumps(summary, indent=2, sort_keys=True))

        report = [
            f"# {scene_name} / {algorithm}",
            "",
            f"- checkpoint: `{metadata['checkpoint_path']}`",
            f"- loaded iteration: `{metadata['loaded_iteration']}`",
            f"- points: `{metadata['num_points']}`",
            f"- feature dim: `{metadata['feature_dim']}`",
            f"- pca variance (3D): `{summary['pca_explained_variance_sum']:.4f}`",
            f"- silhouette score: `{silhouette:.4f}`" if not math.isnan(silhouette) else "- silhouette score: `nan`",
            f"- silhouette sample size: `{sil_info.get('sample_size')}`",
            f"- clusters: `{summary['cluster_count']}`",
            f"- noise points: `{noise_points}`",
            f"- plot: [`scatter_3d.png`](scatter_3d.png)",
        ]
        if algorithm == "hdbscan" and not HAS_HDBSCAN:
            report.append("- hdbscan: unavailable in this environment, skipped")
        report.extend(["", "## Cluster sizes", ""])
        for label, count in sorted(cluster_sizes.items(), key=lambda x: x[0]):
            report.append(f"- {label}: {count}")
        save_text(report_path, "\n".join(report))

        print(
            f"[{scene_name}][{algorithm}] silhouette="
            f"{silhouette:.4f}" if not math.isnan(silhouette) else f"[{scene_name}][{algorithm}] silhouette=nan",
        )
        results.append(summary)

    return results


def write_summary(out_root: Path, results: List[Dict]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in results:
        if row.get("algorithm") == "hdbscan" and row.get("status") == "skipped":
            score = "nan"
        else:
            score = f"{row['silhouette_score']:.4f}" if not math.isnan(row["silhouette_score"]) else "nan"
        rows.append(
            [
                row["scene"],
                row["algorithm"],
                str(row["loaded_iteration"]),
                str(row["num_points"]),
                str(row["feature_dim"]),
                str(row["cluster_count"]),
                score,
                str(row["silhouette_sample_size"]),
                str(row["pca_explained_variance_sum"])[:8],
                str(row["plot_path"]),
                row.get("status", "ok"),
            ]
        )

    md_lines = [
        "# Feature Cluster Summary",
        "",
        "| Scene | Algorithm | Iteration | Points | Dim | Clusters | Silhouette | Silhouette samples | PCA var (3D) | Plot | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.extend(
        [
            "",
            "Interpretation: KMeans is the all-point, fixed-16-cluster diagnostic and is the primary score for 3D separability. HDBSCAN is fit on a sampled subset when requested and its silhouette is computed only on non-noise dense clusters, so those scores can be higher even when the full feature cloud visibly overlaps.",
        ]
    )
    save_text(out_root / "summary.md", "\n".join(md_lines))
    save_text(out_root / "summary.json", json.dumps(results, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster GG identity features in 3D PCA space.")
    parser.add_argument(
        "--model_paths",
        nargs="+",
        default=["output/lerf/ramen", "output/lerf/figurines", "output/lerf/teatime"],
        help="Checkpoint model paths to analyze.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Checkpoint iteration to load, or -1 for the latest available point cloud.",
    )
    parser.add_argument(
        "--out_dir",
        default="analysis/feature_cluster_results",
        help="Output directory for plots and reports.",
    )
    parser.add_argument("--kmeans_clusters", type=int, default=16)
    parser.add_argument("--plot_sample_size", type=int, default=20000)
    parser.add_argument("--silhouette_sample_size", type=int, default=5000)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=500)
    parser.add_argument("--hdbscan_min_samples", type=int, default=50)
    parser.add_argument(
        "--hdbscan_sample_size",
        type=int,
        default=50000,
        help="Fit HDBSCAN on this many sampled PCA points; use 0 for all points.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    for model_path_str in args.model_paths:
        model_path = Path(model_path_str)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        scene_results = analyze_scene(
            model_path=model_path,
            iteration=args.iteration,
            out_root=out_root,
            kmeans_clusters=args.kmeans_clusters,
            plot_sample_size=args.plot_sample_size,
            silhouette_sample_size=args.silhouette_sample_size,
            random_state=args.random_state,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_min_samples=args.hdbscan_min_samples,
            hdbscan_sample_size=args.hdbscan_sample_size,
        )
        all_results.extend(scene_results)

    write_summary(out_root, all_results)
    print(f"Wrote summary to {out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
