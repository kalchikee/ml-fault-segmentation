"""
clustering.py
-------------
HDBSCAN-based 3D clustering of earthquake hypocenters for fault segmentation.

Implements a systematic parameter sweep over min_cluster_size and min_samples
to identify optimal clustering configuration. Cluster quality evaluated by
silhouette coefficient and Davies-Bouldin index.

Reference: McInnes et al. (2017), Campello et al. (2013)
"""

import logging
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

log = logging.getLogger(__name__)


def run_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10,
    cluster_selection_epsilon: float = 2000.0,
    cluster_selection_method: str = "eom",
) -> tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
    """
    Run HDBSCAN clustering on 3D hypocenter feature matrix.

    Parameters
    ----------
    X : np.ndarray, shape (N, 3)
        Feature matrix [x_utm, y_utm, depth_scaled] in meters.
    min_cluster_size : int
        Minimum number of events to form a cluster (fault segment).
    min_samples : int
        Controls noise robustness; higher = more conservative.
    cluster_selection_epsilon : float
        Merge clusters within this distance (meters). Default 2 km.
    cluster_selection_method : str
        "eom" (excess of mass) or "leaf".

    Returns
    -------
    labels : np.ndarray, shape (N,)  Cluster labels (-1 = noise)
    probabilities : np.ndarray, shape (N,)  Soft cluster membership
    clusterer : hdbscan.HDBSCAN  Fitted clusterer object
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1,
    )
    clusterer.fit(X)
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    noise_pct = 100.0 * n_noise / len(labels)

    log.info(
        f"HDBSCAN(min_cluster_size={min_cluster_size}, min_samples={min_samples}): "
        f"{n_clusters} clusters, {n_noise:,} noise ({noise_pct:.1f}%)"
    )

    return labels, probabilities, clusterer


def parameter_sweep(
    X: np.ndarray,
    min_cluster_sizes: list[int] | None = None,
    min_samples_list: list[int] | None = None,
    sample_size: int = 50000,
) -> pd.DataFrame:
    """
    Systematic parameter sweep to optimize HDBSCAN hyperparameters.

    Evaluates each parameter combination using:
    - Silhouette score (higher is better, range -1 to 1)
    - Davies-Bouldin index (lower is better)
    - Number of clusters
    - Noise fraction

    Parameters
    ----------
    X : np.ndarray  Feature matrix
    min_cluster_sizes : list[int]  Values to sweep (default: [50, 100, 200, 500])
    min_samples_list : list[int]  Values to sweep (default: [5, 10, 20])
    sample_size : int  Subsample for silhouette computation (expensive)

    Returns
    -------
    DataFrame with one row per parameter combination, sorted by silhouette score
    """
    if min_cluster_sizes is None:
        min_cluster_sizes = [50, 100, 200, 500]
    if min_samples_list is None:
        min_samples_list = [5, 10, 20]

    combinations = [
        (mcs, ms) for mcs in min_cluster_sizes for ms in min_samples_list
    ]
    log.info(f"Parameter sweep: {len(combinations)} combinations")

    results = []
    for min_cluster_size, min_samples in tqdm(combinations, desc="Parameter sweep"):
        labels, probs, _ = run_hdbscan(X, min_cluster_size, min_samples)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        # Silhouette score requires at least 2 clusters and non-noise points
        non_noise = labels != -1
        silhouette = np.nan
        db_score = np.nan

        if n_clusters >= 2 and non_noise.sum() > sample_size:
            # Subsample for speed
            idx = np.random.choice(
                np.where(non_noise)[0], size=sample_size, replace=False
            )
            try:
                silhouette = silhouette_score(X[idx], labels[idx], metric="euclidean")
                db_score = davies_bouldin_score(X[idx], labels[idx])
            except Exception as e:
                log.warning(f"  Metrics failed: {e}")

        results.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_fraction": n_noise / len(labels),
            "silhouette_score": silhouette,
            "davies_bouldin_score": db_score,
        })

    df = pd.DataFrame(results)
    df = df.sort_values("silhouette_score", ascending=False)
    log.info("Top 5 parameter combinations by silhouette score:")
    log.info(df.head(5).to_string())
    return df


def get_optimal_params(sweep_df: pd.DataFrame) -> dict[str, Any]:
    """Return parameter dict for the highest-silhouette combination."""
    best = sweep_df.iloc[0]
    return {
        "min_cluster_size": int(best["min_cluster_size"]),
        "min_samples": int(best["min_samples"]),
    }


def assign_clusters(
    gdf,
    X: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10,
) -> tuple:
    """
    Run HDBSCAN with optimal parameters and add cluster labels to GeoDataFrame.

    Returns
    -------
    gdf_labeled : GeoDataFrame with 'cluster_id' and 'cluster_prob' columns
    labels : np.ndarray
    """
    labels, probs, clusterer = run_hdbscan(X, min_cluster_size, min_samples)

    gdf = gdf.copy()
    gdf["cluster_id"] = labels
    gdf["cluster_prob"] = probs

    # Cluster size statistics
    cluster_ids = [l for l in set(labels) if l >= 0]
    sizes = [int((labels == c).sum()) for c in cluster_ids]
    if sizes:
        log.info(
            f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
            f"median={int(np.median(sizes))}"
        )

    return gdf, labels, clusterer
