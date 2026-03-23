"""
fault_comparison.py
-------------------
Compare algorithmically detected fault structures against the USGS
Quaternary Fault and Fold Database to assess detection performance
and identify candidate unmapped active faults.

Classification scheme:
  MATCHED  : nearest mapped fault < 5 km, strike discordance < 20°
  PARTIAL  : 5–15 km or 20–40° strike discordance
  UNMATCHED: no mapped fault within 15 km (candidate unmapped structure)
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

QFFDB_URL = "https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS_2021_new.zip"
UTM_CRS = "EPSG:32611"
GEO_CRS = "EPSG:4326"

MATCH_DISTANCE_KM = 5.0
PARTIAL_DISTANCE_KM = 15.0
MATCH_STRIKE_DEG = 20.0
PARTIAL_STRIKE_DEG = 40.0


def download_quaternary_faults(output_dir: Path = Path("data")) -> gpd.GeoDataFrame:
    """Download and return USGS Quaternary Fault traces."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    zip_path = output_dir / "qfaults_us.zip"
    if not zip_path.exists():
        log.info("Downloading USGS Quaternary Fault Database...")
        resp = requests.get(QFFDB_URL, timeout=180, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        log.info(f"  Saved → {zip_path}")

    faults = gpd.read_file(f"zip://{zip_path}")
    faults = faults.to_crs(GEO_CRS)

    # Clip to Southern California study region
    faults = faults.cx[-121.0:-114.0, 32.0:37.0]
    log.info(f"USGS Quaternary Faults (SoCal): {len(faults)} features")
    return faults


def strike_discordance(strike1: float, strike2: float) -> float:
    """
    Minimum angular difference between two fault strikes (0–90°).

    Fault strikes are undirected lines (0–360° but 0°==180°),
    so maximum discordance is 90°.
    """
    diff = abs(strike1 - strike2) % 180
    return min(diff, 180 - diff)


def classify_structures(
    detected: gpd.GeoDataFrame,
    mapped: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Classify each detected fault structure relative to mapped faults.

    Parameters
    ----------
    detected : GeoDataFrame  Detected fault planes with 'strike_deg', geometry
    mapped   : GeoDataFrame  USGS Quaternary Fault traces

    Returns
    -------
    GeoDataFrame  detected with added columns:
        nearest_fault_name, nearest_distance_km, strike_discordance_deg,
        match_class (MATCHED / PARTIAL / UNMATCHED), match_score
    """
    # Reproject both to UTM for metric distance
    det_utm = detected.to_crs(UTM_CRS)
    mapped_utm = mapped.to_crs(UTM_CRS)

    # Extract strike from mapped faults (approximate from geometry)
    mapped_strikes = _estimate_fault_strikes(mapped_utm)

    results = []
    for idx, row in det_utm.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            results.append({
                "nearest_fault_name": None,
                "nearest_distance_km": np.nan,
                "strike_discordance_deg": np.nan,
                "match_class": "UNMATCHED",
                "match_score": 0.0,
            })
            continue

        # Distance from detected trace to all mapped faults
        distances_m = mapped_utm.geometry.distance(geom)
        nearest_idx = distances_m.idxmin()
        nearest_dist_km = distances_m[nearest_idx] / 1000.0

        # Get fault name
        name_cols = ["fault_name", "name", "FAULT_NAME", "NAME"]
        fault_name = "Unknown"
        for col in name_cols:
            if col in mapped.columns:
                fault_name = str(mapped.loc[nearest_idx, col])
                break

        # Strike discordance
        det_strike = row.get("strike_deg", np.nan)
        mapped_strike = mapped_strikes.get(nearest_idx, np.nan)
        disc = (
            strike_discordance(det_strike, mapped_strike)
            if not np.isnan(det_strike) and not np.isnan(mapped_strike)
            else np.nan
        )

        # Classification
        if nearest_dist_km <= MATCH_DISTANCE_KM and (np.isnan(disc) or disc <= MATCH_STRIKE_DEG):
            match_class = "MATCHED"
            match_score = 1.0 - nearest_dist_km / MATCH_DISTANCE_KM
        elif nearest_dist_km <= PARTIAL_DISTANCE_KM or (not np.isnan(disc) and disc <= PARTIAL_STRIKE_DEG):
            match_class = "PARTIAL"
            match_score = 0.5
        else:
            match_class = "UNMATCHED"
            match_score = 0.0

        results.append({
            "nearest_fault_name": fault_name,
            "nearest_distance_km": round(nearest_dist_km, 2),
            "strike_discordance_deg": round(disc, 1) if not np.isnan(disc) else np.nan,
            "match_class": match_class,
            "match_score": match_score,
        })

    result_df = pd.DataFrame(results, index=detected.index)
    detected = detected.copy()
    for col in result_df.columns:
        detected[col] = result_df[col]

    # Summary
    n = len(detected)
    n_matched = (detected["match_class"] == "MATCHED").sum()
    n_partial = (detected["match_class"] == "PARTIAL").sum()
    n_unmatched = (detected["match_class"] == "UNMATCHED").sum()

    log.info(f"Match classification: {n} structures")
    log.info(f"  MATCHED:   {n_matched} ({100*n_matched/n:.0f}%)")
    log.info(f"  PARTIAL:   {n_partial} ({100*n_partial/n:.0f}%)")
    log.info(f"  UNMATCHED: {n_unmatched} ({100*n_unmatched/n:.0f}%) ← candidate unmapped faults")

    # High-priority unmapped structures
    priority = detected[
        (detected["match_class"] == "UNMATCHED") &
        (detected["n_events"] >= 200)
    ].sort_values("n_events", ascending=False)

    if len(priority) > 0:
        log.info(f"\n  HIGH-PRIORITY unmapped structures (N≥200 events):")
        for _, row in priority.iterrows():
            log.info(
                f"    Cluster {row['cluster_id']:3d}: "
                f"strike={row['strike_deg']:.0f}°, "
                f"dip={row['dip_deg']:.0f}°, "
                f"N={row['n_events']}, "
                f"depth={row['depth_top_km']:.0f}–{row['depth_bottom_km']:.0f} km"
            )

    return detected


def compute_performance_metrics(detected: gpd.GeoDataFrame) -> dict:
    """
    Compute precision, recall, and F1 relative to USGS Quaternary Fault Database.

    Note: 'recall' here is approximate — the QFFDB may not capture all active
    structures in the seismicity record, so true recall is a lower bound.
    """
    n_total = len(detected)
    n_matched = (detected["match_class"] == "MATCHED").sum()
    n_partial = (detected["match_class"] == "PARTIAL").sum()
    n_unmatched = (detected["match_class"] == "UNMATCHED").sum()

    precision = n_matched / n_total if n_total > 0 else 0
    recall_proxy = n_matched / (n_matched + n_unmatched) if (n_matched + n_unmatched) > 0 else 0
    f1 = 2 * precision * recall_proxy / (precision + recall_proxy + 1e-10)

    metrics = {
        "n_detected": n_total,
        "n_matched": int(n_matched),
        "n_partial": int(n_partial),
        "n_unmatched": int(n_unmatched),
        "match_rate": round(n_matched / n_total, 3) if n_total > 0 else 0,
        "precision_approx": round(precision, 3),
        "recall_approx": round(recall_proxy, 3),
        "f1_approx": round(f1, 3),
    }

    log.info("Performance metrics (vs USGS Quaternary Fault Database):")
    for k, v in metrics.items():
        log.info(f"  {k}: {v}")

    return metrics


def _estimate_fault_strikes(faults_utm: gpd.GeoDataFrame) -> dict:
    """Estimate fault strike from geometry bearing (degrees CW from north)."""
    strikes = {}
    for idx, row in faults_utm.iterrows():
        geom = row.geometry
        try:
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
                dx = coords[-1][0] - coords[0][0]
                dy = coords[-1][1] - coords[0][1]
                strike = np.degrees(np.arctan2(dx, dy)) % 360
                strikes[idx] = strike
            else:
                strikes[idx] = np.nan
        except Exception:
            strikes[idx] = np.nan
    return strikes
