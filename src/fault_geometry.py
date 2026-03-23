"""
fault_geometry.py
-----------------
PCA-based fault plane extraction from HDBSCAN hypocenter clusters.

For each cluster, fits a planar surface to the 3D hypocenter cloud using
principal component analysis (SVD). Extracts strike, dip, centroid, and
spatial extents. Generates fault trace polygons and surface trace lines.

Theory:
    Earthquake hypocenters on a fault plane lie approximately within a planar
    distribution. The covariance matrix of hypocenter positions has its smallest
    eigenvalue perpendicular to the fault plane (minimum variance direction).
    SVD of the centered coordinate matrix gives:
        - eigenvector 1 (largest variance): along-strike direction
        - eigenvector 2 (intermediate variance): down-dip direction
        - eigenvector 3 (smallest variance): fault plane normal
"""

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

log = logging.getLogger(__name__)

UTM_CRS = "EPSG:32611"
GEO_CRS = "EPSG:4326"

MIN_EVENTS_FOR_PLANE = 50  # Minimum cluster size for reliable PCA


def extract_fault_planes(
    gdf: gpd.GeoDataFrame,
    X: np.ndarray,
    min_events: int = MIN_EVENTS_FOR_PLANE,
) -> gpd.GeoDataFrame:
    """
    Extract fault plane parameters for all clusters via PCA.

    Parameters
    ----------
    gdf : GeoDataFrame with 'cluster_id' column
    X : np.ndarray, shape (N, 3)  Feature matrix [x_utm, y_utm, depth_m]
    min_events : int  Minimum cluster size

    Returns
    -------
    GeoDataFrame with one row per cluster, columns:
        cluster_id, n_events, strike_deg, dip_deg, dip_direction,
        centroid_lat, centroid_lon, centroid_depth_km,
        length_km, width_km, depth_top_km, depth_bottom_km,
        explained_variance_ratio, geometry (surface trace LineString)
    """
    # Use raw UTM coordinates (not depth-weighted) for geometry extraction
    gdf_utm = gdf.to_crs(UTM_CRS) if gdf.crs != UTM_CRS else gdf

    records = []
    cluster_ids = sorted([c for c in gdf["cluster_id"].unique() if c >= 0])

    for cid in cluster_ids:
        mask = gdf["cluster_id"] == cid
        if mask.sum() < min_events:
            continue

        cluster = gdf[mask]
        # Raw UTM coordinates (depth in meters, not scaled)
        x = gdf_utm[mask].geometry.x.values
        y = gdf_utm[mask].geometry.y.values
        z = cluster["depth_km"].values * 1000.0  # meters

        coords = np.column_stack([x, y, z])
        centroid = coords.mean(axis=0)
        centered = coords - centroid

        # SVD for PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Vt rows are eigenvectors; last row is fault plane normal (min variance)
        normal = Vt[2]  # fault plane normal
        along_strike = Vt[0]  # along-strike direction
        along_dip = Vt[1]  # down-dip direction

        # Ensure normal points upward (positive z component)
        if normal[2] < 0:
            normal = -normal

        # ── Strike and Dip ────────────────────────────────────────────────────
        # Strike: azimuth of the fault in the horizontal plane
        # normal = [nx, ny, nz]; strike ⊥ to horizontal projection of normal
        n_horiz = np.array([normal[0], normal[1], 0.0])
        n_horiz_norm = n_horiz / (np.linalg.norm(n_horiz) + 1e-10)

        # Strike direction is perpendicular to horizontal normal
        # Following right-hand rule: strike = 90° from dip direction
        strike_rad = np.arctan2(n_horiz_norm[0], n_horiz_norm[1]) + np.pi / 2
        strike_deg = np.degrees(strike_rad) % 360

        # Dip: angle between fault normal and vertical
        dip_deg = np.degrees(np.arccos(np.clip(np.abs(normal[2]), 0, 1)))

        # Dip direction (azimuth down-dip)
        dip_dir = (strike_deg + 90) % 360

        # ── Spatial Extents ───────────────────────────────────────────────────
        proj_strike = centered @ Vt[0]
        proj_dip = centered @ Vt[1]

        length_km = (proj_strike.max() - proj_strike.min()) / 1000.0
        width_km = (proj_dip.max() - proj_dip.min()) / 1000.0

        depth_top_km = z.min() / 1000.0
        depth_bottom_km = z.max() / 1000.0

        # ── Centroid in Geographic Coordinates ────────────────────────────────
        import pyproj
        transformer = pyproj.Transformer.from_crs(UTM_CRS, GEO_CRS, always_xy=True)
        centroid_lon, centroid_lat = transformer.transform(centroid[0], centroid[1])
        centroid_depth_km = centroid[2] / 1000.0

        # ── Surface Trace ─────────────────────────────────────────────────────
        # Project ±2σ along strike to get fault trace extents
        sigma_strike = proj_strike.std()
        end1 = centroid + 2 * sigma_strike * Vt[0]
        end2 = centroid - 2 * sigma_strike * Vt[0]

        # Project both endpoints to surface (depth=0) using dip geometry
        def to_surface(pt, depth_km, dip_deg, dip_dir_deg):
            horiz_offset = depth_km / np.tan(np.radians(max(dip_deg, 1.0)))
            dip_rad = np.radians(dip_dir_deg)
            dx = horiz_offset * np.sin(dip_rad)
            dy = horiz_offset * np.cos(dip_rad)
            return pt[0] + dx, pt[1] + dy

        x1, y1 = to_surface(end1, end1[2] / 1000, dip_deg, dip_dir)
        x2, y2 = to_surface(end2, end2[2] / 1000, dip_deg, dip_dir)

        lon1, lat1 = transformer.transform(x1, y1)
        lon2, lat2 = transformer.transform(x2, y2)
        trace = LineString([(lon1, lat1), (lon2, lat2)])

        # Variance explained
        explained = (S ** 2) / (S ** 2).sum()

        records.append({
            "cluster_id": int(cid),
            "n_events": int(mask.sum()),
            "strike_deg": round(strike_deg, 1),
            "dip_deg": round(dip_deg, 1),
            "dip_direction": round(dip_dir, 1),
            "centroid_lat": round(centroid_lat, 4),
            "centroid_lon": round(centroid_lon, 4),
            "centroid_depth_km": round(centroid_depth_km, 2),
            "length_km": round(length_km, 2),
            "width_km": round(width_km, 2),
            "depth_top_km": round(depth_top_km, 2),
            "depth_bottom_km": round(depth_bottom_km, 2),
            "explained_var_1": round(explained[0], 3),
            "explained_var_2": round(explained[1], 3),
            "planarity": round(1.0 - explained[2], 3),  # 1 = perfectly planar
            "geometry": trace,
        })

    planes = gpd.GeoDataFrame(records, crs=GEO_CRS)
    log.info(f"Extracted {len(planes)} fault planes from {len(cluster_ids)} clusters")

    # Log high-planarity structures
    high_planarity = planes[planes["planarity"] > 0.85]
    log.info(
        f"  High planarity (>0.85): {len(high_planarity)} structures "
        f"(most geologically coherent)"
    )

    return planes
