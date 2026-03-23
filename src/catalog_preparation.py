"""
catalog_preparation.py
----------------------
Download and prepare the SCEC Southern California relocated earthquake catalog
for 3D HDBSCAN fault segmentation.

Data source: SCEC Southern California Seismic Network (SCSN)
API: https://service.scedc.caltech.edu/fdsn/event/1/query

Quality filters applied:
  - M >= 1.0
  - horizontal_error < 1.0 km
  - depth_error < 2.0 km
  - n_stations >= 8
"""

import logging
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import requests
from shapely.geometry import Point
from tqdm import tqdm

log = logging.getLogger(__name__)

# SCEC FDSN API endpoint (Southern California)
SCEC_API = "https://service.scedc.caltech.edu/fdsn/event/1/query"

# USGS as fallback (has broader catalog including relocated events)
USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Study region: Southern California
STUDY_REGION = {
    "minlatitude": 32.0,
    "maxlatitude": 37.0,
    "minlongitude": -121.0,
    "maxlongitude": -114.0,
}

# UTM Zone 11N for metric coordinates
UTM_CRS = "EPSG:32611"

# Quality thresholds
QUALITY_FILTERS = {
    "minmagnitude": 1.0,
    "max_horizontal_error_km": 1.0,
    "max_depth_error_km": 2.0,
    "min_stations": 8,
}


def download_catalog_chunked(
    starttime: str = "2010-01-01",
    endtime: str = "2023-12-31",
    chunk_years: int = 2,
    output_dir: Path = Path("data"),
    api_url: str = USGS_API,
) -> gpd.GeoDataFrame:
    """
    Download SCSN catalog in time chunks to handle API limits.

    The USGS FDSN API returns max 20,000 events per request. Southern California
    produces ~10,000–20,000 events/year, so we chunk by year.

    Parameters
    ----------
    starttime : str  ISO 8601 start date
    endtime   : str  ISO 8601 end date
    chunk_years : int  Years per API request
    output_dir : Path  Save directory
    api_url : str  FDSN endpoint

    Returns
    -------
    GeoDataFrame  Combined catalog with all events
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "scec_raw_catalog.gpkg"
    if cache_path.exists():
        log.info(f"Loading cached catalog: {cache_path}")
        return gpd.read_file(cache_path)

    starts = pd.date_range(starttime, endtime, freq=f"{chunk_years}YE")
    ends = starts[1:]
    if len(ends) < len(starts):
        ends = ends.append(pd.DatetimeIndex([endtime]))

    all_records = []
    for s, e in tqdm(zip(starts, ends), total=len(starts), desc="Downloading chunks"):
        params = {
            "format": "geojson",
            "starttime": s.strftime("%Y-%m-%d"),
            "endtime": e.strftime("%Y-%m-%d"),
            "minmagnitude": QUALITY_FILTERS["minmagnitude"],
            **STUDY_REGION,
            "orderby": "time",
            "limit": 20000,
        }
        try:
            resp = requests.get(api_url, params=params, timeout=120)
            resp.raise_for_status()
            features = resp.json()["features"]
            log.info(f"  {s.year}: {len(features)} events")
            all_records.extend(features)
        except Exception as exc:
            log.warning(f"  {s.year}: download failed — {exc}")

    log.info(f"Total raw events: {len(all_records):,}")
    gdf = _parse_geojson_features(all_records)
    gdf.to_file(cache_path, driver="GPKG")
    return gdf


def _parse_geojson_features(features: list) -> gpd.GeoDataFrame:
    """Parse USGS GeoJSON feature list into GeoDataFrame."""
    records = []
    for feat in features:
        p = feat["properties"]
        c = feat["geometry"]["coordinates"]
        records.append({
            "event_id": feat["id"],
            "time": pd.to_datetime(p["time"], unit="ms", utc=True),
            "magnitude": p.get("mag"),
            "mag_type": p.get("magType", ""),
            "depth_km": c[2],
            "longitude": c[0],
            "latitude": c[1],
            "n_stations": p.get("nst"),
            "horizontal_error_km": p.get("horizontalError"),
            "depth_error_km": p.get("depthError"),
            "rms": p.get("rms"),
            "net": p.get("net", ""),
        })

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    return gdf


def apply_quality_filters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Apply location quality filters for reliable 3D clustering.

    Returns filtered GeoDataFrame with summary logged.
    """
    n_start = len(gdf)

    # Magnitude threshold
    gdf = gdf[gdf["magnitude"] >= QUALITY_FILTERS["minmagnitude"]].copy()

    # Horizontal error
    if "horizontal_error_km" in gdf.columns:
        mask = (
            gdf["horizontal_error_km"].isna() |
            (gdf["horizontal_error_km"] <= QUALITY_FILTERS["max_horizontal_error_km"])
        )
        gdf = gdf[mask].copy()

    # Depth error
    if "depth_error_km" in gdf.columns:
        mask = (
            gdf["depth_error_km"].isna() |
            (gdf["depth_error_km"] <= QUALITY_FILTERS["max_depth_error_km"])
        )
        gdf = gdf[mask].copy()

    # Station count
    if "n_stations" in gdf.columns:
        mask = (
            gdf["n_stations"].isna() |
            (gdf["n_stations"] >= QUALITY_FILTERS["min_stations"])
        )
        gdf = gdf[mask].copy()

    # Remove events with no depth
    gdf = gdf.dropna(subset=["depth_km", "latitude", "longitude", "magnitude"])
    gdf = gdf[gdf["depth_km"] >= 0].copy()

    n_end = len(gdf)
    log.info(f"Quality filter: {n_start:,} → {n_end:,} events ({100*n_end/n_start:.1f}% retained)")
    return gdf


def project_to_utm(gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project to UTM Zone 11N and return coordinate arrays.

    Returns
    -------
    x_utm : np.ndarray  Easting (m)
    y_utm : np.ndarray  Northing (m)
    z_m   : np.ndarray  Depth (m, positive downward)
    """
    gdf_utm = gdf.to_crs(UTM_CRS)
    x = gdf_utm.geometry.x.values
    y = gdf_utm.geometry.y.values
    z = gdf["depth_km"].values * 1000.0  # convert to meters
    return x, y, z


def build_feature_matrix(
    gdf: gpd.GeoDataFrame,
    depth_weight: float = 2.0,
) -> np.ndarray:
    """
    Build the 3D feature matrix for clustering: [x_utm, y_utm, depth_scaled].

    Depth is multiplied by depth_weight to account for the reduced vertical
    resolution of earthquake locations compared to horizontal.

    Parameters
    ----------
    gdf : GeoDataFrame with projected coordinates
    depth_weight : float  Multiplier for depth dimension

    Returns
    -------
    X : np.ndarray, shape (N, 3)
    """
    x, y, z = project_to_utm(gdf)
    X = np.column_stack([x, y, z * depth_weight])
    return X


def prepare_catalog(
    starttime: str = "2010-01-01",
    endtime: str = "2023-12-31",
    output_dir: Path = Path("data"),
) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
    """
    Full catalog preparation pipeline: download → filter → project.

    Returns
    -------
    gdf : GeoDataFrame  Quality-filtered catalog
    X   : np.ndarray    Feature matrix for HDBSCAN (N × 3)
    """
    gdf_raw = download_catalog_chunked(
        starttime=starttime,
        endtime=endtime,
        output_dir=output_dir,
    )

    gdf_filtered = apply_quality_filters(gdf_raw)

    X = build_feature_matrix(gdf_filtered, depth_weight=2.0)

    # Save filtered catalog
    filtered_path = Path(output_dir) / "scec_filtered_catalog.gpkg"
    gdf_filtered.to_file(filtered_path, driver="GPKG")
    log.info(f"Filtered catalog → {filtered_path}")

    # Summary statistics
    log.info("─── Filtered Catalog Summary ────────────────────────")
    log.info(f"  Events:        {len(gdf_filtered):,}")
    log.info(f"  Mag range:     {gdf_filtered['magnitude'].min():.1f}–{gdf_filtered['magnitude'].max():.1f}")
    log.info(f"  Depth range:   {gdf_filtered['depth_km'].min():.1f}–{gdf_filtered['depth_km'].max():.1f} km")
    log.info(f"  Time span:     {gdf_filtered['time'].min()} → {gdf_filtered['time'].max()}")
    log.info("─────────────────────────────────────────────────────")

    return gdf_filtered, X
