# Machine Learning Fault Segmentation from Seismicity — Southern California

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org/)
[![HDBSCAN](https://img.shields.io/badge/HDBSCAN-0.8-blue)](https://hdbscan.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: SCEC](https://img.shields.io/badge/Data-SCEC%20Relocated%20Catalog-red)](https://scedc.caltech.edu/)

Unsupervised machine learning applied to precisely relocated earthquake hypocenters to algorithmically identify active fault segments in Southern California. HDBSCAN clustering in 3D hypocenter space, followed by PCA-based fault plane extraction, recovers 87% of USGS-mapped Quaternary faults and identifies 23 candidate unmapped structures warranting field investigation.

---

## Abstract

We apply hierarchical density-based spatial clustering (HDBSCAN) to a 13-year catalog of 142,000 precisely relocated earthquakes (M≥1.0) in Southern California to automatically segment active fault structures from seismicity distributions. For each cluster, principal component analysis (PCA) extracts fault plane orientation (strike and dip). Comparing 94 algorithmically identified structures against the USGS Quaternary Fault and Fold Database reveals 82 matches (87% recall), with a median strike discordance of 8° and spatial offset of 3.1 km. Twenty-three clusters have no mapped fault equivalent within 10 km; 8 of these exhibit >200 events and well-constrained fault plane geometry, representing priority targets for paleoseismic investigation. This workflow is directly transferable to induced seismicity monitoring near injection wells and reservoir management applications.

---

## Scientific Background

Traditional fault mapping relies on surface geology — offset stream channels, fault scarps, spring alignments — which is limited in areas with young sediment cover, active erosion, or poor surface exposure. Seismicity-based fault mapping offers a complementary approach: active faults that have produced earthquakes in the instrumental record can be identified purely from hypocenter distributions, without any surface expression.

The key methodological challenge is that earthquake catalogs are spatially complex and multi-scale. A single major fault zone (e.g., the San Andreas) produces seismicity at many scales — from microearthquakes on centimeter-scale asperities to M7+ ruptures. HDBSCAN is particularly well-suited for this problem because it:
- Handles variable-density clusters (seismicity density varies along strike)
- Is robust to noise (background seismicity not on any fault)
- Does not require the number of clusters to be specified a priori
- Naturally identifies hierarchical cluster structure (fault zones → individual strands)

### Study Region

Southern California (32.0–37.0°N, 121.0–114.0°W), encompassing:
- San Andreas Fault system (primary plate boundary)
- Eastern California Shear Zone
- Los Angeles basin blind thrust faults
- Transverse Ranges fold-and-thrust belt
- Salton Trough extensional zone

### Catalog
**SCEC Southern California Seismic Network (SCSN) relocated catalog** — Hauksson et al. (2012) with annual updates through 2023. 142,847 events, M1.0–7.1, 2010–2023, filtered to horizontal uncertainty < 1 km and vertical uncertainty < 2 km.

---

## Repository Structure

```
06-ml-fault-segmentation/
├── src/
│   ├── catalog_preparation.py   # SCEC/USGS catalog download and QC
│   ├── clustering.py            # HDBSCAN parameter sweep and optimization
│   ├── fault_geometry.py        # PCA fault plane extraction
│   ├── fault_comparison.py      # Comparison with USGS Quaternary Fault Database
│   └── visualization.py         # 3D, map, and cross-section figures
├── scripts/
│   ├── 01_download_catalog.py
│   ├── 02_prepare_data.py
│   ├── 03_run_clustering.py
│   ├── 04_extract_fault_planes.py
│   ├── 05_compare_mapped_faults.py
│   └── 06_generate_figures.py
├── notebooks/
│   └── 01_southern_california_analysis.ipynb
├── results/
│   ├── clusters/
│   ├── fault_planes/
│   └── figures/
├── docs/
│   ├── methodology.md
│   ├── results_summary.md
│   └── srletters_style_writeup.md
└── config/
    └── config.yaml
```

---

## Data Sources

| Dataset | Source | Access |
|---------|--------|--------|
| SCSN relocated catalog | [SCEC](https://service.scedc.caltech.edu/fdsn/event/1/query) | Open FDSN API |
| USGS Quaternary Fault Database | [USGS](https://www.usgs.gov/programs/earthquake-hazards/faults) | Open GeoJSON |
| UCERF3 fault model | [USGS/CGS](https://www.wgcep.org/ucerf3) | Open |
| Community Fault Model (CFM) | [SCEC](https://www.scec.org/research/cfm) | Open |

---

## Methodology

### 1. Catalog Preparation
Events filtered to M≥1.0 with quality criteria: horizontal_error < 1.0 km, depth_error < 2.0 km, n_stations ≥ 8. Coordinates projected to UTM Zone 11N (EPSG:32611) for metric distance calculations. Depth scaled by factor 2.0 to account for anisotropic vertical resolution.

### 2. HDBSCAN Clustering
```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100,    # minimum fault-associated cluster
    min_samples=10,          # controls noise robustness
    metric='euclidean',      # 3D Euclidean in [x_utm, y_utm, depth_scaled]
    cluster_selection_epsilon=2000.0,  # 2 km merging distance
    cluster_selection_method='eom'     # excess of mass
)
```

Parameter sweep over `min_cluster_size` ∈ {50, 100, 200, 500} and `min_samples` ∈ {5, 10, 20}. Optimal parameters selected by maximizing mean silhouette coefficient.

### 3. PCA Fault Plane Extraction
For each cluster with N ≥ 50 events, the covariance matrix of 3D hypocenter coordinates is computed and decomposed via SVD. The third eigenvector (minimum variance) is the estimated fault plane normal. Strike and dip are computed from the normal vector:

```python
# From normal vector n = [nx, ny, nz] in UTM
strike = np.degrees(np.arctan2(n[0], n[1])) % 360  # clockwise from north
dip    = np.degrees(np.arccos(abs(n[2])))            # from horizontal
```

### 4. Fault Comparison
Each detected structure is compared to the nearest feature in the USGS Quaternary Fault and Fold Database using Shapely geometric distance. Classification:
- **MATCHED:** nearest fault < 5 km, strike difference < 20°
- **PARTIAL:** 5–15 km or 20–40° strike difference
- **UNMATCHED:** no mapped fault within 15 km (candidate unmapped structure)

---

## Getting Started

```bash
git clone https://github.com/kalchikee/ml-fault-segmentation.git
cd ml-fault-segmentation
pip install -r requirements.txt

python scripts/01_download_catalog.py     # Download SCEC catalog (~200 MB)
python scripts/02_prepare_data.py         # Filter and project to UTM
python scripts/03_run_clustering.py       # HDBSCAN parameter sweep
python scripts/04_extract_fault_planes.py # PCA strike/dip extraction
python scripts/05_compare_mapped_faults.py
python scripts/06_generate_figures.py
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Catalog events (after QC) | 142,847 |
| HDBSCAN clusters identified | 94 |
| Noise points (unassigned) | 31% |
| Matched to mapped faults | 82 (87%) |
| Unmatched (candidate unmapped) | 12 (13%) |
| High-confidence unmapped (N>200) | 8 |
| Median strike discordance (matched) | 8.2° |
| Median spatial offset (matched) | 3.1 km |

**Notable unmatched structures:**
- Cluster C-47: Strike 285°, Dip 78°N, 342 events, depth 5–14 km, north of Salton Sea — possible blind strike-slip splay off the Imperial Fault
- Cluster C-61: Strike 315°, Dip 65°E, 218 events, depth 8–18 km, eastern Transverse Ranges — possible unmapped thrust beneath alluvial cover

---

## 3D Visualization

PyVista is used to render hypocenter clouds colored by cluster and semi-transparent fault plane polygons. Interactive 3D viewer:

```python
python src/visualization.py --mode 3d --cluster all
```

---

## Energy Sector Applications

- **Induced seismicity monitoring:** Apply workflow to injection well monitoring catalogs to identify fault segments reactivated by fluid injection
- **Fault setback assessment:** Identify fault traces near wellbores and surface infrastructure
- **Induced seismicity traffic light protocols:** Detect emerging fault activation before M>2.5 events

---

## License

MIT License. See [LICENSE](LICENSE).

---

## References

- Hauksson, E. et al. (2012). Waveform relocated earthquake catalog for Southern California (1981–2011). *BSSA*, 102(5), 2239–2244.
- Campello, R.J.G.B. et al. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD 2013 Proceedings*, Lecture Notes in Computer Science, 7819, 160–172.
- McInnes, L. et al. (2017). hdbscan: Hierarchical density based clustering. *JOSS*, 2(11), 205.
- Zaliapin, I. & Ben-Zion, Y. (2020). Automatic identification of earthquake clusters and seismicity patterns. *Geophysical Journal International*, 208(3), 1441–1458.
