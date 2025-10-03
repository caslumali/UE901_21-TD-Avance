
# UE901_21 - Advanced Remote Sensing

**Supervised classification of forest tree species in the Toulouse region using Sentinel-2 time series and BD Forêt V2 data**  
📍 *SIGMA Master – Advanced Remote Sensing Module (UE901)*

---

## 🌍 Project Overview

This project was developed as part of the **UE901 – Advanced Remote Sensing** course within the **SIGMA Master’s program** (*“Sciences géomatiques en environnement et aménagement”*), jointly delivered by **Agro Toulouse (ENSAT)** and **Université Toulouse II – Jean Jaurès**.

The objective was to **evaluate the relevance of the French national forest database (BD Forêt V2)** as a source of training samples for supervised classification of forest tree species, using **Sentinel-2 time series** and the **Random Forest algorithm**. The study area covers the **Toulouse region (tile 31TCJ)**.

Two classification scales were explored:
- 🌐 **Pixel level** classification using Sentinel-2 time series  
- 🟩 **Forest stand level** classification aggregated from pixel predictions

---

## 🧭 Methodology

The processing workflow was divided into four main phases:

1. **Filtering BD Forêt V2 data** to select relevant forest classes and create a binary mask.  
2. **Pre-processing Sentinel-2 imagery** (2022–2023) to generate:
   - a **60-band multitemporal raster** (10 bands × 6 dates)  
   - a **6-band NDVI time series raster**
3. **Sample analysis**, including class distribution, phenological profiles of pure stands, and spectral variability analysis.
4. **Supervised classification** using the **Random Forest** algorithm (StratifiedGroupKFold cross-validation, 30 iterations), followed by evaluation metrics and confusion matrices.

A key design choice was to implement a **memory-based pipeline** using GDAL’s `MEM` driver to avoid unnecessary intermediate files and optimize computation. This approach resulted in a **fast, transparent and traceable** processing chain.

---

## 🛰 Data Sources

- **Sentinel-2 L2A** (Theia platform, 2022–2023), 10 m resolution, <15 % cloud cover  
- **BD Forêt V2** (IGN) — national forest inventory database  
- **Study area:** Toulouse region, France (Tile 31TCJ, EPSG:2154)

---

## 🧪 Results

- **High overall accuracy (OA)** of 0.93 for pixel-based classification using all Sentinel-2 bands.  
- **Oak (Quercus)** was dominant in the dataset, resulting in excellent accuracy for this class (F1-score ≈ 97 %) but lower performance for under-represented species.  
- **NDVI time series** alone achieved OA ≈ 0.91 but showed greater class imbalance.  
- Classification at **stand level** provided more stable and interpretable maps but highlighted systematic misclassifications between mixed and pure stands.

The results underline both the potential and the limitations of using BD Forêt V2 for training species classification models, especially in the presence of **high class imbalance** and **phenological variability**.

---

## 🧠 Key Skills & Tools

- **Remote Sensing**: Sentinel-2 time series, NDVI, multi-temporal analysis  
- **Geospatial processing**: GDAL, Rasterstats, QGIS, REGEX for band parsing  
- **Machine Learning**: Random Forest (scikit-learn), StratifiedGroupKFold cross-validation  
- **Programming**: Python (VS Code), logging & error handling for large raster processing  
- **Version control**: Collaborative work with Git & GitHub  
- **HPC / cloud**: Onyxia platform for classification

---

## 📂 Repository Structure

```

UE901_21-TD-Avance/
├── scripts/                 # Python scripts (preprocessing, classification, analysis)
├── libsigma/                # Helper functions and plotting utilities
├── results/                 # Raster outputs, figures, evaluation metrics
├── rapport/                 # Final report (PDF)
├── arborescence.txt         # Folder structure used in the project
└── README.md

```

---

## 🤝 Team

- **Alban Dumont**  
- **Robin Heckendorn**  
- **Lucas Lima** *(mapping & validation)*

---

## 📚 Reference

Dumont A., Heckendorn R., Lima L. (2025). *Supervised classification of forest species in the Toulouse region using Sentinel-2 time series*. UE901 – Advanced Remote Sensing, Master SIGMA, Agro Toulouse.

---

## 📝 License

This project was developed for academic purposes.  
Feel free to explore, cite, or adapt with proper credit.

---

```
