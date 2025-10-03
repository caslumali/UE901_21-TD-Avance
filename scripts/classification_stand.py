#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classification_stand.py

Production d'une carte à l'échelle des peuplements à partir de la carte
des essences forestières à l'échelle du pixel.

Étapes principales :
1) Lecture du shapefile d'échantillons (en GeoDataFrame).
2) Calcul des statistiques zonales (pourcentages des classes présentes).
3) Attribution d'une classe de peuplement pour chaque polygone.
4) Mise à jour du shapefile avec le champ "Code_pred".
5) Génération d'une matrice de confusion (comparaison champ "Code" vs champ "Code_pred").
6) Sauvegarde de la matrice de confusion sous forme d'image.
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Chemin vers votre librairie perso
sys.path.append('libsigma')  # noqa
from plots import plot_cm

# Import des fonctions stand
from my_function import (
    get_pixel_area,
    compute_peuplement_class
)

# ---------------------------------------------------------------------------
# Configuration globale
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

start_time = time.time()
logging.info("Début du traitement de la carte à l'échelle des peuplements.")

# Dictionnaires de chemins (à adapter selon votre arborescence)
CHEMINS = {
    "sample": "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp",
    "classification": "groupe_9/results/data/classif/carte_essences_echelle_pixel.tif"
}
CLASSIF_FOLDER = "groupe_9/results/data/classif"

# Paramètres de classification
SEUIL_CATEGORIE = 75.0  # Pourcentage minimal pour catégoriser majoritairement F ou C
SURFACE_MINI = 20000    # Seuil en m² pour distinguer petits îlots / grands polygones

# Codes de peuplement : adapter aux valeurs de votre nomenclature
CODES_PEUPLEMENT = {
    "feuillus_ilots": 16,
    "melange_feuillus": 15,
    "coniferes_ilots": 27,
    "melange_coniferes": 26,
    "melange_conif_prep_feuil": 28,
    "melange_feuil_prep_conif": 29
}

# ---------------------------------------------------------------------------
# 1) Lecture du raster et calcul de la surface d'un pixel
# ---------------------------------------------------------------------------
raster_path = CHEMINS["classification"]
pixel_surface = get_pixel_area(raster_path)
logging.info(f"Surface d'un pixel : {pixel_surface:.2f} m²")

# ---------------------------------------------------------------------------
# 2) Lecture du shapefile et calcul des statistiques zonales
# ---------------------------------------------------------------------------
gdf_sample = gpd.read_file(CHEMINS["sample"])
logging.info(
    f"Lecture du shapefile : {CHEMINS['sample']} (taille={len(gdf_sample)})")

start_stats = time.time()

stats_zonales = zonal_stats(
    gdf_sample,
    raster_path,
    all_touched=True,
    categorical=True,
    nodata=0
)

end_stats = time.time()
logging.info(
    f"Statistiques zonales calculées en {end_stats - start_stats:.2f} s.")

# ---------------------------------------------------------------------------
# 3) Attribution des classes de peuplements
# ---------------------------------------------------------------------------
classes_predites, pct_feu, pct_conif, surfaces = compute_peuplement_class(
    stats_zonales=stats_zonales,
    pixel_surface=pixel_surface,
    seuil_categorie=SEUIL_CATEGORIE,
    surface_mini=SURFACE_MINI,
    codes_peuplement=CODES_PEUPLEMENT
)

# ---------------------------------------------------------------------------
# 4) Mise à jour du shapefile : champ "Code_pred"
# ---------------------------------------------------------------------------
gdf_sample["Code_pred"] = classes_predites

# Mise à jour le shapefile
gdf_sample.to_file(CHEMINS["sample"], driver="ESRI Shapefile")
logging.info("Shapefile mis à jour avec le champ Code_pred.")

# ---------------------------------------------------------------------------
# 5) Matrice de confusion
# ---------------------------------------------------------------------------
true_labels = gdf_sample["Code"].astype(int).values
pred_labels = gdf_sample["Code_pred"].astype(int).values

# Récupérer toutes les classes possibles (réelles + prédites)
all_labels = sorted(np.unique(np.concatenate([true_labels, pred_labels])))

cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
logging.info("Matrice de confusion calculée.")

# Enregistrer la matrice de confusion en .CSV
cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
cm_df.to_csv(os.path.join(CLASSIF_FOLDER, "confusion_matrix_peuplements.csv"))

# Faire la classification_report
report = classification_report(
    true_labels, pred_labels, labels=all_labels, zero_division=0
)
logging.info(f"Rapport de classification :\n{report}")

# ---------------------------------------------------------------------------
# 6) Sauvegarde de la matrice de confusion
# ---------------------------------------------------------------------------
out_cm_path = os.path.join(CLASSIF_FOLDER, "confusion_matrix_peuplements.png")
plot_cm(
    cm,
    labels=all_labels,
    out_filename=out_cm_path,
    normalize=False,
    cmap="Greens"
)
logging.info(f"Matrice de confusion sauvegardée : {out_cm_path}")

# ---------------------------------------------------------------------------
# Fin
# ---------------------------------------------------------------------------
elapsed = time.time() - start_time
logging.info(f"Fin de la classification stand. Durée : {elapsed:.2f} s.")
