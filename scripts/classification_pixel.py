#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de classification supervisée des essences forestières à l'échelle du pixel,
avec validation croisée répétée (StratifiedGroupKFold) prenant en compte
l'appartenance des pixels à un polygone (ID_poly). Les hyperparamètres
du RandomForest sont fixés selon la consigne.

Étapes principales :
1) Lecture du shapefile d'échantillons (en GeoDataFrame).
2) Création dynamique d'un champ ID_poly (entier) pour chaque polygone (sans
   modifier le shapefile original).
3) Rasterisation pour obtenir deux rasters :
   - "Sample_BD_foret_T31TCJ.tif" (champ "Code")
   - "Sample_id_BD_foret_T31TCJ.tif" (champ "ID_poly")
4) Extraction des échantillons (X, Y, coords) et des groupes (G).
5) Mise en place d'une validation croisée répétée (30 fois × 5 folds).
6) Calcul des métriques moyennes (CM, OA, classification_report).
7) Apprentissage final sur l'ensemble filtré (X_f, Y_f)
8) Prédiction sur TOUT le raster (X_orig)
9) Sauvegarde de la carte en TIFF uint8 (NoData = 0) + figures/statistiques.

@author: Alban Dumont, Lucas Lima, Robin Heckendorn
"""

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
from sklearn.ensemble import RandomForestClassifier as RF

sys.path.append('libsigma')
import classification as cla
import read_and_write as rw
from plots import plot_cm, plot_mean_class_quality

# personal librairies
from my_function import cleanup_temp_files, create_raster_sampleimage, report_from_dict_to_df

# ---------------------------------------------------------------------------
# 1) Logger et paramètres
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
start_time = time.time()
logging.info(
    "Début de la classification supervisée (StratifiedGroupKFold répété).")

# inputs
sample_folder = "groupe_9/results/data/sample"
pretraitees_folder = "groupe_9/results/data/img_pretraitees"
out_classif_folder = "groupe_9/results/data/classif"
os.makedirs(out_classif_folder, exist_ok=True)

shp_original = os.path.join(sample_folder, "Sample_BD_foret_T31TCJ.shp")
image_reference = os.path.join(pretraitees_folder, "masque_foret.tif")
image_filename = os.path.join(pretraitees_folder, "Serie_temp_S2_allbands.tif")

# Fichiers de sortie
raster_code_path = os.path.join(
    out_classif_folder, "Sample_BD_foret_T31TCJ.tif")
raster_id_path = os.path.join(
    out_classif_folder, "Sample_id_BD_foret_T31TCJ.tif")

out_classif = os.path.join(
    out_classif_folder, "carte_essences_echelle_pixel.tif")
out_mean_csv = os.path.join(out_classif_folder, "mean.csv")
out_std_csv = os.path.join(out_classif_folder, "std.csv")
out_fig_acc = os.path.join(out_classif_folder, "accuracy_distribution.png")
out_fig_report = os.path.join(out_classif_folder, "Std_Dev_and_Mean.png")

# Paramètres du RandomForest
nb_iter = 30
n_jobs = -1
nb_folds = 5
max_depth = 50
oob_score = True
max_samples = 0.75
class_weight = "balanced"
n_estimators = 100   # valeur par défaut

# Création du répertoire de sortie si nécessaire
output_dir = os.path.dirname(out_classif)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Répertoire de sortie créé : {output_dir}")

# ---------------------------------------------------------------------------
# 2) Lecture du shapefile + création d'ID_poly
# ---------------------------------------------------------------------------
logging.info("Lecture du shapefile et création ID_poly.")
gdf = gpd.read_file(shp_original)
gdf["ID_poly"] = range(1, len(gdf)+1)

# Création d'un shapefile temporaire sans extension explicite
temp_shp_id_base = os.path.join(out_classif_folder, "tmp_shp_for_ID")
for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
    f_ = temp_shp_id_base + ext
    if os.path.exists(f_):
        os.remove(f_)

gdf.to_file(temp_shp_id_base + ".shp", driver="ESRI Shapefile")
logging.info("Shapefile temporaire ID_poly créé.")

# ---------------------------------------------------------------------------
# 3) Rasterisation
# ---------------------------------------------------------------------------
if not os.path.exists(raster_code_path):
    logging.info("Création du raster Code (pour Y).")
    create_raster_sampleimage(
        sample_vector=shp_original,
        reference_raster=image_reference,
        output_path=raster_code_path,
        attribute="Code"
    )

if not os.path.exists(raster_id_path):
    logging.info("Création du raster ID_poly (pour G).")
    create_raster_sampleimage(
        sample_vector=temp_shp_id_base + ".shp",
        reference_raster=image_reference,
        output_path=raster_id_path,
        attribute="ID_poly"
    )


# ---------------------------------------------------------------------------
# 4) Extraction COMPLETE (X_orig, Y_orig, coords_orig) + G_orig
# ---------------------------------------------------------------------------
logging.info("Obtention du Sample X, Y, t ")
X, Y, t = cla.get_samples_from_roi(image_filename, raster_code_path)
logging.info("Obtention des groupes")
_, groups, _ = cla.get_samples_from_roi(image_filename, raster_id_path)

# ---------------------------------------------------------------------------
# 4) Filtrage pour train/val
# ---------------------------------------------------------------------------
# Valeurs à supprimer
values_to_delete = [15, 16, 26, 27, 28, 29]
Index_Essence_to_Delete = np.where(np.isin(Y, values_to_delete))

Y_skf = np.delete(Y, Index_Essence_to_Delete, 0)
X_skf = np.delete(X, Index_Essence_to_Delete, 0)
groups = np.delete(groups, Index_Essence_to_Delete, 0)

list_cm = []
list_accuracy = []
list_report = []
groups = np.squeeze(groups)

# ---------------------------------------------------------------------------
# 6) Configuration du RandomForest + validation croisée
# ---------------------------------------------------------------------------
# Liste des classes attendues
essence_tree = [11, 12, 13, 14, 21, 22, 23, 24, 25]
clf = RF(max_depth=max_depth,
        n_estimators=n_estimators,
        oob_score=oob_score,
        n_jobs=n_jobs,
        class_weight=class_weight,
        max_samples=max_samples)

# Iter on stratified K fold
logging.info(f"Validation croisée répétée : {nb_iter} x {nb_folds}")
for repetition in range(nb_iter):
    start_rep = time.time()
    logging.info(f"--- Début de la répétition {repetition+1}/{nb_iter} ---")    
    sgkf = StratifiedGroupKFold(n_splits=nb_folds, shuffle=True)

    for fold_i, (train, test) in enumerate(sgkf.split(X_skf, Y_skf, groups=groups)):
        start_fold = time.time()

        # Séparer les ensembles train/test
        logging.info(f"Fold {fold_i+1}/{nb_folds}, répétition {repetition+1}")
        X_train, X_test = X_skf[train], X_skf[test]
        Y_train, Y_test = Y_skf[train], Y_skf[test]

        # vérfie si une essence d'arbre est absent dans le jeu de test
        if not np.array_equal(np.unique(Y_test), np.array(essence_tree)):
            logging.warning(
                f"Classes manquantes dans Y_test: {set(essence_tree) - set(np.unique(Y_test))}")
            continue  # Passer à la prochaine itération

        # 3 --- Train
        Y_train = np.ravel(Y_train)
        clf.fit(X_train, Y_train)

        # 4 --- Test
        Y_predict = clf.predict(X_test)

        list_cm.append(confusion_matrix(Y_test, Y_predict, labels=essence_tree))
        list_accuracy.append(accuracy_score(Y_test, Y_predict))
        report = classification_report(Y_test, Y_predict, labels=essence_tree, 
                                        output_dict=True, zero_division=0)

        # store them
        list_report.append(report_from_dict_to_df(report))

        # Temp du fold
        fold_end_time = time.time()
        fold_duration = fold_end_time - start_fold
        fold_minutes, fold_seconds = divmod(fold_duration, 60)
        logging.info(f"Fold {fold_i+1}/{nb_folds} terminé en "
                     f"{int(fold_minutes)} min et {fold_seconds:.2f} s.")

    # Temp de la répétition
    rep_end_time = time.time()
    rep_duration = rep_end_time - start_rep
    rep_minutes, rep_seconds = divmod(rep_duration, 60)
    logging.info(f"*** Répétition {repetition+1} terminée en "
                 f"{int(rep_minutes)} min et {rep_seconds:.2f} s. ***")

# ---------------------------------------------------------------------------
# 7) Calcul et figures
# ---------------------------------------------------------------------------
arr_acc = np.array(list_accuracy)  # Liste des accuracies de chaque fold
mean_accuracy = arr_acc.mean()  # Moyenne des accuracies
std_accuracy = arr_acc.std()  # Ecart type des accuracies
logging.info(f"Exactitude moyenne sur {nb_iter*nb_folds} folds : "
             f"{mean_accuracy:.3f} ± {std_accuracy:.3f}")

# Rapports de classification
stacked_rep = np.stack([df.to_numpy() for df in list_report], axis=0)
mean_report = stacked_rep.mean(axis=0)  # Moyenne des rapports
std_report = stacked_rep.std(axis=0)  # Ecart type des rapports

# Conversion des résultats en DataFrames pour sauvegarde
df_template = list_report[0]
mean_df_report = pd.DataFrame(
    mean_report, index=df_template.index, columns=df_template.columns)
std_df_report = pd.DataFrame(
    std_report, index=df_template.index, columns=df_template.columns)

# Sauvegarde des rapports moyens et écarts types en CSV
mean_df_report.to_csv(out_mean_csv)
std_df_report.to_csv(out_std_csv)
logging.info(f"Rapports moyens enregistrés : {out_mean_csv}, {out_std_csv}")

# Distribution des accuracies
try:
    plt.figure(figsize=(7, 5))
    plt.hist(arr_acc, bins=15, edgecolor='k', alpha=0.7)
    plt.title("Distribution de l'accuracy (répétée)")
    plt.xlabel("Accuracy")
    plt.ylabel("Fréquence")
    plt.axvline(mean_accuracy, color='red', linestyle='--',
                label=f"Moyenne={mean_accuracy:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fig_acc, dpi=300)
    plt.close()
    logging.info(f"Distribution des accuracies sauvegardée : {out_fig_acc}")
except Exception as e:
    logging.warning(f"Impossible de tracer la distribution d'accuracy : {e}")

try:
    plot_mean_class_quality(list_report, list_accuracy, out_std_dev_mean)
    logging.info(
        f"Figure plot_mean_class_quality sauvegardée : {out_std_dev_mean}")
except Exception as e:
    logging.warning(f"Impossible de tracer la courbe de performance : {e}")

finally:
    # Nettoyage des fichiers temporaires
    cleanup_temp_files(os.path.join(out_classif_folder, "tmp_shp_for_ID"))
    print("Fichiers temporaires supprimés.")

# ---------------------------------------------------------------------------
# 7-bis) Sauvegarde d'une matrice de confusion moyenne
# ---------------------------------------------------------------------------
# Calcul de la matrice de confusion moyenne
# shape = (nb_folds_total, n_classes, n_classes)
arr_cm = np.stack(list_cm, axis=0)
mean_cm = arr_cm.mean(axis=0)       # moyenne sur l'axe des folds
mean_cm = mean_cm.astype(int)

out_cm_pixel = os.path.join(
    out_classif_folder, "confusion_matrix_pixel_mean.png")

# Utilisation de la fonction plot_cm pour tracer/sauvegarder
plot_cm(
    mean_cm,
    labels=essence_tree,
    out_filename=out_cm_pixel,
    normalize=False,   # ou True pour la normaliser en pourcentage
    cmap="Blues"
)
logging.info(f"Matrice de confusion moyenne sauvegardée : {out_cm_pixel}")

# ---------------------------------------------------------------------------
# 8) Apprentissage final sur X_f + prediction sur X_orig
# ---------------------------------------------------------------------------
logging.info("Prédiction sur tout le raster (X_orig) pour la carte finale.")
Y_predict = clf.predict(X)

# reshape
logging.info("Sauvegarde de la nouvelle Carte")
ds = rw.open_image(image_filename)
nb_row, nb_col, _ = rw.get_image_dimension(ds)

img = np.zeros((nb_row, nb_col, 1), dtype='uint8')
img[t[0], t[1], 0] = Y_predict

# write image
rw.write_image(out_classif, img, data_set=ds, nb_band=1)

ds = gdal.Open(out_classif, 1) # The 1 means that you are opening the file to edit it)
rb = ds.GetRasterBand(1) #assuming your raster has 1 band. 
rb.SetNoDataValue(0)
rb = None
ds = None

elapsed = (time.time() - start_time) / 60
logging.info(
    f"=== Fin de la classification pixel. Durée : {elapsed:.2f} minutes. ===")
print(f"Traitement terminé avec succès en {elapsed:.2f} minutes.")
