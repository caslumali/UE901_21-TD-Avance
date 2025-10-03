#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour l'analyse de la variabilité spectrale de la BD forêt.
"""

from my_function import (extraire_valeurs_ndvi_par_classe,
                         extraire_valeurs_ndvi_par_polygone,
                         calculer_centroide_et_distances,
                         plot_barchart_distance_classes,
                         plot_violin_distance_polygons,
                         cleanup_temp_files,
                         log_error_and_raise)
import os
import time

# Définition des chemins d'entrée et de sortie
shapefile_path = "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp"
raster_path = "groupe_9/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"
output_folder = "groupe_9/results/figure/"
groupes = {
    "Autres feuillus": "Pur",
    "Chêne": "Pur",
    "Robinier": "Pur",
    "Peupleraie": "Pur",
    "Douglas": "Pur",
    "Pin laricio ou pin noir": "Pur",
    "Pin maritime": "Pur",
    "Mélange de feuillus": "Mélange",
    "Mélange conifères": "Mélange",
    "Mélange de conifères prépondérants et feuillus": "Mélange",
    "Mélange de feuillus préponderants conifères": "Mélange"
}

# Chemin des fichiers temporaires
temp_class_shp = "temp_classes_dissolues"
temp_poly_shp = "temp_polygones_int_id"

try:
    total_start_time = time.time()
    print("Début du traitement...")

    # Question 1
    print("Extraction des valeurs NDVI par classe...")
    start_time = time.time()
    resultats_par_classe = extraire_valeurs_ndvi_par_classe(
        shapefile_path, raster_path, groupes)
    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(
        f"Extraction des NDVI par classe terminée en {int(minutes)} minutes et {seconds:.2f} secondes.")

    print("Calcul des distances moyennes au centroïde par classe...")
    start_time = time.time()
    distances_par_classe = {}
    for classe, valeurs in resultats_par_classe.items():
        # Passer l'ensemble du dict valeurs, pas juste valeurs["ndvi_means"]
        resultats = calculer_centroide_et_distances({classe: valeurs})
        distances_par_classe[classe] = resultats[classe]["distance_moyenne"]
    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(
        f"Calcul des distances terminé en {int(minutes)} minutes et {seconds:.2f} secondes.")

    print("Génération du diagramme en bâton...")
    start_time = time.time()
    output_barchart = os.path.join(
        output_folder, "diag_baton_dist_centroide_classe.png")
    plot_barchart_distance_classes(
        distances_par_classe, output_barchart, groupes)
    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(
        f"Diagramme en bâton généré en {int(minutes)} minutes et {seconds:.2f} secondes.")

    # Question 2
    print("Extraction des valeurs NDVI par polygone...")
    start_time = time.time()
    resultats_par_polygone = extraire_valeurs_ndvi_par_polygone(
        shapefile_path, raster_path, groupes)
    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(
        f"Extraction des NDVI par polygone terminée en {int(minutes)} minutes et {seconds:.2f} secondes.")

    print("Calcul des distances moyennes au centroïde par polygone...")
    start_time = time.time()
    distances_par_polygone = []
    classes_polygones = []
    for poly_id, result in resultats_par_polygone.items():
        # Passer le dict complet du polygone, et préciser niveau="polygone"
        dist_result = calculer_centroide_et_distances(
            {poly_id: result}, niveau="polygone")

        # Vérifier si on a des valeurs NDVI
        if dist_result[poly_id]["distance_moyenne"] is not None:
            distances_par_polygone.append(
                dist_result[poly_id]["distance_moyenne"])
            classes_polygones.append(result["class"])
        else:
            print(
                f"Pas de NDVI pour le polygone {poly_id}, ce polygone est ignoré.")

    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(
        f"Calcul des distances par polygone terminé en {int(minutes)} minutes et {seconds:.2f} secondes.")

    print("Génération du violin plot...")
    start_time = time.time()
    output_violin_plot = os.path.join(
        output_folder, "violin_plot_dist_centroide_by_poly_by_class.png")
    plot_violin_distance_polygons(
        distances_par_polygone, classes_polygones, output_violin_plot, groupes)
    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(
        f"Violin plot généré en {int(minutes)} minutes et {seconds:.2f} secondes.")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    minutes, seconds = divmod(total_duration, 60)
    print(
        f"Traitement terminé avec succès en {int(minutes)} minutes et {seconds:.2f} secondes.")

except Exception as e:
    log_error_and_raise(f"Une erreur est survenue lors de l'analyse : {e}")

finally:
    # Supprimer les fichiers temporaires
    cleanup_temp_files("temp_classes_dissolues", "temp_polygones_int_id")
    print("Fichiers temporaires supprimés.")
