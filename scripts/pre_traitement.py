#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prétraitement des données Sentinel-2

Ce script effectue les étapes nécessaires pour traiter les bandes Sentinel-2 :
1. Traitement des bandes Sentinel-2 (reprojection, recadrage, rechantillonnage, masquage).
2. Fusion des bandes pour produire une image multitemporelle à 60 bandes.
3. Calcul du NDVI pour chaque date à partir des bandes déjà traitées.
4. Fusion des images NDVI pour produire une série temporelle à 6 bandes.

Créé le 3 décembre 2024
"""
import os
import re  # pour utiliser fonction regex
import time  # Pour calculer le temp de traitement
import logging
from datetime import datetime
from my_function import (find_raster_bands, reproject_raster, clip_raster_to_extent,
                         resample_raster, apply_mask, merge_rasters,
                         calculate_ndvi_from_processed_bands)

# Démarrage du chronomètre de traitement total
total_start_time = time.time()

# Paramètres
clip_vector = "data/project/emprise_etude.shp"
mask_raster = "groupe_9/results/data/img_pretraitees/masque_foret.tif"
input_folder = "data/images/"
band_prefixes = ["FRE_B2", "FRE_B3", "FRE_B4", "FRE_B5",
                 "FRE_B6", "FRE_B7", "FRE_B8", "FRE_B8A", "FRE_B11", "FRE_B12"]
output_path_allbands = "groupe_9/results/data/img_pretraitees/Serie_temp_S2_allbands.tif"
output_path_ndvi = "groupe_9/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"

# Préparer un dictionnaire pour stocker les bandes disponibles pour chaque date
bands_by_date = {}

# Trouver toutes les bandes raster
logging.info("Starting to find raster bands...")
all_band_files = find_raster_bands(input_folder, band_prefixes)
logging.info(f"Found {len(all_band_files)} raster bands.")

# Extraire les dates et organiser les bandes par date et préfixe
for band_file in all_band_files:
    # Extraire la date et le préfixe de la bande à partir du nom de fichier
    basename = os.path.basename(band_file)
    parts = basename.split('_')
    if len(parts) >= 2:
        date_part = parts[1]
        date_str = date_part[:8]  # Obtenir YYYYMMDD
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            date_label = date_obj.strftime('%Y%m%d')
            month_num = date_obj.strftime('%m')
            year_num = date_obj.strftime('%Y')
        except ValueError:
            continue  # Ignorer si la date n'est pas valide
    else:
        continue  # Ignorer si le nom de fichier n'a pas le format correct

    # Identifier le préfixe en utilisant regex
    band_prefix = None
    for prefix in band_prefixes:
        # Créer un modèle regex pour correspondre exactement au préfixe de la bande
        pattern = rf"{prefix}(?!\w)"
        if re.search(pattern, basename):
            band_prefix = prefix
            break
    if not band_prefix:
        continue  # Ignorer si le préfixe de bande n'est pas trouvé

    # Stocker le fichier de bande dans le dictionnaire
    if date_label not in bands_by_date:
        bands_by_date[date_label] = {
            'date_obj': date_obj, 'month_num': month_num, 'year_num': year_num, 'bands': {}}
    bands_by_date[date_label]['bands'][band_prefix] = band_file

# Trier les dates
sorted_dates = sorted(bands_by_date.keys())

# Démarrage du chronomètre pour le traitement du raster multibande
multiband_start_time = time.time()
logging.info("Starting processing of multiband raster (60 bands)...")

# Traiter les bandes et créer la liste des bandes traitées et des noms de bandes
processed_bands = []
band_names = []

for date_label in sorted_dates:
    date_info = bands_by_date[date_label]
    month_num = date_info['month_num']
    year_num = date_info['year_num']
    bands = date_info['bands']

    for band_prefix in band_prefixes:
        if band_prefix in bands:
            band_file = bands[band_prefix]

            # Démarrer la boucle de traitement pour chaque bande
            reprojected = reproject_raster(band_file, "EPSG:2154")
            clipped = clip_raster_to_extent(
                reprojected, clip_vector, nodata_value=0)
            resampled = resample_raster(clipped, 10)
            masked = apply_mask(resampled, mask_raster, nodata_value=0)
            processed_bands.append(masked)

            # Construire le nom de la bande: BXX_XX_XXXX
            band_number = band_prefix.split('_')[1]  # e.g., 'B2'
            band_name = f"{band_number}_{month_num}_{year_num}"
            band_names.append(band_name)
        else:
            logging.warning(
                f"Band {band_prefix} not found for date {date_label}")

# Fusionner toutes les bandes dans le raster à 60 bandes
merge_rasters(processed_bands, output_path_allbands,
              band_names=band_names, pixel_type="UInt16", nodata_value=0)

# Fin du chronomètre pour le traitement des rasters multi-bandes
multiband_end_time = time.time()
multiband_duration = multiband_end_time - multiband_start_time
minutes, seconds = divmod(multiband_duration, 60)
logging.info(
    f"Processing of multiband raster completed in {int(minutes)} minutes and {seconds:.2f} seconds."
)

# Démarrage du chronomètre pour le traitement NDVI
ndvi_start_time = time.time()
logging.info("Starting NDVI calculation...")

# Calculer le NDVI pour chaque mois
ndvi_rasters = []
ndvi_band_names = []

for date_label in sorted_dates:
    date_info = bands_by_date[date_label]
    month_num = date_info['month_num']
    year_num = date_info['year_num']
    bands = date_info['bands']

    if "FRE_B4" in bands and "FRE_B8" in bands:
        red_band_file = bands["FRE_B4"]
        nir_band_file = bands["FRE_B8"]

        # Démarrer la boucle de traitement pour la bande rouge
        red_reprojected = reproject_raster(red_band_file, "EPSG:2154")
        red_clipped = clip_raster_to_extent(
            red_reprojected, clip_vector, nodata_value=0)
        red_resampled = resample_raster(red_clipped, 10)
        red_masked = apply_mask(red_resampled, mask_raster, nodata_value=0)

        # Démarrer la boucle de traitement pour la bande NIR
        nir_reprojected = reproject_raster(nir_band_file, "EPSG:2154")
        nir_clipped = clip_raster_to_extent(
            nir_reprojected, clip_vector, nodata_value=0)
        nir_resampled = resample_raster(nir_clipped, 10)
        nir_masked = apply_mask(nir_resampled, mask_raster, nodata_value=0)

        # Calculater le NDVI
        ndvi = calculate_ndvi_from_processed_bands(
            red_masked, nir_masked, nodata_value=-9999)
        ndvi_rasters.append(ndvi)

        # Construire le nom de la bande: NDVI_XX_XXXX
        ndvi_band_name = f"NDVI_{month_num}_{year_num}"
        ndvi_band_names.append(ndvi_band_name)
    else:
        logging.warning(
            f"Missing bands for NDVI calculation on date {date_label}")

# Merge NDVI rasters into the 6-band raster
merge_rasters(ndvi_rasters, output_path_ndvi, band_names=ndvi_band_names,
              pixel_type="Float32", nodata_value=-9999)

logging.info(f"NDVI raster saved to: {output_path_ndvi}")

# Fin du chronomètre pour le traitement NDVI
ndvi_end_time = time.time()
ndvi_duration = ndvi_end_time - ndvi_start_time
minutes, seconds = divmod(ndvi_duration, 60)
logging.info(
    f"NDVI calculation completed in {int(minutes)} minutes and {seconds:.2f} seconds."
)

# Fin du chronomètre pour tout le traitement
total_end_time = time.time()
total_duration = total_end_time - total_start_time
minutes, seconds = divmod(total_duration, 60)
logging.info(
    f"Total processing time: {int(minutes)} minutes and {seconds:.2f} seconds."
)

print(
    f"Traitement terminé avec succès en {int(minutes)} minutes et {seconds:.2f} secondes")
