#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for Geospatial Data Processing

This module provides functions to filter and reclassify geospatial data,
manage vector files, and process rasters with GDAL, all with integrated logger.

Created on Dec 03, 2024
Last modified: Dec 11, 2024

@author: Alban Dumont, Lucas Lima, Robin Heckendorn
"""
from affine import Affine
from rasterstats import zonal_stats
import os
import logging
import sys
import traceback
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

# Configuration GDAL
gdal.UseExceptions()

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,  # Définit le niveau de journalisation
    # Format du message de log
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # Affiche les logs dans la console (stdout)
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("log.txt")  # Enregistre les logs dans un fichier
    ]
)

# Création de la variable logger
logger = logging.getLogger(__name__)


# ================================= #
# ===  FONCTIONS UTILITAIRES  ===== #
# ================================= #

def log_error_and_raise(message, exception=RuntimeError):
    """
    Enregistre un message d'erreur, inclut la trace d'exécution, et déclenche une exception.

    Paramètres :
    -----------
    message : str
        Message d'erreur à enregistrer et à lever.
    exception : Exception
        Type d'exception à lever (par défaut : RuntimeError).
    """
    logger.error(f"{message}\n{traceback.format_exc()}")
    raise exception(message)


def cleanup_temp_files(*base_file_paths):
    """
    Supprime les fichiers temporaires spécifiés avec toutes leurs extensions associées.

    Paramètres :
    -----------
    *base_file_paths : str
        Chemins de base des fichiers à supprimer (sans extension spécifique).
        Ex : "temp_classes_dissolues" supprimera tous les fichiers "temp_classes_dissolues.*".
    """
    extensions = [
        ".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".fix", ".shp.xml"
    ]  # Extensions associées aux shapefiles et fichiers temporaires

    for base_path in base_file_paths:
        if not base_path or not isinstance(base_path, str):
            logging.warning(f"Chemin de base invalide : {base_path}")
            continue

        for ext in extensions:
            file_path = f"{base_path}{ext}"
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Fichier temporaire supprimé : {file_path}")
                else:
                    logging.debug(
                        f"Fichier non trouvé (aucune action nécessaire) : {file_path}")
            except Exception as e:
                logging.error(
                    f"Erreur lors de la suppression du fichier {file_path} : {e}"
                )


# ================================== #
# === FONCTIONS POUR LE VECTEUR ==== #
# ================================== #
"""
Fonctions pour le traitement vectoriel de la BD Forêt

Ces fonctions permettent de filtrer et reclasser les données vectorielles, 
découper un GeoDataFrame selon une emprise donnée, 
et enregistrer les résultats dans un fichier pour une analyse ultérieure.
"""


def reclassification_dictionary():
    """
    Retourne le dictionnaire de reclassement pour BD Forêt.

    Retourne :
    ---------
    dict
        Un dictionnaire associant le champ `TFV` à un tuple contenant :
        - `Code` (int) : Le code de classification.
        - `Nom` (str) : Le nom de la classification.
    """

    return {
        "Forêt fermée d’un autre feuillu pur": (11, "Autres feuillus"),
        "Forêt fermée de châtaignier pur": (11, "Autres feuillus"),
        "Forêt fermée de hêtre pur": (11, "Autres feuillus"),

        "Forêt fermée de chênes décidus purs": (12, "Chêne"),
        "Forêt fermée de robinier pur": (13, "Robinier"),
        "Peupleraie": (14, "Peupleraie"),
        "Forêt fermée à mélange de feuillus": (15, "Mélange de feuillus"),
        "Forêt fermée de feuillus purs en îlots": (16, "Feuillus en îlots"),

        "Forêt fermée d’un autre conifère pur autre que pin ": (21, "Autres conifères autre que pin"),
        "Forêt fermée de mélèze pur": (21, "Autres conifères autre que pin"),
        "Forêt fermée de sapin ou épicéa": (21, "Autres conifères autre que pin"),
        "Forêt fermée à mélange d’autres conifères": (21, "Autres conifères autre que pin"),

        "Forêt fermée d’un autre pin pur": (22, "Autres pin"),
        "Forêt fermée de pin sylvestre pur": (22, "Autres pin"),
        "Forêt fermée à mélange de pins purs": (22, "Autres pin"),

        "Forêt fermée de douglas pur": (23, "Douglas"),
        "Forêt fermée de pin laricio ou pin noir pur": (24, "Pin laricio ou pin noir"),
        "Forêt fermée de pin maritime pur": (25, "Pin maritime"),
        "Forêt fermée à mélange de conifères": (26, "Mélange conifères"),
        "Forêt fermée de conifères purs en îlots": (27, "Conifères en îlots"),

        "Forêt fermée à mélange de conifères prépondérants et feuillus": (28, "Mélange de conifères prépondérants et feuillus"),
        "Forêt fermée à mélange de feuillus prépondérants et conifères": (29, "Mélange de feuillus préponderants conifères")

    }


def filter_and_reclassify(gdf):
    """
    Filtre et reclasse un GeoDataFrame en fonction du système BD Forêt.

    Paramètres :
    -----------
    gdf : GeoDataFrame
        GeoDataFrame en entrée contenant les données BD Forêt.

    Retourne :
    ---------
    GeoDataFrame
        Un GeoDataFrame filtré et reclassé avec les attributs existants, ainsi que `Nom` et `Code`.
    """
    try:
        logger.info("Début du filtrage et du reclassement du GeoDataFrame.")

        # Obtenir le dictionnaire de reclassement
        reclassification = reclassification_dictionary()

        # Filtrer en fonction des catégories dans le dictionnaire
        categories_to_keep = list(reclassification.keys())
        filtered_gdf = gdf[gdf["TFV"].isin(categories_to_keep)].copy()

        if filtered_gdf.empty:
            log_error_and_raise(
                "Aucune entité trouvée après filtrage basé sur les catégories TFV.")

        logger.info(
            f"GeoDataFrame filtré avec {len(filtered_gdf)} entités basées sur les catégories TFV.")

        # Ajouter les attributs `Nom` et `Code` en fonction du reclassement
        filtered_gdf["Nom"] = filtered_gdf["TFV"].map(
            lambda x: reclassification[x][1])
        filtered_gdf["Code"] = filtered_gdf["TFV"].map(
            lambda x: reclassification[x][0])

        logger.info("Ajout des attributs 'Nom' et 'Code' au GeoDataFrame.")

        return filtered_gdf

    except Exception as e:
        log_error_and_raise(
            f"Erreur lors du filtrage et du reclassement : {e}")


def clip_vector_to_extent(gdf, clip_shapefile):
    """
    Découpe un GeoDataFrame à l'étendue d'un shapefile sans ajouter les attributs du shapefile de découpage.

    Paramètres :
    -----------
    gdf : GeoDataFrame
        GeoDataFrame d'entrée à découper.
    clip_shapefile : str
        Chemin vers le shapefile définissant l'étendue de découpage.

    Retourne :
    ---------
    GeoDataFrame
        GeoDataFrame découpé contenant uniquement les attributs du GeoDataFrame d'entrée.

    Exceptions :
    -----------
    RuntimeError
        Si le processus de découpage échoue.
    """
    try:
        logger.info(
            f"Découpage du GeoDataFrame en utilisant le shapefile : {clip_shapefile}")

        # Lire le shapefile de découpage
        clip_gdf = gpd.read_file(clip_shapefile)
        if clip_gdf.empty:
            log_error_and_raise(
                f"Le shapefile de découpage {clip_shapefile} est vide.")

        # Vérifier et harmoniser le CRS
        if gdf.crs != clip_gdf.crs:
            logger.info(
                "Reprojection du GeoDataFrame pour correspondre au CRS du shapefile de découpage.")
            gdf = gdf.to_crs(clip_gdf.crs)

        # Découper en utilisant geopandas.clip
        clipped_gdf = gpd.clip(gdf, clip_gdf)

        if clipped_gdf.empty:
            log_error_and_raise(
                "Aucune entité trouvée après l'opération de découpage.")

        logger.info(
            f"Le GeoDataFrame découpé contient {len(clipped_gdf)} entités.")

        return clipped_gdf

    except Exception as e:
        log_error_and_raise(f"Erreur lors du découpage du GeoDataFrame : {e}")


def save_vector_file(gdf, output_path):
    """
    Enregistre un GeoDataFrame dans un fichier vecteur (par exemple, Shapefile, GeoJSON).

    Paramètres :
    -----------
    gdf : GeoDataFrame
        GeoDataFrame d'entrée à sauvegarder.
    output_path : str
        Chemin pour sauvegarder le fichier de sortie, y compris le nom du fichier et son extension.

    Retourne :
    ---------
    None
    """
    try:
        logger.info(
            f"Enregistrement du GeoDataFrame dans le fichier : {output_path}")
        gdf.to_file(output_path, driver="ESRI Shapefile")
        logger.info("GeoDataFrame enregistré avec succès.")
    except Exception as e:
        log_error_and_raise(
            f"Erreur lors de l'enregistrement du GeoDataFrame dans le fichier : {e}")


# ========================================= #
# === FONCTIONS POUR LE PRÉ-TRAITEMENT  === #
# ========================================= #
"""
Fonctions pour le Pré-Traitement des Rasters

Ces fonctions permettent de récupérer les propriétés des rasters, reprojeter, 
rasteriser des masques, découper et rééchantillonner les données, appliquer des masques,
calculer le NDVI, fusionner des rasters en multibandes et gérer les fichiers raster.
"""


def find_raster_bands(folder_path, band_prefixes):
    """
    Trouve et filtre les bandes raster en fonction de préfixes.

    Paramètres :
    -----------
    folder_path : str
        Chemin du dossier contenant les fichiers raster.
    band_prefixes : list
        Liste des préfixes de bandes à filtrer (ex : ["FRE_B2", "FRE_B3", "FRE_B4", "FRE_B8"]).

    Retourne :
    ---------
    list
        Liste des chemins des fichiers raster filtrés se terminant par '.tif' et correspondant aux préfixes spécifiés.
    """
    raster_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Vérifie si le fichier se termine par '.tif'
            if file.endswith('.tif'):
                # Vérifie si le fichier contient l'un des préfixes spécifiés
                if any(prefix in file for prefix in band_prefixes):
                    raster_files.append(os.path.join(root, file))
    return raster_files


def get_raster_properties(dataset):
    """
    Récupère les propriétés d'un raster à partir d'un dataset GDAL.

    Paramètres :
    -----------
    dataset : gdal.Dataset
        Dataset GDAL en entrée.

    Retourne :
    ---------
    tuple
        Propriétés du raster : (largeur_pixel, hauteur_pixel, xmin, ymin, xmax, ymax, crs).

    Exceptions :
    -----------
    ValueError
        Si le dataset est None ou si la résolution ne correspond pas à 10m attendus.
    """
    if dataset is None:
        raise ValueError(
            "Le dataset d'entrée est None. Veuillez vérifier le raster en entrée.")

    geotransform = dataset.GetGeoTransform()
    pixel_width = geotransform[1]
    pixel_height = abs(geotransform[5])  # Assure une hauteur de pixel positive
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = xmin + (dataset.RasterXSize * pixel_width)
    # Ajustement pour une hauteur négative
    ymin = ymax - (dataset.RasterYSize * pixel_height)
    crs = dataset.GetProjection()

    # Vérification si la résolution est environ 10m
    if not (abs(pixel_width - 10) < 1e-6 and abs(pixel_height - 10) < 1e-6):
        raise ValueError(
            f"La résolution du raster ne correspond pas à 10m : ({pixel_width}, {pixel_height})")

    return pixel_width, pixel_height, xmin, ymin, xmax, ymax, crs


def reproject_raster(input_raster, target_srs, x_res=10, y_res=10, resample_alg="near"):
    """
    Reprojette un raster vers un système de coordonnées cible en mémoire avec une résolution spécifiée.

    Paramètres :
    -----------
    input_raster : str
        Chemin vers le raster en entrée.
    target_srs : str
        Système de référence spatial cible (ex : 'EPSG:2154').
    x_res : float, optionnel
        Largeur de pixel souhaitée (par défaut : 10).
    y_res : float, optionnel
        Hauteur de pixel souhaitée (par défaut : 10).
    resample_alg : str, optionnel
        Algorithme de rééchantillonnage à utiliser (par défaut : "bilinear").

    Retourne :
    ---------
    gdal.Dataset
        Raster reprojeté en mémoire.

    Exceptions :
    -----------
    RuntimeError
        Si la reprojection échoue.
    """
    try:
        logger.info(
            f"Reprojection du raster : {input_raster} vers {target_srs} avec une résolution de {x_res}x{y_res}")
        reprojected = gdal.Warp(
            '', input_raster, dstSRS=target_srs, format="MEM",
            xRes=x_res, yRes=y_res, resampleAlg=resample_alg
        )
        if reprojected is None:
            log_error_and_raise(
                f"Échec de la reprojection du raster : {input_raster}")
        return reprojected
    except Exception as e:
        log_error_and_raise(f"Erreur pendant la reprojection : {e}")


def create_forest_mask(mask_vector, reference_raster, clip_vector, output_path):
    """
    Crée un raster masque pour les forêts aligné avec un raster de référence, avec :
        - 1 pour les zones forestières
        - 0 pour les zones non-forestières
        - 99 pour les zones sans données (NoData).

    Paramètres :
    -----------
    mask_vector : str
        Chemin vers le fichier vecteur représentant les zones forestières.
    reference_raster : str
        Chemin vers le raster de référence.
    clip_vector : str
        Chemin vers le shapefile définissant l'étendue de la zone d'étude.
    output_path : str
        Chemin pour sauvegarder le raster masque final.

    Exceptions :
    -----------
    RuntimeError
        Si une étape du processus échoue.
    """
    try:
        logger.info("Création du masque forestier...")

        # Étape 1 : Reprojection du raster de référence pour obtenir les propriétés souhaitées
        logger.info("Reprojection du raster de référence vers EPSG:2154...")
        ref_raster_reprojected = reproject_raster(
            reference_raster, "EPSG:2154")
        pixel_width, pixel_height, xmin, ymin, xmax, ymax, crs = get_raster_properties(
            ref_raster_reprojected)

        # Libérer le raster reprojeté après extraction des propriétés
        ref_raster_reprojected = None

        # Alignement des bornes pour correspondre à la résolution
        xmin_aligned = xmin - (xmin % pixel_width)
        ymin_aligned = ymin - (ymin % pixel_height)
        xmax_aligned = xmax + (pixel_width - (xmax % pixel_width))
        ymax_aligned = ymax + (pixel_height - (ymax % pixel_height))

        # Calcul du nombre de pixels en directions x et y
        x_pixels = int((xmax_aligned - xmin_aligned) / pixel_width)
        y_pixels = int((ymax_aligned - ymin_aligned) / pixel_height)

        # Étape 2 : Création d'un raster en mémoire pour rasteriser le masque vecteur
        logger.info("Rasterisation du masque forestier...")
        mem_driver = gdal.GetDriverByName('MEM')
        out_raster = mem_driver.Create(
            '', x_pixels, y_pixels, 1, gdal.GDT_Byte)
        out_raster.SetGeoTransform(
            (xmin_aligned, pixel_width, 0, ymax_aligned, 0, -pixel_height))
        out_raster.SetProjection(crs)

        # Initialiser le raster avec la valeur 0 (non-forêt)
        # Les bandes sont indexées à partir de 1
        out_band = out_raster.GetRasterBand(1)
        out_band.Fill(0)
        out_band.SetNoDataValue(99)

        # Ouvrir le masque vecteur
        vector_ds = gdal.OpenEx(mask_vector, gdal.OF_VECTOR)
        if vector_ds is None:
            log_error_and_raise(
                f"Impossible d'ouvrir le fichier vecteur : {mask_vector}")

        # Rasteriser le masque vecteur avec la valeur 1 (forêt)
        err = gdal.RasterizeLayer(
            out_raster, [1], vector_ds.GetLayer(), burn_values=[1])
        if err != 0:
            log_error_and_raise("Échec de la rasterisation.")

        # Fermer le fichier vecteur
        vector_ds = None

        # Étape 3 : Découpage du rasterisé avec la zone d'étude
        logger.info("Découpage du masque forestier avec la zone d'étude...")
        clipped_mask = clip_raster_to_extent(
            out_raster, clip_vector, nodata_value=99)

        if clipped_mask is None:
            log_error_and_raise("Échec du découpage du masque forestier.")

        # Étape 4 : Sauvegarde du masque découpé
        # Supprimer le fichier existant avant sauvegarde
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"Fichier existant supprimé : {output_path}")
            except PermissionError:
                log_error_and_raise(
                    f"Impossible de supprimer le fichier existant : {output_path}")

        driver = gdal.GetDriverByName('GTiff')
        output_ds = driver.CreateCopy(
            output_path, clipped_mask, options=["COMPRESS=LZW"])
        if output_ds is None:
            log_error_and_raise(
                f"Échec de la sauvegarde du masque forestier : {output_path}")

        # Finaliser et fermer les datasets
        output_ds.FlushCache()
        output_ds = None
        out_raster = None
        clipped_mask = None

        logger.info(f"Masque forestier sauvegardé : {output_path}")

    except Exception as e:
        log_error_and_raise(
            f"Erreur pendant la création du masque forestier : {e}")


def clip_raster_to_extent(input_raster, clip_vector, nodata_value):
    """
    Découpe un raster selon l'étendue d'un shapefile.

    Paramètres :
    -----------
    input_raster : gdal.Dataset ou str
        Raster en mémoire ou chemin vers le fichier raster en entrée.
    clip_vector : str
        Chemin vers le shapefile utilisé pour le découpage.
    nodata_value : int ou float
        Valeur NoData à attribuer dans le raster de sortie.

    Retourne :
    ---------
    gdal.Dataset
        Raster découpé en mémoire.

    Exceptions :
    -----------
    RuntimeError
        Si le processus de découpage échoue.
    """
    try:
        logger.info(f"Découpage du raster avec le shapefile : {clip_vector}")
        if not os.path.exists(clip_vector):
            log_error_and_raise(
                f"Shapefile introuvable : {clip_vector}", FileNotFoundError)

        # Découpage avec GDAL.Warp
        clipped = gdal.Warp(
            '', input_raster, format="MEM",
            cutlineDSName=clip_vector, cropToCutline=True,
            dstNodata=nodata_value
        )
        if clipped is None:
            log_error_and_raise(
                f"Échec du découpage du raster avec le shapefile : {clip_vector}")

        # Libérer la mémoire si le raster en entrée est en mémoire
        if isinstance(input_raster, gdal.Dataset):
            input_raster = None

        return clipped

    except Exception as e:
        log_error_and_raise(f"Erreur lors du découpage : {e}")


def resample_raster(input_raster, pixel_size):
    """
    Rééchantillonne un raster à une résolution spécifique.

    Paramètres :
    -----------
    input_raster : gdal.Dataset
        Raster en mémoire.
    pixel_size : float
        Taille de pixel souhaitée.

    Retourne :
    ---------
    gdal.Dataset
        Raster rééchantillonné en mémoire.

    Exceptions :
    -----------
    RuntimeError
        Si le processus de rééchantillonnage échoue.
    """
    try:
        logger.info(
            f"Rééchantillonnage du raster à une taille de pixel : {pixel_size}")
        resampled = gdal.Warp('', input_raster, format="MEM",
                              xRes=pixel_size, yRes=pixel_size, resampleAlg="near")
        if resampled is None:
            log_error_and_raise("Échec du rééchantillonnage du raster.")
        return resampled
    except Exception as e:
        log_error_and_raise(f"Erreur lors du rééchantillonnage : {e}")


def apply_mask(input_raster, mask_raster_path, nodata_value):
    """
    Applique un masque raster à un raster, en définissant les zones non forestières à la valeur NoData.

    Paramètres :
    -----------
    input_raster : gdal.Dataset
        Raster d'entrée en mémoire.
    mask_raster_path : str
        Chemin vers le fichier raster du masque.
    nodata_value : int ou float
        Valeur pour les zones NoData (non-forest et données invalides).

    Retourne :
    ---------
    gdal.Dataset
        Raster masqué en mémoire.

    Exceptions :
    -----------
    RuntimeError
        Si le processus d'application du masque échoue.
    """
    try:
        logger.info("Application du masque au raster...")

        # Ouvrir le raster du masque
        mask_ds = gdal.Open(mask_raster_path)
        if mask_ds is None:
            log_error_and_raise(
                f"Impossible d'ouvrir le raster du masque : {mask_raster_path}")

        # Vérifier que les dimensions du raster d'entrée et du masque correspondent
        if (input_raster.RasterXSize != mask_ds.RasterXSize) or (input_raster.RasterYSize != mask_ds.RasterYSize):
            log_error_and_raise(
                "Le raster d'entrée et le raster du masque ont des dimensions différentes.")

        # Lire les données des rasters d'entrée et du masque sous forme de tableaux
        input_band = input_raster.GetRasterBand(1)
        input_data = input_band.ReadAsArray()

        mask_band = mask_ds.GetRasterBand(1)
        mask_data = mask_band.ReadAsArray()

        # Appliquer le masque : définir les zones non forestières à la valeur NoData
        input_data = np.where(mask_data == 1, input_data, nodata_value)

        # Créer un nouveau raster en mémoire pour contenir les données masquées
        driver = gdal.GetDriverByName('MEM')
        out_ds = driver.Create('', input_raster.RasterXSize,
                               input_raster.RasterYSize, 1, input_band.DataType)
        out_ds.SetGeoTransform(input_raster.GetGeoTransform())
        out_ds.SetProjection(input_raster.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(input_data)
        out_band.SetNoDataValue(nodata_value)

        # Libérer les ensembles de données
        mask_ds = None
        input_raster = None

        return out_ds

    except Exception as e:
        log_error_and_raise(f"Erreur lors de l'application du masque : {e}")


def calculate_ndvi_from_processed_bands(red_raster, nir_raster, nodata_value=-9999):
    """
    Calcule le NDVI à partir des rasters des bandes rouge et NIR déjà traitées.

    Paramètres :
    -----------
    red_raster : gdal.Dataset
        Ensemble de données GDAL pour la bande rouge traitée.
    nir_raster : gdal.Dataset
        Ensemble de données GDAL pour la bande NIR traitée.
    nodata_value : float, optionnel
        Valeur NoData à appliquer dans le raster de sortie (par défaut : -9999).

    Retourne :
    ---------
    gdal.Dataset
        Raster en mémoire contenant les valeurs NDVI.

    Exceptions :
    -----------
    RuntimeError
        Si une étape du processus échoue.
    """
    try:
        logger.info("Calcul du NDVI à partir des bandes traitées...")

        # Vérifier que les rasters ont les mêmes dimensions
        if (red_raster.RasterXSize != nir_raster.RasterXSize) or (red_raster.RasterYSize != nir_raster.RasterYSize):
            log_error_and_raise(
                "Les rasters rouge et NIR n'ont pas les mêmes dimensions.")

        # Lire les données sous forme de tableaux
        red_band = red_raster.GetRasterBand(1)
        nir_band = nir_raster.GetRasterBand(1)
        red_data = red_band.ReadAsArray().astype('float32')
        nir_data = nir_band.ReadAsArray().astype('float32')

        # Obtenir les valeurs NoData
        red_nodata = red_band.GetNoDataValue()
        nir_nodata = nir_band.GetNoDataValue()

        # Créer un masque pour les valeurs NoData
        mask = (red_data == red_nodata) | (nir_data == nir_nodata)

        # Supprimer les avertissements liés à la division
        np.seterr(divide='ignore', invalid='ignore')

        # Calculer le NDVI
        ndvi = (nir_data - red_data) / (nir_data + red_data)
        ndvi = np.where(mask, nodata_value, ndvi)

        # Créer un raster en mémoire pour stocker les valeurs NDVI
        driver = gdal.GetDriverByName('MEM')
        ndvi_raster = driver.Create(
            '', red_raster.RasterXSize, red_raster.RasterYSize, 1, gdal.GDT_Float32)
        ndvi_raster.SetGeoTransform(red_raster.GetGeoTransform())
        ndvi_raster.SetProjection(red_raster.GetProjection())
        ndvi_band = ndvi_raster.GetRasterBand(1)
        ndvi_band.WriteArray(ndvi)
        ndvi_band.SetNoDataValue(nodata_value)

        # Libérer les ensembles de données
        red_raster = None
        nir_raster = None

        return ndvi_raster

    except Exception as e:
        log_error_and_raise(f"Erreur lors du calcul du NDVI : {e}")


def merge_rasters(raster_list, output_path, band_names, pixel_type, compression="LZW", nodata_value=None):
    """
    Fusionne plusieurs rasters en un raster multibande unique.

    Paramètres :
    -----------
    raster_list : list
        Liste des ensembles de données GDAL à fusionner.
    output_path : str
        Chemin pour sauvegarder le raster multibande final.
    band_names : list, optionnel
        Liste des noms de bandes correspondant aux rasters.
    pixel_type : str
        Type de données du raster de sortie (ex : "UInt16", "Float32").
    compression : str
        Type de compression pour le raster de sortie (par défaut : "LZW").
    nodata_value : int ou float, optionnel
        Valeur NoData à appliquer dans les bandes du raster de sortie.

    Retourne :
    ---------
    None

    Exceptions :
    -----------
    RuntimeError
        Si le processus de fusion échoue.
    """
    try:
        logger.info("Fusion des rasters en un raster multibande unique...")

        # Construction du VRT (Virtual Dataset)
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg="nearest", addAlpha=False, separate=True)
        vrt = gdal.BuildVRT('', raster_list, options=vrt_options)
        if vrt is None:
            log_error_and_raise(
                "Échec de la construction du VRT pour la fusion des rasters.")

        # Définir les descriptions des bandes et les valeurs NoData si fournies
        for i in range(vrt.RasterCount):
            band = vrt.GetRasterBand(i + 1)
            if band_names and i < len(band_names):
                band.SetDescription(band_names[i])
            if nodata_value is not None:
                band.SetNoDataValue(nodata_value)

        # Préparation des options de création
        creation_options = [f"COMPRESS={compression}", "BIGTIFF=YES"]

        # Conversion du VRT en GeoTIFF
        gdal.Translate(output_path, vrt, format="GTiff",
                       creationOptions=creation_options,
                       outputType=getattr(gdal, f"GDT_{pixel_type}"))

        # Libération des ensembles de données
        vrt = None
        for ds in raster_list:
            ds = None

        logger.info(f"Raster multibande sauvegardé : {output_path}")

    except Exception as e:
        log_error_and_raise(f"Erreur lors de la fusion des rasters : {e}")


# ======================================== #
# ===    FONCTIONS POUR L'ANALYSE      === #
# ===        DES ÉCHANTILLONS          === #
# ======================================== #

def sample_data_analysis(shapefile_path, raster_path, classes_a_conserver, output_dir):
    """
    Analyse les données à partir d'un shapefile et d'un raster pour générer des graphiques :
    - Diagramme bâton du nombre de polygones par classe.
    - Diagramme bâton du nombre de pixels par classe.
    - Violon plot de la distribution des pixels par classe.

    :param shapefile_path: Chemin vers le fichier shapefile.
    :param raster_path: Chemin vers le fichier raster.
    :param classes_a_conserver: Liste des classes à conserver dans l'analyse.
    :param output_dir: Répertoire où les graphiques seront sauvegardés.
    """
    if not validate_inputs(shapefile_path, raster_path, output_dir):
        return

    gdf = load_and_filter_shapefile(shapefile_path, classes_a_conserver)

    create_bar_plot_polygons_by_class(gdf, output_dir)

    stats = calculate_zonal_stats(gdf, raster_path)
    create_bar_plot_pixels_by_class(stats, output_dir)

    create_violin_plot_pixel_distribution(stats, output_dir)

    logger.info("Analyse terminée. Les graphiques ont été sauvegardés.")

# Validation des entrées


def validate_inputs(shapefile_path, raster_path, output_dir):
    """
    Valide les chemins d'entrée et crée le répertoire de sortie si nécessaire.

    :param shapefile_path: Chemin vers le fichier shapefile.
    :param raster_path: Chemin vers le fichier raster.
    :param output_dir: Répertoire où les graphiques seront sauvegardés.
    :return: True si les entrées sont valides, False sinon.
    """
    if not os.path.exists(shapefile_path):
        logger.error(f"Le fichier shapefile {shapefile_path} n'existe pas.")
        return False

    if not os.path.exists(raster_path):
        logger.error(f"Le fichier raster {raster_path} n'existe pas.")
        return False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return True

# Chargement et filtrage du shapefile


def load_and_filter_shapefile(shapefile_path, classes_a_conserver):
    """
    Charge le shapefile et filtre les données en fonction des classes spécifiées.

    :param shapefile_path: Chemin vers le fichier shapefile.
    :param classes_a_conserver: Liste des classes à conserver.
    :return: GeoDataFrame filtré.
    """
    logger.info("Chargement et filtrage du shapefile...")
    gdf = gpd.read_file(shapefile_path)
    return gdf[gdf['Nom'].isin(classes_a_conserver)]

# Création d'un diagramme bâton pour les polygones


def create_bar_plot_polygons_by_class(gdf, output_dir):
    """
    Crée un diagramme bâton représentant le nombre de polygones par classe.

    :param gdf: GeoDataFrame contenant les données filtrées.
    :param output_dir: Répertoire où le graphique sera sauvegardé.
    """
    logger.info(
        "Création du diagramme bâton du nombre de polygones par classe...")

    # Compter les polygones par classe et trier dans l'ordre décroissant
    polygones_par_classe = gdf['Nom'].value_counts(
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    ax = polygones_par_classe.plot(
        kind='bar', color='skyblue', edgecolor='black')

    # Ajouter les annotations au-dessus des barres
    for i, value in enumerate(polygones_par_classe):
        ax.text(
            i, value + 0.5, str(value), ha='center', va='bottom', fontsize=10, color='black'
        )

    plt.title('Nombre de polygones par classe', fontsize=16)
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Nombre de polygones', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()

    # Enregistrer le graphique
    output_path = os.path.join(output_dir, 'diag_baton_nb_poly_by_class.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Diagramme bâton des polygones sauvegardé avec succès.")


# Calcul des statistiques zonales

def calculate_zonal_stats(gdf, raster_path):
    """
    Calcule les statistiques zonales pour les polygones dans le GeoDataFrame à partir du raster.

    :param gdf: GeoDataFrame contenant les données filtrées.
    :param raster_path: Chemin vers le fichier raster.
    :return: Liste des statistiques zonales.
    """
    logger.info("Calcul des statistiques zonales...")
    return zonal_stats(gdf, raster_path, stats=['sum'], geojson_out=True)

# Création d'un diagramme bâton pour les pixels


def create_bar_plot_pixels_by_class(stats, output_dir):
    """
    Crée un diagramme bâton représentant le nombre total de pixels par classe.

    :param stats: Liste des statistiques zonales calculées.
    :param output_dir: Répertoire où le graphique sera sauvegardé.
    """
    logger.info("Création du diagramme bâton du nombre de pixels par classe...")

    pixel_counts = {}
    for feature in stats:
        classe = feature['properties']['Nom']
        pixel_counts[classe] = pixel_counts.get(
            classe, 0) + feature['properties']['sum']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        pixel_counts.keys(),
        pixel_counts.values(),
        color='lightgreen',
        edgecolor='black'
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(pixel_counts.values()) * 0.02,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.ylim(0, max(pixel_counts.values()) * 1.1)
    plt.subplots_adjust(top=0.9)
    plt.title('Nombre de pixels par classe', fontsize=16)
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Nombre de pixels', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'diag_baton_nb_pix_by_class.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Diagramme bâton des pixels sauvegardé avec succès.")

# Création d'un violon plot pour la distribution des pixels


def create_violin_plot_pixel_distribution(stats, output_dir):
    """
    Crée un violon plot représentant la distribution des pixels par classe de polygone.

    :param stats: Liste des statistiques zonales calculées.
    :param output_dir: Répertoire où le graphique sera sauvegardé.
    """
    logger.info(
        "Création du violon plot de la distribution des pixels par classe...")

    # Regrouper les valeurs par classe
    distribution_par_classe = {}
    for feature in stats:
        classe = feature['properties']['Nom']
        valeur = feature['properties']['sum']
        distribution_par_classe.setdefault(classe, []).append(valeur)

    # Conversion en listes
    classes = list(distribution_par_classe.keys())
    distributions = list(distribution_par_classe.values())

    # Configuration du style du graphique
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))

    # Création du violon plot
    parts = plt.violinplot(
        distributions,
        showmeans=True,
        showmedians=True,
        showextrema=False,
        widths=0.7
    )

    for pc in parts['bodies']:
        pc.set_facecolor('#D3D3D3')
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    # Personnalisation des lignes (médianes et moyennes)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('blue')
        parts['cmeans'].set_linewidth(1.5)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(2)

    # Configurations des axes et titres
    plt.xticks(
        range(1, len(classes) + 1),
        classes,
        rotation=45,
        fontsize=12
    )
    plt.title('Distribution des Pixels par Classe de Polygone',
              fontsize=18, fontweight='bold')
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Nombre de Pixels', fontsize=14)

    # Ajout des limites de l'axe Y
    plt.ylim(bottom=0, top=15000)

    # Légende
    plt.legend([
        plt.Line2D([0], [0], color='red', lw=2),
        plt.Line2D([0], [0], color='blue', lw=1.5),
        mpatches.Patch(facecolor='#D3D3D3', edgecolor='black', alpha=0.8)
    ], ['Médiane', 'Moyenne', 'Distribution'], loc='upper right', fontsize=10)

    plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(
        output_dir, 'violin_plot_nb_pix_by_poly_by_class.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Violon plot sauvegardé avec succès : {output_path}")

# ======================================== #
# ===  PHÉNOLOGIE DES PEUPLEMENTS PURS === #
# ======================================== #


def load_and_filter_shapefile(shapefile_path, classes_interet):
    """
    Charge un shapefile et filtre les entités selon les classes d'intérêt.
    """
    logger.info("Chargement du shapefile.")
    gdf = gpd.read_file(shapefile_path)
    initial_count = len(gdf)
    gdf_filtered = gdf[gdf['Nom'].isin(classes_interet)]
    filtered_count = len(gdf_filtered)
    logger.info(
        f"Filtrage des classes d'intérêt: {filtered_count} sur {initial_count} entités conservées.")
    return gdf_filtered


def load_raster_and_metadata(raster_path):
    """
    Charge un raster et récupère ses métadonnées.
    """
    logger.info("Ouverture du raster NDVI.")
    dataset = gdal.Open(raster_path)
    if dataset is None:
        logger.error(f"Impossible d'ouvrir le raster: {raster_path}")
        return None, None, None
    nb_bands = dataset.RasterCount
    geo_transform = dataset.GetGeoTransform()
    affine_transform = Affine.from_gdal(*geo_transform)
    logger.info(
        f"Raster chargé avec {nb_bands} bandes et transformation affine : {affine_transform}.")
    return dataset, nb_bands, affine_transform


def extract_dates_from_bands(dataset, nb_bands):
    """
    Extrait les dates à partir des descriptions des bandes du raster.
    """
    dates = []
    for band_idx in range(1, nb_bands + 1):
        band = dataset.GetRasterBand(band_idx)
        desc = band.GetDescription()
        parts = desc.split("NDVI_")[-1].split("_") if "NDVI_" in desc else []
        month = parts[0] if len(
            parts) > 0 and parts[0].isdigit() else f"{band_idx:02d}"
        year = parts[1] if len(parts) > 1 and parts[1].isdigit() else "2022"
        dates.append(np.datetime64(f"{year}-{month}"))
    logger.info(f"Dates extraites des bandes : {dates}.")
    return dates


def compute_zonal_statistics(dataset, gdf, classes_interet, affine_transform, nb_bands):
    """
    Calcule les statistiques zonales (moyenne et écart type) pour chaque classe et bande.
    """
    stats_mean = {classe: [] for classe in classes_interet}
    stats_std = {classe: [] for classe in classes_interet}

    for band_idx in range(1, nb_bands + 1):
        logger.info(f"Traitement de la bande {band_idx}/{nb_bands}.")
        band = dataset.GetRasterBand(band_idx)
        ndvi = band.ReadAsArray()
        nodata_value = band.GetNoDataValue() or -999

        for classe in classes_interet:
            gdf_classe = gdf[gdf['Nom'] == classe]
            zs = zonal_stats(
                vectors=gdf_classe,
                raster=ndvi,
                affine=affine_transform,
                stats=['mean', 'std'],
                geojson_out=False,
                nodata=nodata_value
            )
            values = [stat['mean'] for stat in zs if stat['mean'] is not None]
            if values:
                stats_mean[classe].append(np.mean(values))
                stats_std[classe].append(np.std(values))
            else:
                stats_mean[classe].append(np.nan)
                stats_std[classe].append(np.nan)
    return stats_mean, stats_std


def plot_ndvi_time_series(dates, stats_mean, stats_std, classes_interet, output_plot_path=None):
    """
    Trace les séries temporelles NDVI pour chaque classe.
    """
    logger.info("Création des graphiques des séries temporelles.")
    plt.figure(figsize=(15, 10))
    num_classes = len(classes_interet)
    cols = 2
    rows = (num_classes + 1) // cols

    for idx, classe in enumerate(classes_interet, 1):
        plt.subplot(rows, cols, idx)
        mean = np.array(stats_mean[classe])
        std = np.array(stats_std[classe])

        plt.plot(dates, mean, '-o', label="NDVI : Moyenne ± Écart type")
        plt.fill_between(dates, mean - std, mean + std,
                         color='skyblue', alpha=0.5)

        plt.title(f"Série Temporelle NDVI - {classe}")
        plt.xlabel("Date")
        plt.ylabel("NDVI")
        plt.ylim(0.5, 1)
        plt.xticks(dates, [d.astype('datetime64[M]').astype(str)
                   for d in dates], rotation=45, ha='right')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    if output_plot_path:
        plt.savefig(output_plot_path)
        logger.info(f"Graphique sauvegardé sous {output_plot_path}.")
    plt.show()


def process_ndvi(raster_path, shapefile_path, classes_interet, output_plot_path=None):
    """
    Traite un raster NDVI multibandes et un shapefile pour calculer les statistiques NDVI
    globales par classe et tracer des séries temporelles.
    """
    logger.info("Début du traitement du NDVI.")
    gdf = load_and_filter_shapefile(shapefile_path, classes_interet)
    dataset, nb_bands, affine_transform = load_raster_and_metadata(raster_path)
    if dataset is None:
        return
    dates = extract_dates_from_bands(dataset, nb_bands)
    stats_mean, stats_std = compute_zonal_statistics(
        dataset, gdf, classes_interet, affine_transform, nb_bands)
    plot_ndvi_time_series(dates, stats_mean, stats_std,
                          classes_interet, output_plot_path)


# ======================================== #
# ===   FONCTIONS POUR L'ANALYSE DE    === #
# ===     LA VARIABILITÉ SPECTRALE     === #
# ======================================== #
"""
Fonctions pour l'Analyse de Variabilité Spectrale

Ces fonctions servent à extraire les valeurs NDVI à l’échelle du pixel, 
par classe ou par polygone, à calculer le centroïde et les distances
au centroïde des classes et des polygones, ainsi qu’à générer des graphiques.
"""


def extraire_valeurs_ndvi_par_classe(shapefile_path, raster_path, groupes):
    """
    Extrait les valeurs NDVI pour chaque classe au niveau du pixel, 
    mais seulement pour les classes présentes dans le dictionnaire 'groupes'.
    """
    try:
        logger.info(
            "Chargement du shapefile et du raster NDVI pour extraction par classe...")
        gdf = gpd.read_file(shapefile_path)
        if 'Nom' not in gdf.columns:
            log_error_and_raise(
                "La colonne 'Nom' est requise dans le shapefile.")

        # Filtrer uniquement les classes présentes dans 'groupes'
        classes_valide = set(groupes.keys())
        gdf = gdf[gdf['Nom'].isin(classes_valide)]
        if gdf.empty:
            log_error_and_raise(
                "Aucune classe dans le shapefile ne correspond aux classes du dictionnaire 'groupes'.")

        # Dissoudre par classe
        logger.info("Dissolution des polygones par classe...")
        gdf_classes = gdf.dissolve(by='Nom')
        gdf_classes['ClassID'] = range(1, len(gdf_classes) + 1)

        temp_class_shp = "temp_classes_dissolues.shp"
        gdf_classes.to_file(temp_class_shp)

        src = gdal.Open(raster_path)
        if src is None:
            log_error_and_raise("Impossible d'ouvrir le raster NDVI.")

        geotransform = src.GetGeoTransform()
        projection = src.GetProjection()
        xsize = src.RasterXSize
        ysize = src.RasterYSize
        n_bandes = src.RasterCount

        logger.info("Rasterisation des classes...")
        driver = gdal.GetDriverByName('MEM')
        class_raster = driver.Create('', xsize, ysize, 1, gdal.GDT_Int16)
        class_raster.SetGeoTransform(geotransform)
        class_raster.SetProjection(projection)

        band = class_raster.GetRasterBand(1)
        band.Fill(0)  # 0 = aucune classe
        band.SetNoDataValue(0)

        ds = ogr.Open(temp_class_shp)
        layer = ds.GetLayer()

        gdal.RasterizeLayer(class_raster, [1], layer, options=[
                            "ATTRIBUTE=ClassID"])

        logger.info("Lecture de toutes les bandes NDVI...")
        ndvi_data = src.ReadAsArray()

        class_arr = class_raster.ReadAsArray()

        resultats_par_classe = {}
        for idx, row in gdf_classes.iterrows():
            classe = str(idx)  # idx est la classe (Nom)
            class_id = row['ClassID']
            mask_classe = (class_arr == class_id)
            pixels_classe = ndvi_data[:, mask_classe].T
            total_pixels = pixels_classe.shape[0]
            resultats_par_classe[classe] = {
                "total_pixels": total_pixels,
                "ndvi_means": pixels_classe
            }

        logger.info("Extraction des valeurs NDVI par classe terminée.")
        return resultats_par_classe

    except Exception as e:
        log_error_and_raise(
            f"Erreur lors de l'extraction des valeurs NDVI par classe : {e}")


def extraire_valeurs_ndvi_par_polygone(shapefile_path, raster_path, groupes):
    """
    Extrait les valeurs NDVI pour chaque polygone, uniquement pour les classes présentes dans 'groupes'.
    """
    try:
        logger.info(
            "Chargement du shapefile et du raster NDVI pour extraction par polygone...")
        gdf = gpd.read_file(shapefile_path)
        if 'Nom' not in gdf.columns:
            log_error_and_raise(
                "La colonne 'Nom' est requise dans le shapefile.")

        # Filtrer les classes
        classes_valide = set(groupes.keys())
        gdf = gdf[gdf['Nom'].isin(classes_valide)]
        if gdf.empty:
            log_error_and_raise(
                "Aucune classe dans le shapefile ne correspond aux classes du dictionnaire 'groupes'.")

        if 'PolyIntID' not in gdf.columns:
            gdf['PolyIntID'] = range(1, len(gdf) + 1)

        temp_poly_shp = "temp_polygones_int_id.shp"
        gdf.to_file(temp_poly_shp)

        src = gdal.Open(raster_path)
        if src is None:
            log_error_and_raise("Impossible d'ouvrir le raster NDVI.")

        geotransform = src.GetGeoTransform()
        projection = src.GetProjection()
        xsize = src.RasterXSize
        ysize = src.RasterYSize
        n_bandes = src.RasterCount

        logger.info("Rasterisation des polygones...")
        driver = gdal.GetDriverByName('MEM')
        poly_raster = driver.Create('', xsize, ysize, 1, gdal.GDT_Int32)
        poly_raster.SetGeoTransform(geotransform)
        poly_raster.SetProjection(projection)

        band = poly_raster.GetRasterBand(1)
        band.Fill(0)
        band.SetNoDataValue(0)

        ds = ogr.Open(temp_poly_shp)
        layer = ds.GetLayer()

        gdal.RasterizeLayer(poly_raster, [1], layer, options=[
                            "ATTRIBUTE=PolyIntID"])

        logger.info("Lecture de toutes les bandes NDVI...")
        ndvi_data = src.ReadAsArray()

        poly_arr = poly_raster.ReadAsArray()

        resultats_par_polygone = {}
        for i, row in gdf.iterrows():
            poly_int_id = row['PolyIntID']
            poly_class = row['Nom']
            original_id = row['ID'] if 'ID' in gdf.columns else None
            mask_poly = (poly_arr == poly_int_id)
            pixels_poly = ndvi_data[:, mask_poly].T
            total_pixels = pixels_poly.shape[0]

            resultats_par_polygone[poly_int_id] = {
                "ndvi_means": pixels_poly,
                "total_pixels": total_pixels,
                "class": poly_class,
                "original_id": original_id
            }

        logger.info("Extraction des valeurs NDVI par polygone terminée.")
        return resultats_par_polygone

    except Exception as e:
        log_error_and_raise(
            f"Erreur lors de l'extraction des valeurs NDVI par polygone : {e}")


def calculer_centroide_et_distances(valeurs_ndvi, niveau="classe", classes=None):
    """
    Calcule le centroïde et les distances moyennes au centroïde.

    Paramètres :
    -----------
    valeurs_ndvi : dict
        Dictionnaire contenant les NDVI par classe ou par polygone.
        Les valeurs NDVI doivent être un array (N_observations, N_bandes) où:
        - Si niveau="classe": N_observations = N_pixels de la classe
        - Si niveau="polygone": N_observations = N_pixels du polygone
    niveau : str
        Niveau d'analyse, "classe" ou "polygone".
    classes : dict, optionnel
        Dictionnaire associant chaque polygone (ID) à sa classe.

    Retourne :
    ---------
    dict
        {
          cle: {
             "centroide": array (n_bandes,) ou None,
             "distance_moyenne": float ou None,
             "classe": str ou None
          }
        }
        Si pas de NDVI, distance_moyenne et centroide = None.
    """
    try:
        resultats = {}

        for cle, valeurs in valeurs_ndvi.items():
            valeurs_array = valeurs["ndvi_means"]
            if valeurs_array is None or valeurs_array.size == 0:
                logger.warning(f"Aucune valeur NDVI pour {cle}. Ignoré.")
                # On retourne quand même une entrée pour éviter KeyError plus tard
                classe_associee = valeurs.get("class", None)
                if niveau == "polygone" and classes and (cle in classes):
                    classe_associee = classes[cle]

                resultats[cle] = {
                    "centroide": None,
                    "distance_moyenne": None,
                    "classe": classe_associee
                }
                continue

            # Assurer deux dimensions
            if valeurs_array.ndim == 1:
                valeurs_array = valeurs_array[np.newaxis, :]

            centroide = np.mean(valeurs_array, axis=0)  # (n_bandes,)
            distances = np.sqrt(np.sum((valeurs_array - centroide)**2, axis=1))
            distance_moyenne = np.mean(distances)

            classe_associee = valeurs.get("class", None)
            if niveau == "polygone" and classes and (cle in classes):
                classe_associee = classes[cle]

            resultats[cle] = {
                "centroide": centroide,
                "distance_moyenne": distance_moyenne,
                "classe": classe_associee
            }

        return resultats

    except Exception as e:
        log_error_and_raise(
            f"Erreur lors du calcul du centroïde et des distances : {e}")


def plot_barchart_distance_classes(distances_par_classe, output_path, groupes):
    """
    Génère un diagramme en bâton des distances moyennes au centroïde pour chaque classe.

    Paramètres :
    -----------
    distances_par_classe : dict
        Dictionnaire {classe: distance_moyenne}
    output_path : str
        Chemin de sauvegarde du graphique.
    groupes : dict
        Dictionnaire associant chaque classe à "Pur" ou "Mélange".
    """
    # Préparation des données
    data = [(classe, distance, groupes.get(classe, "Autre"))
            for classe, distance in distances_par_classe.items()]
    data_sorted = sorted(data, key=lambda x: (x[2] != "Pur", x[0]))

    # Décomposition des données pour le graphique
    classes = [item[0] for item in data_sorted]
    distances = [item[1] for item in data_sorted]
    couleurs = ['#FF9999' if item[2] ==
                "Pur" else '#66B3FF' for item in data_sorted]

    # Création du graphique
    fig, ax = plt.subplots(figsize=(12, 6))

    # Dessiner les barres
    bars = ax.bar(classes, distances, color=couleurs, edgecolor="black")

    # Ajouter des étiquettes sur chaque barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 50, f"{height:.2f}",
                ha='center', va='bottom', fontsize=9)

    # Personnalisation des axes et du titre
    ax.set_title("Distance moyenne des pixels au centroïde de leur classe",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Classe", fontsize=12)
    ax.set_ylabel("Distance moyenne au centroïde", fontsize=12)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)

    # Ajustement de la limite supérieure de Y
    ax.set_ylim(0, max(distances) * 1.2)

    # Ajout d'une légende
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor='#FF9999', edgecolor="black", label="Pur"),
                      plt.Rectangle((0, 0), 1, 1, facecolor='#66B3FF', edgecolor="black", label="Mélange")]

    ax.legend(handles=legend_handles, title="Essences", loc="upper right")

    # Amélioration de l'affichage
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Sauvegarde du graphique
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique sauvegardé : {output_path}")


def plot_violin_distance_polygons(distances_par_polygone, classes_polygones, output_path, groupes):
    """
    Génère un violin plot des distances moyennes au centroïde par classe, à l'échelle des polygones.
    Les classes pures (Pur) sont affichées en premier (en rouge), et les classes mélange (Mélange) en second (en bleu).
    Un violon par classe.

    Paramètres :
    -----------
    distances_par_polygone : list of float
        Liste des distances moyennes au centroïde pour chaque polygone.
    classes_polygones : list of str
        Liste des classes associées à chaque polygone, dans le même ordre que distances_par_polygone.
    output_path : str
        Chemin pour sauvegarder l'image.
    groupes : dict
        Dictionnaire associant chaque classe à "Pur" ou "Mélange".

    Note :
    Cette fonction suit la même logique de tri que le bar plot : d'abord les classes "Pur" puis "Mélange",
    classées alphabétiquement dans chaque groupe.
    """

    plt.style.use('ggplot')

    # Agréger les distances par classe
    distances_par_classe = {}
    for dist, cls in zip(distances_par_polygone, classes_polygones):
        if cls not in distances_par_classe:
            distances_par_classe[cls] = []
        distances_par_classe[cls].append(dist)

    # Déterminer le groupe (Pur/Mélange) de chaque classe
    classes_data = [(cls, np.median(distances_par_classe[cls]), groupes.get(cls, "Autre"))
                    for cls in distances_par_classe]

    # Trier les données : Pur d'abord, puis Mélange, puis Autre (si existe), par ordre alphabétique dans chaque groupe
    classes_data_sorted = sorted(
        classes_data, key=lambda x: (x[2] != "Pur", x[0]))

    # Extraire la liste des classes triées et des données correspondantes
    classes_ordre = [item[0] for item in classes_data_sorted]
    data = [distances_par_classe[cls] for cls in classes_ordre]
    groupes_ordre = [groupes.get(cls, "Autre") for cls in classes_ordre]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Création du violon plot par classe
    violin_parts = ax.violinplot(
        data, showmeans=True, showmedians=True, showextrema=True)

    # Appliquer les couleurs en fonction du groupe
    # Pur (rouge), Mélange (bleu)
    for i, pc in enumerate(violin_parts['bodies']):
        if groupes_ordre[i] == "Pur":
            pc.set_facecolor('#FF9999')
        else:
            pc.set_facecolor('#66B3FF')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # Personnalisation des lignes (médianes, extrêmes, etc.)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = violin_parts[partname]
        vp.set_edgecolor('black')  # Couleur noire pour les lignes principales
        vp.set_linewidth(1)

    # Personnalisation des axes et du titre
    ax.set_xticks(range(1, len(classes_ordre) + 1))
    ax.set_xticklabels(classes_ordre, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel("Distance moyenne au centroïde", fontsize=12)
    ax.set_title("Distribution des distances moyennes par polygone, par classe",
                 fontsize=14, fontweight="bold")

    # Légende
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#FF9999',
                      edgecolor='black', label="Pur"),  # Classes pures
        plt.Rectangle((0, 0), 1, 1, facecolor='#66B3FF',
                      edgecolor='black', label="Mélange"),  # Classes mélanges
        Line2D([0], [0], color='red', linestyle='-',
               linewidth=1, label="Médiane")  # Line rouge
    ]

    fig.subplots_adjust(right=0.8)  # Réserver un espace pour la légende
    ax.legend(handles=legend_handles, title="Essences",
              loc="upper right", bbox_to_anchor=(1.13, 1.015))

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique sauvegardé : {output_path}")


def report_from_dict_to_df(dict_report):
    """
    Convertit un rapport de classification en un DataFrame Pandas.

    Cette fonction prend un dictionnaire généré par un rapport
    de classification (issu de `classification_report` de scikit-learn),
    le transforme en DataFrame et élimine les colonnes et lignes non pertinentes.

    Paramètres :
    ------------
    dict_report : dict
        Le dictionnaire contenant les métriques du rapport de classification.

    Retours :
    ---------
    report_df : pandas.DataFrame
        Un DataFrame contenant uniquement les métriques par classe (précision, rappel, F1).

    Exceptions :
    ------------
    Retourne un DataFrame vide si l'entrée est invalide.
    """
    try:
        # Conversion du dictionnaire en DataFrame
        report_df = pd.DataFrame.from_dict(dict_report)

        # Vérifier et supprimer les colonnes non pertinentes si elles existent
        cols_to_drop = ['accuracy', 'macro avg', 'weighted avg', 'micro avg']
        for col in cols_to_drop:
            if col in report_df.columns:
                report_df.drop(columns=col, inplace=True)

        # Vérifier et supprimer la ligne "support" si elle existe
        if 'support' in report_df.index:
            report_df.drop(index='support', inplace=True)

        # Retourner le DataFrame nettoyé
        return report_df

    except Exception as e:
        logging.warning(f"Erreur lors de la conversion du rapport : {e}")
        return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur


def create_raster_sampleimage(sample_vector, reference_raster, output_path, attribute):
    """
    Crée un raster d'échantillonnage à partir d'un vecteur et d'un raster de référence.

    Cette fonction génère un raster en utilisant un vecteur d'échantillons 
    et un raster de référence, en rasterisant selon un attribut donné.

    Paramètres :
    ------------
    sample_vector : str
        Chemin vers le fichier vecteur contenant les échantillons.
    reference_raster : str
        Chemin vers le raster de référence définissant la géométrie et la projection.
    output_path : str
        Chemin pour sauvegarder le raster de sortie.
    attribute : str
        Nom de l'attribut dans le vecteur à utiliser pour la rasterisation.

    Retours :
    ---------
    Aucun. Un fichier raster est généré à l'emplacement `output_path`.

    Exceptions :
    ------------
    Lève une exception si le fichier vecteur ou raster de référence ne peut pas être ouvert,
    ou si la rasterisation échoue.
    """
    try:
        # Vérifier l'existence des fichiers d'entrée
        if not os.path.exists(sample_vector):
            raise FileNotFoundError(
                f"Fichier vecteur introuvable : {sample_vector}")
        if not os.path.exists(reference_raster):
            raise FileNotFoundError(
                f"Raster de référence introuvable : {reference_raster}")

        # Ouvrir le raster de référence
        ref_ds = gdal.Open(reference_raster, gdal.GA_ReadOnly)
        if ref_ds is None:
            raise RuntimeError(
                f"Impossible d'ouvrir le raster de référence : {reference_raster}")

        # Extraire les propriétés du raster de référence
        geo_transform = ref_ds.GetGeoTransform()
        projection = ref_ds.GetProjection()
        x_pixels = ref_ds.RasterXSize
        y_pixels = ref_ds.RasterYSize

        # Déterminer le type de données pour le raster de sortie
        raster_dtype = gdal.GDT_UInt16

        # Créer un raster en mémoire
        out_raster = gdal.GetDriverByName("MEM").Create(
            "", x_pixels, y_pixels, 1, raster_dtype
        )
        out_raster.SetGeoTransform(geo_transform)
        out_raster.SetProjection(projection)
        out_band = out_raster.GetRasterBand(1)
        out_band.SetNoDataValue(0)  # NoData = 0
        out_band.Fill(0)  # Remplir avec la valeur NoData

        # Ouvrir le vecteur
        vector_ds = gdal.OpenEx(sample_vector, gdal.OF_VECTOR)
        if vector_ds is None:
            raise RuntimeError(
                f"Impossible d'ouvrir le fichier vecteur : {sample_vector}")
        vector_layer = vector_ds.GetLayer()

        # Rasteriser le vecteur
        options = [f"ATTRIBUTE={attribute}"]
        gdal.RasterizeLayer(out_raster, [1], vector_layer, options=options)

        # Sauvegarder le raster sur disque
        driver = gdal.GetDriverByName("GTiff")
        driver.CreateCopy(output_path, out_raster, options=["COMPRESS=LZW"])

        logging.info(
            f"Raster d'échantillonnage créé avec succès : {output_path}")

    except Exception as e:
        logging.error(
            f"Erreur lors de la création du raster d'échantillonnage : {e}")
        raise


def get_pixel_area(raster_path):
    """
    Renvoie la surface d'un pixel (en m²) à partir de la géotransformation
    du raster.

    Paramètres
    ----------
    raster_path : str
        Chemin vers le fichier raster.

    Retourne
    -------
    float
        Surface d'un pixel en m².
    """
    raster = gdal.Open(raster_path)
    if raster is None:
        raise FileNotFoundError(
            f"Impossible d'ouvrir le raster : {raster_path}")

    geo_transform = raster.GetGeoTransform()
    # Taille du pixel
    pixel_width = abs(geo_transform[1])
    pixel_height = abs(geo_transform[5])
    pixel_surface = pixel_width * pixel_height
    raster = None  # fermeture
    return pixel_surface


def classify_peuplement_feuillus_coniferes(pourcentage_feuillus,
                                           pourcentage_coniferes,
                                           seuil,
                                           surf,
                                           surf_mini,
                                           feuillus_ilots,
                                           coniferes_ilots,
                                           melange_feuillus,
                                           melange_coniferes,
                                           melange_conif_prep_feuil,
                                           melange_feuil_prep_conif):
    """
    Classifie un polygone en fonction des pourcentages de feuillus/conifères et
    de sa superficie, selon les règles de décision décrites dans la consigne.

    Paramètres
    ----------
    pourcentage_feuillus : float
        Pourcentage de feuillus dans le polygone.
    pourcentage_coniferes : float
        Pourcentage de conifères dans le polygone.
    seuil : float
        Seuil de pourcentage pour catégoriser majoritairement feuillu ou conifère.
    surf : float
        Superficie totale du polygone (m²).
    surf_mini : float
        Seuil de superficie pour différencier petits îlots et grands polygones.
    feuillus_ilots : int
        Code de classe pour "îlots feuillus".
    coniferes_ilots : int
        Code de classe pour "îlots conifères".
    melange_feuillus : int
        Code de classe pour "peuplement feuillus" (mélange).
    melange_coniferes : int
        Code de classe pour "peuplement conifères" (mélange).
    melange_conif_prep_feuil : int
        Code de classe pour "mélange conifères à prédominance feuillus".
    melange_feuil_prep_conif : int
        Code de classe pour "mélange feuillus à prédominance conifères".

    Retourne
    -------
    int
        Code correspondant à la classe attribuée.
    """
    if surf < surf_mini:
        # Petits îlots
        if pourcentage_feuillus > seuil:
            return feuillus_ilots
        elif pourcentage_coniferes > seuil:
            return coniferes_ilots
        elif pourcentage_coniferes > pourcentage_feuillus:
            return melange_conif_prep_feuil
        else:
            return melange_feuil_prep_conif
    else:
        # Grands polygones
        if pourcentage_feuillus > seuil:
            return melange_feuillus
        elif pourcentage_coniferes > seuil:
            return melange_coniferes
        elif pourcentage_coniferes > pourcentage_feuillus:
            return melange_conif_prep_feuil
        else:
            return melange_feuil_prep_conif


def compute_peuplement_class(stats_zonales,
                             pixel_surface,
                             seuil_categorie,
                             surface_mini,
                             codes_peuplement):
    """
    À partir des statistiques zonales et du code de classification,
    calcule la classe de peuplement pour chaque polygone.

    Paramètres
    ----------
    stats_zonales : list of dict
        Sortie de la fonction compute_zonal_statistics().
    pixel_surface : float
        Surface d'un pixel (m²).
    seuil_categorie : float
        Seuil pour distinguer majoritairement feuillus ou conifères (en %).
    surface_mini : float
        Seuil de superficie pour différencier les petits îlots des grands polygones.
    codes_peuplement : dict
        Dictionnaire contenant les codes de classes (ex. "feuillus_ilots": 16, etc.).

    Retourne
    -------
    classes_predites : list of int
        Liste des classes prédites (une par polygone).
    percentages_feuillus : list of float
        Liste des pourcentages feuillus.
    percentages_coniferes : list of float
        Liste des pourcentages conifères.
    surfaces : list of float
        Liste des surfaces des polygones (m²).
    """
    classes_predites = []
    percentages_feuillus = []
    percentages_coniferes = []
    surfaces = []

    # Récupération des différents codes
    feuillus_ilots = codes_peuplement["feuillus_ilots"]
    melange_feuillus = codes_peuplement["melange_feuillus"]
    coniferes_ilots = codes_peuplement["coniferes_ilots"]
    melange_coniferes = codes_peuplement["melange_coniferes"]
    melange_conif_prep_feuil = codes_peuplement["melange_conif_prep_feuil"]
    melange_feuil_prep_conif = codes_peuplement["melange_feuil_prep_conif"]

    # Pour chaque polygone
    for stats_poly in stats_zonales:
        total_pixels = sum(stats_poly.values())
        # Éviter la division par zéro
        if total_pixels == 0:
            classes_predites.append(0)  # ou un code "inconnu"
            percentages_feuillus.append(0)
            percentages_coniferes.append(0)
            surfaces.append(0)
            continue

        # Pourcentage de feuillus (ex. classes entre 11 et 14, selon votre nomenclature)
        pourcentage_feuillus = sum(
            (count / total_pixels) * 100
            for cat, count in stats_poly.items()
            # Exemple : classes feuillus = 11,12,13,14 -> adapter selon vos codes
            if 10 < int(cat) < 15
        )
        # Pourcentage de conifères (ex. classes entre 21 et 25)
        pourcentage_coniferes = sum(
            (count / total_pixels) * 100
            for cat, count in stats_poly.items()
            if 20 < int(cat) < 26
        )
        surface_poly = total_pixels * pixel_surface

        # Classification via la fonction
        classe_predite = classify_peuplement_feuillus_coniferes(
            pourcentage_feuillus,
            pourcentage_coniferes,
            seuil_categorie,
            surface_poly,
            surface_mini,
            feuillus_ilots,
            coniferes_ilots,
            melange_feuillus,
            melange_coniferes,
            melange_conif_prep_feuil,
            melange_feuil_prep_conif
        )

        classes_predites.append(classe_predite)
        percentages_feuillus.append(pourcentage_feuillus)
        percentages_coniferes.append(pourcentage_coniferes)
        surfaces.append(surface_poly)

    return classes_predites, percentages_feuillus, percentages_coniferes, surfaces
