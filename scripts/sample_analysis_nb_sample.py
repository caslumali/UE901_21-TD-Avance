#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse des échantillons de la BD forêt

Ce script permet de réaliser une analyse des échantillons de la BD forêt filtrée.

1. Produit un diagramme bâton du nombre de polygones par classe.
2. Produit un diagramme bâton du nombre de pixels par classe
3. Produit un violin plot de la distribution du nombre de pixels par classe de polygone.

Créé le 5 décembre 2024
"""

from my_function import sample_data_analysis

shapefile_path = "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp"
raster_path = "groupe_9/results/data/img_pretraitees/masque_foret.tif"
classes_a_conserver = [
    "Autres conifères autre que pin",
    "Autres feuillus",
    "Autres pin",
    "Chêne",
    "Douglas",
    "Peupleraie",
    "Pin laricio ou pin noir",
    "Pin maritime",
    "Robinier"
]

output_dir = "groupe_9/results/figure"

sample_data_analysis(shapefile_path, raster_path,
                     classes_a_conserver, output_dir)
