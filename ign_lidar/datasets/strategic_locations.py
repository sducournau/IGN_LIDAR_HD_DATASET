"""
Module des localisations stratégiques et validation WFS.

Ce module contient la base de données des localisations stratégiques
pour la construction de datasets IA diversifiés pour la segmentation 3D
et l'extraction de bâtiments LOD2/LOD3.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

logger = logging.getLogger(__name__)

# ============================================================================
# BASE DE DONNÉES DES LOCALISATIONS STRATÉGIQUES
# ============================================================================

STRATEGIC_LOCATIONS = {
    # ========================================================================
    # MONUMENTS HISTORIQUES & PATRIMOINE UNESCO
    # ========================================================================
    "versailles_chateau": {
        "bbox": (2.110, 48.800, 2.135, 48.810),
        "category": "heritage_palace",
        "characteristics": ["chateau_royal", "architecture_classique", "toitures_complexes", "lod3"],
        "priority": 1,
        "target_tiles": 6
    },
    "reims_cathedrale": {
        "bbox": (4.030, 49.250, 4.040, 49.258),
        "category": "heritage_religious",
        "characteristics": ["cathedrale_gothique", "architecture_medievale", "geometrie_complexe"],
        "priority": 1,
        "target_tiles": 4
    },
    "carcassonne_cite": {
        "bbox": (2.360, 43.204, 2.370, 43.212),
        "category": "heritage_fortress",
        "characteristics": ["cite_medievale", "remparts", "tours_defense"],
        "priority": 1,
        "target_tiles": 5
    },
    "chambord_chateau": {
        "bbox": (1.515, 47.613, 1.525, 47.619),
        "category": "heritage_palace",
        "characteristics": ["chateau_renaissance", "toitures_sculptees", "tourelles"],
        "priority": 1,
        "target_tiles": 5
    },
    "avignon_palais_papes": {
        "bbox": (4.805, 43.950, 4.812, 43.954),
        "category": "heritage_palace",
        "characteristics": ["palais_gothique", "fortification", "architecture_14eme"],
        "priority": 1,
        "target_tiles": 4
    },
    "fontainebleau_chateau": {
        "bbox": (2.695, 48.400, 2.705, 48.407),
        "category": "heritage_palace",
        "characteristics": ["chateau_renaissance", "ailes_multiples", "cours_interieures"],
        "priority": 1,
        "target_tiles": 5
    },
    
    # ========================================================================
    # URBAIN DENSE - CENTRES HISTORIQUES
    # ========================================================================
    "paris_haussmann": {
        "bbox": (2.315, 48.865, 2.355, 48.880),
        "category": "urban_dense",
        "characteristics": ["haussmannien", "haute_densite", "toits_zinc", "lod2"],
        "priority": 1,
        "target_tiles": 10
    },
    "paris_marais": {
        "bbox": (2.355, 48.855, 2.365, 48.865),
        "category": "urban_dense",
        "characteristics": ["hotels_particuliers", "17eme_siecle", "cours_interieures"],
        "priority": 1,
        "target_tiles": 6
    },
    "paris_ile_cite": {
        "bbox": (2.340, 48.850, 2.355, 48.858),
        "category": "urban_dense_historic",
        "characteristics": ["notre_dame", "architecture_medievale", "densité_exceptionnelle"],
        "priority": 1,
        "target_tiles": 5
    },
    "lyon_presquile": {
        "bbox": (4.825, 45.755, 4.840, 45.770),
        "category": "urban_dense",
        "characteristics": ["centre_historique", "immeubles_19eme", "traboules"],
        "priority": 1,
        "target_tiles": 7
    },
    "lyon_vieux_lyon": {
        "bbox": (4.825, 45.760, 4.832, 45.768),
        "category": "urban_dense_historic",
        "characteristics": ["renaissance", "facades_colorees", "ruelles_etroites"],
        "priority": 1,
        "target_tiles": 5
    },
    "marseille_vieux_port": {
        "bbox": (5.365, 43.292, 5.380, 43.300),
        "category": "urban_dense",
        "characteristics": ["mediterraneen", "immeubles_balcons", "port"],
        "priority": 1,
        "target_tiles": 6
    },
    "strasbourg_centre": {
        "bbox": (7.745, 48.580, 7.755, 48.588),
        "category": "urban_dense_historic",
        "characteristics": ["colombages", "cathedrale", "alsacien", "unesco"],
        "priority": 1,
        "target_tiles": 6
    },
    "nantes_bouffay": {
        "bbox": (-1.555, 47.210, -1.547, 47.218),
        "category": "urban_dense",
        "characteristics": ["centre_medieval", "architecture_18eme", "mixte"],
        "priority": 1,
        "target_tiles": 5
    },
    "nice_vieille_ville": {
        "bbox": (7.275, 43.695, 7.282, 43.700),
        "category": "urban_dense_coastal",
        "characteristics": ["baroque", "facades_colorees", "rues_etroites"],
        "priority": 1,
        "target_tiles": 5
    },
    
    # ========================================================================
    # URBAIN MODERNE - TOURS & GRATTE-CIELS
    # ========================================================================
    "paris_defense": {
        "bbox": (2.225, 48.885, 2.245, 48.900),
        "category": "urban_modern",
        "characteristics": ["gratte_ciel", "architecture_moderne", "verre", "grande_arche"],
        "priority": 1,
        "target_tiles": 8
    },
    "lyon_part_dieu": {
        "bbox": (4.853, 45.758, 4.863, 45.765),
        "category": "urban_modern",
        "characteristics": ["tours_bureaux", "tour_crayon", "dalle_urbaine"],
        "priority": 1,
        "target_tiles": 5
    },
    "nanterre_prefecture": {
        "bbox": (2.205, 48.890, 2.215, 48.900),
        "category": "urban_modern",
        "characteristics": ["tours_bureaux", "quartier_affaires", "architecture_70s"],
        "priority": 1,
        "target_tiles": 4
    },
    "paris_13eme_moderne": {
        "bbox": (2.365, 48.825, 2.380, 48.840),
        "category": "urban_modern",
        "characteristics": ["bibliotheque_nationale", "tours_verre", "contemporain"],
        "priority": 1,
        "target_tiles": 5
    },
    
    # ========================================================================
    # PÉRIURBAIN / BANLIEUE - DIVERSITÉ
    # ========================================================================
    "paris_suburbs_nord": {
        "bbox": (2.360, 48.925, 2.410, 48.950),
        "category": "suburban_social",
        "characteristics": ["grands_ensembles", "barres_immeubles", "tours_hlm"],
        "priority": 2,
        "target_tiles": 7
    },
    "paris_suburbs_ouest_pavillons": {
        "bbox": (2.180, 48.840, 2.220, 48.870),
        "category": "suburban_residential",
        "characteristics": ["pavillonnaire", "maisons_individuelles", "lotissements"],
        "priority": 1,
        "target_tiles": 8
    },
    "paris_suburbs_est_mix": {
        "bbox": (2.440, 48.850, 2.480, 48.880),
        "category": "suburban_mixed",
        "characteristics": ["mixte", "barres_pavillons", "tissu_heterogene"],
        "priority": 2,
        "target_tiles": 6
    },
    "lyon_suburbs_villeurbanne": {
        "bbox": (4.875, 45.762, 4.895, 45.778),
        "category": "suburban_social",
        "characteristics": ["grands_ensembles", "architecture_60s", "gratte_ciel_villeurbanne"],
        "priority": 2,
        "target_tiles": 5
    },
    "marseille_suburbs_nord": {
        "bbox": (5.385, 43.340, 5.420, 43.365),
        "category": "suburban_social",
        "characteristics": ["grands_ensembles", "architecture_mediterraneenne", "relief"],
        "priority": 2,
        "target_tiles": 5
    },
    "toulouse_suburbs_colomiers": {
        "bbox": (1.305, 43.605, 1.335, 43.625),
        "category": "suburban_residential",
        "characteristics": ["pavillonnaire", "brique", "aeronautique_proche"],
        "priority": 2,
        "target_tiles": 4
    },
    
    # ========================================================================
    # RURAL TRADITIONNEL - VILLAGES CIBLÉS
    # ========================================================================
    # PROVENCE - Villages spécifiques avec bâti
    "provence_gordes": {
        "bbox": (5.196, 43.908, 5.206, 43.918),
        "category": "rural_traditional",
        "characteristics": ["pierre_provencale", "village_perche", "toits_tuiles", "dense"],
        "priority": 1,
        "target_tiles": 3
    },
    "provence_roussillon": {
        "bbox": (5.288, 43.898, 5.298, 43.908),
        "category": "rural_traditional",
        "characteristics": ["ocre", "village_perche", "facades_colorees"],
        "priority": 1,
        "target_tiles": 2
    },
    "provence_menerbes": {
        "bbox": (5.195, 43.828, 5.205, 43.838),
        "category": "rural_traditional",
        "characteristics": ["village_luberon", "pierre", "toits_tuiles"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # ALSACE - Villages vignerons précis
    "alsace_riquewihr": {
        "bbox": (7.295, 48.163, 7.305, 48.173),
        "category": "rural_traditional",
        "characteristics": ["colombages_alsaciens", "village_vigneron", "medieval"],
        "priority": 1,
        "target_tiles": 3
    },
    "alsace_eguisheim": {
        "bbox": (7.302, 48.038, 7.312, 48.048),
        "category": "rural_traditional",
        "characteristics": ["colombages", "circulaire", "village_vigneron"],
        "priority": 1,
        "target_tiles": 3
    },
    
    # PÉRIGORD - Villages concentrés
    "perigord_sarlat": {
        "bbox": (1.210, 44.885, 1.225, 44.898),
        "category": "rural_traditional",
        "characteristics": ["pierre_blonde", "medieval", "centre_historique"],
        "priority": 1,
        "target_tiles": 3
    },
    "perigord_domme": {
        "bbox": (1.210, 44.798, 1.220, 44.808),
        "category": "rural_traditional",
        "characteristics": ["bastide", "village_perche", "pierre_perigourdine"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # AUVERGNE - Villages de pierre volcanique
    "auvergne_salers": {
        "bbox": (2.492, 45.138, 2.502, 45.148),
        "category": "rural_traditional",
        "characteristics": ["pierre_volcanique", "village_medieval", "toits_lauzes"],
        "priority": 1,
        "target_tiles": 2
    },
    "auvergne_besse": {
        "bbox": (2.935, 45.503, 2.945, 45.513),
        "category": "rural_traditional",
        "characteristics": ["pierre_volcanique", "village_altitude", "architecture_auvergnate"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # JURA - Villages et fermes comtoises
    "jura_baume_les_messieurs": {
        "bbox": (5.642, 46.708, 5.652, 46.718),
        "category": "rural_traditional",
        "characteristics": ["village_reculee", "abbaye", "fermes_comtoises"],
        "priority": 1,
        "target_tiles": 2
    },
    "jura_arbois": {
        "bbox": (5.770, 46.900, 5.780, 46.910),
        "category": "rural_traditional",
        "characteristics": ["village_vigneron", "pierre_jura", "toits_tuiles"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # PAYS BASQUE - Villages concentrés
    "pays_basque_espelette": {
        "bbox": (-1.452, 43.332, -1.442, 43.342),
        "category": "rural_traditional",
        "characteristics": ["facades_blanches_rouges", "village_typique", "colombages_basques"],
        "priority": 1,
        "target_tiles": 2
    },
    "pays_basque_ainhoa": {
        "bbox": (-1.395, 43.320, -1.385, 43.330),
        "category": "rural_traditional",
        "characteristics": ["bastide_navarraise", "facades_basques", "village_rue"],
        "priority": 1,
        "target_tiles": 2
    },
    "pays_basque_sare": {
        "bbox": (-1.582, 43.310, -1.572, 43.320),
        "category": "rural_traditional",
        "characteristics": ["village_montagne", "architecture_labourdine", "fermes"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # BRETAGNE - Villages ciblés
    "bretagne_dinan": {
        "bbox": (-2.048, 48.450, -2.038, 48.460),
        "category": "rural_traditional",
        "characteristics": ["cite_medievale", "colombages", "remparts"],
        "priority": 1,
        "target_tiles": 2
    },
    "bretagne_locronan": {
        "bbox": (-4.183, 48.095, -4.173, 48.105),
        "category": "rural_traditional",
        "characteristics": ["village_granit", "place_medievale", "pierres"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # NORMANDIE - Villages et bourgs ciblés
    "normandie_etretat": {
        "bbox": (0.200, 49.705, 0.215, 49.715),
        "category": "coastal_traditional",
        "characteristics": ["village_cotier", "falaises", "architecture_normande"],
        "priority": 1,
        "target_tiles": 2
    },
    "normandie_beuvron_en_auge": {
        "bbox": (-0.040, 49.190, -0.030, 49.200),
        "category": "rural_traditional",
        "characteristics": ["colombages_normands", "village_fleuri", "fermes"],
        "priority": 1,
        "target_tiles": 2
    },
    "normandie_lyons_la_foret": {
        "bbox": (1.473, 49.395, 1.483, 49.405),
        "category": "rural_traditional",
        "characteristics": ["colombages", "village_foret", "halles"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # BOURGOGNE - Villages viticoles
    "bourgogne_beaune": {
        "bbox": (4.833, 47.018, 4.843, 47.028),
        "category": "rural_traditional",
        "characteristics": ["hospices", "toits_tuiles_vernissees", "medieval"],
        "priority": 1,
        "target_tiles": 2
    },
    "bourgogne_vezelay": {
        "bbox": (3.745, 47.463, 3.755, 47.473),
        "category": "rural_traditional",
        "characteristics": ["basilique", "village_perche", "pierre"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # LOIRE - Villages de tuffeau
    "loire_chinon": {
        "bbox": (0.235, 47.163, 0.245, 47.173),
        "category": "rural_traditional",
        "characteristics": ["tuffeau", "medieval", "forteresse"],
        "priority": 1,
        "target_tiles": 2
    },
    "loire_amboise": {
        "bbox": (0.980, 47.408, 0.990, 47.418),
        "category": "rural_traditional",
        "characteristics": ["chateau", "tuffeau", "renaissance"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # ========================================================================
    # CÔTIER - DIVERSITÉ LITTORALE
    # ========================================================================
    "cote_azur_nice": {
        "bbox": (7.260, 43.690, 7.290, 43.710),
        "category": "coastal_urban",
        "characteristics": ["promenade_anglais", "belle_epoque", "hotels_palaces"],
        "priority": 1,
        "target_tiles": 6
    },
    "cote_azur_cannes": {
        "bbox": (7.010, 43.545, 7.030, 43.560),
        "category": "coastal_urban",
        "characteristics": ["villas_luxe", "belle_epoque", "architecture_balneaire"],
        "priority": 1,
        "target_tiles": 5
    },
    "atlantique_arcachon": {
        "bbox": (-1.175, 44.650, -1.155, 44.670),
        "category": "coastal_residential",
        "characteristics": ["villas_arcachonnaises", "balneaire", "pilotis"],
        "priority": 1,
        "target_tiles": 5
    },
    "bretagne_concarneau": {
        "bbox": (-3.915, 47.870, -3.905, 47.878),
        "category": "coastal_historic",
        "characteristics": ["ville_close", "fortifications", "port", "granit"],
        "priority": 1,
        "target_tiles": 5
    },
    "bretagne_quimper": {
        "bbox": (-4.110, 47.990, -4.095, 48.000),
        "category": "coastal_traditional",
        "characteristics": ["maisons_colombages", "cathedrale", "architecture_bretonne"],
        "priority": 1,
        "target_tiles": 5
    },
    "normandie_cabourg": {
        "bbox": (-0.115, 49.285, -0.105, 49.295),
        "category": "coastal_residential",
        "characteristics": ["villas_balneaires", "architecture_19eme", "front_mer"],
        "priority": 1,
        "target_tiles": 4
    },
    "biarritz_villas": {
        "bbox": (-1.565, 43.480, -1.550, 43.492),
        "category": "coastal_residential",
        "characteristics": ["villas_basques", "belle_epoque", "architecture_balneaire"],
        "priority": 1,
        "target_tiles": 4
    },
    
    # ========================================================================
    # MONTAGNE - VILLAGES ET STATIONS CIBLÉS
    # ========================================================================
    # ALPES - Stations et villages précis
    "alpes_chamonix_centre": {
        "bbox": (6.867, 45.920, 6.877, 45.930),
        "category": "mountain_resort",
        "characteristics": ["station_ski", "chalets", "hotels_montagne"],
        "priority": 1,
        "target_tiles": 3
    },
    "alpes_megeve_centre": {
        "bbox": (6.616, 45.855, 6.626, 45.865),
        "category": "mountain_resort",
        "characteristics": ["village_luxe", "chalets_haut_gamme", "station"],
        "priority": 1,
        "target_tiles": 2
    },
    "alpes_val_isere": {
        "bbox": (6.975, 45.445, 6.985, 45.455),
        "category": "mountain_resort",
        "characteristics": ["station_altitude", "immeubles_ski", "chalets"],
        "priority": 1,
        "target_tiles": 2
    },
    "alpes_saint_veran": {
        "bbox": (6.870, 44.700, 6.880, 44.710),
        "category": "mountain_traditional",
        "characteristics": ["village_altitude", "chalets_traditionnels", "fustes"],
        "priority": 1,
        "target_tiles": 2
    },
    "alpes_beaufort": {
        "bbox": (6.570, 45.715, 6.580, 45.725),
        "category": "mountain_traditional",
        "characteristics": ["village_savoyard", "chalets_pierre", "toits_lauzes"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # PYRÉNÉES - Villages et stations ciblés
    "pyrenees_cauterets": {
        "bbox": (-0.115, 42.888, -0.105, 42.898),
        "category": "mountain_resort",
        "characteristics": ["station_thermale", "architecture_19eme", "hotels"],
        "priority": 1,
        "target_tiles": 2
    },
    "pyrenees_saint_lary": {
        "bbox": (0.320, 42.815, 0.330, 42.825),
        "category": "mountain_resort",
        "characteristics": ["station_ski", "village_montagne", "chalets"],
        "priority": 1,
        "target_tiles": 2
    },
    "pyrenees_villages_cerdagne": {
        "bbox": (1.985, 42.460, 1.995, 42.470),
        "category": "mountain_traditional",
        "characteristics": ["villages_catalans", "pierre_montagne", "ardoise"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # VOSGES - Villages ciblés
    "vosges_la_bresse": {
        "bbox": (6.877, 48.003, 6.887, 48.013),
        "category": "mountain_traditional",
        "characteristics": ["fermes_vosgiennes", "station_ski", "toits_pentus"],
        "priority": 1,
        "target_tiles": 2
    },
    "vosges_gerardmer": {
        "bbox": (6.875, 48.070, 6.885, 48.080),
        "category": "mountain_resort",
        "characteristics": ["ville_montagne", "hotels", "villas"],
        "priority": 1,
        "target_tiles": 2
    },
    
    # ========================================================================
    # INFRASTRUCTURES MAJEURES
    # ========================================================================
    "cdg_airport": {
        "bbox": (2.535, 49.005, 2.575, 49.025),
        "category": "infrastructure_airport",
        "characteristics": ["terminaux_aeriens", "hangars", "structures_complexes", "geometrie_courbe"],
        "priority": 1,
        "target_tiles": 6
    },
    "orly_airport": {
        "bbox": (2.355, 48.720, 2.375, 48.735),
        "category": "infrastructure_airport",
        "characteristics": ["terminaux", "zones_fret", "structures_aeriennes"],
        "priority": 2,
        "target_tiles": 4
    },
    "port_marseille": {
        "bbox": (5.335, 43.315, 5.365, 43.335),
        "category": "infrastructure_port",
        "characteristics": ["terminaux_portuaires", "entrepots", "grues", "containers"],
        "priority": 1,
        "target_tiles": 5
    },
    "port_nantes_saint_nazaire": {
        "bbox": (-2.200, 47.270, -2.180, 47.285),
        "category": "infrastructure_port",
        "characteristics": ["port_containers", "chantiers_navals", "industries"],
        "priority": 1,
        "target_tiles": 4
    },
    "gare_montparnasse": {
        "bbox": (2.318, 48.840, 2.322, 48.844),
        "category": "infrastructure_rail",
        "characteristics": ["gare_tgv", "architecture_moderne", "dalle_urbaine"],
        "priority": 1,
        "target_tiles": 3
    },
    "gare_lyon_part_dieu": {
        "bbox": (4.857, 45.760, 4.862, 45.763),
        "category": "infrastructure_rail",
        "characteristics": ["gare", "dalle_urbaine", "structures_complexes"],
        "priority": 2,
        "target_tiles": 3
    },
    "zone_industrielle_lyon": {
        "bbox": (4.900, 45.580, 4.950, 45.610),
        "category": "infrastructure_industrial",
        "characteristics": ["usines", "entrepots_logistique", "hangars", "zones_activites"],
        "priority": 2,
        "target_tiles": 4
    },
    "raffinerie_fos": {
        "bbox": (4.850, 43.400, 4.900, 43.430),
        "category": "infrastructure_industrial",
        "characteristics": ["raffinerie", "cuves", "structures_industrielles", "petrochimie"],
        "priority": 2,
        "target_tiles": 3
    },
    "data_center_paris": {
        "bbox": (2.560, 48.950, 2.580, 48.965),
        "category": "infrastructure_tech",
        "characteristics": ["datacenters", "batiments_techniques", "logistique"],
        "priority": 2,
        "target_tiles": 2
    },
    
    # ========================================================================
    # ZONES MIXTES - VILLES MOYENNES
    # ========================================================================
    "toulouse_centre": {
        "bbox": (1.440, 43.600, 1.455, 43.610),
        "category": "mixed_urban",
        "characteristics": ["brique_toulousaine", "centre_historique", "place_capitole"],
        "priority": 1,
        "target_tiles": 6
    },
    "bordeaux_centre": {
        "bbox": (-0.580, 44.835, -0.570, 44.845),
        "category": "mixed_urban",
        "characteristics": ["pierre_bordeaux", "architecture_18eme", "unesco"],
        "priority": 1,
        "target_tiles": 6
    },
    "lille_centre": {
        "bbox": (3.055, 50.630, 3.070, 50.640),
        "category": "mixed_urban",
        "characteristics": ["architecture_flamande", "brique_rouge", "mixte"],
        "priority": 1,
        "target_tiles": 5
    },
    "montpellier_centre": {
        "bbox": (3.875, 43.608, 3.885, 43.615),
        "category": "mixed_urban",
        "characteristics": ["centre_historique", "architecture_languedocienne", "mixte"],
        "priority": 1,
        "target_tiles": 5
    },
    "rennes_centre": {
        "bbox": (-1.685, 48.110, -1.675, 48.118),
        "category": "mixed_urban",
        "characteristics": ["colombages", "apres_incendie", "mixte_ancien_moderne"],
        "priority": 1,
        "target_tiles": 4
    },
    "nice_centre": {
        "bbox": (7.265, 43.695, 7.275, 43.705),
        "category": "mixed_urban",
        "characteristics": ["architecture_belle_epoque", "balcons", "mediterraneen"],
        "priority": 1,
        "target_tiles": 4
    },
    
    # ========================================================================
    # ZONES SPÉCIFIQUES - CAMPUS & COMPLEXES
    # ========================================================================
    "saclay_campus": {
        "bbox": (2.145, 48.710, 2.175, 48.730),
        "category": "campus_research",
        "characteristics": ["campus_universitaire", "batiments_recherche", "architecture_contemporaine"],
        "priority": 2,
        "target_tiles": 4
    },
    "cite_universitaire_paris": {
        "bbox": (2.335, 48.818, 2.345, 48.825),
        "category": "campus_university",
        "characteristics": ["pavillons_internationaux", "architecture_20eme", "diversite_styles"],
        "priority": 2,
        "target_tiles": 3
    },
    "hopital_pitie_salpetriere": {
        "bbox": (2.360, 48.835, 2.368, 48.842),
        "category": "infrastructure_hospital",
        "characteristics": ["hopital", "batiments_historiques", "pavillons"],
        "priority": 2,
        "target_tiles": 3
    },
    "stade_france": {
        "bbox": (2.358, 48.920, 2.365, 48.926),
        "category": "infrastructure_sport",
        "characteristics": ["stade", "toiture_complexe", "structure_metallique"],
        "priority": 2,
        "target_tiles": 2
    },
    "parc_disneyland": {
        "bbox": (2.775, 48.865, 2.790, 48.875),
        "category": "infrastructure_leisure",
        "characteristics": ["parc_attractions", "hotels_thematiques", "structures_fantaisie"],
        "priority": 2,
        "target_tiles": 4
    },
}


# ============================================================================
# FONCTIONS DE VALIDATION ET SÉLECTION
# ============================================================================

def validate_locations_via_wfs(downloader) -> Dict[str, Dict]:
    """
    Valider toutes les localisations stratégiques via le WFS IGN.
    
    Args:
        downloader: Instance de IGNLiDARDownloader
    
    Returns:
        Dict avec le statut de chaque localisation
    """
    logger.info("=" * 70)
    logger.info("🔍 VALIDATION DES LOCALISATIONS STRATÉGIQUES VIA WFS IGN")
    logger.info("=" * 70)
    
    validation_results = {}
    category_stats = {}
    
    for location_name, config in STRATEGIC_LOCATIONS.items():
        logger.info(f"\nVérification: {location_name}")
        logger.info(f"  Catégorie: {config['category']}")
        logger.info(f"  BBox: {config['bbox']}")
        
        try:
            tiles_data = downloader.fetch_available_tiles(bbox=config['bbox'])
            
            available = tiles_data and 'features' in tiles_data
            tile_count = len(tiles_data.get('features', [])) if available else 0
            
            validation_results[location_name] = {
                'config': config,
                'available': available,
                'tile_count': tile_count,
                'status': 'OK' if available and tile_count > 0 else 'NO_TILES'
            }
            
            # Stats par catégorie
            category = config['category']
            if category not in category_stats:
                category_stats[category] = {
                    'locations': 0,
                    'tiles': 0,
                    'valid': 0
                }
            
            category_stats[category]['locations'] += 1
            if available and tile_count > 0:
                category_stats[category]['tiles'] += tile_count
                category_stats[category]['valid'] += 1
                logger.info(f"  ✅ {tile_count} tuiles disponibles")
            else:
                logger.warning(f"  ❌ Aucune tuile disponible")
            
            time.sleep(0.5)  # Respecter l'API
            
        except Exception as e:
            logger.error(f"  ❌ Erreur: {e}")
            validation_results[location_name] = {
                'config': config,
                'available': False,
                'tile_count': 0,
                'status': 'ERROR',
                'error': str(e)
            }
    
    # Résumé par catégorie
    logger.info("\n" + "=" * 70)
    logger.info("📊 RÉSUMÉ PAR CATÉGORIE")
    logger.info("=" * 70)
    
    for category, stats in sorted(category_stats.items()):
        logger.info(
            f"  {category:30s}: {stats['valid']:2d}/{stats['locations']:2d} "
            f"locations, {stats['tiles']:4d} tuiles"
        )
    
    total_valid = sum(s['valid'] for s in category_stats.values())
    total_tiles = sum(s['tiles'] for s in category_stats.values())
    
    logger.info(
        f"\n✅ TOTAL: {total_valid} localisations valides, "
        f"{total_tiles} tuiles disponibles"
    )
    
    return validation_results


def download_diverse_tiles(
    downloader,
    validation_results: Dict,
    download_dir: Path,
    max_total_tiles: int = 60
) -> List[Path]:
    """
    Télécharger des tuiles diversifiées selon la validation.
    
    Args:
        downloader: Instance de IGNLiDARDownloader
        validation_results: Résultats de validate_locations_via_wfs()
        download_dir: Répertoire de destination
        max_total_tiles: Nombre maximum de tuiles à télécharger
    
    Returns:
        Liste des fichiers LAZ téléchargés
    """
    logger.info("\n" + "=" * 70)
    logger.info("📥 TÉLÉCHARGEMENT STRATÉGIQUE DES TUILES")
    logger.info("=" * 70)
    
    # Filtrer et trier par priorité
    valid_locations = [
        (name, result) for name, result in validation_results.items()
        if result['available'] and result['tile_count'] > 0
    ]
    
    valid_locations.sort(
        key=lambda x: (x[1]['config']['priority'], -x[1]['tile_count'])
    )
    
    all_downloaded = []
    
    for location_name, result in valid_locations:
        if len(all_downloaded) >= max_total_tiles:
            logger.info(f"\n✅ Objectif de {max_total_tiles} tuiles atteint!")
            break
        
        config = result['config']
        category = config['category']
        
        # Créer le répertoire par catégorie
        category_dir = download_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculer combien de tuiles télécharger
        remaining = max_total_tiles - len(all_downloaded)
        target = min(config['target_tiles'], result['tile_count'], remaining)
        
        logger.info(f"\n📍 {location_name}")
        logger.info(f"   Catégorie: {category}")
        logger.info(
            f"   Caractéristiques: {', '.join(config['characteristics'])}"
        )
        logger.info(f"   Téléchargement: {target} tuiles")
        
        try:
            # Récupérer les tuiles
            tiles_data = downloader.fetch_available_tiles(bbox=config['bbox'])
            features = tiles_data.get('features', [])
            
            # Sélection aléatoire pour diversité
            random.shuffle(features)
            selected_features = features[:target]
            
            # Télécharger chaque tuile
            for i, feature in enumerate(selected_features):
                try:
                    properties = feature.get('properties', {})
                    tile_name = properties.get(
                        'name',
                        properties.get('nom_dalle', f'tile_{i}')
                    )
                    
                    # Nettoyer le nom
                    tile_name = tile_name.replace('.copc.laz', '')
                    tile_name = tile_name.replace('.laz', '')
                    output_path = category_dir / f"{tile_name}.laz"
                    
                    # Skip si déjà téléchargé
                    if (output_path.exists() and
                            output_path.stat().st_size > 1024 * 1024):
                        logger.info(f"     ✓ Existe déjà: {tile_name}")
                        all_downloaded.append(output_path)
                        continue
                    
                    # Télécharger
                    tile_url = properties.get('url')
                    if tile_url:
                        logger.info(
                            f"     📥 Téléchargement {i+1}/{target}: "
                            f"{tile_name}"
                        )
                        
                        import requests
                        response = requests.get(
                            tile_url,
                            stream=True,
                            timeout=600
                        )
                        response.raise_for_status()
                        
                        with open(output_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        # Validation basique
                        if output_path.stat().st_size > 1024 * 1024:
                            all_downloaded.append(output_path)
                            size_mb = output_path.stat().st_size / (1024 * 1024)
                            logger.info(
                                f"     ✅ Téléchargé: {size_mb:.1f} MB"
                            )
                        else:
                            logger.warning(
                                f"     ⚠️ Fichier trop petit, supprimé"
                            )
                            output_path.unlink()
                        
                        time.sleep(1)  # Respecter le serveur
                    
                except Exception as e:
                    logger.error(f"     ❌ Erreur tuile {i}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"   ❌ Erreur localisation: {e}")
            continue
        
        logger.info(
            f"   Progress global: {len(all_downloaded)}/{max_total_tiles} "
            f"tuiles"
        )
        time.sleep(2)
    
    logger.info(
        f"\n✅ Téléchargement terminé: {len(all_downloaded)} tuiles"
    )
    return all_downloaded


def get_categories() -> List[str]:
    """Retourne la liste des catégories de bâtiments disponibles."""
    categories = set()
    for config in STRATEGIC_LOCATIONS.values():
        categories.add(config['category'])
    return sorted(categories)


def get_locations_by_category(category: str) -> Dict[str, Dict]:
    """Retourne toutes les localisations d'une catégorie donnée."""
    return {
        name: config
        for name, config in STRATEGIC_LOCATIONS.items()
        if config['category'] == category
    }


def get_locations_by_priority(priority: int) -> Dict[str, Dict]:
    """Retourne toutes les localisations d'une priorité donnée."""
    return {
        name: config
        for name, config in STRATEGIC_LOCATIONS.items()
        if config['priority'] == priority
    }


def get_total_target_tiles() -> int:
    """Calcule le nombre total de tuiles ciblées."""
    return sum(config['target_tiles']
               for config in STRATEGIC_LOCATIONS.values())


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'STRATEGIC_LOCATIONS',
    'validate_locations_via_wfs',
    'download_diverse_tiles',
    'get_categories',
    'get_locations_by_category',
    'get_locations_by_priority',
    'get_total_target_tiles',
]
