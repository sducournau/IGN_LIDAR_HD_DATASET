#!/usr/bin/env python3
"""
Script de test des endpoints WFS de l'IGN

Ce script vérifie que tous les layers WFS utilisés dans le projet sont accessibles
et retournent des données correctement depuis le service Géoplateforme de l'IGN.
"""

import requests
from urllib.parse import urlencode
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from datetime import datetime
import sys


@dataclass
class LayerTestResult:
    """Résultat du test d'un layer WFS."""
    layer_name: str
    status: str  # 'success', 'error', 'skipped'
    http_code: Optional[int] = None
    feature_count: Optional[int] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None


class WFSEndpointTester:
    """Testeur d'endpoints WFS IGN."""
    
    # Configuration WFS IGN
    WFS_URL = "https://data.geopf.fr/wfs"
    VERSION = "2.0.0"
    OUTPUT_FORMAT = "application/json"
    CRS = "EPSG:2154"
    
    # Bbox de test (zone de Versailles)
    TEST_BBOX = (650000, 6860000, 651000, 6861000)
    
    # Layers à tester (issus de IGNWFSConfig)
    LAYERS_TO_TEST = {
        # Layers principaux
        'buildings': 'BDTOPO_V3:batiment',
        'roads': 'BDTOPO_V3:troncon_de_route',
        'railways': 'BDTOPO_V3:troncon_de_voie_ferree',
        'water': 'BDTOPO_V3:surface_hydrographique',
        'vegetation': 'BDTOPO_V3:zone_de_vegetation',
        'sports': 'BDTOPO_V3:terrain_de_sport',
        
        # Layers additionnels
        'cemeteries': 'BDTOPO_V3:cimetiere',
        'power_lines': 'BDTOPO_V3:ligne_electrique',
        'constructions': 'BDTOPO_V3:construction_surfacique',
        'reservoirs': 'BDTOPO_V3:reservoir',
        
        # Layers désactivés (connus comme non disponibles)
        # 'bridges': None,  # BDTOPO_V3:pont - NOT AVAILABLE
        # 'parking': None,  # BDTOPO_V3:parking - NOT AVAILABLE
    }
    
    def __init__(self, timeout: int = 30):
        """
        Initialise le testeur.
        
        Args:
            timeout: Timeout pour les requêtes HTTP en secondes
        """
        self.timeout = timeout
        self.results: List[LayerTestResult] = []
    
    def test_wfs_capabilities(self) -> bool:
        """
        Teste GetCapabilities pour vérifier que le service WFS est disponible.
        
        Returns:
            True si le service est disponible, False sinon
        """
        print("=" * 80)
        print("TEST 1: GetCapabilities - Vérification du service WFS")
        print("=" * 80)
        
        params = {
            "SERVICE": "WFS",
            "REQUEST": "GetCapabilities"
        }
        
        url = f"{self.WFS_URL}?{urlencode(params)}"
        
        try:
            print(f"URL: {url}")
            response = requests.get(url, timeout=self.timeout)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Service WFS disponible")
                print(f"Content-Type: {response.headers.get('Content-Type')}")
                print(f"Taille de la réponse: {len(response.content)} bytes")
                return True
            else:
                print(f"❌ Service WFS indisponible (HTTP {response.status_code})")
                return False
                
        except Exception as e:
            print(f"❌ Erreur de connexion: {e}")
            return False
    
    def test_single_layer(self, friendly_name: str, layer_name: str) -> LayerTestResult:
        """
        Teste un layer WFS spécifique.
        
        Args:
            friendly_name: Nom convivial du layer
            layer_name: Nom technique du layer (ex: BDTOPO_V3:batiment)
            
        Returns:
            Résultat du test
        """
        if layer_name is None:
            return LayerTestResult(
                layer_name=friendly_name,
                status='skipped',
                error_message="Layer désactivé (connu comme non disponible)"
            )
        
        bbox = self.TEST_BBOX
        params = {
            "SERVICE": "WFS",
            "VERSION": self.VERSION,
            "REQUEST": "GetFeature",
            "TYPENAME": layer_name,
            "OUTPUTFORMAT": self.OUTPUT_FORMAT,
            "SRSNAME": self.CRS,
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{self.CRS}",
            "COUNT": 100,  # Limite pour test rapide
        }
        
        url = f"{self.WFS_URL}?{urlencode(params)}"
        
        start_time = datetime.now()
        
        try:
            response = requests.get(url, timeout=self.timeout)
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    feature_count = len(data.get('features', []))
                    
                    return LayerTestResult(
                        layer_name=layer_name,
                        status='success',
                        http_code=200,
                        feature_count=feature_count,
                        response_time_ms=elapsed_ms
                    )
                except json.JSONDecodeError as e:
                    return LayerTestResult(
                        layer_name=layer_name,
                        status='error',
                        http_code=200,
                        error_message=f"Réponse JSON invalide: {e}",
                        response_time_ms=elapsed_ms
                    )
            else:
                # Essayer de récupérer le message d'erreur
                error_msg = response.text[:500] if response.text else f"HTTP {response.status_code}"
                
                return LayerTestResult(
                    layer_name=layer_name,
                    status='error',
                    http_code=response.status_code,
                    error_message=error_msg,
                    response_time_ms=elapsed_ms
                )
                
        except requests.Timeout:
            return LayerTestResult(
                layer_name=layer_name,
                status='error',
                error_message=f"Timeout après {self.timeout}s"
            )
        except Exception as e:
            return LayerTestResult(
                layer_name=layer_name,
                status='error',
                error_message=str(e)
            )
    
    def test_all_layers(self) -> None:
        """Teste tous les layers WFS configurés."""
        print("\n" + "=" * 80)
        print("TEST 2: GetFeature - Test de tous les layers")
        print("=" * 80)
        print(f"Bbox de test: {self.TEST_BBOX} (zone de Versailles)")
        print(f"Nombre de layers à tester: {len(self.LAYERS_TO_TEST)}")
        print()
        
        for i, (friendly_name, layer_name) in enumerate(self.LAYERS_TO_TEST.items(), 1):
            print(f"[{i}/{len(self.LAYERS_TO_TEST)}] Test: {friendly_name} ({layer_name})")
            
            result = self.test_single_layer(friendly_name, layer_name)
            self.results.append(result)
            
            # Afficher le résultat
            if result.status == 'success':
                print(f"    ✅ Succès - {result.feature_count} features trouvées "
                      f"({result.response_time_ms:.0f}ms)")
            elif result.status == 'skipped':
                print(f"    ⏭️  Ignoré - {result.error_message}")
            else:
                print(f"    ❌ Erreur - {result.error_message}")
                if result.http_code:
                    print(f"       HTTP {result.http_code}")
            print()
    
    def print_summary(self) -> None:
        """Affiche un résumé des tests."""
        print("=" * 80)
        print("RÉSUMÉ DES TESTS")
        print("=" * 80)
        
        success_count = sum(1 for r in self.results if r.status == 'success')
        error_count = sum(1 for r in self.results if r.status == 'error')
        skipped_count = sum(1 for r in self.results if r.status == 'skipped')
        total_count = len(self.results)
        
        print(f"\nStatistiques:")
        print(f"  Total de layers testés: {total_count}")
        print(f"  ✅ Succès: {success_count}")
        print(f"  ❌ Erreurs: {error_count}")
        print(f"  ⏭️  Ignorés: {skipped_count}")
        
        # Taux de succès
        if total_count > 0:
            success_rate = (success_count / (total_count - skipped_count)) * 100 if (total_count - skipped_count) > 0 else 0
            print(f"  Taux de succès: {success_rate:.1f}%")
        
        # Détails des erreurs
        errors = [r for r in self.results if r.status == 'error']
        if errors:
            print(f"\n❌ Layers en erreur ({len(errors)}):")
            for result in errors:
                print(f"  - {result.layer_name}")
                print(f"    Erreur: {result.error_message}")
        
        # Détails des succès avec statistiques
        successes = [r for r in self.results if r.status == 'success']
        if successes:
            print(f"\n✅ Layers fonctionnels ({len(successes)}):")
            for result in successes:
                print(f"  - {result.layer_name}: {result.feature_count} features "
                      f"({result.response_time_ms:.0f}ms)")
            
            # Temps de réponse moyen
            avg_time = sum(r.response_time_ms for r in successes) / len(successes)
            print(f"\n  Temps de réponse moyen: {avg_time:.0f}ms")
        
        print("\n" + "=" * 80)
    
    def save_report(self, output_file: str = "wfs_test_report.json") -> None:
        """
        Sauvegarde le rapport de test en JSON.
        
        Args:
            output_file: Chemin du fichier de sortie
        """
        report = {
            'test_date': datetime.now().isoformat(),
            'wfs_url': self.WFS_URL,
            'test_bbox': self.TEST_BBOX,
            'results': [
                {
                    'layer_name': r.layer_name,
                    'status': r.status,
                    'http_code': r.http_code,
                    'feature_count': r.feature_count,
                    'error_message': r.error_message,
                    'response_time_ms': r.response_time_ms
                }
                for r in self.results
            ],
            'summary': {
                'total': len(self.results),
                'success': sum(1 for r in self.results if r.status == 'success'),
                'error': sum(1 for r in self.results if r.status == 'error'),
                'skipped': sum(1 for r in self.results if r.status == 'skipped'),
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Rapport sauvegardé: {output_file}")
    
    def run_all_tests(self) -> bool:
        """
        Exécute tous les tests WFS.
        
        Returns:
            True si tous les tests ont réussi, False sinon
        """
        print("\n" + "=" * 80)
        print("TEST DES ENDPOINTS WFS IGN")
        print("=" * 80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Service WFS: {self.WFS_URL}")
        print(f"Version WFS: {self.VERSION}")
        print(f"CRS: {self.CRS}")
        print()
        
        # Test 1: GetCapabilities
        if not self.test_wfs_capabilities():
            print("\n⚠️  Le service WFS n'est pas disponible. Tests annulés.")
            return False
        
        # Test 2: GetFeature pour tous les layers
        self.test_all_layers()
        
        # Résumé
        self.print_summary()
        
        # Sauvegarde du rapport
        self.save_report()
        
        # Retourner True seulement si tous les layers non-ignorés ont réussi
        error_count = sum(1 for r in self.results if r.status == 'error')
        return error_count == 0


def main():
    """Point d'entrée principal."""
    tester = WFSEndpointTester(timeout=30)
    
    success = tester.run_all_tests()
    
    # Code de sortie pour intégration CI/CD
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
