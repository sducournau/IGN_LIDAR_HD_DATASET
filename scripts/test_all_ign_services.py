#!/usr/bin/env python3
"""
Script de test complet des services IGN utilisés dans le projet

Ce script teste:
1. WFS (BD TOPO V3) - Données vectorielles topographiques
2. WMS (RGE ALTI / LiDAR HD MNT) - Modèles numériques de terrain
3. WFS (BD Forêt V2) - Données forestières
4. WMS (Orthophotos RGB) - Images aériennes

Utilise les mêmes endpoints que le code de production.
"""

import requests
from urllib.parse import urlencode
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from datetime import datetime
import sys


@dataclass
class ServiceTestResult:
    """Résultat du test d'un service IGN."""
    service_name: str
    service_type: str  # 'WFS' ou 'WMS'
    endpoint: str
    status: str  # 'success', 'error'
    http_code: Optional[int] = None
    details: Optional[str] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None


class IGNServiceTester:
    """Testeur complet des services IGN."""
    
    # ========================================================================
    # Configuration des services
    # ========================================================================
    
    # WFS - BD TOPO V3
    WFS_BDTOPO_URL = "https://data.geopf.fr/wfs"
    WFS_VERSION = "2.0.0"
    
    # WMS - RGE ALTI / LiDAR HD MNT
    WMS_DTM_URL = "https://data.geopf.fr/wms-r/wms"
    WMS_VERSION = "1.3.0"
    
    # WMS - Orthophotos RGB
    WMS_RGB_URL = "https://data.geopf.fr/wms-r"
    
    # Test parameters
    TEST_BBOX_LAMBERT93 = (650000, 6860000, 651000, 6861000)  # Versailles
    TEST_CRS = "EPSG:2154"
    TEST_TIMEOUT = 30
    
    def __init__(self):
        """Initialise le testeur."""
        self.results: List[ServiceTestResult] = []
    
    # ========================================================================
    # Tests WFS
    # ========================================================================
    
    def test_wfs_capabilities(self) -> ServiceTestResult:
        """Teste GetCapabilities WFS pour BD TOPO."""
        print("\n" + "=" * 80)
        print("TEST WFS: GetCapabilities - BD TOPO V3")
        print("=" * 80)
        
        params = {
            "SERVICE": "WFS",
            "REQUEST": "GetCapabilities"
        }
        
        url = f"{self.WFS_BDTOPO_URL}?{urlencode(params)}"
        print(f"URL: {url}")
        
        start_time = datetime.now()
        
        try:
            response = requests.get(url, timeout=self.TEST_TIMEOUT)
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                result = ServiceTestResult(
                    service_name="BD TOPO V3 WFS",
                    service_type="WFS",
                    endpoint=self.WFS_BDTOPO_URL,
                    status='success',
                    http_code=200,
                    details=f"Capabilities document: {len(response.content)} bytes",
                    response_time_ms=elapsed_ms
                )
                print(f"✅ Service WFS BD TOPO disponible ({elapsed_ms:.0f}ms)")
                print(f"   Taille: {len(response.content)} bytes")
            else:
                result = ServiceTestResult(
                    service_name="BD TOPO V3 WFS",
                    service_type="WFS",
                    endpoint=self.WFS_BDTOPO_URL,
                    status='error',
                    http_code=response.status_code,
                    error_message=f"HTTP {response.status_code}",
                    response_time_ms=elapsed_ms
                )
                print(f"❌ Service WFS BD TOPO indisponible (HTTP {response.status_code})")
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return ServiceTestResult(
                service_name="BD TOPO V3 WFS",
                service_type="WFS",
                endpoint=self.WFS_BDTOPO_URL,
                status='error',
                error_message=str(e)
            )
    
    def test_wfs_layer_sample(self) -> ServiceTestResult:
        """Teste un layer WFS spécifique (bâtiments)."""
        print("\n" + "=" * 80)
        print("TEST WFS: GetFeature - Exemple avec BDTOPO_V3:batiment")
        print("=" * 80)
        
        bbox = self.TEST_BBOX_LAMBERT93
        params = {
            "SERVICE": "WFS",
            "VERSION": self.WFS_VERSION,
            "REQUEST": "GetFeature",
            "TYPENAME": "BDTOPO_V3:batiment",
            "OUTPUTFORMAT": "application/json",
            "SRSNAME": self.TEST_CRS,
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{self.TEST_CRS}",
            "COUNT": 10,
        }
        
        url = f"{self.WFS_BDTOPO_URL}?{urlencode(params)}"
        print(f"Bbox test: {bbox}")
        
        start_time = datetime.now()
        
        try:
            response = requests.get(url, timeout=self.TEST_TIMEOUT)
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                feature_count = len(data.get('features', []))
                
                result = ServiceTestResult(
                    service_name="BD TOPO Buildings Layer",
                    service_type="WFS",
                    endpoint=self.WFS_BDTOPO_URL,
                    status='success',
                    http_code=200,
                    details=f"{feature_count} features récupérées",
                    response_time_ms=elapsed_ms
                )
                print(f"✅ Layer BDTOPO_V3:batiment accessible ({elapsed_ms:.0f}ms)")
                print(f"   Features: {feature_count}")
            else:
                result = ServiceTestResult(
                    service_name="BD TOPO Buildings Layer",
                    service_type="WFS",
                    endpoint=self.WFS_BDTOPO_URL,
                    status='error',
                    http_code=response.status_code,
                    error_message=response.text[:200],
                    response_time_ms=elapsed_ms
                )
                print(f"❌ Erreur HTTP {response.status_code}")
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return ServiceTestResult(
                service_name="BD TOPO Buildings Layer",
                service_type="WFS",
                endpoint=self.WFS_BDTOPO_URL,
                status='error',
                error_message=str(e)
            )
    
    # ========================================================================
    # Tests WMS - DTM (RGE ALTI / LiDAR HD MNT)
    # ========================================================================
    
    def test_wms_dtm_capabilities(self) -> ServiceTestResult:
        """Teste GetCapabilities WMS pour les MNT."""
        print("\n" + "=" * 80)
        print("TEST WMS: GetCapabilities - RGE ALTI / LiDAR HD MNT")
        print("=" * 80)
        
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetCapabilities"
        }
        
        url = f"{self.WMS_DTM_URL}?{urlencode(params)}"
        print(f"URL: {url}")
        
        start_time = datetime.now()
        
        try:
            response = requests.get(url, timeout=self.TEST_TIMEOUT)
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                # Vérifier que les layers attendus sont présents
                content = response.text
                has_lidar_hd = "LIDAR-HD_MNT" in content
                has_rge_alti = "ELEVATION.ELEVATIONGRIDCOVERAGE" in content
                
                details = []
                if has_lidar_hd:
                    details.append("LiDAR HD MNT layer présent")
                if has_rge_alti:
                    details.append("RGE ALTI layer présent")
                
                result = ServiceTestResult(
                    service_name="DTM WMS (RGE ALTI / LiDAR HD)",
                    service_type="WMS",
                    endpoint=self.WMS_DTM_URL,
                    status='success',
                    http_code=200,
                    details="; ".join(details) if details else "Capabilities OK",
                    response_time_ms=elapsed_ms
                )
                print(f"✅ Service WMS DTM disponible ({elapsed_ms:.0f}ms)")
                for detail in details:
                    print(f"   - {detail}")
            else:
                result = ServiceTestResult(
                    service_name="DTM WMS (RGE ALTI / LiDAR HD)",
                    service_type="WMS",
                    endpoint=self.WMS_DTM_URL,
                    status='error',
                    http_code=response.status_code,
                    error_message=f"HTTP {response.status_code}",
                    response_time_ms=elapsed_ms
                )
                print(f"❌ Service WMS DTM indisponible (HTTP {response.status_code})")
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return ServiceTestResult(
                service_name="DTM WMS (RGE ALTI / LiDAR HD)",
                service_type="WMS",
                endpoint=self.WMS_DTM_URL,
                status='error',
                error_message=str(e)
            )
    
    def test_wms_dtm_getmap(self) -> ServiceTestResult:
        """Teste GetMap WMS pour récupérer un MNT."""
        print("\n" + "=" * 80)
        print("TEST WMS: GetMap - Téléchargement d'un échantillon de MNT")
        print("=" * 80)
        
        bbox = self.TEST_BBOX_LAMBERT93
        width, height = 100, 100  # Petite taille pour test rapide
        
        # Test avec RGE ALTI (disponibilité plus large)
        params = {
            "SERVICE": "WMS",
            "VERSION": self.WMS_VERSION,
            "REQUEST": "GetMap",
            "LAYERS": "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES",  # RGE ALTI
            "STYLES": "",
            "FORMAT": "image/geotiff",
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "WIDTH": width,
            "HEIGHT": height,
            "CRS": self.TEST_CRS,
        }
        
        url = f"{self.WMS_DTM_URL}?{urlencode(params)}"
        print(f"Bbox test: {bbox}")
        print(f"Résolution: {width}x{height}")
        print(f"Layer: RGE ALTI")
        
        start_time = datetime.now()
        
        try:
            response = requests.get(url, timeout=self.TEST_TIMEOUT)
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                is_geotiff = 'tiff' in content_type.lower() or 'image' in content_type.lower()
                
                result = ServiceTestResult(
                    service_name="DTM WMS GetMap",
                    service_type="WMS",
                    endpoint=self.WMS_DTM_URL,
                    status='success',
                    http_code=200,
                    details=f"GeoTIFF {len(response.content)} bytes, Content-Type: {content_type}",
                    response_time_ms=elapsed_ms
                )
                print(f"✅ GetMap réussi ({elapsed_ms:.0f}ms)")
                print(f"   Taille: {len(response.content)} bytes")
                print(f"   Content-Type: {content_type}")
                
                if not is_geotiff:
                    print(f"   ⚠️  Warning: Content-Type inattendu (attendu: image/tiff)")
            else:
                result = ServiceTestResult(
                    service_name="DTM WMS GetMap",
                    service_type="WMS",
                    endpoint=self.WMS_DTM_URL,
                    status='error',
                    http_code=response.status_code,
                    error_message=response.text[:200],
                    response_time_ms=elapsed_ms
                )
                print(f"❌ Erreur HTTP {response.status_code}")
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return ServiceTestResult(
                service_name="DTM WMS GetMap",
                service_type="WMS",
                endpoint=self.WMS_DTM_URL,
                status='error',
                error_message=str(e)
            )
    
    # ========================================================================
    # Tests WMS - Orthophotos RGB
    # ========================================================================
    
    def test_wms_rgb_capabilities(self) -> ServiceTestResult:
        """Teste GetCapabilities WMS pour les orthophotos."""
        print("\n" + "=" * 80)
        print("TEST WMS: GetCapabilities - Orthophotos RGB")
        print("=" * 80)
        
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetCapabilities"
        }
        
        url = f"{self.WMS_RGB_URL}?{urlencode(params)}"
        print(f"URL: {url}")
        
        start_time = datetime.now()
        
        try:
            response = requests.get(url, timeout=self.TEST_TIMEOUT)
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                content = response.text
                has_ortho = "ORTHOIMAGERY.ORTHOPHOTOS" in content
                
                result = ServiceTestResult(
                    service_name="RGB Orthophotos WMS",
                    service_type="WMS",
                    endpoint=self.WMS_RGB_URL,
                    status='success',
                    http_code=200,
                    details="Orthophotos layer présent" if has_ortho else "Capabilities OK",
                    response_time_ms=elapsed_ms
                )
                print(f"✅ Service WMS Orthophotos disponible ({elapsed_ms:.0f}ms)")
                if has_ortho:
                    print(f"   - Layer HR.ORTHOIMAGERY.ORTHOPHOTOS présent")
            else:
                result = ServiceTestResult(
                    service_name="RGB Orthophotos WMS",
                    service_type="WMS",
                    endpoint=self.WMS_RGB_URL,
                    status='error',
                    http_code=response.status_code,
                    error_message=f"HTTP {response.status_code}",
                    response_time_ms=elapsed_ms
                )
                print(f"❌ Service WMS Orthophotos indisponible (HTTP {response.status_code})")
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return ServiceTestResult(
                service_name="RGB Orthophotos WMS",
                service_type="WMS",
                endpoint=self.WMS_RGB_URL,
                status='error',
                error_message=str(e)
            )
    
    # ========================================================================
    # Tests WFS - BD Forêt
    # ========================================================================
    
    def test_wfs_bdforet(self) -> ServiceTestResult:
        """Teste WFS BD Forêt V2."""
        print("\n" + "=" * 80)
        print("TEST WFS: GetFeature - BD Forêt V2")
        print("=" * 80)
        
        bbox = self.TEST_BBOX_LAMBERT93
        params = {
            "SERVICE": "WFS",
            "VERSION": self.WFS_VERSION,
            "REQUEST": "GetFeature",
            "TYPENAME": "BDFORET_V2:formation_vegetale",
            "OUTPUTFORMAT": "application/json",
            "SRSNAME": self.TEST_CRS,
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{self.TEST_CRS}",
            "COUNT": 10,
        }
        
        url = f"{self.WFS_BDTOPO_URL}?{urlencode(params)}"
        print(f"Bbox test: {bbox}")
        
        start_time = datetime.now()
        
        try:
            response = requests.get(url, timeout=self.TEST_TIMEOUT)
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                feature_count = len(data.get('features', []))
                
                result = ServiceTestResult(
                    service_name="BD Forêt V2 WFS",
                    service_type="WFS",
                    endpoint=self.WFS_BDTOPO_URL,
                    status='success',
                    http_code=200,
                    details=f"{feature_count} formations végétales récupérées",
                    response_time_ms=elapsed_ms
                )
                print(f"✅ Layer BDFORET_V2:formation_vegetale accessible ({elapsed_ms:.0f}ms)")
                print(f"   Features: {feature_count}")
            else:
                result = ServiceTestResult(
                    service_name="BD Forêt V2 WFS",
                    service_type="WFS",
                    endpoint=self.WFS_BDTOPO_URL,
                    status='error',
                    http_code=response.status_code,
                    error_message=response.text[:200],
                    response_time_ms=elapsed_ms
                )
                print(f"❌ Erreur HTTP {response.status_code}")
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return ServiceTestResult(
                service_name="BD Forêt V2 WFS",
                service_type="WFS",
                endpoint=self.WFS_BDTOPO_URL,
                status='error',
                error_message=str(e)
            )
    
    # ========================================================================
    # Exécution et rapport
    # ========================================================================
    
    def run_all_tests(self) -> bool:
        """Exécute tous les tests."""
        print("\n" + "=" * 80)
        print("TEST COMPLET DES SERVICES IGN GÉOPLATEFORME")
        print("=" * 80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Zone de test: {self.TEST_BBOX_LAMBERT93} (Versailles, Lambert 93)")
        print()
        
        # Test WFS BD TOPO
        self.results.append(self.test_wfs_capabilities())
        self.results.append(self.test_wfs_layer_sample())
        
        # Test WMS DTM
        self.results.append(self.test_wms_dtm_capabilities())
        self.results.append(self.test_wms_dtm_getmap())
        
        # Test WMS RGB
        self.results.append(self.test_wms_rgb_capabilities())
        
        # Test WFS BD Forêt
        self.results.append(self.test_wfs_bdforet())
        
        # Résumé
        self.print_summary()
        
        # Sauvegarde du rapport
        self.save_report()
        
        # Retourner True seulement si tous les tests ont réussi
        error_count = sum(1 for r in self.results if r.status == 'error')
        return error_count == 0
    
    def print_summary(self) -> None:
        """Affiche un résumé des tests."""
        print("\n" + "=" * 80)
        print("RÉSUMÉ GLOBAL DES TESTS")
        print("=" * 80)
        
        success_count = sum(1 for r in self.results if r.status == 'success')
        error_count = sum(1 for r in self.results if r.status == 'error')
        total_count = len(self.results)
        
        print(f"\nStatistiques globales:")
        print(f"  Total de services testés: {total_count}")
        print(f"  ✅ Succès: {success_count}")
        print(f"  ❌ Erreurs: {error_count}")
        
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
            print(f"  Taux de succès: {success_rate:.1f}%")
        
        # Par type de service
        wfs_results = [r for r in self.results if r.service_type == 'WFS']
        wms_results = [r for r in self.results if r.service_type == 'WMS']
        
        print(f"\n📊 Services WFS: {sum(1 for r in wfs_results if r.status == 'success')}/{len(wfs_results)} OK")
        print(f"📊 Services WMS: {sum(1 for r in wms_results if r.status == 'success')}/{len(wms_results)} OK")
        
        # Détails des erreurs
        errors = [r for r in self.results if r.status == 'error']
        if errors:
            print(f"\n❌ Services en erreur ({len(errors)}):")
            for result in errors:
                print(f"  - {result.service_name}")
                print(f"    Endpoint: {result.endpoint}")
                print(f"    Erreur: {result.error_message}")
        
        # Services fonctionnels
        successes = [r for r in self.results if r.status == 'success']
        if successes:
            print(f"\n✅ Services fonctionnels ({len(successes)}):")
            for result in successes:
                time_str = f" ({result.response_time_ms:.0f}ms)" if result.response_time_ms else ""
                print(f"  - {result.service_name}{time_str}")
                if result.details:
                    print(f"    {result.details}")
        
        print("\n" + "=" * 80)
    
    def save_report(self, output_file: str = "ign_services_test_report.json") -> None:
        """Sauvegarde le rapport de test en JSON."""
        report = {
            'test_date': datetime.now().isoformat(),
            'test_bbox': self.TEST_BBOX_LAMBERT93,
            'results': [
                {
                    'service_name': r.service_name,
                    'service_type': r.service_type,
                    'endpoint': r.endpoint,
                    'status': r.status,
                    'http_code': r.http_code,
                    'details': r.details,
                    'error_message': r.error_message,
                    'response_time_ms': r.response_time_ms
                }
                for r in self.results
            ],
            'summary': {
                'total': len(self.results),
                'success': sum(1 for r in self.results if r.status == 'success'),
                'error': sum(1 for r in self.results if r.status == 'error'),
                'wfs_success': sum(1 for r in self.results if r.service_type == 'WFS' and r.status == 'success'),
                'wms_success': sum(1 for r in self.results if r.service_type == 'WMS' and r.status == 'success'),
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Rapport détaillé sauvegardé: {output_file}")


def main():
    """Point d'entrée principal."""
    tester = IGNServiceTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
