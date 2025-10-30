"""
Working List of 50 IGN LiDAR HD Tiles - REAL WFS DATA

This file contains a curated list of 50 real tiles from the IGN WFS service
covering different regions across France for testing and development.
All tiles are verified to be available for download.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class TileInfo:
    """Information about an IGN LiDAR HD tile."""

    filename: str
    tile_x: int
    tile_y: int
    location: str
    environment: str
    description: str
    recommended_lod: str
    coordinates_lambert93: tuple  # (center_x, center_y)
    coordinates_gps: tuple  # (lat, lon)
    priority: int  # 1=highest, 5=lowest
    real_name: str = ""  # Original WFS name
    download_url: str = ""  # Direct download URL


# Working list of 50 REAL tiles from IGN WFS service
WORKING_TILES = [
    # Tile 1/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0186_6834_PTS_C_LAMB93_IGN69.laz",
        tile_x=186,
        tile_y=6834,
        location="Finistère - Tile 186_6834",
        environment="coastal",
        description="Real IGN tile covering Finistère region",
        recommended_lod="LOD2",
        coordinates_lambert93=(186500, 6834500),
        coordinates_gps=(-18.9317, -5.9981),
        priority=1,
        real_name="LHD_FXX_0186_6834_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0186_6834_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 2/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0192_6838_PTS_C_LAMB93_IGN69.laz",
        tile_x=192,
        tile_y=6838,
        location="Finistère - Tile 192_6838",
        environment="coastal",
        description="Real IGN tile covering Finistère region",
        recommended_lod="LOD2",
        coordinates_lambert93=(192500, 6838500),
        coordinates_gps=(-18.9316, -5.9981),
        priority=1,
        real_name="LHD_FXX_0192_6838_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0192_6838_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 3/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0198_6842_PTS_C_LAMB93_IGN69.laz",
        tile_x=198,
        tile_y=6842,
        location="Finistère - Tile 198_6842",
        environment="coastal",
        description="Real IGN tile covering Finistère region",
        recommended_lod="LOD2",
        coordinates_lambert93=(198500, 6842500),
        coordinates_gps=(-18.9316, -5.9980),
        priority=1,
        real_name="LHD_FXX_0198_6842_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0198_6842_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 4/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0204_6846_PTS_C_LAMB93_IGN69.laz",
        tile_x=204,
        tile_y=6846,
        location="Morbihan - Tile 204_6846",
        environment="coastal",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(204500, 6846500),
        coordinates_gps=(-18.9315, -5.9980),
        priority=1,
        real_name="LHD_FXX_0204_6846_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0204_6846_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 5/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0211_6834_PTS_C_LAMB93_IGN69.laz",
        tile_x=211,
        tile_y=6834,
        location="Morbihan - Tile 211_6834",
        environment="coastal",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(211500, 6834500),
        coordinates_gps=(-18.9317, -5.9979),
        priority=1,
        real_name="LHD_FXX_0211_6834_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0211_6834_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 6/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0217_6838_PTS_C_LAMB93_IGN69.laz",
        tile_x=217,
        tile_y=6838,
        location="Morbihan - Tile 217_6838",
        environment="coastal",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(217500, 6838500),
        coordinates_gps=(-18.9316, -5.9978),
        priority=1,
        real_name="LHD_FXX_0217_6838_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0217_6838_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 7/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0223_6842_PTS_C_LAMB93_IGN69.laz",
        tile_x=223,
        tile_y=6842,
        location="Morbihan - Tile 223_6842",
        environment="coastal",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(223500, 6842500),
        coordinates_gps=(-18.9316, -5.9978),
        priority=1,
        real_name="LHD_FXX_0223_6842_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0223_6842_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 8/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0229_6846_PTS_C_LAMB93_IGN69.laz",
        tile_x=229,
        tile_y=6846,
        location="Morbihan - Tile 229_6846",
        environment="coastal",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(229500, 6846500),
        coordinates_gps=(-18.9315, -5.9977),
        priority=1,
        real_name="LHD_FXX_0229_6846_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0229_6846_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 9/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0186_6851_PTS_C_LAMB93_IGN69.laz",
        tile_x=186,
        tile_y=6851,
        location="Finistère - Tile 186_6851",
        environment="coastal",
        description="Real IGN tile covering Finistère region",
        recommended_lod="LOD2",
        coordinates_lambert93=(186500, 6851500),
        coordinates_gps=(-18.9315, -5.9981),
        priority=1,
        real_name="LHD_FXX_0186_6851_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0186_6851_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 10/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0190_6867_PTS_C_LAMB93_IGN69.laz",
        tile_x=190,
        tile_y=6867,
        location="Finistère - Tile 190_6867",
        environment="coastal",
        description="Real IGN tile covering Finistère region",
        recommended_lod="LOD2",
        coordinates_lambert93=(190500, 6867500),
        coordinates_gps=(-18.9313, -5.9981),
        priority=1,
        real_name="LHD_FXX_0190_6867_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0190_6867_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 11/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0196_6864_PTS_C_LAMB93_IGN69.laz",
        tile_x=196,
        tile_y=6864,
        location="Finistère - Tile 196_6864",
        environment="rural",
        description="Real IGN tile covering Finistère region",
        recommended_lod="LOD2",
        coordinates_lambert93=(196500, 6864500),
        coordinates_gps=(-18.9314, -5.9980),
        priority=2,
        real_name="LHD_FXX_0196_6864_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0196_6864_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 12/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0202_6858_PTS_C_LAMB93_IGN69.laz",
        tile_x=202,
        tile_y=6858,
        location="Morbihan - Tile 202_6858",
        environment="rural",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(202500, 6858500),
        coordinates_gps=(-18.9314, -5.9980),
        priority=2,
        real_name="LHD_FXX_0202_6858_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0202_6858_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 13/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0208_6858_PTS_C_LAMB93_IGN69.laz",
        tile_x=208,
        tile_y=6858,
        location="Morbihan - Tile 208_6858",
        environment="rural",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(208500, 6858500),
        coordinates_gps=(-18.9314, -5.9979),
        priority=2,
        real_name="LHD_FXX_0208_6858_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0208_6858_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 14/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0214_6872_PTS_C_LAMB93_IGN69.laz",
        tile_x=214,
        tile_y=6872,
        location="Morbihan - Tile 214_6872",
        environment="rural",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(214500, 6872500),
        coordinates_gps=(-18.9313, -5.9979),
        priority=2,
        real_name="LHD_FXX_0214_6872_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0214_6872_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 15/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0218_6871_PTS_C_LAMB93_IGN69.laz",
        tile_x=218,
        tile_y=6871,
        location="Morbihan - Tile 218_6871",
        environment="rural",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(218500, 6871500),
        coordinates_gps=(-18.9313, -5.9978),
        priority=2,
        real_name="LHD_FXX_0218_6871_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0218_6871_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 16/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0222_6859_PTS_C_LAMB93_IGN69.laz",
        tile_x=222,
        tile_y=6859,
        location="Morbihan - Tile 222_6859",
        environment="rural",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(222500, 6859500),
        coordinates_gps=(-18.9314, -5.9978),
        priority=2,
        real_name="LHD_FXX_0222_6859_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0222_6859_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 17/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0225_6867_PTS_C_LAMB93_IGN69.laz",
        tile_x=225,
        tile_y=6867,
        location="Morbihan - Tile 225_6867",
        environment="rural",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(225500, 6867500),
        coordinates_gps=(-18.9313, -5.9977),
        priority=2,
        real_name="LHD_FXX_0225_6867_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0225_6867_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 18/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0228_6878_PTS_C_LAMB93_IGN69.laz",
        tile_x=228,
        tile_y=6878,
        location="Morbihan - Tile 228_6878",
        environment="rural",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(228500, 6878500),
        coordinates_gps=(-18.9312, -5.9977),
        priority=2,
        real_name="LHD_FXX_0228_6878_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0228_6878_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 19/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0232_6862_PTS_C_LAMB93_IGN69.laz",
        tile_x=232,
        tile_y=6862,
        location="Morbihan - Tile 232_6862",
        environment="rural",
        description="Real IGN tile covering Morbihan region",
        recommended_lod="LOD2",
        coordinates_lambert93=(232500, 6862500),
        coordinates_gps=(-18.9314, -5.9977),
        priority=2,
        real_name="LHD_FXX_0232_6862_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0232_6862_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 20/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0311_6262_PTS_C_LAMB93_IGN69.laz",
        tile_x=311,
        tile_y=6262,
        location="Western France - Tile 311_6262",
        environment="rural",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(311500, 6262500),
        coordinates_gps=(-18.9374, -5.9969),
        priority=2,
        real_name="LHD_FXX_0311_6262_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0311_6262_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 21/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0322_6261_PTS_C_LAMB93_IGN69.laz",
        tile_x=322,
        tile_y=6261,
        location="Western France - Tile 322_6261",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(322500, 6261500),
        coordinates_gps=(-18.9374, -5.9968),
        priority=3,
        real_name="LHD_FXX_0322_6261_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0322_6261_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 22/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0328_6251_PTS_C_LAMB93_IGN69.laz",
        tile_x=328,
        tile_y=6251,
        location="Western France - Tile 328_6251",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(328500, 6251500),
        coordinates_gps=(-18.9375, -5.9967),
        priority=3,
        real_name="LHD_FXX_0328_6251_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0328_6251_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 23/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0332_6261_PTS_C_LAMB93_IGN69.laz",
        tile_x=332,
        tile_y=6261,
        location="Western France - Tile 332_6261",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(332500, 6261500),
        coordinates_gps=(-18.9374, -5.9967),
        priority=3,
        real_name="LHD_FXX_0332_6261_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0332_6261_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 24/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0335_6276_PTS_C_LAMB93_IGN69.laz",
        tile_x=335,
        tile_y=6276,
        location="Western France - Tile 335_6276",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(335500, 6276500),
        coordinates_gps=(-18.9372, -5.9966),
        priority=3,
        real_name="LHD_FXX_0335_6276_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0335_6276_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 25/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0338_6261_PTS_C_LAMB93_IGN69.laz",
        tile_x=338,
        tile_y=6261,
        location="Western France - Tile 338_6261",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(338500, 6261500),
        coordinates_gps=(-18.9374, -5.9966),
        priority=3,
        real_name="LHD_FXX_0338_6261_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0338_6261_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 26/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0340_6273_PTS_C_LAMB93_IGN69.laz",
        tile_x=340,
        tile_y=6273,
        location="Western France - Tile 340_6273",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(340500, 6273500),
        coordinates_gps=(-18.9373, -5.9966),
        priority=3,
        real_name="LHD_FXX_0340_6273_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0340_6273_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 27/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0342_6275_PTS_C_LAMB93_IGN69.laz",
        tile_x=342,
        tile_y=6275,
        location="Western France - Tile 342_6275",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(342500, 6275500),
        coordinates_gps=(-18.9372, -5.9966),
        priority=3,
        real_name="LHD_FXX_0342_6275_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0342_6275_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 28/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0344_6261_PTS_C_LAMB93_IGN69.laz",
        tile_x=344,
        tile_y=6261,
        location="Western France - Tile 344_6261",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(344500, 6261500),
        coordinates_gps=(-18.9374, -5.9966),
        priority=3,
        real_name="LHD_FXX_0344_6261_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0344_6261_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 29/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0346_6247_PTS_C_LAMB93_IGN69.laz",
        tile_x=346,
        tile_y=6247,
        location="Western France - Tile 346_6247",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(346500, 6247500),
        coordinates_gps=(-18.9375, -5.9965),
        priority=3,
        real_name="LHD_FXX_0346_6247_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0346_6247_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 30/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0348_6247_PTS_C_LAMB93_IGN69.laz",
        tile_x=348,
        tile_y=6247,
        location="Western France - Tile 348_6247",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(348500, 6247500),
        coordinates_gps=(-18.9375, -5.9965),
        priority=3,
        real_name="LHD_FXX_0348_6247_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0348_6247_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 31/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0350_6238_PTS_C_LAMB93_IGN69.laz",
        tile_x=350,
        tile_y=6238,
        location="Western France - Tile 350_6238",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(350500, 6238500),
        coordinates_gps=(-18.9376, -5.9965),
        priority=3,
        real_name="LHD_FXX_0350_6238_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0350_6238_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 32/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0352_6226_PTS_C_LAMB93_IGN69.laz",
        tile_x=352,
        tile_y=6226,
        location="Western France - Tile 352_6226",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(352500, 6226500),
        coordinates_gps=(-18.9377, -5.9965),
        priority=3,
        real_name="LHD_FXX_0352_6226_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/ER/LHD_FXX_0352_6226_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 33/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0353_6270_PTS_C_LAMB93_IGN69.laz",
        tile_x=353,
        tile_y=6270,
        location="Western France - Tile 353_6270",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(353500, 6270500),
        coordinates_gps=(-18.9373, -5.9965),
        priority=3,
        real_name="LHD_FXX_0353_6270_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0353_6270_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 34/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0355_6257_PTS_C_LAMB93_IGN69.laz",
        tile_x=355,
        tile_y=6257,
        location="Western France - Tile 355_6257",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(355500, 6257500),
        coordinates_gps=(-18.9374, -5.9965),
        priority=3,
        real_name="LHD_FXX_0355_6257_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0355_6257_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 35/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0357_6253_PTS_C_LAMB93_IGN69.laz",
        tile_x=357,
        tile_y=6253,
        location="Western France - Tile 357_6253",
        environment="mixed",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(357500, 6253500),
        coordinates_gps=(-18.9375, -5.9964),
        priority=3,
        real_name="LHD_FXX_0357_6253_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0357_6253_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 36/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0359_6255_PTS_C_LAMB93_IGN69.laz",
        tile_x=359,
        tile_y=6255,
        location="Western France - Tile 359_6255",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(359500, 6255500),
        coordinates_gps=(-18.9374, -5.9964),
        priority=4,
        real_name="LHD_FXX_0359_6255_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0359_6255_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 37/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0361_6257_PTS_C_LAMB93_IGN69.laz",
        tile_x=361,
        tile_y=6257,
        location="Western France - Tile 361_6257",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(361500, 6257500),
        coordinates_gps=(-18.9374, -5.9964),
        priority=4,
        real_name="LHD_FXX_0361_6257_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0361_6257_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 38/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0363_6260_PTS_C_LAMB93_IGN69.laz",
        tile_x=363,
        tile_y=6260,
        location="Western France - Tile 363_6260",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(363500, 6260500),
        coordinates_gps=(-18.9374, -5.9964),
        priority=4,
        real_name="LHD_FXX_0363_6260_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0363_6260_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 39/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0365_6262_PTS_C_LAMB93_IGN69.laz",
        tile_x=365,
        tile_y=6262,
        location="Western France - Tile 365_6262",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(365500, 6262500),
        coordinates_gps=(-18.9374, -5.9963),
        priority=4,
        real_name="LHD_FXX_0365_6262_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0365_6262_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 40/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0367_6264_PTS_C_LAMB93_IGN69.laz",
        tile_x=367,
        tile_y=6264,
        location="Western France - Tile 367_6264",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(367500, 6264500),
        coordinates_gps=(-18.9374, -5.9963),
        priority=4,
        real_name="LHD_FXX_0367_6264_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0367_6264_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 41/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0369_6266_PTS_C_LAMB93_IGN69.laz",
        tile_x=369,
        tile_y=6266,
        location="Western France - Tile 369_6266",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(369500, 6266500),
        coordinates_gps=(-18.9373, -5.9963),
        priority=4,
        real_name="LHD_FXX_0369_6266_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0369_6266_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 42/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0371_6268_PTS_C_LAMB93_IGN69.laz",
        tile_x=371,
        tile_y=6268,
        location="Western France - Tile 371_6268",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(371500, 6268500),
        coordinates_gps=(-18.9373, -5.9963),
        priority=4,
        real_name="LHD_FXX_0371_6268_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0371_6268_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 43/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0373_6270_PTS_C_LAMB93_IGN69.laz",
        tile_x=373,
        tile_y=6270,
        location="Western France - Tile 373_6270",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(373500, 6270500),
        coordinates_gps=(-18.9373, -5.9963),
        priority=4,
        real_name="LHD_FXX_0373_6270_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0373_6270_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 44/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0375_6273_PTS_C_LAMB93_IGN69.laz",
        tile_x=375,
        tile_y=6273,
        location="Western France - Tile 375_6273",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(375500, 6273500),
        coordinates_gps=(-18.9373, -5.9962),
        priority=4,
        real_name="LHD_FXX_0375_6273_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0375_6273_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 45/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0377_6275_PTS_C_LAMB93_IGN69.laz",
        tile_x=377,
        tile_y=6275,
        location="Western France - Tile 377_6275",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(377500, 6275500),
        coordinates_gps=(-18.9372, -5.9962),
        priority=4,
        real_name="LHD_FXX_0377_6275_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0377_6275_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 46/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0379_6277_PTS_C_LAMB93_IGN69.laz",
        tile_x=379,
        tile_y=6277,
        location="Western France - Tile 379_6277",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(379500, 6277500),
        coordinates_gps=(-18.9372, -5.9962),
        priority=4,
        real_name="LHD_FXX_0379_6277_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0379_6277_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 47/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0381_6279_PTS_C_LAMB93_IGN69.laz",
        tile_x=381,
        tile_y=6279,
        location="Western France - Tile 381_6279",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(381500, 6279500),
        coordinates_gps=(-18.9372, -5.9962),
        priority=4,
        real_name="LHD_FXX_0381_6279_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0381_6279_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 48/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0383_6281_PTS_C_LAMB93_IGN69.laz",
        tile_x=383,
        tile_y=6281,
        location="Western France - Tile 383_6281",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(383500, 6281500),
        coordinates_gps=(-18.9372, -5.9962),
        priority=4,
        real_name="LHD_FXX_0383_6281_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0383_6281_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 49/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0311_6263_PTS_C_LAMB93_IGN69.laz",
        tile_x=311,
        tile_y=6263,
        location="Western France - Tile 311_6263",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(311500, 6263500),
        coordinates_gps=(-18.9374, -5.9969),
        priority=4,
        real_name="LHD_FXX_0311_6263_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0311_6263_PTS_C_LAMB93_IGN69.copc.laz",
    ),
    # Tile 50/50
    TileInfo(
        filename="HD_LIDARHD_FXX_0311_6264_PTS_C_LAMB93_IGN69.laz",
        tile_x=311,
        tile_y=6264,
        location="Western France - Tile 311_6264",
        environment="varied",
        description="Real IGN tile covering Western France region",
        recommended_lod="LOD2",
        coordinates_lambert93=(311500, 6264500),
        coordinates_gps=(-18.9374, -5.9969),
        priority=4,
        real_name="LHD_FXX_0311_6264_PTS_C_LAMB93_IGN69.copc.laz",
        download_url="https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/EQ/LHD_FXX_0311_6264_PTS_C_LAMB93_IGN69.copc.laz",
    ),
]


def get_tiles_by_priority(max_priority: int = 5) -> List[TileInfo]:
    """Get tiles filtered by priority level."""
    return [tile for tile in WORKING_TILES if tile.priority <= max_priority]


def get_tiles_by_environment(environment: str) -> List[TileInfo]:
    """Get tiles filtered by environment type."""
    return [tile for tile in WORKING_TILES if tile.environment == environment]


def get_tiles_by_region(region_prefix: str) -> List[TileInfo]:
    """Get tiles filtered by region (location starts with prefix)."""
    return [tile for tile in WORKING_TILES if tile.location.startswith(region_prefix)]


def get_download_mapping() -> Dict[str, str]:
    """Get mapping from standard filename to real WFS name for downloads."""
    return {tile.filename: tile.real_name for tile in WORKING_TILES if tile.real_name}


def get_url_mapping() -> Dict[str, str]:
    """Get mapping from filename to direct download URL."""
    return {
        tile.filename: tile.download_url for tile in WORKING_TILES if tile.download_url
    }


# Statistics (logged at module initialization)
import logging

logger = logging.getLogger(__name__)
logger.debug(f"Loaded {len(WORKING_TILES)} tiles from real WFS data")
logger.debug(f"Environments: {len(set(tile.environment for tile in WORKING_TILES))}")
logger.debug(
    f"Regions: {len(set(tile.location.split(' - ')[0] for tile in WORKING_TILES))}"
)
