"""澳门附近的局部等距近似，供航程与前端使用同一套公式。"""

from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_M = 6_378_137.0
LAT0_DEG = 22.16
LAT0_RAD = math.radians(LAT0_DEG)

MODEL = {
    "lat0_deg": LAT0_DEG,
    "earth_radius_m": EARTH_RADIUS_M,
    "airspeed_ms": 12.0,
    "height_airspeed_ms": 7.0,
    "min_groundspeed_ms": 3.0,
    "power_w": 280.0,
    "headwind_power_per_ms": 0.06,
    "nofly_eps_m": 0.4,
    "score": "distance_m + flight_time_s * airspeed_ms",
}


def to_xy(lon: float, lat: float) -> tuple[float, float]:
    x = math.radians(lon) * EARTH_RADIUS_M * math.cos(LAT0_RAD)
    y = math.radians(lat) * EARTH_RADIUS_M
    return x, y


def to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = math.degrees(x / (EARTH_RADIUS_M * math.cos(LAT0_RAD)))
    lat = math.degrees(y / EARTH_RADIUS_M)
    return lon, lat


def wind_vector(from_deg: float, speed_ms: float) -> np.ndarray:
    """气象来向转为东、北分量。正值指向风的去向。"""
    to_rad = math.radians(from_deg + 180.0)
    return np.array(
        [speed_ms * math.sin(to_rad), speed_ms * math.cos(to_rad)],
        dtype=float,
    )


def ring_xy(ring: list[list[float]]) -> list[tuple[float, float]]:
    return [to_xy(lon, lat) for lon, lat in ring]


def exterior_lonlat(poly, ndigits: int = 7) -> list[list[float]]:
    coords = []
    for x, y in poly.exterior.coords:
        lon, lat = to_lonlat(x, y)
        coords.append([round(lon, ndigits), round(lat, ndigits)])
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords
