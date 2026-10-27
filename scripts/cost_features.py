"""航段特征与教学标签。几何仍在 learn_cost.py，这里只放可单测的公式。"""

from __future__ import annotations

import math

import numpy as np

from geo import MODEL

FEATURES = [
    "length_m",
    "turn_deg",
    "height_frac",
    "pop_density",
    "nofly_dist_m",
    "wind_angle_deg",
    "detect_density",
]

FEATURE_ZH = {
    "length_m": "航段长度",
    "turn_deg": "转角",
    "height_frac": "限高比例",
    "pop_density": "人口密度",
    "nofly_dist_m": "距禁飞区",
    "wind_angle_deg": "风向夹角",
    "detect_density": "检测密度",
}

# 教学附加项，不是法规。检测密度每 1 个单位大约相当于多飞这么多米的综合代价。
PERCEPTION_M_PER_DENSITY = 80.0
TURN_M_PER_DEG = 0.35
SIGMA_M = 450.0

# 低空走廊里值得加价的 COCO 类。其余检测仍记录，但不进风险。
CLASS_WEIGHTS = {
    "person": 1.0,
    "bicycle": 0.6,
    "car": 0.8,
    "motorcycle": 0.7,
    "bus": 1.0,
    "truck": 1.0,
    "bird": 1.2,
    "airplane": 1.5,
    "boat": 0.5,
    "traffic light": 0.3,
    "stop sign": 0.3,
}


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def turn_deg(chord: np.ndarray, edge: np.ndarray) -> float:
    """本边航向与这一段任务直线的夹角。可视图边权不能依赖飞到这里之前的真实转弯。"""
    return angle_between(chord, edge)


def wind_angle_deg(heading: np.ndarray, wind_from_deg: float) -> float:
    """0° 是顶风，180° 是顺风。heading 为东、北分量。"""
    rad = math.radians(wind_from_deg)
    upwind = np.array([math.sin(rad), math.cos(rad)], dtype=float)
    return angle_between(heading, upwind)


def segment_physical(
    length: float,
    height_m: float,
    wind_angle: float,
    wind_speed: float,
    model: dict | None = None,
) -> tuple[float, float]:
    """返回 (综合代价, 飞行秒)。与 score_model 的地速公式一致。"""
    model = model or MODEL
    if length <= 1e-9:
        return 0.0, 0.0
    height_m = min(max(height_m, 0.0), length)
    headwind = wind_speed * math.cos(math.radians(wind_angle))
    gs_out = max(model["min_groundspeed_ms"], model["airspeed_ms"] - headwind)
    gs_in = max(model["min_groundspeed_ms"], model["height_airspeed_ms"] - headwind)
    outside = length - height_m
    seconds = 0.0
    if outside > 0:
        seconds += outside / gs_out
    if height_m > 0:
        seconds += height_m / gs_in
    return length + seconds * model["airspeed_ms"], seconds


def teaching_label(physical: float, turn: float, detect_density: float) -> float:
    return physical + PERCEPTION_M_PER_DENSITY * detect_density + TURN_M_PER_DEG * turn


def site_risk(boxes: list[dict], weights: dict | None = None) -> float:
    weights = CLASS_WEIGHTS if weights is None else weights
    total = 0.0
    for box in boxes:
        total += float(weights.get(box["cls"], 0.0)) * float(box["conf"])
    return total


def detection_density(x: float, y: float, sites: list[dict], sigma: float = SIGMA_M) -> float:
    """拍摄点风险的指数核。sites 用局部投影米制坐标，字段 x、y、risk。"""
    if sigma <= 0:
        return 0.0
    total = 0.0
    for site in sites:
        distance = math.hypot(x - site["x"], y - site["y"])
        total += float(site["risk"]) * math.exp(-distance / sigma)
    return total
