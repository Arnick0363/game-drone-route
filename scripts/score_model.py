"""航线评分。禁飞失败；限高与风写进飞行时间；人口和电磁只解释、不计分。"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from geo import MODEL, ring_xy, to_xy, wind_vector


def _polygon(rings_lonlat: list[list[list[float]]]) -> Polygon:
    shell = ring_xy(rings_lonlat[0])
    holes = [ring_xy(ring) for ring in rings_lonlat[1:]]
    poly = Polygon(shell, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda part: part.area)
    return poly


def feature_polygon(feature: dict) -> Polygon:
    if "rings" in feature:
        return _polygon(feature["rings"])
    return _polygon([feature["ring"]])


def _heading(line: LineString) -> np.ndarray | None:
    if line.length < 1e-6:
        return None
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    vec = np.array([x1 - x0, y1 - y0], dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return None
    return vec / norm


def _ground_speed(airspeed: float, heading: np.ndarray, wind: np.ndarray, model: dict) -> tuple[float, float]:
    headwind = float(-np.dot(wind, heading))
    speed = max(model["min_groundspeed_ms"], airspeed - headwind)
    return speed, headwind


def _point_deep_inside(point: Point, poly: Polygon, eps: float) -> bool:
    if not poly.contains(point):
        return False
    boundary = poly.boundary
    return boundary.distance(point) > eps


def segment_hits_nofly(line: LineString, poly: Polygon, eps: float) -> bool:
    """进入多边形内部才算穿越。贴着边界的可视图边允许通过。"""
    if line.is_empty or line.length <= 0:
        return False
    start = Point(line.coords[0])
    end = Point(line.coords[-1])
    if _point_deep_inside(start, poly, eps) or _point_deep_inside(end, poly, eps):
        return True
    inside_len = float(poly.intersection(line).length)
    if inside_len <= eps:
        return False
    boundary_len = float(poly.boundary.intersection(line).length)
    return inside_len - boundary_len > eps


def evaluate_route(
    route_lonlat: list[list[float]],
    *,
    nofly: list[dict] | None = None,
    height: list[dict] | None = None,
    population: list[dict] | None = None,
    em: list[dict] | None = None,
    wind: dict | None = None,
    model: dict | None = None,
) -> dict:
    model = model or MODEL
    nofly = nofly or []
    height = height or []
    population = population or []
    em = em or []
    wind = wind or {"from_deg": 0.0, "speed_ms": 0.0}
    eps = float(model["nofly_eps_m"])

    if len(route_lonlat) < 2:
        return {
            "legal": False,
            "hit_names": [],
            "distance_m": 0.0,
            "flight_time_s": 0.0,
            "height_m": 0.0,
            "score": None,
            "energy_wh": 0.0,
            "headwind_m": 0.0,
            "exposure_density": None,
            "exposure_covered_m": 0.0,
            "exposure_top_zone": None,
            "exposure_top_m": 0.0,
            "em_m": 0.0,
            "segments": [],
            "reason": "航线至少需要两个点",
        }

    nofly_polys = [(item["name"], feature_polygon(item)) for item in nofly]
    height_union = None
    if height:
        from shapely.ops import unary_union

        height_union = unary_union([feature_polygon(item) for item in height])
    em_union = None
    if em:
        from shapely.ops import unary_union

        em_union = unary_union([feature_polygon(item) for item in em])
    pop_polys = [(item["name"], float(item["density"]), feature_polygon(item)) for item in population]
    wind_vec = wind_vector(float(wind.get("from_deg", 0)), float(wind.get("speed_ms", 0)))

    distance = 0.0
    flight_time = 0.0
    height_m = 0.0
    energy = 0.0
    headwind_m = 0.0
    em_m = 0.0
    hit_names: list[str] = []
    segments = []
    pop_length = {name: 0.0 for name, _, _ in pop_polys}

    for index in range(len(route_lonlat) - 1):
        a = to_xy(*route_lonlat[index])
        b = to_xy(*route_lonlat[index + 1])
        line = LineString([a, b])
        length = float(line.length)
        heading = _heading(line)
        hit = None
        seg_height = 0.0
        seg_em = 0.0
        seg_time = 0.0
        if heading is not None and length > 0:
            for name, poly in nofly_polys:
                if segment_hits_nofly(line, poly, eps):
                    hit = name
                    if name not in hit_names:
                        hit_names.append(name)
                    break
            if height_union is not None and not height_union.is_empty:
                seg_height = float(line.intersection(height_union).length)
                seg_height = min(seg_height, length)
            outside = max(0.0, length - seg_height)
            gs_out, headwind = _ground_speed(model["airspeed_ms"], heading, wind_vec, model)
            gs_in, _ = _ground_speed(model["height_airspeed_ms"], heading, wind_vec, model)
            if outside > 0:
                seg_time += outside / gs_out
            if seg_height > 0:
                seg_time += seg_height / gs_in
            if headwind > 1.0:
                headwind_m += length
            power = model["power_w"] * (1.0 + model["headwind_power_per_ms"] * max(headwind, 0.0))
            energy += power * seg_time / 3600.0
            if em_union is not None and not em_union.is_empty:
                seg_em = float(line.intersection(em_union).length)
            for name, _, poly in pop_polys:
                pop_length[name] += float(line.intersection(poly).length)
        distance += length
        flight_time += seg_time
        height_m += seg_height
        em_m += seg_em
        segments.append(
            {
                "index": index,
                "length_m": round(length, 3),
                "height_m": round(seg_height, 3),
                "em_m": round(seg_em, 3),
                "nofly": hit,
            }
        )

    covered = 0.0
    weighted = 0.0
    top_name = None
    top_len = 0.0
    peak_name = None
    peak_len = 0.0
    peak_density = None
    peak_value = 0.0
    for name, density, _ in pop_polys:
        length = pop_length[name]
        if length <= 0.5:
            continue
        covered += length
        weighted += length * density
        if length > top_len:
            top_len = length
            top_name = name
        if length * density > peak_value:
            peak_value = length * density
            peak_name = name
            peak_len = length
            peak_density = density
    exposure = (weighted / covered) if covered > 1.0 else None
    uncovered = max(0.0, distance - covered)
    legal = not hit_names
    score = (distance + flight_time * model["airspeed_ms"]) if legal else None
    return {
        "legal": legal,
        "hit_names": hit_names,
        "distance_m": distance,
        "flight_time_s": flight_time,
        "height_m": height_m,
        "score": score,
        "energy_wh": energy,
        "headwind_m": headwind_m,
        "exposure_density": exposure,
        "exposure_covered_m": covered,
        "exposure_top_zone": top_name,
        "exposure_top_m": top_len,
        "exposure_peak_zone": peak_name,
        "exposure_peak_m": peak_len,
        "exposure_peak_density": peak_density,
        "uncovered_m": uncovered,
        "em_m": em_m,
        "segments": segments,
        "reason": None if legal else f"穿越禁飞区：{'、'.join(hit_names)}",
    }
