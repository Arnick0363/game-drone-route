"""把 OSM、统计局分区和关卡定义做成前端能直接读的场景与基准航线。"""

from __future__ import annotations

import json
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from geo import MODEL, exterior_lonlat, to_lonlat, to_xy
from score_model import evaluate_route
from visibility import route_through

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

DISCLAIMER = "禁飞区、限高区和电磁谨慎带都是教学演示近似，不是澳门法定管制空域，也不能当作飞行批准。"


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_osm() -> dict:
    return json.loads((RAW / "osm_macau.json").read_text(encoding="utf-8"))


def load_population() -> list[dict]:
    transformer = Transformer.from_crs("EPSG:8433", "EPSG:4326", always_xy=True)
    raw = json.loads((RAW / "dsec_zones_2021.json").read_text(encoding="utf-8"))
    zones = []
    for feature in raw["features"]:
        attrs = feature["attributes"]
        density = int(attrs.get("density") or 0)
        if density <= 0:
            continue
        rings_ll = []
        for ring in feature["geometry"]["rings"]:
            rings_ll.append([list(transformer.transform(x, y)) for x, y in ring])
        poly = Polygon(rings_ll[0], rings_ll[1:] if len(rings_ll) > 1 else None)
        if not poly.is_valid:
            poly = poly.buffer(0)
        pieces = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
        for piece in pieces:
            projected = shapely_transform(piece)
            simplified = projected.simplify(32, preserve_topology=True)
            parts = list(simplified.geoms) if simplified.geom_type == "MultiPolygon" else [simplified]
            for part in parts:
                if part.is_empty or part.area < 8000:
                    continue
                zones.append(
                    {
                        "name": attrs.get("Desc_CN") or attrs.get("Desc_C"),
                        "density": density,
                        "rings": polygon_rings(part),
                    }
                )
    return zones


def shapely_transform(poly):
    return Polygon(
        [to_xy(x, y) for x, y in poly.exterior.coords],
        [[to_xy(x, y) for x, y in hole.coords] for hole in poly.interiors],
    )


def polygon_rings(poly) -> list[list[list[float]]]:
    rings = [exterior_lonlat(poly)]
    for hole in poly.interiors:
        coords = []
        for x, y in hole.coords:
            lon, lat = to_lonlat(x, y)
            coords.append([round(lon, 7), round(lat, 7)])
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        rings.append(coords)
    return rings


def as_feature(poly, name: str, reason: str, source: str) -> list[dict]:
    if poly.is_empty:
        return []
    parts = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    features = []
    for part in parts:
        if part.area < 200:
            continue
        features.append(
            {
                "name": name,
                "reason": reason,
                "source": source,
                "rings": polygon_rings(part),
            }
        )
    return features


def circle(lon: float, lat: float, radius: float, resolution: int = 7):
    return Point(to_xy(lon, lat)).buffer(radius, quad_segs=resolution)


def runway_line(osm: dict) -> LineString:
    parts = []
    for element in osm["elements"]:
        tags = element.get("tags") or {}
        if tags.get("aeroway") != "runway" or "16/34" not in tags.get("ref", ""):
            continue
        geom = element.get("geometry") or []
        if len(geom) >= 2:
            parts.append(LineString([to_xy(p["lon"], p["lat"]) for p in geom]))
    merged = unary_union(parts)
    if merged.geom_type == "LineString":
        return merged
    coords = []
    for part in merged.geoms:
        coords.extend(list(part.coords))
    north = max(coords, key=lambda item: item[1])
    south = min(coords, key=lambda item: item[1])
    return LineString([south, north])


def approach_polygon(line: LineString, length: float, half_width: float, which: str):
    south, north = line.coords[0], line.coords[-1]
    dx, dy = north[0] - south[0], north[1] - south[1]
    norm = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / norm, dy / norm
    origin = north if which == "north" else south
    sign = 1 if which == "north" else -1
    end = (origin[0] + sign * ux * length, origin[1] + sign * uy * length)
    px, py = -uy * half_width, ux * half_width
    return Polygon(
        [
            (origin[0] + px, origin[1] + py),
            (origin[0] - px, origin[1] - py),
            (end[0] - px, end[1] - py),
            (end[0] + px, end[1] + py),
        ]
    )


def parse_height(tags: dict) -> float | None:
    raw = tags.get("height")
    if raw is None:
        return None
    try:
        return float(str(raw).split(";")[0].replace("m", "").strip())
    except ValueError:
        return None


def canyon_polygons(osm: dict, stops: list[dict]) -> list:
    boxes = []
    for element in osm["elements"]:
        if element.get("type") != "way":
            continue
        height = parse_height(element.get("tags") or {})
        if height is None or height < 30:
            continue
        geom = element.get("geometry") or []
        if len(geom) < 4:
            continue
        lon = sum(p["lon"] for p in geom) / len(geom)
        lat = sum(p["lat"] for p in geom) / len(geom)
        if lon < 113.536 or not (22.12 <= lat <= 22.175 and lon <= 113.61):
            continue
        footprint = Polygon([to_xy(p["lon"], p["lat"]) for p in geom])
        if not footprint.is_valid or footprint.area < 20:
            continue
        boxes.append(footprint.buffer(38))
    if not boxes:
        return []
    pads = unary_union([Point(to_xy(s["lon"], s["lat"])).buffer(48) for s in stops])
    corridor = unary_union(
        [
            LineString([to_xy(a["lon"], a["lat"]), to_xy(b["lon"], b["lat"])]).buffer(420)
            for a, b in zip(stops, stops[1:])
        ]
    )
    merged = unary_union(boxes).difference(pads).intersection(corridor)
    return [part for part in explode_polygons(merged.simplify(16, preserve_topology=True)) if part.area >= 9000]


def explode_polygons(geom) -> list:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts = []
        for child in geom.geoms:
            parts.extend(explode_polygons(child))
        return parts
    return []


def point_clear(lon: float, lat: float, polys: list, margin: float) -> bool:
    point = Point(to_xy(lon, lat))
    return all(poly.distance(point) >= margin for poly in polys)


def leg_pairs(stops: list[dict]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pairs = []
    for left, right in zip(stops, stops[1:]):
        pairs.append(((left["lon"], left["lat"]), (right["lon"], right["lat"])))
    return pairs


def anchor_circle(spec: dict, stops: list[dict]) -> dict:
    """把圆心挪到兴趣点向指定航段作垂线的垂足上，让直线真正穿过禁飞区。"""
    spec = dict(spec)
    if "anchor_leg" not in spec:
        return spec
    start, goal = stops[spec["anchor_leg"]], stops[spec["anchor_leg"] + 1]
    line = LineString([to_xy(start["lon"], start["lat"]), to_xy(goal["lon"], goal["lat"])])
    origin = Point(to_xy(spec["lon"], spec["lat"]))
    foot = line.interpolate(line.project(origin))
    lon, lat = to_lonlat(foot.x, foot.y)
    spec["source"] = (
        f"{spec['source']}兴趣点原坐标 {spec['lon']:.5f}, {spec['lat']:.5f}；"
        f"圆心放在它到第 {spec['anchor_leg'] + 1} 段任务直线的垂足（{lon:.5f}, {lat:.5f}）。"
        "若把圆心留在兴趣点上，圆要很大才挡得住航线，会吞掉旁边的配送点。"
    )
    spec["lon"], spec["lat"] = lon, lat
    return spec


def try_circle(spec: dict, stops: list[dict]) -> tuple[dict, object, float] | None:
    radii = spec["radii"]
    blocked_need = spec.get("min_block_m", 12)
    for radius in radii:
        poly = circle(spec["lon"], spec["lat"], radius)
        if not point_clear_all(stops, [poly], spec.get("margin_m", 70)):
            continue
        blocked = 0.0
        for (lon1, lat1), (lon2, lat2) in leg_pairs(stops):
            line = LineString([to_xy(lon1, lat1), to_xy(lon2, lat2)])
            blocked += line.intersection(poly).length
        if blocked >= blocked_need:
            feature = as_feature(poly, spec["name"], spec["reason"], spec["source"])[0]
            feature["reason"] = f"{feature['reason']}教学半径 {radius:.0f} 米。"
            feature["radius_m"] = radius
            return feature, poly, radius
    return None


def point_clear_all(stops: list[dict], polys: list, margin: float) -> bool:
    return all(point_clear(stop["lon"], stop["lat"], polys, margin) for stop in stops)


def path_for(stops: list[dict], features: list[dict]) -> list[list[float]]:
    xy_path = route_through(
        [to_xy(stop["lon"], stop["lat"]) for stop in stops],
        features,
        MODEL["nofly_eps_m"],
    )
    return [[round(lon, 7), round(lat, 7)] for lon, lat in (to_lonlat(x, y) for x, y in xy_path)]


def keep_optional(stops: list[dict], required: list[tuple[dict, object]], optional_specs: list[dict]) -> list[tuple[dict, object]]:
    chosen = list(required)
    for spec in optional_specs:
        trial = try_circle(anchor_circle(spec, stops), stops)
        if trial is None:
            print(f"  跳过 {spec['name']}：半径挡不住航段，或离任务点太近")
            continue
        features = [item[0] for item in chosen] + [trial[0]]
        try:
            path_for(stops, features)
        except RuntimeError as exc:
            print(f"  跳过 {spec['name']}：加入后无路可走 ({exc})")
            continue
        chosen.append((trial[0], trial[1]))
        print(f"  保留 {spec['name']} r={trial[2]:.0f}m")
    return chosen


def metrics_of(path: list[list[float]], level: dict, population: list[dict]) -> dict:
    result = evaluate_route(
        path,
        nofly=level["nofly"],
        height=level["height"],
        population=population,
        em=level["em"],
        wind=level["wind"],
    )
    public = {key: value for key, value in result.items() if key != "segments"}
    public["segments"] = result["segments"]
    return public


def build_level(spec: dict, nofly_pairs: list[tuple[dict, object]], height_polys: list, em_polys: list, population: list[dict]) -> tuple[dict, dict]:
    stops = [spec["vertiport"], *spec["deliveries"]]
    nofly = [item[0] for item in nofly_pairs]
    if not point_clear_all(stops, [item[1] for item in nofly_pairs], 40):
        raise RuntimeError(f"{spec['id']} 任务点进入禁飞区或离边界不足 40 m")
    height = []
    for index, poly in enumerate(height_polys, start=1):
        height.extend(
            as_feature(
                poly,
                f"城市峡谷 {index}",
                "由高度不低于 30 米的建筑足迹聚合而成，演示峡谷里要降速。",
                spec["height_source"],
            )
        )
    em = []
    for poly in em_polys:
        em.extend(
            as_feature(
                poly,
                "电磁谨慎带",
                "跑道向外扩出的演示环，提醒靠近机场导航设施，本身不算失败。",
                "以 OSM 跑道中心线外扩，教学宽度，不是电磁环境实测。",
            )
        )
    level = {
        "id": spec["id"],
        "order": spec["order"],
        "title": spec["title"],
        "subtitle": spec["subtitle"],
        "story": spec["story"],
        "disclaimer": DISCLAIMER,
        "task": "按医嘱顺序送达全部配送点。航线是途经点之间的直线，不沿地面路网。",
        "wind": spec["wind"],
        "battery_wh": spec["battery_wh"],
        "capture_radius_m": spec["capture_radius_m"],
        "vertiport": spec["vertiport"],
        "deliveries": spec["deliveries"],
        "nofly": nofly,
        "height": height,
        "em": em,
        "height_rule": spec["height_source"],
    }
    path = path_for(stops, nofly)
    metrics = metrics_of(path, level, population)
    if not metrics["legal"]:
        raise RuntimeError(f"{spec['id']} 基准航线仍穿越禁飞区：{metrics['reason']}")
    print(
        f"{spec['id']} 航程 {metrics['distance_m']:.0f} m，"
        f"时间 {metrics['flight_time_s']:.0f} s，"
        f"限高 {metrics['height_m']:.0f} m，"
        f"代价 {metrics['score']:.0f}，"
        f"暴露 {metrics['exposure_density'] and round(metrics['exposure_density'])}"
    )
    baseline = {"path": path, "metrics": {k: v for k, v in metrics.items() if k != "segments"}}
    return level, baseline


def write_svg(levels: list[dict], baselines: dict) -> None:
    width, height = 1100, 420
    panels = []
    for index, level in enumerate(levels):
        points = []
        for feature in level["nofly"] + level["height"] + level["em"]:
            for ring in feature["rings"]:
                points.extend(ring)
        points.append([level["vertiport"]["lon"], level["vertiport"]["lat"]])
        for stop in level["deliveries"]:
            points.append([stop["lon"], stop["lat"]])
        lons = [p[0] for p in points]
        lats = [p[1] for p in points]
        minx, maxx = min(lons) - 0.004, max(lons) + 0.004
        miny, maxy = min(lats) - 0.003, max(lats) + 0.003
        ox, oy, pw, ph = 20 + index * 360, 36, 340, 360

        def px(lon, lat):
            x = ox + (lon - minx) / (maxx - minx) * pw
            y = oy + (maxy - lat) / (maxy - miny) * ph
            return x, y

        def ring_path(ring: list[list[float]]) -> str:
            cmds = []
            for i, (lon, lat) in enumerate(ring):
                x, y = px(lon, lat)
                cmds.append(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}")
            return " ".join(cmds) + " Z"

        chunks = [f'<text x="{ox}" y="24" font-size="16" font-family="Microsoft YaHei, sans-serif">{level["title"]}</text>']
        for feature in level["em"]:
            chunks.append(f'<path d="{ring_path(feature["rings"][0])}" fill="#6d5bd0" fill-opacity="0.18" stroke="#6d5bd0"/>')
        for feature in level["height"]:
            chunks.append(f'<path d="{ring_path(feature["rings"][0])}" fill="#e6a317" fill-opacity="0.45" stroke="#b7791f"/>')
        for feature in level["nofly"]:
            chunks.append(f'<path d="{ring_path(feature["rings"][0])}" fill="#d64545" fill-opacity="0.55" stroke="#9b2c2c"/>')
        route = baselines[level["id"]]["path"]
        cmds = []
        for i, (lon, lat) in enumerate(route):
            x, y = px(lon, lat)
            cmds.append(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}")
        chunks.append(f'<path d="{" ".join(cmds)}" fill="none" stroke="#0f766e" stroke-width="2.4"/>')
        sx, sy = px(level["vertiport"]["lon"], level["vertiport"]["lat"])
        chunks.append(f'<rect x="{sx-4:.1f}" y="{sy-4:.1f}" width="8" height="8" fill="#0f172a"/>')
        for stop in level["deliveries"]:
            x, y = px(stop["lon"], stop["lat"])
            chunks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#15803d"/>')
        panels.append("".join(chunks))
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f4f7f8"/>'
        + "".join(panels)
        + "</svg>"
    )
    path = ROOT / "assets" / "levels.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_fixtures() -> None:
    square = [[113.54, 22.15], [113.545, 22.15], [113.545, 22.155], [113.54, 22.155], [113.54, 22.15]]
    nofly = [{"name": "方块禁飞", "rings": [square]}]
    height = [{"name": "东半限高", "rings": [[[113.5425, 22.148], [113.548, 22.148], [113.548, 22.157], [113.5425, 22.157], [113.5425, 22.148]]]}]
    cases = []
    samples = [
        ("向东逆风", [[113.535, 22.152], [113.55, 22.152]], {"from_deg": 90, "speed_ms": 6}, nofly, []),
        ("向西顺风", [[113.55, 22.152], [113.535, 22.152]], {"from_deg": 90, "speed_ms": 6}, [], []),
        ("无风穿限高", [[113.538, 22.152], [113.55, 22.152]], {"from_deg": 0, "speed_ms": 0}, [], height),
        ("穿越禁飞", [[113.538, 22.1525], [113.547, 22.1525]], {"from_deg": 0, "speed_ms": 0}, nofly, []),
        ("贴边外通过", [[113.538, 22.1492], [113.547, 22.1492]], {"from_deg": 0, "speed_ms": 0}, nofly, []),
    ]
    for name, route, wind, red, yellow in samples:
        result = evaluate_route(route, nofly=red, height=yellow, wind=wind)
        cases.append(
            {
                "name": name,
                "route": route,
                "wind": wind,
                "nofly": red,
                "height": yellow,
                "expect": {
                    "legal": result["legal"],
                    "distance_m": round(result["distance_m"], 3),
                    "flight_time_s": round(result["flight_time_s"], 3),
                    "height_m": round(result["height_m"], 3),
                    "score": None if result["score"] is None else round(result["score"], 3),
                },
            }
        )
    dump(OUT / "fixtures" / "geometry_cases.json", {"model": MODEL, "cases": cases})


def mission_specs() -> list[dict]:
    def stop(name, lon, lat, payload, order, note):
        return {
            "id": f"D{order}",
            "name": name,
            "lon": lon,
            "lat": lat,
            "payload": payload,
            "order": order,
            "note": note,
        }

    return [
        {
            "id": "L1",
            "order": 1,
            "title": "路环沿岸 · 开阔空域",
            "subtitle": "约束少，先学会按顺序送",
            "story": "傍晚把路环和石排湾的样本送到海洋花园卫生中心。这一带居住密度低，空域开阔，只有一处工业设施的演示禁飞需要绕开。",
            "wind": {"from_deg": 90, "speed_ms": 3.2, "note": "沿岸偏东风，风速不大，用来熟悉地速而不是制造逆风困境。"},
            "battery_wh": 70,
            "capture_radius_m": 100,
            "height_source": "本关不设限高。",
            "vertiport": {
                "name": "路环卫生站出发锚点",
                "lon": 113.55120,
                "lat": 22.11793,
                "note": "借用 OSM 卫生站坐标作为临时起飞点，不表示这里有无人机起降场。",
            },
            "deliveries": [
                stop("公共卫生临床中心", 113.55632, 22.12434, "咽拭子样本", 1, "OSM 诊所节点"),
                stop("石排湾卫生中心", 113.56362, 22.12877, "社区常规用药", 2, "OSM 诊所节点"),
                stop("海洋花园卫生中心", 113.54229, 22.15883, "外送检验试剂", 3, "OSM 诊所节点"),
            ],
            "circles": [
                {
                    "name": "路环工业设施",
                    "lon": 113.55248,
                    "lat": 22.12361,
                    "radii": [180, 170, 160],
                    "anchor_leg": 0,
                    "reason": "路环沿岸工业设施的演示禁飞，挡住卫生站到临床中心的直线。",
                    "source": "中心取 OSM 工业兴趣点，半径是教学设定。",
                    "optional": False,
                }
            ],
            "use_runway": False,
            "use_canyon": False,
        },
        {
            "id": "L2",
            "order": 2,
            "title": "半岛老城 · 密集禁飞",
            "subtitle": "历史城区、机关和学校把直线封住",
            "story": "从外港把血浆和药品按顺序送到仁伯爵综合医院、下环、镜湖、塔石，最后进入黑沙环。老城不能拉直线，禁飞圈一多，绕行会贴着全澳最密的居住区飞。",
            "wind": {"from_deg": 30, "speed_ms": 4.5, "note": "偏北风的均匀近似，不代表某一天的实况。"},
            "battery_wh": 95,
            "capture_radius_m": 78,
            "height_source": "本关故意不叠限高，让禁飞圈本身成为难点。",
            "vertiport": {
                "name": "外港客运码头演示起降点",
                "lon": 113.55914,
                "lat": 22.19733,
                "note": "借用码头坐标，把水陆联运节点想象成低空物流交接点。",
            },
            "deliveries": [
                stop("仁伯爵综合医院", 113.54624, 22.19362, "急救血浆", 1, "OSM 医院节点"),
                stop("下环卫生中心", 113.53447, 22.19145, "社区补货", 2, "OSM 诊所节点"),
                stop("镜湖医院", 113.54272, 22.19900, "专科药品", 3, "OSM 医院节点"),
                stop("塔石卫生中心", 113.54662, 22.19828, "门急诊耗材", 4, "OSM 诊所节点"),
                stop("黑沙环卫生中心", 113.55473, 22.21223, "高密度社区用药", 5, "OSM 诊所节点"),
            ],
            "circles": [
                {
                    "name": "中央联络办周边",
                    "lon": 113.54940,
                    "lat": 22.19450,
                    "radii": [150, 130, 110],
                    "reason": "机关设施周边的演示禁飞。",
                    "source": "中心取 OSM 兴趣点，半径是教学设定。",
                    "optional": False,
                },
                {
                    "name": "政府总部周边",
                    "lon": 113.53730,
                    "lat": 22.19106,
                    "radii": [160, 140, 120],
                    "reason": "政府总部一带的演示禁飞。",
                    "source": "中心取 OSM 兴趣点，半径是教学设定。",
                    "optional": False,
                },
                {
                    "name": "大三巴历史核心",
                    "lon": 113.54084,
                    "lat": 22.19746,
                    "radii": [150, 130, 110],
                    "reason": "历史城区核心的演示禁飞，不是遗产区官方边界。",
                    "source": "中心取 OSM 大三巴牌坊，半径是教学设定。",
                    "optional": False,
                },
                {
                    "name": "东南学校",
                    "lon": 113.53893,
                    "lat": 22.19952,
                    "radii": [180, 150, 120],
                    "reason": "学校上空的演示禁飞。",
                    "source": "中心取 OSM 学校，半径是教学设定。",
                    "optional": True,
                },
                {
                    "name": "二龙喉公立学校",
                    "lon": 113.55093,
                    "lat": 22.20007,
                    "radii": [180, 150, 120],
                    "reason": "学校上空的演示禁飞。",
                    "source": "中心取 OSM 学校，半径是教学设定。",
                    "optional": True,
                },
                {
                    "name": "妇联学校",
                    "lon": 113.55732,
                    "lat": 22.20565,
                    "radii": [200, 170, 140],
                    "reason": "学校上空的演示禁飞。",
                    "source": "中心取 OSM 学校，半径是教学设定。",
                    "optional": True,
                },
            ],
            "use_runway": False,
            "use_canyon": False,
        },
        {
            "id": "L3",
            "order": 3,
            "title": "离岛转运 · 净空与峡谷",
            "subtitle": "机场、油库和高层建筑叠在同一趟任务里",
            "story": "先从石排湾把样本送到氹仔，再经嘉模转到科大医院，最后送到九澳的社会及卫生服务大楼。南下和北上会遇到高层建筑组成的城市峡谷；去九澳的直线会擦过油库，东边则是跑道净空。",
            "wind": {"from_deg": 60, "speed_ms": 6.0, "note": "离岛偏东风取 6 m/s 的均匀风，用来比较航向，不是机场气象报文。"},
            "battery_wh": 105,
            "capture_radius_m": 88,
            "height_source": "OSM 中高度不低于 30 米的建筑，缓冲 38 米后融合，再挖去任务点周围 48 米，只保留离直线任务走廊 420 米以内、面积不小于 9000 平方米的碎片。经度小于 113.536 的建筑不进入，以免把横琴算进澳门。这是城市峡谷演示，不是法定限高区。",
            "vertiport": {
                "name": "石排湾卫生中心出发锚点",
                "lon": 113.56362,
                "lat": 22.12877,
                "note": "借用卫生中心坐标作为演示起飞点，不表示这里有无人机起降场。",
            },
            "deliveries": [
                stop("镜湖医院氹仔医疗中心", 113.55421, 22.15152, "上送样本", 1, "OSM 医院节点"),
                stop("湖畔嘉模卫生中心", 113.56132, 22.15847, "门诊药品", 2, "OSM 诊所节点"),
                stop("科大医院", 113.56818, 22.15438, "冷链试剂", 3, "OSM 医院节点，靠近澳门科技大学"),
                stop("九澳社会及卫生服务大楼", 113.58903, 22.13059, "离岛末端配送", 4, "OSM 建筑物中心"),
            ],
            "circles": [
                {
                    "name": "九澳油库",
                    "lon": 113.58983028,
                    "lat": 22.13386146,
                    "radii": [160, 150, 140],
                    "margin_m": 60,
                    "anchor_leg": 3,
                    "reason": "燃油设施的演示禁飞，挡在科大医院到九澳的直线上。",
                    "source": "中心取 OSM 九澳油库，半径是教学设定。",
                    "optional": False,
                },
                {
                    "name": "圣善学校",
                    "lon": 113.55842,
                    "lat": 22.15313,
                    "radii": [160, 150],
                    "anchor_leg": 1,
                    "reason": "学校上空的演示禁飞，挡在镜湖氹仔到嘉模的直线上。",
                    "source": "中心取 OSM 学校，半径是教学设定。",
                    "optional": True,
                },
            ],
            "use_runway": True,
            "use_canyon": True,
        },
    ]


def main() -> None:
    population = load_population()
    dump(
        OUT / "population.json",
        {
            "source": "澳门统计暨普查局 2021 人口普查统计分区人口密度（Census GIS 字段 density）。几何原始坐标为澳门网格，服务标注 wkid 3064，但按该代码转换会偏到几度附近；这里改用 EPSG:8433 转到 WGS84。边界简化约 32 米。",
            "density_unit": "人/平方公里",
            "zones": population,
        },
    )
    dump(OUT / "model.json", MODEL)
    osm = load_osm()
    runway = runway_line(osm)
    print(f"跑道折线 {runway.length:.0f} m，人口分区 {len(population)} 个")

    levels = []
    baselines = {}
    for spec in mission_specs():
        print(f"\n== {spec['id']} ==")
        stops = [spec["vertiport"], *spec["deliveries"]]
        required = []
        optional = []
        for circle_spec in spec["circles"]:
            (optional if circle_spec.get("optional") else required).append(circle_spec)
        required_pairs = []
        for circle_spec in required:
            fitted = try_circle(anchor_circle(circle_spec, stops), stops)
            if fitted is None:
                raise RuntimeError(f"{spec['id']} 无法放置禁飞圈：{circle_spec['name']}")
            required_pairs.append((fitted[0], fitted[1]))
            print(f"  禁飞 {circle_spec['name']} r={fitted[2]:.0f}m")
        if spec["use_runway"]:
            runway_poly = runway.buffer(430, quad_segs=3).simplify(12, preserve_topology=True)
            if not point_clear_all(stops, [runway_poly], 50):
                raise RuntimeError("跑道缓冲覆盖了任务点，请减小半径")
            runway_features = as_feature(
                runway_poly,
                "澳门国际机场跑道净空",
                "沿 OSM 16/34 跑道中心线外扩约 430 米的演示禁飞，不是障碍物限制面。",
                "OSM aeroway=runway，ref=16/34；缓冲半径为教学设定。",
            )
            required_pairs.extend((feature, runway_poly) for feature in runway_features)
            south_approach = approach_polygon(runway, 1700, 260, "south")
            hits = 0.0
            for (lon1, lat1), (lon2, lat2) in leg_pairs(stops):
                hits += LineString([to_xy(lon1, lat1), to_xy(lon2, lat2)]).intersection(south_approach).length
            if hits > 20 and point_clear_all(stops, [south_approach], 45):
                for feature in as_feature(
                    south_approach,
                    "跑道南端演示进近带",
                    "从跑道南端沿方位外推约 1.7 公里、全宽约 520 米。教学用的进近保护带，不是公布程序。",
                    "方位来自 OSM 跑道两端，长度和宽度是教学设定。",
                ):
                    required_pairs.append((feature, south_approach))
                print(f"  南端进近带切入直线 {hits:.0f} m")
            else:
                print(f"  南端进近带未启用，直线切入 {hits:.0f} m")
            em_outer = runway.buffer(1100, quad_segs=3).simplify(20, preserve_topology=True)
            em_poly = em_outer.difference(runway_poly.buffer(15))
            em_polys = [em_poly]
        else:
            em_polys = []
        chosen = keep_optional(stops, required_pairs, optional)
        height_polys = canyon_polygons(osm, stops) if spec["use_canyon"] else []
        print(f"  城市峡谷碎片 {len(height_polys)} 个")
        level, baseline = build_level(spec, chosen, height_polys, em_polys, population)
        levels.append(level)
        baselines[spec["id"]] = baseline
        dump(OUT / "levels" / f"{spec['id'].lower()}.json", level)

    dump(
        OUT / "baseline.json",
        {
            "method": "以起降点、配送点和禁飞区顶点做可视图，Dijkstra 求避障最短路。边权是局部投影下的欧氏距离。限高和风不进入寻路，只在事后用同一套评分计算。",
            "model": MODEL,
            "levels": baselines,
        },
    )
    summary = []
    for level in levels:
        item = baselines[level["id"]]["metrics"]
        summary.append(
            {
                "id": level["id"],
                "title": level["title"],
                "nofly": len(level["nofly"]),
                "height": len(level["height"]),
                "deliveries": len(level["deliveries"]),
                "distance_m": round(item["distance_m"], 1),
                "flight_time_s": round(item["flight_time_s"], 1),
                "height_m": round(item["height_m"], 1),
                "score": round(item["score"], 1),
                "energy_wh": round(item["energy_wh"], 2),
                "battery_pct": round(100 * item["energy_wh"] / level["battery_wh"], 1),
                "exposure_density": None if item["exposure_density"] is None else round(item["exposure_density"], 1),
                "exposure_top_zone": item["exposure_top_zone"],
                "exposure_peak_zone": item["exposure_peak_zone"],
                "exposure_peak_density": item["exposure_peak_density"],
                "em_m": round(item["em_m"], 1),
                "uncovered_m": round(item["uncovered_m"], 1),
            }
        )
    dump(OUT / "summary.json", {"levels": summary, "disclaimer": DISCLAIMER})
    write_svg(levels, baselines)
    write_fixtures()
    print("\n已写出 data/levels、baseline.json、summary.json、assets/levels.svg")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
