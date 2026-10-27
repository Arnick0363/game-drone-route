"""用梯度提升树学习航段教学代价，并把它当作可视图边权。

标签 = 现有综合代价 + 检测密度附加 + 转角附加。
留出集上报告 MAE、R² 和 Spearman。线性回归只作对照，不拿去寻路。
样本全部由公式生成，这是教学实验，不是飞行记录。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from cost_features import (
    FEATURE_ZH,
    FEATURES,
    PERCEPTION_M_PER_DENSITY,
    SIGMA_M,
    TURN_M_PER_DEG,
    detection_density,
    segment_physical,
    teaching_label,
    turn_deg,
    wind_angle_deg,
)
from geo import MODEL, to_lonlat, to_xy
from score_model import evaluate_route, feature_polygon
from visibility import obstacle_vertices, segment_clear, shortest_leg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cost_model.json"
CHART = ROOT / "assets" / "cost-fit.png"
SEED = 0
LEG_CAP = 700


class SurrogateWeight:
    def __init__(self, model, ctx: "LevelContext"):
        self.model = model
        self.ctx = ctx

    def batch(self, nodes, pairs, start, goal):
        rows = []
        keep = []
        for i, j in pairs:
            measured = measure_edge(nodes[i], nodes[j], start, goal, self.ctx)
            if measured is None:
                keep.append(1.0)
                continue
            row, _physical = measured
            rows.append(row)
            keep.append(None)
        if rows:
            predicted = self.model.predict(np.asarray(rows, dtype=float))
            cursor = 0
            for index, value in enumerate(keep):
                if value is None:
                    keep[index] = max(float(predicted[cursor]), 1.0)
                    cursor += 1
        return keep


class LevelContext:
    def __init__(self, level: dict, population: list[dict], sites: list[dict]):
        self.level = level
        self.eps = float(MODEL["nofly_eps_m"])
        self.wind_from = float(level["wind"]["from_deg"])
        self.wind_speed = float(level["wind"]["speed_ms"])
        self.nofly_polys = [feature_polygon(item) for item in level["nofly"]]
        self.nofly_union = unary_union(self.nofly_polys) if self.nofly_polys else None
        height_polys = [feature_polygon(item) for item in level["height"]]
        self.height_union = unary_union(height_polys) if height_polys else None
        self.pop_geoms = []
        self.pop_densities = []
        for zone in population:
            poly = feature_polygon(zone)
            if poly.is_empty:
                continue
            self.pop_geoms.append(poly)
            self.pop_densities.append(float(zone["density"]))
        self.pop_tree = STRtree(self.pop_geoms) if self.pop_geoms else None
        self.sites = sites
        self.sigma = SIGMA_M


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def perception_sites(payload: dict | None) -> list[dict]:
    if not payload or payload.get("status") != "ok":
        return []
    sites = []
    for site in payload.get("sites") or []:
        x, y = to_xy(site["lon"], site["lat"])
        sites.append({"x": x, "y": y, "risk": float(site.get("risk") or 0), "lon": site["lon"], "lat": site["lat"]})
    return sites


def measure_edge(a, b, start, goal, ctx: LevelContext):
    line = LineString([a, b])
    length = float(line.length)
    if length < 1e-6:
        return None
    heading = np.array([b[0] - a[0], b[1] - a[1]], dtype=float) / length
    chord = np.array([goal[0] - start[0], goal[1] - start[1]], dtype=float)
    turn = turn_deg(chord, heading)
    height = 0.0
    if ctx.height_union is not None and not ctx.height_union.is_empty:
        height = min(length, float(line.intersection(ctx.height_union).length))
    pop = population_density(line, ctx)
    if ctx.nofly_union is None or ctx.nofly_union.is_empty:
        nofly_dist = 5000.0
    else:
        nofly_dist = float(line.distance(ctx.nofly_union))
    wind_angle = wind_angle_deg(heading, ctx.wind_from)
    detect = segment_detection(line, ctx)
    row = [length, turn, height / length, pop, nofly_dist, wind_angle, detect]
    physical, _seconds = segment_physical(length, height, wind_angle, ctx.wind_speed)
    return row, physical


def population_density(line: LineString, ctx: LevelContext) -> float:
    if ctx.pop_tree is None:
        return 0.0
    covered = 0.0
    weighted = 0.0
    for index in ctx.pop_tree.query(line):
        part = float(line.intersection(ctx.pop_geoms[int(index)]).length)
        if part <= 0:
            continue
        covered += part
        weighted += part * ctx.pop_densities[int(index)]
    if covered < 1.0:
        return 0.0
    return weighted / covered


def segment_detection(line: LineString, ctx: LevelContext) -> float:
    if not ctx.sites:
        return 0.0
    total = 0.0
    for fraction in (0.25, 0.5, 0.75):
        point = line.interpolate(fraction, normalized=True)
        total += detection_density(point.x, point.y, ctx.sites, ctx.sigma)
    return total / 3.0


def thin_ring(ring: list[list[float]], limit: int = 8) -> list[tuple[float, float]]:
    points = [to_xy(lon, lat) for lon, lat in ring]
    if len(points) > 1 and math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < 0.1:
        points = points[:-1]
    if len(points) <= limit:
        return points
    step = len(points) / limit
    return [points[int(i * step)] for i in range(limit)]


def height_waypoints(level: dict, nofly_polys: list, eps: float) -> list[tuple[float, float]]:
    raw = []
    for feature in level["height"]:
        raw.extend(thin_ring(feature["rings"][0], limit=8))
    kept = []
    for point in raw:
        marker = Point(point)
        blocked = False
        for poly in nofly_polys:
            if poly.contains(marker) and poly.boundary.distance(marker) > eps:
                blocked = True
                break
        if not blocked:
            kept.append(point)
    deduped = []
    for point in kept:
        if all(math.hypot(point[0] - other[0], point[1] - other[1]) > 12 for other in deduped):
            deduped.append(point)
    return deduped


def extras_near(start, goal, points, corridor_m: float = 700, limit: int = 70):
    if not points:
        return []
    line = LineString([start, goal])
    ranked = sorted(points, key=lambda point: line.distance(Point(point)))
    return [point for point in ranked if line.distance(Point(point)) <= corridor_m][:limit]


def stops_xy(level: dict) -> list[tuple[float, float]]:
    stops = [level["vertiport"], *level["deliveries"]]
    return [to_xy(stop["lon"], stop["lat"]) for stop in stops]


def sample_level(ctx: LevelContext, extras: list, cap: int = LEG_CAP):
    rng = np.random.default_rng(SEED + ctx.level["order"])
    rows = []
    labels = []
    stops = stops_xy(ctx.level)
    for start, goal in zip(stops, stops[1:]):
        near = extras_near(start, goal, extras)
        tail = obstacle_vertices(ctx.nofly_polys, ctx.eps) + near
        nodes = [start, goal]
        for point in tail:
            if math.hypot(point[0] - start[0], point[1] - start[1]) < 1:
                continue
            if math.hypot(point[0] - goal[0], point[1] - goal[1]) < 1:
                continue
            if any(math.hypot(point[0] - other[0], point[1] - other[1]) < 1 for other in nodes):
                continue
            nodes.append(point)
        pairs = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if segment_clear(nodes[i], nodes[j], ctx.nofly_polys, ctx.eps):
                    pairs.append((i, j))
        touch = [pair for pair in pairs if 0 in pair or 1 in pair]
        touch_set = set(touch)
        rest = [pair for pair in pairs if pair not in touch_set]
        rng.shuffle(rest)
        chosen = touch + rest[: max(0, cap - len(touch))]
        for i, j in chosen:
            measured = measure_edge(nodes[i], nodes[j], start, goal, ctx)
            if measured is None:
                continue
            row, physical = measured
            rows.append(row)
            labels.append(teaching_label(physical, row[1], row[6]))
        print(f"  {ctx.level['id']} 航段节点 {len(nodes)}，畅通边 {len(pairs)}，抽入 {len(chosen)}")
    return rows, labels


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        rank = np.empty(len(values), dtype=float)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start
            while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
                end += 1
            rank[order[start : end + 1]] = (start + end) / 2.0
            start = end + 1
        return rank

    left_rank = ranks(np.asarray(left, dtype=float))
    right_rank = ranks(np.asarray(right, dtype=float))
    if float(left_rank.std()) < 1e-12 or float(right_rank.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def fit_models(rows: np.ndarray, labels: np.ndarray):
    x_train, x_test, y_train, y_test = train_test_split(rows, labels, test_size=0.2, random_state=SEED)
    tree = GradientBoostingRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        random_state=SEED,
    )
    linear = LinearRegression()
    tree.fit(x_train, y_train)
    linear.fit(x_train, y_train)
    tree_pred = tree.predict(x_test)
    linear_pred = linear.predict(x_test)
    report = {
        "n": int(len(labels)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "seed": SEED,
        "test_size": 0.2,
        "gbdt": {
            "mae": float(mean_absolute_error(y_test, tree_pred)),
            "r2": float(r2_score(y_test, tree_pred)),
            "spearman": spearman(y_test, tree_pred),
            "importance": {name: float(value) for name, value in zip(FEATURES, tree.feature_importances_)},
        },
        "linear": {
            "mae": float(mean_absolute_error(y_test, linear_pred)),
            "r2": float(r2_score(y_test, linear_pred)),
            "spearman": spearman(y_test, linear_pred),
            "coef": {name: float(value) for name, value in zip(FEATURES, linear.coef_)},
            "intercept": float(linear.intercept_),
        },
    }
    return tree, report, y_test, tree_pred, linear_pred


def write_chart(y_true, tree_pred, linear_pred) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"散点图跳过：{exc}")
        return None
    CHART.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(y_true, tree_pred, s=14, c="#2C3E50", alpha=0.7, label="梯度提升树")
    ax.scatter(y_true, linear_pred, s=12, c="#E67E22", alpha=0.35, label="线性回归")
    low = float(min(np.min(y_true), np.min(tree_pred), np.min(linear_pred)))
    high = float(max(np.max(y_true), np.max(tree_pred), np.max(linear_pred)))
    ax.plot([low, high], [low, high], color="#C0392B", linewidth=1, label="理想对角线")
    ax.set_xlabel("教学标签")
    ax.set_ylabel("留出集预测")
    ax.set_title("航段代价：留出集")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART, dpi=120)
    plt.close(fig)
    return "assets/cost-fit.png"


def plan_level(ctx: LevelContext, model, extras: list) -> dict:
    stops = stops_xy(ctx.level)
    learned_xy = []
    distance_xy = []
    for start, goal in zip(stops, stops[1:]):
        near = extras_near(start, goal, extras)
        learned_leg = shortest_leg(
            start,
            goal,
            ctx.nofly_polys,
            ctx.eps,
            weight_fn=SurrogateWeight(model, ctx),
            extra_points=near,
        )
        distance_leg = shortest_leg(start, goal, ctx.nofly_polys, ctx.eps, extra_points=near)
        if learned_xy:
            learned_xy.extend(learned_leg[1:])
            distance_xy.extend(distance_leg[1:])
        else:
            learned_xy.extend(learned_leg)
            distance_xy.extend(distance_leg)
    learned = pack_path(learned_xy, stops, ctx, model)
    distance = pack_path(distance_xy, stops, ctx, model)
    same = same_geometry(distance_xy, learned_xy)
    learned["same_geometry"] = same
    distance["same_geometry"] = same
    return {
        "distance": distance,
        "learned": learned,
        "same_geometry": same,
        "note": route_note(distance["metrics"], learned["metrics"], same),
    }


def pack_path(path_xy, stops, ctx: LevelContext, model) -> dict:
    lonlat = [[round(lon, 7), round(lat, 7)] for lon, lat in (to_lonlat(x, y) for x, y in path_xy)]
    result = evaluate_route(
        lonlat,
        nofly=ctx.level["nofly"],
        height=ctx.level["height"],
        em=ctx.level["em"],
        population=load_population_zones(),
        wind=ctx.level["wind"],
    )
    teaching, predicted, detect_mean = accumulate(path_xy, stops, ctx, model)
    metrics = {key: value for key, value in result.items() if key != "segments"}
    metrics["teaching_cost"] = teaching
    metrics["predicted_cost"] = predicted
    metrics["detect_density"] = detect_mean
    return {"path": lonlat, "metrics": metrics}


POPULATION_ZONES: list[dict] | None = None


def load_population_zones() -> list[dict]:
    global POPULATION_ZONES
    if POPULATION_ZONES is None:
        POPULATION_ZONES = load_json(ROOT / "data" / "population.json")["zones"]
    return POPULATION_ZONES


def accumulate(path_xy, stops, ctx: LevelContext, model):
    teaching = 0.0
    predicted = 0.0
    detect_weighted = 0.0
    length = 0.0
    cursor = 0
    for start, goal in zip(stops, stops[1:]):
        end = cursor
        while end < len(path_xy) - 1 and math.hypot(path_xy[end][0] - goal[0], path_xy[end][1] - goal[1]) > 2.5:
            end += 1
        for left, right in zip(path_xy[cursor:end], path_xy[cursor + 1 : end + 1]):
            measured = measure_edge(left, right, start, goal, ctx)
            if measured is None:
                continue
            row, physical = measured
            teaching += teaching_label(physical, row[1], row[6])
            detect_weighted += row[0] * row[6]
            length += row[0]
            predicted += max(float(model.predict(np.asarray(row, dtype=float).reshape(1, -1))[0]), 1.0)
        cursor = end
    detect_mean = detect_weighted / length if length else 0.0
    return teaching, predicted, detect_mean


def same_geometry(left, right, tol_m: float = 20.0) -> bool:
    if len(left) < 2 or len(right) < 2:
        return False
    a = LineString(left)
    b = LineString(right)
    deviation = max(max(b.distance(Point(point)) for point in a.coords), max(a.distance(Point(point)) for point in b.coords))
    return deviation <= tol_m


def route_note(distance: dict, learned: dict, same: bool) -> str:
    delta_m = learned["distance_m"] - distance["distance_m"]
    delta_score = learned["score"] - distance["score"]
    delta_teach = learned["teaching_cost"] - distance["teaching_cost"]
    delta_height = learned["height_m"] - distance["height_m"]
    if same:
        text = "两条折线几乎重合。"
    elif abs(delta_m) < 1:
        text = "航程几乎一样，但折线分开了。"
    else:
        verb = "多" if delta_m >= 0 else "少"
        text = f"学习代价路比距离最短路{verb}飞 {abs(delta_m):.0f} 米。"
    if delta_height <= -1:
        text += f"限高段少 {abs(delta_height):.0f} 米。"
    elif delta_height >= 1:
        text += f"限高段多 {delta_height:.0f} 米。"
    if delta_score < -1:
        text += f"按航程加飞行时间的综合代价低 {abs(delta_score):.0f}。"
    elif delta_score > 1:
        text += f"综合代价高 {delta_score:.0f}。寻路最小化教学标签。排行榜仍按航程加飞行时间计。"
    else:
        text += "综合代价几乎一样。"
    if delta_teach < -1:
        text += f"教学标签低 {abs(delta_teach):.0f}。"
    elif delta_teach > 1:
        text += f"教学标签高 {delta_teach:.0f}。代理模型和标签还有差距。"
    return text


def main() -> None:
    population = load_population_zones()
    perception_path = ROOT / "data" / "perception.json"
    perception = load_json(perception_path) if perception_path.exists() else None
    sites = perception_sites(perception)
    if not sites:
        reason = (perception or {}).get("reason") or "没有 perception.json"
        print(f"检测密度按 0 处理：{reason}")

    rows = []
    labels = []
    contexts = []
    extras_by_level = {}
    for name in ("l1", "l2", "l3"):
        level = load_json(ROOT / "data" / "levels" / f"{name}.json")
        ctx = LevelContext(level, population, sites)
        extras = height_waypoints(level, ctx.nofly_polys, ctx.eps)
        extras_by_level[level["id"]] = extras
        contexts.append(ctx)
        print(f"\n== 抽样 {level['id']}，限高绕行点 {len(extras)} ==")
        part_rows, part_labels = sample_level(ctx, extras)
        rows.extend(part_rows)
        labels.extend(part_labels)

    matrix = np.asarray(rows, dtype=float)
    target = np.asarray(labels, dtype=float)
    if len(target) < 30:
        raise RuntimeError(f"教学样本只有 {len(target)} 条，不够拟合")
    model, report, y_test, tree_pred, linear_pred = fit_models(matrix, target)
    chart = write_chart(y_test, tree_pred, linear_pred)
    print(
        f"\n样本 {report['n']}，留出 {report['n_test']}。"
        f"树 MAE {report['gbdt']['mae']:.2f}，R2 {report['gbdt']['r2']:.4f}，"
        f"Spearman {report['gbdt']['spearman']:.4f}"
    )
    print(
        f"线性 MAE {report['linear']['mae']:.2f}，R2 {report['linear']['r2']:.4f}，"
        f"Spearman {report['linear']['spearman']:.4f}"
    )
    ranked = sorted(report["gbdt"]["importance"].items(), key=lambda item: item[1], reverse=True)
    print("重要性", ", ".join(f"{FEATURE_ZH[name]} {value:.3f}" for name, value in ranked))

    levels = {}
    for ctx in contexts:
        print(f"\n== 规划 {ctx.level['id']} ==")
        packed = plan_level(ctx, model, extras_by_level[ctx.level["id"]])
        levels[ctx.level["id"]] = packed
        distance = packed["distance"]["metrics"]
        learned = packed["learned"]["metrics"]
        print(
            f"  距离 {distance['distance_m']:.0f} m / 代价 {distance['score']:.0f}；"
            f"学习 {learned['distance_m']:.0f} m / 代价 {learned['score']:.0f}；"
            f"重合 {packed['same_geometry']}"
        )
        print(f"  {packed['note']}")
        if not learned["legal"] or not distance["legal"]:
            raise RuntimeError(f"{ctx.level['id']} 规划航线进入禁飞区")

    payload = {
        "experiment": "教学实验。标签由综合代价公式加检测附加和转角附加生成，不是真实飞行记录。",
        "planner": "梯度提升树的预测值作为有向可视图边权，Dijkstra 求路。风和转角随方向变化，所以两个方向各算一次。对照是同一批顶点上只最小化距离的 Dijkstra。限高边界上每片最多 8 个点，可作为绕行顶点，本身不是禁飞。",
        "features": FEATURES,
        "feature_names_zh": FEATURE_ZH,
        "label": "physical + 80 * detect_density + 0.35 * turn_deg",
        "perception_m_per_density": PERCEPTION_M_PER_DENSITY,
        "turn_m_per_deg": TURN_M_PER_DEG,
        "sigma_m": SIGMA_M,
        "perception_status": None if perception is None else perception.get("status"),
        "perception_reason": None if perception is None else perception.get("reason"),
        "chart": chart,
        "model_family": {
            "planner": "sklearn.ensemble.GradientBoostingRegressor",
            "baseline": "sklearn.linear_model.LinearRegression",
            "hyperparameters": {
                "n_estimators": 120,
                "max_depth": 3,
                "learning_rate": 0.08,
                "subsample": 0.9,
                "random_state": SEED,
            },
        },
        **report,
        "levels": levels,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {OUT}")


if __name__ == "__main__":
    main()
