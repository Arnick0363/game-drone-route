"""禁飞区顶点可视图 + Dijkstra。边权是欧氏距离，只绕开禁飞，不绕限高。"""

from __future__ import annotations

import networkx as nx
from shapely.geometry import LineString, Point

from score_model import feature_polygon, segment_hits_nofly


def _dedupe(points: list[tuple[float, float]], tol: float = 0.8) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for point in points:
        if all((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 > tol * tol for other in kept):
            kept.append(point)
    return kept


def obstacle_vertices(polys: list, eps: float) -> list[tuple[float, float]]:
    raw: list[tuple[float, float]] = []
    for poly in polys:
        if poly.is_empty:
            continue
        parts = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
        for part in parts:
            raw.extend((x, y) for x, y in part.exterior.coords[:-1])
    unique = _dedupe(raw, tol=1.0)
    visible = []
    for x, y in unique:
        point = Point(x, y)
        blocked = False
        for poly in polys:
            if poly.contains(point) and poly.boundary.distance(point) > eps:
                blocked = True
                break
        if not blocked:
            visible.append((x, y))
    return visible


def _bbox_miss(a: tuple[float, float], b: tuple[float, float], poly) -> bool:
    minx, miny, maxx, maxy = poly.bounds
    if max(a[0], b[0]) < minx - 0.01 or min(a[0], b[0]) > maxx + 0.01:
        return True
    if max(a[1], b[1]) < miny - 0.01 or min(a[1], b[1]) > maxy + 0.01:
        return True
    return False


def segment_clear(a: tuple[float, float], b: tuple[float, float], polys: list, eps: float) -> bool:
    if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < 1e-8:
        return False
    line = LineString([a, b])
    for poly in polys:
        if _bbox_miss(a, b, poly):
            continue
        if segment_hits_nofly(line, poly, eps):
            return False
    return True


def _edge_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def shortest_leg(
    start: tuple[float, float],
    goal: tuple[float, float],
    polys: list,
    eps: float,
    weight_fn=None,
    extra_points: list | None = None,
) -> list[tuple[float, float]]:
    """避障最短路。weight_fn 为空时边权是距离；有代价函数时直线畅通也要进图，因为绕行可能更便宜。"""
    if weight_fn is None and segment_clear(start, goal, polys, eps):
        return [start, goal]
    extras = []
    for point in extra_points or []:
        if _edge_length(point, start) < 1 or _edge_length(point, goal) < 1:
            continue
        extras.append(point)
    tail = _dedupe([*obstacle_vertices(polys, eps), *extras], tol=1.0)
    nodes = [start, goal, *tail]
    # 风和转角随方向变化，所以两个方向各有一条边。距离在两个方向上相同。
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(nodes)))
    pairs = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if segment_clear(nodes[i], nodes[j], polys, eps):
                pairs.append((i, j))
                pairs.append((j, i))
    if weight_fn is None:
        weights = [_edge_length(nodes[i], nodes[j]) for i, j in pairs]
    elif hasattr(weight_fn, "batch"):
        weights = list(weight_fn.batch(nodes, pairs, start, goal))
    else:
        weights = [float(weight_fn(nodes[i], nodes[j], start, goal)) for i, j in pairs]
    for (i, j), weight in zip(pairs, weights):
        if weight is None or weight != weight or weight <= 0:
            weight = _edge_length(nodes[i], nodes[j])
        graph.add_edge(i, j, weight=float(weight))
    if not graph.has_node(0) or not graph.has_node(1) or not nx.has_path(graph, 0, 1):
        raise RuntimeError("可视图不连通，这一段没有避开禁飞区的折线")
    path_index = nx.shortest_path(graph, 0, 1, weight="weight")
    return [nodes[index] for index in path_index]


def route_through(
    stops_xy: list[tuple[float, float]],
    nofly_features: list[dict],
    eps: float,
    weight_fn=None,
    extra_points: list | None = None,
) -> list[tuple[float, float]]:
    polys = [feature_polygon(item) for item in nofly_features]
    full: list[tuple[float, float]] = []
    for start, goal in zip(stops_xy, stops_xy[1:]):
        leg = shortest_leg(
            start,
            goal,
            polys,
            eps,
            weight_fn=weight_fn,
            extra_points=extra_points,
        )
        if full:
            full.extend(leg[1:])
        else:
            full.extend(leg)
    return full
