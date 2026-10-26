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


def shortest_leg(
    start: tuple[float, float],
    goal: tuple[float, float],
    polys: list,
    eps: float,
) -> list[tuple[float, float]]:
    if segment_clear(start, goal, polys, eps):
        return [start, goal]
    nodes = [start, goal, *obstacle_vertices(polys, eps)]
    graph = nx.Graph()
    for index, point in enumerate(nodes):
        graph.add_node(index, pos=point)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if segment_clear(a, b, polys, eps):
                weight = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                graph.add_edge(i, j, weight=weight)
    if 0 not in graph or 1 not in graph or not nx.has_path(graph, 0, 1):
        raise RuntimeError("可视图不连通，这一段没有避开禁飞区的折线")
    path_index = nx.shortest_path(graph, 0, 1, weight="weight")
    return [nodes[index] for index in path_index]


def route_through(stops_xy: list[tuple[float, float]], nofly_features: list[dict], eps: float) -> list[tuple[float, float]]:
    polys = [feature_polygon(item) for item in nofly_features]
    full: list[tuple[float, float]] = []
    for start, goal in zip(stops_xy, stops_xy[1:]):
        leg = shortest_leg(start, goal, polys, eps)
        if full:
            full.extend(leg[1:])
        else:
            full.extend(leg)
    return full
