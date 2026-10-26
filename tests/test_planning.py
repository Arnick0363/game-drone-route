import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geo import MODEL, to_xy  # noqa: E402
from score_model import evaluate_route  # noqa: E402
from visibility import route_through  # noqa: E402


def calm(route, **kwargs):
    return evaluate_route(route, wind={"from_deg": 0, "speed_ms": 0}, **kwargs)


class ScoreTests(unittest.TestCase):
    def test_calm_score_is_twice_distance(self):
        result = calm([[113.54, 22.16], [113.55, 22.16]])
        self.assertTrue(result["legal"])
        self.assertAlmostEqual(result["flight_time_s"], result["distance_m"] / 12, places=6)
        self.assertAlmostEqual(result["score"], 2 * result["distance_m"], places=4)

    def test_east_wind_changes_time_with_heading(self):
        east = [[113.54, 22.16], [113.552, 22.16]]
        into = evaluate_route(east, wind={"from_deg": 90, "speed_ms": 6})
        away = evaluate_route(list(reversed(east)), wind={"from_deg": 90, "speed_ms": 6})
        self.assertAlmostEqual(into["flight_time_s"], into["distance_m"] / 6, places=4)
        self.assertAlmostEqual(away["flight_time_s"], away["distance_m"] / 18, places=4)
        self.assertAlmostEqual(into["score"], 3 * into["distance_m"], places=3)
        self.assertAlmostEqual(away["score"], away["distance_m"] * 5 / 3, places=3)
        self.assertGreater(into["flight_time_s"], away["flight_time_s"])

    def test_height_slows_only_the_overlapping_part(self):
        route = [[113.538, 22.152], [113.55, 22.152]]
        free = calm(route)
        limited = calm(
            route,
            height=[{
                "name": "半段",
                "rings": [[[113.544, 22.149], [113.552, 22.149], [113.552, 22.155], [113.544, 22.155], [113.544, 22.149]]],
            }],
        )
        self.assertGreater(limited["height_m"], 100)
        self.assertLess(limited["height_m"], free["distance_m"])
        outside = free["distance_m"] - limited["height_m"]
        expected = outside / 12 + limited["height_m"] / 7
        self.assertAlmostEqual(limited["flight_time_s"], expected, places=3)
        self.assertGreater(limited["score"], free["score"])

    def test_crossing_fails_but_grazing_boundary_does_not(self):
        box = {
            "name": "方块",
            "rings": [[[113.54, 22.15], [113.545, 22.15], [113.545, 22.155], [113.54, 22.155], [113.54, 22.15]]],
        }
        blocked = calm([[113.538, 22.1525], [113.547, 22.1525]], nofly=[box])
        clear = calm([[113.538, 22.149], [113.547, 22.149]], nofly=[box])
        self.assertFalse(blocked["legal"])
        self.assertIsNone(blocked["score"])
        self.assertIn("方块", blocked["hit_names"])
        self.assertTrue(clear["legal"])

    def test_visibility_route_avoids_interior_and_is_longer(self):
        poly = {
            "name": "圆",
            "rings": [circle_ring(113.55, 22.16, 180)],
        }
        start = (113.545, 22.16)
        goal = (113.555, 22.16)
        straight = calm([[*start], [*goal]], nofly=[poly])
        self.assertFalse(straight["legal"])
        path_xy = route_through([to_xy(*start), to_xy(*goal)], [poly], MODEL["nofly_eps_m"])
        from geo import to_lonlat

        path = [list(to_lonlat(x, y)) for x, y in path_xy]
        flown = calm(path, nofly=[poly])
        self.assertTrue(flown["legal"])
        self.assertGreater(flown["distance_m"], straight["distance_m"] + 40)
        self.assertLess(hypot_m(path[0], start), 1)
        self.assertLess(hypot_m(path[-1], goal), 1)


class BuiltLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels = {
            key: json.loads((ROOT / "data" / "levels" / f"{key}.json").read_text(encoding="utf-8"))
            for key in ("l1", "l2", "l3")
        }
        cls.baseline = json.loads((ROOT / "data" / "baseline.json").read_text(encoding="utf-8"))
        cls.population = json.loads((ROOT / "data" / "population.json").read_text(encoding="utf-8"))["zones"]

    def test_constraint_density_increases(self):
        self.assertEqual(len(self.levels["l1"]["nofly"]), 1)
        self.assertEqual(self.levels["l1"]["height"], [])
        self.assertGreaterEqual(len(self.levels["l2"]["nofly"]), 3)
        self.assertEqual(self.levels["l2"]["height"], [])
        self.assertGreaterEqual(len(self.levels["l3"]["nofly"]), 2)
        self.assertGreaterEqual(len(self.levels["l3"]["height"]), 5)
        self.assertTrue(any("跑道" in item["name"] for item in self.levels["l3"]["nofly"]))

    def test_baselines_are_legal_and_visit_in_order(self):
        for key, level in self.levels.items():
            record = self.baseline["levels"][level["id"]]
            result = evaluate_route(
                record["path"],
                nofly=level["nofly"],
                height=level["height"],
                em=level["em"],
                population=self.population,
                wind=level["wind"],
            )
            self.assertTrue(result["legal"], level["id"])
            self.assertAlmostEqual(result["score"], record["metrics"]["score"], delta=1.0)
            self.assertAlmostEqual(result["score"], result["distance_m"] + result["flight_time_s"] * 12, delta=0.1)
            stops = [level["vertiport"], *level["deliveries"]]
            cursor = 0
            for stop in stops:
                found = False
                for point in record["path"][cursor:]:
                    if hypot_m(point, (stop["lon"], stop["lat"])) < 3:
                        found = True
                        break
                    cursor += 1
                self.assertTrue(found, stop["name"])

    def test_peninsula_exposure_is_higher_than_coloane(self):
        low = self.baseline["levels"]["L1"]["metrics"]["exposure_density"]
        high = self.baseline["levels"]["L2"]["metrics"]["exposure_density"]
        self.assertLess(low, 20000)
        self.assertGreater(high, 50000)
        self.assertLess(self.baseline["levels"]["L2"]["metrics"]["uncovered_m"], 400)
        self.assertGreater(self.baseline["levels"]["L3"]["metrics"]["height_m"], 300)

    def test_fixture_file_matches_python(self):
        payload = json.loads((ROOT / "data" / "fixtures" / "geometry_cases.json").read_text(encoding="utf-8"))
        for case in payload["cases"]:
            result = evaluate_route(case["route"], nofly=case["nofly"], height=case["height"], wind=case["wind"])
            self.assertEqual(result["legal"], case["expect"]["legal"], case["name"])
            self.assertAlmostEqual(result["distance_m"], case["expect"]["distance_m"], delta=0.05)


def circle_ring(lon, lat, radius, n=16):
    from shapely.geometry import Point

    poly = Point(to_xy(lon, lat)).buffer(radius, quad_segs=4)
    ring = []
    from geo import to_lonlat

    for x, y in poly.exterior.coords:
        ring.append([round(v, 7) for v in to_lonlat(x, y)])
    return ring


def hypot_m(a, b):
    ax, ay = to_xy(a[0], a[1])
    bx, by = to_xy(b[0], b[1])
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


if __name__ == "__main__":
    unittest.main()
