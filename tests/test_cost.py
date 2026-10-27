import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cost_features import (  # noqa: E402
    FEATURES,
    detection_density,
    segment_physical,
    site_risk,
    teaching_label,
    turn_deg,
    wind_angle_deg,
)
from geo import MODEL, to_xy  # noqa: E402
from score_model import evaluate_route  # noqa: E402
from visibility import shortest_leg  # noqa: E402


class FeatureTests(unittest.TestCase):
    def test_wind_angle_matches_headwind_time(self):
        heading = np.array([1.0, 0.0])
        self.assertAlmostEqual(wind_angle_deg(heading, 90), 0.0, places=5)
        self.assertAlmostEqual(wind_angle_deg(np.array([-1.0, 0.0]), 90), 180.0, places=5)
        cost, seconds = segment_physical(120.0, 0.0, 0.0, 6.0)
        self.assertAlmostEqual(seconds, 120.0 / 6.0, places=5)
        self.assertAlmostEqual(cost, 120.0 + seconds * 12.0, places=5)

    def test_physical_matches_route_score(self):
        route = [[113.54, 22.16], [113.552, 22.16]]
        result = evaluate_route(route, wind={"from_deg": 90, "speed_ms": 6})
        start = to_xy(*route[0])
        goal = to_xy(*route[1])
        length = math.hypot(goal[0] - start[0], goal[1] - start[1])
        heading = np.array([goal[0] - start[0], goal[1] - start[1]]) / length
        angle = wind_angle_deg(heading, 90)
        cost, _seconds = segment_physical(length, 0.0, angle, 6.0)
        self.assertAlmostEqual(cost, result["score"], places=3)

    def test_turn_and_detection_kernel(self):
        self.assertAlmostEqual(turn_deg(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 90.0, places=5)
        sites = [{"x": 0.0, "y": 0.0, "risk": 2.0}]
        self.assertAlmostEqual(detection_density(0, 0, sites, 450), 2.0, places=6)
        self.assertLess(detection_density(900, 0, sites, 450), 2.0 * math.exp(-1.5))
        boxes = [{"cls": "car", "conf": 0.5}, {"cls": "banana", "conf": 0.9}]
        self.assertAlmostEqual(site_risk(boxes), 0.4, places=6)
        self.assertAlmostEqual(teaching_label(10, 20, 0.5), 10 + 80 * 0.5 + 0.35 * 20, places=6)

    def test_weight_prefers_the_northern_side(self):
        from shapely.geometry import Polygon

        box = Polygon([(-25, -25), (25, -25), (25, 25), (-25, 25)])

        def weight(a, b, _start, _goal):
            midpoint_y = (a[1] + b[1]) / 2
            length = math.hypot(a[0] - b[0], a[1] - b[1])
            return length * (1.0 if midpoint_y >= -1 else 40.0)

        path = shortest_leg((-160, 0), (160, 0), [box], MODEL["nofly_eps_m"], weight_fn=weight)
        self.assertGreater(max(point[1] for point in path), 20)
        self.assertLess(min(point[1] for point in path), 5)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        perception_path = ROOT / "data" / "perception.json"
        cost_path = ROOT / "data" / "cost_model.json"
        if not perception_path.exists() or not cost_path.exists():
            raise unittest.SkipTest("先运行 perceive.py 和 learn_cost.py")
        cls.perception = json.loads(perception_path.read_text(encoding="utf-8"))
        cls.cost = json.loads(cost_path.read_text(encoding="utf-8"))

    def test_perception_boxes_are_recomputable(self):
        self.assertEqual(self.perception["status"], "ok")
        self.assertGreaterEqual(len(self.perception["sites"]), 1)
        for site in self.perception["sites"]:
            self.assertAlmostEqual(site_risk(site["boxes"]), site["risk"], places=3)
            self.assertTrue(site["page_url"].startswith("https://"))
            for box in site["boxes"]:
                self.assertGreaterEqual(box["conf"], self.perception["conf"])
                self.assertLessEqual(box["conf"], 1)
                self.assertEqual(len(box["xyxy"]), 4)

    def test_cost_model_metrics_and_legal_routes(self):
        self.assertEqual(self.cost["features"], FEATURES)
        self.assertGreater(self.cost["n_test"], 10)
        for key in ("mae", "r2", "spearman"):
            self.assertTrue(math.isfinite(self.cost["gbdt"][key]))
            self.assertTrue(math.isfinite(self.cost["linear"][key]))
        importance = sum(self.cost["gbdt"]["importance"].values())
        self.assertAlmostEqual(importance, 1.0, places=4)
        levels = {
            key: json.loads((ROOT / "data" / "levels" / f"{key}.json").read_text(encoding="utf-8"))
            for key in ("l1", "l2", "l3")
        }
        population = json.loads((ROOT / "data" / "population.json").read_text(encoding="utf-8"))["zones"]
        for key, level in levels.items():
            record = self.cost["levels"][level["id"]]
            for kind in ("distance", "learned"):
                result = evaluate_route(
                    record[kind]["path"],
                    nofly=level["nofly"],
                    height=level["height"],
                    em=level["em"],
                    population=population,
                    wind=level["wind"],
                )
                self.assertTrue(result["legal"], f"{level['id']} {kind}")
                self.assertAlmostEqual(result["score"], record[kind]["metrics"]["score"], delta=1.0)


if __name__ == "__main__":
    unittest.main()
