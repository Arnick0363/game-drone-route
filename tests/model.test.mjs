import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { detectionAt, evaluateRoute, routeDetection } from "../js/model.js";

const payload = JSON.parse(readFileSync(new URL("../data/fixtures/geometry_cases.json", import.meta.url), "utf8"));

for (const item of payload.cases) {
  test(item.name, () => {
    const result = evaluateRoute(item.route, {
      nofly: item.nofly,
      height: item.height,
      wind: item.wind,
      model: payload.model,
    });
    assert.equal(result.legal, item.expect.legal);
    assert.ok(Math.abs(result.distance_m - item.expect.distance_m) < 1);
    assert.ok(Math.abs(result.height_m - item.expect.height_m) < 2);
    if (item.expect.score == null) {
      assert.equal(result.score, null);
    } else {
      assert.ok(Math.abs(result.flight_time_s - item.expect.flight_time_s) < 0.5);
      assert.ok(Math.abs(result.score - item.expect.score) < 8);
    }
  });
}

test("detection kernel is one at the camera and fades with distance", () => {
  const model = payload.model;
  const sites = [{ lon: 113.54, lat: 22.16, risk: 2 }];
  assert.ok(Math.abs(detectionAt(113.54, 22.16, sites, 450, model) - 2) < 1e-9);
  const away = detectionAt(113.55, 22.16, sites, 450, model);
  assert.ok(away < 2);
  assert.ok(away > 0);
  const route = routeDetection([[113.54, 22.16], [113.541, 22.16]], sites, 450, model);
  assert.ok(route > 0);
  assert.ok(route < 2);
});

test("stored routes keep a finite detection density", () => {
  const cost = JSON.parse(readFileSync(new URL("../data/cost_model.json", import.meta.url), "utf8"));
  const perception = JSON.parse(readFileSync(new URL("../data/perception.json", import.meta.url), "utf8"));
  assert.equal(perception.status, "ok");
  assert.ok(perception.sites.every((site) => Array.isArray(site.boxes)));
  for (const id of ["L1", "L2", "L3"]) {
    const learned = cost.levels[id].learned;
    const distance = cost.levels[id].distance;
    assert.ok(learned.path.length >= 2);
    assert.ok(distance.path.length >= 2);
    assert.equal(typeof learned.metrics.detect_density, "number");
    assert.ok(Number.isFinite(learned.metrics.score));
  }
  const jsDensity = routeDetection(cost.levels.L2.distance.path, perception.sites, cost.sigma_m);
  assert.ok(Math.abs(jsDensity - cost.levels.L2.distance.metrics.detect_density) < 0.02);
});

test("real baseline stays close to the python score", () => {
  const baseline = JSON.parse(readFileSync(new URL("../data/baseline.json", import.meta.url), "utf8"));
  const population = JSON.parse(readFileSync(new URL("../data/population.json", import.meta.url), "utf8")).zones;
  for (const id of ["L1", "L2", "L3"]) {
    const level = JSON.parse(readFileSync(new URL(`../data/levels/${id.toLowerCase()}.json`, import.meta.url), "utf8"));
    const result = evaluateRoute(baseline.levels[id].path, {
      nofly: level.nofly,
      height: level.height,
      em: level.em,
      population,
      wind: level.wind,
      model: baseline.model,
    });
    assert.equal(result.legal, true, id);
    const stored = baseline.levels[id].metrics.score;
    assert.ok(Math.abs(result.score - stored) / stored < 0.01, `${id} ${result.score} vs ${stored}`);
  }
});
