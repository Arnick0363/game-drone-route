import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { evaluateRoute } from "../js/model.js";

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
