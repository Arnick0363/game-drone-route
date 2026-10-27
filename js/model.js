/** 与 scripts/score_model.py 同一套投影、禁飞判定和评分。 */

export const FALLBACK_MODEL = {
  lat0_deg: 22.16,
  earth_radius_m: 6378137,
  airspeed_ms: 12,
  height_airspeed_ms: 7,
  min_groundspeed_ms: 3,
  power_w: 280,
  headwind_power_per_ms: 0.06,
  nofly_eps_m: 0.4,
};

export function project(lon, lat, model = FALLBACK_MODEL) {
  const lat0 = (model.lat0_deg * Math.PI) / 180;
  const radius = model.earth_radius_m;
  return [(lon * Math.PI / 180) * radius * Math.cos(lat0), (lat * Math.PI / 180) * radius];
}

export function windVector(fromDeg, speed) {
  const to = ((fromDeg + 180) * Math.PI) / 180;
  return [speed * Math.sin(to), speed * Math.cos(to)];
}

export function unproject(x, y, model = FALLBACK_MODEL) {
  const lat0 = (model.lat0_deg * Math.PI) / 180;
  const radius = model.earth_radius_m;
  return [(x / (radius * Math.cos(lat0))) * 180 / Math.PI, (y / radius) * 180 / Math.PI];
}

export function detectionAt(lon, lat, sites, sigma, model = FALLBACK_MODEL) {
  if (!sites || !sites.length || !(sigma > 0)) return 0;
  const [x, y] = project(lon, lat, model);
  let total = 0;
  for (const site of sites) {
    const [sx, sy] = project(site.lon, site.lat, model);
    const distance = Math.hypot(x - sx, y - sy);
    total += Number(site.risk || 0) * Math.exp(-distance / sigma);
  }
  return total;
}

export function routeDetection(route, sites, sigma, model = FALLBACK_MODEL) {
  if (!route || route.length < 2 || !sites || !sites.length) return 0;
  let weighted = 0;
  let length = 0;
  for (let index = 0; index < route.length - 1; index += 1) {
    const a = project(route[index][0], route[index][1], model);
    const b = project(route[index + 1][0], route[index + 1][1], model);
    const segment = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (segment < 1e-6) continue;
    let local = 0;
    for (const fraction of [0.25, 0.5, 0.75]) {
      const [lon, lat] = unproject(a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction, model);
      local += detectionAt(lon, lat, sites, sigma, model);
    }
    weighted += segment * (local / 3);
    length += segment;
  }
  return length > 0 ? weighted / length : 0;
}

function hypot(dx, dy) {
  return Math.hypot(dx, dy);
}

function ringOpen(ring) {
  if (ring.length > 1) {
    const a = ring[0];
    const b = ring[ring.length - 1];
    if (a[0] === b[0] && a[1] === b[1]) return ring.slice(0, -1);
  }
  return ring;
}

function pointInRing(point, ring) {
  const poly = ringOpen(ring);
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0];
    const yi = poly[i][1];
    const xj = poly[j][0];
    const yj = poly[j][1];
    const cross = (yi > point[1]) !== (yj > point[1]);
    if (cross) {
      const x = ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi;
      if (point[0] < x) inside = !inside;
    }
  }
  return inside;
}

function distPointSeg(point, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return hypot(point[0] - a[0], point[1] - a[1]);
  let t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  return hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy));
}

function distToRing(point, ring) {
  const poly = ringOpen(ring);
  let best = Infinity;
  for (let i = 0; i < poly.length; i++) {
    best = Math.min(best, distPointSeg(point, poly[i], poly[(i + 1) % poly.length]));
  }
  return best;
}

function ringsOf(feature) {
  return feature.rings || [feature.ring];
}

function contains(point, rings, holesCountAsOut = true) {
  if (!pointInRing(point, rings[0])) return false;
  if (!holesCountAsOut) return true;
  for (let i = 1; i < rings.length; i++) {
    if (pointInRing(point, rings[i])) return false;
  }
  return true;
}

function segmentTs(a, b, ring) {
  const poly = ringOpen(ring);
  const ts = [0, 1];
  for (let i = 0; i < poly.length; i++) {
    const t = segmentHitT(a, b, poly[i], poly[(i + 1) % poly.length]);
    if (t !== null && t > 1e-8 && t < 1 - 1e-8) ts.push(t);
  }
  ts.sort((left, right) => left - right);
  const unique = [];
  for (const t of ts) {
    if (!unique.length || t - unique[unique.length - 1] > 1e-7) unique.push(t);
  }
  return unique;
}

function segmentHitT(a, b, c, d) {
  const rx = b[0] - a[0];
  const ry = b[1] - a[1];
  const sx = d[0] - c[0];
  const sy = d[1] - c[1];
  const denom = rx * sy - ry * sx;
  if (Math.abs(denom) < 1e-9) return null;
  const qpx = c[0] - a[0];
  const qpy = c[1] - a[1];
  const t = (qpx * sy - qpy * sx) / denom;
  const u = (qpx * ry - qpy * rx) / denom;
  if (t < -1e-8 || t > 1 + 1e-8 || u < -1e-8 || u > 1 + 1e-8) return null;
  return Math.min(1, Math.max(0, t));
}

function clipLength(a, b, rings, eps, deepOnly) {
  const splits = segmentTs(a, b, rings[0]);
  for (let h = 1; h < rings.length; h++) {
    for (const t of segmentTs(a, b, rings[h])) splits.push(t);
  }
  splits.sort((left, right) => left - right);
  const ts = [];
  for (const t of splits) {
    if (!ts.length || t - ts[ts.length - 1] > 1e-7) ts.push(Math.min(1, Math.max(0, t)));
  }
  const length = hypot(b[0] - a[0], b[1] - a[1]);
  let inside = 0;
  for (let i = 0; i < ts.length - 1; i++) {
    const t0 = ts[i];
    const t1 = ts[i + 1];
    if (t1 - t0 < 1e-8) continue;
    const tm = (t0 + t1) / 2;
    const point = [a[0] + (b[0] - a[0]) * tm, a[1] + (b[1] - a[1]) * tm];
    if (!contains(point, rings)) continue;
    if (deepOnly && distToRing(point, rings[0]) <= eps) continue;
    inside += (t1 - t0) * length;
  }
  return inside;
}

function groundSpeed(airspeed, heading, wind, model) {
  const headwind = -(wind[0] * heading[0] + wind[1] * heading[1]);
  return { speed: Math.max(model.min_groundspeed_ms, airspeed - headwind), headwind };
}

export function evaluateRoute(route, options) {
  const model = options.model || FALLBACK_MODEL;
  const nofly = options.nofly || [];
  const height = options.height || [];
  const population = options.population || [];
  const em = options.em || [];
  const windSpec = options.wind || { from_deg: 0, speed_ms: 0 };
  const eps = model.nofly_eps_m;
  const wind = windVector(windSpec.from_deg || 0, windSpec.speed_ms || 0);
  const noflyXY = nofly.map((feature) => ({ name: feature.name, rings: ringsOf(feature).map((ring) => ring.map(([lon, lat]) => project(lon, lat, model))) }));
  const heightXY = height.map((feature) => ringsOf(feature).map((ring) => ring.map(([lon, lat]) => project(lon, lat, model))));
  const emXY = em.map((feature) => ringsOf(feature).map((ring) => ring.map(([lon, lat]) => project(lon, lat, model))));
  const popXY = population.map((zone) => ({
    name: zone.name,
    density: zone.density,
    rings: zone.rings.map((ring) => ring.map(([lon, lat]) => project(lon, lat, model))),
  }));

  if (!route || route.length < 2) {
    return emptyResult("航线至少需要两个点");
  }

  let distance = 0;
  let flightTime = 0;
  let heightM = 0;
  let energy = 0;
  let headwindM = 0;
  let emM = 0;
  const hitNames = [];
  const segments = [];
  const popLength = Object.fromEntries(popXY.map((zone) => [zone.name, 0]));

  for (let index = 0; index < route.length - 1; index++) {
    const a = project(route[index][0], route[index][1], model);
    const b = project(route[index + 1][0], route[index + 1][1], model);
    const length = hypot(b[0] - a[0], b[1] - a[1]);
    let hit = null;
    let segHeight = 0;
    let segEm = 0;
    let segTime = 0;
    if (length > 1e-6) {
      const heading = [(b[0] - a[0]) / length, (b[1] - a[1]) / length];
      for (const obstacle of noflyXY) {
        const interior = clipLength(a, b, obstacle.rings, eps, true);
        const deepStart = contains(a, obstacle.rings) && distToRing(a, obstacle.rings[0]) > eps;
        const deepEnd = contains(b, obstacle.rings) && distToRing(b, obstacle.rings[0]) > eps;
        if (interior > eps || deepStart || deepEnd) {
          hit = obstacle.name;
          if (!hitNames.includes(obstacle.name)) hitNames.push(obstacle.name);
          break;
        }
      }
      for (const rings of heightXY) segHeight += clipLength(a, b, rings, eps, false);
      segHeight = Math.min(segHeight, length);
      const outside = Math.max(0, length - segHeight);
      const cruise = groundSpeed(model.airspeed_ms, heading, wind, model);
      const slow = groundSpeed(model.height_airspeed_ms, heading, wind, model);
      if (outside > 0) segTime += outside / cruise.speed;
      if (segHeight > 0) segTime += segHeight / slow.speed;
      if (cruise.headwind > 1) headwindM += length;
      const power = model.power_w * (1 + model.headwind_power_per_ms * Math.max(cruise.headwind, 0));
      energy += (power * segTime) / 3600;
      for (const rings of emXY) segEm += clipLength(a, b, rings, eps, false);
      for (const zone of popXY) popLength[zone.name] += clipLength(a, b, zone.rings, eps, false);
    }
    distance += length;
    flightTime += segTime;
    heightM += segHeight;
    emM += segEm;
    segments.push({ index, length_m: length, height_m: segHeight, em_m: segEm, nofly: hit });
  }

  let covered = 0;
  let weighted = 0;
  let topName = null;
  let topLen = 0;
  let peakName = null;
  let peakLen = 0;
  let peakDensity = null;
  let peakValue = 0;
  for (const zone of popXY) {
    const length = popLength[zone.name];
    if (length <= 0.5) continue;
    covered += length;
    weighted += length * zone.density;
    if (length > topLen) {
      topLen = length;
      topName = zone.name;
    }
    if (length * zone.density > peakValue) {
      peakValue = length * zone.density;
      peakName = zone.name;
      peakLen = length;
      peakDensity = zone.density;
    }
  }
  const legal = hitNames.length === 0;
  return {
    legal,
    hit_names: hitNames,
    distance_m: distance,
    flight_time_s: flightTime,
    height_m: heightM,
    score: legal ? distance + flightTime * model.airspeed_ms : null,
    energy_wh: energy,
    headwind_m: headwindM,
    exposure_density: covered > 1 ? weighted / covered : null,
    exposure_covered_m: covered,
    exposure_top_zone: topName,
    exposure_top_m: topLen,
    exposure_peak_zone: peakName,
    exposure_peak_m: peakLen,
    exposure_peak_density: peakDensity,
    uncovered_m: Math.max(0, distance - covered),
    em_m: emM,
    segments,
    reason: legal ? null : `穿越禁飞区：${hitNames.join("、")}`,
  };
}

function emptyResult(reason) {
  return {
    legal: false,
    hit_names: [],
    distance_m: 0,
    flight_time_s: 0,
    height_m: 0,
    score: null,
    energy_wh: 0,
    headwind_m: 0,
    exposure_density: null,
    exposure_covered_m: 0,
    exposure_top_zone: null,
    exposure_top_m: 0,
    exposure_peak_zone: null,
    exposure_peak_m: 0,
    exposure_peak_density: null,
    uncovered_m: 0,
    em_m: 0,
    segments: [],
    reason,
  };
}

export function formatMeters(value) {
  if (value >= 1000) return `${(value / 1000).toFixed(2)} km`;
  return `${Math.round(value)} m`;
}

export function formatDuration(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const min = Math.floor(total / 60);
  const sec = total % 60;
  return `${min}:${String(sec).padStart(2, "0")}`;
}

export function explain(player, ai) {
  const lines = [];
  if (!player.legal) {
    lines.push(`航线进入了「${player.hit_names[0] || "禁飞区"}」。禁飞是硬约束，本局失败，不写入成绩榜。`);
    if (ai && ai.legal) {
      lines.push(`参考航线只绕开禁飞区，航程 ${formatMeters(ai.distance_m)}。它优化的是距离，不是综合代价。`);
    }
    return lines;
  }
  if (!ai || !ai.legal) return lines;
  const extra = player.distance_m - ai.distance_m;
  const scoreDelta = player.score - ai.score;
  if (extra > 40 && scoreDelta < -15) {
    lines.push(`你比避障最短路多飞了 ${formatMeters(extra)}，综合代价反而低了 ${Math.abs(scoreDelta).toFixed(0)}。多出来的距离换掉了一部分限高或逆风时间。`);
  } else if (extra > 40 && scoreDelta > 30) {
    lines.push(`你比避障最短路多飞了 ${formatMeters(extra)}，综合代价也更高。这次绕行没有换来足够的时间。`);
  } else if (Math.abs(scoreDelta) <= 30 && Math.abs(extra) <= 40) {
    lines.push("这条航线和避障最短路几乎重合。它只保证躲开禁飞；限高和逆风要靠你自己决定绕不绕。");
  } else if (scoreDelta < 0) {
    lines.push("你的综合代价低于参考航线。参考航线只求禁飞约束下的最短距离，限高和逆风是飞完才结算的。");
  } else {
    lines.push("参考航线更短，按同一套规则综合代价也更低。在这组风和限高下，绕路并不划算。");
  }
  if (player.height_m > 5 || ai.height_m > 5) {
    lines.push(`城市峡谷限高段：你 ${formatMeters(player.height_m)}，参考 ${formatMeters(ai.height_m)}。限高内空速按 7 m/s，外面按 12 m/s。`);
  }
  if (player.exposure_density != null && ai.exposure_density != null) {
    lines.push(`落入 2021 分区的航段，长度加权人口密度：你 ${Math.round(player.exposure_density)} 人/km²，参考 ${Math.round(ai.exposure_density)} 人/km²。这个数不进排行榜。`);
    if (player.uncovered_m > 200) {
      lines.push(`另有 ${formatMeters(player.uncovered_m)} 没有落入分区（南岸、新填海或水域），不参加上面的平均。`);
    }
    if (player.exposure_peak_zone) {
      lines.push(`按「长度 × 密度」贡献最大的是${player.exposure_peak_zone}（密度 ${Math.round(player.exposure_peak_density)} 人/km²，穿过 ${formatMeters(player.exposure_peak_m)}）。`);
    }
  }
  if (player.em_m > 5) {
    lines.push(`有 ${formatMeters(player.em_m)} 进入机场电磁谨慎带。它只作提示，穿过不算失败。`);
  }
  return lines;
}
