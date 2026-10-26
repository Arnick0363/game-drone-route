import { evaluateRoute, explain, formatDuration, formatMeters } from "./model.js";

const BOARD_KEY = "g2-macau-drone-board-v1";
const state = {
  model: null,
  population: [],
  baseline: null,
  levels: [],
  level: null,
  phase: "pick",
  points: [],
  delivered: 0,
  startedAt: 0,
  planningS: 0,
  callsign: "规划员",
  player: null,
  ai: null,
};

const layers = {};
let map;
let toastTimer;
let clockTimer;

const briefEl = document.querySelector("#brief");
const sideEl = document.querySelector("#side");
const pickerEl = document.querySelector("#picker");
const missionEl = document.querySelector("#mission-list");

document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const action = target.dataset.action;
  if (action === "rules") document.querySelector("#rules").showModal();
  if (action === "pick") openLevel(target.dataset.level);
  if (action === "back") showPicker();
  if (action === "start") startPlanning();
  if (action === "undo") undo();
  if (action === "clear") clearRoute();
  if (action === "submit") submit();
  if (action === "retry") retry();
  if (action === "next") openLevel(target.dataset.level);
  if (action === "forget") {
    localStorage.removeItem(BOARD_KEY);
    render();
  }
});

document.addEventListener("input", (event) => {
  if (event.target.id === "callsign") state.callsign = event.target.value.trim() || "规划员";
});

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea")) return;
  if (event.key === "Backspace") {
    event.preventDefault();
    undo();
  }
  if (event.key === "Enter" && state.phase === "play") submit();
});

init().catch((error) => {
  briefEl.innerHTML = `<p>场景没有加载：${esc(error.message)}。请在项目根目录运行 <code>python -m http.server 8080</code>。</p>`;
});

async function init() {
  const [model, populationFile, baseline, l1, l2, l3] = await Promise.all([
    loadJson("data/model.json"),
    loadJson("data/population.json"),
    loadJson("data/baseline.json"),
    loadJson("data/levels/l1.json"),
    loadJson("data/levels/l2.json"),
    loadJson("data/levels/l3.json"),
  ]);
  state.model = model;
  state.population = populationFile.zones;
  state.baseline = baseline;
  state.levels = [l1, l2, l3];
  missionEl.innerHTML = state.levels.map(missionCard).join("");
  map = L.map("map", { zoomControl: false, doubleClickZoom: false }).setView([22.16, 113.56], 13);
  L.control.zoom({ position: "bottomleft" }).addTo(map);
  L.control.scale({ imperial: false, position: "bottomleft" }).addTo(map);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap 贡献者",
  }).addTo(map);
  layers.population = L.layerGroup().addTo(map);
  layers.em = L.layerGroup().addTo(map);
  layers.height = L.layerGroup().addTo(map);
  layers.nofly = L.layerGroup().addTo(map);
  layers.route = L.layerGroup().addTo(map);
  layers.markers = L.layerGroup().addTo(map);
  layers.cursor = L.polyline([], { color: "#0f5f5a", weight: 2, dashArray: "4 6", interactive: false }).addTo(map);
  map.on("click", onClick);
  map.on("dblclick", onDouble);
  map.on("mousemove", onMove);
  drawPopulation();
  window.setInterval(paintClock, 250);
  window.__drone = {
    state,
    map: () => map,
    start: startPlanning,
    add: (lon, lat) => addPoint({ lng: lon, lat }),
    submit,
  };
}

async function loadJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(url);
  return response.json();
}

function missionCard(level) {
  const tags = [
    `${level.deliveries.length} 个配送点`,
    `${level.nofly.length} 处禁飞`,
    level.height.length ? `${level.height.length} 片限高` : "无限高",
  ];
  return `<button class="mission" data-action="pick" data-level="${level.id}">
    <strong>${esc(level.title)}</strong>
    <span>${esc(level.subtitle)}</span>
    ${tags.map((tag) => `<em class="tag">${esc(tag)}</em>`).join("")}
  </button>`;
}

function openLevel(id) {
  const level = state.levels.find((item) => item.id === id);
  if (!level) return;
  state.level = level;
  state.phase = "brief";
  state.points = [];
  state.delivered = 0;
  state.player = null;
  state.ai = null;
  state.planningS = 0;
  pickerEl.classList.add("hidden");
  drawLevel();
  render();
  map.invalidateSize();
  map.fitBounds(levelBounds(level), { paddingTopLeft: [360, 90], paddingBottomRight: [330, 40] });
}

function showPicker() {
  state.phase = "pick";
  state.level = null;
  pickerEl.classList.remove("hidden");
  clearDrawings();
}

function startPlanning() {
  if (!state.level || state.phase === "play") return;
  const port = state.level.vertiport;
  state.points = [{ lon: port.lon, lat: port.lat, kind: "start" }];
  state.delivered = 0;
  state.phase = "play";
  state.startedAt = Date.now();
  state.player = null;
  state.ai = null;
  render();
  drawRoute();
}

function retry() {
  if (!state.level) return;
  state.phase = "brief";
  state.points = [];
  state.delivered = 0;
  state.player = null;
  state.ai = null;
  render();
  drawRoute();
}

function onClick(event) {
  if (state.phase !== "play") {
    if (state.phase === "brief") toast("先点「开始规划」，计时从这一刻起，不计入综合代价。");
    return;
  }
  window.clearTimeout(clockTimer);
  clockTimer = window.setTimeout(() => addPoint(event.latlng), 220);
}

function onDouble(event) {
  window.clearTimeout(clockTimer);
  if (state.phase !== "play") return;
  if (state.delivered < state.level.deliveries.length) addPoint(event.latlng);
  if (state.delivered >= state.level.deliveries.length) submit();
}

function onMove(event) {
  if (state.phase !== "play" || !state.points.length) {
    layers.cursor.setLatLngs([]);
    return;
  }
  const last = state.points[state.points.length - 1];
  layers.cursor.setLatLngs([[last.lat, last.lon], [event.latlng.lat, event.latlng.lng]]);
}

function addPoint(latlng) {
  if (state.phase !== "play") return;
  const level = state.level;
  const next = level.deliveries[state.delivered];
  const here = { lon: latlng.lng, lat: latlng.lat, kind: "via" };
  if (next) {
    const distance = metersBetween(here, next);
    if (distance <= level.capture_radius_m) {
      state.points.push({ lon: next.lon, lat: next.lat, kind: "delivery", id: next.id });
      state.delivered += 1;
      toast(`已送达 ${next.name}`);
      render();
      drawRoute();
      return;
    }
    const later = level.deliveries.slice(state.delivered + 1).find((stop) => metersBetween(here, stop) <= level.capture_radius_m);
    if (later) {
      toast(`要按顺序送。下一站是${next.name}`);
      return;
    }
  }
  const last = state.points[state.points.length - 1];
  if (last && metersBetween(last, here) < 15) return;
  if (state.points.length >= 40) {
    toast("途经点太多了，先撤销再重画。");
    return;
  }
  state.points.push(here);
  render();
  drawRoute();
}

function undo() {
  if (state.phase !== "play" || state.points.length <= 1) return;
  const removed = state.points.pop();
  if (removed.kind === "delivery") state.delivered -= 1;
  render();
  drawRoute();
}

function clearRoute() {
  if (state.phase !== "play") return;
  const port = state.level.vertiport;
  state.points = [{ lon: port.lon, lat: port.lat, kind: "start" }];
  state.delivered = 0;
  render();
  drawRoute();
}

function submit() {
  const level = state.level;
  if (!level || state.phase !== "play") return;
  if (state.delivered < level.deliveries.length) {
    const next = level.deliveries[state.delivered];
    toast(`还没送到${next.name}。双击只在全部送达后结束。`);
    return;
  }
  state.planningS = (Date.now() - state.startedAt) / 1000;
  state.phase = "review";
  const route = state.points.map((point) => [point.lon, point.lat]);
  state.player = evaluateRoute(route, context(level));
  state.ai = evaluateRoute(state.baseline.levels[level.id].path, context(level));
  if (state.player.legal) remember(level, state.player);
  layers.cursor.setLatLngs([]);
  render();
  drawRoute();
}

function context(level) {
  return {
    nofly: level.nofly,
    height: level.height,
    em: level.em,
    population: state.population,
    wind: level.wind,
    model: state.model,
  };
}

function remember(level, player) {
  const board = loadBoard();
  const rows = board[level.id] || [];
  rows.push({
    score: Math.round(player.score),
    distance_m: Math.round(player.distance_m),
    flight_time_s: Math.round(player.flight_time_s),
    planning_s: Math.round(state.planningS),
    callsign: state.callsign,
    when: new Date().toISOString(),
  });
  rows.sort((a, b) => a.score - b.score);
  board[level.id] = rows.slice(0, 5);
  try {
    localStorage.setItem(BOARD_KEY, JSON.stringify(board));
  } catch {
    toast("这台浏览器没能写下成绩榜。");
  }
}

function loadBoard() {
  try {
    return JSON.parse(localStorage.getItem(BOARD_KEY) || "{}");
  } catch {
    return {};
  }
}

function render() {
  const level = state.level;
  if (!level) {
    briefEl.innerHTML = "";
    sideEl.innerHTML = "";
    return;
  }
  const next = level.deliveries[state.delivered];
  const preview = state.phase === "play" && state.points.length > 1
    ? evaluateRoute(state.points.map((point) => [point.lon, point.lat]), context(level))
    : null;
  briefEl.innerHTML = `
    <p class="kicker">第 ${level.order} 关</p>
    <h2>${esc(level.title)}</h2>
    <p class="sub">${esc(level.subtitle)}</p>
    <p class="story">${esc(level.story)}</p>
    <p class="fine">${esc(level.vertiport.note)}</p>
    <ol class="stops">
      <li class="done"><span class="idx">起</span><span>${esc(level.vertiport.name)}</span></li>
      ${level.deliveries.map((stop, index) => {
        const cls = index < state.delivered ? "done" : index === state.delivered ? "next" : "";
        return `<li class="${cls}"><span class="idx">${index + 1}</span><span>${esc(stop.name)}<em class="payload">${esc(stop.payload)}</em></span></li>`;
      }).join("")}
    </ol>
    ${reviewBlock()}
    <div class="row">
      ${state.phase === "brief" ? '<button class="primary" data-action="start">开始规划</button>' : ""}
      ${state.phase === "play" ? '<button class="primary" data-action="submit">提交航线</button><button class="ghost" data-action="undo">撤销</button><button class="ghost" data-action="clear">清空</button>' : ""}
      ${state.phase === "review" ? `<button class="primary" data-action="retry">再飞一次</button>${nextLevelButton()}` : ""}
      <button class="ghost" data-action="back">返回任务</button>
    </div>`;
  sideEl.innerHTML = `
    <div class="metrics">
      <div class="metric"><span>规划用时</span><strong id="clock">${formatDuration(currentPlanning())}</strong></div>
      <div class="metric"><span>已送达</span><strong>${state.delivered}/${level.deliveries.length}</strong></div>
      <div class="metric"><span>航程</span><strong>${preview ? formatMeters(preview.distance_m) : state.player ? formatMeters(state.player.distance_m) : "—"}</strong></div>
      <div class="metric"><span>综合代价</span><strong>${scoreText(preview || state.player)}</strong></div>
    </div>
    <p class="fine">下一站：${next && state.phase !== "review" ? esc(next.name) : "—"}。点在 ${level.capture_radius_m} 米内会吸附。</p>
    <p class="fine">风从 ${windName(level.wind.from_deg)} 来，${level.wind.speed_ms} m/s。${esc(level.wind.note)}</p>
    <p class="fine">${batteryText(preview || state.player, level)}</p>
    <label class="fine">成绩榜代号
      <input id="callsign" class="callsign" value="${esc(state.callsign)}" maxlength="12" />
    </label>
    <div class="legend">
      <span><i class="swatch" style="background:#b42318"></i>禁飞</span>
      <span><i class="swatch" style="background:#d97706"></i>限高</span>
      <span><i class="swatch" style="background:#6d5bd0"></i>电磁谨慎</span>
      <span><i class="swatch" style="background:#7a1f1f"></i>高人口密度</span>
    </div>
    <h3 style="margin-top:14px;font-size:14px">本关成绩榜</h3>
    <p class="fine">只记录成功局，代价越低越好。存在这台浏览器里。</p>
    ${boardTable(level.id)}
    <button class="ghost" data-action="forget">清空本机成绩</button>`;
}

function reviewBlock() {
  if (state.phase !== "review" || !state.player) return "";
  const player = state.player;
  const ai = state.ai;
  const lines = explain(player, ai).map((line) => `<p>${esc(line)}</p>`).join("");
  return `<div class="debrief">
    <h3>${player.legal ? "这趟可以飞" : "这趟不能飞"}</h3>
    <table class="compare">
      <thead><tr><th></th><th>你</th><th>参考</th></tr></thead>
      <tbody>
        ${row("航程", formatMeters(player.distance_m), formatMeters(ai.distance_m))}
        ${row("飞行时间", formatDuration(player.flight_time_s), formatDuration(ai.flight_time_s))}
        ${row("限高段", formatMeters(player.height_m), formatMeters(ai.height_m))}
        ${row("综合代价", scoreText(player), scoreText(ai))}
        ${row("人口加权", densityText(player), densityText(ai))}
        ${row("电磁段", formatMeters(player.em_m), formatMeters(ai.em_m))}
        ${row("估算耗电", `${player.energy_wh.toFixed(1)} Wh`, `${ai.energy_wh.toFixed(1)} Wh`)}
      </tbody>
    </table>
    ${lines}
    <p class="fine">规划用时 ${formatDuration(state.planningS)}，不在综合代价里。参考航线是禁飞约束下的最短路。</p>
  </div>`;
}

function row(label, left, right) {
  return `<tr><td>${label}</td><td>${left}</td><td>${right}</td></tr>`;
}

function nextLevelButton() {
  const index = state.levels.findIndex((item) => item.id === state.level.id);
  const next = state.levels[index + 1];
  if (!next) return "";
  return `<button class="ghost" data-action="next" data-level="${next.id}">下一关</button>`;
}

function scoreText(result) {
  if (!result) return "—";
  if (!result.legal) return "失败";
  return Math.round(result.score).toLocaleString("zh-CN");
}

function densityText(result) {
  if (!result || result.exposure_density == null) return "—";
  return `${Math.round(result.exposure_density)}`;
}

function batteryText(result, level) {
  if (!result) return `演示电池 ${level.battery_wh} Wh。飞起来之后按 280 W 估算耗电。`;
  const pct = Math.round((100 * result.energy_wh) / level.battery_wh);
  const warn = pct > 100 ? "按这个估法，演示电池不够。" : `约占演示电池 ${pct}%。`;
  return `估算耗电 ${result.energy_wh.toFixed(1)} Wh，${warn}电量不进排行榜。`;
}

function boardTable(id) {
  const rows = loadBoard()[id] || [];
  if (!rows.length) return `<p class="fine">还没有成功记录。</p>`;
  return `<table class="board"><thead><tr><th>代号</th><th>代价</th><th>航程</th></tr></thead><tbody>
    ${rows.map((rowItem) => `<tr><td>${esc(rowItem.callsign)}</td><td>${rowItem.score}</td><td>${formatMeters(rowItem.distance_m)}</td></tr>`).join("")}
  </tbody></table>`;
}

function currentPlanning() {
  if (state.phase === "play") return (Date.now() - state.startedAt) / 1000;
  return state.planningS;
}

function paintClock() {
  const clock = document.querySelector("#clock");
  if (clock && state.phase === "play") clock.textContent = formatDuration(currentPlanning());
}

function drawPopulation() {
  layers.population.clearLayers();
  for (const zone of state.population) {
    const polygon = L.polygon(toLeafletRings(zone.rings), {
      stroke: false,
      fillColor: popColor(zone.density),
      fillOpacity: 0.28,
      interactive: false,
    });
    layers.population.addLayer(polygon);
  }
}

function drawLevel() {
  clearDrawings();
  const level = state.level;
  drawPolygons(layers.em, level.em, { color: "#6d5bd0", weight: 1, fillColor: "#6d5bd0", fillOpacity: 0.12 });
  drawPolygons(layers.height, level.height, { color: "#b54708", weight: 1, fillColor: "#f59e0b", fillOpacity: 0.32 });
  drawPolygons(layers.nofly, level.nofly, { color: "#9b1c1c", weight: 1.4, fillColor: "#b42318", fillOpacity: 0.38 });
  drawMarkers();
  drawRoute();
}

function drawPolygons(group, features, style) {
  group.clearLayers();
  for (const feature of features) {
    group.addLayer(L.polygon(toLeafletRings(feature.rings), { ...style, interactive: false }));
  }
}

function drawMarkers() {
  layers.markers.clearLayers();
  const level = state.level;
  const port = level.vertiport;
  layers.markers.addLayer(L.marker([port.lat, port.lon], {
    icon: L.divIcon({ className: "", html: `<div class="start-pin">起</div>`, iconSize: [22, 22], iconAnchor: [11, 11] }),
    interactive: false,
  }));
  level.deliveries.forEach((stop, index) => {
    const cls = index === state.delivered && state.phase !== "review" ? "drop-pin next" : "drop-pin";
    layers.markers.addLayer(L.marker([stop.lat, stop.lon], {
      icon: L.divIcon({ className: "", html: `<div class="${cls}">${index + 1}</div>`, iconSize: [22, 22], iconAnchor: [11, 11] }),
      interactive: false,
    }));
  });
  const next = level.deliveries[state.delivered];
  if (next && state.phase === "play") {
    layers.markers.addLayer(L.circle([next.lat, next.lon], {
      radius: level.capture_radius_m,
      color: "#1a8f86",
      weight: 1,
      dashArray: "4 4",
      fillOpacity: 0.05,
      interactive: false,
    }));
  }
}

function drawRoute() {
  layers.route.clearLayers();
  drawMarkers();
  if (state.points.length > 1) {
    const latlngs = state.points.map((point) => [point.lat, point.lon]);
    layers.route.addLayer(L.polyline(latlngs, { color: "#0f5f5a", weight: 4, interactive: false }));
    const result = state.player || (state.phase === "play"
      ? evaluateRoute(state.points.map((point) => [point.lon, point.lat]), context(state.level))
      : null);
    if (result) paintSegments(result, latlngs);
  }
  if (state.phase === "review" && state.baseline) {
    const ai = state.baseline.levels[state.level.id].path.map(([lon, lat]) => [lat, lon]);
    layers.route.addLayer(L.polyline(ai, { color: "#334155", weight: 3, dashArray: "8 7", interactive: false }));
  }
}

function paintSegments(result, latlngs) {
  result.segments.forEach((segment) => {
    const pair = [latlngs[segment.index], latlngs[segment.index + 1]];
    if (segment.nofly) {
      layers.route.addLayer(L.polyline(pair, { color: "#dc2626", weight: 7, interactive: false }));
    } else if (segment.height_m > 8) {
      layers.route.addLayer(L.polyline(pair, { color: "#d97706", weight: 6, interactive: false }));
    }
  });
}

function clearDrawings() {
  layers.em.clearLayers();
  layers.height.clearLayers();
  layers.nofly.clearLayers();
  layers.route.clearLayers();
  layers.markers.clearLayers();
  layers.cursor.setLatLngs([]);
}

function levelBounds(level) {
  const bounds = L.latLngBounds([]);
  const extendRings = (features) => features.forEach((feature) => feature.rings.forEach((ring) => ring.forEach(([lon, lat]) => bounds.extend([lat, lon]))));
  extendRings(level.nofly);
  extendRings(level.height);
  extendRings(level.em);
  bounds.extend([level.vertiport.lat, level.vertiport.lon]);
  level.deliveries.forEach((stop) => bounds.extend([stop.lat, stop.lon]));
  return bounds;
}

function toLeafletRings(rings) {
  return rings.map((ring) => ring.map(([lon, lat]) => [lat, lon]));
}

function metersBetween(a, b) {
  const [ax, ay] = projectPoint(a.lon, a.lat);
  const [bx, by] = projectPoint(b.lon, b.lat);
  return Math.hypot(ax - bx, ay - by);
}

function projectPoint(lon, lat) {
  const lat0 = (state.model.lat0_deg * Math.PI) / 180;
  const radius = state.model.earth_radius_m;
  return [(lon * Math.PI / 180) * radius * Math.cos(lat0), (lat * Math.PI / 180) * radius];
}

function popColor(density) {
  if (density >= 120000) return "#7a1f1f";
  if (density >= 90000) return "#c05621";
  if (density >= 60000) return "#dd8a1a";
  if (density >= 30000) return "#e6c15a";
  return "#c5d4b8";
}

function windName(deg) {
  const names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"];
  return names[Math.round(deg / 45) % 8];
}

function toast(message) {
  const node = document.querySelector("#toast");
  node.textContent = message;
  node.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => node.classList.remove("show"), 2400);
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}
