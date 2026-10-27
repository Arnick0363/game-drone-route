import { evaluateRoute, explain, formatDuration, formatMeters, routeDetection } from "./model.js";

const BOARD_KEY = "g2-macau-drone-board-v1";
const state = {
  model: null,
  population: [],
  baseline: null,
  perception: null,
  costModel: null,
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
  learned: null,
  layerOn: {
    nofly: true,
    height: true,
    population: true,
    wind: true,
    detect: true,
    compare: true,
    em: true,
  },
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

document.addEventListener("change", (event) => {
  const input = event.target.closest("[data-layer]");
  if (!input) return;
  state.layerOn[input.dataset.layer] = input.checked;
  if (input.dataset.layer === "compare") drawCompare();
  applyLayers();
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
  const [model, populationFile, baseline, perception, costModel, l1, l2, l3] = await Promise.all([
    loadJson("data/model.json"),
    loadJson("data/population.json"),
    loadJson("data/baseline.json"),
    loadOptional("data/perception.json"),
    loadOptional("data/cost_model.json"),
    loadJson("data/levels/l1.json"),
    loadJson("data/levels/l2.json"),
    loadJson("data/levels/l3.json"),
  ]);
  state.model = model;
  state.population = populationFile.zones;
  state.baseline = baseline;
  state.perception = perception;
  state.costModel = costModel;
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
  layers.wind = L.layerGroup().addTo(map);
  layers.detect = L.layerGroup().addTo(map);
  layers.compare = L.layerGroup().addTo(map);
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

async function loadOptional(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
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
  state.learned = null;
  state.planningS = 0;
  state.layerOn.compare = true;
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
  state.learned = null;
  state.layerOn.compare = false;
  render();
  drawRoute();
  drawCompare();
  applyLayers();
}

function retry() {
  if (!state.level) return;
  state.phase = "brief";
  state.points = [];
  state.delivered = 0;
  state.player = null;
  state.ai = null;
  state.learned = null;
  state.layerOn.compare = true;
  render();
  drawRoute();
  drawCompare();
  applyLayers();
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
  const distancePath = state.costModel?.levels?.[level.id]?.distance?.path || state.baseline.levels[level.id].path;
  state.ai = evaluateRoute(distancePath, context(level));
  state.learned = state.costModel?.levels?.[level.id]?.learned || null;
  state.layerOn.compare = true;
  if (state.player.legal) remember(level, state.player);
  layers.cursor.setLatLngs([]);
  render();
  drawRoute();
  drawCompare();
  applyLayers();
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
    <h3 style="margin-top:14px;font-size:14px">图层</h3>
    ${layerToggles()}
    <div class="legend">
      <span><i class="swatch" style="background:#b42318"></i>禁飞</span>
      <span><i class="swatch" style="background:#d97706"></i>限高</span>
      <span><i class="swatch" style="background:#6d5bd0"></i>电磁谨慎</span>
      <span><i class="swatch" style="background:#7a1f1f"></i>高人口密度</span>
      <span><i class="swatch" style="background:#db2777"></i>检测样本</span>
      <span><i class="swatch" style="background:#334155"></i>距离最短</span>
      <span><i class="swatch" style="background:#c2410c"></i>学习代价</span>
    </div>
    ${modelCard()}
    ${routeCompare()}
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
  const learned = state.learned?.metrics;
  return `<div class="debrief">
    <h3>${player.legal ? "这趟可以飞" : "这趟不能飞"}</h3>
    <table class="compare">
      <thead><tr><th></th><th>你</th><th>距离最短</th><th>学习代价</th></tr></thead>
      <tbody>
        ${row3("航程", formatMeters(player.distance_m), formatMeters(ai.distance_m), learned ? formatMeters(learned.distance_m) : "—")}
        ${row3("飞行时间", formatDuration(player.flight_time_s), formatDuration(ai.flight_time_s), learned ? formatDuration(learned.flight_time_s) : "—")}
        ${row3("限高段", formatMeters(player.height_m), formatMeters(ai.height_m), learned ? formatMeters(learned.height_m) : "—")}
        ${row3("综合代价", scoreText(player), scoreText(ai), learned && learned.legal ? Math.round(learned.score).toLocaleString("zh-CN") : "—")}
        ${row3("人口加权", densityText(player), densityText(ai), learned && learned.exposure_density != null ? String(Math.round(learned.exposure_density)) : "—")}
        ${row3("检测密度", detectText(playerDetect()), detectText(state.costModel?.levels?.[state.level.id]?.distance?.metrics?.detect_density), detectText(learned?.detect_density))}
      </tbody>
    </table>
    ${lines}
    ${state.costModel?.levels?.[state.level.id]?.note ? `<p>${esc(state.costModel.levels[state.level.id].note)}</p>` : ""}
    <p class="fine">规划用时 ${formatDuration(state.planningS)}，不进综合代价。距离最短路只最小化米数。学习代价路把梯度提升树的预测值当作边权。</p>
  </div>`;
}

function row3(label, player, distance, learned) {
  return `<tr><td>${label}</td><td>${player}</td><td>${distance}</td><td>${learned}</td></tr>`;
}

function playerDetect() {
  if (!state.player || !state.points.length) return null;
  return routeDetection(state.points.map((point) => [point.lon, point.lat]), state.perception?.sites || [], state.costModel?.sigma_m || state.perception?.sigma_m || 450, state.model);
}

function detectText(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return Number(value).toFixed(3);
}

function layerToggles() {
  const items = [
    ["nofly", "禁飞"],
    ["height", "限高"],
    ["population", "人口"],
    ["wind", "风"],
    ["detect", "检测风险"],
    ["compare", "对比航线"],
    ["em", "电磁"],
  ];
  return `<div class="layers">${items.map(([key, label]) => `<label><input type="checkbox" data-layer="${key}" ${state.layerOn[key] ? "checked" : ""}/>${label}</label>`).join("")}</div>`;
}

function modelCard() {
  const model = state.costModel;
  if (!model?.gbdt) return `<p class="fine">代价模型还没有生成。先运行 scripts/learn_cost.py。</p>`;
  const ranked = Object.entries(model.gbdt.importance).sort((left, right) => right[1] - left[1]);
  const top = ranked[0];
  const topName = model.feature_names_zh?.[top[0]] || top[0];
  const chart = model.chart ? `<img class="fit-chart" src="${esc(model.chart)}" alt="留出集上预测代价和教学标签的散点" />` : "";
  return `<div class="model-card">
    <h3>教学代价模型</h3>
    <p class="fine">梯度提升树留出集 MAE ${model.gbdt.mae.toFixed(1)}，R² ${model.gbdt.r2.toFixed(3)}，排序相关 ${model.gbdt.spearman.toFixed(3)}。样本 ${model.n}，其中 ${model.n_test} 条留出。最重要的特征是${esc(topName)}。</p>
    <p class="fine">线性回归对照 MAE ${model.linear.mae.toFixed(1)}，R² ${model.linear.r2.toFixed(3)}。寻路用树，不用这条直线。</p>
    ${chart}
  </div>`;
}

function routeCompare() {
  const item = state.costModel?.levels?.[state.level?.id];
  if (!item) return "";
  const distance = item.distance.metrics;
  const learned = item.learned.metrics;
  return `<table class="compare">
    <thead><tr><th></th><th>距离最短</th><th>学习代价</th></tr></thead>
    <tbody>
      <tr><td>航程</td><td>${formatMeters(distance.distance_m)}</td><td>${formatMeters(learned.distance_m)}</td></tr>
      <tr><td>综合代价</td><td>${Math.round(distance.score).toLocaleString("zh-CN")}</td><td>${Math.round(learned.score).toLocaleString("zh-CN")}</td></tr>
      <tr><td>限高段</td><td>${formatMeters(distance.height_m)}</td><td>${formatMeters(learned.height_m)}</td></tr>
      <tr><td>检测密度</td><td>${detectText(distance.detect_density)}</td><td>${detectText(learned.detect_density)}</td></tr>
    </tbody>
  </table>
  <p class="fine">${esc(item.note)}</p>`;
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
  drawWind();
  drawDetect();
  drawCompare();
  applyLayers();
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
  if (state.phase === "review") drawCompare();
}

function drawCompare() {
  if (!layers.compare) return;
  layers.compare.clearLayers();
  const item = state.costModel?.levels?.[state.level?.id];
  if (!item) return;
  const distance = item.distance.path.map(([lon, lat]) => [lat, lon]);
  const learned = item.learned.path.map(([lon, lat]) => [lat, lon]);
  layers.compare.addLayer(L.polyline(distance, { color: "#334155", weight: 3, dashArray: "8 7", interactive: false }));
  layers.compare.addLayer(L.polyline(learned, { color: "#c2410c", weight: 4, interactive: false }));
}

function drawWind() {
  layers.wind.clearLayers();
  const level = state.level;
  if (!level) return;
  const bearing = (Number(level.wind.from_deg) + 180) % 360;
  const icon = L.divIcon({
    className: "",
    html: `<div class="wind-pin"><b style="transform:rotate(${bearing}deg)">▲</b><em>${level.wind.speed_ms}</em></div>`,
    iconSize: [54, 28],
    iconAnchor: [14, 14],
  });
  const bounds = levelBounds(level);
  const center = bounds.getCenter();
  layers.wind.addLayer(L.marker(center, { icon, interactive: false, zIndexOffset: 300 }));
  layers.wind.addLayer(L.marker([level.vertiport.lat, level.vertiport.lon], { icon, interactive: false, zIndexOffset: 300 }));
}

function drawDetect() {
  layers.detect.clearLayers();
  const sites = state.perception?.status === "ok" ? state.perception.sites || [] : [];
  const level = state.level;
  if (!level || !sites.length) return;
  const bounds = levelBounds(level).pad(0.6);
  for (const site of sites) {
    if (!bounds.contains([site.lat, site.lon])) continue;
    const marker = L.circleMarker([site.lat, site.lon], {
      radius: 7 + Math.min(Number(site.risk) || 0, 8),
      color: "#9d174d",
      weight: 1,
      fillColor: "#db2777",
      fillOpacity: 0.5,
    });
    marker.bindTooltip(detectTip(site), { sticky: true });
    layers.detect.addLayer(marker);
  }
}

function detectTip(site) {
  const names = {
    person: "人", bicycle: "自行车", car: "汽车", motorcycle: "摩托车", bus: "公交",
    truck: "卡车", bird: "鸟", airplane: "飞机", boat: "船", "traffic light": "信号灯", "stop sign": "停车标志",
  };
  const counts = {};
  for (const box of site.boxes || []) {
    const label = names[box.cls] || box.cls;
    counts[label] = (counts[label] || 0) + 1;
  }
  const summary = Object.entries(counts).map(([label, count]) => `${label} ${count}`).join("、") || "没有计入风险的目标";
  const credit = site.license ? ` ${site.license}` : "";
  return `${site.title}：${summary}。风险 ${Number(site.risk).toFixed(2)}。${credit}`;
}

function applyLayers() {
  const groups = [
    ["population", layers.population],
    ["em", layers.em],
    ["height", layers.height],
    ["nofly", layers.nofly],
    ["wind", layers.wind],
    ["detect", layers.detect],
    ["compare", layers.compare],
  ];
  for (const [name, group] of groups) {
    if (!group || !map) continue;
    if (state.layerOn[name]) {
      if (!map.hasLayer(group)) group.addTo(map);
    } else if (map.hasLayer(group)) {
      map.removeLayer(group);
    }
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
  layers.wind.clearLayers();
  layers.detect.clearLayers();
  layers.compare.clearLayers();
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
