"""用 Ultralytics YOLO11n 对公开地理标记照片做检测，写成拍摄点风险。

做法对应 Himat Shah 的 Yolo11_Wood_Surfaces：YOLO("yolo11n.pt")、imgsz=640、device=cpu。
这里只做预测，不用他的木材节疤数据和训练轮次。检测框只来自本次推理。
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from cost_features import CLASS_WEIGHTS, SIGMA_M, site_risk

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "perception"
WEIGHTS = ROOT / "data" / "raw" / "weights" / "yolo11n.pt"
OUT = ROOT / "data" / "perception.json"
CATALOG = RAW / "catalog.json"

HEADERS = {
    "User-Agent": "MacauLowAltitudeTeaching/1.0 (educational portfolio; local research replay)"
}
CONF = 0.25
IMGSZ = 640
MAX_SITES = 8

# 文件页上的相机坐标，不是推测。Commons 查询接口返回 429 时用这三张。
# 960px 是文件页列出的预览宽度，640px 缩略图会 400。
CURATED = [
    {
        "title": "File:Macau - Senado Square.jpg",
        "lat": 22.1933948435,
        "lon": 113.5397222824,
        "thumb_url": "https://thumb.wikimedia.org/wikipedia/commons/thumb/1/15/Macau_-_Senado_Square.jpg/960px-Macau_-_Senado_Square.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Macau_-_Senado_Square.jpg",
        "mime": "image/jpeg",
        "license": "CC BY 2.0",
        "artist": "Tristan Schmurr (kewl)",
        "attribution": "kewl, CC BY 2.0, via Wikimedia Commons",
        "file": "senado_a.jpg",
        "scene": "街景。文件页 GPS 约海拔 10 米。",
    },
    {
        "title": "File:Largo do Senado 1110301.jpg",
        "lat": 22 + 11 / 60 + 37.24 / 3600,
        "lon": 113 + 32 / 60 + 23.89 / 3600,
        "thumb_url": "https://thumb.wikimedia.org/wikipedia/commons/thumb/3/3d/Largo_do_Senado_1110301.jpg/960px-Largo_do_Senado_1110301.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Largo_do_Senado_1110301.jpg",
        "mime": "image/jpeg",
        "license": "CC BY-SA 4.0",
        "artist": "K.Y.K.Z.K.",
        "attribution": "K.Y.K.Z.K., CC BY-SA 4.0, via Wikimedia Commons",
        "file": "senado_b.jpg",
        "scene": "街景。文件页 GPS 约海拔 5 米。",
    },
    {
        "title": "File:East Lobby Venetian Macau 澳門威尼斯人東大堂 - panoramio.jpg",
        "lat": 22 + 8 / 60 + 53.58 / 3600,
        "lon": 113 + 33 / 60 + 42.26 / 3600,
        "thumb_url": "https://thumb.wikimedia.org/wikipedia/commons/thumb/0/0a/East_Lobby_Venetian_Macau_%E6%BE%B3%E9%96%80%E5%A8%81%E5%B0%BC%E6%96%AF%E4%BA%BA%E6%9D%B1%E5%A4%A7%E5%A0%82_-_panoramio.jpg/960px-East_Lobby_Venetian_Macau_%E6%BE%B3%E9%96%80%E5%A8%81%E5%B0%BC%E6%96%AF%E4%BA%BA%E6%9D%B1%E5%A4%A7%E5%A0%82_-_panoramio.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:East_Lobby_Venetian_Macau_%E6%BE%B3%E9%96%80%E5%A8%81%E5%B0%BC%E6%96%AF%E4%BA%BA%E6%9D%B1%E5%A4%A7%E5%A0%82_-_panoramio.jpg",
        "mime": "image/jpeg",
        "license": "CC BY 3.0",
        "artist": "lienyuan lee",
        "attribution": "lienyuan lee, CC BY 3.0, via Wikimedia Commons",
        "file": "venetian.jpg",
        "scene": "威尼斯人东大堂室内，不是低空航拍。坐标是文件页上的相机位置。",
    },
]


def dump(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_json(params: dict) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{query}",
        headers=HEADERS,
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())


def strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value or "")
    return re.sub(r"\s+", " ", text).strip()


def anchors() -> list[tuple[str, float, float]]:
    points = []
    for path in sorted((ROOT / "data" / "levels").glob("l*.json")):
        level = json.loads(path.read_text(encoding="utf-8"))
        points.append((level["id"], level["vertiport"]["lat"], level["vertiport"]["lon"]))
        for stop in level["deliveries"]:
            points.append((level["id"], stop["lat"], stop["lon"]))
    return points


def nearest_level(lat: float, lon: float, points: list[tuple[str, float, float]]) -> str:
    from geo import to_xy

    best_name = points[0][0]
    best = float("inf")
    px, py = to_xy(lon, lat)
    for name, alat, alon in points:
        ax, ay = to_xy(alon, alat)
        dist = (px - ax) ** 2 + (py - ay) ** 2
        if dist < best:
            best = dist
            best_name = name
    return best_name


def image_record(title: str, lat: float, lon: float, info: dict) -> dict | None:
    mime = info.get("mime") or ""
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        return None
    thumb = info.get("thumburl") or info.get("url")
    width = int(info.get("thumbwidth") or info.get("width") or 0)
    if not thumb or width < 280:
        return None
    meta = info.get("extmetadata") or {}
    return {
        "title": title,
        "lat": lat,
        "lon": lon,
        "thumb_url": thumb,
        "page_url": info.get("descriptionurl"),
        "mime": mime,
        "license": strip_html((meta.get("LicenseShortName") or {}).get("value", "")),
        "artist": strip_html((meta.get("Artist") or {}).get("value", ""))[:180],
        "attribution": strip_html((meta.get("Attribution") or {}).get("value", ""))[:220],
    }


def geosearch(points: list[tuple[str, float, float]]) -> list[dict]:
    found: dict[str, dict] = {}
    for _, lat, lon in points:
        payload = fetch_json(
            {
                "action": "query",
                "list": "geosearch",
                "gscoord": f"{lat}|{lon}",
                "gsradius": 1500,
                "gslimit": 6,
                "gsnamespace": 6,
                "format": "json",
            }
        )
        for item in payload.get("query", {}).get("geosearch", []):
            title = item.get("title") or ""
            lowered = title.lower()
            if any(word in lowered for word in ("map", "flag", "logo", "svg", "icon", "locator", "coat")):
                continue
            found[title] = {"title": title, "lat": item["lat"], "lon": item["lon"]}
    if not found:
        return []
    titles = list(found)
    records = []
    for start in range(0, len(titles), 8):
        chunk = titles[start : start + 8]
        payload = fetch_json(
            {
                "action": "query",
                "titles": "|".join(chunk),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": 640,
                "format": "json",
            }
        )
        for page in payload.get("query", {}).get("pages", {}).values():
            title = page.get("title")
            info = (page.get("imageinfo") or [None])[0]
            anchor = found.get(title)
            if not info or not anchor:
                continue
            record = image_record(title, anchor["lat"], anchor["lon"], info)
            if record:
                records.append(record)
    return records


def materialize_curated(points: list[tuple[str, float, float]]) -> list[dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    stored = []
    for record in CURATED:
        dest = RAW / record["file"]
        if not dest.exists() or dest.stat().st_size < 2000:
            fetch_bytes(record["thumb_url"], dest)
        item = dict(record)
        item["level_hint"] = nearest_level(item["lat"], item["lon"], points)
        stored.append(item)
    return stored


def choose_sites(points: list[tuple[str, float, float]]) -> tuple[list[dict], str]:
    if CATALOG.exists():
        cached = json.loads(CATALOG.read_text(encoding="utf-8"))
        if cached and all((RAW / item["file"]).exists() for item in cached):
            return cached, "复用 data/raw/perception/catalog.json"
    errors = []
    records = []
    source = ""
    try:
        records = geosearch(points[:1])
        source = "Wikimedia Commons geosearch，半径 1500 米"
    except Exception as exc:
        errors.append(f"geosearch 失败：{exc}")
    if len(records) < 1:
        records = materialize_curated(points)
        source = "Commons 查询接口不可用（" + "；".join(errors) + "）。改用三张文件页上带相机坐标的公开照片，预览宽 960 像素。"
        CATALOG.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return records, source

    from geo import to_xy

    def distance_to_route(record: dict) -> float:
        x, y = to_xy(record["lon"], record["lat"])
        best = float("inf")
        for _, lat, lon in points:
            ax, ay = to_xy(lon, lat)
            best = min(best, (x - ax) ** 2 + (y - ay) ** 2)
        return best

    records.sort(key=distance_to_route)
    chosen = records[:MAX_SITES]
    RAW.mkdir(parents=True, exist_ok=True)
    stored = []
    for index, record in enumerate(chosen, start=1):
        suffix = ".png" if record["mime"] == "image/png" else ".jpg"
        filename = f"site_{index:02d}{suffix}"
        dest = RAW / filename
        if not dest.exists() or dest.stat().st_size < 2000:
            fetch_bytes(record["thumb_url"], dest)
        record = dict(record)
        record["file"] = filename
        record["level_hint"] = nearest_level(record["lat"], record["lon"], points)
        stored.append(record)
    known = {item.get("title") for item in stored}
    for item in materialize_curated(points):
        if item["title"] not in known:
            stored.append(item)
    source += "；另加三张文件页上带相机坐标的照片（议事亭前地两张、威尼斯人东大堂一张）。"
    CATALOG.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    return stored, source


def resolve_weights() -> tuple[Path | None, str]:
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if WEIGHTS.exists() and WEIGHTS.stat().st_size > 1_000_000:
        return WEIGHTS, "data/raw/weights/yolo11n.pt，来自 Ultralytics 发布的 COCO nano 权重，本地复用"
    mirrors = [
        (
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
            "Ultralytics 发布的 yolo11n.pt",
        ),
        (
            "https://github.com/adeelshah110/Yolo11_Wood_Surfaces/raw/master/yolo11n.pt",
            "adeelshah110/Yolo11_Wood_Surfaces 仓库里的 yolo11n.pt（COCO nano 预训练，不是木材微调权重）",
        ),
    ]
    errors = []
    for url, label in mirrors:
        try:
            fetch_bytes(url, WEIGHTS)
            if WEIGHTS.exists() and WEIGHTS.stat().st_size > 1_000_000:
                return WEIGHTS, label
            errors.append(f"{label} 文件过小")
        except Exception as exc:
            errors.append(f"{label}：{exc}")
        if WEIGHTS.exists() and WEIGHTS.stat().st_size < 1_000_000:
            WEIGHTS.unlink()
    return None, "；".join(errors)


def run_detector(sites: list[dict], weight: Path) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(weight))
    detected = []
    for site in sites:
        path = RAW / site["file"]
        results = model.predict(source=str(path), imgsz=IMGSZ, device="cpu", conf=CONF, verbose=False)
        result = results[0]
        names = result.names
        boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = names[cls_id] if isinstance(names, dict) else names[cls_id]
                conf = float(box.conf[0])
                xyxy = [round(float(value), 1) for value in box.xyxy[0].tolist()]
                boxes.append({"cls": str(name), "conf": round(conf, 4), "xyxy": xyxy})
        height, width = result.orig_shape
        risk = round(site_risk(boxes), 4)
        detected.append(
            {
                **site,
                "width": int(width),
                "height": int(height),
                "boxes": boxes,
                "risk": risk,
            }
        )
        print(f"  {site['title']}: {len(boxes)} 框, 风险 {risk}")
    return detected


def main() -> None:
    points = anchors()
    try:
        sites, image_source = choose_sites(points)
    except Exception as exc:
        dump(
            {
                "status": "failed",
                "reason": f"公开图片没有取到：{exc}",
                "model": "yolo11n",
                "sites": [],
                "class_weights": CLASS_WEIGHTS,
                "sigma_m": SIGMA_M,
                "conf": CONF,
            }
        )
        print(f"感知失败：{exc}")
        return

    weight, weight_source = resolve_weights()
    if weight is None:
        dump(
            {
                "status": "failed",
                "reason": f"YOLO11n 权重没有得到：{weight_source}",
                "model": "yolo11n",
                "image_source": image_source,
                "sites": [],
                "class_weights": CLASS_WEIGHTS,
                "sigma_m": SIGMA_M,
                "conf": CONF,
            }
        )
        print(f"权重失败：{weight_source}")
        return

    try:
        detected = run_detector(sites, weight)
    except Exception as exc:
        dump(
            {
                "status": "failed",
                "reason": f"YOLO11n 推理失败：{exc}",
                "model": "yolo11n",
                "weights_source": weight_source,
                "image_source": image_source,
                "sites": [],
                "class_weights": CLASS_WEIGHTS,
                "sigma_m": SIGMA_M,
                "conf": CONF,
            }
        )
        print(f"推理失败：{exc}")
        return

    dump(
        {
            "status": "ok",
            "reason": None,
            "model": "yolo11n",
            "imgsz": IMGSZ,
            "device": "cpu",
            "conf": CONF,
            "weights_source": weight_source,
            "image_source": image_source,
            "sigma_m": SIGMA_M,
            "class_weights": CLASS_WEIGHTS,
            "limitation": "权重是 COCO 预训练的 YOLO11n，不是用澳门航拍训出来的。框在照片像素里，地图上只保留拍摄点坐标和风险和。",
            "borrowed_from": "https://github.com/adeelshah110/Yolo11_Wood_Surfaces/blob/master/Yolo_v1.ipynb",
            "sites": detected,
        }
    )
    print(f"已写出 {OUT}，样本 {len(detected)} 张")


if __name__ == "__main__":
    main()
