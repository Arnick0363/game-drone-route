"""下载可公开访问的源数据，供场景构建离线复用。

- OpenStreetMap：机场、医院/卫生中心、带高度的建筑物（Overpass）
- 澳门统计暨普查局 Census GIS：2021 人口普查 23 个统计分区的人口密度

网络失败时不覆盖已有文件。
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = "must-portfolio-g2/1.0 (academic low-altitude routing demo)"

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

OVERPASS_QUERY = """
[out:json][timeout:120];
(
  way["aeroway"~"aerodrome|runway"](22.10,113.52,22.22,113.65);
  relation["aeroway"="aerodrome"](22.10,113.52,22.22,113.65);
  node["amenity"~"hospital|clinic|ferry_terminal"](22.10,113.52,22.22,113.62);
  way["amenity"~"hospital|clinic"](22.10,113.52,22.22,113.62);
  node["name"~"澳門科技大學|澳门科技大学|科大醫院|鏡湖|仁伯爵"](22.10,113.52,22.22,113.62);
  way["name"~"澳門科技大學|澳门科技大学|科大醫院"](22.10,113.52,22.22,113.62);
  way["building"]["height"](22.10,113.52,22.22,113.62);
  node["man_made"="communications_tower"](22.18,113.53,22.21,113.56);
);
out geom tags;
""".strip()

DSEC_QUERY = (
    "https://www.dsec.gov.mo/CensusGIS/rest/services/Production2021/"
    "Census2021_Density23_TW_v2/MapServer/1/query"
)


def _get(url: str, data: bytes | None = None, timeout: int = 120) -> bytes:
    req = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": UA, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_overpass(dest: Path) -> None:
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        try:
            body = urllib.parse.urlencode({"data": OVERPASS_QUERY}).encode()
            raw = _get(url, data=body, timeout=150)
            payload = json.loads(raw.decode("utf-8"))
            if "elements" not in payload:
                raise RuntimeError(f"Overpass 响应缺少 elements: {url}")
            dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            print(f"OSM {len(payload['elements'])} elements -> {dest} via {url}")
            return
        except Exception as exc:  # noqa: BLE001 — 换镜像继续
            last_error = exc
            print(f"Overpass 失败 {url}: {exc}")
    raise RuntimeError(f"全部 Overpass 镜像失败: {last_error}")


def fetch_dsec(dest: Path) -> None:
    # 服务把 outSR=4326 转错到 4°E 附近。这里保存澳门网格原始坐标，构建时用 EPSG:8433。
    params = urllib.parse.urlencode(
        {
            "where": "1=1",
            "outFields": "Zona,Desc_C,Desc_CN,Desc_C1,Desc_P,density,hh_count",
            "returnGeometry": "true",
            "f": "json",
        }
    )
    raw = _get(f"{DSEC_QUERY}?{params}", timeout=90)
    payload = json.loads(raw.decode("utf-8"))
    features = payload.get("features") or []
    if len(features) < 10 or payload.get("spatialReference", {}).get("wkid") != 3064:
        raise RuntimeError(f"DSEC 响应异常: n={len(features)} sr={payload.get('spatialReference')}")
    dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"DSEC {len(features)} zones -> {dest}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    fetch_overpass(RAW / "osm_macau.json")
    fetch_dsec(RAW / "dsec_zones_2021.json")


if __name__ == "__main__":
    main()
