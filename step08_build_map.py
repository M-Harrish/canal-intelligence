"""Step 8: interactive map -> ayacut_map.html

A self-contained page (no server, no live API calls) showing every canal
segment coloured by desilting priority. Clicking a segment opens a panel with
its command area, conveyance health score, the satellite chip the score was
computed from, and a button that runs the pre-computed blockage simulation.

Every failure simulation is computed up front and embedded as JSON, so the
demo responds instantly and works offline.

Run:
  python step08_build_map.py chips   # download segment chips (once, ~4 min)
  python step08_build_map.py         # build the map
"""

import base64
import json
import pickle
import sys
import warnings

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd

import assumptions
import config
import step07_simulate_failure as sim

warnings.filterwarnings("ignore")

SEG_CHIPS_DIR = config.DATA_DIR / "seg_chips"
OUT_HTML = config.PROJECT_ROOT / "ayacut_map.html"
CHIP_PX = 128


def fetch_segment_chips():
    import urllib.request

    import ee
    ee.Initialize(project=config.EE_PROJECT)
    SEG_CHIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Chips only need geometry, so they can be fetched before ranking exists.
    src = (config.RANKED_GPKG if config.RANKED_GPKG.exists()
           else config.SEGMENTS_GPKG)
    segs = gpd.read_file(src).to_crs(config.CRS_WGS84)
    segs = segs[(segs["is_river"] == 0) & (segs["can_type"] != "Connector")]
    print(f"fetching chips for {len(segs)} segments from {src.name}")
    minx, miny, maxx, maxy = config.AOI_BBOX
    aoi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    img = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi).filterDate(*config.DRY_WINDOW)
           .linkCollection(csp, ["cs_cdf"])
           .map(lambda im: im.updateMask(
               im.select("cs_cdf").gte(config.CSP_CLEAR_THRESHOLD)))
           .median().select(["B4", "B3", "B2"]))

    for i, r in enumerate(segs.itertuples()):
        path = SEG_CHIPS_DIR / f"{r.seg_id}.png"
        if path.exists():
            continue
        mid = r.geometry.interpolate(0.5, normalized=True)
        region = ee.Geometry.Point([mid.x, mid.y]).buffer(300).bounds()
        url = img.getThumbURL({"region": region, "dimensions": CHIP_PX,
                               "format": "png", "min": 200, "max": 2500})
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"  {r.seg_id}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(segs)} chips")
    print(f"chips in {SEG_CHIPS_DIR}")


def chip_data_uri(seg_id):
    path = SEG_CHIPS_DIR / f"{seg_id}.png"
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def precompute_simulations(segs):
    with open(config.GRAPH_PKL, "rb") as f:
        D = pickle.load(f)
    cells = sim.load_cells_with_crop()

    sources = [n for n, d in D.nodes(data=True) if d.get("is_source")]
    before = set()
    for s in sources:
        before |= nx_descendants(D, s)

    out = {}
    for r in segs.itertuples():
        edge = next(((u, v) for u, v, d in D.edges(data=True)
                     if d["seg_id"] == r.seg_id), None)
        if edge is None:
            continue
        H = D.copy()
        H.remove_edge(*edge)
        after = set()
        for s in sources:
            after |= nx_descendants(H, s)
        lost_nodes = before - after
        lost = {d["seg_id"] for u, v, d in D.edges(data=True)
                if u in lost_nodes or d["seg_id"] == r.seg_id}

        lc = cells[cells["seg_id"].isin(lost)]
        by_crop = lc.groupby("crop")["area_ha"].sum().to_dict()
        ha = float(lc["area_ha"].sum())
        out[r.seg_id] = {
            "segments_cut": len(lost),
            "ha_lost": ha,
            "by_crop": {k: float(v) for k, v in by_crop.items()},
            "revenue": ha * assumptions.gross_revenue_per_ha(),
            "cost": assumptions.cost_to_desilt(r.length_m, r.can_type),
            "lost_ids": sorted(lost),
        }
    print(f"precomputed {len(out)} failure simulations")
    return out


def nx_descendants(G, n):
    import networkx as nx
    return nx.descendants(G, n) | {n}


PANEL_CSS = """
<style>
#ay-panel{position:fixed;top:10px;right:10px;width:330px;max-height:94vh;
 overflow-y:auto;background:#12161c;color:#e8eaed;font-family:system-ui,sans-serif;
 font-size:13px;border-radius:10px;padding:14px;z-index:9999;
 box-shadow:0 6px 24px rgba(0,0,0,.5);line-height:1.45}
#ay-panel h3{margin:0 0 2px;font-size:16px}
#ay-panel .sub{color:#9aa4b2;font-size:12px;margin-bottom:10px}
#ay-panel img{width:100%;border-radius:6px;border:1px solid #2a3441;margin:8px 0}
.ay-row{display:flex;justify-content:space-between;padding:4px 0;
 border-bottom:1px solid #222c38}
.ay-row b{font-weight:600}
.ay-btn{width:100%;margin-top:10px;padding:9px;border:0;border-radius:6px;
 background:#c0392b;color:#fff;font-size:14px;font-weight:600;cursor:pointer}
.ay-btn:hover{background:#e04b3a}
.ay-sim{margin-top:10px;padding:10px;background:#1b2230;border-radius:6px;
 border-left:3px solid #c0392b}
.ay-warn{margin-top:10px;font-size:11px;color:#8b95a3;font-style:italic}
.ay-tag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;
 font-weight:700;letter-spacing:.3px}
.t-sourced{background:#1e4620;color:#7ee787}
.t-derived{background:#3a3410;color:#e3b341}
.t-assumed{background:#4a1d1d;color:#ff7b72}
#ay-legend{position:fixed;bottom:18px;left:10px;background:#12161c;color:#e8eaed;
 padding:10px 12px;border-radius:8px;font-family:system-ui;font-size:12px;
 z-index:9999;box-shadow:0 4px 16px rgba(0,0,0,.5)}
#ay-legend .bar{height:9px;width:190px;border-radius:3px;margin:6px 0 3px;
 background:linear-gradient(to right,#2ecc71,#f1c40f,#e67e22,#c0392b)}
</style>
"""

PANEL_HTML = """
<div id="ay-panel">
  <h3>AYACUT</h3>
  <div class="sub">Canal desilting prioritisation &middot; Cauvery delta</div>
  <div id="ay-body">
    <p style="color:#9aa4b2">Click any canal segment on the map.</p>
    <p style="color:#9aa4b2;font-size:12px">Colour = desilting priority
    (command area &times; conveyance risk &times; network centrality).
    Red is most urgent.</p>
  </div>
</div>
<div id="ay-legend">
  <b>Desilting priority</b>
  <div class="bar"></div>
  <div style="display:flex;justify-content:space-between;color:#9aa4b2">
    <span>low</span><span>urgent</span></div>
</div>
"""

PANEL_JS = """
<script>
const AY_SIM = __SIMS__;
const AY_CHIPS = __CHIPS__;
const AY_REV_PER_HA = __REVHA__;
const AY_COST_TAG = "__COSTTAG__";
function ayFmt(n,d){return n.toLocaleString('en-IN',{maximumFractionDigits:d||0});}
function ayShow(p){
  const s = AY_SIM[p.seg_id] || null;
  let html = `<h3>${p.can_name||'(unnamed canal)'}</h3>
    <div class="sub">${p.can_type} &middot; ${p.prj_name||''} &middot;
      rank #${p.rank} of ${p.n_total}</div>`;
  const chip = AY_CHIPS[p.seg_id];
  if(chip) html += `<img src="${chip}" alt="satellite chip">
    <div style="color:#9aa4b2;font-size:11px;margin-top:-4px">
    Sentinel-2 dry season, ~600 m across, segment midpoint</div>`;
  html += `
    <div class="ay-row"><span>Length</span><b>${ayFmt(p.length_m/1000,1)} km</b></div>
    <div class="ay-row"><span>Command area</span><b>${ayFmt(p.command_area_ha)} ha</b></div>
    <div class="ay-row"><span>Conveyance health</span><b>${p.health_score.toFixed(2)}</b></div>
    <div class="ay-row"><span>Network centrality</span><b>${p.bc_norm.toFixed(2)}</b></div>
    <div class="ay-row"><span>Priority score</span><b>${ayFmt(p.priority)}</b></div>
    <div class="ay-row"><span>Desilting cost</span><b>&#8377;${ayFmt(p.cost)}
      <span class="ay-tag t-derived">${AY_COST_TAG}</span></b></div>`;
  if(s){
    html += `<button class="ay-btn" onclick="aySim('${p.seg_id}')">
      &#9888; SIMULATE BLOCKAGE</button><div id="ay-simout"></div>`;
  }
  html += `<div class="ay-warn">Health score reflects surface signatures of
    impaired conveyance observed by satellite. It is not a measurement of
    bed-level sediment and does not replace ground inspection.</div>`;
  document.getElementById('ay-body').innerHTML = html;
}
function aySim(id){
  const s = AY_SIM[id]; if(!s) return;
  let crops = '';
  for(const [k,v] of Object.entries(s.by_crop))
    crops += `<div class="ay-row"><span>${k}</span><b>${ayFmt(v)} ha</b></div>`;
  document.getElementById('ay-simout').innerHTML = `
    <div class="ay-sim">
      <b style="color:#ff7b72">IF THIS SEGMENT FAILS</b>
      <div class="ay-row"><span>Segments cut off</span><b>${s.segments_cut}</b></div>
      <div class="ay-row"><span>Command area lost</span><b>${ayFmt(s.ha_lost)} ha</b></div>
      ${crops}
      <div class="ay-row"><span>Gross revenue at risk</span>
        <b>&#8377;${ayFmt(s.revenue/1e7,2)} cr</b></div>
      <div class="ay-row"><span>Cost to desilt</span>
        <b>&#8377;${ayFmt(s.cost)}</b></div>
      <div class="ay-row"><span>Benefit &divide; cost</span>
        <b>${s.cost>0?ayFmt(s.revenue/s.cost,1):'—'}&times;</b></div>
      <div class="ay-warn">Gross revenue at paddy MSP
        (&#8377;${ayFmt(AY_REV_PER_HA)}/ha/yr, single season, total supply loss
        assumed). Crop classes are inferred from Sentinel-2 seasonality, not a
        crop survey. Order-of-magnitude figure for prioritisation only.</div>
    </div>`;
}
</script>
"""


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "chips":
        return fetch_segment_chips()

    if not config.RANKED_GPKG.exists():
        print("no ranked_segments.gpkg — run step05_rank_segments.py first")
        return 1

    segs = gpd.read_file(config.RANKED_GPKG)
    sims = precompute_simulations(segs)
    segs = segs.to_crs(config.CRS_WGS84)

    segs["cost"] = [assumptions.cost_to_desilt(r.length_m, r.can_type)
                    for r in segs.itertuples()]
    segs["n_total"] = len(segs)
    chips = {s: u for s in segs["seg_id"] if (u := chip_data_uri(s))}
    print(f"{len(chips)}/{len(segs)} segments have a satellite chip")

    ctr = segs.geometry.union_all().centroid
    m = folium.Map(location=[ctr.y, ctr.x], zoom_start=10, tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite").add_to(m)
    folium.TileLayer("CartoDB positron", name="Map").add_to(m)

    colors = cm.LinearColormap(
        ["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"], vmin=0, vmax=1)

    def style(feat):
        p = feat["properties"]
        w = {"Main Canal": 6, "Branch Canal": 5, "Distributary": 4}.get(
            p["can_type"], 3)
        return {"color": colors(p["priority_norm"]), "weight": w, "opacity": 0.9}

    fields = ["seg_id", "can_name", "can_type", "prj_name", "rank", "length_m",
              "command_area_ha", "health_score", "bc_norm", "priority",
              "priority_norm", "cost", "n_total"]
    gj = folium.GeoJson(
        segs[fields + ["geometry"]].to_json(),
        style_function=style,
        highlight_function=lambda f: {"weight": 9, "color": "#00d4ff"},
        tooltip=folium.GeoJsonTooltip(
            fields=["can_name", "can_type", "rank", "command_area_ha"],
            aliases=["Canal", "Type", "Priority rank", "Command area (ha)"]),
        name="Canal segments",
    )
    gj.add_to(m)
    # topleft, not the default topright — the info panel sits top-right
    folium.LayerControl(position="topleft", collapsed=False).add_to(m)

    js = (PANEL_JS
          .replace("__SIMS__", json.dumps(sims))
          .replace("__CHIPS__", json.dumps(chips))
          .replace("__REVHA__", f"{assumptions.gross_revenue_per_ha():.0f}")
          .replace("__COSTTAG__", assumptions.DESILT_COST_PER_KM.status))
    m.get_root().html.add_child(folium.Element(PANEL_CSS + PANEL_HTML))
    m.get_root().html.add_child(folium.Element(js))
    # root.script renders BEFORE the map's own JS, so defer until load —
    # referencing the geojson variable directly here would throw and kill
    # the whole script block (blank map).
    m.get_root().script.add_child(folium.Element(
        f"window.addEventListener('load', function(){{"
        f"  {gj.get_name()}.eachLayer(function(l){{"
        f"    l.on('click', function(e){{ ayShow(l.feature.properties);"
        f"      L.DomEvent.stopPropagation(e); }});"
        f"  }});"
        f"}});"
    ))

    m.save(OUT_HTML)
    size_mb = OUT_HTML.stat().st_size / 1e6
    print(f"wrote {OUT_HTML} ({size_mb:.1f} MB)")
    print("open it in a browser — no server needed")


if __name__ == "__main__":
    sys.exit(main())
