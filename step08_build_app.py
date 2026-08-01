

import json
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

import assumptions
import config
import crop_economics

warnings.filterwarnings("ignore")


def load_segments():
    segs = gpd.read_file(config.RANKED_GPKG).to_crs(config.CRS_WGS84)
    segs["district"] = [
        crop_economics.district_for_segment(s, p)
        for s, p in zip(segs["seg_id"], segs["prj_name"])
    ]
    return segs


def precompute_simulations(segs):
    """Run every segment's failure simulation once, at build time — that is
    what lets the app work offline with no Python backend behind it."""
    import contextlib
    import io
    import pickle

    import step07_simulate_failure as s7

    with open(config.GRAPH_PKL, "rb") as f:
        G = pickle.load(f)
    ranked = gpd.read_file(config.RANKED_GPKG)
    cells = s7.load_cells_with_crop()

    out = {}
    for i, sid in enumerate(segs["seg_id"]):
        try:
            # simulate() prints a full report; silence it for the bulk run
            with contextlib.redirect_stdout(io.StringIO()):
                r = s7.simulate(G, sid, cells, ranked)
            if r:
                out[sid] = {
                    k: (None if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in r.items() if k != "geometry"
                }
        except Exception as e:
            print(f"  sim failed for {sid}: {e}")
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(segs)} simulations")
    print(f"  {len(out)} simulations precomputed")
    return out


SEG_CHIPS_DIR = config.DATA_DIR / "seg_chips"
CHIP_PX = 128


def fetch_segment_chips():
    """One dry-season true-colour thumbnail per segment. Run once (~4 min).
    Only needs geometry, so it can run before the ranking exists."""
    import urllib.request

    import ee
    ee.Initialize(project=config.EE_PROJECT)
    SEG_CHIPS_DIR.mkdir(parents=True, exist_ok=True)

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
    return 0


def reach_names(canals):
    """Give every reach a unique, human-meaningful label.

    raw can_name is unusable in a picker: 68 of 133 segments are unnamed in the
    KML, and the named ones repeat — "Pullambadi Channel" alone is 18 edges.

    Derived labels come from real network data (parent canal, class, chainage)
    and are never invented — a plausible-looking Tamil canal name would read as
    official when nothing sourced it.

    Chainage runs along a reach's OWN canal, not from the Grand Anicut: "km 3.3
    of Pullambadi" is what an engineer would say.

    Returns {seg_id: {"name", "canal", "reach", "km_from", "km_to"}}.
    """
    import pickle

    import networkx as nx

    with open(config.GRAPH_PKL, "rb") as f:
        G = pickle.load(f)

    edge_of = {d["seg_id"]: (u, v) for u, v, d in G.edges(data=True) if "seg_id" in d}
    sources = [n for n, d in G.nodes(data=True) if d.get("is_source")]

    # network chainage: metres from the nearest oriented source to each node
    dist = {}
    for s in sources:
        for n, m in nx.single_source_dijkstra_path_length(
                G, s, weight="length_m").items():
            dist[n] = min(dist.get(n, float("inf")), m)

    def blank(v):
        return str(v).strip() if v is not None else ""

    raw, length = {}, {}
    for r in canals.itertuples():
        nm = blank(r.can_name)
        raw[r.seg_id] = "" if nm.lower() in ("", "nan", "none", "null") else nm
        length[r.seg_id] = float(r.length_m)

    def start_of(sid):
        uv = edge_of.get(sid)
        return dist.get(uv[0], float("inf")) if uv else float("inf")

    # head chainage of each named canal, so its reaches are measured from its
    # own offtake rather than from the anicut
    head = {}
    for sid, nm in raw.items():
        if nm:
            head[nm] = min(head.get(nm, float("inf")), start_of(sid))

    def named_parent(sid):
        """Nearest named canal upstream — what an unnamed reach takes off from."""
        uv = edge_of.get(sid)
        if not uv:
            return None
        cur, seen = uv[0], set()
        for _ in range(80):
            step = None
            for p in G.predecessors(cur):
                psid = G[p][cur].get("seg_id")
                if psid and psid not in seen:
                    step = (p, psid)
                    break
            if step is None:
                return None
            cur, psid = step
            seen.add(psid)
            if raw.get(psid):
                return raw[psid]
        return None

    def km(v):
        return "?" if v is None or v != v or v == float("inf") else f"{v / 1000:.1f}"

    # reach index within each named canal, ordered downstream
    order = {}
    for nm in set(v for v in raw.values() if v):
        sids = sorted([s for s, v in raw.items() if v == nm], key=start_of)
        for i, s in enumerate(sids):
            order[s] = (i + 1, len(sids))

    out = {}
    for sid, nm in raw.items():
        uv = edge_of.get(sid)
        a = start_of(sid)
        typ = str(canals.loc[canals["seg_id"] == sid, "can_type"].iloc[0])
        prj = str(canals.loc[canals["seg_id"] == sid, "prj_name"].iloc[0])

        if nm:
            # Graph distance isn't chainage along one line once a canal
            # branches, so label the offtake point, never a span.
            base = a - head.get(nm, 0.0)
            i, n = order.get(sid, (1, 1))
            pos = "head reach" if i == 1 else f"reach {i}/{n}"
            reach = f"{pos} · {length[sid] / 1000:.1f} km"
            if i > 1:
                reach += f" · from km {km(base)}"
            name = nm if n == 1 else f"{nm} · {pos}"
            canal = nm
        else:
            par = named_parent(sid)
            if par:
                off = a - head.get(par, 0.0)
                name = f"{typ} off {par} at km {km(off)}"
                # "Distributary" -> "Distributaries", not "Distributarys"
                plural = f"{typ[:-1]}ies" if typ.endswith("y") else f"{typ}s"
                canal = f"{plural} off {par}"
                reach = f"at km {km(off)} · {length[sid] / 1000:.1f} km"
            else:
                name = f"{typ}, {prj} at km {km(a)}"
                canal = f"{prj} · unnamed reaches"
                reach = f"{typ} at km {km(a)} · {length[sid] / 1000:.1f} km"
        out[sid] = {"name": name, "canal": canal, "reach": reach,
                    "km_from": round(a / 1000, 2)}

    # Two offtakes can share a parent and a chainage to one decimal; reach
    # length separates them, and it's the next thing visible on the ground.
    seen = {}
    for sid, v in sorted(out.items(), key=lambda kv: kv[1]["name"]):
        seen.setdefault(v["name"], []).append(sid)
    for nm, sids in seen.items():
        if len(sids) > 1:
            for sid in sids:
                out[sid]["name"] = f"{nm} ({length[sid] / 1000:.1f} km)"

    n_derived = sum(1 for s, v in raw.items() if not v)
    print(f"  named {len(out)} reaches ({n_derived} derived from network position, "
          f"{len(set(v['canal'] for v in out.values()))} canal groups)")
    return out


def build_payload():
    segs = load_segments()
    canals = segs[(segs["is_river"] == 0) & (segs["can_type"] != "Connector")].copy()
    print(f"{len(canals)} canal segments")
    names = reach_names(canals)

    sims = precompute_simulations(canals)

    feats = []
    for r in canals.itertuples():
        coords = [[round(x, 5), round(y, 5)] for x, y in r.geometry.coords]
        nm = names.get(r.seg_id, {})
        feats.append({
            "id": r.seg_id,
            "name": nm.get("name") or (r.can_name or "").strip() or "(unnamed reach)",
            # canal this reach belongs to, for grouping the pickers
            "canal": nm.get("canal") or (r.can_name or "").strip() or r.prj_name,
            # position within that canal, shown inside the group
            "reach": nm.get("reach", ""),
            # network chainage of the reach's upstream end, for ordering
            "kmFrom": nm.get("km_from", 0),
            # True where the name was derived rather than read from the KML
            "derived": not (r.can_name or "").strip(),
            "type": r.can_type,
            "project": r.prj_name,
            "district": r.district.title(),
            "km": round(r.length_m / 1000, 2),
            "area": round(r.command_area_ha),
            "local": round(r.local_area_ha),
            "health": round(float(r.health_score), 3),
            "cent": round(float(r.bc_norm), 3),
            "priority": round(float(r.priority)),
            "pnorm": round(float(r.priority_norm), 4),
            "rank": int(r.rank),
            "cost": round(assumptions.cost_to_desilt(r.length_m, r.can_type)),
            "coords": coords,
        })

    districts = {}
    for d in sorted({f["district"] for f in feats}):
        mix = crop_economics.crop_mix(d)
        districts[d] = {
            "mix": [[c, round(float(s), 4)] for c, s in mix.head(6).items()],
            "rev_ha": round(crop_economics.revenue_per_ha(d)),
            "water_mm": round(crop_economics.water_demand_mm(d)),
            "irr_dep": round(crop_economics.irrigation_dependency(d), 3),
            "rain_dev": round(crop_economics.rainfall_deficit(d) * -100, 1),
        }

    method = str(canals["method"].iloc[0]) if "method" in canals else "unknown"
    # No thumbnails embedded: a 128 px crop of a 5-15 m channel shows nothing
    # and cost 80% of the file size. fetch_segment_chips() is still there.
    return {
        "segments": feats,
        "sims": sims,
        "districts": districts,
        "method": method,
        "desilt_rate": assumptions.DESILT_COST_PER_KM.value,
        "bounds": [[float(canals.total_bounds[1]), float(canals.total_bounds[0])],
                   [float(canals.total_bounds[3]), float(canals.total_bounds[2])]],
    }


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Canal Intelligence — canal desilting priority</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:ui-sans-serif,system-ui,'Segoe UI',sans-serif;
     background:#0d1117;color:#e6edf3;height:100vh;display:flex;flex-direction:column}
header{padding:12px 20px;border-bottom:1px solid #21262d;display:flex;
       align-items:center;gap:22px;flex-shrink:0;background:#0d1117}
.brand{font-size:19px;font-weight:650;letter-spacing:.02em}
.brand span{color:#7d8590;font-weight:400;font-size:13px;margin-left:10px}
nav{display:flex;gap:4px;margin-left:auto}
nav button{background:transparent;border:1px solid transparent;color:#7d8590;
  padding:7px 15px;border-radius:7px;cursor:pointer;font-size:14px;font-weight:500}
nav button:hover{color:#e6edf3;background:#161b22}
nav button.on{background:#1f6feb;color:#fff}
main{flex:1;min-height:0;position:relative}
.view{position:absolute;inset:0;display:none}
.view.on{display:flex}
#map{flex:1;background:#0d1117}
.side{width:390px;flex-shrink:0;border-left:1px solid #21262d;overflow-y:auto;
      background:#0d1117;padding:18px}
.pane{flex:1;overflow-y:auto;padding:24px 30px}
h2{margin:0 0 4px;font-size:17px;font-weight:620}
.sub{color:#7d8590;font-size:13px;margin-bottom:18px;line-height:1.5}
.kv{display:flex;justify-content:space-between;padding:7px 0;
    border-bottom:1px solid #1c2128;font-size:13.5px}
.kv span:first-child{color:#8b949e}
.kv b{font-weight:600}
.cap{color:#6e7681;font-size:11.5px;margin-bottom:14px}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;
      font-weight:600;letter-spacing:.03em}
.btn{width:100%;padding:11px;border:0;border-radius:8px;cursor:pointer;
     font-size:14px;font-weight:600;margin-top:14px}
.btn-red{background:#da3633;color:#fff}
.btn-red:hover{background:#f85149}
.btn-blue{background:#1f6feb;color:#fff}
.btn-blue:hover{background:#388bfd}
.btn:disabled{background:#21262d;color:#6e7681;cursor:not-allowed}
.warn{background:#2d1a0e;border:1px solid #5a3a1a;color:#e3b341;padding:9px 12px;
      border-radius:7px;font-size:12px;line-height:1.5;margin:14px 0}
.legend{position:absolute;bottom:18px;left:18px;background:#161b22ee;
  border:1px solid #21262d;padding:11px 13px;border-radius:9px;z-index:500;font-size:12px}
.bar{height:9px;width:190px;border-radius:5px;margin:7px 0 5px;
     background:linear-gradient(90deg,#2ea043,#89b32d,#d29922,#e8590c,#f85149)}
.wkey{display:flex;align-items:center;gap:7px;margin-top:5px;color:#8b949e;font-size:11px}
.wkey i{display:inline-block;border-radius:2px;background:#8b949e;flex-shrink:0}
.lrow{display:flex;justify-content:space-between;color:#8b949e;font-size:11px}
.lkey{display:flex;align-items:center;gap:7px;margin-top:6px;color:#8b949e;font-size:11.5px}
.lkey i{width:15px;height:3px;border-radius:2px;flex-shrink:0;display:inline-block}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:#8b949e;font-weight:600;padding:9px 10px;
   border-bottom:1px solid #21262d;font-size:12px;text-transform:uppercase;
   letter-spacing:.04em}
td{padding:11px 10px;border-bottom:1px solid #1c2128;vertical-align:top}
tr.top td{background:#11261a}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pick{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));
      gap:8px;margin:14px 0 20px}
.pick label{display:flex;gap:9px;align-items:flex-start;padding:10px 12px;
  border:1px solid #21262d;border-radius:8px;cursor:pointer;background:#0f1520}
.pick label:hover{border-color:#30363d;background:#161b22}
.pick input:checked+div{color:#e6edf3}
.pick .nm{font-size:13px;font-weight:600}
.pick .mt{font-size:11.5px;color:#7d8590;margin-top:2px}
input[type=search]{width:100%;padding:10px 13px;border-radius:8px;
  border:1px solid #21262d;background:#0f1520;color:#e6edf3;font-size:14px}
.verdict{background:#0f2417;border:1px solid #1f6f3f;border-radius:10px;
         padding:16px 18px;margin-bottom:18px}
.verdict h3{margin:0 0 7px;font-size:15px;color:#3fb950}
.verdict p{margin:0;font-size:13.5px;line-height:1.6;color:#c9d1d9}
.muted{color:#6e7681}
.empty{color:#6e7681;text-align:center;padding:60px 20px;font-size:14px}
/* views that stack a map above a scrolling pane */
.vsplit{flex-direction:column}
.mapstrip{height:340px;flex-shrink:0;position:relative;border-bottom:1px solid #21262d}
.mapstrip>div[id]{position:absolute;inset:0;background:#0d1117}
/* horizontal bar chart — one series, so no legend; every bar is direct-labelled */
.chart{margin:6px 0 20px}
.crow{display:grid;grid-template-columns:190px 1fr auto;gap:12px;align-items:center;
      padding:3px 0;font-size:13px}
.crow .cnm{color:#8b949e;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ctrack{background:#161b22;border-radius:5px;height:19px;overflow:hidden}
.cfill{height:100%;border-radius:0 4px 4px 0;min-width:2px;
       transition:width .25s ease}
.cval{font-variant-numeric:tabular-nums;font-weight:600;min-width:56px;text-align:right}
.axnote{color:#6e7681;font-size:11.5px;margin:2px 0 0 202px}
.dwrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:0 26px}
</style></head><body>
<header>
  <div class="brand">Canal Intelligence <span>Cauvery delta · canal desilting priority</span></div>
  <nav>
    <button class="on" data-v="map">Canal heatmap</button>
    <button data-v="cmp">Canal priority</button>
    <button data-v="sim">Simulation</button>
  </nav>
</header>
<main>
  <div class="view on" id="v-map">
    <div id="map"></div>
    <div class="legend">
      <div style="font-weight:600;margin-bottom:2px">Network importance</div>
      <div class="bar"></div>
      <div class="lrow"><span>minor</span><span>critical</span></div>
      <div class="wkey"><i style="width:15px;height:2px"></i>
        <i style="width:15px;height:8px"></i>thickness shows the same</div>
      <div style="color:#6e7681;font-size:11px;margin-top:6px;max-width:190px;
                  line-height:1.45">Betweenness centrality, by percentile rank
        within this network — how much of the delta depends on this one reach.
        Not canal condition.</div>
    </div>
    <div class="side" id="side"><div class="empty">Click any canal on the map.</div></div>
  </div>
  <div class="view vsplit" id="v-cmp">
    <div class="mapstrip">
      <div id="pmap"></div>
      <div class="legend">
        <div style="font-weight:600;margin-bottom:4px">Selection</div>
        <div class="lkey"><i style="background:#2ea043"></i>clean first</div>
        <div class="lkey"><i style="background:#1f6feb"></i>also selected</div>
        <div class="lkey"><i style="background:#30363d"></i>not selected</div>
      </div>
    </div>
    <div class="pane" id="cmppane"></div>
  </div>
  <div class="view vsplit" id="v-sim">
    <div class="mapstrip">
      <div id="smap"></div>
      <div class="legend">
        <div style="font-weight:600;margin-bottom:4px">Blockage effect</div>
        <div class="lkey"><i style="background:#da3633"></i>blocked reach</div>
        <div class="lkey"><i style="background:#d29922"></i>loses supply</div>
        <div class="lkey"><i style="background:#2ea043"></i>still supplied</div>
      </div>
    </div>
    <div class="pane" id="simpane"></div>
  </div>
</main>
<script>
const D = __DATA__;
const byId = {}; D.segments.forEach(s => byId[s.id] = s);
const inr = n => n==null ? '—' : '₹' + Number(n).toLocaleString('en-IN',{maximumFractionDigits:0});
const cr  = n => n==null ? '—' : '₹' + (n/1e7).toFixed(2) + ' cr';
const ha  = n => n==null ? '—' : Number(n).toLocaleString('en-IN') + ' ha';
/* KML canal names carry & and quotes, and go into an optgroup label attr. */
const esc = s => String(s==null ? '' : s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const mix = (a,b,f) => {
  const p = h => [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];
  const A=p(a), B=p(b);
  return `rgb(${Math.round(A[0]+(B[0]-A[0])*f)},${Math.round(A[1]+(B[1]-A[1])*f)},${Math.round(A[2]+(B[2]-A[2])*f)})`;
};
/* Criticality ramp: green (minor) -> amber -> red (critical). Red/green is
   unreadable for deuteranopia and protanopia, so LINE WEIGHT carries the same
   variable — thick AND red, thin AND green. Never let the ramp be the only
   channel. */
const HEAT = ['#2ea043','#89b32d','#d29922','#e8590c','#f85149'];
function heat(t){
  t = Math.max(0,Math.min(1,t));
  const x = t*(HEAT.length-1), i = Math.min(HEAT.length-2, Math.floor(x));
  return mix(HEAT[i], HEAT[i+1], x-i);
}
const heatWt = t => 2 + 6*Math.max(0,Math.min(1,t));
/* Betweenness is right-skewed (median 0.05, max 1.0), so a linear ramp paints
   three quarters of the network the same dim colour. Rank instead. */
const CENTS = D.segments.map(s=>s.cent).sort((a,b)=>a-b);
function centPct(v){
  let lo=0, hi=CENTS.length;
  while(lo<hi){ const m=(lo+hi)>>1; if(CENTS[m]<v) lo=m+1; else hi=m; }
  return CENTS.length>1 ? lo/(CENTS.length-1) : 0;
}
const wt = s => s.type==='Main Canal' ? 4.5 : 2.8;

function buildMap(elId, styler){
  const m = L.map(elId,{zoomControl:true}).fitBounds(D.bounds);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    {attribution:'© OpenStreetMap © CARTO',maxZoom:19}).addTo(m);
  const ls = {};
  D.segments.forEach(s => {
    const ln = L.polyline(s.coords.map(c=>[c[1],c[0]]), styler(s));
    ln.addTo(m); ls[s.id] = ln;
  });
  return {map:m, lines:ls};
}

/* Where a canal's water actually goes: everything downstream, rolled up by
   district. No village-level ayacut register exists here, so that is as fine
   as it honestly gets. */
function areasCovered(id){
  const sim = D.sims[id] || {};
  const byD = sim.area_by_district || {};
  const names = [];
  (sim.lost_seg_ids || []).forEach(i => {
    const s = byId[i];
    // list the canal, not every reach, and only where the KML names it
    if(s && i !== id && !s.derived && !names.includes(s.canal)) names.push(s.canal);
  });
  return {byD, names};
}

/* ------------------------------------------------------- 1. canal heatmap */
const H = buildMap('map', s => ({color:heat(centPct(s.cent)),
                                 weight:heatWt(centPct(s.cent)), opacity:.95}));
const map = H.map, lines = H.lines;
D.segments.forEach(s => {
  const ln = lines[s.id], p = centPct(s.cent);
  const band = p>=.8 ? 'critical' : p>=.6 ? 'high' : p>=.35 ? 'moderate' : 'minor';
  ln.on('click', e => { L.DomEvent.stopPropagation(e); select(s.id); });
  ln.on('mouseover', () => ln.setStyle({weight:heatWt(p)+3,opacity:1}));
  ln.on('mouseout',  () => ln.setStyle({weight:heatWt(p),opacity:.95}));
  ln.bindTooltip(`${s.name} · ${s.type}<br><b>${band}</b> — carries more of the
    network than ${Math.round(p*100)}% of canals`,{sticky:true});
});

let selected = null;
function select(id){
  selected = id; const s = byId[id];
  Object.values(lines).forEach(l=>l.setStyle({dashArray:null}));
  lines[id].setStyle({dashArray:'6 5'});
  const dist = D.districts[s.district] || {};
  const ac = areasCovered(id);
  const dRows = Object.entries(ac.byD);
  const p = centPct(s.cent), c = heat(p);
  const band = p>=.8 ? 'CRITICAL' : p>=.6 ? 'HIGH' : p>=.35 ? 'MODERATE' : 'MINOR';
  document.getElementById('side').innerHTML = `
    <h2>${s.name}</h2>
    <div class="sub">${s.type} · ${s.project} · ${s.district}<br>
      <span class="pill" style="background:${c}22;color:${c}">${band} — top
      ${Math.max(1,Math.round((1-p)*100))}% by network importance</span></div>
    <div class="kv"><span>Length</span><b>${s.km} km</b></div>
    <div class="kv"><span>Command area</span><b>${ha(s.area)}</b></div>
    <div class="kv"><span>Conveyance health</span><b>${s.health.toFixed(2)}</b></div>
    <div class="kv"><span>District revenue</span><b>${inr(dist.rev_ha)}/ha/yr</b></div>
    <h2 style="font-size:14px;margin:20px 0 6px">Command area covered</h2>
    ${dRows.length ? dRows.map(([d,v])=>
        `<div class="kv"><span>${d.charAt(0)+d.slice(1).toLowerCase()}</span><b>${ha(Math.round(v))}</b></div>`).join('')
      : `<div class="kv"><span>${s.district}</span><b>${ha(s.area)}</b></div>`}
    ${ac.names.length ? `<div class="cap" style="margin-top:10px">Reaches served —
      ${ac.names.slice(0,12).join(', ')}${ac.names.length>12?` and ${ac.names.length-12} more`:''}</div>` : ''}
    <button class="btn btn-red" onclick="goSim('${id}')">Simulate a blockage here</button>
    <button class="btn btn-blue" onclick="addCompare('${id}')">Add to priority list</button>`;
}

/* ---------------------------------------------------------------- nav */
/* Leaflet can't measure a display:none container — nudge on first show. */
function wake(m, fitFn){ setTimeout(()=>{ m.invalidateSize(); if(fitFn) fitFn(); },60); }
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('on'));
  document.getElementById('v-'+b.dataset.v).classList.add('on');
  const v = b.dataset.v;
  if(v==='map') setTimeout(()=>map.invalidateSize(),50);
  if(v==='sim'){ renderSim(); wake(SM.map, fitSim); }
  if(v==='cmp'){ renderCompare(); wake(PM.map, fitPriority); }
});
function show(v){ document.querySelector(`nav button[data-v=${v}]`).click(); }
function goSim(id){ simTargets = new Set([id]); show('sim'); }

/* ------------------------------------------------------- 3. simulation */
const SM = buildMap('smap', s => ({color:'#1a7f37', weight:wt(s)*0.7, opacity:.42}));
/* Starts empty — a canal pre-selected on load reads as a finding. */
let simTargets = new Set();

/* Combined impact of blocking several reaches at once. Unioning the individual
   cut-off sets is exact, not an approximation: the graph is a near-tree (359
   nodes, 361 edges) and no sampled pair stranded anything the two didn't
   separately. Local areas sum back to the simulated total to the hectare. */
function combinedSim(){
  const lost = new Set();
  simTargets.forEach(id => ((D.sims[id]||{}).lost_seg_ids || []).forEach(x => lost.add(x)));
  simTargets.forEach(id => lost.add(id));
  const cut = [...lost].filter(x => byId[x] && !simTargets.has(x));
  const byD = {};
  let area = 0;
  [...lost].forEach(x => {
    const s = byId[x]; if(!s) return;
    byD[s.district] = (byD[s.district] || 0) + s.local;
    area += s.local;
  });
  const cost = [...simTargets].reduce((a,id)=>a+(byId[id]?byId[id].cost:0), 0);
  // revenue scales with each district's own crop value, so rebuild per district
  let rev = 0;
  Object.entries(byD).forEach(([d,v]) => {
    rev += v * ((D.districts[d]||{}).rev_ha || 0);
  });
  return {lost, cut, byD: Object.fromEntries(
    Object.entries(byD).sort((a,b)=>b[1]-a[1])), area, cost, rev};
}

/* Paint the consequence: blocked reaches, then every sub-branch that loses its
   path back to the Grand Anicut. */
function paintSim(){
  const {lost} = combinedSim();
  D.segments.forEach(s => {
    const ln = SM.lines[s.id];
    let tip;
    if(simTargets.has(s.id)){
      ln.setStyle({color:'#da3633',weight:Math.max(6,wt(s)+2),opacity:1,dashArray:null});
      tip = 'blocked here';
    } else if(lost.has(s.id)){
      ln.setStyle({color:'#d29922',weight:wt(s),opacity:.95,dashArray:'5 4'});
      tip = 'loses supply';
    } else {
      ln.setStyle({color:'#1a7f37',weight:wt(s)*0.7,opacity:.42,dashArray:null});
      tip = 'still supplied';
    }
    ln.unbindTooltip(); ln.bindTooltip(`${s.name} — ${tip}`,{sticky:true});
  });
}
function fitSim(){
  const {lost} = combinedSim();
  const pts = [];
  lost.forEach(i => { const s = byId[i];
    if(s) s.coords.forEach(c => pts.push([c[1],c[0]])); });
  if(pts.length) SM.map.fitBounds(L.latLngBounds(pts).pad(0.15));
  else SM.map.fitBounds(D.bounds);
}
function toggleSim(id){
  simTargets.has(id) ? simTargets.delete(id) : simTargets.add(id);
  renderSim(); fitSim();
}
D.segments.forEach(s => SM.lines[s.id].on('click', e => {
  L.DomEvent.stopPropagation(e); toggleSim(s.id);
}));

function renderSim(){
  const C = combinedSim();
  /* Grouped by canal — a flat list repeated "Pullambadi Channel" 18 times with
     nothing to tell the rows apart. Reaches sit under their canal in downstream
     order, and stay individually pickable: blocking the head cuts everything
     below it, blocking the tail cuts almost nothing. */
  const groups = {};
  D.segments.slice().filter(s => !simTargets.has(s.id))
    .forEach(s => (groups[s.canal || s.name] = groups[s.canal || s.name] || []).push(s));
  const opts = Object.keys(groups).sort((a,b)=>
      Math.min(...groups[a].map(s=>s.rank)) - Math.min(...groups[b].map(s=>s.rank)))
    .map(canal => {
      const rows = groups[canal].sort((a,b)=>a.kmFrom-b.kmFrom)
        .map(s=>`<option value="${s.id}">${esc(s.reach || s.name)}</option>`).join('');
      // A single-reach canal needs no heading, and should carry its own name
      // rather than the plural group label ("Distributaries off X").
      const one = groups[canal][0];
      return groups[canal].length === 1
        ? `<option value="${one.id}">${esc(one.name)} (${one.type}, ${one.km} km)</option>`
        : `<optgroup label="${esc(canal)} — ${groups[canal].length} reaches">${rows}</optgroup>`;
    }).join('');
  const byD = Object.entries(C.byD);
  const totD = C.area || 1;
  const cutNames = [...new Set(C.cut.filter(i=>byId[i] && !byId[i].derived)
                                    .map(i=>byId[i].canal))];
  // crop split: inferred paddy/other ratio applied to the combined area, not
  // summed across overlapping simulations
  const cropShare = {};
  simTargets.forEach(id => {
    Object.entries(D.sims[id]||{}).filter(([k])=>k.startsWith('ha_'))
      .forEach(([k,v]) => { cropShare[k.slice(3)] = (cropShare[k.slice(3)]||0) + v; });
  });
  const cropTot = Object.values(cropShare).reduce((a,b)=>a+b,0) || 1;
  const crops = Object.entries(cropShare).filter(([,v])=>v>0)
    .map(([c,v]) => [c, v/cropTot*C.area]);
  const none = simTargets.size === 0;
  const multi = simTargets.size > 1;
  const head = none ? null : byId[[...simTargets][0]];
  const dist = head ? (D.districts[head.district] || {}) : {};
  document.getElementById('simpane').innerHTML = `
    <h2>What breaks if ${none?'a canal blocks':multi?'these canals block':'this canal blocks'}?</h2>
    <div class="sub">Removes the selected reaches from the network, then recomputes
      which command area can still be reached from the Grand Anicut. The map shows
      blocked reaches in red and every sub-branch that loses supply in amber —
      click any canal on the map to add or remove it.</div>
    <div style="display:flex;flex-wrap:wrap;gap:7px;margin:14px 0 10px">
      ${[...simTargets].map(id=>`<span class="pill" style="background:#3d1517;
        color:#f85149;border:1px solid #5c2427;padding:5px 11px;font-size:12.5px">
        ${byId[id].name} · ${byId[id].km} km
        <b style="cursor:pointer;margin-left:7px" onclick="toggleSim('${id}')">×</b>
        </span>`).join('')}
    </div>
    <select id="simsel" style="width:100%;padding:11px;border-radius:8px;
      border:1px solid #21262d;background:#0f1520;color:#e6edf3;font-size:14px">
      <option value="">${none?'Select a canal to block…':'+ block another canal…'}</option>${opts}</select>
    ${none ? `<div class="cap" style="margin-top:26px;padding:26px;text-align:center;
      border:1px dashed #21262d;border-radius:10px;font-size:13px;line-height:1.7">
      No canal is blocked. Pick a reach from the list above, or click any canal on the
      map, to see which command area loses supply.</div>` : `
    <div class="dwrap" style="margin-top:24px">
      <div>
        <h2 style="font-size:15px">Combined impact</h2>
        <div class="kv"><span>Canals blocked</span><b>${simTargets.size}</b></div>
        <div class="kv"><span>Canal reaches cut off</span><b>${C.cut.length}</b></div>
        <div class="kv"><span>Command area lost</span><b>${ha(Math.round(C.area))}</b></div>
        ${crops.map(([c,v])=>`<div class="kv"><span class="muted">&nbsp;&nbsp;${c}</span><b>${ha(Math.round(v))}</b></div>`).join('')}
        <div class="kv"><span>Gross revenue at risk</span><b style="color:#f85149">${cr(C.rev)}</b></div>
      </div>
      <div>
        <h2 style="font-size:15px">Which areas lose supply</h2>
        <div class="sub" style="margin-bottom:10px">Command area cut off, by district.</div>
        ${byD.length ? byD.map(([d,v])=>`<div class="kv">
            <span>${d.charAt(0)+d.slice(1).toLowerCase()}</span>
            <b>${ha(Math.round(v))} <span class="muted" style="font-weight:400">
            ${(v/totD*100).toFixed(0)}%</span></b></div>`).join('')
          : '<div class="muted" style="font-size:13px">Nothing downstream is cut off.</div>'}
        ${cutNames.length ? `<div class="cap" style="margin-top:12px">Named reaches that go dry —
          ${cutNames.slice(0,14).join(', ')}${cutNames.length>14?` and ${cutNames.length-14} more`:''}</div>`:''}
      </div>
      <div>
        <h2 style="font-size:15px">Why that area is worth it</h2>
        <div class="sub" style="margin-bottom:10px">Reported cropping pattern,
          Tamil Nadu crop statistics.</div>
        ${(dist.mix||[]).map(([c,sh])=>`<div class="kv"><span>${c}</span><b>${(sh*100).toFixed(1)}%</b></div>`).join('')}
        <div class="kv"><span>Revenue per hectare</span><b>${inr(dist.rev_ha)}/yr</b></div>
        <div class="kv"><span>Crop water need</span><b>${dist.water_mm} mm/yr</b></div>
        <div class="kv"><span>Rain covers</span><b>${Math.round((1-dist.irr_dep)*100)}% of demand</b></div>
      </div>
    </div>
    <div class="warn"><b>Read this before quoting the rupee figure.</b>
      Revenue is GROSS at MSP for a single season and assumes the cut-off area loses
      its supply entirely. Paddy is priced at the sourced MSP; every other crop price
      is an ASSUMED value.${multi?` Overlapping
      command area is counted once, so the combined figure is not the sum of the
      canals taken separately.`:''}</div>`}`;
  document.getElementById('simsel').onchange = e => {
    if(e.target.value) toggleSim(e.target.value); };
  paintSim();
}

/* ------------------------------------------------------- 2. canal priority */
/* One canal is many graph segments — 133 segments, 27 distinct names, 18 of
   them "Pullambadi Channel". You desilt a canal, not a graph edge, so group
   the reaches before ranking. */
function canalGroups(){
  const g = {};
  D.segments.forEach(s => {
    // Group on the canal, not s.name — that carries a reach number and would
    // split Pullambadi into 18. Derived names stay separate: a minor off it at
    // km 29 and another at km 41 are different jobs.
    const key = s.derived ? s.id : (s.canal || s.name);
    (g[key] = g[key] || []).push(s);
  });
  const out = Object.entries(g).map(([key, segs]) => {
    const rep = segs.slice().sort((a,b)=>b.area-a.area)[0];   // head reach
    const km = segs.reduce((a,s)=>a+s.km, 0);
    return {
      key, segs, rep, type: rep.type, district: rep.district,
      name: rep.derived ? rep.name : (rep.canal || rep.name),
      km: Math.round(km*10)/10,
      // command area is cumulative downstream — the head reach already
      // commands everything below it, so summing reaches double-counts
      area: Math.max(...segs.map(s=>s.area)),
      // you clean the whole canal, so cost does sum
      cost: segs.reduce((a,s)=>a+s.cost, 0),
      // length-weighted: a long bad reach should outweigh a short clean one
      health: km ? segs.reduce((a,s)=>a+s.health*s.km, 0)/km : rep.health,
      // worst single-point failure anywhere on the canal
      rev: Math.max(...segs.map(s=>(D.sims[s.id]||{}).revenue_at_risk_inr || 0)),
      cut: Math.max(...segs.map(s=>(D.sims[s.id]||{}).segments_cut_off || 0)),
    };
  });
  // Two unnamed reaches can share project, district AND length, putting the
  // same text in the picker twice. Suffix any collision that survives.
  const seen = {};
  out.forEach(g => {
    const n = seen[g.name] = (seen[g.name] || 0) + 1;
    if(n > 1) g.name = `${g.name} #${n}`;
  });
  return out;
}
const GROUPS = canalGroups();
const GBYKEY = {}; GROUPS.forEach(g => GBYKEY[g.key] = g);
const KEYOF  = {}; GROUPS.forEach(g => g.segs.forEach(s => KEYOF[s.id] = g.key));

/* Priority = revenue at risk + poor health + command area. Revenue and area are
   normalised within the current selection, so the score answers "of these,
   which first" rather than pretending to be absolute. */
const W_REV = 0.40, W_HEALTH = 0.35, W_AREA = 0.25;
function scoreRows(rows){
  const mxR = Math.max(...rows.map(r=>r.rev), 0);
  const mxA = Math.max(...rows.map(r=>r.area), 0);
  rows.forEach(r => {
    r.nRev   = mxR ? r.rev/mxR : 0;
    r.impair = 1 - r.health;
    r.nArea  = mxA ? r.area/mxA : 0;
    r.score  = (W_REV*r.nRev + W_HEALTH*r.impair + W_AREA*r.nArea) * 100;
  });
  return rows.sort((a,b) => b.score - a.score);
}

const PM = buildMap('pmap', s => ({color:'#30363d', weight:wt(s)*0.7, opacity:.5}));
let picked = new Set();
function addCompare(id){ picked.add(KEYOF[id]); show('cmp'); }
D.segments.forEach(s => PM.lines[s.id].on('click', e => {
  L.DomEvent.stopPropagation(e);
  const k = KEYOF[s.id];
  picked.has(k) ? picked.delete(k) : picked.add(k);
  renderCompare();
}));

function paintPriority(firstKey){
  const pickedSegs = new Set();
  picked.forEach(k => (GBYKEY[k]||{segs:[]}).segs.forEach(s => pickedSegs.add(s.id)));
  const firstSegs = new Set(((GBYKEY[firstKey]||{segs:[]}).segs).map(s=>s.id));
  D.segments.forEach(s => {
    const ln = PM.lines[s.id];
    let tip = s.name;
    if(firstSegs.has(s.id)){
      ln.setStyle({color:'#2ea043',weight:Math.max(6,wt(s)+2),opacity:1});
      tip += ' — clean first';
    } else if(pickedSegs.has(s.id)){
      ln.setStyle({color:'#1f6feb',weight:wt(s)+1,opacity:.95});
      tip += ' — selected';
    } else {
      ln.setStyle({color:'#30363d',weight:wt(s)*0.7,opacity:.5});
    }
    ln.unbindTooltip(); ln.bindTooltip(tip,{sticky:true});
  });
}
function fitPriority(){
  const pts = [];
  picked.forEach(k => (GBYKEY[k]||{segs:[]}).segs.forEach(
    s => s.coords.forEach(c => pts.push([c[1],c[0]]))));
  if(pts.length) PM.map.fitBounds(L.latLngBounds(pts).pad(0.2));
  else PM.map.fitBounds(D.bounds);
}

function renderCompare(){
  const q = (document.getElementById('cmpq')?.value || '').toLowerCase();
  const list = GROUPS.slice()
    .sort((a,b) => b.area - a.area)
    .filter(g => !q || g.name.toLowerCase().includes(q) || g.district.toLowerCase().includes(q));
  document.getElementById('cmppane').innerHTML = `
    <h2>Which of these should we clean first?</h2>
    <div class="sub">Pick the canals actually on the table this season — on the map
      above, or in the list below. Each entry is a whole canal, with all of its
      reaches rolled together. The ranking is over your selection only, so it
      answers the real question: not "is this canal bad" but "of these, which
      first".</div>
    <div id="verdict"></div>
    <h2 style="font-size:15px;margin-top:26px">Canals on the table
      <span class="muted" style="font-weight:400;font-size:13px">
      — ${picked.size} of ${GROUPS.length} selected</span></h2>
    <input type="search" id="cmpq" placeholder="Filter by canal or district…"
           value="${q}" style="margin:10px 0">
    <div class="pick">${list.slice(0,40).map(g=>`
      <label><input type="checkbox" value="${g.key}" ${picked.has(g.key)?'checked':''}>
        <div><div class="nm">${g.name}</div>
        <div class="mt">${g.type} · ${g.district} · ${ha(g.area)} · ${g.km} km
        ${g.segs.length>1?` · ${g.segs.length} reaches`:''}</div></div>
      </label>`).join('')}</div>
    ${list.length>40?`<div class="muted" style="font-size:12.5px">
      Showing 40 of ${list.length} — use the filter to narrow down.</div>`:''}`;
  const box = document.getElementById('cmpq');
  box.oninput = () => renderCompare();
  box.onkeydown = e => { if(e.key==='Enter') e.preventDefault(); };
  document.querySelectorAll('.pick input').forEach(i => i.onchange = () => {
    i.checked ? picked.add(i.value) : picked.delete(i.value); verdict(); });
  verdict();
}

/* One series, so no legend — the heading names the measure. Direct-labelling
   every bar is also what keeps the two-colour fill legible for tritan viewers. */
function chart(rows){
  const max = Math.max(...rows.map(r=>r.score), 1);
  return `
    <h2 style="font-size:15px;margin-top:24px">Priority score</h2>
    <div class="sub">Revenue at risk (${W_REV*100}%), poor conveyance health
      (${W_HEALTH*100}%) and command area (${W_AREA*100}%), combined. Longer is
      more urgent.</div>
    <div class="chart">${rows.map((r,i)=>`
      <div class="crow">
        <div class="cnm" title="${r.name}">${i+1}. ${r.name}</div>
        <div class="ctrack"><div class="cfill" style="width:${Math.max(1.5,r.score/max*100)}%;
          background:${i===0?'#2ea043':'#1f6feb'}"></div></div>
        <div class="cval" style="color:${i===0?'#2ea043':'#e6edf3'}">${r.score.toFixed(0)}</div>
      </div>`).join('')}</div>
    <div class="axnote">0 — ${max.toFixed(0)} priority score</div>`;
}

function verdict(){
  const el = document.getElementById('verdict'); if(!el) return;
  const sel = [...picked].map(k=>GBYKEY[k]).filter(Boolean);
  if(sel.length < 2){
    el.innerHTML = `<div class="empty">Select at least two canals to compare.</div>`;
    paintPriority(null); return; }
  const rows = scoreRows(sel.map(g => Object.assign({}, g)));
  const w = rows[0], runner = rows[1];
  const gap = runner.score ? (w.score - runner.score) : 0;
  paintPriority(w.key);
  el.innerHTML = `
    <div class="verdict">
      <h3>Do ${w.name} first</h3>
      <p>It scores <b>${w.score.toFixed(0)}</b> against ${runner.name}'s
      <b>${runner.score.toFixed(0)}</b>${gap>0?` — ${gap.toFixed(0)} points clear`:''}.
      It puts <b>${cr(w.rev)}</b> of gross revenue at risk, runs a conveyance
      health of <b>${w.health.toFixed(2)}</b>, and commands
      <b>${ha(w.area)}</b>${w.segs.length>1?` across ${w.segs.length} reaches`:''}.</p>
    </div>
    ${chart(rows)}
    <table><thead><tr>
      <th>Order</th><th>Canal</th><th class="num">Revenue at risk</th>
      <th class="num">Health</th><th class="num">Command area</th>
      <th class="num">Score</th></tr></thead><tbody>
      ${rows.map((r,i)=>`<tr class="${i===0?'top':''}">
        <td><b>${i+1}</b></td>
        <td><b>${r.name}</b><div class="mt muted">${r.type} · ${r.district} ·
          ${r.km} km${r.segs.length>1?` · ${r.segs.length} reaches`:''}</div></td>
        <td class="num">${cr(r.rev)}</td>
        <td class="num">${r.health.toFixed(2)}</td>
        <td class="num">${ha(r.area)}</td>
        <td class="num"><b>${r.score.toFixed(0)}</b></td></tr>`).join('')}
    </tbody></table>
    <div class="warn">Score = ${W_REV}×revenue at risk + ${W_HEALTH}×(1 − health)
      + ${W_AREA}×command area, with revenue and area normalised against the
      largest in your selection — so scores move when the selection changes, by
      design. Health is
      ${D.method.includes('PROVISIONAL')?'a <b>PROVISIONAL heuristic</b>, not a trained model — label points in step 3 and retrain before using this to award work.':'from the trained condition model.'}</div>`;
}
select(D.segments.slice().sort((a,b)=>a.rank-b.rank)[0].id);
</script></body></html>"""


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "chips":
        return fetch_segment_chips()

    payload = build_payload()
    out = config.PROJECT_ROOT / "ayacut_app.html"

    def jsonable(o):
        # numpy scalars leak in from geopandas/pandas columns
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")

    html = HTML.replace("__DATA__",
                        json.dumps(payload, separators=(",", ":"), default=jsonable))
    out.write_text(html, encoding="utf-8")
    print(f"\nwrote {out} ({len(html)/1e6:.1f} MB)")
    print("open it in a browser — no server needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
