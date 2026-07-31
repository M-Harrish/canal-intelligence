"""Step 8: build the AYACUT web app -> ayacut_app.html

One self-contained file, no server needed. Three sections:

  PRIORITY MAP  every segment coloured by desilting priority, click for detail
  SIMULATION    block a canal, see what is cut off and what it costs
  COMPARE       pick several canals, get a ranked verdict on which to do first

The compare section is the one a department actually needs: the question is
never "is this canal bad" in isolation, it is "we can afford two of these five
this year — which two". That comparison is made explicit here, with the
reasoning shown rather than just a score.

Run:  python step08_build_app.py            # build the app
      python step08_build_app.py chips      # refresh segment thumbnails first
"""

import base64
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
    """Run the failure simulation for every segment once, at build time.

    Doing all of them here is what makes the app work offline: clicking a canal
    reads a precomputed result instead of needing a Python backend.
    """
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
    """One dry-season true-colour thumbnail per segment, embedded in the app.

    Run once (~4 min). Chips only need geometry, so this can run before the
    ranking exists — useful for doing the slow download in parallel with the
    rest of the pipeline.
    """
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


def load_chips():
    chips = {}
    if not SEG_CHIPS_DIR.exists():
        return chips
    for p in SEG_CHIPS_DIR.glob("*.png"):
        chips[p.stem] = base64.b64encode(p.read_bytes()).decode()
    return chips


def build_payload():
    segs = load_segments()
    canals = segs[(segs["is_river"] == 0) & (segs["can_type"] != "Connector")].copy()
    print(f"{len(canals)} canal segments")

    sims = precompute_simulations(canals)
    chips = load_chips()
    print(f"{len(chips)} segment thumbnails")

    feats = []
    for r in canals.itertuples():
        coords = [[round(x, 5), round(y, 5)] for x, y in r.geometry.coords]
        feats.append({
            "id": r.seg_id,
            "name": (r.can_name or "").strip() or "(unnamed reach)",
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
    return {
        "segments": feats,
        "sims": sims,
        "chips": chips,
        "districts": districts,
        "method": method,
        "desilt_rate": assumptions.DESILT_COST_PER_KM.value,
        "bounds": [[float(canals.total_bounds[1]), float(canals.total_bounds[0])],
                   [float(canals.total_bounds[3]), float(canals.total_bounds[2])]],
    }


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<title>AYACUT — canal desilting priority</title>
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
.chip{width:100%;border-radius:8px;border:1px solid #21262d;display:block;margin:12px 0 6px}
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
     background:linear-gradient(90deg,#2da44e,#d29922,#da3633)}
.lrow{display:flex;justify-content:space-between;color:#8b949e;font-size:11px}
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
</style></head><body>
<header>
  <div class="brand">AYACUT <span>Cauvery delta · canal desilting priority</span></div>
  <nav>
    <button class="on" data-v="map">Priority map</button>
    <button data-v="sim">Simulation</button>
    <button data-v="cmp">Compare canals</button>
  </nav>
</header>
<main>
  <div class="view on" id="v-map">
    <div id="map"></div>
    <div class="legend">
      <div style="font-weight:600;margin-bottom:2px">Desilting priority</div>
      <div class="bar"></div>
      <div class="lrow"><span>low</span><span>urgent</span></div>
    </div>
    <div class="side" id="side"><div class="empty">Click any canal on the map.</div></div>
  </div>
  <div class="view" id="v-sim"><div class="pane" id="simpane"></div></div>
  <div class="view" id="v-cmp"><div class="pane" id="cmppane"></div></div>
</main>
<script>
const D = __DATA__;
const byId = {}; D.segments.forEach(s => byId[s.id] = s);
const inr = n => n==null ? '—' : '₹' + Number(n).toLocaleString('en-IN',{maximumFractionDigits:0});
const cr  = n => n==null ? '—' : '₹' + (n/1e7).toFixed(2) + ' cr';
const ha  = n => n==null ? '—' : Number(n).toLocaleString('en-IN') + ' ha';
function colour(p){ const t=Math.max(0,Math.min(1,p));
  return t<0.5 ? `rgb(${Math.round(45+180*t*2)},${Math.round(164-30*t*2)},78)`
               : `rgb(${Math.round(225-7*(t-.5)*2)},${Math.round(134-80*(t-.5)*2)},${Math.round(25+26*(t-.5)*2)})`; }

/* ---------------------------------------------------------------- map */
const map = L.map('map',{zoomControl:true}).fitBounds(D.bounds);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  {attribution:'© OpenStreetMap © CARTO',maxZoom:19}).addTo(map);
const lines = {};
D.segments.forEach(s => {
  const ln = L.polyline(s.coords.map(c=>[c[1],c[0]]),
    {color:colour(s.pnorm), weight:s.type==='Main Canal'?4.5:2.8, opacity:.92});
  ln.on('click', e => { L.DomEvent.stopPropagation(e); select(s.id); });
  ln.on('mouseover', () => ln.setStyle({weight:7,opacity:1}));
  ln.on('mouseout',  () => ln.setStyle({weight:s.type==='Main Canal'?4.5:2.8,opacity:.92}));
  ln.bindTooltip(`${s.name} · rank ${s.rank}`,{sticky:true});
  ln.addTo(map); lines[s.id]=ln;
});

let selected = null;
function select(id){
  selected = id; const s = byId[id], sim = D.sims[id] || {};
  Object.values(lines).forEach(l=>l.setStyle({dashArray:null}));
  lines[id].setStyle({dashArray:'6 5'});
  const dist = D.districts[s.district] || {};
  const chip = D.chips[id];
  document.getElementById('side').innerHTML = `
    <h2>${s.name}</h2>
    <div class="sub">${s.type} · ${s.project} · ${s.district}<br>
      <span class="pill" style="background:${colour(s.pnorm)}22;color:${colour(s.pnorm)}">
      RANK ${s.rank} OF ${D.segments.length}</span></div>
    ${chip?`<img class="chip" src="data:image/png;base64,${chip}">
      <div class="cap">Sentinel-2 dry season, segment midpoint</div>`:''}
    <div class="kv"><span>Length</span><b>${s.km} km</b></div>
    <div class="kv"><span>Command area</span><b>${ha(s.area)}</b></div>
    <div class="kv"><span>Conveyance health</span><b>${s.health.toFixed(2)}</b></div>
    <div class="kv"><span>Network centrality</span><b>${s.cent.toFixed(2)}</b></div>
    <div class="kv"><span>Priority score</span><b>${Number(s.priority).toLocaleString('en-IN')}</b></div>
    <div class="kv"><span>Desilting cost</span><b>${inr(s.cost)}</b></div>
    <div class="kv"><span>District revenue</span><b>${inr(dist.rev_ha)}/ha/yr</b></div>
    <div class="kv"><span>Rainfall vs normal</span><b>${dist.rain_dev>0?'+':''}${dist.rain_dev}%</b></div>
    <button class="btn btn-red" onclick="goSim('${id}')">Simulate a blockage here</button>
    <button class="btn btn-blue" onclick="addCompare('${id}')">Add to comparison</button>`;
}

/* ---------------------------------------------------------------- nav */
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('on'));
  document.getElementById('v-'+b.dataset.v).classList.add('on');
  if(b.dataset.v==='map') setTimeout(()=>map.invalidateSize(),50);
  if(b.dataset.v==='sim') renderSim();
  if(b.dataset.v==='cmp') renderCompare();
});
function show(v){ document.querySelector(`nav button[data-v=${v}]`).click(); }
function goSim(id){ simTarget = id; show('sim'); }

/* ---------------------------------------------------------------- simulation */
let simTarget = D.segments.slice().sort((a,b)=>a.rank-b.rank)[0].id;
function renderSim(){
  const opts = D.segments.slice().sort((a,b)=>a.rank-b.rank)
    .map(s=>`<option value="${s.id}" ${s.id===simTarget?'selected':''}>
      #${s.rank} · ${s.name} (${s.type}, ${s.km} km)</option>`).join('');
  const s = byId[simTarget], sim = D.sims[simTarget] || {};
  const dist = D.districts[s.district] || {};
  const crops = Object.entries(sim).filter(([k])=>k.startsWith('ha_'))
    .map(([k,v])=>[k.slice(3), v]).filter(([,v])=>v>0);
  document.getElementById('simpane').innerHTML = `
    <h2>What breaks if this canal blocks?</h2>
    <div class="sub">Removes the segment from the network, then recomputes which
      command area can still be reached from the Grand Anicut.</div>
    <select id="simsel" style="width:100%;padding:11px;border-radius:8px;
      border:1px solid #21262d;background:#0f1520;color:#e6edf3;font-size:14px">${opts}</select>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:26px;margin-top:24px">
      <div>
        <h2 style="font-size:15px">Impact</h2>
        <div class="kv"><span>Segments cut off</span><b>${sim.segments_cut_off ?? '—'}</b></div>
        <div class="kv"><span>Command area lost</span><b>${ha(sim.area_lost_ha)}</b></div>
        ${crops.map(([c,v])=>`<div class="kv"><span class="muted">&nbsp;&nbsp;${c}</span><b>${ha(Math.round(v))}</b></div>`).join('')}
        <div class="kv"><span>Gross revenue at risk</span><b style="color:#f85149">${cr(sim.revenue_at_risk_inr)}</b></div>
        <div class="kv"><span>Cost to desilt</span><b>${inr(sim.desilt_cost_inr)}</b></div>
        <div class="kv"><span>Benefit : cost</span><b style="color:#3fb950">${sim.benefit_cost_ratio?Number(sim.benefit_cost_ratio).toFixed(0)+'×':'—'}</b></div>
      </div>
      <div>
        <h2 style="font-size:15px">Why that area is worth it</h2>
        <div class="sub" style="margin-bottom:10px">${s.district} reported cropping pattern,
          Tamil Nadu crop statistics.</div>
        ${(dist.mix||[]).map(([c,sh])=>`<div class="kv"><span>${c}</span><b>${(sh*100).toFixed(1)}%</b></div>`).join('')}
        <div class="kv"><span>Revenue per hectare</span><b>${inr(dist.rev_ha)}/yr</b></div>
        <div class="kv"><span>Crop water need</span><b>${dist.water_mm} mm/yr</b></div>
        <div class="kv"><span>Rainfall vs normal</span><b>${dist.rain_dev>0?'+':''}${dist.rain_dev}%</b></div>
        <div class="kv"><span>Rain covers</span><b>${Math.round((1-dist.irr_dep)*100)}% of demand</b></div>
      </div>
    </div>
    <div class="warn"><b>Read this before quoting the rupee figure.</b>
      Revenue is GROSS at MSP for a single season and assumes the cut-off area loses
      its supply entirely. Paddy is priced at the sourced MSP; every other crop price
      is an ASSUMED value. Desilting cost is a programme average
      (${inr(D.desilt_rate)}/km), not a schedule of rates.</div>`;
  document.getElementById('simsel').onchange = e => { simTarget = e.target.value; renderSim(); };
}

/* ---------------------------------------------------------------- compare */
let picked = new Set();
function addCompare(id){ picked.add(id); show('cmp'); }
function renderCompare(){
  const q = (document.getElementById('cmpq')?.value || '').toLowerCase();
  const list = D.segments.slice().sort((a,b)=>a.rank-b.rank)
    .filter(s => !q || s.name.toLowerCase().includes(q) || s.district.toLowerCase().includes(q));
  document.getElementById('cmppane').innerHTML = `
    <h2>Which of these should we clean first?</h2>
    <div class="sub">Pick the canals actually on the table this season. The ranking
      below is over your selection only, so it answers the real question — not
      "is this canal bad" but "of these, which buys the most protected hectares
      per rupee".</div>
    <div id="verdict"></div>
    <h2 style="font-size:15px;margin-top:26px">Canals on the table
      <span class="muted" style="font-weight:400;font-size:13px">
      — ${picked.size} selected</span></h2>
    <input type="search" id="cmpq" placeholder="Filter by canal or district…"
           value="${q}" style="margin:10px 0">
    <div class="pick">${list.slice(0,40).map(s=>`
      <label><input type="checkbox" value="${s.id}" ${picked.has(s.id)?'checked':''}>
        <div><div class="nm">${s.name}</div>
        <div class="mt">${s.type} · ${s.district} · ${ha(s.area)} · rank ${s.rank}</div></div>
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
function verdict(){
  const el = document.getElementById('verdict'); if(!el) return;
  const sel = [...picked].map(id=>byId[id]).filter(Boolean);
  if(sel.length < 2){
    el.innerHTML = `<div class="empty">Select at least two canals to compare.</div>`; return; }
  const rows = sel.map(s => {
    const sim = D.sims[s.id] || {};
    const rev = sim.revenue_at_risk_inr || 0;
    // Hectares this spend protects = command area weighted by how impaired the
    // segment looks. A healthy canal protects little for the same money.
    const prot = s.area * (1 - s.health);
    return {s, sim, rev, prot, perRupee: s.cost ? prot / s.cost * 1e6 : 0};
  }).sort((a,b) => b.perRupee - a.perRupee);
  const w = rows[0], runner = rows[1];
  const factor = runner.perRupee ? (w.perRupee / runner.perRupee) : 0;
  el.innerHTML = `
    <div class="verdict">
      <h3>Do ${w.s.name} first</h3>
      <p>It protects <b>${ha(Math.round(w.prot))}</b> of at-risk command area for
      <b>${inr(w.s.cost)}</b> — ${factor>1.05?`about <b>${factor.toFixed(1)}×</b> more
      hectares per rupee than ${runner.s.name}`:`narrowly ahead of ${runner.s.name}`}.
      Its conveyance health is <b>${w.s.health.toFixed(2)}</b> and it carries
      <b>${ha(w.s.area)}</b> of command area${w.sim.segments_cut_off?`, with
      <b>${w.sim.segments_cut_off}</b> segments downstream of it`:''}.</p>
    </div>
    <table><thead><tr>
      <th>Order</th><th>Canal</th><th class="num">Command area</th>
      <th class="num">Health</th><th class="num">At-risk ha</th>
      <th class="num">Cost</th><th class="num">Ha protected / ₹10 lakh</th>
      <th class="num">Revenue at risk</th></tr></thead><tbody>
      ${rows.map((r,i)=>`<tr class="${i===0?'top':''}">
        <td><b>${i+1}</b></td>
        <td><b>${r.s.name}</b><div class="mt muted">${r.s.type} · ${r.s.district}</div></td>
        <td class="num">${ha(r.s.area)}</td>
        <td class="num">${r.s.health.toFixed(2)}</td>
        <td class="num">${ha(Math.round(r.prot))}</td>
        <td class="num">${inr(r.s.cost)}</td>
        <td class="num"><b>${r.perRupee.toFixed(0)}</b></td>
        <td class="num">${cr(r.rev)}</td></tr>`).join('')}
    </tbody></table>
    <div class="warn">Ranking is on <b>at-risk hectares per rupee</b>: command area ×
      (1 − conveyance health) ÷ desilting cost. Health is
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
