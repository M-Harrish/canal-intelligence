"""Step 3: hand-labelling helper -> data/labels.csv

Downloads a satellite chip for each of ~300 stratified sample points and serves
a local web page where you click a class for each one. Labels are appended to
data/labels.csv as you go, so you can stop and resume any time.

Classes:
  flowing     water visible in the channel, clear conveyance
  dry         no water, but the channel is open (normal in the dry season)
  choked      channel obscured by vegetation growing in it
  encroached  built structures / fields over the alignment
  unclear     can't tell (kept out of training)

Usage:
  python step03_label_points.py fetch    # download chips (run once, ~5 min)
  python step03_label_points.py label    # open the labelling UI
  python step03_label_points.py status   # how many labelled so far
"""

import json
import sys
import threading
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

import config

warnings.filterwarnings("ignore")

PORT = 8712


def choose_points():
    """Stratified sample across canal type and dry-season NDVI difference, so
    the training set spans healthy and suspect reaches rather than whatever
    the network happens to have most of."""
    df = pd.read_csv(config.FEATURES_CSV).drop_duplicates("point_id")
    df = df[df["dry_nobs"].fillna(0) >= 5]

    df["ndvi_bin"] = pd.qcut(df["dry_ndvi_diff"], 4, labels=False, duplicates="drop")
    groups = df.groupby(["can_type", "ndvi_bin"], dropna=False)
    per_group = max(1, config.N_LABEL_POINTS // max(1, groups.ngroups))
    picked = groups.apply(
        lambda g: g.sample(min(len(g), per_group), random_state=config.LABEL_SEED)
    ).reset_index(drop=True)

    if len(picked) < config.N_LABEL_POINTS:
        rest = df[~df["point_id"].isin(picked["point_id"])]
        extra = rest.sample(
            min(len(rest), config.N_LABEL_POINTS - len(picked)),
            random_state=config.LABEL_SEED,
        )
        picked = pd.concat([picked, extra], ignore_index=True)
    picked = picked.sample(frac=1, random_state=config.LABEL_SEED).head(config.N_LABEL_POINTS)
    print(f"selected {len(picked)} points across {picked['can_type'].nunique()} canal types")
    return picked.reset_index(drop=True)


ESRI_EXPORT = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
    "MapServer/export?bbox={minx},{miny},{maxx},{maxy}&bboxSR=4326"
    "&imageSR=3857&size={px},{px}&format=jpg&f=image"
)


def fetch_hr_chip(lat, lon, path, centerline=None):
    """Sub-metre Esri World Imagery, which is what you actually label from.

    Sentinel-2 is 10 m, so a 5-15 m canal is one or two pixels — you cannot
    judge whether a channel is overgrown from that. Labelling ground truth
    from the sharpest available imagery and training on the coarser
    operational features is standard remote-sensing practice, not a
    shortcut: the label describes reality, the features describe what the
    satellite the model runs on can see.

    The canal centreline is drawn on the chip. The source KML is a national
    dataset with roughly 10-15 m positional accuracy, and delta canals very
    often run right beside a road — without the line drawn on, a labeller
    cannot tell which of two parallel features they are being asked to
    judge. Treat the line as "the canal is about here", not as exact.
    """
    import io
    import math
    import urllib.request

    from PIL import Image, ImageDraw

    dlat = config.CHIP_BUFFER_M / 110_574.0
    dlon = config.CHIP_BUFFER_M / (111_320.0 * math.cos(math.radians(lat)))
    minx, miny = lon - dlon, lat - dlat
    maxx, maxy = lon + dlon, lat + dlat
    px = config.CHIP_HR_PIXELS
    url = ESRI_EXPORT.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy, px=px)

    raw = urllib.request.urlopen(url, timeout=60).read()
    if centerline is None:
        path.write_bytes(raw)
        return

    im = Image.open(io.BytesIO(raw)).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")

    def to_px(x, y):
        return ((x - minx) / (maxx - minx) * px,
                (1 - (y - miny) / (maxy - miny)) * px)

    pts = [to_px(x, y) for x, y in centerline.coords]
    d.line(pts, fill=(0, 240, 255, 110), width=9)   # soft halo
    d.line(pts, fill=(0, 240, 255, 235), width=2)   # crisp centre
    im.save(path, quality=92)


_QUEUE_CACHE = {}


def chip_is_good(path):
    """True if the file exists, decodes, and is not an all-black tile.

    A half-written JPEG (interrupted download) and a genuinely black tile both
    render as a black box in the browser, which the labeller would read as
    "dark imagery" rather than "broken file". Checking here lets the server
    silently re-fetch instead of showing a lie.
    """
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        from PIL import Image, ImageStat
        return ImageStat.Stat(Image.open(path).convert("L")).mean[0] >= 5
    except Exception:
        return False


def fetch_one_hr(point_id, path):
    """Fetch a single high-res chip, used by the labelling server on demand."""
    import geopandas as gpd

    if not _QUEUE_CACHE:
        q = pd.read_csv(config.DATA_DIR / "label_queue.csv").set_index("point_id")
        lines = (gpd.read_file(config.SEGMENTS_GPKG).to_crs(config.CRS_WGS84)
                 .set_index("seg_id")["geometry"])
        _QUEUE_CACHE["q"], _QUEUE_CACHE["lines"] = q, lines

    r = _QUEUE_CACHE["q"].loc[point_id]
    config.CHIPS_HR_DIR.mkdir(parents=True, exist_ok=True)
    fetch_hr_chip(r.lat, r.lon, path, _QUEUE_CACHE["lines"].get(r.seg_id))


def placeholder_svg(msg):
    """Shown instead of a black box when imagery cannot be fetched."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">'
        f'<rect width="400" height="400" fill="#1b2230"/>'
        f'<text x="200" y="195" fill="#8b95a3" font-family="system-ui" '
        f'font-size="17" text-anchor="middle">{msg}</text>'
        f'<text x="200" y="222" fill="#5f6875" font-family="system-ui" '
        f'font-size="13" text-anchor="middle">press 5 (unclear) and move on</text>'
        f'</svg>'
    ).encode()


def fetch_chips():
    """Download a true-colour dry-season chip per point. Dry season is when an
    open channel is most distinguishable from a vegetated one."""
    import ee
    import urllib.request

    ee.Initialize(project=config.EE_PROJECT)
    config.CHIPS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHIPS_HR_DIR.mkdir(parents=True, exist_ok=True)

    pts = choose_points()
    pts[["point_id", "seg_id", "lat", "lon", "can_type", "prj_name"]].to_csv(
        config.DATA_DIR / "label_queue.csv", index=False
    )

    import geopandas as gpd
    lines = (gpd.read_file(config.SEGMENTS_GPKG).to_crs(config.CRS_WGS84)
             .set_index("seg_id")["geometry"])

    minx, miny, maxx, maxy = config.AOI_BBOX
    aoi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    img = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(*config.DRY_WINDOW)
        .linkCollection(csp, ["cs_cdf"])
        .map(lambda im: im.updateMask(im.select("cs_cdf").gte(config.CSP_CLEAR_THRESHOLD)))
        .median()
        .select(["B4", "B3", "B2"])
    )

    for i, r in pts.iterrows():
        hr = config.CHIPS_HR_DIR / f"{r.point_id}.jpg"
        if not hr.exists():
            try:
                fetch_hr_chip(r.lat, r.lon, hr, lines.get(r.seg_id))
            except Exception as e:
                print(f"  {r.point_id}: high-res failed ({e})")

        path = config.CHIPS_DIR / f"{r.point_id}.png"
        if not path.exists():
            region = (ee.Geometry.Point([r.lon, r.lat])
                      .buffer(config.CHIP_BUFFER_M).bounds())
            url = img.getThumbURL({
                "region": region,
                "dimensions": config.CHIP_PIXELS,
                "format": "png",
                "min": 200, "max": 2500,
            })
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"  {r.point_id}: sentinel chip failed ({e})")

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(pts)} points")
    print(f"chips in {config.CHIPS_DIR} and {config.CHIPS_HR_DIR}")


def load_state():
    queue = pd.read_csv(config.DATA_DIR / "label_queue.csv")
    done = {}
    if config.LABELS_CSV.exists():
        d = pd.read_csv(config.LABELS_CSV)
        done = dict(zip(d["point_id"], d["label"]))
    return queue, done


GUIDE = [
    ("flowing", "Dark/blue water line visible along the channel"),
    ("dry", "Channel clearly there as a line, but empty — bare or sandy"),
    ("choked", "Green vegetation growing ON the channel line itself"),
    ("encroached", "Buildings, roads or fields sitting over the alignment"),
    ("unclear", "Genuinely cannot tell — dropped from training"),
]

PAGE = """
<!doctype html><meta charset=utf-8><title>AYACUT labelling</title>
<style>
 body{font-family:system-ui;margin:0;background:#0e1116;color:#eee;
      display:flex;flex-direction:column;align-items:center;padding:14px 0}
 .imgs{display:flex;gap:14px;align-items:flex-start}
 .pane{width:460px}
 .pane.s2{width:230px}
 .wrap{position:relative;line-height:0}
 .pane img{width:100%;aspect-ratio:1;border:2px solid #2a3441;background:#000;
           display:block;border-radius:6px}
 .pane.s2 img{image-rendering:pixelated}
 .cap{color:#8b95a3;font-size:12px;margin-top:6px;text-align:center;
      line-height:1.45}
 .cap b{color:#c9d1d9}
 .wrap .x{position:absolute;left:50%;top:50%;width:18px;height:18px;
   margin:-9px 0 0 -9px;border:2px solid #ff3b3b;border-radius:50%;
   pointer-events:none;box-shadow:0 0 0 1px rgba(0,0,0,.6)}
 .meta{margin:9px 0 2px;color:#9aa4b2;font-size:13px}
 .btns{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;justify-content:center}
 button{padding:9px 15px;font-size:14px;border:0;border-radius:6px;cursor:pointer;
        background:#22304a;color:#fff}
 button:hover{background:#33486e}
 .k{opacity:.55;font-size:12px}
 .prog{margin-top:10px;color:#7d8792;font-size:13px}
 .guide{margin-top:10px;font-size:12px;color:#8b95a3;max-width:720px;
        border-top:1px solid #222c38;padding-top:8px}
 .guide div{padding:1px 0}
 .guide b{color:#c9d1d9;display:inline-block;min-width:92px}
 .undo{background:#3a2a2a;font-size:12px;padding:6px 12px}
</style>
<div class=imgs>
  <div class=pane>
    <div class=wrap><img id=hr><div class=x></div></div>
    <div class=cap><b>Judge the cyan line</b> — ~300 m across.
      Source geometry is good to ~10-15 m, so read the channel nearest the
      line rather than the exact pixel under the dot.</div>
  </div>
  <div class="pane s2">
    <div class=wrap><img id=chip><div class=x></div></div>
    <div class=cap><b>What the model sees</b><br>Sentinel-2, 10 m, dry season</div>
  </div>
</div>
<div class=meta id=meta></div>
<div class=btns id=btns></div>
<div class=prog id=prog></div>
<div class=guide id=guide></div>
<script>
const CLASSES = %CLASSES%;
const GUIDE = %GUIDE%;
let cur = null, last = null;
async function next(){
  const r = await fetch('/next'); const j = await r.json();
  if(j.done){ document.querySelector('.imgs').innerHTML =
      '<h2>All done — data/labels.csv written</h2>';
    document.getElementById('btns').innerHTML=''; return; }
  cur = j;
  document.getElementById('hr').src   = '/chip_hr?id='+j.point_id;
  document.getElementById('chip').src = '/chip?id='+j.point_id;
  document.getElementById('meta').textContent =
    j.can_type+'  ·  '+j.prj_name+'  ·  '+j.point_id;
  document.getElementById('prog').textContent =
    j.done_n+' / '+j.total+' labelled'+(last?'   (last: '+last+' — press U to undo)':'');
}
function send(label){
  if(!cur) return;
  last = label;
  fetch('/label?id='+cur.point_id+'&label='+label).then(next);
}
function undo(){ fetch('/undo').then(()=>{ last=null; next(); }); }
CLASSES.forEach((c,i)=>{
  const b=document.createElement('button');
  b.innerHTML=c+' <span class=k>['+(i+1)+']</span>';
  b.onclick=()=>send(c); document.getElementById('btns').appendChild(b);
});
const ub=document.createElement('button');
ub.className='undo'; ub.innerHTML='undo <span class=k>[U]</span>';
ub.onclick=undo; document.getElementById('btns').appendChild(ub);
document.getElementById('guide').innerHTML =
  GUIDE.map(g=>'<div><b>'+g[0]+'</b> '+g[1]+'</div>').join('');
window.onkeydown=e=>{
  if(e.key==='u'||e.key==='U') return undo();
  const i=parseInt(e.key)-1;
  if(i>=0&&i<CLASSES.length) send(CLASSES[i]); };
next();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == "/":
            page = (PAGE.replace("%CLASSES%", json.dumps(config.LABEL_CLASSES))
                        .replace("%GUIDE%", json.dumps(GUIDE)))
            return self._send(200, "text/html; charset=utf-8", page.encode())

        if u.path == "/next":
            queue, done = load_state()
            todo = queue[~queue["point_id"].isin(done)]
            if todo.empty:
                return self._send(200, "application/json", b'{"done":true}')
            r = todo.iloc[0]
            return self._send(200, "application/json", json.dumps({
                "point_id": r.point_id, "can_type": str(r.can_type),
                "prj_name": str(r.prj_name), "done_n": len(done), "total": len(queue),
            }).encode())

        if u.path == "/chip":
            path = config.CHIPS_DIR / f"{q['id'][0]}.png"
            if not path.exists():
                return self._send(404, "text/plain", b"missing chip")
            return self._send(200, "image/png", path.read_bytes())

        if u.path == "/chip_hr":
            pid = q["id"][0]
            path = config.CHIPS_HR_DIR / f"{pid}.jpg"
            # Fetch on demand. A pre-download pass can die halfway (network,
            # a killed background job) and a missing file renders as a silent
            # black box, which is worse than a slow one — the labeller cannot
            # tell "not downloaded" from "genuinely dark imagery".
            if not chip_is_good(path):
                path.unlink(missing_ok=True)
                try:
                    fetch_one_hr(pid, path)
                except Exception as e:
                    print(f"  on-demand fetch failed for {pid}: {e}")
                    return self._send(200, "image/svg+xml",
                                      placeholder_svg("imagery unavailable"))
            return self._send(200, "image/jpeg", path.read_bytes())

        if u.path == "/undo":
            if config.LABELS_CSV.exists():
                d = pd.read_csv(config.LABELS_CSV)
                if len(d):
                    d.iloc[:-1].to_csv(config.LABELS_CSV, index=False)
            return self._send(200, "application/json", b'{"ok":true}')

        if u.path == "/label":
            pid, label = q["id"][0], q["label"][0]
            queue, _ = load_state()
            row = queue[queue["point_id"] == pid].iloc[0]
            header = not config.LABELS_CSV.exists()
            pd.DataFrame([{
                "point_id": pid, "seg_id": row.seg_id, "label": label,
                "lat": row.lat, "lon": row.lon,
            }]).to_csv(config.LABELS_CSV, mode="a", header=header, index=False)
            return self._send(200, "application/json", b'{"ok":true}')

        self._send(404, "text/plain", b"nope")


def label_ui():
    if not (config.DATA_DIR / "label_queue.csv").exists():
        print("run `python step03_label_points.py fetch` first")
        return 1
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"labelling UI at {url}   (keys 1-{len(config.LABEL_CLASSES)}, Ctrl-C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped — labels saved")


def status():
    queue, done = load_state()
    print(f"{len(done)} / {len(queue)} labelled")
    if done:
        print(pd.Series(list(done.values())).value_counts().to_string())


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "label"
    return {"fetch": fetch_chips, "label": label_ui, "status": status}[cmd]()


if __name__ == "__main__":
    sys.exit(main())
