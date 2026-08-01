"""Step 1: canal KML -> noded network graph.

Pipeline:
  1. Filter the national canal layer to the Cauvery delta AOI -> data/canals.gpkg
  2. Explode multilines, reproject to UTM 44N
  3. Node the network: snap endpoints together and split lines at T-junctions
  4. Build an undirected graph, report connected components
  5. Orient edges by BFS from the Grand Anicut source point
  6. Write data/segments.gpkg (edges with stable seg_id) and data/graph.pkl

Run:  python step01_build_graph.py
"""

import hashlib
import pickle
import sys
from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyogrio
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import substring

import config


def load_and_filter():
    print(f"Reading {config.RAW_KML.name} ...")
    df = pyogrio.read_dataframe(
        config.RAW_KML,
        columns=["prj_name", "can_name", "can_type", "basin", "river", "state", "length_km"],
    )
    n_total = len(df)
    df = df[(df["state"] == config.FILTER_STATE) & (df["basin"] == config.FILTER_BASIN)]
    minx, miny, maxx, maxy = config.AOI_BBOX
    b = df.geometry.bounds
    df = df[(b["maxx"] >= minx) & (b["minx"] <= maxx) & (b["maxy"] >= miny) & (b["miny"] <= maxy)]
    df = df.set_crs(config.CRS_WGS84, allow_override=True)
    print(f"  {n_total} canals nationally -> {len(df)} in AOI "
          f"({df['length_km'].astype(float).sum():.0f} km)")

    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_file(config.CANALS_GPKG, driver="GPKG")
    print(f"  wrote {config.CANALS_GPKG}")
    df["is_river"] = 0
    return df


def load_rivers():
    """Rivers connect the canal systems; flagged so ranking can exclude them."""
    if not config.RIVERS_GPKG.exists():
        print("  no rivers.gpkg — run step00_fetch_rivers.py for a connected network")
        return None
    rv = gpd.read_file(config.RIVERS_GPKG)
    rv = rv[rv["waterway"] == "river"]
    rv = gpd.GeoDataFrame({
        "prj_name": "RIVER",
        "can_name": rv["name"].fillna("river"),
        "can_type": "River",
        "is_river": 1,
        "geometry": rv.geometry,
    }, crs=rv.crs)
    print(f"  {len(rv)} river connector lines loaded")
    return rv


def explode_lines(df):
    df = df.to_crs(config.CRS_UTM)
    df = df.explode(index_parts=False).reset_index(drop=True)
    df = df[df.geometry.geom_type == "LineString"]
    df = df[df.geometry.length > 0]
    print(f"  exploded to {len(df)} LineStrings")
    return df


def node_network(df):
    """Split lines at T-junctions and where endpoints nearly coincide."""
    tol = config.SNAP_TOLERANCE_M
    lines = list(df.geometry)
    tree = STRtree(lines)

    # distance along each line at which it must be cut
    cut_dists = defaultdict(set)
    for i, line in enumerate(lines):
        for pt in (Point(line.coords[0]), Point(line.coords[-1])):
            for j in tree.query(pt.buffer(tol)):
                j = int(j)
                if j == i:
                    continue
                other = lines[j]
                if other.distance(pt) <= tol:
                    d = other.project(pt)
                    if tol < d < other.length - tol:  # interior hit -> T-junction
                        cut_dists[j].add(round(d, 1))

    segments = []
    for i, line in enumerate(lines):
        attrs = df.iloc[i][["prj_name", "can_name", "can_type", "is_river"]].to_dict()
        dists = sorted(cut_dists.get(i, set()))
        breaks = [0.0] + dists + [line.length]
        for a, b in zip(breaks[:-1], breaks[1:]):
            if b - a < config.MIN_SEGMENT_M:
                continue
            segments.append((substring(line, a, b), attrs))
    print(f"  noded into {len(segments)} segments "
          f"({sum(len(v) for v in cut_dists.values())} T-junction cuts)")
    return segments


def cluster_endpoints(segments):
    """Union-find over segment endpoints within tolerance -> canonical node ids."""
    tol = config.SNAP_TOLERANCE_M
    pts = []
    for geom, _ in segments:
        pts.append(geom.coords[0])
        pts.append(geom.coords[-1])
    pts = np.array(pts)

    parent = list(range(len(pts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = STRtree([Point(p) for p in pts])
    for i, p in enumerate(pts):
        for j in tree.query(Point(p).buffer(tol)):
            j = int(j)
            if np.hypot(*(pts[j] - p)) <= tol:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    clusters = defaultdict(list)
    for i in range(len(pts)):
        clusters[find(i)].append(i)
    node_xy = {}
    point_to_node = {}
    for root, members in clusters.items():
        centroid = pts[members].mean(axis=0)
        nid = (round(float(centroid[0]), 1), round(float(centroid[1]), 1))
        node_xy[nid] = centroid
        for m in members:
            point_to_node[m] = nid
    print(f"  {len(pts)} endpoints -> {len(node_xy)} graph nodes")
    return point_to_node


def stable_seg_id(geom):
    key = f"{geom.coords[0][0]:.0f},{geom.coords[0][1]:.0f}|" \
          f"{geom.coords[-1][0]:.0f},{geom.coords[-1][1]:.0f}|{geom.length:.0f}"
    return "SEG_" + hashlib.md5(key.encode()).hexdigest()[:10]


def build_graph(segments, point_to_node):
    G = nx.Graph()
    rows = []
    for k, (geom, attrs) in enumerate(segments):
        u = point_to_node[2 * k]
        v = point_to_node[2 * k + 1]
        if u == v:
            continue
        sid = stable_seg_id(geom)
        data = dict(seg_id=sid, length_m=geom.length, geometry=geom, **attrs)
        # keep the longer edge if two canals run between the same node pair
        if G.has_edge(u, v) and G[u][v]["length_m"] >= geom.length:
            continue
        G.add_edge(u, v, **data)
    for u, v, d in G.edges(data=True):
        rows.append(d | {"u": u, "v": v})
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, rows


def report_components(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    print(f"  connected components: {len(comps)}")
    for i, c in enumerate(comps[:10]):
        km = sum(d["length_m"] for u, v, d in G.subgraph(c).edges(data=True)) / 1000
        print(f"    component {i}: {len(c)} nodes, {km:.1f} km")
    if len(comps) > config.MAX_EXPECTED_COMPONENTS:
        print(f"  WARNING: more than {config.MAX_EXPECTED_COMPONENTS} components — "
              f"consider raising SNAP_TOLERANCE_M (now {config.SNAP_TOLERANCE_M} m)")
    return comps


def _split_edge(G, u, v, dist_along):
    """Split edge (u,v) at dist_along its geometry; return the new middle node."""
    d = G[u][v]
    geom = d["geometry"]
    pt = geom.interpolate(dist_along)
    w = (round(pt.x, 1), round(pt.y, 1))
    if w == u or w == v:
        return u if w == u else v
    g1, g2 = substring(geom, 0, dist_along), substring(geom, dist_along, geom.length)
    base = {k: val for k, val in d.items() if k not in ("seg_id", "length_m", "geometry")}
    G.remove_edge(u, v)
    G.add_edge(u, w, seg_id=stable_seg_id(g1), length_m=g1.length, geometry=g1, **base)
    G.add_edge(w, v, seg_id=stable_seg_id(g2), length_m=g2.length, geometry=g2, **base)
    return w


def attach_components(G):
    """Link floating components to the main network with virtual connectors,
    standing in for offtakes the source data never mapped. River-only
    fragments that never attach are dropped."""
    tol = config.ATTACH_TOLERANCE_M

    def main_comp():
        lat, lon = config.SOURCE_LATLON
        src = gpd.GeoSeries([Point(lon, lat)], crs=config.CRS_WGS84).to_crs(config.CRS_UTM)[0]
        best = min(G.nodes, key=lambda n: np.hypot(n[0] - src.x, n[1] - src.y))
        return nx.node_connected_component(G, best)

    n_links = 0
    while True:
        main = main_comp()
        edges = [(u, v) for u, v in G.edges if u in main]
        tree = STRtree([G[u][v]["geometry"] for u, v in edges])

        best = None  # (dist, node, edge_idx)
        for comp in nx.connected_components(G):
            if next(iter(comp)) in main or not any(
                not G[u][v].get("is_river") for u, v in G.subgraph(comp).edges
            ):
                continue
            for n in comp:
                idx, dist = tree.query_nearest(Point(n), return_distance=True)
                if len(idx) and (best is None or dist[0] < best[0]):
                    best = (float(dist[0]), n, int(idx[0]))
        if best is None or best[0] > tol:
            break

        dist, n, ei = best
        u, v = edges[ei]
        target_geom = G[u][v]["geometry"]
        w = _split_edge(G, u, v, target_geom.project(Point(n)))
        n_links += 1
        if w != n:
            cg = LineString([w, n])
            G.add_edge(w, n, seg_id=f"CONN_{n_links:03d}",
                       length_m=cg.length, geometry=cg, prj_name="CONNECTOR",
                       can_name="virtual offtake link", can_type="Connector",
                       is_river=0, is_virtual=1)
        elif not nx.has_path(G, n, next(iter(main))):
            # same node back, components still separate — bail before looping
            print(f"    WARNING: could not attach component near {n}")
            break
        print(f"    connector {n_links}: {dist:.0f} m link near node {n}")

    main = main_comp()
    drop = []
    for comp in nx.connected_components(G):
        if next(iter(comp)) not in main and not any(
            not G[u][v].get("is_river") for u, v in G.subgraph(comp).edges
        ):
            drop.extend(comp)
    if drop:
        km = sum(d["length_m"] for u, v, d in G.subgraph(drop).edges(data=True)) / 1000
        G.remove_nodes_from(drop)
        print(f"  attached {n_links} components; dropped {km:.0f} km of "
              f"unattached river fragments")
    return G


def orient(G, comps):
    """BFS outward from the node nearest the source. Each component gets its
    own local source: whichever of its nodes is nearest the global one."""
    lat, lon = config.SOURCE_LATLON
    src_pt = gpd.GeoSeries([Point(lon, lat)], crs=config.CRS_WGS84).to_crs(config.CRS_UTM)[0]

    D = nx.DiGraph()
    for ci, comp in enumerate(comps):
        nodes = list(comp)
        dists = [np.hypot(n[0] - src_pt.x, n[1] - src_pt.y) for n in nodes]
        local_src = nodes[int(np.argmin(dists))]
        if ci == 0 or ci < len(comps):
            tag = "SOURCE" if ci == 0 else f"component {ci} local source"
            print(f"  {tag}: node {local_src}, {min(dists)/1000:.2f} km from Grand Anicut")
        order = nx.single_source_shortest_path_length(G.subgraph(comp), local_src)
        for u, v, d in G.subgraph(comp).edges(data=True):
            if order[u] <= order[v]:
                D.add_edge(u, v, **d)
            else:
                D.add_edge(v, u, **d)
        D.nodes[local_src]["is_source"] = True
        for n in comp:
            D.nodes[n]["component"] = ci
    return D


def write_outputs(D):
    rows = []
    for u, v, d in D.edges(data=True):
        rows.append({
            "seg_id": d["seg_id"],
            "prj_name": d.get("prj_name"),
            "can_name": d.get("can_name"),
            "can_type": d.get("can_type"),
            "is_river": d.get("is_river", 0),
            "length_m": d["length_m"],
            "from_node": str(u),
            "to_node": str(v),
            "component": D.nodes[u].get("component", -1),
            "geometry": d["geometry"],
        })
    gdf = gpd.GeoDataFrame(rows, crs=config.CRS_UTM)
    dup = gdf["seg_id"].duplicated().sum()
    if dup:
        print(f"  WARNING: {dup} duplicate seg_ids")
    gdf.to_file(config.SEGMENTS_GPKG, driver="GPKG")
    print(f"  wrote {config.SEGMENTS_GPKG} ({len(gdf)} segments)")

    # geometry objects don't pickle cleanly across nx versions — store WKB
    P = nx.DiGraph()
    P.add_nodes_from(D.nodes(data=True))
    for u, v, d in D.edges(data=True):
        d2 = {k: v2 for k, v2 in d.items() if k != "geometry"}
        d2["wkb"] = d["geometry"].wkb
        P.add_edge(u, v, **d2)
    with open(config.GRAPH_PKL, "wb") as f:
        pickle.dump(P, f)
    print(f"  wrote {config.GRAPH_PKL}")


def main():
    df = load_and_filter()
    rivers = load_rivers()
    if rivers is not None:
        df = gpd.GeoDataFrame(
            pd.concat([df, rivers], ignore_index=True), crs=df.crs
        )
    df = explode_lines(df)
    segments = node_network(df)
    point_to_node = cluster_endpoints(segments)
    G, _ = build_graph(segments, point_to_node)
    report_components(G)
    G = attach_components(G)
    print("after attachment:")
    comps = report_components(G)
    D = orient(G, comps)
    write_outputs(D)
    print("Step 1 done.")


if __name__ == "__main__":
    sys.exit(main())
