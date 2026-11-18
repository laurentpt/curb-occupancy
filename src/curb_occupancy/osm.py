"""OpenStreetMap street geometry: the street's centreline, cross streets and curb cuts.

Data (c) OpenStreetMap contributors, ODbL. `fetch` queries the Overpass API for every
`highway` way in a bounding box and caches the JSON (config `street.osm_json`).

* street ways: car roads named `street.name` (or, without a name, the road most GPS samples
  of the reference pass are closest to), chained node to node into one polyline;
* crossings: other ways sharing a node with the street. Car roads = intersections (with the
  cross street's name and the side(s) it joins from); `highway=service` (alleys, driveways,
  parking aisles) = curb cuts. Footways, cycleways, paths and steps are ignored (crosswalks
  and bike links do not interrupt the curb).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CAR_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
             "residential", "living_street", "road", "motorway_link", "trunk_link",
             "primary_link", "secondary_link", "tertiary_link"}
CURB_CUTS = {"service"}


def fetch(bbox: tuple, path: str | Path, timeout: int = 120) -> dict:
    """bbox = (south, west, north, east) degrees."""
    import urllib.parse
    import urllib.request
    s, w, n, e = bbox
    q = f'[out:json][timeout:{timeout}];way["highway"]({s},{w},{n},{e});out body geom;'
    req = urllib.request.Request(OVERPASS_URL, data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "curb-occupancy"})
    with urllib.request.urlopen(req, timeout=timeout + 30) as r:
        d = json.load(r)
    d["query_bbox"] = list(bbox)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d))
    return d


def _ways(osm: dict) -> list[dict]:
    return [e for e in osm["elements"] if e.get("type") == "way" and "geometry" in e]


def node_coords(osm: dict) -> dict:
    """node id -> (lat, lon) for every node of the extract's ways."""
    return _node_coords(_ways(osm))


def _node_coords(ways) -> dict:
    out = {}
    for w in ways:
        for nid, g in zip(w["nodes"], w["geometry"]):
            out[nid] = (g["lat"], g["lon"])
    return out


def street_ways(osm: dict, name: str | None = None, track_ll: np.ndarray | None = None):
    """Car-road ways of the street: by name (case-insensitive), or the name most track points
    are nearest to. Returns (ways, name)."""
    from .street import local_xy
    roads = [w for w in _ways(osm) if w.get("tags", {}).get("highway") in CAR_ROADS]
    if name is None:
        if track_ll is None:
            raise ValueError("street.name or a GPS track is needed to pick the street")
        lat0, lon0 = float(track_ll[:, 0].mean()), float(track_ll[:, 1].mean())
        pts = local_xy(track_ll[::10, 0], track_ll[::10, 1], lat0, lon0)
        votes = Counter()
        segs = []
        for w in roads:
            g = np.array([(p["lat"], p["lon"]) for p in w["geometry"]])
            segs.append((w.get("tags", {}).get("name"), local_xy(g[:, 0], g[:, 1], lat0, lon0)))
        for q in pts:
            best = min(((_dist_polyline(q, P), n) for n, P in segs if n), default=None)
            if best is not None and best[0] < 15:
                votes[best[1]] += 1
        if not votes:
            raise ValueError("no named road within 15 m of the GPS track")
        name = votes.most_common(1)[0][0]
    sel = [w for w in roads if w.get("tags", {}).get("name", "").casefold() == name.casefold()]
    if not sel:
        raise ValueError(f"no car road named {name!r} in the OSM extract")
    return sel, name


def _dist_polyline(q, P) -> float:
    a, b = P[:-1], P[1:]
    d = b - a
    L2 = np.maximum((d ** 2).sum(1), 1e-9)
    u = np.clip(((q - a) * d).sum(1) / L2, 0, 1)
    return float(np.hypot(*(q - (a + u[:, None] * d)).T).min())


def _adjacency(ways):
    adj = defaultdict(set)
    for w in ways:
        for a, b in zip(w["nodes"][:-1], w["nodes"][1:]):
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    return adj


def components(ways: list[dict]) -> list[list[dict]]:
    """Ways grouped into connected pieces (a street name can be reused for separate
    stretches)."""
    adj = _adjacency(ways)
    comp, k = {}, 0
    for n in adj:
        if n in comp:
            continue
        stack = [n]
        while stack:
            m = stack.pop()
            if m not in comp:
                comp[m] = k
                stack.extend(adj[m])
        k += 1
    groups = defaultdict(list)
    for w in ways:
        groups[comp[w["nodes"][0]]].append(w)
    return list(groups.values())


def chain(ways: list[dict]) -> list[int] | None:
    """Node ids of the street as one path (ways joined at shared nodes, any direction), or
    None if the ways do not form a single simple path (dual carriageways, branches)."""
    adj = _adjacency(ways)
    if any(len(v) > 2 for v in adj.values()):
        return None
    ends = [n for n, v in adj.items() if len(v) == 1]
    if len(ends) != 2:
        return None
    path, prev, cur = [ends[0]], None, ends[0]
    while True:
        nxt = [n for n in adj[cur] if n != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        path.append(cur)
    return path if len(path) == len(adj) else None


def crossings(osm: dict, street: list[dict], path: list[int]) -> list[dict]:
    """Ways sharing a node with the street path: {node, i (index on the path), kind
    ('intersection' | 'curb_cut'), name, highway, service, neighbours (lat/lon of the joining
    way's nodes next to the shared node)}."""
    on_path = {n: i for i, n in enumerate(path)}
    ids = {w["id"] for w in street}
    coords = _node_coords(_ways(osm))
    out = []
    for w in _ways(osm):
        if w["id"] in ids:
            continue
        tags = w.get("tags", {})
        hw = tags.get("highway")
        kind = "intersection" if hw in CAR_ROADS else "curb_cut" if hw in CURB_CUTS else None
        if kind is None:
            continue
        for k, n in enumerate(w["nodes"]):
            if n not in on_path:
                continue
            nb = [w["nodes"][j] for j in (k - 1, k + 1) if 0 <= j < len(w["nodes"])]
            out.append({"node": n, "i": on_path[n], "kind": kind, "way": w["id"],
                        "name": tags.get("name", ""), "highway": hw,
                        "service": tags.get("service", ""),
                        "neighbours": [coords[m] for m in nb if m in coords and m not in on_path]})
    return out


def street_polyline(osm: dict, name: str | None, track_ll: np.ndarray):
    """(lat/lon polyline of the street ordered along the track, crossings, name, path ok)."""
    from .street import local_xy, project_polyline
    ways, name = street_ways(osm, name, track_ll)
    lat0, lon0 = float(track_ll[:, 0].mean()), float(track_ll[:, 1].mean())
    pts = local_xy(track_ll[::20, 0], track_ll[::20, 1], lat0, lon0)

    def support(ws):  # track samples within 20 m of this piece of the street
        P = [local_xy(*np.array([(g["lat"], g["lon"]) for g in w["geometry"]]).T, lat0, lon0)
             for w in ws]
        return sum(min(_dist_polyline(q, Q) for Q in P) < 20 for q in pts)
    pieces = components(ways)
    ways = max(pieces, key=support) if len(pieces) > 1 else ways
    coords = _node_coords(ways)
    path = chain(ways)
    chained = path is not None
    if not chained:
        # fallback: street nodes near the track, ordered by their position along the track
        lat0, lon0 = float(track_ll[:, 0].mean()), float(track_ll[:, 1].mean())
        T = local_xy(track_ll[:, 0], track_ll[:, 1], lat0, lon0)
        X = np.r_[0, np.cumsum(np.hypot(*np.diff(T, axis=0).T))]
        nodes = list(coords)
        pr = project_polyline(local_xy(*np.array([coords[n] for n in nodes]).T, lat0, lon0), T, X)
        keep = [(x, n) for (d, x, _), n in zip(pr, nodes) if d < 20]
        path = [n for _, n in sorted(keep)]
    ll = np.array([coords[n] for n in path])
    return ll, path, crossings(osm, ways, path), name, chained
