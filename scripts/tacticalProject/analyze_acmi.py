#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    return EARTH_RADIUS_M * c


def distance_3d_m(p1: "Point", p2: "Point") -> float:
    ground = haversine_m(p1.lat, p1.lon, p2.lat, p2.lon)
    dz = p2.alt - p1.alt
    return math.sqrt(ground * ground + dz * dz)


def displacement_details(p1: "Point", p2: "Point") -> Optional[Dict[str, float]]:
    dt = p2.t - p1.t
    if dt <= 0:
        return None
    ground = haversine_m(p1.lat, p1.lon, p2.lat, p2.lon)
    dz = p2.alt - p1.alt
    d3 = math.sqrt(ground * ground + dz * dz)
    return {
        "t0": p1.t,
        "t1": p2.t,
        "dt": dt,
        "ground_m": ground,
        "dz_m": dz,
        "d3_m": d3,
        "speed_mps": d3 / dt,
    }


def speed_timeline(track: "ObjectTrack", start_t: float, duration_s: float) -> List[Dict[str, float]]:
    if duration_s <= 0:
        return []
    pts = [p for p in track.points if start_t <= p.t <= start_t + duration_s]
    out: List[Dict[str, float]] = []
    for p0, p1 in zip(pts, pts[1:]):
        d = displacement_details(p0, p1)
        if d:
            out.append(d)
    return out


@dataclass(frozen=True)
class Point:
    t: float
    lon: float
    lat: float
    alt: float


@dataclass
class ObjectTrack:
    obj_id: str
    name: Optional[str] = None
    color: Optional[str] = None
    type: Optional[str] = None
    points: List[Point] = None
    _times: List[float] = None

    def __post_init__(self) -> None:
        if self.points is None:
            self.points = []
        if self._times is None:
            self._times = []

    def add_point(self, p: Point) -> None:
        # Assume file is time-ordered.
        self.points.append(p)
        self._times.append(p.t)

    def first_point(self) -> Optional[Point]:
        return self.points[0] if self.points else None

    def second_point(self) -> Optional[Point]:
        return self.points[1] if len(self.points) >= 2 else None

    def speed_from_first_two(self) -> Optional[float]:
        p1 = self.first_point()
        p2 = self.second_point()
        if not p1 or not p2:
            return None
        details = displacement_details(p1, p2)
        return None if details is None else float(details["speed_mps"])

    def speed_details_first_two(self) -> Optional[Dict[str, float]]:
        p1 = self.first_point()
        p2 = self.second_point()
        if not p1 or not p2:
            return None
        return displacement_details(p1, p2)

    def pos_at_or_before(self, t: float) -> Optional[Point]:
        if not self.points:
            return None
        idx = bisect_right(self._times, t) - 1
        if idx < 0:
            return None
        return self.points[idx]


def parse_acmi_text(path: Path) -> Dict[str, ObjectTrack]:
    tracks: Dict[str, ObjectTrack] = {}
    current_time: Optional[float] = None

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                try:
                    current_time = float(line[1:])
                except ValueError:
                    current_time = None
                continue

            if current_time is None:
                continue

            # Object update line: <ID>,T=lon|lat|alt|...,Name=...,Color=...,Type=...
            if "," not in line:
                continue

            parts = line.split(",")
            obj_id = parts[0].strip()
            if not obj_id:
                continue

            fields: Dict[str, str] = {}
            for part in parts[1:]:
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                fields[k.strip()] = v.strip()

            track = tracks.get(obj_id)
            if track is None:
                track = ObjectTrack(obj_id=obj_id)
                tracks[obj_id] = track

            if "Name" in fields and track.name is None:
                track.name = fields["Name"]
            if "Color" in fields and track.color is None:
                track.color = fields["Color"]
            if "Type" in fields and track.type is None:
                track.type = fields["Type"]

            if "T" in fields:
                t_parts = fields["T"].split("|")
                if len(t_parts) >= 3:
                    try:
                        lon = float(t_parts[0])
                        lat = float(t_parts[1])
                        alt = float(t_parts[2])
                    except ValueError:
                        continue
                    track.add_point(Point(t=current_time, lon=lon, lat=lat, alt=alt))

    return tracks


def _name_looks_like_missile(name: Optional[str]) -> bool:
    if not name:
        return False
    n = name.strip().upper()
    # Conservative keyword set; extend if needed.
    return any(
        k in n
        for k in (
            "AIM-",
            "AMRAAM",
            "R-",
            "PL-",
            "SD-",
            "MISSILE",
            "MICA",
        )
    )


def classify_track_ids(tracks: Dict[str, ObjectTrack]) -> Tuple[Dict[str, ObjectTrack], Dict[str, ObjectTrack]]:
    """Return (aircraft_tracks, missile_tracks) with robust inference.

    Some ACMI exports omit `Type=...`; in that case we infer:
    - aircraft: 5-char IDs like A0100/B0200
    - missiles: IDs longer than 5 whose first 5 chars match an aircraft ID
    """
    aircraft_ids = {obj_id for obj_id in tracks.keys() if len(obj_id) == 5 and obj_id[0] in {"A", "B"}}

    aircraft: Dict[str, ObjectTrack] = {}
    missiles: Dict[str, ObjectTrack] = {}

    for obj_id, tr in tracks.items():
        if not tr.first_point():
            continue

        # If Type is present, trust it.
        if tr.type:
            if "Weapon" in tr.type and "Missile" in tr.type:
                missiles[obj_id] = tr
                continue
            if "Air" in tr.type:
                aircraft[obj_id] = tr
                continue

        # No Type: infer from ID linkage and/or name.
        if len(obj_id) == 5 and obj_id in aircraft_ids:
            aircraft[obj_id] = tr
            continue

        shooter_id = infer_shooter_id(obj_id)
        if shooter_id and shooter_id in aircraft_ids and len(obj_id) > 5:
            missiles[obj_id] = tr
            continue

        if _name_looks_like_missile(tr.name):
            missiles[obj_id] = tr
            continue

        # Fallback: treat as aircraft-like track.
        aircraft[obj_id] = tr

    return aircraft, missiles


def infer_side(track: ObjectTrack) -> Optional[str]:
    # Prefer stable side tags, but fallback to ID prefix.
    if track.color:
        c = track.color.lower()
        if c in {"red", "blue"}:
            return c
    if track.obj_id.startswith("A"):
        return "red"
    if track.obj_id.startswith("B"):
        return "blue"
    return None


def infer_shooter_id(missile_id: str) -> Optional[str]:
    # In these logs: A0100 (aircraft) -> A01002 (missile)
    return missile_id[:5] if len(missile_id) >= 5 else None


def summarize(acmi_path: Path, out_json: Optional[Path], timeline_seconds: float = 0.0) -> int:
    tracks = parse_acmi_text(acmi_path)

    aircraft, missiles = classify_track_ids(tracks)

    # Aircraft initial states
    aircraft_init: Dict[str, Dict[str, Any]] = {}
    for obj_id, tr in sorted(aircraft.items()):
        p0 = tr.first_point()
        v0 = tr.speed_from_first_two()
        v0_details = tr.speed_details_first_two()
        aircraft_init[obj_id] = {
            "name": tr.name,
            "side": infer_side(tr),
            "t0": p0.t,
            "lon0": p0.lon,
            "lat0": p0.lat,
            "alt0_m": p0.alt,
            "speed0_mps": v0,
            "speed0_details": v0_details,
        }

    # Initial red-blue separations (pairwise)
    reds = [tr for tr in aircraft.values() if infer_side(tr) == "red" and tr.first_point()]
    blues = [tr for tr in aircraft.values() if infer_side(tr) == "blue" and tr.first_point()]

    pairwise: List[Dict[str, Any]] = []
    for r in reds:
        for b in blues:
            pr0 = r.first_point()
            pb0 = b.first_point()
            d0 = distance_3d_m(pr0, pb0)

            # Rough initial range-rate using the next sample if available
            rr = None
            pr1 = r.second_point()
            pb1 = b.second_point()
            if pr1 and pb1 and pr1.t == pb1.t and pr1.t > pr0.t:
                d1 = distance_3d_m(pr1, pb1)
                rr = (d1 - d0) / (pr1.t - pr0.t)

            pairwise.append(
                {
                    "red": r.obj_id,
                    "blue": b.obj_id,
                    "range0_m": d0,
                    "range_rate_mps": rr,
                }
            )

    pairwise_sorted = sorted(pairwise, key=lambda x: x["range0_m"])
    ranges = [p["range0_m"] for p in pairwise_sorted]
    sep_summary = {
        "pairs": pairwise_sorted,
        "min_m": min(ranges) if ranges else None,
        "mean_m": (sum(ranges) / len(ranges)) if ranges else None,
        "max_m": max(ranges) if ranges else None,
    }

    # Missile launches
    launches: List[Dict[str, Any]] = []
    for mid, mtr in sorted(missiles.items()):
        mp0 = mtr.first_point()
        m_speed0 = mtr.speed_from_first_two()
        m_speed0_details = mtr.speed_details_first_two()

        shooter_id = infer_shooter_id(mid)
        shooter_tr = aircraft.get(shooter_id) if shooter_id else None
        shooter_p = shooter_tr.pos_at_or_before(mp0.t) if shooter_tr else None

        side = infer_side(mtr) or (infer_side(shooter_tr) if shooter_tr else None)

        # Infer target as nearest enemy aircraft at launch time.
        target_id = None
        target_range_m = None
        shooter_target_range_m = None

        enemy_tracks = (
            [tr for tr in aircraft.values() if infer_side(tr) == "blue"]
            if side == "red"
            else [tr for tr in aircraft.values() if infer_side(tr) == "red"]
            if side == "blue"
            else []
        )

        best: Tuple[Optional[str], Optional[float]] = (None, None)
        for etr in enemy_tracks:
            ep = etr.pos_at_or_before(mp0.t)
            if not ep:
                continue
            d = distance_3d_m(mp0, ep)
            if best[1] is None or d < best[1]:
                best = (etr.obj_id, d)

        target_id, target_range_m = best
        if shooter_p and target_id:
            target_p = aircraft[target_id].pos_at_or_before(mp0.t)
            if target_p:
                shooter_target_range_m = distance_3d_m(shooter_p, target_p)

        timeline = speed_timeline(mtr, mp0.t, timeline_seconds) if timeline_seconds > 0 else []
        peak_speed = None
        peak_at_t = None
        if timeline:
            best = max(timeline, key=lambda x: x.get("speed_mps", float("-inf")))
            peak_speed = float(best["speed_mps"])
            peak_at_t = float(best["t1"])

        launches.append(
            {
                "missile_id": mid,
                "missile_name": mtr.name,
                "side": side,
                "t_launch": mp0.t,
                "launch_lon": mp0.lon,
                "launch_lat": mp0.lat,
                "launch_alt_m": mp0.alt,
                "missile_speed0_mps": m_speed0,
                "missile_speed0_details": m_speed0_details,
                "timeline_seconds": timeline_seconds if timeline_seconds > 0 else None,
                "speed_timeline": timeline if timeline_seconds > 0 else None,
                "peak_speed_mps_within_window": peak_speed,
                "peak_speed_time_s": peak_at_t,
                "shooter_id": shooter_id,
                "target_id_inferred": target_id,
                "missile_to_target_range_m": target_range_m,
                "shooter_to_target_range_m": shooter_target_range_m,
            }
        )

    # Print a human-readable summary
    print(f"ACMI: {acmi_path}")
    print(f"Aircraft: {len(aircraft)} | Missiles: {len(missiles)}")

    print("\nInitial aircraft states (t0):")
    for obj_id, info in aircraft_init.items():
        sp = info["speed0_mps"]
        sp_s = f"{sp:.1f}" if isinstance(sp, (int, float)) else "N/A"
        print(
            f"- {obj_id} ({info.get('side')}) lat={info['lat0']:.6f} lon={info['lon0']:.6f} alt={info['alt0_m']:.1f}m v0={sp_s} m/s"
        )

    if sep_summary["min_m"] is not None:
        print("\nInitial red-blue separations (t0):")
        print(
            f"- min={sep_summary['min_m']:.1f} m | mean={sep_summary['mean_m']:.1f} m | max={sep_summary['max_m']:.1f} m"
        )
        print("- closest pairs:")
        for p in sep_summary["pairs"][: min(6, len(sep_summary["pairs"]))]:
            rr = p.get("range_rate_mps")
            rr_s = f"{rr:+.1f} m/s" if isinstance(rr, (int, float)) else "N/A"
            print(f"  - {p['red']} vs {p['blue']}: {p['range0_m']:.1f} m (range_rate {rr_s})")

    if launches:
        print("\nMissile launches (first appearance):")
        for l in launches:
            v0 = l.get("missile_speed0_mps")
            v0_s = f"{v0:.1f}" if isinstance(v0, (int, float)) else "N/A"
            r_mt = l.get("missile_to_target_range_m")
            r_mt_s = f"{r_mt/1000:.2f} km" if isinstance(r_mt, (int, float)) else "N/A"
            r_st = l.get("shooter_to_target_range_m")
            r_st_s = f"{r_st/1000:.2f} km" if isinstance(r_st, (int, float)) else "N/A"
            peak_v = l.get("peak_speed_mps_within_window")
            peak_t = l.get("peak_speed_time_s")
            peak_s = (
                f" peak={peak_v:.1f} m/s @t={peak_t:.2f}s" if isinstance(peak_v, (int, float)) and isinstance(peak_t, (int, float)) else ""
            )
            print(
                f"- t={l['t_launch']:.2f}s {l['missile_id']} ({l.get('missile_name')}, {l.get('side')}) v0={v0_s} m/s shooter={l.get('shooter_id')} target≈{l.get('target_id_inferred')} missile->target={r_mt_s} shooter->target={r_st_s}{peak_s}"
            )
            if timeline_seconds and l.get("speed_timeline"):
                print(f"  timeline (first {timeline_seconds:.1f}s, segment speeds):")
                for seg in l["speed_timeline"]:
                    print(
                        f"    - {seg['t0']:.2f}->{seg['t1']:.2f} dt={seg['dt']:.2f}s d3={seg['d3_m']:.1f}m v={seg['speed_mps']:.1f} m/s"
                    )

    result = {
        "acmi_path": str(acmi_path),
        "aircraft": aircraft_init,
        "initial_red_blue_separation": sep_summary,
        "missile_launches": launches,
    }

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {out_json}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Tacview text ACMI (.txt.acmi) and extract initial speeds/separations and missile launch metrics.")
    parser.add_argument("acmi", nargs="?", type=Path, default=None, help="Path to .txt.acmi file (optional; defaults to a sample/latest file)")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--debug-id",
        type=str,
        default=None,
        help="Optional object ID (e.g., A01002) to print the two frames used for speed estimation",
    )
    parser.add_argument(
        "--timeline-seconds",
        type=float,
        default=0.0,
        help="If >0, print per-segment speed timeline for each missile for the first N seconds after launch",
    )
    args = parser.parse_args()

    # PyCharm one-click runs typically pass no args; default to a useful missile speed window.
    if len(sys.argv) == 1 and float(args.timeline_seconds) == 0.0:
        args.timeline_seconds = 10.0

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "scripts" / "tacticalProject" / "tactical_simulation_results"

    if args.acmi is None:
        # Prefer a stable sample file if present, otherwise pick the newest .acmi.
        preferred = results_dir / "0104_001710_auto_select_2v2.txt.acmi"
        if preferred.exists():
            args.acmi = preferred
        else:
            candidates = sorted(results_dir.glob("*.acmi"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise SystemExit(
                    "No ACMI file provided and none found under: "
                    f"{results_dir} (expected e.g. 1_v1.txt.acmi)"
                )
            args.acmi = candidates[0]

    # Make relative paths work regardless of IDE working directory.
    if not args.acmi.is_absolute():
        args.acmi = (repo_root / args.acmi).resolve()

    if not args.acmi.exists():
        raise SystemExit(f"File not found: {args.acmi}")

    # For one-click runs, emit a JSON file by default (can be overridden via --out-json).
    if args.out_json is None:
        args.out_json = Path(str(args.acmi) + ".analysis.json")

    rc = summarize(args.acmi, args.out_json, timeline_seconds=float(args.timeline_seconds))

    if args.debug_id:
        tracks = parse_acmi_text(args.acmi)
        tr = tracks.get(args.debug_id)
        if tr is None:
            print(f"\n[debug] Object not found: {args.debug_id}")
        else:
            d = tr.speed_details_first_two()
            p0 = tr.first_point()
            p1 = tr.second_point()
            print(f"\n[debug] {args.debug_id} name={tr.name} type={tr.type}")
            if p0:
                print(f"[debug] p0: t={p0.t:.2f} lon={p0.lon:.8f} lat={p0.lat:.8f} alt={p0.alt:.2f}")
            if p1:
                print(f"[debug] p1: t={p1.t:.2f} lon={p1.lon:.8f} lat={p1.lat:.8f} alt={p1.alt:.2f}")
            if d:
                print(
                    "[debug] dt={dt:.3f}s ground={ground_m:.2f}m dz={dz_m:.2f}m d3={d3_m:.2f}m => v={speed_mps:.2f}m/s".format(
                        **d
                    )
                )
            else:
                print("[debug] Not enough points to compute speed")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
