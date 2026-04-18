import csv
import math
import sys
from pathlib import Path


def nav_heading_from_v(vx: float, vy: float) -> float:
    """Convert velocity components (x=east, y=north) to nav heading deg (0=north, clockwise)."""
    math_angle = math.degrees(math.atan2(vy, vx))
    return (90.0 - math_angle) % 360.0


def speed(vx: float, vy: float, vz: float) -> float:
    return math.sqrt(vx * vx + vy * vy + vz * vz)


def load_rows(path: Path):
    by_agent = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            aid = r["agent_id"]
            t = float(r["time"])
            x = float(r["x"])
            y = float(r["y"])
            z = float(r["z"])
            vx = float(r["vx"])
            vy = float(r["vy"])
            vz = float(r["vz"])
            by_agent.setdefault(aid, []).append((t, x, y, z, vx, vy, vz))
    return by_agent


def summarize(path: Path):
    data = load_rows(path)

    agents = sorted(data.keys())
    print(f"== {path.name} ==")
    print(f"agents: {', '.join(agents)}")

    for aid in agents:
        rows = data[aid]
        t0, x0, y0, z0, vx0, vy0, vz0 = rows[0]
        t1, x1, y1, z1, vx1, vy1, vz1 = rows[-1]
        s0 = speed(vx0, vy0, vz0)
        s1 = speed(vx1, vy1, vz1)
        smin = min(speed(vx, vy, vz) for _, _, _, _, vx, vy, vz in rows)
        h0 = nav_heading_from_v(vx0, vy0)
        h1 = nav_heading_from_v(vx1, vy1)
        print(
            f"{aid}: t[{t0:.1f}->{t1:.1f}] pos0=({x0:.0f},{y0:.0f},{z0:.0f}) pos1=({x1:.0f},{y1:.0f},{z1:.0f}) "
            f"hdg0={h0:.1f} hdg1={h1:.1f} spd0={s0:.1f} spd1={s1:.1f} spd_min={smin:.1f}"
        )

    # Simple lateral separation metrics for A0100/A0200
    if "A0100" in data and "A0200" in data:
        a0 = {t: (x, y) for t, x, y, *_ in data["A0100"]}
        a2 = {t: (x, y) for t, x, y, *_ in data["A0200"]}
        common_t = sorted(set(a0.keys()) & set(a2.keys()))
        if common_t:
            sep = [abs(a2[t][1] - a0[t][1]) for t in common_t]
            print(f"A0100/A0200 lateral |dy|: start={sep[0]:.0f}m max={max(sep):.0f}m end={sep[-1]:.0f}m")

    print()


def main(argv):
    if len(argv) < 2:
        print("Usage: python analyze_trajectory_csv.py <trajectory.csv> [more.csv ...]")
        return 2

    for p in argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"Missing: {path}")
            continue
        summarize(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
