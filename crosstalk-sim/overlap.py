"""
Find broadside signal-over-signal overlap on ITA-PCB.

Two outer/inner layer pairs are only 3 mil apart:
    TopLayer   over MidLayer1
    BottomLayer over MidLayer2
Where an outer-layer signal runs *parallel over* an inner-layer signal (of a
different, non-plane net) within ~1 trace width laterally, we get the severe
broadside coupling characterised by the LTspice sweep (18-47% of swing).

This script sums, per net-pair, the parallel-overlap length, and buckets it by
the connector/COB rooms so we can see whether it clusters at the VHDCI ends.

Overlap on VSS/VDD (the intended reference) is GOOD and excluded.
"""
import json, math, collections, os

HERE = r"C:\Users\Public\Documents\Altium\NEXUS-connectors\ITA-PCB\crosstalk-sim"
SRC = (r"C:\Users\danlo\.claude\projects\C--Users-Public-Documents-Altium-"
       r"NEXUS-connectors\ed538ba0-c05e-4ea9-8fff-656a046b6734\tool-results\\"
       r"mcp-eda-agent-obj_query-1783739979381.txt")

QUIET = {"VSS", "VDD", "V5V", "GND", ""}          # planes / references: overlap is fine
ANGLE_PARALLEL = math.radians(12)                 # within this = "parallel"
PERP_STRONG = 8.0                                 # mil, centreline offset: direct stack
PERP_MOD = 16.0                                   # mil, moderate broadside
MIN_REPORT = 50.0                                 # mil, ignore tiny overlaps

ROOMS = {   # name: (x0,y0,x1,y1) mil
    "cob-bottom": (5258.0, 3660.0, 8493.0, 3870.0),
    "cob-left":   (4865.0, 3977.0, 5075.0, 7212.0),
    "cob-right":  (8570.0, 3968.0, 8780.0, 7203.0),
    "cob-top":    (5255.0, 7380.0, 8490.0, 7590.0),
    "vhcdi-1": (3275.0, 4915.0, 3650.0, 6605.0),
    "vhcdi-2": (4335.0, 8210.0, 6025.0, 8585.0),
    "vhcdi-3": (7820.9, 8225.0, 9510.9, 8600.0),
    "vhcdi-4": (10140.0, 4941.0, 10515.0, 6631.0),
    "vhcdi-5": (7720.9, 2605.0, 9410.9, 2980.0),
    "vhcdi-6": (4229.1, 2635.0, 5919.1, 3010.0),
}

def load():
    d = json.load(open(SRC, encoding="utf-8"))
    segs = collections.defaultdict(list)   # layer -> list of (x1,y1,x2,y2,net)
    for o in d["objects"]:
        L = o["Layer"]
        if L not in ("TopLayer", "MidLayer1", "MidLayer2", "BottomLayer"):
            continue
        x1, y1, x2, y2 = (float(o[k]) for k in ("X1", "Y1", "X2", "Y2"))
        if abs(x1-x2) < 1e-6 and abs(y1-y2) < 1e-6:
            continue
        segs[L].append((x1, y1, x2, y2, o["Net"]))
    return segs

def which_room(x, y):
    for name, (x0, y0, x1, y1) in ROOMS.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return None

def parallel_overlap(a, b):
    """Return (overlap_len, perp_offset, mid_x, mid_y) or None if not parallel/overlapping."""
    ax1, ay1, ax2, ay2, _ = a
    bx1, by1, bx2, by2, _ = b
    ux, uy = ax2-ax1, ay2-ay1
    la = math.hypot(ux, uy)
    ux, uy = ux/la, uy/la
    vx, vy = bx2-bx1, by2-by1
    lb = math.hypot(vx, vy)
    vx, vy = vx/lb, vy/lb
    cross = abs(ux*vy - uy*vx)                     # sin(angle)
    if cross > math.sin(ANGLE_PARALLEL):
        return None
    # perp offset of b's midpoint from a's infinite line
    mbx, mby = (bx1+bx2)/2, (by1+by2)/2
    perp = abs((mbx-ax1)*uy - (mby-ay1)*ux)
    if perp > PERP_MOD:
        return None
    # project b endpoints onto a's axis (param from a's start, in mil)
    t_b1 = (bx1-ax1)*ux + (by1-ay1)*uy
    t_b2 = (bx2-ax1)*ux + (by2-ay1)*uy
    lo, hi = sorted((t_b1, t_b2))
    ov = max(0.0, min(la, hi) - max(0.0, lo))
    if ov < 1.0:
        return None
    # midpoint of the overlapping region in xy
    tc = (max(0.0, lo) + min(la, hi)) / 2
    mx, my = ax1+ux*tc, ay1+uy*tc
    return ov, perp, mx, my

HOTSPOTS = []   # (mx, my, ov, label) for the location map

def analyze(outer, inner, segs, label):
    O, I = segs[outer], segs[inner]
    pair_strong = collections.Counter()   # (outer_net, inner_net) -> mil
    pair_mod = collections.Counter()
    room_strong = collections.Counter()
    for a in O:
        an = a[4]
        for b in I:
            bn = b[4]
            if an == bn:                   # same net = own routing, not crosstalk
                continue
            if an in QUIET or bn in QUIET:  # over a plane = intended reference, fine
                continue
            r = parallel_overlap(a, b)
            if not r:
                continue
            ov, perp, mx, my = r
            key = (an, bn)
            if perp <= PERP_STRONG:
                pair_strong[key] += ov
                room_strong[which_room(mx, my)] += ov
                if ov >= 20:
                    HOTSPOTS.append((mx, my, ov, label))
            else:
                pair_mod[key] += ov
    tot_s = sum(pair_strong.values())
    tot_m = sum(pair_mod.values())
    print(f"\n===== {label}  ({outer} over {inner}) =====")
    print(f"strong-stack (<= {PERP_STRONG:.0f} mil offset): "
          f"{tot_s:,.0f} mil total across {sum(1 for v in pair_strong.values() if v>=MIN_REPORT)} net-pairs")
    print(f"moderate ({PERP_STRONG:.0f}-{PERP_MOD:.0f} mil offset): {tot_m:,.0f} mil total")
    print(f"top strong-stack net-pairs (outer <- inner, overlap mil):")
    for (an, bn), v in sorted(pair_strong.items(), key=lambda kv: -kv[1])[:15]:
        if v < MIN_REPORT:
            break
        print(f"   {v:7.0f}  {an:28s} <- {bn}")
    print("strong-stack overlap by room:")
    for r, v in sorted(room_strong.items(), key=lambda kv: -kv[1]):
        print(f"   {v:8.0f} mil  {r}")
    return pair_strong, room_strong

def plot_map(segs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(11, 9))
    # faint routing context (all signal segs)
    for L, col in (("TopLayer", "#cfe6ff"), ("MidLayer1", "#ffe0cf"),
                   ("BottomLayer", "#d9f2d9"), ("MidLayer2", "#f2d9f2")):
        for x1, y1, x2, y2, _ in segs[L]:
            ax.plot([x1, x2], [y1, y2], color=col, lw=0.4, zorder=1)
    # rooms
    for name, (x0, y0, x1, y1) in ROOMS.items():
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                     edgecolor="0.4", lw=1.0, ls="--", zorder=2))
        ax.text(x0, y1+40, name, fontsize=7, color="0.3")
    # hotspots
    for mx, my, ov, label in HOTSPOTS:
        c = "#c00000" if "BOTTOM" in label else "#0040c0"
        ax.scatter([mx], [my], s=ov*0.6+8, c=c, alpha=0.7,
                   edgecolors="k", linewidths=0.3, zorder=3)
    ax.scatter([], [], c="#c00000", label="Bottom-over-Mid2 stack")
    ax.scatter([], [], c="#0040c0", label="Top-over-Mid1 stack")
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlabel("x (mil)"); ax.set_ylabel("y (mil)")
    ax.set_title("ITA broadside signal-over-signal stacks (marker size ~ overlap length)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p = os.path.join(HERE, "overlap_map.png")
    fig.savefig(p, dpi=115); print("saved", p)

if __name__ == "__main__":
    segs = load()
    print("segment counts:", {k: len(v) for k, v in segs.items()})
    analyze("TopLayer", "MidLayer1", segs, "TOP / MID1")
    analyze("BottomLayer", "MidLayer2", segs, "BOTTOM / MID2")
    plot_map(segs)
