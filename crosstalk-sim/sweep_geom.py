"""
Design charts for 'how close can traces be?' on ITA-PCB.

Uses the same 2D field solver (fieldsolve.Grid) to compute the saturated
near-end coupling coefficient Kb = 0.25*(Lm/Ls + Cm/Cself) for a coplanar
aggressor+victim pair, as a function of:
  (a) edge-to-edge spacing s, at the current 3-mil reference height, and
  (b) reference-plane height h, at the fixed 6-mil spacing (why the SAME 6 mil
      is benign now but was 'too close' on a poorly-referenced 2-layer board).

Kb is the saturated NEXT (long parallel run, fast edge) = worst-case, edge-
independent -> a conservative spacing rule.  Double it for a victim flanked by
aggressors on both sides.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fieldsolve import Grid, MIL, EPS0, MU0, spice_caps

W, T, ER = 10.0, 1.4, 4.2

def kb_coplanar(s, h):
    """Backward coupling coeff for two W-wide traces, gap s, over gnd at height h."""
    box_w = 2*W + s + 60
    box_h = h + 40
    dx = 0.25 if h <= 12 else 0.5
    g = Grid(W_mil=box_w, H_mil=box_h, dx_mil=dx)
    g.add_ground_plane_bottom()
    g.set_eps_box(0, box_w, 0, h, ER)
    xc = box_w/2
    g.add_conductor_box(xc - s/2 - W, xc - s/2, h, h+T)        # line 0
    g.add_conductor_box(xc + s/2, xc + s/2 + W, h, h+T)        # line 1
    Cd = g.maxwell_C(g.eps.copy())
    Ca = g.maxwell_C(np.ones_like(g.eps))
    L = MU0*EPS0*np.linalg.inv(Ca)
    Ls, Lm = L[0,0], L[0,1]
    Cg, Cmut = spice_caps(Cd)
    Cself = Cg[0]; Cm12 = Cmut[0,1]
    Kb = 0.25*(Lm/Ls + Cm12/Cself)
    Z0 = np.sqrt(np.sqrt((Ls+Lm)/Cself)*np.sqrt((Ls-Lm)/(Cself+2*Cm12)))
    return Kb, Z0

# (a) spacing sweep at h = 3 (current board)
spaces = [4,5,6,8,10,12,16,20]
kb_s = [(s,)+kb_coplanar(s, 3.0) for s in spaces]
# (b) reference-height sweep at s = 6
heights = [3,6,10,16,24,40,59]
kb_h = [(h,)+kb_coplanar(6.0, h) for h in heights]

print("=== coplanar NEXT vs spacing, h=3mil (current stackup) ===")
for s,kb,z in kb_s:
    print(f"  gap {s:2.0f} mil : NEXT {kb*100:4.1f}% (1 side), {2*kb*100:4.1f}% (2 sides)   Z0={z:.0f}")
print("=== coplanar NEXT vs reference height, s=6mil ===")
for h,kb,z in kb_h:
    print(f"  h {h:2.0f} mil : NEXT {kb*100:4.1f}% (1 side)   Z0={z:.0f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
s_ = [r[0] for r in kb_s]
ax[0].plot(s_, [r[1]*100 for r in kb_s], "o-", label="1 aggressor")
ax[0].plot(s_, [2*r[1]*100 for r in kb_s], "s--", label="2 aggressors (bus)")
ax[0].axvline(6, color="r", ls=":", lw=1); ax[0].text(6.1, ax[0].get_ylim()[1]*0.9, "6 mil", color="r")
ax[0].set_xlabel("edge-to-edge spacing (mil)"); ax[0].set_ylabel("saturated NEXT (% swing)")
ax[0].set_title("Coplanar coupling vs spacing  (h = 3 mil reference)")
ax[0].grid(alpha=0.3); ax[0].legend()
h_ = [r[0] for r in kb_h]
ax[1].plot(h_, [r[1]*100 for r in kb_h], "o-", color="C3")
ax[1].axvline(3, color="g", ls=":", lw=1); ax[1].text(3.3, 2, "current\n3 mil", color="g", fontsize=8)
ax[1].axvline(59, color="0.4", ls=":", lw=1); ax[1].text(45, ax[1].get_ylim()[1]*0.5, "2-layer\n(~59 mil)", fontsize=8)
ax[1].set_xlabel("distance to reference plane h (mil)"); ax[1].set_ylabel("saturated NEXT (% swing)")
ax[1].set_title("Same 6-mil gap: coupling vs reference distance")
ax[1].grid(alpha=0.3)
fig.suptitle("ITA-PCB coplanar spacing design charts")
fig.tight_layout()
p = r"C:\Users\Public\Documents\Altium\NEXUS-connectors\ITA-PCB\crosstalk-sim\spacing_charts.png"
fig.savefig(p, dpi=115); print("saved", p)
