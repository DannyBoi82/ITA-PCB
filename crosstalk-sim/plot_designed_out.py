"""
"Broadside risk designed out" — one figure for sign-off.

Worst-case coupled noise (max of |NEXT|,|FEXT|, % of aggressor swing) vs edge
rate, comparing the OLD broadside signal-over-signal overlap (the reason for the
relayout, now eliminated on all three boards) against the ONLY mechanism left in
the redone Signal/plane/plane/Signal stacks: coplanar same-layer coupling.

All numbers are measured from the LTspice RLGC sweeps in this repo:
  - broadside_* / coplanar_* : ITA-PCB/crosstalk-sim/run_xtalk.py
  - HE 7-mil Top             : HE-PCB/xtalk_sim/he_top7_4ns.py
"""
import numpy as np
import matplotlib.pyplot as plt

edges_ps = np.array([100, 300, 1000, 3000, 4000])   # x axis

# --- OLD, now-eliminated broadside (Top-over-inner-signal, 3 mil, no plane) ---
bro_1000 = [45.9, 47.5, 21.7, 7.4, 5.6]   # 1000 mil (1 in) overlap, worst=NEXT
bro_2000 = [43.9, 46.7, 36.9, 14.7, 11.1]  # 2000 mil (2 in) overlap

# --- NEW coplanar, 3000 mil (3 in) fully-adjacent run, worst of NEXT/FEXT ---
he_top7  = [27.8, 10.5, 9.5, 3.2, 2.4]     # HE Top, 7 mil to plane (loosest)
tight    = [18.8, 6.9, 3.9, 1.3, 1.0]      # ITA/CNFET + HE Bottom, ~3 mil to plane

fig, ax = plt.subplots(figsize=(9, 6))

# stress-test region: everything faster than the real 4 ns edge (right of it)
ax.axvspan(90, 3600, color="0.93", zorder=0)
ax.axhline(5, color="green", ls=":", lw=1.4, label="5% design budget")

ax.plot(edges_ps, bro_2000, "s--", color="#b30000", lw=2, ms=7,
        label="OLD broadside overlap, 2 in  (ELIMINATED)")
ax.plot(edges_ps, bro_1000, "^--", color="#e34a33", lw=2, ms=7,
        label="OLD broadside overlap, 1 in  (ELIMINATED)")
ax.plot(edges_ps, he_top7, "o-", color="#f39c12", lw=2.2, ms=7,
        label="NEW HE Top (7 mil), 3 in run")
ax.plot(edges_ps, tight, "o-", color="#2166ac", lw=2.2, ms=7,
        label="NEW ITA/CNFET + HE Bottom (3 mil), 3 in run")

# real operating point
ax.axvline(4000, color="k", lw=1.2)
ax.annotate("real IO edge\n25 MHz, 4 ns", xy=(4000, 40), xytext=(1500, 41),
            fontsize=10, ha="center",
            arrowprops=dict(arrowstyle="->", lw=1.1))
ax.annotate("faster than reality\n(stress test)", xy=(200, 33), fontsize=9,
            color="0.35", style="italic")

ax.set_xscale("log")
ax.set_xticks(edges_ps)
ax.set_xticklabels([f"{e/1000:g} ns" if e >= 1000 else f"{e} ps" for e in edges_ps])
ax.invert_xaxis()   # fast edges left, real slow edge right
ax.set_xlabel("aggressor rise/fall time  (faster → right)")
ax.set_ylabel("worst-case coupled noise  (% of 1.8 V swing)")
ax.set_ylim(0, 50)
ax.set_title("NEXUS breakout boards: broadside crosstalk risk designed out\n"
             "coplanar coupling is bounded at every edge; 4 ns operating point is trivial",
             fontsize=11)
ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
ax.grid(True, which="both", alpha=0.3)

out = r"C:\Users\Public\Documents\Altium\NEXUS-connectors\ITA-PCB\crosstalk-sim\crosstalk_designed_out.png"
fig.tight_layout()
fig.savefig(out, dpi=150)
print("wrote", out)
