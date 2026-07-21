"""
Generate + run LTspice coupled-line crosstalk netlists for ITA-PCB.

Two-conductor (aggressor + victim) RLGC ladder built from the per-unit-length
L,C matrices extracted by fieldsolve.py (rlgc.npz).  Sweeps aggressor edge rate
and coupled length; measures near-end (NEXT) and far-end (FEXT) victim voltage
via LTspice .meas, parsed from the .log.

Terminations: all four line ends matched to the in-situ single-line Z0 so the
reported numbers are the *intrinsic* coupling (no reflection artifacts).  An
unterminated (high-Z) receiver would see up to ~2x these voltages.
Aggressor swing normalised to 1 V -> results are % of swing.
"""
import numpy as np, subprocess, re, os, itertools

HERE = r"C:\Users\Public\Documents\Altium\NEXUS-connectors\ITA-PCB\crosstalk-sim"
LTSPICE = r"C:\Users\danlo\AppData\Local\Programs\ADI\LTspice\LTspice.exe"
MIL = 25.4e-6
d = np.load(os.path.join(HERE, "rlgc.npz"))

def params(L, C):
    Ls, Lm = L[0, 0], L[0, 1]
    Cg = C[0, 0] + C[0, 1]          # Maxwell row-sum -> cap to ground
    Cm = -C[0, 1]                   # mutual cap
    Z0 = np.sqrt((Ls) / (Cg))       # approx single-line (neighbour grounded)
    # better in-situ Z0 = sqrt(Zeven*Zodd)
    Le, Lo = Ls + Lm, Ls - Lm
    Ce, Co = Cg, Cg + 2 * Cm
    Z0 = np.sqrt(np.sqrt(Le / Ce) * np.sqrt(Lo / Co))
    return Ls, Lm, Cg, Cm, Z0

CASES = {
    "coplanar": params(d["cop_L"], d["cop_C"]),
    "broadside": params(d["bro_L"], d["bro_C"]),
}
# 4 ns = 10% of the 25 MHz IO period (chip IO edge rate, confirmed 2026-07-21).
# 100 ps..3 ns retained only as the (now-irrelevant) fast-edge reference.
EDGES = [100e-12, 300e-12, 1e-9, 3e-9, 4e-9]
LENGTHS_MIL = {
    "coplanar": [500, 1500, 3000],
    "broadside": [250, 500, 1000, 1500, 2000],
}
N = 200   # ladder sections

def netlist(case, length_mil, tr):
    Ls, Lm, Cg, Cm, Z0 = CASES[case]
    ell = length_mil * MIL
    dd = ell / N
    kL = Lm / Ls
    Vh = 1.0
    tpd = np.sqrt(Ls * (Cg + 2 * Cm))            # rough odd-mode delay per m (upper bnd)
    tflight = ell * np.sqrt(Ls * Cg)
    tstop = max(6 * tflight, 8 * tr) + 2e-9
    lines = [f"* {case} len={length_mil}mil tr={tr*1e12:.0f}ps Z0={Z0:.1f}"]
    # source: trapezoid on aggressor near end through Rs=Z0
    lines.append(f"Vsrc s 0 PULSE(0 {Vh} 0 {tr:.4e} {tr:.4e} {tstop*2:.4e} {tstop*4:.4e})")
    lines.append(f"Rs s a0 {Z0:.4f}")
    lines.append(f"RaF a{N} 0 {Z0:.4f}")         # aggressor far term
    lines.append(f"RvN b0 0 {Z0:.4f}")           # victim near end (NEXT)
    lines.append(f"RvF b{N} 0 {Z0:.4f}")         # victim far end (FEXT)
    for i in range(N):
        lines.append(f"La{i} a{i} a{i+1} {Ls*dd:.6e}")
        lines.append(f"Lb{i} b{i} b{i+1} {Ls*dd:.6e}")
        lines.append(f"K{i} La{i} Lb{i} {kL:.6f}")
    for i in range(N + 1):
        w = dd * (0.5 if i in (0, N) else 1.0)
        lines.append(f"Cag{i} a{i} 0 {Cg*w:.6e}")
        lines.append(f"Cbg{i} b{i} 0 {Cg*w:.6e}")
        lines.append(f"Cm{i} a{i} b{i} {Cm*w:.6e}")
    lines.append(f".tran 0 {tstop:.4e} 0 {tr/20:.4e}")
    lines.append(".meas TRAN next_max MAX V(b0)")
    lines.append(".meas TRAN next_min MIN V(b0)")
    lines.append(f".meas TRAN fext_max MAX V(b{N})")
    lines.append(f".meas TRAN fext_min MIN V(b{N})")
    lines.append(".meas TRAN aggr MAX V(a0)")
    lines.append(".backanno")
    lines.append(".end")
    return "\n".join(lines), Z0

def parse_log(path):
    txt = open(path, encoding="latin-1").read()
    out = {}
    for key in ("next_max", "next_min", "fext_max", "fext_min", "aggr"):
        m = re.search(rf"{key}:.*?=([-\d.eE+]+)", txt)
        out[key] = float(m.group(1)) if m else float("nan")
    return out

def run():
    rows = []
    for case in CASES:
        for length_mil, tr in itertools.product(LENGTHS_MIL[case], EDGES):
            net, Z0 = netlist(case, length_mil, tr)
            base = f"{case}_{length_mil}_{int(tr*1e12)}ps"
            npath = os.path.join(HERE, base + ".net")
            open(npath, "w").write(net)
            subprocess.run([LTSPICE, "-b", "-Run", npath],
                           cwd=HERE, timeout=300,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log = os.path.join(HERE, base + ".log")
            r = parse_log(log)
            aggr = r["aggr"] if r["aggr"] else 0.5      # actual line swing
            next_pk = max(abs(r["next_max"]), abs(r["next_min"])) / aggr
            fext_pk = max(abs(r["fext_max"]), abs(r["fext_min"])) / aggr
            rows.append((case, length_mil, tr, Z0, next_pk, fext_pk, aggr))
            print(f"{case:10s} L={length_mil:5d}mil tr={tr*1e12:6.0f}ps  "
                  f"NEXT={next_pk*100:6.1f}%  FEXT={fext_pk*100:6.1f}%  of aggr swing")
    np.save(os.path.join(HERE, "results.npy"), np.array(rows, dtype=object))
    return rows

if __name__ == "__main__":
    for c in CASES:
        Ls, Lm, Cg, Cm, Z0 = CASES[c]
        print(f"{c}: Z0={Z0:.1f} ohm  Ls={Ls*1e9:.0f}nH/m Lm={Lm*1e9:.0f} "
              f"Cg={Cg*1e12:.0f}pF/m Cm={Cm*1e12:.0f}  kL={Lm/Ls:.3f}")
    print()
    run()
