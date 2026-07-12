"""Plots for ITA-PCB crosstalk sweep: summary curves + representative waveforms."""
import numpy as np, os, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_xtalk as rx

HERE = rx.HERE
rows = np.load(os.path.join(HERE, "results.npy"), allow_pickle=True)
# rows: case,length_mil,tr,Z0,next_pct,fext_pct,aggr
def sub(case):
    return [r for r in rows if r[0] == case]
EDGES = rx.EDGES
edge_lbl = ["100 ps", "300 ps", "1 ns", "3 ns"]

# ---- binary .raw parser (LTspice transient) ----
def read_raw(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\xff\xfe" or raw[1:2] == b"\x00":  # UTF-16LE (LTspice default, often no BOM)
        enc = "utf-16-le"; marker = "Binary:\n".encode(enc)
    else:
        enc = "latin-1"; marker = b"Binary:\n"
    hdr_end = raw.find(marker) + len(marker)
    hdr = raw[:hdr_end].decode(enc, errors="ignore")
    nvar = int([l for l in hdr.splitlines() if "No. Variables" in l][0].split(":")[1])
    npts = int([l for l in hdr.splitlines() if "No. Points" in l][0].split(":")[1])
    names = []
    grab = False
    for l in hdr.splitlines():
        if l.startswith("Variables:"): grab = True; continue
        if grab and l.startswith("\t"):
            names.append(l.split("\t")[2])
    data = raw[hdr_end:]
    # LTspice: time = float64, vars = float32 (real transient)
    vals = np.zeros((npts, nvar))
    off = 0
    import struct
    for p in range(npts):
        vals[p, 0] = struct.unpack_from("<d", data, off)[0]; off += 8
        for v in range(1, nvar):
            vals[p, v] = struct.unpack_from("<f", data, off)[0]; off += 4
    return names, vals

def waveform(case, length_mil, tr):
    net, Z0 = rx.netlist(case, length_mil, tr)
    base = f"wf_{case}_{length_mil}_{int(tr*1e12)}"
    npath = os.path.join(HERE, base + ".net")
    open(npath, "w").write(net)
    subprocess.run([rx.LTSPICE, "-b", "-Run", npath], cwd=HERE, timeout=300,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    names, vals = read_raw(os.path.join(HERE, base + ".raw"))
    idx = {n.lower(): i for i, n in enumerate(names)}
    t = vals[:, 0]
    def V(n): return vals[:, idx[f"v({n})"]]
    return t, V(f"a0"), V("b0"), V(f"b{rx.N}")

# ======== Figure 1: summary curves ========
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
for j, case in enumerate(("coplanar", "broadside")):
    S = sub(case)
    lengths = sorted(set(r[1] for r in S))
    for metric, a in ((4, ax[j, 0]), (5, ax[j, 1])):
        for L in lengths:
            ys = [next(r[metric] for r in S if r[1] == L and abs(r[2]-e) < 1e-15)*100
                  for e in EDGES]
            a.plot(range(len(EDGES)), ys, "o-", label=f"{L} mil coupled")
        a.set_xticks(range(len(EDGES))); a.set_xticklabels(edge_lbl)
        a.set_xlabel("aggressor edge (rise time)")
        a.set_ylabel("% of aggressor swing")
        a.grid(True, alpha=0.3); a.legend(fontsize=8)
        kind = "NEXT (near-end)" if metric == 4 else "FEXT (far-end)"
        a.set_title(f"{case.upper()}  —  {kind}")
# annotate the danger line
for a in ax[1]:
    a.axhline(20, color="r", ls="--", lw=1, alpha=0.6)
    a.text(0.02, 21, "~20% caution", color="r", fontsize=8)
fig.suptitle("ITA-PCB single-aggressor crosstalk (matched terminations; "
             "2-sided bus ≈ 2× coplanar NEXT)", fontsize=12)
fig.tight_layout()
f1 = os.path.join(HERE, "xtalk_summary.png")
fig.savefig(f1, dpi=110); print("saved", f1)

# ======== Figure 2: representative waveforms ========
cases = [("coplanar", 3000, 300e-12), ("broadside", 500, 300e-12)]
fig2, ax2 = plt.subplots(1, 2, figsize=(13, 4.6))
for k, (case, L, tr) in enumerate(cases):
    t, a0, b0, bN = waveform(case, L, tr)
    sw = a0.max() if a0.max() else 0.5
    ax2[k].plot(t*1e9, a0/sw*100, "k", lw=1.5, label="aggressor V(a0)")
    ax2[k].plot(t*1e9, b0/sw*100, "C3", label="victim NEXT (near)")
    ax2[k].plot(t*1e9, bN/sw*100, "C0", label="victim FEXT (far)")
    ax2[k].set_xlabel("time (ns)"); ax2[k].set_ylabel("% of aggressor swing")
    ax2[k].set_title(f"{case.upper()}  {L} mil, 300 ps edge")
    ax2[k].grid(True, alpha=0.3); ax2[k].legend(fontsize=8)
    ax2[k].set_xlim(0, min(t.max()*1e9, (6 if case=="coplanar" else 4)))
fig2.suptitle("Representative victim waveforms (300 ps aggressor edge)")
fig2.tight_layout()
f2 = os.path.join(HERE, "xtalk_waveforms.png")
fig2.savefig(f2, dpi=110); print("saved", f2)

# ======== absolute mV table ========
print("\nAbsolute worst-case victim (single aggressor), mV:")
print(f"{'case/len/edge':32s} {'%swing':>7} {'@1.8V':>7} {'@3.3V':>7}")
for r in rows:
    pk = max(r[4], r[5]) * 100
    print(f"{r[0]:9s} {r[1]:5d}mil {int(r[2]*1e12):5d}ps  {pk:6.1f}% "
          f"{pk/100*1800:6.0f}  {pk/100*3300:6.0f}")
