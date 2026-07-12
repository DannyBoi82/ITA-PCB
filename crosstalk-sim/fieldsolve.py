"""
2D electrostatic field solver for ITA-PCB crosstalk analysis.

Extracts per-unit-length capacitance (C) and inductance (L) matrices for a
multiconductor transmission-line cross-section using a finite-difference
solution of div(eps*grad(phi)) = 0 on a uniform grid, then the nodal-charge
method for the Maxwell capacitance matrix.

  L = mu0 * eps0 * inv(C_air)      (TEM approximation)

Geometry is described in mils; solved in SI.  Ground = potential-0 conductors
plus the outer box (Dirichlet 0).

Validation target: a single 10-mil-wide, 1.4-mil-thick trace 3 mil above a
ground plane in eps_r 4.2 should give Z0 ~= 24 ohm (matches Altium/IPC-2141).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MIL = 25.4e-6            # m
EPS0 = 8.8541878128e-12  # F/m
MU0 = 4e-7 * np.pi       # H/m
C_LIGHT = 1.0 / np.sqrt(EPS0 * MU0)


class Grid:
    """Uniform rectangular FDM grid over a box [0,W]x[0,H] in mils."""
    def __init__(self, W_mil, H_mil, dx_mil=0.25):
        self.dx = dx_mil * MIL
        self.nx = int(round(W_mil / dx_mil)) + 1
        self.ny = int(round(H_mil / dx_mil)) + 1
        self.dx_mil = dx_mil
        self.eps = np.ones((self.ny, self.nx))          # relative permittivity per node
        self.fixed = np.full((self.ny, self.nx), -1, np.int32)  # conductor id, -1 = free
        # conductor 0 is reserved for GROUND (the reference)
        self.n_cond = 1

    def _idx(self, xa, xb, ya, yb):
        """node index ranges (inclusive) for a mil box."""
        i0 = int(round(xa / self.dx_mil)); i1 = int(round(xb / self.dx_mil))
        j0 = int(round(ya / self.dx_mil)); j1 = int(round(yb / self.dx_mil))
        return max(i0, 0), min(i1, self.nx - 1), max(j0, 0), min(j1, self.ny - 1)

    def set_eps_box(self, xa, xb, ya, yb, er):
        i0, i1, j0, j1 = self._idx(xa, xb, ya, yb)
        self.eps[j0:j1 + 1, i0:i1 + 1] = er

    def add_ground_box(self, xa, xb, ya, yb):
        i0, i1, j0, j1 = self._idx(xa, xb, ya, yb)
        self.fixed[j0:j1 + 1, i0:i1 + 1] = 0

    def add_ground_plane_bottom(self):
        self.fixed[0, :] = 0

    def add_conductor_box(self, xa, xb, ya, yb):
        cid = self.n_cond
        self.n_cond += 1
        i0, i1, j0, j1 = self._idx(xa, xb, ya, yb)
        self.fixed[j0:j1 + 1, i0:i1 + 1] = cid
        return cid

    def build_stiffness(self, eps_field):
        """Assemble the eps-weighted 5-point Laplacian K (N x N, sparse)."""
        ny, nx = self.ny, self.nx
        N = nx * ny
        def nid(j, i): return j * nx + i
        rows, cols, vals = [], [], []
        diag = np.zeros(N)
        # edge permittivity = arithmetic mean of the two adjacent node eps
        for j in range(ny):
            for i in range(nx):
                p = nid(j, i)
                for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    jj, ii = j + dj, i + di
                    if 0 <= jj < ny and 0 <= ii < nx:
                        ee = 0.5 * (eps_field[j, i] + eps_field[jj, ii])
                        rows.append(p); cols.append(nid(jj, ii)); vals.append(-ee)
                        diag[p] += ee
        rows.extend(range(N)); cols.extend(range(N)); vals.extend(diag)
        return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    def maxwell_C(self, eps_field):
        """Return Maxwell capacitance matrix (F/m) for signal conductors 1..n_cond-1."""
        ny, nx = self.ny, self.nx
        N = nx * ny
        K = self.build_stiffness(eps_field)
        fixed = self.fixed.ravel()
        free = np.where(fixed < 0)[0]
        Kff = K[free][:, free].tocsc()
        lu = spla.splu(Kff)
        nsig = self.n_cond - 1
        Cm = np.zeros((nsig, nsig))
        Kdense_rows = {}
        for k in range(1, self.n_cond):        # excite conductor k -> 1V
            phi = np.zeros(N)
            drive = np.where(fixed == k)[0]
            phi[drive] = 1.0
            # RHS for free nodes: -K_free,fixed * phi_fixed
            rhs = -(K[free][:, :] @ phi)[:]     # phi already has fixed values set
            phi[free] = lu.solve(rhs)
            # charge on each signal conductor j = sum over its nodes of (K phi)
            Kphi = K @ phi
            for j in range(1, self.n_cond):
                nodes = np.where(fixed == j)[0]
                Cm[j - 1, k - 1] = Kphi[nodes].sum() * EPS0
        return Cm


def spice_caps(Cm):
    """Convert Maxwell matrix -> (Cground[i], Cmutual[i,j]) in F/m."""
    n = Cm.shape[0]
    Cg = np.array([Cm[i].sum() for i in range(n)])   # row sum -> cap to ground
    Cmut = -Cm.copy()
    np.fill_diagonal(Cmut, 0.0)
    return Cg, Cmut


def analyze(name, grid_builder):
    g = grid_builder()
    eps_diel = g.eps.copy()
    eps_air = np.ones_like(g.eps)
    Cd = g.maxwell_C(eps_diel)     # with dielectric
    Ca = g.maxwell_C(eps_air)      # vacuum
    L = MU0 * EPS0 * np.linalg.inv(Ca)   # H/m
    C = Cd                                 # F/m (Maxwell form)
    n = C.shape[0]
    # per-line self/mutual (Maxwell diagonal is self-to-all; use L & C directly)
    Ls = np.diag(L).copy()
    Cs = np.diag(C).copy()
    Z0 = np.sqrt(Ls / Cs)                  # crude self Z (ignores coupling)
    # proper single-line Z0 uses L,C with victim grounded => modal below
    print(f"\n=== {name} : {n} signal conductor(s) ===")
    np.set_printoptions(precision=4, suppress=False)
    print("L (nH/m):\n", L * 1e9)
    print("C (pF/m):\n", C * 1e12)
    if n == 1:
        z0 = np.sqrt(L[0, 0] / C[0, 0])
        vp = 1 / np.sqrt(L[0, 0] * C[0, 0])
        print(f"Single-line Z0 = {z0:.2f} ohm,  vp = {vp/1e8:.3f}e8 m/s,  "
              f"eps_eff = {(C_LIGHT/vp)**2:.2f},  tpd = {1e12/vp*0.0254:.1f} ps/inch")
    if n >= 2:
        Ls_, Lm = L[0, 0], L[0, 1]
        Cg, Cmut = spice_caps(C)
        Cself_tot = C[0, 0]                     # total self (to gnd + mutual)
        # even/odd modal analysis for the symmetric 2-line pair (lines 0,1)
        Cself = Cg[0]                           # cap to ground
        Cm12 = Cmut[0, 1]
        Le, Lo = Ls_ + Lm, Ls_ - Lm
        Ce, Co = Cself, Cself + 2 * Cm12
        Ze, Zo = np.sqrt(Le / Ce), np.sqrt(Lo / Co)
        Z0e = np.sqrt(Ze * Zo)                  # single-line Z0 with neighbor present
        # weak-coupling backward/forward coefficients
        Kb = 0.25 * (Lm / Ls_ + Cm12 / Cself)
        tpd = np.sqrt(Ls_ * Cself)              # s/m (approx per-line)
        Kf_coeff = 0.5 * (Cm12 / Cself - Lm / Ls_)   # * (len/tr) * V, sign per convention
        print(f"line0 Cground={Cg[0]*1e12:.2f} pF/m, Cmutual={Cm12*1e12:.2f} pF/m")
        print(f"Ls={Ls_*1e9:.1f} nH/m  Lm={Lm*1e9:.1f} nH/m  (Lm/Ls={Lm/Ls_:.3f})")
        print(f"Cm/Cself = {Cm12/Cself:.3f}")
        print(f"Zeven={Ze:.1f}  Zodd={Zo:.1f}  Z0(single,in-situ)={Z0e:.1f} ohm")
        print(f"Backward (NEXT) coeff Kb = {Kb:.4f}  ({Kb*100:.1f}% of swing, saturated)")
        print(f"Forward (FEXT) coeff (Cm/C - Lm/L)/2 = {Kf_coeff:.4f}  "
              f"[FEXT = -coeff * (Tflight/tr) * Vswing]")
        return dict(name=name, L=L, C=C, Cg=Cg, Cmut=Cmut, Ls=Ls_, Lm=Lm,
                    Cself=Cself, Cm12=Cm12, Ze=Ze, Zo=Zo, Z0=Z0e, Kb=Kb,
                    Kf_coeff=Kf_coeff, tpd=tpd)
    return dict(name=name, L=L, C=C)


# ---------------- cross-section builders ----------------
W, T, H_top = 10.0, 1.4, 3.0     # trace width, copper thickness, top prepreg (mil)
GAP = 6.0                        # min edge-edge in the fan-out rooms
ER_PP, ER_CORE = 4.2, 4.4
CORE = 51.0

def cs_single():
    """One microstrip trace over ground plane, h=3 mil, for validation."""
    g = Grid(W_mil=60, H_mil=40, dx_mil=0.25)
    g.add_ground_plane_bottom()
    g.set_eps_box(0, 60, 0, H_top, ER_PP)          # prepreg
    xc = 30
    g.add_conductor_box(xc - W/2, xc + W/2, H_top, H_top + T)
    return g

def cs_coplanar():
    """Victim-Aggressor-Victim, 3 microstrips, 6-mil gaps, over ground at 3 mil.
       Returns aggressor(centre)=line index? We order conductors as added:
       0=left victim, 1=aggressor(centre), 2=right victim.  For 2-line coupling
       coeffs we use centre<->neighbour; but analyze() uses lines 0,1 -> so we
       instead add centre first then a neighbour so [0]=aggr,[1]=victim."""
    g = Grid(W_mil=80, H_mil=40, dx_mil=0.25)
    g.add_ground_plane_bottom()
    g.set_eps_box(0, 80, 0, H_top, ER_PP)
    pitch = W + GAP
    xc = 40
    g.add_conductor_box(xc - W/2, xc + W/2, H_top, H_top + T)             # 0 aggressor (centre)
    g.add_conductor_box(xc + GAP + W/2, xc + GAP + 3*W/2, H_top, H_top+T) # 1 right victim
    g.add_conductor_box(xc - GAP - 3*W/2, xc - GAP - W/2, H_top, H_top+T) # 2 left victim
    return g

def cs_broadside():
    """Top victim directly above a Mid1 aggressor (3 mil), VSS fill flanking the
       aggressor on the Mid1 layer at 6-mil gap, Mid2 ground plane 51 mil below.
       Conductor order: 0 = Mid1 aggressor (lower), 1 = Top victim (upper)."""
    Hbox = CORE + H_top + 30
    g = Grid(W_mil=80, H_mil=Hbox, dx_mil=0.5)   # coarser: taller domain
    g.add_ground_plane_bottom()                  # Mid2 / VDD plane at y=0
    y_mid1 = CORE
    y_top = CORE + H_top
    g.set_eps_box(0, 80, 0, y_mid1, ER_CORE)     # 51-mil core below Mid1
    g.set_eps_box(0, 80, y_mid1, y_top, ER_PP)   # 3-mil prepreg Mid1->Top
    xc = 40
    # lower aggressor on Mid1
    g.add_conductor_box(xc - W/2, xc + W/2, y_mid1, y_mid1 + T)           # 0 aggressor
    # upper victim on Top, directly above
    g.add_conductor_box(xc - W/2, xc + W/2, y_top, y_top + T)             # 1 victim
    # VSS fill flanking the aggressor on Mid1 (ground)
    g.add_ground_box(xc + GAP + W/2, 80, y_mid1, y_mid1 + T)
    g.add_ground_box(0, xc - GAP - W/2, y_mid1, y_mid1 + T)
    return g


if __name__ == "__main__":
    analyze("VALIDATION single microstrip (target Z0~24)", cs_single)
    r_cop = analyze("COPLANAR 6-mil fanout (Top/Bottom, gnd 3 mil below)", cs_coplanar)
    r_bro = analyze("BROADSIDE Top-over-Mid1 (3 mil), VSS flank + Mid2 51 mil", cs_broadside)
    np.savez(r"C:\Users\Public\Documents\Altium\NEXUS-connectors\ITA-PCB\crosstalk-sim\rlgc.npz",
             cop_L=r_cop["L"], cop_C=r_cop["C"],
             bro_L=r_bro["L"], bro_C=r_bro["C"])
    print("\nsaved rlgc.npz")
