"""
2D plane-strain finite-element model of the bent sensor laminate (scikit-fem).

Physics (identical to the COMSOL design in comsol/build_model.py):
  * solid layers (PI, PET, PMMA, PDMS, rigid island): plane-strain linear elasticity,
    biquadratic 9-node quadrilaterals (ElementQuad2) on a graded tensor-product mesh
  * nanometre films: Au 50 nm and the conformal Al2O3 5 nm cap run along the whole strip as plane-strain
    membranes (stiffness E t / (1 - nu^2)) bonded to the substrate surface; MoS2 (0.65 nm) exists only
    in the channel
  * MoS2 channel ("flake", length L_ch): own axial displacement u_m (1D quadratic elements) with membrane
    stiffness, attached to the substrate surface at MoS2 mid-plane height, u_s = u_x - h dv/dx, through a
    tangential interface spring k (thin elastic layer). "tied" = exact bonding via a Lagrange multiplier.
  * pure bending: both end faces rigid, rotations +-kappa L/2, u_y = 0; axial translation of the right end
    is a free global unknown, so the axial force is zero and the neutral plane is an outcome.
Units inside the solver: um, uN, MPa (so E in MPa, membrane stiffness in uN/um = N/m, k in uN/um^3).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from skfem import MeshQuad, ElementVector, ElementQuad2, Basis, FacetBasis, BilinearForm, asm
from skfem.helpers import sym_grad, trace, ddot

L = 10000.0         # strip length (um)
LF = 10.0           # channel (flake) length (um)
W_ISL = 3000.0      # rigid island width (um)


# ---------------------------------------------------------------- mesh
def x_breaks(island):
    a = [0.0, LF / 2 - 2, LF / 2, LF / 2 + 2, LF / 2 + 20, LF / 2 + 200, 1000.0]
    if island:
        a += [W_ISL / 2 - 100, W_ISL / 2, W_ISL / 2 + 100]
    a += [L / 2]
    pos = sorted(set(a))
    return sorted(set([-x for x in pos] + pos))


def h_target(xm, m):
    ax = abs(xm)
    if abs(ax - LF / 2) <= 2:
        h = 0.2
    elif ax < LF / 2 + 20:
        h = 1.0
    elif ax < LF / 2 + 200:
        h = 10.0
    elif abs(ax - W_ISL / 2) <= 100:
        h = 5.0
    elif ax < 1000:
        h = 50.0
    else:
        h = 150.0
    return h / m


def nodes_1d(breaks, sizes):
    out = [breaks[0]]
    for a, b, n in zip(breaks[:-1], breaks[1:], sizes):
        out.extend(np.linspace(a, b, n + 1)[1:])
    return np.array(out)


# ---------------------------------------------------------------- model
class Laminate:
    """
    rows: list of (name, y0, y1, material) bottom -> top; material = (E [MPa], nu) or "island"
    dev_y: height of the device line (top of the device substrate)
    films: dict(Et_all, Et_flake, h_m): stiffness of the continuous bonded films (Au + conformal Al2O3 cap),
           of the MoS2 channel flake (uN/um), and MoS2 mid-plane height above the substrate (um)
    island: None or dict(row, E, nu, E_out, nu_out)
    """

    def __init__(self, rows, dev_y, films, level, island=None, sub_rows=("sub",)):
        self.rows, self.dev_y, self.films, self.island = rows, dev_y, films, island
        m = 2 ** (level - 1)
        xb = x_breaks(island is not None)
        nx = [max(1, int(round((b - a) / h_target(0.5 * (a + b), m)))) for a, b in zip(xb[:-1], xb[1:])]
        yb = [rows[0][1]] + [r[2] for r in rows]
        ny = [2 * m] * len(rows)
        self.x = nodes_1d(xb, nx)
        self.y = nodes_1d(yb, ny)
        self.mesh = MeshQuad.init_tensor(self.x, self.y)
        self.basis = Basis(self.mesh, ElementVector(ElementQuad2()), intorder=4)
        self.n_elem = self.mesh.t.shape[1]
        self.ndof2 = self.basis.N
        self.sub = [r for r in rows if r[0] == "sub"][0]
        self._assemble_solid()
        self._assemble_films()
        self._boundary()

    # ---- solid layers
    def _materials(self):
        c = self.mesh.p[:, self.mesh.t].mean(axis=1)
        E = np.zeros(self.n_elem)
        nu = np.zeros(self.n_elem)
        for name, y0, y1, mat in self.rows:
            sel = (c[1] > y0) & (c[1] < y1)
            if mat == "island":
                isl = sel & (np.abs(c[0]) < W_ISL / 2)
                out = sel & ~isl
                E[isl], nu[isl] = self.island["E"], self.island["nu"]
                E[out], nu[out] = self.island["E_out"], self.island["nu_out"]
            else:
                E[sel], nu[sel] = mat
        assert np.all(E > 0), "element without material"
        self.E_el, self.nu_el = E, nu
        return E, nu

    def _assemble_solid(self):
        E, nu = self._materials()
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))      # plane strain
        mu = E / (2 * (1 + nu))
        nq = self.basis.X.shape[1]
        LAM = np.repeat(lam[:, None], nq, axis=1)
        MU = np.repeat(mu[:, None], nq, axis=1)

        @BilinearForm
        def elasticity(u, v, w):
            eu, ev = sym_grad(u), sym_grad(v)
            return 2 * w.mu * ddot(eu, ev) + w.lam * trace(eu) * trace(ev)

        self.K_solid = asm(elasticity, self.basis, lam=LAM, mu=MU)

    # ---- films on the device line and the flake
    def _line_facets(self, y, xmin=-np.inf, xmax=np.inf):
        mp = self.mesh.p[:, self.mesh.facets].mean(axis=1)
        dx = np.abs(np.diff(self.mesh.p[0, self.mesh.facets], axis=0))[0]
        return np.nonzero(np.isclose(mp[1], y, atol=1e-9) & (dx > 0) & (mp[0] > xmin) & (mp[0] < xmax))[0]

    def _assemble_films(self):
        f = self.films
        fac_all = self._line_facets(self.dev_y)

        @BilinearForm
        def membrane(u, v, w):
            return w.Et * u.grad[0][0] * v.grad[0][0]

        fb_all = FacetBasis(self.mesh, self.basis.elem, facets=fac_all, intorder=6)
        self.K_films = asm(membrane, fb_all, Et=f["Et_all"])

        # flake: quadrature points on the flake facets, 2D trace operator and 1D P2 flake basis
        fac_fl = self._line_facets(self.dev_y, xmin=-LF / 2, xmax=LF / 2)
        fb = FacetBasis(self.mesh, self.basis.elem, facets=fac_fl, intorder=6)
        X = fb.global_coordinates().value[0].ravel()            # x of every quadrature point
        W = fb.dx.ravel()                                        # quadrature weights (um)
        npt = X.size
        rows, cols, vals, vals_e = [], [], [], []
        for i in range(fb.Nbfun):
            bf = fb.basis[i][0]
            us = (bf.value[0] - f["h_m"] * bf.grad[1][0]).ravel()   # u_x - h dv/dx at MoS2 height
            ex = bf.grad[0][0].ravel()                               # du_x/dx on the line
            dof = np.repeat(fb.element_dofs[i][:, None], fb.dx.shape[1], axis=1).ravel()
            rows.append(np.arange(npt)); cols.append(dof); vals.append(us); vals_e.append(ex)
        R, Cc = np.concatenate(rows), np.concatenate(cols)
        self.Cs = sp.csr_matrix((np.concatenate(vals), (R, Cc)), shape=(npt, self.ndof2))
        self.Ex = sp.csr_matrix((np.concatenate(vals_e), (R, Cc)), shape=(npt, self.ndof2))
        # 1D quadratic elements = flake facets sorted by x
        xe = np.unique(self.mesh.p[0, self.mesh.facets[:, fac_fl]])
        ne = xe.size - 1
        self.nm = 2 * ne + 1
        self.xm = np.concatenate([np.ravel(np.column_stack([xe[:-1], 0.5 * (xe[:-1] + xe[1:])])), [xe[-1]]])
        e = np.clip(np.searchsorted(xe, X, side="right") - 1, 0, ne - 1)
        a, b = xe[e], xe[e + 1]
        xi = (X - a) / (b - a)
        N = np.column_stack([2 * (xi - 0.5) * (xi - 1), 4 * xi * (1 - xi), 2 * xi * (xi - 0.5)])
        dN = np.column_stack([4 * xi - 3, 4 - 8 * xi, 4 * xi - 1]) / (b - a)[:, None]
        cm = np.column_stack([2 * e, 2 * e + 1, 2 * e + 2])
        R3 = np.repeat(np.arange(npt), 3)
        self.Cm = sp.csr_matrix((N.ravel(), (R3, cm.ravel())), shape=(npt, self.nm))
        self.Dm = sp.csr_matrix((dN.ravel(), (R3, cm.ravel())), shape=(npt, self.nm))
        self.Wq = sp.diags(W)
        self.fl_len = W.sum()
        # strain probes on the device line (T) and the substrate bottom (B): under the channel (local
        # curvature, rigid island) and far from it, 300-900 um, where the laminate is uniform (neutral plane)
        far = lambda y: np.concatenate([self._line_facets(y, -900, -300), self._line_facets(y, 300, 900)])
        fbs = {"T": fb,
               "B": FacetBasis(self.mesh, self.basis.elem, facets=self._line_facets(self.sub[1], -LF / 2, LF / 2), intorder=6),
               "Tf": FacetBasis(self.mesh, self.basis.elem, facets=far(self.dev_y), intorder=6),
               "Bf": FacetBasis(self.mesh, self.basis.elem, facets=far(self.sub[1]), intorder=6)}
        self.probe = {}
        for key, fbp in fbs.items():
            rr, cc, vv = [], [], []
            w = fbp.dx
            for i in range(fbp.Nbfun):
                g = (fbp.basis[i][0].grad[0][0] * w).sum(axis=1)       # integral over each facet
                rr.append(np.zeros(g.size, int)); cc.append(fbp.element_dofs[i]); vv.append(g)
            self.probe[key] = sp.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                                            shape=(1, self.ndof2)) / w.sum()

    # ---- pure-bending boundary conditions (rigid end faces)
    def _boundary(self):
        D = self.basis.get_dofs(lambda x: np.isclose(x[0], -L / 2))
        E = self.basis.get_dofs(lambda x: np.isclose(x[0], L / 2))
        self.lx, self.ly = D.all(["u^1"]), D.all(["u^2"])
        self.rx, self.ry = E.all(["u^1"]), E.all(["u^2"])
        self.yc = 0.5 * (self.y[0] + self.y[-1])

    # ---- solve
    def solve(self, kappa_um, kind="spring", k=0.0):
        """kappa in 1/um; kind 'spring' (interface stiffness k, uN/um^3) or 'tied' (exact bonding)."""
        n2, nm = self.ndof2, self.nm
        f = self.films
        K = (self.K_solid + self.K_films).tocsr()
        Kmm = f["Et_flake"] * (self.Dm.T @ self.Wq @ self.Dm)
        blocks = [[K, None], [None, Kmm]]
        if kind == "spring":
            blocks[0][0] = K + k * (self.Cs.T @ self.Wq @ self.Cs)
            blocks[0][1] = -k * (self.Cs.T @ self.Wq @ self.Cm)
            blocks[1][0] = blocks[0][1].T
            blocks[1][1] = Kmm + k * (self.Cm.T @ self.Wq @ self.Cm)
            A = sp.bmat(blocks, format="csr")
            nl = 0
        else:   # Lagrange multiplier lambda (same P2 space as u_m): integral mu (u_m - u_s) = 0
            G_m = self.Cm.T @ self.Wq @ self.Cm
            G_s = -(self.Cm.T @ self.Wq @ self.Cs)
            A = sp.bmat([[K, None, G_s.T], [None, Kmm, G_m.T], [G_s, G_m, None]], format="csr")
            nl = nm
        ntot = n2 + nm + nl
        # transformation full = T r + g ; extra reduced unknown U_R (right-end axial translation)
        phiL, phiR = kappa_um * L / 2, -kappa_um * L / 2
        g = np.zeros(ntot)
        yl = self.basis.doflocs[1, self.lx]
        yr = self.basis.doflocs[1, self.rx]
        g[self.lx] = -phiL * (yl - self.yc)
        g[self.rx] = -phiR * (yr - self.yc)
        fixed = np.zeros(ntot, bool)
        fixed[np.concatenate([self.lx, self.ly, self.rx, self.ry])] = True
        free = np.nonzero(~fixed)[0]
        nfree = free.size
        rows = np.concatenate([free, self.rx])
        cols = np.concatenate([np.arange(nfree), np.full(self.rx.size, nfree)])
        T = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(ntot, nfree + 1))
        Ar = (T.T @ A @ T).tocsc()
        r = spsolve(Ar, -(T.T @ (A @ g)))
        U = T @ r + g
        u2, um = U[:n2], U[n2:n2 + nm]
        eps_ch = (um[-1] - um[0]) / LF                         # channel-averaged MoS2 strain (exact for P2)
        eps_surf = float((self.Wq @ (self.Ex @ u2)).sum() / self.fl_len)
        e = {k: float(self.probe[k] @ u2) for k in self.probe}
        t_sub = self.sub[2] - self.sub[1]
        return dict(eps_ch=eps_ch, eps_surf=eps_surf, eT=e["T"], eB=e["B"],
                    kap_fe=(e["T"] - e["B"]) / t_sub / kappa_um,                     # local, under the channel
                    kap_far=(e["Tf"] - e["Bf"]) / t_sub / kappa_um,
                    zn=self.sub[1] + (-e["Bf"] / (e["Tf"] - e["Bf"])) * t_sub,       # neutral plane, uniform region
                    eps_bonded_far=e["Tf"] + self.films["h_m"] * (e["Tf"] - e["Bf"]) / t_sub,
                    epk_um=eps_ch / kappa_um, u=u2, um=um, dofs=int(Ar.shape[0]))

    # ---- post-processing: element-average axial strain
    def element_strain(self, u):
        from skfem import Functional
        ux = self.basis.interpolate(u)

        @Functional
        def exx(w):
            return w["u"].grad[0][0]
        vals = exx.elemental(self.basis, u=ux)
        area = Functional(lambda w: 1.0 + 0 * w.x[0]).elemental(self.basis)
        return vals / area
