"""
Open-source finite-element method for the paper (scikit-fem), replacing COMSOL.

  1. laminate mechanics (fem/laminate.py): mesh convergence, interface-spring calibration (eta),
     seven structures, strain-field images
  2. Debye length from a 1D Poisson-Boltzmann FE solve (fem/debye.py)
  3. sensor chain (Stage 3-6 equations of motion_artifact_sim.py) driven by the FE strain per curvature
  4. post-processing shared with the COMSOL pipeline (comsol/assemble.py): fem/results.json with the same
     keys as figures/results_python.json, fem/comparison.csv, figs 3-8 in fem/figures, fem/FEM_METHODS.md

Usage: python fem/run_fem.py [--levels 1,2,3,4,5]
"""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "comsol")]

import motion_artifact_sim as ms            # noqa: E402
from params import EXTRA                    # noqa: E402  (Poisson ratios etc., shared with the COMSOL design)
from assemble import STRUCTS, PREBENT_RATIO, assemble   # noqa: E402
from laminate import Laminate, LF, W_ISL    # noqa: E402

P = ms.P
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
MPa = 1e-6                                   # Pa -> MPa
UM = 1e6                                     # m -> um
KAPPA = 1e-6                                 # 1 1/m expressed in 1/um
RT = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def nu(name, nu_on=1.0):
    return EXTRA[name][0] * nu_on


# ---------------------------------------------------------------- stack definitions (um, MPa)
def films(cap_on=1, nu_on=1.0):
    Et = lambda E, t, n: E * MPa * t * UM / (1 - n ** 2)
    Au = Et(P["E_Au"], P["t_Au"], nu("nu_Au", nu_on))
    Al = Et(P["E_Al2O3"], P["t_Al2O3"], nu("nu_Al2O3", nu_on))
    Mo = Et(P["E_MoS2"], P["t_MoS2"], nu("nu_MoS2", nu_on))
    return dict(Et_all=Au + cap_on * Al, Et_flake=Mo, h_m=(P["t_Au"] + P["t_MoS2"] / 2) * UM)


def base(level, t_sub=25.0, E=None, nu_s=None, cap_on=1, nu_on=1.0):
    E = P["E_PI"] * MPa if E is None else E
    nu_s = nu("nu_PI", nu_on) if nu_s is None else nu_s
    return Laminate([("sub", 0.0, t_sub, (E, nu_s))], t_sub, films(cap_on, nu_on), level)


def capped(level, t_cap, E_cap, nu_cap, nu_on=1.0):
    t = 25.0
    return Laminate([("sub", 0.0, t, (P["E_PI"] * MPa, nu("nu_PI", nu_on))), ("cap", t, t + t_cap, (E_cap, nu_cap))],
                    t, films(1, nu_on), level)


def island(level, nu_on=1.0):
    tp, ti = EXTRA["t_PDMS"][0] * UM, EXTRA["t_isl"][0] * UM
    pd = (EXTRA["E_PDMS"][0] * MPa, nu("nu_PDMS", nu_on))
    isl = dict(E=EXTRA["E_isl"][0] * MPa, nu=nu("nu_isl", nu_on), E_out=pd[0], nu_out=pd[1])
    return Laminate([("pdms", -tp, -ti, pd), ("isl", -ti, 0.0, "island"), ("sub", 0.0, 25.0, (P["E_PI"] * MPa, nu("nu_PI", nu_on)))],
                    25.0, films(1, nu_on), level, island=isl)


def timed(key, fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    RT[key] = RT.get(key, 0.0) + time.perf_counter() - t0
    return out


def seed_k(eta, Et):
    """Shear-lag seed: <eps_flake>/eps_surf = 1 - tanh(a)/a, a = (L/2) sqrt(k/Et)."""
    from scipy.optimize import brentq
    a = brentq(lambda a: 1 - math.tanh(a) / a - eta, 1e-3, 1e4)
    return (2 * a / LF) ** 2 * Et


def calibrate(lam, eta):
    """Secant on log10 k so that eps_MoS2 / eps_substrate-surface (channel averages, same solve) = eta."""
    def ratio(logk):
        r = timed("calibration", lam.solve, KAPPA, "spring", 10 ** logk)
        return r["eps_ch"] / r["eps_surf"]
    x0 = math.log10(seed_k(eta, lam.films["Et_flake"]))
    x1 = x0 + 0.3
    f0, f1 = ratio(x0) - eta, ratio(x1) - eta
    hist = [(x0, f0 + eta), (x1, f1 + eta)]
    for _ in range(40):
        if abs(f1) < 1e-6:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0, x1 = x1, f1, x2
        f1 = ratio(x1) - eta
        hist.append((x1, f1 + eta))
    return 10 ** x1, f1 + eta, hist


# ---------------------------------------------------------------- 1. mechanics
def run_mechanics(levels):
    raw = {}
    k_seed = seed_k(P["eta_thinALD"], films()["Et_flake"])
    log("mesh convergence (baseline)")
    conv = []
    for lev in levels:
        t0 = time.perf_counter()
        lam = base(lev)
        rt = lam.solve(KAPPA, "tied")
        rs = lam.solve(KAPPA, "spring", k_seed)
        conv.append(dict(level=lev, n_elem=int(lam.n_elem), dofs=rs["dofs"], epk_tied_um=rt["epk_um"],
                         epk_spring_um=rs["epk_um"], zn_um=rs["zn"], kappa_fe_over_kappa=rs["kap_far"],
                         time_s=time.perf_counter() - t0))
        log(f"  level {lev}: {lam.n_elem} elements, {rs['dofs']} DOFs, eps/kappa spring {rs['epk_um']:.5f} um, "
            f"tied {rt['epk_um']:.5f} um, z_n {rs['zn']:.5f} um, {conv[-1]['time_s']:.1f} s")
        del lam
    fin = conv[-1]
    for r in conv:
        r["d_spring_pct"] = (r["epk_spring_um"] - fin["epk_spring_um"]) / fin["epk_spring_um"] * 100
        r["d_tied_pct"] = (r["epk_tied_um"] - fin["epk_tied_um"]) / fin["epk_tied_um"] * 100
    ok = [r["level"] for r in conv if abs(r["d_spring_pct"]) < 1 and abs(r["d_tied_pct"]) < 1]
    prod = max(3, min(ok)) if ok else levels[-1]
    raw["convergence"], raw["production_level"] = conv, prod
    RT["convergence"] = sum(r["time_s"] for r in conv)
    log("production level", prod)

    log("calibrating the interface spring")
    lam_b = base(prod)
    k_ald, eta_ald, h_ald = calibrate(lam_b, P["eta_thinALD"])
    lam_bare = base(prod, cap_on=0)
    k_bare, eta_bare, h_bare = calibrate(lam_bare, P["eta_bare"])
    rt = lam_b.solve(KAPPA, "tied")
    raw["calib"] = dict(k_bare=k_bare * 1e18, eta_bare=eta_bare, k_ALD=k_ald * 1e18, eta_ALD=eta_ald,
                        eta_polymer=rt["eps_ch"] / rt["eps_surf"], units="N/m^3",
                        hist_bare=h_bare, hist_ALD=h_ald)
    log(f"  k_ALD = {k_ald*1e18:.4e} N/m^3 (eta {eta_ald:.6f}); k_bare = {k_bare*1e18:.4e} N/m^3 (eta {eta_bare:.6f}); "
        f"bonded ratio {raw['calib']['eta_polymer']:.5f}")

    log("structures")
    epk, sols = {}, {}
    r = timed("baseline", lam_b.solve, KAPPA, "spring", k_ald)
    epk[STRUCTS[0]] = r["epk_um"] / UM
    raw["zn_m"] = r["zn"] / UM
    raw["baseline_eps_surf_per_kappa_m"] = r["eps_surf"] / KAPPA / UM
    raw["baseline_bonded_far_per_kappa_m"] = r["eps_bonded_far"] / KAPPA / UM
    raw["eps_motion_direct"] = {n: r["eps_ch"] / KAPPA / ms.MOTIONS[n]["R_min"] for n in ("wrist", "elbow", "walking", "hold")}
    raw["baseline_tied_epk_m"] = rt["epk_um"] / UM
    sols["baseline_bent"] = (lam_b, r)
    raw["bare_epk_m"] = lam_bare.solve(KAPPA, "spring", k_bare)["epk_um"] / UM
    del lam_bare

    for name, kw in ((STRUCTS[2], dict(t_sub=12.5)),
                     (STRUCTS[3], dict(t_sub=2.5, E=P["E_PET"] * MPa, nu_s=nu("nu_PET")))):
        lam = base(prod, **kw)
        epk[name] = timed(name, lam.solve, KAPPA, "spring", k_ald)["epk_um"] / UM
        log(f"  {name}: {epk[name]*UM:.5f} um")

    lam = base(prod, nu_on=0.0)
    r0 = lam.solve(KAPPA, "tied")
    raw["nu0_check"] = dict(zn_m=r0["zn"] / UM, epk_tied_m=r0["epk_um"] / UM, bonded_far_m=r0["eps_bonded_far"] / KAPPA / UM)
    fl = films(1, 0.0)
    E = [P["E_PI"] * MPa, fl["Et_all"]]
    raw["nu0_check"]["zn_theory_same_films_m"] = (E[0] * 25.0 * 12.5 + E[1] * 25.0) / (E[0] * 25.0 + E[1]) / UM

    lam = capped(prod, P["t_PMMA"] if "t_PMMA" in P else EXTRA["t_PMMA"][0] * UM, P["E_PMMA"] * MPa, nu("nu_PMMA"))
    epk[STRUCTS[1]] = timed(STRUCTS[1], lam.solve, KAPPA, "tied")["epk_um"] / UM
    log(f"  {STRUCTS[1]}: {epk[STRUCTS[1]]*UM:.5f} um")
    lam = capped(prod, EXTRA["t_buried_cap"][0] * UM, P["E_PI"] * MPa, nu("nu_PI"))
    r = timed(STRUCTS[4], lam.solve, KAPPA, "tied")
    epk[STRUCTS[4]] = r["epk_um"] / UM
    raw["buried_zn_m"] = r["zn"] / UM
    sols["buried_channel"] = (lam, r)
    log(f"  {STRUCTS[4]}: {epk[STRUCTS[4]]*UM:.5f} um")
    lam = capped(prod, 25.0, P["E_PI"] * MPa, nu("nu_PI"))
    raw["sym_epk_m"] = lam.solve(KAPPA, "tied")["epk_um"] / UM
    log(f"  symmetric 25/25 um stack: {raw['sym_epk_m']*UM:.6f} um")

    lam = island(prod)
    r = timed(STRUCTS[5], lam.solve, KAPPA, "spring", k_ald)
    epk[STRUCTS[5]] = r["epk_um"] / UM
    raw["island_local_kappa_ratio"] = r["kap_fe"]
    raw["island_ratio_to_baseline"] = epk[STRUCTS[5]] / epk[STRUCTS[0]]
    sols["rigid_island"] = (lam, r)
    log(f"  {STRUCTS[5]}: {epk[STRUCTS[5]]*UM:.5f} um, ratio to baseline {raw['island_ratio_to_baseline']:.4f}, "
        f"local curvature ratio {r['kap_fe']:.4f}")

    epk[STRUCTS[6]] = epk[STRUCTS[0]] * PREBENT_RATIO
    raw["epk"] = {n: epk[n] for n in STRUCTS}
    raw["runtimes"] = dict(RT)
    return raw, sols


# ---------------------------------------------------------------- strain-field images
def strain_images(sols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    views = {"baseline_bent": (-40, 40, -2, 27, "Baseline stack (25 um PI), wrist flexion R = 25 mm"),
             "buried_channel": (-40, 40, -2, 50, "Buried channel (25 um PI / 23 um PI), R = 25 mm"),
             "rigid_island": (-2500, 2500, -110, 30, "Rigid island (3 mm, 70 GPa) in PDMS under 25 um PI, R = 25 mm")}
    os.makedirs(FIGS, exist_ok=True)
    kw = 1 / (ms.MOTIONS["wrist"]["R_min"] * UM)       # wrist curvature in 1/um; the model is linear
    for name, (lam, r) in sols.items():
        x0, x1, y0, y1, title = views[name]
        ex = lam.element_strain(r["u"]) * (kw / KAPPA) * 100    # % at wrist curvature
        pts = lam.mesh.p[:, lam.mesh.t]                          # (2, 4, nelem)
        c = pts.mean(axis=1)
        sel = (c[0] > x0) & (c[0] < x1) & (c[1] > y0) & (c[1] < y1)
        polys = np.transpose(pts[:, :, sel], (2, 1, 0))
        fig, ax = plt.subplots(figsize=(7.1, 2.6))
        pc = PolyCollection(polys, array=ex[sel], cmap="RdBu_r", edgecolors="none")
        lim = np.max(np.abs(ex[sel]))
        pc.set_clim(-lim, lim)
        ax.add_collection(pc)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.axhline(lam.dev_y, color="k", lw=0.4)
        ax.plot([-LF / 2, LF / 2], [lam.dev_y, lam.dev_y], color="#e6a700", lw=2.5, solid_capstyle="butt",
                label="MoS$_2$ channel")
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)"); ax.set_title(title, fontsize=8)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        cb = fig.colorbar(pc, ax=ax, pad=0.01); cb.set_label("axial strain $\\varepsilon_{xx}$ (%)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"fem_strain_{name}.png"), dpi=600)
        plt.close(fig)
        log("  image", name)


# ---------------------------------------------------------------- 3. sensor chain driven by FE strain
def chain_exports(mech, debye):
    """Stage 3-6 (equations of motion_artifact_sim.py) with the FE eps/kappa; same export format as the COMSOL run."""
    epk = mech["epk"]
    s_base = epk[STRUCTS[0]]
    FS = ms.dVT_max_sweat() / P["V_ov"]
    ng = ms.NGML_TO_NM
    dI = lambda e, V=P["V_ov"]: float(ms.rel_current_change_strain(e, V))
    motions = {}
    for n in ["wrist", "elbow", "walking", "hold"]:
        e = s_base / ms.MOTIONS[n]["R_min"]
        S = FS * P["K_D_nM"] / (30 * ng + P["K_D_nM"]) ** 2
        c30, c50 = ms.false_concentration_ngml(e, 30.0), ms.false_concentration_ngml(e, 50.0)
        motions[n] = dict(eps=e, dI=dI(e), lin_false30=dI(e) / S / ng,
                          C_false30=float(c30) if np.isfinite(c30) else 0.0, sat30=bool(~np.isfinite(c30)),
                          C_false50=float(c50) if np.isfinite(c50) else 0.0, sat50=bool(~np.isfinite(c50)))
    e_w = s_base / 25e-3
    R0 = ms.response(50 * ng)
    rc0 = P["r_c"]; P["r_c"] = 0.0
    datye = dI(0.007, 0.145)
    P["r_c"] = rc0
    chain = dict(
        FS=FS, V_ov=P["V_ov"], dVT_max_V=ms.dVT_max_sweat(), K_D_nM=P["K_D_nM"], ngml_to_nM=ng,
        scalars=dict(GF_ch_0p3=ms.GF_channel(0.3), GF_c=ms.GF_contact(), GF_tot_0p3=ms.GF_total_small_signal(0.3),
                     lambda_D_nm=ms.debye_length_nm(0.05), dVT_sweat_mV=ms.dVT_max_sweat() * 1e3, FS_pct=FS * 100,
                     K_D_ngml=P["K_D_nM"] / ng, k_on=P["k_on"],
                     tau_100ngml_s=1 / (P["k_on"] * 100 * ng * 1e-9 + P["k_off"]),
                     R_phys_low_pct=ms.response(8.16 * ng) * 100, R_phys_high_pct=ms.response(141.7 * ng) * 100,
                     john_sum=147 + 55, li_xt=P["kappa_x"] * 0.0034 * 100, datye=datye,
                     GF_ch_0p145=float(ms.GF_channel(0.145)), lambda_100mM_formula_nm=ms.debye_length_nm(0.1)),
        motions=motions, xtalk_floor=P["kappa_x"] * e_w * R0 / FS,
        mitigation={n: dict(wrist=dI(epk[n] / 25e-3) / FS, elbow=dI(epk[n] / 40e-3) / FS,
                            walking=dI(epk[n] / 300e-3) / FS, eps_wrist=epk[n] / 25e-3) for n in STRUCTS})
    raw = dict(mech=mech, chain=chain, debye=debye, runtimes=dict(RT),
               comsol_version=f"scikit-fem {__import__('skfem').__version__}, Python {platform.python_version()}")
    json.dump(raw, open(os.path.join(DATA, "raw.json"), "w"), indent=1)

    SW = {}
    Vov = np.linspace(0.05, 1.0, 200)
    SW["gate_V"] = Vov
    for rc in (0.3, 0.0, 0.6):
        P["r_c"] = rc
        SW[f"gate_ratio_rc{rc:g}"] = np.array([dI(e_w, V) / (ms.dVT_max_sweat() / V) for V in Vov])
    P["r_c"] = 0.3
    eg = np.linspace(-1e-3, 1e-3, 4001)
    SW["transfer_eps"], SW["transfer_dI"] = eg, ms.rel_current_change_strain(eg, P["V_ov"])
    e4 = np.linspace(-0.1, 0.1, 400) / 100
    SW["fig4a_eps"] = e4
    for V in (0.1, 0.3, 0.6):
        SW[f"fig4a_V{V:g}"] = ms.rel_current_change_strain(e4, V)
    for rc, k in ((0.0, "fig4a_chonly"), (1.0, "fig4a_conly")):
        P["r_c"] = rc; SW[k] = ms.rel_current_change_strain(e4, 0.3)
    P["r_c"] = 0.3
    Cg = np.logspace(-1, 3.2, 300)
    SW["fig4b_C"] = Cg
    for I in (0.001, 0.05, 0.16):
        P["I_sweat_M"] = I
        SW[f"fig4b_R_I{I:g}"], SW[f"fig4b_lam_I{I:g}"] = ms.response(Cg * ng), np.full_like(Cg, ms.debye_length_nm(I))
    P["I_sweat_M"] = 0.05
    C0s = np.linspace(5, 140, 200)
    SW["fig5a_C0"] = C0s
    rcs = np.linspace(0, 0.8, 100)
    SW["fig5b_rc"] = rcs
    for n in ("walking", "elbow", "wrist"):
        e = s_base / ms.MOTIONS[n]["R_min"]
        SW[f"fig5a_cf_{n}"] = np.array([ms.false_concentration_ngml(e, c) for c in C0s])
        vals = []
        for r in rcs:
            P["r_c"] = r; vals.append(dI(e) / FS * 100)
        P["r_c"] = 0.3
        SW[f"fig5b_{n}"] = np.array(vals)
    mism = np.linspace(0, 0.2, 50)
    SW["fig7b_mism"] = mism
    SW["fig7b_resid"] = np.abs(dI(e_w) * mism + P["kappa_x"] * e_w * R0) / FS * 100
    SW["fig7b_resid_noX"] = np.abs(dI(e_w) * mism) / FS * 100
    np.savez_compressed(os.path.join(DATA, "raw_sweeps.npz"), **SW)

    for tag, n in (("baseline", STRUCTS[0]), ("thin", STRUCTS[3]), ("island", STRUCTS[5])):
        t = np.arange(0, 1800, 0.05)
        Ct = np.where(t < 300, 30.0, 100.0) * ng
        th = ms.binding_kinetics(t, Ct)
        out = ms.scenario(eps_per_kappa=epk[n])
        eps = out["eps"]
        R_bio0 = ms.dVT_max_sweat() * th / P["V_ov"]
        art = ms.rel_current_change_strain(eps, P["V_ov"])
        np.savez_compressed(os.path.join(DATA, f"raw_time_{tag}.npz"), t=t, kap=out["kappa"], eps=eps, theta=th,
                            R_bio0=R_bio0, R_sens=R_bio0 * (1 + P["kappa_x"] * eps) + art, R_ref=art * 1.05,
                            Cb_ngml=Ct / ng)
    t3 = np.linspace(0, 4, 2000)
    M3 = {"t": t3}
    for n in ("wrist", "elbow", "walking"):
        M3[f"kap_{n}"] = ms.curvature(t3, ms.MOTIONS[n]); M3[f"eps_{n}"] = s_base * M3[f"kap_{n}"]
    np.savez_compressed(os.path.join(DATA, "raw_motion.npz"), **M3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,3,4,5")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    t0 = time.perf_counter()
    mech, sols = run_mechanics(tuple(int(x) for x in a.levels.split(",")))
    json.dump({"mech": mech}, open(os.path.join(DATA, "raw_mech.json"), "w"), indent=1)
    strain_images(sols)
    del sols
    from debye import debye_lengths
    debye = timed("debye", debye_lengths)
    log(f"Debye length (FE Poisson-Boltzmann): 50 mM {debye['lambda_50mM_nm']:.5f} nm, 100 mM {debye['lambda_100mM_nm']:.5f} nm")
    RT["total_s"] = time.perf_counter() - t0
    chain_exports(mech, debye)
    py = json.load(open(os.path.join(ROOT, "figures", "results_python.json")))
    assemble(DATA, HERE, python_results=py)
    subprocess.run([sys.executable, os.path.join(ROOT, "make_figures.py"), "--source", "fem"], check=True, cwd=ROOT)
    subprocess.run([sys.executable, os.path.join(HERE, "write_methods.py")], check=True, cwd=ROOT)
    log("done: fem/results.json, fem/comparison.csv, fem/figures, fem/FEM_METHODS.md")


if __name__ == "__main__":
    main()
