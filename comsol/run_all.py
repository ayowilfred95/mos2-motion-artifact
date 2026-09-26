"""
Run every COMSOL solve used in the paper and write the results.

  stage 1 (COMSOL):  mechanics (mesh convergence, eta calibration, 7 structures, images),
                     sensor chain (stationary sweeps + 1800 s time-dependent scenario), Debye 1D.
                     Raw exports -> comsol/data/raw.json, raw_sweeps.npz, raw_time_*.npz, raw_motion.npz
  stage 2 (post):    assemble.py -> comsol/results.json, comsol/data/curves.npz, comsol/comparison.csv
                     make_figures.py --source comsol -> comsol/figures/fig3..fig8
                     write_methods.py -> comsol/COMSOL_METHODS.md

Usage:
  python comsol/run_all.py                 # everything
  python comsol/run_all.py --stage mech    # only the mechanics model (+ convergence report)
  python comsol/run_all.py --stage post    # re-assemble from existing raw exports (no COMSOL needed)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import params as prm                                            # noqa: E402
import build_model as bm                                        # noqa: E402
from comsol_util import dataset_of_study, eval_global, eval_nodes, setp_first, Timer  # noqa: E402
from assemble import STRUCTS, PREBENT_RATIO, assemble          # noqa: E402

DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
T = Timer()
INNER = {"ref": 0, "wrist": 1, "elbow": 2, "walking": 3, "hold": 4}   # index into build_model.KAPPA_LIST


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ====================================================================== mechanics
class Mech:
    def __init__(self, client, level0=1):
        self.model = bm.build_mechanics(client, level=level0)
        self.j = self.model.java
        self.specs = bm.mech_specs()
        self.defaults = dict(bm.prm.mech_params())
        self.defaults.update({"t_cap": "t_PMMA", "E_cap": "E_PMMA", "nu_cap": "nu_PMMA*nu_on"})

    def set(self, **kw):
        for k, v in kw.items():
            self.j.param().set(k, str(v))

    def reset(self, **extra):
        for k, v in self.defaults.items():
            self.j.param().set(k, v)
        self.set(**extra)

    def remesh(self, c, level):
        self.j.component(c).geom(f"geom_{c}").run()
        bm.set_mesh_level(self.model, c, self.specs[c], level)

    def solve(self, c, kind):
        with T(f"mech_{c}_{kind}"):
            self.j.study(f"std_{c}_{kind}").run()
        return dataset_of_study(self.model, f"std_{c}_{kind}")

    def ev(self, c, kind, names, ds=None):
        ds = ds or dataset_of_study(self.model, f"std_{c}_{kind}")
        vals = eval_global(self.model, [f"{c}.{n}" for n in names], ds)
        return {n: vals[i] for i, n in enumerate(names)}

    def ndofs(self, c, kind):
        for s in self.j.sol().tags():
            if str(self.j.sol(s).study()) == f"std_{c}_{kind}":
                try:
                    return int(len(self.j.sol(s).getU()))
                except Exception:
                    return -1
        return -1

    def nelem(self, c):
        try:
            return int(self.j.component(c).mesh(f"mesh_{c}").getNumElem())
        except Exception:
            return -1


def eta_seed_k(eta, Et):
    """Shear-lag seed for the spring stiffness: <eps_flake>/eps_surf = 1 - tanh(a)/a, a = (L/2) sqrt(k/Et)."""
    from scipy.optimize import brentq
    a = brentq(lambda a: 1 - math.tanh(a) / a - eta, 1e-3, 1e4)
    return (2 * a / bm.LF) ** 2 * Et


def run_mechanics(client, levels=(1, 2, 3, 4, 5)):
    raw = {"runtimes": {}}
    M = Mech(client, level0=levels[0])
    P = prm.P
    Et_ald = P["E_MoS2"] * P["t_MoS2"] / (1 - 0.27 ** 2) + P["E_Al2O3"] * P["t_Al2O3"] / (1 - 0.24 ** 2)
    Et_bare = P["E_MoS2"] * P["t_MoS2"] / (1 - 0.27 ** 2)
    k_seed = eta_seed_k(P["eta_thinALD"], Et_ald)

    # ------------------------------------------------ 1. mesh convergence (baseline, both bonding models)
    log("mesh convergence")
    M.reset(k_int=f"{k_seed}[N/m^3]")
    conv = []
    for lev in levels:
        t0 = time.perf_counter()
        M.remesh("base", lev)
        M.solve("base", "tied")
        rt = M.ev("base", "tied", ["epk_fe", "zn_fe"])
        M.solve("base", "spring")
        rs = M.ev("base", "spring", ["epk_fe", "zn_fe", "kap_fe"])
        row = dict(level=lev, n_elem=M.nelem("base"), dofs=M.ndofs("base", "spring"),
                   epk_tied_um=rt["epk_fe"][0] * 1e6, epk_spring_um=rs["epk_fe"][0] * 1e6,
                   zn_um=rs["zn_fe"][0] * 1e6, kappa_fe_over_kappa=rs["kap_fe"][0],
                   time_s=time.perf_counter() - t0)
        conv.append(row)
        log("  level", lev, row)
    fin = conv[-1]
    for r in conv:
        r["d_spring_pct"] = (r["epk_spring_um"] - fin["epk_spring_um"]) / fin["epk_spring_um"] * 100
        r["d_tied_pct"] = (r["epk_tied_um"] - fin["epk_tied_um"]) / fin["epk_tied_um"] * 100
    ok = [r["level"] for r in conv if abs(r["d_spring_pct"]) < 1 and abs(r["d_tied_pct"]) < 1]
    prod = max(3, min(ok)) if ok else levels[-1]
    raw["convergence"] = conv
    raw["production_level"] = prod
    log("production mesh level", prod)
    for c in M.specs:
        M.remesh(c, prod)

    # ------------------------------------------------ 2. calibrate the Thin Elastic Layer stiffness
    def calibrate(eta, cap_on, k0):
        M.reset(cap_on=cap_on)
        M.solve("base", "tied")
        e_tied = M.ev("base", "tied", ["eps_ch"])["eps_ch"][0]

        def ratio(logk):
            M.set(k_int=f"{10 ** logk}[N/m^3]")
            M.solve("base", "spring")
            return M.ev("base", "spring", ["eps_ch"])["eps_ch"][0] / e_tied
        x0, x1 = math.log10(k0), math.log10(k0) + 0.3
        f0, f1 = ratio(x0) - eta, ratio(x1) - eta
        hist = [(x0, f0 + eta), (x1, f1 + eta)]
        for _ in range(30):
            if abs(f1) < 1e-5:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            x0, f0 = x1, f1
            x1, f1 = x2, ratio(x2) - eta
            hist.append((x1, f1 + eta))
        return 10 ** x1, f1 + eta, hist

    log("calibrating k_int")
    with T("calibration"):
        k_bare, eta_bare, h_bare = calibrate(P["eta_bare"], 0, eta_seed_k(P["eta_bare"], Et_bare))
        k_ald, eta_ald, h_ald = calibrate(P["eta_thinALD"], 1, k_seed)
    raw["calib"] = dict(k_bare=k_bare, eta_bare=eta_bare, k_ALD=k_ald, eta_ALD=eta_ald,
                        eta_polymer="pointwise constraint (exact bonding)",
                        hist_bare=h_bare, hist_ALD=h_ald)
    log("  k_bare", k_bare, "eta", eta_bare, " k_ALD", k_ald, "eta", eta_ald)
    M.j.param().set("k_bare", f"{k_bare}[N/m^3]")
    M.j.param().set("k_ALD", f"{k_ald}[N/m^3]")
    M.defaults["k_int"] = "k_ALD"
    M.j.param().set("k_int", "k_ALD")

    epk = {}
    # ------------------------------------------------ 3. baseline + substrate variants (spring, eta = 0.8)
    log("structures")
    M.reset(k_int="k_ALD")
    M.remesh("base", prod)
    ds = M.solve("base", "spring")
    r = M.ev("base", "spring", ["epk_fe", "zn_fe", "eps_ch", "eps_surf", "kap_fe"], ds)
    epk[STRUCTS[0]] = r["epk_fe"][INNER["ref"]]
    raw["zn_m"] = r["zn_fe"][0]
    raw["eps_motion_direct"] = {k: r["eps_ch"][i] for k, i in INNER.items() if k != "ref"}
    raw["baseline_eps_surf_per_kappa_m"] = r["eps_surf"][0]
    export_strain_image(M, "base", ds, INNER["wrist"], (-40e-6, 40e-6, -2e-6, 27e-6), "baseline_bent")
    M.solve("base", "tied")
    raw["baseline_tied_epk_m"] = M.ev("base", "tied", ["epk_fe"])["epk_fe"][0]

    for name, kw in ((STRUCTS[2], dict(t_sub="12.5e-6[m]")),
                     (STRUCTS[3], dict(t_sub="2.5e-6[m]", E_sub="E_PET", nu_sub="nu_PET*nu_on"))):
        M.reset(k_int="k_ALD", **kw)
        M.remesh("base", prod)
        M.solve("base", "spring")
        epk[name] = M.ev("base", "spring", ["epk_fe"])["epk_fe"][0]
        log("  ", name, epk[name])

    # nu = 0 check (plane strain == 1D beam theory) and bare-flake check
    M.reset(k_int="k_ALD", nu_on=0)
    M.remesh("base", prod)
    M.solve("base", "tied")
    rt = M.ev("base", "tied", ["epk_fe", "zn_fe"])
    raw["nu0_check"] = dict(epk_tied_m=rt["epk_fe"][0], zn_m=rt["zn_fe"][0])
    M.reset(k_int="k_bare", cap_on=0)
    M.solve("base", "spring")
    raw["bare_epk_m"] = M.ev("base", "spring", ["epk_fe"])["epk_fe"][0]

    # ------------------------------------------------ 4. capped stacks (eta = 1, tied)
    for name, kw, img in ((STRUCTS[1], dict(t_cap="t_PMMA", E_cap="E_PMMA", nu_cap="nu_PMMA*nu_on"), None),
                          (STRUCTS[4], dict(t_cap="t_buried_cap", E_cap="E_PI", nu_cap="nu_PI*nu_on"), "buried_channel")):
        M.reset(**kw)
        M.remesh("capped", prod)
        ds = M.solve("capped", "tied")
        rr = M.ev("capped", "tied", ["epk_fe", "zn_fe"], ds)
        epk[name] = rr["epk_fe"][0]
        if img:
            raw["buried_zn_m"] = rr["zn_fe"][0]
            export_strain_image(M, "capped", ds, INNER["wrist"], (-40e-6, 40e-6, -2e-6, 50e-6), img)
        log("  ", name, epk[name])
    M.reset(t_cap="25e-6[m]", E_cap="E_PI", nu_cap="nu_PI*nu_on")
    M.remesh("capped", prod)
    M.solve("capped", "tied")
    raw["sym_epk_m"] = M.ev("capped", "tied", ["epk_fe"])["epk_fe"][0]

    # ------------------------------------------------ 5. rigid island (eta = 0.8)
    M.reset(k_int="k_ALD")
    M.remesh("island", prod)
    ds = M.solve("island", "spring")
    ri = M.ev("island", "spring", ["epk_fe", "kap_fe"], ds)
    epk[STRUCTS[5]] = ri["epk_fe"][0]
    raw["island_local_kappa_ratio"] = ri["kap_fe"][0]
    raw["island_ratio_to_baseline"] = epk[STRUCTS[5]] / epk[STRUCTS[0]]
    export_strain_image(M, "island", ds, INNER["wrist"], (-2.5e-3, 2.5e-3, -110e-6, 30e-6), "rigid_island")
    log("  ", STRUCTS[5], epk[STRUCTS[5]], "ratio", raw["island_ratio_to_baseline"])

    # ------------------------------------------------ 6. prebent channel: published ratio on COMSOL baseline
    epk[STRUCTS[6]] = epk[STRUCTS[0]] * PREBENT_RATIO
    raw["epk"] = {n: epk[n] for n in STRUCTS}

    # restore the baseline state, solve everything once more and save
    M.reset(k_int="k_ALD")
    for c in M.specs:
        M.remesh(c, prod)
    M.set(t_cap="t_buried_cap", E_cap="E_PI", nu_cap="nu_PI*nu_on")
    M.remesh("capped", prod)
    for c, kind in (("base", "spring"), ("base", "tied"), ("capped", "tied"), ("island", "spring")):
        M.solve(c, kind)
    M.model.save(os.path.join(HERE, "mos2_patch_mechanics.mph"))
    raw["runtimes"] = dict(T.t)
    return raw


def export_strain_image(M, c, ds, inner, view, name):
    """COMSOL-rendered strain field (PNG, 600 dpi) of the current solution."""
    j = M.j
    try:
        tag = f"pg_{name}"
        if tag in [str(t) for t in j.result().tags()]:
            j.result().remove(tag)
        vt = f"view_{name}"
        if vt in [str(t) for t in j.component(c).view().tags()]:
            j.component(c).view().remove(vt)
        pg = bm.add_strain_plot(M.model, c, ds, inner + 1, view, name)
        pg.run()
        et = f"img_{name}"
        if et in [str(t) for t in j.result().export().tags()]:
            j.result().export().remove(et)
        img = j.result().export().create(et, "Image")
        setp_first(img, ["sourceobject", "plotgroup"], tag)
        os.makedirs(FIGS, exist_ok=True)
        setp_first(img, ["pngfilename", "filename"], os.path.join(FIGS, f"comsol_strain_{name}.png"))
        for k, v in (("size", "manualprint"), ("unit", "mm"), ("width", "170"), ("height", "70"),
                     ("resolution", "600"), ("imagetype", "png")):
            setp_first(img, [k], v, required=False)
        img.run()
        log("  image", name)
    except Exception as e:
        log("  image export failed for", name, ":", e)


# ====================================================================== sensor chain
class Chain:
    def __init__(self, client, epk_base):
        self.model = bm.build_chain(client)
        self.j = self.model.java
        self.defaults = prm.chain_params()
        self.defaults["epk"] = f"{epk_base}[m]"

    def reset(self, **kw):
        for k, v in self.defaults.items():
            self.j.param().set(k, v)
        for k, v in kw.items():
            self.j.param().set(k, str(v))

    def sweep(self, pname, values, unit, exprs, units=None, **fixed):
        self.reset(**fixed)
        step = self.j.study("std_stat").feature("stat")
        step.setIndex("pname", pname, 0)
        step.setIndex("plistarr", " ".join(repr(float(v)) for v in values), 0)
        step.setIndex("punit", unit, 0)
        with T("chain_stationary"):
            self.j.study("std_stat").run()
        vals = eval_global(self.model, exprs, dataset_of_study(self.model, "std_stat"), units)
        return {e: vals[i] for i, e in enumerate(exprs)}

    def transient(self, study, exprs, units, **fixed):
        self.reset(**fixed)
        with T(f"chain_{study}"):
            self.j.study(study).run()
        vals = eval_global(self.model, exprs, dataset_of_study(self.model, study), units)
        return {e: vals[i] for i, e in enumerate(exprs)}


def run_chain(client, epk):
    base = epk[STRUCTS[0]]
    C = Chain(client, base)
    R = dict(motions={}, mitigation={}, scalars={})
    kw = {"wrist": 1 / 0.025, "elbow": 1 / 0.040, "walking": 1 / 0.300, "hold": 1 / 0.030}
    log("chain: scalars")
    ex = {"GF_ch": "1", "GF_c": "1", "GF_tot": "1", "lambda_D": "nm", "dVT_max": "mV", "dVT_max*1": "V", "FS": "1",
          "K_D_ngml": "ng/ml", "k_on": "1/(M*s)", "V_ov": "V", "K_D": "nM", "1[ng/ml]/MW": "nM",
          "john_sum": "1", "li_xt": "1"}
    s = C.sweep("kap_s", [0], "1/m", list(ex), list(ex.values()))
    v = {k: s[k][0] for k in ex}
    sc = R["scalars"]
    sc.update(GF_ch_0p3=v["GF_ch"], GF_c=v["GF_c"], GF_tot_0p3=v["GF_tot"], lambda_D_nm=v["lambda_D"],
              dVT_sweat_mV=v["dVT_max"], FS_pct=v["FS"] * 100, K_D_ngml=v["K_D_ngml"], k_on=v["k_on"],
              john_sum=v["john_sum"], li_xt=v["li_xt"])
    R.update(FS=v["FS"], V_ov=v["V_ov"], dVT_max_V=v["dVT_max*1"], K_D_nM=v["K_D"], ngml_to_nM=v["1[ng/ml]/MW"])
    sc["tau_100ngml_s"] = C.sweep("C0", [100], "ng/ml", ["tau_on"], ["s"])["tau_on"][0]
    rp = C.sweep("C0", [prm.P["C_phys_low_ngml"], prm.P["C_phys_high_ngml"]], "ng/ml", ["R_bio0"])["R_bio0"]
    sc["R_phys_low_pct"], sc["R_phys_high_pct"] = rp[0] * 100, rp[1] * 100
    d = C.sweep("eps_s", [0.007], "", ["dI_strain", "GF_ch"], r_c=0, V_ov="0.145[V]")
    sc["datye"], sc["GF_ch_0p145"] = d["dI_strain"][0], d["GF_ch"][0]
    sc["lambda_100mM_formula_nm"] = C.sweep("I_ion", [0.1], "M", ["lambda_D"], ["nm"])["lambda_D"][0]

    log("chain: motions")
    names = list(kw)
    ks = [kw[n] for n in names]
    m30 = C.sweep("kap_s", ks, "1/m", ["eps", "dI_strain", "C_false_lin", "C_false", "sat"],
                  ["1", "1", "ng/ml", "ng/ml", "1"], C0="30[ng/ml]")
    m50 = C.sweep("kap_s", ks, "1/m", ["C_false", "sat"], ["ng/ml", "1"], C0="50[ng/ml]")
    for i, n in enumerate(names):
        R["motions"][n] = dict(eps=m30["eps"][i], dI=m30["dI_strain"][i], lin_false30=m30["C_false_lin"][i],
                               C_false30=m30["C_false"][i], sat30=bool(m30["sat"][i] > 0.5),
                               C_false50=m50["C_false"][i], sat50=bool(m50["sat"][i] > 0.5))
    R["xtalk_floor"] = C.sweep("kap_s", [kw["wrist"]], "1/m", ["xt_floor"], C0="50[ng/ml]")["xt_floor"][0]

    log("chain: mitigation")
    for n in STRUCTS:
        r = C.sweep("kap_s", [kw["wrist"], kw["elbow"], kw["walking"]], "1/m", ["art_FS", "eps"], epk=f"{epk[n]}[m]")
        R["mitigation"][n] = dict(wrist=r["art_FS"][0], elbow=r["art_FS"][1], walking=r["art_FS"][2],
                                  eps_wrist=r["eps"][0])

    log("chain: sweeps")
    SW = {}
    Vov = np.linspace(0.05, 1.0, 200)
    SW["gate_V"] = Vov
    for rc in (0.3, 0.0, 0.6):
        SW[f"gate_ratio_rc{rc:g}"] = C.sweep("V_ov", Vov, "V", ["art_FS"], kap_s=f"{kw['wrist']}[1/m]", r_c=rc)["art_FS"]
    e4 = np.linspace(-0.1, 0.1, 400) / 100
    SW["fig4a_eps"] = e4
    for V in (0.1, 0.3, 0.6):
        SW[f"fig4a_V{V:g}"] = C.sweep("eps_s", e4, "", ["dI_strain"], V_ov=f"{V}[V]")["dI_strain"]
    SW["fig4a_chonly"] = C.sweep("eps_s", e4, "", ["dI_strain"], r_c=0)["dI_strain"]
    SW["fig4a_conly"] = C.sweep("eps_s", e4, "", ["dI_strain"], r_c=1)["dI_strain"]
    Cg = np.logspace(-1, 3.2, 300)
    SW["fig4b_C"] = Cg
    for I in (0.001, 0.05, 0.16):
        r = C.sweep("C0", Cg, "ng/ml", ["R_bio0", "lambda_D"], ["1", "nm"], I_ion=f"{I}[M]")
        SW[f"fig4b_R_I{I:g}"], SW[f"fig4b_lam_I{I:g}"] = r["R_bio0"], r["lambda_D"]
    C0s = np.linspace(5, 140, 200)
    SW["fig5a_C0"] = C0s
    for n in ("walking", "elbow", "wrist"):
        r = C.sweep("C0", C0s, "ng/ml", ["C_false", "sat"], ["ng/ml", "1"], kap_s=f"{kw[n]}[1/m]")
        SW[f"fig5a_cf_{n}"] = np.where(r["sat"] > 0.5, np.inf, r["C_false"])
    rcs = np.linspace(0, 0.8, 100)
    SW["fig5b_rc"] = rcs
    for n in ("walking", "elbow", "wrist"):
        SW[f"fig5b_{n}"] = C.sweep("r_c", rcs, "", ["art_FS"], kap_s=f"{kw[n]}[1/m]")["art_FS"] * 100
    mism = np.linspace(0, 0.2, 50)
    SW["fig7b_mism"] = mism
    r = C.sweep("mism", mism, "", ["resid_twin", "resid_twin_noX"], kap_s=f"{kw['wrist']}[1/m]", C0="50[ng/ml]")
    SW["fig7b_resid"], SW["fig7b_resid_noX"] = r["resid_twin"] * 100, r["resid_twin_noX"] * 100
    eg = np.linspace(-1e-3, 1e-3, 4001)
    SW["transfer_eps"], SW["transfer_dI"] = eg, C.sweep("eps_s", eg, "", ["dI_strain"])["dI_strain"]
    np.savez_compressed(os.path.join(DATA, "raw_sweeps.npz"), **SW)

    log("chain: 1800 s scenario")
    ex = ["t", "kap", "eps", "theta", "R_bio0", "R_sens", "R_ref", "Cb"]
    un = ["s", "1/m", "1", "1", "1", "1", "1", "ng/ml"]
    for tag, n in (("baseline", STRUCTS[0]), ("thin", STRUCTS[3]), ("island", STRUCTS[5])):
        r = C.transient("std_time", ex, un, epk=f"{epk[n]}[m]", mot=0, mism=0.05)
        np.savez_compressed(os.path.join(DATA, f"raw_time_{tag}.npz"), t=r["t"], kap=r["kap"], eps=r["eps"],
                            theta=r["theta"], R_bio0=r["R_bio0"], R_sens=r["R_sens"], R_ref=r["R_ref"], Cb_ngml=r["Cb"])
        log("  ", tag, "points:", len(r["t"]))
    M3 = {}
    for mi, n in ((1, "wrist"), (2, "elbow"), (3, "walking")):
        r = C.transient("std_motion", ["t", "kap", "eps"], ["s", "1/m", "1"], mot=mi)
        M3["t"], M3[f"kap_{n}"], M3[f"eps_{n}"] = r["t"], r["kap"], r["eps"]
    np.savez_compressed(os.path.join(DATA, "raw_motion.npz"), **M3)
    C.reset()
    C.model.save(os.path.join(HERE, "mos2_sensor_chain.mph"))
    return R


# ====================================================================== Debye 1D
def run_debye(client):
    model = bm.build_debye(client)
    with T("debye"):
        model.java.study("std1").run()
    ds = dataset_of_study(model, "std1")
    coords, vals = eval_nodes(model, "V", ds, "V")
    x = coords.ravel() * 1e9                            # nm
    vals = vals.reshape(-1, x.size)
    order = np.argsort(x)
    x = x[order]
    out = {}
    for i, c in enumerate((50, 100)):
        V = vals[i][order]
        lam = 1.0
        for _ in range(3):                              # fit ln V on [0, 4 lambda] (linear PB regime)
            sel = (x <= 4 * lam) & (V > 0)
            lam = -1.0 / np.polyfit(x[sel], np.log(V[sel]), 1)[0]
        out[f"lambda_{c}mM_nm"] = float(lam)
    model.save(os.path.join(HERE, "debye_1d.mph"))
    return out


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "mech", "comsol", "post"], default="all")
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--levels", default="1,2,3,4,5")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    raw_path = os.path.join(DATA, "raw.json")

    if a.stage in ("all", "mech", "comsol"):
        import mph
        client = mph.start(cores=a.cores)
        t0 = time.perf_counter()
        raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
        if a.stage in ("all", "mech", "comsol") and (a.stage != "comsol" or "mech" not in raw):
            raw["mech"] = run_mechanics(client, tuple(int(x) for x in a.levels.split(",")))
            json.dump(raw, open(raw_path, "w"), indent=1)
            print_convergence(raw["mech"])
        if a.stage in ("all", "comsol"):
            client.clear()
            raw["chain"] = run_chain(client, raw["mech"]["epk"])
            json.dump(raw, open(raw_path, "w"), indent=1)
            try:
                client.clear()
                raw["debye"] = run_debye(client)
            except Exception as e:
                log("debye model skipped:", e)
                raw["debye"] = None
            raw["runtimes"] = dict(T.t)
            raw["runtimes"]["total_s"] = time.perf_counter() - t0
            raw["comsol_version"] = str(client.version)
            json.dump(raw, open(raw_path, "w"), indent=1)

    if a.stage in ("all", "post"):
        py = json.load(open(os.path.join(ROOT, "figures", "results_python.json")))
        assemble(DATA, HERE, python_results=py)
        subprocess.run([sys.executable, os.path.join(ROOT, "make_figures.py"), "--source", "comsol"], check=True, cwd=ROOT)
        subprocess.run([sys.executable, os.path.join(HERE, "write_methods.py")], check=True, cwd=ROOT)
        log("wrote comsol/results.json, comsol/comparison.csv, comsol/figures, comsol/COMSOL_METHODS.md")


def print_convergence(mech):
    print("\nMesh convergence (baseline, kappa = 1 1/m)")
    print(f"{'level':>5} {'elements':>9} {'DOFs':>9} {'eps/k spring (um)':>18} {'d%':>7} {'eps/k tied (um)':>16} {'d%':>7} {'z_n (um)':>9} {'t (s)':>7}")
    for r in mech["convergence"]:
        print(f"{r['level']:>5} {r['n_elem']:>9} {r['dofs']:>9} {r['epk_spring_um']:>18.5f} {r['d_spring_pct']:>7.3f} "
              f"{r['epk_tied_um']:>16.5f} {r['d_tied_pct']:>7.3f} {r['zn_um']:>9.4f} {r['time_s']:>7.1f}")
    print("production level:", mech["production_level"])


if __name__ == "__main__":
    main()
