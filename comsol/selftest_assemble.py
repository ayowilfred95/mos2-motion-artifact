"""
Self-test of the post-processing path (no COMSOL needed).

Writes raw exports in exactly the format run_all.py produces, but filled from the reference Python
model, runs assemble.assemble() on them and checks that the result reproduces
figures/results_python.json. This validates filtering, inversion, block statistics, key layout and
the comparison table. Output goes to a temporary directory, never to comsol/results.json.

Usage: python comsol/selftest_assemble.py [outdir]
"""
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import motion_artifact_sim as ms                          # noqa: E402
from assemble import STRUCTS, assemble, flatten, _f       # noqa: E402

P = ms.P


def python_raw(d):
    os.makedirs(d, exist_ok=True)
    L, i, eta = ms.stack()
    s_base, zn, _ = ms.channel_strain_per_curvature(L, i, eta)
    cat = ms.mitigation_catalogue()
    FS = ms.dVT_max_sweat() / P["V_ov"]
    ng = ms.NGML_TO_NM
    dI = lambda e, V=P["V_ov"]: float(ms.rel_current_change_strain(e, V))
    motions = {}
    for n in ["wrist", "elbow", "walking", "hold"]:
        e = s_base / ms.MOTIONS[n]["R_min"]
        C0 = 30 * ng
        S = FS * P["K_D_nM"] / (C0 + P["K_D_nM"]) ** 2
        c30, c50 = ms.false_concentration_ngml(e, 30.0), ms.false_concentration_ngml(e, 50.0)
        motions[n] = dict(eps=e, dI=dI(e), lin_false30=dI(e) / S / ng,
                          C_false30=float(c30) if np.isfinite(c30) else 0.0, sat30=bool(~np.isfinite(c30)),
                          C_false50=float(c50) if np.isfinite(c50) else 0.0, sat50=bool(~np.isfinite(c50)))
    e_w = s_base / 25e-3
    R0 = ms.response(50 * ng)
    P_rc = P["r_c"]; P["r_c"] = 0.0
    datye = dI(0.007, 0.145)
    P["r_c"] = P_rc
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
        mitigation={n: dict(wrist=dI(cat[n] / 25e-3) / FS, elbow=dI(cat[n] / 40e-3) / FS,
                            walking=dI(cat[n] / 300e-3) / FS, eps_wrist=cat[n] / 25e-3) for n in STRUCTS})
    mech = dict(zn_m=zn, epk={n: cat[n] for n in STRUCTS},
                sym_epk_m=float(ms.channel_strain_per_curvature(*ms.stack(("PI", 25e-6), ("PI", 25e-6)))[0]))
    json.dump(dict(mech=mech, chain=chain, debye=None), open(os.path.join(d, "raw.json"), "w"), indent=1)

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
    np.savez_compressed(os.path.join(d, "raw_sweeps.npz"), **SW)

    for tag, epk in (("baseline", s_base), ("thin", cat[STRUCTS[3]]), ("island", s_base * 0.042)):
        t = np.arange(0, 1800, 0.05)
        Ct = np.where(t < 300, 30.0, 100.0) * ng
        th = ms.binding_kinetics(t, Ct)
        out = ms.scenario(eps_per_kappa=epk)
        eps = out["eps"]
        R_bio0 = ms.dVT_max_sweat() * th / P["V_ov"]
        art = ms.rel_current_change_strain(eps, P["V_ov"])
        np.savez_compressed(os.path.join(d, f"raw_time_{tag}.npz"), t=t, kap=out["kappa"], eps=eps, theta=th,
                            R_bio0=R_bio0, R_sens=R_bio0 * (1 + P["kappa_x"] * eps) + art, R_ref=art * 1.05,
                            Cb_ngml=Ct / ng)
    t3 = np.linspace(0, 4, 2000)
    M3 = {"t": t3}
    for n in ("wrist", "elbow", "walking"):
        M3[f"kap_{n}"] = ms.curvature(t3, ms.MOTIONS[n]); M3[f"eps_{n}"] = s_base * M3[f"kap_{n}"]
    np.savez_compressed(os.path.join(d, "raw_motion.npz"), **M3)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="selftest_")
    data = os.path.join(out, "data")
    python_raw(data)
    py = json.load(open(os.path.join(ROOT, "figures", "results_python.json")))
    res = assemble(data, out, python_results=py)
    fr, fp = flatten(json.loads(json.dumps(res, default=str))), flatten(py)
    worst, bad = 0.0, []
    for k in fp:
        c, p = _f(fr[k]), _f(fp[k])
        if math.isinf(c) or math.isinf(p):
            ok = math.isinf(c) and math.isinf(p)
            d = 0.0 if ok else math.inf
        else:
            d = abs(c - p) / max(abs(p), 1e-30)
        worst = max(worst, d)
        if d > 1e-6:
            bad.append((k, c, p, d))
    print(f"{len(fp)} quantities compared; worst relative difference {worst:.3e}")
    for b in bad:
        print("  MISMATCH", b)
    print("comparison table:", os.path.join(out, "comparison.csv"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
