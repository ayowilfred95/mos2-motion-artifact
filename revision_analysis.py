"""
Sensor-chain analyses added for the reviewer's comments (pre-submission review, 2026-10-22).
Uses the equations of motion_artifact_sim.py and the Elmer strain per curvature (elmer/data/raw_mech.json).

  R2  twin-sensor derivation: worked numbers at the wrist peak, 50 ng/mL, 5 % mismatch
  R3  cross-talk floor vs true concentration (expression + figure elmer/figures/fig7c_floor.png)
  R4  zero-phase (offline, forward-backward) vs causal (real-time) 2nd-order Butterworth, fc = 10 mHz
  R6  worked examples current -> occupancy -> concentration for walking and wrist flexion at 30 ng/mL
  R9  dynamic scenario for the 12.5 um PI substrate (and all designs with the causal filter)
  R13 sensitivity to wrist bending radius and strain-transfer efficiency
  R15 robustness of the headline numbers to the influential electrochemical parameters

Writes elmer/data/revision_analysis.json. Usage: python revision_analysis.py
"""
import copy
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt   # noqa: E402

import motion_artifact_sim as m   # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
EPK = json.load(open(os.path.join(ROOT, "elmer", "data", "raw_mech.json")))["mech"]["epk"]
DESIGNS = {"baseline": EPK["Baseline: 25 um PI, 5 nm Al2O3"], "pi_12p5": EPK["Thinner substrate (12.5 um PI)"],
           "pet_2p5": EPK["Ultrathin substrate (2.5 um PET)"], "island_fe": EPK["Rigid island (4.2 % strain ratio)"]}
P0 = copy.deepcopy(m.P)
FS_PCT = lambda: m.dVT_max_sweat() / m.P["V_ov"] * 100       # full-scale response, % of I0
NG = m.NGML_TO_NM


def theta(c_ngml):
    return m.theta_eq(c_ngml * NG)


# ------------------------------------------------------------------ R6 worked examples
def worked_example(eps, c0=30.0):
    th = theta(c0)
    fs = m.dVT_max_sweat() / m.P["V_ov"]
    r_true = fs * th
    xt = 1 + m.P["kappa_x"] * eps
    art = m.rel_current_change_strain(eps, m.P["V_ov"])
    r_meas = r_true * xt + art
    th_app = r_meas / fs
    c_app = m.K_D_ngml * th_app / (1 - th_app) if th_app < 1 else float("inf")
    return dict(C0_ngml=c0, eps_pct=eps * 100, theta=th, dVT_mV=m.dVT_max_sweat() * 1e3, FS_pct=fs * 100,
                R_true_pct=r_true * 100, crosstalk_factor=xt, strain_artifact_pct=art * 100, R_meas_pct=r_meas * 100,
                theta_app=th_app, C_app_ngml=c_app, C_false_ngml=c_app - c0 if np.isfinite(c_app) else float("inf"),
                saturated=bool(th_app >= 1),
                linearised_false_ngml=art / (fs * m.K_D_ngml / (c0 + m.K_D_ngml) ** 2))


m.K_D_ngml = m.P["K_D_nM"] / NG


# ------------------------------------------------------------------ R2 twin derivation
def twin_example(eps, c0=50.0, delta=0.05):
    fs = m.dVT_max_sweat() / m.P["V_ov"]
    th = theta(c0)
    art = m.rel_current_change_strain(eps, m.P["V_ov"])
    sens = fs * th * (1 + m.P["kappa_x"] * eps) + art           # sensing channel, eq. (10)
    ref = (1 + delta) * art                                       # reference channel: no binding, gain 1+delta
    twin = sens - ref
    ideal = fs * th
    xtalk = fs * th * m.P["kappa_x"] * eps
    mism = -delta * art
    return dict(C0_ngml=c0, eps_pct=eps * 100, delta=delta, FS_pct=fs * 100, theta=th,
                R_sense_pct=sens * 100, R_ref_pct=ref * 100, R_twin_pct=twin * 100, R_ideal_pct=ideal * 100,
                crosstalk_residual_pctFS=xtalk / fs * 100, mismatch_residual_pctFS=mism / fs * 100,
                total_residual_pctFS=(twin - ideal) / fs * 100)


# ------------------------------------------------------------------ R3 floor vs concentration
def floor_pctFS(c_ngml, eps):
    """Cross-talk floor of twin subtraction (delta = 0) as % of full scale: 100 * theta(C) * kappa_x * eps."""
    return 100 * theta(c_ngml) * m.P["kappa_x"] * eps


# ------------------------------------------------------------------ R4 / R9 dynamic scenario with filter choice
SEG_PAPER = [(300, 600, "walking"), (600, 900, "wrist"), (900, 1200, "hold"), (1200, 1500, "elbow")]
SEG_FOREARM = [(0, 900, "walking_a"), (900, 1800, "walking_b")]     # forearm: walking only, step at 900 s


def scenario(epk, causal=False, fs=20.0, T=1800.0, mismatch=0.05, fc=0.01, seg=SEG_PAPER, t_step=300.0,
             motion=True):
    """mismatch = delta = (reference gain - sensing gain) / sensing gain (Table 5 used +0.05; -0.05 is the
    worst case, where mismatch and cross-talk residuals add)."""
    t = np.arange(0, T, 1.0 / fs)
    c_true = np.where(t < t_step, 30.0, 100.0) * NG
    th = m.binding_kinetics(t, c_true)
    kap = np.zeros_like(t)
    for a, b, name in (seg if motion else []):
        idx = (t >= a) & (t < b)
        kap[idx] = m.curvature(t[idx] - a, m.MOTIONS[name.split("_")[0]])
    eps = epk * kap
    r_bio = m.dVT_max_sweat() * th / m.P["V_ov"]
    r_sens = r_bio * (1 + m.P["kappa_x"] * eps) + m.rel_current_change_strain(eps, m.P["V_ov"])
    r_twin = r_sens - m.rel_current_change_strain(eps, m.P["V_ov"]) * (1 + mismatch)
    sos = butter(2, fc / (fs / 2), output="sos")
    if causal:   # real-time: forward filtering only, initial state at steady state of the first sample
        f = lambda x: sosfilt(sos, x, zi=sosfilt_zi(sos) * x[0])[0]
    else:        # offline: forward-backward (zero phase), scipy default odd-extension padding
        f = lambda x: sosfiltfilt(sos, x)
    c_of = lambda r: m.invert_response(r) / NG
    out = dict(t=t, C_ideal=c_of(r_bio), C_raw=c_of(r_sens), C_filt=c_of(f(r_sens)),
               C_twin=c_of(r_twin), C_twin_filt=c_of(f(r_twin)), segments=seg)
    return out


def errors(out):
    return {k: m.seg_errors(out, k) for k in ("C_raw", "C_filt", "C_twin", "C_twin_filt")}


def motion_only_errors(epk, causal, mismatch, seg=SEG_PAPER, t_step=300.0):
    """Error caused by motion alone: processed reading with motion minus the same processing without motion
    (removes the filter's own response to the cortisol step). RMS / max per block, ng/mL."""
    a = scenario(epk, causal=causal, mismatch=mismatch, seg=seg, t_step=t_step)
    b = scenario(epk, causal=causal, mismatch=mismatch, seg=seg, t_step=t_step, motion=False)
    res = {}
    for k in ("C_filt", "C_twin_filt"):
        res[k] = {}
        for lo, hi, name in seg:
            idx = (a["t"] >= lo) & (a["t"] < hi)
            e = a[k][idx] - b[k][idx]
            res[k][name] = ([float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))]
                            if np.all(np.isfinite(e)) else ["inf", "inf"])
    return res


def step_response_delay(fs=20.0, fc=0.01):
    """Delay of the 2nd-order Butterworth: causal group delay at DC, and 50 % rise time of a step for both."""
    sos = butter(2, fc / (fs / 2), output="sos")
    t = np.arange(0, 600, 1 / fs)
    x = (t >= 300).astype(float)
    yc = sosfilt(sos, x)
    yz = sosfiltfilt(sos, x)
    t50_c = t[np.argmax(yc >= 0.5)] - 300
    t50_z = t[np.argmax(yz >= 0.5)] - 300
    t10_z, t90_z = t[np.argmax(yz >= 0.1)] - 300, t[np.argmax(yz >= 0.9)] - 300
    return dict(causal_50pct_delay_s=float(t50_c), zero_phase_50pct_delay_s=float(t50_z),
                zero_phase_10_90_s=[float(t10_z), float(t90_z)],
                causal_dc_group_delay_s=float(np.sqrt(2) / (2 * np.pi * fc)),
                padtype="odd", padlen=int(3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))))


# ------------------------------------------------------------------ R13 / R15 sensitivity
def headline(epk_base=DESIGNS["baseline"], R_wrist=25e-3, eta_scale=1.0):
    e_w = epk_base * eta_scale / R_wrist
    e_walk = epk_base * eta_scale / 0.3
    fs = FS_PCT()
    art_w = m.rel_current_change_strain(e_w, m.P["V_ov"]) * 100 / fs * 100        # % of full scale
    contact_share = (m.P["r_c"] * m.GF_contact()) / m.GF_total_small_signal(m.P["V_ov"])
    return dict(wrist_artifact_pctFS=art_w, filter_offset_pctFS=art_w / 2,
                walking_false_at30_ngml=float(m.false_concentration_ngml(e_walk, 30.0)),
                floor_at50_pctFS=floor_pctFS(50.0, e_w), contact_share=contact_share,
                wrist_saturates_at30=bool(not np.isfinite(m.false_concentration_ngml(e_w, 30.0))))


def with_params(**kw):
    m.P.clear(); m.P.update(copy.deepcopy(P0)); m.P.update(kw)
    m.P["k_on"] = m.P["k_off"] / (m.P["K_D_nM"] * 1e-9)
    m.K_D_ngml = m.P["K_D_nM"] / NG


def robustness():
    cases = {"dVT_vac_V": [0.027, 0.1, 0.3, 0.78], "K_D_nM": [10.0, 50.0, 200.0], "kappa_x": [0.0, 42.5, 85.0, 170.0],
             "r_c": [0.0, 0.3, 0.6], "I_sweat_M": [0.02, 0.05, 0.1], "d_charge_nm": [1.0, 2.0, 3.0],
             "GF_max": [100.0, 200.0, 300.0], "V_ov": [0.1, 0.3, 0.6]}
    out = {}
    for k, vals in cases.items():
        out[k] = []
        for v in vals:
            with_params(**{k: v})
            out[k].append(dict(value=v, **headline()))
    with_params()
    return out


def kd_bounds():
    """Appendix D: the two K_D values reported for the aptamer of [1] (30 pM on the FET, 500 nM in solution)."""
    out = []
    for kd in (0.03, 50.0, 500.0):
        with_params(K_D_nM=kd)
        th = lambda c: c / (c + m.K_D_ngml)
        e_pet = DESIGNS["pet_2p5"] / 25e-3
        ta = th(100.0) * (1 + m.P["kappa_x"] * e_pet)      # matched twin, peak wrist strain, before filtering
        twin = m.K_D_ngml * ta / (1 - ta) - 100.0 if ta < 1 else float("inf")
        out.append(dict(K_D_nM=kd, theta_phys_range=[th(8.16), th(141.7)], pet_twin_peak_err_at100_ngml=twin, **headline()))
    with_params()
    return out


MC_RANGES = {   # joint uncertainty: (low, high, scale); same bounds as Tables D1 and D2, plus beta and delta
    "dVT_vac_V": (0.027, 0.78, "log"), "K_D_nM": (10.0, 500.0, "log"), "kappa_x": (0.0, 170.0, "lin"),
    "r_c": (0.0, 0.6, "lin"), "dPhiB_per_strain_eV": (-8.25, -2.75, "lin"), "I_sweat_M": (0.02, 0.1, "lin"),
    "d_charge_nm": (1.0, 3.0, "lin"), "GF_max": (100.0, 300.0, "lin"),
    "R_wrist_m": (20e-3, 40e-3, "lin"), "eta": (0.5, 1.0, "lin"), "delta": (-0.05, 0.05, "lin")}


def monte_carlo(n=10000, seed=1):
    """Appendix D: all uncertain inputs drawn together; medians, 95 % intervals and Spearman ranking."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    draws = {k: (np.exp(rng.uniform(np.log(lo), np.log(hi), n)) if sc == "log" else rng.uniform(lo, hi, n))
             for k, (lo, hi, sc) in MC_RANGES.items()}
    out = {k: np.empty(n) for k in ("wrist_pctFS", "walking_false_at30_ngml", "pet_twin_err_at100_ngml",
                                     "base_twin_err_at100_ngml", "contact_share")}
    model_keys = [k for k in MC_RANGES if k not in ("R_wrist_m", "eta", "delta")]
    for i in range(n):
        with_params(**{k: float(draws[k][i]) for k in model_keys})
        sc = draws["eta"][i] / 0.8                                   # Elmer eps/kappa includes eta = 0.8
        h = headline(R_wrist=draws["R_wrist_m"][i], eta_scale=sc)
        out["wrist_pctFS"][i] = h["wrist_artifact_pctFS"]
        out["walking_false_at30_ngml"][i] = h["walking_false_at30_ngml"]
        out["contact_share"][i] = h["contact_share"]
        fs = m.dVT_max_sweat() / m.P["V_ov"]
        th = theta(100.0)
        for key, d in (("pet_twin_err_at100_ngml", "pet_2p5"), ("base_twin_err_at100_ngml", "baseline")):
            eps = DESIGNS[d] * sc / draws["R_wrist_m"][i]
            res = fs * th * m.P["kappa_x"] * eps - draws["delta"][i] * m.rel_current_change_strain(eps, m.P["V_ov"])
            ta = (fs * th + res) / fs                                # twin output read through the calibration
            out[key][i] = abs(m.K_D_ngml * ta / (1 - ta) - 100.0) if 0 <= ta < 1 else np.inf
    with_params()
    summ = {}
    for k, v in out.items():
        fin = np.isfinite(v)
        vs = np.sort(v)                                              # saturated draws (inf) sort last
        summ[k] = dict(median=float(vs[n // 2]), p2_5=float(vs[int(0.025 * n)]), p97_5=float(vs[int(0.975 * n)]),
                       frac_saturated=float(1 - fin.mean()),
                       spearman={p: float(abs(spearmanr(draws[p], np.where(fin, v, 1e9))[0])) for p in MC_RANGES})
    summ["P_pet_twin_below_1ngml"] = float(np.mean(out["pet_twin_err_at100_ngml"] < 1.0))
    summ["P_base_twin_below_1ngml"] = float(np.mean(out["base_twin_err_at100_ngml"] < 1.0))
    summ["P_wrist_above_10pctFS"] = float(np.mean(out["wrist_pctFS"] > 10.0))
    summ["P_contact_share_above_half"] = float(np.mean(out["contact_share"] > 0.5))
    summ["n"] = n
    return summ


def sensitivity_motion():
    rows = []
    for R in (20e-3, 25e-3, 30e-3, 35e-3, 40e-3):
        for eta in (0.5, 0.8, 1.0):
            h = headline(R_wrist=R, eta_scale=eta / 0.8)     # Elmer eps/kappa includes eta = 0.8
            rows.append(dict(R_mm=R * 1e3, eta=eta, **h))
    return rows


def fig_floor(path):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})   # as make_figures.py
    c = np.logspace(0, np.log10(300), 300)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for name, R, col in (("wrist (R = 25 mm)", 25e-3, "#c0392b"), ("elbow (R = 40 mm)", 40e-3, "#2980b9"),
                         ("walking (1/Δκ = 300 mm)", 0.3, "#27ae60")):
        e = DESIGNS["baseline"] / R
        ax.plot(c, floor_pctFS(c, e), color=col, lw=1.2, label=name)
    ax.axvspan(8.16, 141.7, color="#2ecc71", alpha=0.12, lw=0)
    ax.axvline(50, color="#555", ls=":", lw=0.8)
    ax.text(53, 0.55, "50 ng/mL (Fig. 7b)", fontsize=6, color="#555")
    ax.text(9, 2.95, "physiological range", fontsize=6, color="#1e8449")
    ax.set_xscale("log"); ax.set_ylim(0, 3.3)
    ax.set_xlabel("True cortisol (ng/mL)"); ax.set_ylabel("Cross-talk floor\n(% of full scale)")
    ax.legend(frameon=False, fontsize=6, loc="center left"); ax.grid(alpha=0.3, lw=0.4, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=1200, bbox_inches="tight"); plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k not in ("t",) and not isinstance(v, np.ndarray)}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else "inf"
    if isinstance(o, np.bool_):
        return bool(o)
    return o


if __name__ == "__main__":
    with_params()
    ew = DESIGNS["baseline"] / 25e-3
    res = {}
    res["R6_worked_walking"] = worked_example(DESIGNS["baseline"] / 0.3)
    res["R6_worked_wrist"] = worked_example(ew)
    res["R2_twin_example"] = twin_example(ew)
    res["R3_floor_expression"] = "floor(%FS) = 100 * theta(C) * kappa_x * eps_ch, theta = C/(C+K_D)"
    res["R3_floor_wrist"] = {f"{c:g}": floor_pctFS(c, ew) for c in (8.16, 30, 50, 100, 141.7)}
    res["R3_floor_wrist_limit_theta1"] = 100 * m.P["kappa_x"] * ew
    res["R4_filter"] = step_response_delay()
    dyn = {}
    for name, epk in DESIGNS.items():
        dyn[name] = {"zero_phase": errors(scenario(epk)), "causal": errors(scenario(epk, causal=True))}
    res["R4_R9_dynamic"] = dyn
    worst = {}
    for name, epk in DESIGNS.items():
        o = scenario(epk, mismatch=-0.05)
        worst[name] = {k: m.seg_errors(o, k) for k in ("C_twin", "C_twin_filt")}
    res["R2_worst_case_mismatch_minus5"] = worst
    res["R4_R9_motion_only"] = {name: {f"{'causal' if c else 'zero_phase'}_delta{d:+.2f}":
                                       motion_only_errors(epk, c, d) for c in (False, True) for d in (0.05, -0.05)}
                                for name, epk in DESIGNS.items()}
    res["R2_matched_delta0"] = {name: {k: m.seg_errors(o, k) for k in ("C_twin", "C_twin_filt")}
                                for name, epk in DESIGNS.items() for o in [scenario(epk, mismatch=0.0)]}
    # largest |delta| (worst sign) that keeps the twin + zero-phase LPF maximum error <= 1 ng/mL in every flexion block
    tol = {}
    for name, epk in DESIGNS.items():
        ok = None
        for d in np.arange(0.0, 0.201, 0.0025):
            o = scenario(epk, mismatch=-d)
            e = m.seg_errors(o, "C_twin_filt")
            if max(e[b][1] for b in ("wrist", "hold", "elbow")) <= 1.0:
                ok = float(d)
            else:
                break
        tol[name] = ok
    res["R2_mismatch_tolerance_for_1ngml"] = tol
    res["R9_forearm_walking_only"] = {name: {f"{'causal' if c else 'zero_phase'}":
                                             motion_only_errors(epk, c, -0.05, seg=SEG_FOREARM, t_step=900.0)
                                             for c in (False, True)} for name, epk in DESIGNS.items()}
    res["R13_motion_sensitivity"] = sensitivity_motion()
    res["R15_robustness"] = robustness()
    res["KD_bounds"] = kd_bounds()
    res["monte_carlo"] = monte_carlo()
    fig_floor(os.path.join(ROOT, "elmer", "figures", "fig7c_floor.png"))
    json.dump(clean(res), open(os.path.join(ROOT, "elmer", "data", "revision_analysis.json"), "w"), indent=1)
    print("wrote elmer/data/revision_analysis.json and elmer/figures/fig7c_floor.png")
