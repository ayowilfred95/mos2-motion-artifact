"""
Assemble comsol/results.json, comsol/data/curves.npz and comsol/comparison.csv from the raw COMSOL
exports written by run_all.py (comsol/data/raw.json + raw_*.npz).

Only post-processing happens here (Butterworth filtering, inversion through the calibration curve,
per-block error statistics, argmin/interp on COMSOL sweeps). Every input number is a COMSOL export.
"""
import csv
import json
import math
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

STRUCTS = ["Baseline: 25 um PI, 5 nm Al2O3", "Polymer encapsulation (1 um PMMA)", "Thinner substrate (12.5 um PI)",
           "Ultrathin substrate (2.5 um PET)", "Neutral-plane buried channel + extended gate (1 um tol.)",
           "Rigid island (4.2 % strain ratio)", "Prebent channel (1e-3 strain ratio)"]
PREBENT_RATIO = 1e-3          # published strain ratio, applied to the COMSOL baseline (no 2D geometry)
SEGMENTS = [(300, 600, "walking"), (600, 900, "wrist"), (900, 1200, "hold"), (1200, 1500, "elbow")]
FS_SAMPLE = 20.0              # Hz, output every 0.05 s
FC = 0.01                     # Hz, low-pass corner


# ---------------------------------------------------------------- post-processing primitives
def lpf(x, fc=FC, fs=FS_SAMPLE):
    sos = butter(2, fc / (fs / 2), output="sos")
    return sosfiltfilt(sos, x)


def invert(R, V_ov, dVT, K_D_nM):
    """Apparent concentration (nM) through the strain-free calibration curve; inf if >= full scale."""
    th = np.clip(R * V_ov / dVT, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(th < 1.0, K_D_nM * th / (1.0 - th), np.inf)


def seg_errors(t, C, C_ideal):
    res = {}
    for a, b, name in [(0, 300, "rest"), *SEGMENTS, (1500, 1800, "rest2")]:
        idx = (t >= a) & (t < b)
        e = C[idx] - C_ideal[idx]
        if np.any(~np.isfinite(e)):
            res[name] = (math.inf, math.inf)
        else:
            res[name] = (float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e))))
    return res


def scenario_readings(ts, ch):
    """C_ideal/C_raw/C_filt/C_twin/C_twin_filt (ng/mL) from an exported COMSOL time series."""
    ng = ch["ngml_to_nM"]
    inv = lambda R: invert(R, ch["V_ov"], ch["dVT_max_V"], ch["K_D_nM"]) / ng
    R_sens, R_ref, R_bio0 = ts["R_sens"], ts["R_ref"], ts["R_bio0"]
    R_twin = R_sens - R_ref
    return dict(C_ideal=inv(R_bio0), C_raw=inv(R_sens), C_filt=inv(lpf(R_sens)),
                C_twin=inv(R_twin), C_twin_filt=inv(lpf(R_twin)))


def filter_vs_frequency(eps_grid, dI_grid, epk, FS):
    """Fig. 7a: LPF residual vs motion frequency using the exported COMSOL strain->current transfer curve."""
    from scipy.interpolate import CubicSpline
    tr = CubicSpline(eps_grid, dI_grid)
    freqs = np.logspace(-3, 0.5, 40)
    uni, bi = [], []
    for f in freqs:
        tt = np.arange(0, max(600, 20 / f), 1 / FS_SAMPLE)
        rf = lpf(tr(epk * 40 * 0.5 * (1 - np.cos(2 * np.pi * f * tt))))
        uni.append(np.mean(rf[len(rf) // 4: -len(rf) // 4]) / FS * 100)
        rf2 = lpf(tr(epk * 40 * 0.5 * np.sin(2 * np.pi * f * tt)))
        bi.append(np.sqrt(np.mean(rf2[len(rf2) // 4: -len(rf2) // 4] ** 2)) / FS * 100)
    return freqs, np.array(uni), np.array(bi)


# ---------------------------------------------------------------- assembly
def _inf(x, sat):
    return math.inf if sat else float(x)


def assemble(data_dir, out_dir, python_results=None):
    raw = json.load(open(os.path.join(data_dir, "raw.json")))
    sw = dict(np.load(os.path.join(data_dir, "raw_sweeps.npz")))
    mech, ch = raw["mech"], raw["chain"]
    FS = ch["FS"]
    RES, CUR = {}, {}

    # motions (Stage 1-3)
    for name in ["walking", "elbow", "wrist"]:
        m = ch["motions"][name]
        RES[f"art_{name}_pct"] = m["dI"] * 100
        RES[f"eps_{name}_pct"] = m["eps"] * 100

    # dynamic scenario
    for key, struct in (("dyn_baseline", "baseline"), ("dyn_thin", "thin"), ("dyn_island", "island")):
        ts = dict(np.load(os.path.join(data_dir, f"raw_time_{struct}.npz")))
        rd = scenario_readings(ts, ch)
        RES[key] = {k: seg_errors(ts["t"], rd[k], rd["C_ideal"]) for k in ["C_raw", "C_filt", "C_twin", "C_twin_filt"]}
        if struct in ("baseline", "thin"):
            for k, v in rd.items():
                CUR[f"fig6_{struct}_{k}"] = v
            CUR[f"fig6_{struct}_t"] = ts["t"]
            CUR[f"fig6_{struct}_kappa"] = ts["kap"]
            CUR[f"fig6_{struct}_eps"] = ts["eps"]
            CUR[f"fig6_{struct}_C_true"] = ts["Cb_ngml"]

    RES["xtalk_floor_pct_FS"] = ch["xtalk_floor"] * 100

    # mitigation catalogue
    RES["mitigation_pct_FS"] = {}
    for n in STRUCTS:
        m = ch["mitigation"][n]
        RES["mitigation_pct_FS"][n] = dict(wrist=m["wrist"] * 100, elbow=m["elbow"] * 100,
                                           walking=m["walking"] * 100, eps_wrist_pct=m["eps_wrist"] * 100)

    # gate-bias optimum (fig. 8b)
    Vov, ratio = sw["gate_V"], sw["gate_ratio_rc0.3"] * 100
    imin = int(np.argmin(ratio))
    RES["gate_opt"] = dict(Vov=float(Vov[imin]), ratio=float(ratio[imin]),
                           ratio_at_0p3=float(np.interp(0.3, Vov, ratio)), ratio_at_0p1=float(np.interp(0.1, Vov, ratio)))

    RES["zn_um"] = mech["zn_m"] * 1e6
    RES["eps_per_kappa_um"] = mech["epk"]["Baseline: 25 um PI, 5 nm Al2O3"] * 1e6
    for k in ["lambda_D_nm", "dVT_sweat_mV", "FS_pct", "R_phys_low_pct", "R_phys_high_pct", "GF_ch_0p3", "GF_c",
              "GF_tot_0p3", "K_D_ngml", "k_on", "tau_100ngml_s"]:
        RES[k] = ch["scalars"][k]
    for name in ["wrist", "elbow", "walking", "hold"]:
        m = ch["motions"][name]
        RES[f"lin_false_{name}_ngml"] = m["lin_false30"]
        RES[f"exact_false_{name}_at30"] = _inf(m["C_false30"], m["sat30"])
        RES[f"exact_false_{name}_at50"] = _inf(m["C_false50"], m["sat50"])
    RES["check_john_sum_meV"] = ch["scalars"]["john_sum"]
    RES["check_debye_100mM_nm"] = raw["debye"]["lambda_100mM_nm"] if raw.get("debye") else ch["scalars"]["lambda_100mM_formula_nm"]
    RES["check_datye_channel_only_0p7_Vov0p145"] = ch["scalars"]["datye"] * 100
    RES["GF_ch_0p145"] = ch["scalars"]["GF_ch_0p145"]
    RES["check_li_29pct"] = ch["scalars"]["li_xt"]
    RES["check_symmetric_stack_strain"] = mech["sym_epk_m"]

    # key order and key set identical to the Python results file
    if python_results is not None:
        missing = set(python_results) ^ set(RES)
        if missing:
            raise KeyError(f"key mismatch vs Python results: {sorted(missing)}")
        RES = {k: RES[k] for k in python_results}

    # ------------------------------------------------ curves for figs 3-8
    for k, v in sw.items():
        CUR[k] = v
    ts3 = dict(np.load(os.path.join(data_dir, "raw_motion.npz")))
    for k, v in ts3.items():
        CUR[f"fig3_{k}"] = v
    f, uni, bi = filter_vs_frequency(sw["transfer_eps"], sw["transfer_dI"],
                                     mech["epk"]["Baseline: 25 um PI, 5 nm Al2O3"], FS)
    CUR["fig7a_freqs"], CUR["fig7a_uni"], CUR["fig7a_bi"] = f, uni, bi
    CUR["fig8a_names"] = np.array(STRUCTS)
    for k, band in (("wrist", "w"), ("elbow", "e"), ("walking", "walk")):
        CUR[f"fig8a_{band}"] = np.array([RES["mitigation_pct_FS"][n][k] for n in STRUCTS])
    CUR["FS"] = np.array(FS)
    CUR["eps_per_kappa"] = np.array(mech["epk"]["Baseline: 25 um PI, 5 nm Al2O3"])

    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    json.dump(RES, open(os.path.join(out_dir, "results.json"), "w"), indent=1, default=str)
    np.savez_compressed(os.path.join(out_dir, "data", "curves.npz"), **CUR)
    if python_results is not None:
        write_comparison(RES, python_results, os.path.join(out_dir, "comparison.csv"))
    return RES


# ---------------------------------------------------------------- comparison table
def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (list, tuple)) and len(v) == 2 and not isinstance(v[0], (list, dict)):
            out[key + ".rms"] = v[0]
            out[key + ".max"] = v[1]
        else:
            out[key] = v
    return out


def _f(x):
    if isinstance(x, str):
        return float(x.replace("Infinity", "inf"))
    return float(x)


def write_comparison(res, py, path):
    fc, fp = flatten(res), flatten(py)
    rows = []
    for k in fp:
        c, p = _f(fc[k]), _f(fp[k])
        if math.isinf(c) and math.isinf(p):
            diff, note = "", "both saturated"
        elif math.isinf(c) or math.isinf(p):
            diff, note = "", "saturation differs"
        elif p == 0:
            diff, note = ("0" if c == 0 else ""), ("" if c == 0 else "reference is zero")
        else:
            diff, note = f"{(c - p) / abs(p) * 100:.3f}", ""
            if abs((c - p) / p) > 0.05:
                note = "> 5 %: see COMSOL_METHODS.md"
        rows.append([k, repr(c), repr(p), diff, note])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "comsol", "python", "percent_difference", "note"])
        w.writerows(rows)
    return rows
