"""
Motion-artifact simulation chain for a flexible liquid-gated MoS2 FET cortisol sensor.

Stage 1  body motion  -> time-varying patch curvature kappa(t)
Stage 2  curvature    -> strain in MoS2 channel (multilayer neutral-plane mechanics + vdW transfer)
Stage 3  strain       -> channel piezoresistance (gate-dependent gauge factor), Schottky barrier shift,
                         and strain-dependent biomarker sensitivity (cross-talk)
Stage 4  compact liquid-gated FET model with aptamer Langmuir kinetics and Debye screening in sweat
Stage 5  strain-equivalent false cortisol concentration; mitigation sweeps
Stage 6  dynamic (time-domain) analysis: low-pass filtering vs twin-sensor subtraction

All parameters are collected in PARAMS with a source tag. Units: SI unless stated.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt
import json

kT_eV = 0.025852   # thermal energy at 300 K (eV)
MW_CORT = 362.46   # g/mol
NGML_TO_NM = 1e3 / MW_CORT * 1e0   # 1 ng/mL = 1e-6 g/L /362.46 g/mol = 2.759e-9 M = 2.759 nM
NGML_TO_NM = 1e-6 / MW_CORT * 1e9  # = 2.7589 nM per ng/mL

PARAMS = {
    # ---- Layer stack (bottom -> top). E in Pa, t in m ----
    "E_PI": 2.5e9, "E_PET": 4.0e9, "E_Au": 79e9, "E_MoS2": 270e9, "E_Al2O3": 180e9, "E_PMMA": 3.0e9,
    "t_MoS2": 0.65e-9, "t_Au": 50e-9, "t_Al2O3": 5e-9,
    # ---- van der Waals strain transfer efficiency (Vladimirov 2025: encapsulation ~doubles transfer) ----
    "eta_bare": 0.5, "eta_thinALD": 0.8, "eta_polymer": 1.0,
    # ---- Stage 3 coefficients ----
    "GF_min": 15.0, "GF_max": 200.0, "V_gf": 0.12, "V0_gf": 0.05,   # gate-dependent channel gauge factor (Datye 2022)
    "dPhiB_per_strain_eV": -5.5,        # -55 meV per 1 % strain (John 2020, bilayer)
    "kappa_x": 85.0,                    # sensitivity cross-talk: 29 % per 0.34 % strain (Li 2019)
    "r_c": 0.30,                        # fraction of total resistance in the two Schottky contacts at zero strain
    # ---- Stage 4 electrochemistry / binding ----
    "K_D_nM": 50.0, "k_off": 2e-3,      # cortisol-aptamer affinity (bracketed by nM-uM literature values)
    "dVT_vac_V": 0.100,                 # cortisol-induced threshold shift, unscreened (Boschetto 2023: 27-780 mV)
    "d_charge_nm": 2.0,                 # effective distance of redistributed aptamer charge from surface
    "I_sweat_M": 0.050,                 # sweat ionic strength (~50 mM)
    "V_ov": 0.30,                       # gate overdrive V_G - V_T (V)
    # ---- Physiology ----
    "C_phys_low_ngml": 8.16, "C_phys_high_ngml": 141.7,   # Aleksandrova 2026
}
P = PARAMS
P["k_on"] = P["k_off"] / (P["K_D_nM"] * 1e-9)   # M^-1 s^-1

# ---------------------------------------------------------------- Stage 2: mechanics
def neutral_plane(layers):
    """layers: list of (E, t) from bottom to top. Returns z_n (m) and z-centroids (m)."""
    z = 0.0; num = 0.0; den = 0.0; cents = []
    for E, t in layers:
        zc = z + t / 2; cents.append(zc)
        num += E * t * zc; den += E * t; z += t
    return num / den, cents

def stack(substrate=("PI", 25e-6), cap=("Al2O3", 5e-9), buried=False):
    """Return layer list and index of MoS2 layer. cap: ('Al2O3', t) | ('PMMA', t) | ('none',0)."""
    Es = P["E_PI"] if substrate[0] == "PI" else P["E_PET"]
    layers = [(Es, substrate[1]), (P["E_Au"], P["t_Au"]), (P["E_MoS2"], P["t_MoS2"])]
    if cap[0] == "Al2O3":
        layers.append((P["E_Al2O3"], cap[1])); eta = P["eta_thinALD"]
    elif cap[0] == "PMMA":
        layers.append((P["E_Al2O3"], P["t_Al2O3"])); layers.append((P["E_PMMA"], cap[1])); eta = P["eta_polymer"]
    elif cap[0] == "PI":   # symmetric buried channel (remote / extended gate architecture)
        layers.append((P["E_Al2O3"], P["t_Al2O3"])); layers.append((P["E_PI"], cap[1])); eta = P["eta_polymer"]
    else:
        eta = P["eta_bare"]
    return layers, 2, eta

def channel_strain_per_curvature(layers, i_mos2, eta, misalign=0.0):
    """Strain in MoS2 per unit curvature (dimensionless per 1/m): eps = eta*(z_MoS2 - z_n)*kappa."""
    z_n, cents = neutral_plane(layers)
    return eta * (cents[i_mos2] - z_n + misalign), z_n, cents

# ---------------------------------------------------------------- Stage 3: strain -> electrical
def GF_channel(V_ov):
    return P["GF_min"] + (P["GF_max"] - P["GF_min"]) * np.exp(-(V_ov - P["V0_gf"]) / P["V_gf"])

def GF_contact():
    return -P["dPhiB_per_strain_eV"] / kT_eV        # ~213

def rel_current_change_strain(eps, V_ov):
    """Two-resistor model: R = R_ch exp(-GF eps) + R_c exp(dPhiB eps/kT). Returns dI/I = 1/R - 1."""
    r_c = P["r_c"]; r_ch = 1 - r_c
    R_ch = r_ch * np.exp(-GF_channel(V_ov) * eps)      # reduces to 1 - GF*eps for small strain
    R_c = r_c * np.exp(P["dPhiB_per_strain_eV"] * eps / kT_eV)
    R = R_ch + R_c
    return 1.0 / R - 1.0

def GF_total_small_signal(V_ov):
    return (1 - P["r_c"]) * GF_channel(V_ov) + P["r_c"] * GF_contact()

# ---------------------------------------------------------------- Stage 4: binding + FET
def debye_length_nm(I_M):
    return 0.304 / np.sqrt(I_M)     # nm, monovalent electrolyte at 25 C

def dVT_max_sweat():
    return P["dVT_vac_V"] * np.exp(-P["d_charge_nm"] / debye_length_nm(P["I_sweat_M"]))

def theta_eq(C_nM):
    return C_nM / (C_nM + P["K_D_nM"])

def response(C_nM, eps=0.0, V_ov=None):
    """Normalized biomarker response dI/I (linear regime): dVT/V_ov * (1 + kappa_x eps)."""
    V_ov = P["V_ov"] if V_ov is None else V_ov
    return dVT_max_sweat() * theta_eq(C_nM) / V_ov * (1.0 + P["kappa_x"] * eps)

def invert_response(R, V_ov=None):
    """Apparent concentration (nM) read back through the strain-free calibration curve. inf if > full scale."""
    V_ov = P["V_ov"] if V_ov is None else V_ov
    th = np.clip(R * V_ov / dVT_max_sweat(), 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(th < 1.0, P["K_D_nM"] * th / (1.0 - th), np.inf)
    return C

def false_concentration_ngml(eps, C0_ngml=50.0, V_ov=None):
    """Strain-equivalent false cortisol: apparent - true, at operating point C0 (ng/mL)."""
    V_ov = P["V_ov"] if V_ov is None else V_ov
    C0 = C0_ngml * NGML_TO_NM
    R_true = response(C0, 0.0, V_ov)
    R_meas = R_true * (1 + P["kappa_x"] * eps) + rel_current_change_strain(eps, V_ov)
    C_app = invert_response(R_meas, V_ov)
    return (C_app - C0) / NGML_TO_NM

# ---------------------------------------------------------------- Stage 1: motions
MOTIONS = {
    # name: (R_min [m], frequency [Hz], waveform, description)
    "wrist":   dict(R_min=25e-3, f=0.5,  kind="flex",  label="Wrist flexion (0.5 Hz)"),
    "elbow":   dict(R_min=40e-3, f=0.5,  kind="flex",  label="Elbow flexion (0.5 Hz)"),
    "walking": dict(R_min=300e-3, f=1.8, kind="osc",   label="Walking, forearm (1.8 Hz)"),
    "hold":    dict(R_min=30e-3, f=0.0,  kind="hold",  label="Sustained wrist flexion"),
}

def curvature(t, m):
    k = 1.0 / m["R_min"]
    if m["kind"] == "flex":   # unipolar: flat -> bent -> flat
        return k * 0.5 * (1 - np.cos(2 * np.pi * m["f"] * t))
    if m["kind"] == "osc":    # bipolar oscillation about calibrated static curvature
        return k * np.sin(2 * np.pi * m["f"] * t)
    if m["kind"] == "hold":
        return k * np.ones_like(t)

# ---------------------------------------------------------------- Stage 6: dynamic scenario
def binding_kinetics(t, C_nM_t):
    th = np.zeros_like(t); dt = t[1] - t[0]
    for i in range(1, len(t)):
        dth = P["k_on"] * C_nM_t[i] * 1e-9 * (1 - th[i-1]) - P["k_off"] * th[i-1]
        th[i] = th[i-1] + dt * dth
    return th

def scenario(fs=20.0, T=1800.0, eps_per_kappa=None, mismatch=0.05, fc=0.01):
    t = np.arange(0, T, 1.0 / fs)
    # true cortisol: 30 ng/mL, stress rise to 100 ng/mL beginning at 300 s
    C_true = np.where(t < 300, 30.0, 100.0) * NGML_TO_NM
    th = binding_kinetics(t, C_true)
    # motion schedule
    kap = np.zeros_like(t)
    seg = [(300, 600, "walking"), (600, 900, "wrist"), (900, 1200, "hold"), (1200, 1500, "elbow")]
    for a, b, name in seg:
        idx = (t >= a) & (t < b)
        kap[idx] = curvature(t[idx] - a, MOTIONS[name])
    eps = eps_per_kappa * kap
    R_bio = dVT_max_sweat() * th / P["V_ov"]
    R_sens = R_bio * (1 + P["kappa_x"] * eps) + rel_current_change_strain(eps, P["V_ov"])
    R_ref = rel_current_change_strain(eps, P["V_ov"]) * (1 + mismatch)   # twin reference channel, no aptamer
    R_twin = R_sens - R_ref
    sos = butter(2, fc / (fs / 2), output="sos")
    R_filt = sosfiltfilt(sos, R_sens)
    R_twin_filt = sosfiltfilt(sos, R_twin)
    C_of = lambda R: invert_response(R) / NGML_TO_NM
    out = dict(t=t, kappa=kap, eps=eps, theta=th, C_true=C_true / NGML_TO_NM,
               C_ideal=C_of(R_bio), C_raw=C_of(R_sens), C_filt=C_of(R_filt),
               C_twin=C_of(R_twin), C_twin_filt=C_of(R_twin_filt), segments=seg)
    return out

def seg_errors(out, key):
    """RMS and max |C - C_ideal| per segment (ng/mL); inf if saturated."""
    res = {}
    t = out["t"]
    segs = [(0, 300, "rest"), *out["segments"], (1500, 1800, "rest2")]
    for a, b, name in segs:
        idx = (t >= a) & (t < b)
        e = out[key][idx] - out["C_ideal"][idx]
        if np.any(~np.isfinite(e)):
            res[name] = (np.inf, np.inf)
        else:
            res[name] = (float(np.sqrt(np.mean(e**2))), float(np.max(np.abs(e))))
    return res

# ---------------------------------------------------------------- Stage 5: mitigation catalogue
def mitigation_catalogue():
    """Channel strain per unit curvature for each design. Returns dict name -> (eps/kappa [m], notes)."""
    cat = {}
    L, i, eta = stack(("PI", 25e-6), ("Al2O3", 5e-9)); s, zn, c = channel_strain_per_curvature(L, i, eta)
    cat["Baseline: 25 um PI, 5 nm Al2O3"] = s
    L, i, eta = stack(("PI", 25e-6), ("PMMA", 1e-6)); s, _, _ = channel_strain_per_curvature(L, i, eta)
    cat["Polymer encapsulation (1 um PMMA)"] = s
    L, i, eta = stack(("PI", 12.5e-6), ("Al2O3", 5e-9)); s, _, _ = channel_strain_per_curvature(L, i, eta)
    cat["Thinner substrate (12.5 um PI)"] = s
    L, i, eta = stack(("PET", 2.5e-6), ("Al2O3", 5e-9)); s, _, _ = channel_strain_per_curvature(L, i, eta)
    cat["Ultrathin substrate (2.5 um PET)"] = s
    L, i, eta = stack(("PI", 25e-6), ("PI", 25e-6)); s, _, _ = channel_strain_per_curvature(L, i, eta, misalign=1e-6)
    cat["Neutral-plane buried channel + extended gate (1 um tol.)"] = s
    cat["Rigid island (4.2 % strain ratio)"] = cat["Baseline: 25 um PI, 5 nm Al2O3"] * 0.042
    cat["Prebent channel (1e-3 strain ratio)"] = cat["Baseline: 25 um PI, 5 nm Al2O3"] * 1e-3
    return cat

if __name__ == "__main__":
    np.set_printoptions(precision=4)
    L, i, eta = stack(); s, zn, cents = channel_strain_per_curvature(L, i, eta)
    print("Neutral plane z_n = %.3f um ; MoS2 centroid = %.3f um ; eps/kappa = %.3e m" % (zn*1e6, cents[i]*1e6, s))
    print("lambda_D(sweat) = %.3f nm ; dVT_max,sweat = %.2f mV ; full-scale R = %.3f %%" %
          (debye_length_nm(P["I_sweat_M"]), dVT_max_sweat()*1e3, dVT_max_sweat()/P["V_ov"]*100))
    print("GF_channel(V_ov=0.3) = %.1f ; GF_contact = %.1f ; GF_total = %.1f" % (GF_channel(0.3), GF_contact(), GF_total_small_signal(0.3)))
    for name, m in MOTIONS.items():
        kmax = 1/m["R_min"]; e = s*kmax
        print("%-28s kappa_max=%.2f 1/m  eps_ch=%.4f %%  dI/I=%.2f %%  C_false@50=%s ng/mL" %
              (name, kmax, e*100, rel_current_change_strain(e, P["V_ov"])*100, false_concentration_ngml(e)))
    print("Physiological signal: R(8.16)=%.3f%% R(141.7)=%.3f%%" % (response(8.16*NGML_TO_NM)*100, response(141.7*NGML_TO_NM)*100))
    for k, v in mitigation_catalogue().items():
        e = v / 25e-3
        print("%-58s eps=%.5f %%  C_false=%s" % (k, e*100, false_concentration_ngml(e)))
