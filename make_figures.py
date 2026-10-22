"""
Figures for the manuscript.

  python make_figures.py                  # reference Python model -> figures/ and figures/results.json
  python make_figures.py --source elmer   # figs 3-8 from elmer/data/curves.npz -> elmer/figures/
                                          # (also: comsol, fem; figs 1-2 are schematics, Python only)
"""
import argparse
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from motion_artifact_sim import *

_ap = argparse.ArgumentParser()
_ap.add_argument("--source", choices=["python", "comsol", "fem", "elmer"], default="python")
SRC = _ap.parse_args().source

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],   # JCE: Helvetica/Arial lettering
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0})
OUT = "figures" if SRC == "python" else os.path.join(SRC, "figures")
os.makedirs(OUT, exist_ok=True)
COL = 3.45; DBL = 7.1
RES = {}


def figs_1_2():
    # ------------------------------------------------------------------ Fig.1 block diagram
    fig, ax = plt.subplots(figsize=(DBL, 2.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    stages = [("Stage 1\nBody motion", "wrist, elbow,\nwalking\n" + r"$\kappa(t)=1/R(t)$"),
              ("Stage 2\nMultilayer\nmechanics", "neutral plane,\nvdW transfer " + r"$\eta$" + "\n" + r"$\varepsilon_{ch}(t)$"),
              ("Stage 3\nStrain to\nelectronics", r"$GF_{ch}(V_{ov})$, $\Delta\Phi_B$," + "\ncross-talk " + r"$\kappa_x$"),
              ("Stage 4\nCompact FET\n+ binding", "Langmuir kinetics,\nDebye screening\n" + r"$\Delta I/I_0$"),
              ("Stage 5\nError metric", "strain-equivalent\nfalse cortisol\n" + r"$C_{false}$"),
              ("Stage 6\nDynamics &\nmitigation", "LPF, twin sensor,\n7 structural\ndesigns")]
    w = 14; gap = (100 - 6 * w) / 7
    for k, (h, b) in enumerate(stages):
        x = gap + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 13), w, 14, boxstyle="round,pad=0.4", fc="#dfe9f5" if k < 5 else "#f5e6d3", ec="k", lw=0.7))
        ax.text(x + w / 2, 20, h, ha="center", va="center", fontsize=7.5, weight="bold")
        ax.add_patch(FancyBboxPatch((x, 1.5), w, 9.5, boxstyle="round,pad=0.3", fc="white", ec="#666", lw=0.5))
        ax.text(x + w / 2, 6.3, b, ha="center", va="center", fontsize=6.3)
        if k < 5:
            ax.add_patch(FancyArrowPatch((x + w + 0.5, 20), (x + w + gap - 0.5, 20), arrowstyle="-|>", mutation_scale=9, lw=0.8, color="k"))
    ax.annotate("", xy=(gap + 5 * (w + gap) + w / 2, 27.5), xytext=(gap + w / 2, 27.5),
                arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=-0.15", lw=0.7, ls="--", color="#8a3b0c"))
    ax.text(50, 29.3, "design sweep loop (substrate, encapsulation, geometry, gate bias)", ha="center", fontsize=6.5, color="#8a3b0c")
    fig.savefig(f"{OUT}/fig1_framework.png", dpi=1200, bbox_inches="tight"); plt.close(fig)

    # ------------------------------------------------------------------ Fig.2 cross-section + neutral plane
    fig, axs = plt.subplots(1, 2, figsize=(DBL, 2.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]; ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 60)
    layers = [("Skin (adhesive, compliant)", 0, 8, "#f3d9c6"), ("Polyimide substrate (25 um)", 8, 30, "#f0c36d"),
              ("Au source / drain (50 nm)", 30, 34, None), ("MoS2 channel (0.65 nm)", 34, 37, "#5c7fb8"),
              ("Al2O3 passivation (5 nm)", 37, 40, "#b0b0b0"), ("Aptamer layer + sweat (liquid gate)", 40, 52, "#cfe5f7")]
    for name, y0, y1, c in layers:
        if c is None:
            ax.add_patch(Rectangle((10, y0), 22, y1 - y0, fc="#d4af37", ec="k", lw=0.5)); ax.add_patch(Rectangle((68, y0), 22, y1 - y0, fc="#d4af37", ec="k", lw=0.5))
            ax.text(93, (y0 + y1) / 2, name, va="center", fontsize=6.3)
        else:
            ax.add_patch(Rectangle((10, y0), 80, y1 - y0, fc=c, ec="k", lw=0.5)); ax.text(93, (y0 + y1) / 2, name, va="center", fontsize=6.3)
    ax.plot([32, 68], [55, 55], color="#1f4e79", lw=0.8); ax.text(50, 57.5, "reference electrode, $V_G$", ha="center", fontsize=6.3, color="#1f4e79")
    ax.axhline(8 + 22 * 13.4 / 25.0, xmin=0.08, xmax=0.92, ls="--", color="r", lw=0.9)
    ax.text(12, 8 + 22 * 13.4 / 25.0 + 1.2, "neutral plane $z_n$", color="r", fontsize=6.5)
    ax.annotate("", xy=(5, 35.5), xytext=(5, 8 + 22 * 13.4 / 25.0), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(1, 27, "$z_{ch}-z_n$", fontsize=6.5, rotation=90, va="center")
    ax.text(50, 2, "(a)", ha="center", fontsize=8)
    # (b) strain profile vs thickness
    ax = axs[1]
    ts = np.logspace(np.log10(1e-6), np.log10(100e-6), 60)
    for cap, lab, c in [(("Al2O3", 5e-9), "thin Al$_2$O$_3$ cap ($\\eta$=0.8)", "k"), (("PMMA", 1e-6), "1 um PMMA cap ($\\eta$=1.0)", "#c0392b"), (("none", 0), "bare ($\\eta$=0.5)", "#7f8c8d")]:
        e = []
        for t in ts:
            L, i, eta = stack(("PI", t), cap); s, _, _ = channel_strain_per_curvature(L, i, eta); e.append(s / 25e-3 * 100)
        ax.loglog(ts * 1e6, e, color=c, label=lab)
    ax.axhline(0.0031 * 1, ls=":", color="#555", lw=0.7)
    ax.set_xlabel("Substrate thickness (um)"); ax.set_ylabel("Channel strain at R = 25 mm (%)")
    ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=0.3, which="both", lw=0.4)
    ax.text(0.03, 0.93, "(b)", transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_stack.png", dpi=1200, bbox_inches="tight"); plt.close(fig)



# ------------------------------------------------------------------ data for figs 3-8 from the reference Python model
def python_curves():
    """Curves for figs 3-8 (same keys as comsol/data/curves.npz) and the manuscript numbers in RES."""
    C = {}
    L, i, eta = stack(); s_base, zn, cents = channel_strain_per_curvature(L, i, eta)
    FS = dVT_max_sweat() / P["V_ov"]
    C["FS"] = FS; C["eps_per_kappa"] = s_base
    # Fig.3
    t = np.linspace(0, 4, 2000); C["fig3_t"] = t
    for name in ["wrist", "elbow", "walking"]:
        k = curvature(t, MOTIONS[name]); C[f"fig3_kap_{name}"] = k; C[f"fig3_eps_{name}"] = s_base * k
    # Fig.4
    eps = np.linspace(-0.1, 0.1, 400) / 100; C["fig4a_eps"] = eps
    for V in [0.1, 0.3, 0.6]:
        C[f"fig4a_V{V:g}"] = rel_current_change_strain(eps, V)
    P_rc = P["r_c"]; P["r_c"] = 0.0; C["fig4a_chonly"] = rel_current_change_strain(eps, 0.3)
    P["r_c"] = 1.0; C["fig4a_conly"] = rel_current_change_strain(eps, 0.3); P["r_c"] = P_rc
    Cg = np.logspace(-1, 3.2, 300); C["fig4b_C"] = Cg
    for Iion in (0.001, 0.05, 0.16):
        P["I_sweat_M"] = Iion; C[f"fig4b_R_I{Iion:g}"] = response(Cg * NGML_TO_NM)
        C[f"fig4b_lam_I{Iion:g}"] = np.full_like(Cg, debye_length_nm(Iion))
    P["I_sweat_M"] = 0.05
    # Fig.5
    C0s = np.linspace(5, 140, 200); C["fig5a_C0"] = C0s
    for name in ["walking", "elbow", "wrist"]:
        e = s_base / MOTIONS[name]["R_min"]
        C[f"fig5a_cf_{name}"] = np.array([false_concentration_ngml(e, c0) for c0 in C0s])
        RES[f"art_{name}_pct"] = rel_current_change_strain(e, P["V_ov"]) * 100
        RES[f"eps_{name}_pct"] = e * 100
    rcs = np.linspace(0, 0.8, 100); C["fig5b_rc"] = rcs
    for name in ["walking", "elbow", "wrist"]:
        e = s_base / MOTIONS[name]["R_min"]; vals = []
        for r in rcs:
            P["r_c"] = r; vals.append(rel_current_change_strain(e, P["V_ov"]) / FS * 100)
        P["r_c"] = 0.3; C[f"fig5b_{name}"] = np.array(vals)
    # Fig.6
    out = scenario(eps_per_kappa=s_base)
    L2, i2, eta2 = stack(("PET", 2.5e-6), ("Al2O3", 5e-9)); s_thin, _, _ = channel_strain_per_curvature(L2, i2, eta2)
    out2 = scenario(eps_per_kappa=s_thin)
    for tag, o in (("baseline", out), ("thin", out2)):
        C[f"fig6_{tag}_t"] = o["t"]; C[f"fig6_{tag}_kappa"] = o["kappa"]; C[f"fig6_{tag}_eps"] = o["eps"]
        C[f"fig6_{tag}_C_true"] = o["C_true"]
        for k in ["C_ideal", "C_raw", "C_filt", "C_twin", "C_twin_filt"]:
            C[f"fig6_{tag}_{k}"] = o[k]
    RES["dyn_baseline"] = {k: seg_errors(out, k) for k in ["C_raw", "C_filt", "C_twin", "C_twin_filt"]}
    RES["dyn_thin"] = {k: seg_errors(out2, k) for k in ["C_raw", "C_filt", "C_twin", "C_twin_filt"]}
    s_isl = s_base * 0.042
    out3 = scenario(eps_per_kappa=s_isl)
    RES["dyn_island"] = {k: seg_errors(out3, k) for k in ["C_raw", "C_filt", "C_twin", "C_twin_filt"]}
    # Fig.7
    fs_ = 20.0; fc = 0.01
    freqs = np.logspace(-3, 0.5, 40); resid_uni = []; resid_bi = []
    sos = butter(2, fc / (fs_ / 2), output="sos")
    for f in freqs:
        tt = np.arange(0, max(600, 20 / f), 1 / fs_)
        k = 40 * 0.5 * (1 - np.cos(2 * np.pi * f * tt)); e = s_base * k
        r = rel_current_change_strain(e, P["V_ov"]); rf = sosfiltfilt(sos, r)
        resid_uni.append(np.mean(rf[len(rf)//4: -len(rf)//4]) / FS * 100)
        k2 = 40 * 0.5 * np.sin(2 * np.pi * f * tt); e2 = s_base * k2
        r2 = rel_current_change_strain(e2, P["V_ov"]); rf2 = sosfiltfilt(sos, r2)
        resid_bi.append(np.sqrt(np.mean(rf2[len(rf2)//4: -len(rf2)//4]**2)) / FS * 100)
    C["fig7a_freqs"] = freqs; C["fig7a_uni"] = np.array(resid_uni); C["fig7a_bi"] = np.array(resid_bi)
    mism = np.linspace(0, 0.2, 50); e_w = s_base / 25e-3
    R0 = response(50 * NGML_TO_NM)
    art = rel_current_change_strain(e_w, P["V_ov"])
    C["fig7b_mism"] = mism
    C["fig7b_resid"] = np.abs(art * mism + P["kappa_x"] * e_w * R0) / FS * 100
    C["fig7b_resid_noX"] = np.abs(art * mism) / FS * 100
    RES["xtalk_floor_pct_FS"] = float(P["kappa_x"] * e_w * R0 / FS * 100)
    # Fig.8
    cat = mitigation_catalogue()
    names = list(cat.keys()); vals_w = []; vals_e = []; vals_walk = []
    for n in names:
        for R, arr in [(25e-3, vals_w), (40e-3, vals_e), (300e-3, vals_walk)]:
            e = cat[n] / R; arr.append(rel_current_change_strain(e, P["V_ov"]) / FS * 100)
    C["fig8a_names"] = np.array(names); C["fig8a_w"] = np.array(vals_w); C["fig8a_e"] = np.array(vals_e)
    C["fig8a_walk"] = np.array(vals_walk)
    RES["mitigation_pct_FS"] = {n: dict(wrist=vals_w[k], elbow=vals_e[k], walking=vals_walk[k], eps_wrist_pct=cat[n] / 25e-3 * 100) for k, n in enumerate(names)}
    Vov = np.linspace(0.05, 1.0, 200); e_w = s_base / 25e-3
    ratio = [rel_current_change_strain(e_w, V) / (dVT_max_sweat() / V) * 100 for V in Vov]
    C["gate_V"] = Vov; C["gate_ratio_rc0.3"] = np.array(ratio) / 100
    for r in (0.0, 0.6):
        P["r_c"] = r; C[f"gate_ratio_rc{r:g}"] = np.array([rel_current_change_strain(e_w, V) / (dVT_max_sweat() / V) for V in Vov])
    P["r_c"] = 0.3
    imin = int(np.argmin(ratio)); RES["gate_opt"] = dict(Vov=float(Vov[imin]), ratio=float(ratio[imin]), ratio_at_0p3=float(np.interp(0.3, Vov, ratio)), ratio_at_0p1=float(np.interp(0.1, Vov, ratio)))
    # misc numbers
    RES["zn_um"] = zn * 1e6; RES["eps_per_kappa_um"] = s_base * 1e6
    RES["lambda_D_nm"] = debye_length_nm(0.05); RES["dVT_sweat_mV"] = dVT_max_sweat() * 1e3; RES["FS_pct"] = FS * 100
    RES["R_phys_low_pct"] = response(8.16 * NGML_TO_NM) * 100; RES["R_phys_high_pct"] = response(141.7 * NGML_TO_NM) * 100
    RES["GF_ch_0p3"] = GF_channel(0.3); RES["GF_c"] = GF_contact(); RES["GF_tot_0p3"] = GF_total_small_signal(0.3)
    RES["K_D_ngml"] = P["K_D_nM"] / NGML_TO_NM; RES["k_on"] = P["k_on"]
    RES["tau_100ngml_s"] = 1 / (P["k_on"] * 100 * NGML_TO_NM * 1e-9 + P["k_off"])
    for name in ["wrist", "elbow", "walking", "hold"]:
        e = s_base / MOTIONS[name]["R_min"]; C0 = 30 * NGML_TO_NM
        S = dVT_max_sweat() / P["V_ov"] * P["K_D_nM"] / (C0 + P["K_D_nM"])**2   # per nM
        RES[f"lin_false_{name}_ngml"] = float(rel_current_change_strain(e, P["V_ov"]) / S / NGML_TO_NM)
        RES[f"exact_false_{name}_at30"] = float(false_concentration_ngml(e, 30.0))
        RES[f"exact_false_{name}_at50"] = float(false_concentration_ngml(e, 50.0))
    RES["check_john_sum_meV"] = 147 + 55
    RES["check_debye_100mM_nm"] = debye_length_nm(0.1)
    P_rc = P["r_c"]; P["r_c"] = 0.0
    RES["check_datye_channel_only_0p7_Vov0p145"] = float(rel_current_change_strain(0.007, 0.145) * 100)
    RES["GF_ch_0p145"] = float(GF_channel(0.145))
    P["r_c"] = P_rc
    RES["check_li_29pct"] = float(P["kappa_x"] * 0.0034 * 100)
    RES["check_symmetric_stack_strain"] = float(channel_strain_per_curvature(*stack(("PI", 25e-6), ("PI", 25e-6)))[0])
    return C


def comsol_curves():
    d = np.load(os.path.join(SRC, "data", "curves.npz"))
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


# ------------------------------------------------------------------ figs 3-8 (plotting only)
def plot_figs_3_8(C):
    FS = C["FS"]; s_base = C["eps_per_kappa"]
    # Fig.3 motion inputs
    fig, axs = plt.subplots(2, 1, figsize=(COL, 3.0), sharex=True)
    t = C["fig3_t"]
    for name, c in [("wrist", "#c0392b"), ("elbow", "#2980b9"), ("walking", "#27ae60")]:
        axs[0].plot(t, C[f"fig3_kap_{name}"], color=c, label=MOTIONS[name]["label"]); axs[1].plot(t, C[f"fig3_eps_{name}"] * 100, color=c)
    axs[0].set_ylabel("Curvature (1/m)"); axs[1].set_ylabel("Channel strain (%)"); axs[1].set_xlabel("Time (s)")
    axs[0].legend(frameon=False, fontsize=6.3); [a.grid(alpha=0.3, lw=0.4) for a in axs]
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_motion.png", dpi=1200, bbox_inches="tight"); plt.close(fig)

    # Fig.4 strain->current decomposition & calibration
    fig, axs = plt.subplots(1, 2, figsize=(DBL, 2.5))
    eps = C["fig4a_eps"]
    ax = axs[0]
    for V in [0.1, 0.3, 0.6]:
        ax.plot(eps * 100, C[f"fig4a_V{V:g}"] * 100, label=f"total, $V_{{ov}}$ = {V:.1f} V")
    ax.plot(eps * 100, C["fig4a_chonly"] * 100, "k--", label="channel only ($V_{ov}$ = 0.3 V)")
    ax.plot(eps * 100, C["fig4a_conly"] * 100 * 0.3, "k:", label="contacts only (30 % of R)")
    ax.axhspan(-7.66 / 2, 7.66 / 2, color="#f9e79f", alpha=0.5); ax.text(-0.098, 3.0, "full-scale cortisol\nresponse window", fontsize=6)
    for nm, xx in [("walking", 0.0031), ("elbow", 0.0232), ("wrist", 0.0372)]:
        ax.axvline(xx, color="#888", lw=0.5, ls="-."); ax.text(xx, -9.5, nm, rotation=90, fontsize=6, ha="right", va="bottom")
    ax.set_xlabel("Channel strain (%)"); ax.set_ylabel("Current change (%)"); ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0.08,1.0)); ax.grid(alpha=0.3, lw=0.4)
    ax.text(0.02, 0.93, "(a)", transform=ax.transAxes)
    ax = axs[1]
    Cg = C["fig4b_C"]
    for Iion, lab, c in [(0.001, "1 mM (buffer-diluted)", "#27ae60"), (0.05, "50 mM (sweat)", "k"), (0.16, "160 mM (1x PBS)", "#c0392b")]:
        ax.semilogx(Cg, C[f"fig4b_R_I{Iion:g}"] * 100, color=c, label=lab + f",  $\\lambda_D$={C[f'fig4b_lam_I{Iion:g}'][0]:.2f} nm")
    ax.axvspan(8.16, 141.7, color="#d5f5e3", alpha=0.6); ax.text(30, 9.5, "physiological\nsweat range", fontsize=6, ha="center")
    ax.set_xlabel("Cortisol concentration (ng/mL)"); ax.set_ylabel("Response  $\\Delta I/I_0$ (%)"); ax.legend(frameon=False, fontsize=6, loc="lower right"); ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.text(0.03, 0.92, "(b)", transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_electrical.png", dpi=1200, bbox_inches="tight"); plt.close(fig)

    # Fig.5 false concentration vs motion, vs operating point
    fig, axs = plt.subplots(1, 2, figsize=(DBL, 2.5))
    ax = axs[0]
    C0s = C["fig5a_C0"]
    for name, c in [("walking", "#27ae60"), ("elbow", "#2980b9"), ("wrist", "#c0392b")]:
        cf = C[f"fig5a_cf_{name}"]
        ax.semilogy(C0s, np.where(np.isfinite(cf), cf, 1e4), color=c, label=MOTIONS[name]["label"] + " (peak)")
    ax.axhline(8.16, ls="--", color="#555", lw=0.7); ax.text(6, 9.5, "low end of physiological range (8.16 ng/mL)", fontsize=6)
    ax.axhline(141.7, ls="--", color="#555", lw=0.7); ax.text(6, 165, "high end (141.7 ng/mL)", fontsize=6)
    ax.set_ylim(0.5, 2e4); ax.set_xlabel("True cortisol at operating point (ng/mL)"); ax.set_ylabel("False cortisol $C_{false}$ (ng/mL)")
    ax.text(60, 6000, "reading saturates\n(> full scale)", fontsize=6, ha="center", color="#c0392b")
    ax.legend(frameon=False, fontsize=6, loc="lower right"); ax.grid(alpha=0.3, lw=0.4, which="both"); ax.text(0.03, 0.93, "(a)", transform=ax.transAxes)
    ax = axs[1]
    for name, c in [("walking", "#27ae60"), ("elbow", "#2980b9"), ("wrist", "#c0392b")]:
        ax.plot(C["fig5b_rc"], C[f"fig5b_{name}"], color=c, label=MOTIONS[name]["label"])
    ax.axhline(100, color="k", lw=0.6, ls="--"); ax.text(0.02, 106, "artifact = full physiological scale", fontsize=6)
    ax.set_xlabel("Contact fraction of total resistance $r_c$"); ax.set_ylabel("Peak artifact (% of full scale)"); ax.grid(alpha=0.3, lw=0.4); ax.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(0.0,0.62)); ax.set_ylim(0,115)
    ax.text(0.03, 0.92, "(b)", transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_false_conc.png", dpi=1200, bbox_inches="tight"); plt.close(fig)

    # Fig.6 dynamic scenario
    b = lambda k: C[f"fig6_baseline_{k}"]; th = lambda k: C[f"fig6_thin_{k}"]
    fig, axs = plt.subplots(4, 1, figsize=(DBL, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[0.8, 0.8, 1.6, 0.9]))
    t = b("t")
    axs[0].plot(t, b("kappa"), color="#555", lw=0.5); axs[0].set_ylabel("Curvature\n(1/m)")
    for a_, b_, nm in [(300, 600, "walking"), (600, 900, "wrist"), (900, 1200, "hold"), (1200, 1500, "elbow")]:
        axs[0].text((a_ + b_) / 2, 44, MOTIONS[nm]["label"].split(" (")[0], ha="center", fontsize=6)
    axs[0].set_ylim(-5, 52)
    axs[1].plot(t, b("eps") * 100, color="#8e44ad", lw=0.5); axs[1].set_ylabel("Channel\nstrain (%)")
    ax = axs[2]
    raw = np.where(np.isfinite(b("C_raw")), b("C_raw"), 1e3)
    ax.plot(t, raw, color="#bbb", lw=0.4, label="raw reading")
    ax.plot(t, b("C_filt"), color="#2980b9", lw=1.0, label="low-pass filtered ($f_c$ = 10 mHz)")
    ax.plot(t, b("C_twin_filt"), color="#c0392b", lw=1.0, label="twin-sensor subtraction + LPF (5 % mismatch)")
    ax.plot(t, b("C_ideal"), "k--", lw=1.0, label="ideal (motion-free) reading")
    ax.plot(t, b("C_true"), color="#27ae60", lw=0.8, ls=":", label="true sweat cortisol")
    ax.set_ylim(0, 400); ax.set_ylabel("Cortisol reading (ng/mL)"); ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
    ax.text(1350, 300, "saturated", ha="center", fontsize=6, color="#888", rotation=90)
    ax = axs[3]
    ax.plot(t, b("C_filt") - b("C_ideal"), color="#2980b9", lw=0.8, label="LPF only"); ax.plot(t, b("C_twin_filt") - b("C_ideal"), color="#c0392b", lw=0.8, label="twin + LPF")
    ax.plot(th("t"), th("C_twin_filt") - th("C_ideal"), color="#8e44ad", lw=0.8, label="twin + LPF, 2.5 um PET")
    ax.set_ylim(-4, 6); ax.set_ylabel("Error vs ideal\n(ng/mL)"); ax.set_xlabel("Time (s)"); ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper right")
    ax.axhline(0,color="k",lw=0.4)
    ax.text(310, 4.5, "filter response at\ncortisol step", fontsize=6)
    [a.grid(alpha=0.3, lw=0.4) for a in axs]
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_dynamic.png", dpi=1200, bbox_inches="tight"); plt.close(fig)

    # Fig.7 filter vs frequency and twin-sensor mismatch
    fc = 0.01
    fig, axs = plt.subplots(1, 2, figsize=(DBL, 2.4))
    ax = axs[0]
    ax.semilogx(C["fig7a_freqs"], C["fig7a_uni"], color="#c0392b", label="unipolar flexion (flat -> bent -> flat)")
    ax.semilogx(C["fig7a_freqs"], C["fig7a_bi"], color="#2980b9", label="bipolar oscillation (about calibration)")
    ax.axvline(fc, ls="--", color="#555", lw=0.7); ax.text(fc * 1.15, 1.5, "$f_c$ = 10 mHz", fontsize=6)
    ax.axvspan(0.5, 2.5, color="#eee"); ax.text(1.1, 9, "body\nmotion\nband", ha="center", fontsize=6); ax.set_ylim(-1, 30)
    ax.text(0.03, 17.5, "DC component of unipolar motion passes the filter", fontsize=6, color="#c0392b")
    ax.set_xlabel("Motion frequency (Hz)"); ax.set_ylabel("Residual after LPF\n(% of full scale)"); ax.legend(frameon=False, fontsize=6, loc="upper right"); ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.text(0.03, 0.92, "(a)", transform=ax.transAxes)
    ax = axs[1]
    ax.plot(C["fig7b_mism"] * 100, C["fig7b_resid"], color="k", label="with sensitivity cross-talk ($\\kappa_x$ = 85)")
    ax.plot(C["fig7b_mism"] * 100, C["fig7b_resid_noX"], color="k", ls="--", label="without cross-talk")
    ax.set_xlabel("Twin-sensor gauge mismatch (%)"); ax.set_ylabel("Residual wrist artifact\n(% of full scale)"); ax.legend(frameon=False, fontsize=6); ax.grid(alpha=0.3, lw=0.4)
    ax.text(0.03, 0.92, "(b)", transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig7_filter_twin.png", dpi=1200, bbox_inches="tight"); plt.close(fig)

    # Fig.8 mitigation comparison + gate bias
    fig, axs = plt.subplots(1, 2, figsize=(DBL, 2.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axs[0]
    names = [str(n) for n in C["fig8a_names"]]
    if SRC == "elmer":   # FE geometries replace the analytic ratios; keys in results.json stay unchanged
        names = [n.replace("Rigid island (4.2 % strain ratio)", "Rigid island (FE, 100 um PDMS, end moment)")
                  .replace(" (1 um tol.)", " (FE, 0.98 um offset)") for n in names]
    y = np.arange(len(names))
    ax.barh(y + 0.27, C["fig8a_w"], 0.26, color="#c0392b", label="wrist (R = 25 mm)")
    ax.barh(y, C["fig8a_e"], 0.26, color="#2980b9", label="elbow (R = 40 mm)")
    ax.barh(y - 0.27, C["fig8a_walk"], 0.26, color="#27ae60", label="walking (R = 300 mm)")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([n.replace(" + extended gate", "\n+ extended gate") for n in names], fontsize=6)
    ax.axvline(100, color="k", ls="--", lw=0.7); ax.text(100, -0.75, "100 % FS", fontsize=6, ha="center")
    ax.axvline(10, color="#888", ls=":", lw=0.7); ax.text(10, -0.75, "10 % FS", fontsize=6, ha="center")
    ax.set_xlabel("Peak motion artifact (% of full-scale cortisol response)"); ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3); ax.invert_yaxis()
    ax.text(0.01, 1.02, "(a)", transform=ax.transAxes)
    ax = axs[1]
    Vov = C["gate_V"]
    ax.plot(Vov, C["gate_ratio_rc0.3"] * 100, "k", label="$r_c$ = 0.3 (baseline)")
    for r, c, ls in [(0.0, "#2980b9", "--"), (0.6, "#c0392b", ":")]:
        ax.plot(Vov, C[f"gate_ratio_rc{r:g}"] * 100, color=c, ls=ls, label=f"$r_c$ = {r:.1f}")
    ax.set_xlabel("Gate overdrive $V_{ov}$ (V)"); ax.set_ylabel("Wrist artifact / full-scale signal (%)"); ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0.1,1.0)); ax.grid(alpha=0.3, lw=0.4)
    ax.text(0.03, 0.93, "(b)", transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig8_mitigation.png", dpi=1200, bbox_inches="tight"); plt.close(fig)


if SRC == "python":
    figs_1_2()
    plot_figs_3_8(python_curves())
    json.dump(RES, open(f"{OUT}/results.json", "w"), indent=1, default=str)
    print(json.dumps(RES, indent=1, default=str))
else:
    plot_figs_3_8(comsol_curves())
    print("figs 3-8 written to", OUT)
