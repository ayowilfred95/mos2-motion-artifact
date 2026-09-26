"""
Single source of COMSOL parameters.

All values that also exist in the reference Python model are taken from motion_artifact_sim.PARAMS
so the two implementations cannot drift apart. Quantities that the 1D analytic model does not need
(Poisson ratios for plane strain, flake length, PDMS thickness, ...) are listed in EXTRA with a
source tag and are reported in COMSOL_METHODS.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from motion_artifact_sim import PARAMS as P, MOTIONS, kT_eV, MW_CORT, NGML_TO_NM  # noqa: E402

# ---------------------------------------------------------------- extra parameters (COMSOL only)
EXTRA = {
    # name: (value, COMSOL unit string, source tag)
    "L_strip": (10e-3, "m", "prompt: 10 mm strip"),
    "L_flake": (10e-6, "m", "assumed MoS2 channel length 10 um (typical flexible MoS2 FET); only sets the calibrated k_int, not eta"),
    "nu_PI": (0.34, "1", "prompt"),
    "nu_Au": (0.44, "1", "bulk Au, standard handbook value"),
    "nu_MoS2": (0.27, "1", "monolayer MoS2, DFT/nanoindentation literature 0.25-0.29"),
    "nu_Al2O3": (0.24, "1", "ALD alumina, literature 0.22-0.24"),
    "nu_PET": (0.38, "1", "PET film, handbook value"),
    "nu_PMMA": (0.35, "1", "PMMA, handbook value"),
    "E_PDMS": (2e6, "Pa", "prompt: PDMS 2 MPa"),
    "nu_PDMS": (0.49, "1", "PDMS, nearly incompressible"),
    "E_isl": (70e9, "Pa", "prompt: rigid island 70 GPa"),
    "nu_isl": (0.22, "1", "glass-like island"),
    "w_isl": (3e-3, "m", "prompt: 3 mm island"),
    "t_isl": (20e-6, "m", "prompt: 20 um island"),
    "t_PDMS": (100e-6, "m", "assumed PDMS carrier thickness 100 um (not specified in prompt)"),
    "t_buried_cap": (23e-6, "m", "top PI 25 um - 2 um -> ~1 um channel offset from neutral plane"),
    "t_PMMA": (1e-6, "m", "prompt"),
    "kappa_ref": (1.0, "1/m", "reference curvature for the linear solve"),
    "T_room": (298.15, "K", "Debye model temperature"),
    "epsr_w": (78.5, "1", "water relative permittivity at 25 C"),
    "psi0_PB": (1e-3, "V", "surface potential for linear-regime Poisson-Boltzmann (<< kT/e)"),
    "Phi_B_lit_meV": (147.0, "1", "John 2020 zero-strain barrier term used in check_john_sum_meV"),
}

# ---------------------------------------------------------------- mechanics parameter strings
def mech_params():
    """COMSOL Parameters for model 1 (strings with units)."""
    d = {
        "E_PI": f"{P['E_PI']}[Pa]", "E_PET": f"{P['E_PET']}[Pa]", "E_Au": f"{P['E_Au']}[Pa]",
        "E_MoS2": f"{P['E_MoS2']}[Pa]", "E_Al2O3": f"{P['E_Al2O3']}[Pa]", "E_PMMA": f"{P['E_PMMA']}[Pa]",
        "t_MoS2": f"{P['t_MoS2']}[m]", "t_Au": f"{P['t_Au']}[m]", "t_Al2O3": f"{P['t_Al2O3']}[m]",
        "eta_bare": str(P["eta_bare"]), "eta_thinALD": str(P["eta_thinALD"]), "eta_polymer": str(P["eta_polymer"]),
        # plane-strain switch: nu_on = 1 -> physical Poisson ratios, 0 -> nu = 0 (reproduces 1D beam theory)
        "nu_on": "1",
        # substrate (baseline component is swept through these)
        "t_sub": "25e-6[m]", "E_sub": "E_PI", "nu_sub": "nu_PI*nu_on",
        # cap membrane on the flake: 1 -> Al2O3 5 nm cap, 0 -> bare
        "cap_on": "1",
        # interfacial spring (Thin Elastic Layer, tangential stiffness per area)
        "k_int": "1e15[N/m^3]",
        "kappa": "kappa_ref",
    }
    for k, (v, u, _) in EXTRA.items():
        d[k] = f"{v}[{u}]" if u != "1" else f"{v}"
    # plane-strain membrane stiffnesses of the nanometre layers, E t / (1 - nu^2)
    d["Et_Au"] = "E_Au*t_Au/(1-(nu_on*nu_Au)^2)"
    d["Et_MoS2"] = "E_MoS2*t_MoS2/(1-(nu_on*nu_MoS2)^2)"
    d["Et_Al2O3"] = "E_Al2O3*t_Al2O3/(1-(nu_on*nu_Al2O3)^2)"
    d["Et_flake"] = "Et_MoS2 + cap_on*Et_Al2O3"
    d["h_m"] = "t_Au + t_MoS2/2"   # MoS2 mid-plane height above the substrate surface
    return d


# ---------------------------------------------------------------- chain parameter strings
def chain_params():
    """COMSOL Parameters for model 2 (strings with units)."""
    M = MOTIONS
    return {
        "kT": f"{kT_eV}[V]", "MW": f"{MW_CORT}[g/mol]",
        "GF_min": str(P["GF_min"]), "GF_max": str(P["GF_max"]), "V_gf": f"{P['V_gf']}[V]", "V0_gf": f"{P['V0_gf']}[V]",
        "dPhiB_ps": f"{P['dPhiB_per_strain_eV']}[V]", "kappa_x": str(P["kappa_x"]), "r_c": str(P["r_c"]),
        "K_D": f"{P['K_D_nM']}[nM]", "k_off": f"{P['k_off']}[1/s]", "k_on": "k_off/K_D",
        "dVT_vac": f"{P['dVT_vac_V']}[V]", "d_charge": f"{P['d_charge_nm']}[nm]", "I_ion": f"{P['I_sweat_M']}[M]",
        "V_ov": f"{P['V_ov']}[V]",
        "C_low": f"{P['C_phys_low_ngml']}[ng/ml]", "C_high_phys": f"{P['C_phys_high_ngml']}[ng/ml]",
        # motions
        "R_wrist": f"{M['wrist']['R_min']}[m]", "f_flex": f"{M['wrist']['f']}[Hz]",
        "R_elbow": f"{M['elbow']['R_min']}[m]",
        "R_walk": f"{M['walking']['R_min']}[m]", "f_walk": f"{M['walking']['f']}[Hz]",
        "R_hold": f"{M['hold']['R_min']}[m]",
        # scenario
        "C0": "30[ng/ml]", "C_step": "100[ng/ml]", "t_step": "300[s]",
        "mot": "0",               # 0 = 1800 s schedule, 1 wrist, 2 elbow, 3 walking, 4 hold (from t = 0)
        "kap_s": "0[1/m]",        # static curvature for stationary sweeps
        "eps_s": "0",             # directly imposed strain for transfer-curve sweeps
        "epk": "9.29e-6[m]",      # eps/kappa from model 1 (overwritten by run_all.py with the COMSOL value)
        "mism": "0.05",           # twin-sensor gauge mismatch
        "Phi_lit": f"{EXTRA['Phi_B_lit_meV'][0]}",
    }


def chain_variables():
    """Global variables of model 2 (exact definitions reported in COMSOL_METHODS.md)."""
    blk = lambda a, b: f"(t>={a}[s])*(t<{b}[s])"
    sched = (f"{blk(300, 600)}*sin(2*pi*f_walk*(t-300[s]))/R_walk"
             f" + {blk(600, 900)}*0.5*(1-cos(2*pi*f_flex*(t-600[s])))/R_wrist"
             f" + {blk(900, 1200)}/R_hold"
             f" + {blk(1200, 1500)}*0.5*(1-cos(2*pi*f_flex*(t-1200[s])))/R_elbow")
    single = ("(mot==1)*0.5*(1-cos(2*pi*f_flex*t))/R_wrist + (mot==2)*0.5*(1-cos(2*pi*f_flex*t))/R_elbow"
              " + (mot==3)*sin(2*pi*f_walk*t)/R_walk + (mot==4)/R_hold")
    return {
        "kap_t": f"if(mot==0, {sched}, {single})",
        "kap": "kap_s + kap_t",
        "eps": "epk*kap + eps_s",
        "GF_ch": "GF_min + (GF_max-GF_min)*exp(-(V_ov-V0_gf)/V_gf)",
        "GF_c": "-dPhiB_ps/kT",
        "GF_tot": "(1-r_c)*GF_ch + r_c*GF_c",
        "dPhiB": "dPhiB_ps*eps",
        "R_rel": "(1-r_c)*exp(-GF_ch*eps) + r_c*exp(dPhiB/kT)",
        "dI_strain": "1/R_rel - 1",
        "xtalk": "1 + kappa_x*eps",
        "lambda_D": "0.304[nm]/sqrt(I_ion/1[M])",
        "dVT_max": "dVT_vac*exp(-d_charge/lambda_D)",
        "FS": "dVT_max/V_ov",
        "Cb": "C0 + (t>=t_step)*(C_step-C0)",
        "Cb_M": "Cb/MW",
        "R_bio0": "dVT_max*theta/V_ov",            # ideal, motion-free reading
        "R_bio": "R_bio0*xtalk",
        "R_sens": "R_bio + dI_strain",
        "R_ref": "dI_strain*(1+mism)",
        "R_twin": "R_sens - R_ref",
        "th_app": "max(R_sens*V_ov/dVT_max, 0)",
        "sat": "th_app>=1",
        "C_app": "K_D*th_app/max(1-th_app, 1e-12)",
        "C_false": "(C_app - Cb_M)*MW",
        "S_lin": "FS*K_D/(Cb_M+K_D)^2",
        "C_false_lin": "dI_strain/S_lin*MW",
        "art_FS": "dI_strain/FS",
        "resid_twin": "abs(dI_strain*mism + kappa_x*eps*R_bio0)/FS",
        "resid_twin_noX": "abs(dI_strain*mism)/FS",
        "xt_floor": "kappa_x*eps*R_bio0/FS",
        "tau_on": "1/(k_on*Cb_M + k_off)",
        "K_D_ngml": "K_D*MW",
        "john_sum": "Phi_lit + (-dPhiB_ps/1[V])*0.01*1000",
        "li_xt": "kappa_x*0.0034*100",
    }
