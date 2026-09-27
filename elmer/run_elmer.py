"""
Run every Elmer model of the paper and assemble the results.

  1. mesh convergence (baseline slice, 4 refinement levels)
  2. the seven structures (+ nu = 0 and symmetric-stack checks)
  3. sensor chain (Stage 3-6 equations of motion_artifact_sim.py) driven by the Elmer strain per curvature
  4. elmer/results.json (same keys as figures/results_python.json), elmer/comparison.csv, figs 3-8 in
     elmer/figures, elmer/ELMER_METHODS.md

Elmer runs inside the OrbStack Linux machine "elmer" (see HOW_TO_OPEN_ELMER.md).
Usage: python elmer/run_elmer.py
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "comsol"), os.path.join(ROOT, "fem")]

import make_models as mm                                  # noqa: E402
from assemble import STRUCTS, PREBENT_RATIO, assemble    # noqa: E402

DATA = os.path.join(HERE, "data")
RT = {}
H_M = mm.T_AU + mm.T_M / 2                                # MoS2 mid-plane height above the substrate surface
ETA = {"spring": mm.P["eta_thinALD"], "tied": mm.P["eta_polymer"]}
EXTRA_T_ISL = mm.EXTRA["t_isl"][0]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def elmer(d, name):
    """Mesh (ElmerGrid) and solve (ElmerSolver) a model folder inside the Linux machine."""
    cmd = (f"cd '{d}' && rm -rf results {name} && ElmerGrid 1 2 {name}.grd -decimals 12 > elmergrid.log 2>&1 "
           f"&& ElmerSolver case.sif > solver.log 2>&1")
    t0 = time.perf_counter()
    r = subprocess.run(["orb", "-m", "elmer", "bash", "-c", cmd])
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"Elmer failed for {name}; see {d}/solver.log")
    return dt


def mesh_size(d, name):
    h = open(os.path.join(d, name, "mesh.header")).read().split()
    return int(h[0]), int(h[1])            # nodes, elements


def profile(d, x_index=0):
    """Axial strain through the thickness on SaveLine polyline x_index (0 = channel centre)."""
    a = np.loadtxt(os.path.join(d, "results", "lines.dat"))
    xs = np.unique(np.round(a[:, 3], 12))
    m = np.isclose(a[:, 3], xs[x_index], atol=1e-10)
    o = np.argsort(a[m, 4])
    return a[m, 4][o], a[m, 13][o]


def slice_result(d, t_sub, t_cap_top=None):
    """Neutral plane, curvature, strain at the MoS2 mid-plane; x = 0 (identical to x = 50 um in pure bending)."""
    y, e = profile(d, 0)
    sub = y <= t_sub
    k, c = np.polyfit(y[sub], e[sub], 1)
    e_m = np.interp(t_sub + H_M, y, e)
    y2, e2 = profile(d, 1)
    k2 = np.polyfit(y2[y2 <= t_sub], e2[y2 <= t_sub], 1)[0]
    return dict(kappa=k, zn=-c / k, eps_mos2=e_m, epk_bonded=e_m / k, uniformity=abs(k2 / k - 1))


def run_slice(name, nx=mm.NX_SEG):
    layers, title = mm.STACKS[name]
    d, info = mm.model(name, layers, title, nx=nx)
    dt = elmer(d, name)
    RT[f"elmer_{name}"] = RT.get(f"elmer_{name}", 0) + dt
    r = slice_result(d, layers[0][1])
    r["nodes"], r["elements"] = mesh_size(d, name)
    r["time_s"] = dt
    return r


def island_result(d, info):
    """Channel strain and strip-average curvature (total rotation / length) of the rigid-island strip."""
    a = np.loadtxt(os.path.join(d, "results", "lines.dat"))
    x, y, e = a[:, 3], a[:, 4], a[:, 13]
    ts = info["t_sub"]
    thick = np.isclose(x, 1e-9, atol=1e-12) & (y >= 0) & (y <= ts)
    k0, c0 = np.polyfit(y[thick], e[thick], 1)                   # local curvature at the channel
    e_m = np.polyval([k0, c0], ts + H_M)                          # MoS2 mid-plane (bonded films, Kirchhoff)
    bot = np.isclose(y, 1e-9, atol=1e-12)
    top = np.isclose(y, ts - 1e-9, atol=1e-12)
    xb, eb = x[bot][np.argsort(x[bot])], e[bot][np.argsort(x[bot])]
    xt, et = x[top][np.argsort(x[top])], e[top][np.argsort(x[top])]
    k = (np.interp(xb, xt, et) - eb) / (ts - 2e-9)               # curvature along the strip
    far = xb < info["L_half"] - 200e-6                            # exclude the loaded end zone ...
    k_far_end = k[far][-1]
    rot = np.trapz(k[far], xb[far]) + k_far_end * (info["L_half"] - xb[far][-1])   # ... and extend it
    k_mean = rot / info["L_half"]
    return dict(epk_bonded=e_m / k_mean, k_local=k0, k_mean=k_mean, k_local_over_mean=k0 / k_mean,
                k_unsupported=float(np.median(k[(xb > 2.5e-3) & far])))


def run_island(scale=1, t_pdms=None, name="rigid_island"):
    title = None if t_pdms is None else (f"Rigid island: 3 mm x 20 um island (70 GPa) in {t_pdms*1e6:g} um PDMS "
                                         "under 25 um PI")
    kw = dict(name=name, scale=scale, t_pdms=t_pdms)
    if title:
        kw["title"] = title
    d, info = mm.model_island(**kw)
    RT[f"elmer_{name}"] = RT.get(f"elmer_{name}", 0) + elmer(d, name)
    r = island_result(d, info)
    r["nodes"], r["elements"] = mesh_size(d, name)
    # neutral plane under the channel (fit of the through-thickness strain in the island + PI)
    a = np.loadtxt(os.path.join(d, "results", "lines.dat"))
    thick = np.isclose(a[:, 3], 1e-9, atol=1e-12) & (a[:, 4] >= -EXTRA_T_ISL) & (a[:, 4] <= info["t_sub"])
    k, c = np.polyfit(a[thick, 4], a[thick, 13], 1)
    r["zn_local_um"] = -c / k * 1e6                      # measured from the PI bottom (island top); negative = inside island
    r["channel_offset_um"] = (info["t_sub"] + H_M) * 1e6 - r["zn_local_um"]
    return r


def main():
    os.makedirs(DATA, exist_ok=True)
    t_all = time.perf_counter()
    mech = {}

    # ---- 1. mesh convergence: refine along x (element length) and through the substrate together
    log("mesh convergence (baseline)")
    conv = []
    base_layers, base_title = mm.STACKS["baseline"]
    for lev, (nx, npi) in enumerate([(1700, 4), (2400, 6), (3400, 8), (4800, 12)], start=1):
        mm.STACKS["baseline"] = ([("PI", 25e-6, npi)] + mm.FILMS, base_title)
        r = run_slice("baseline", nx)
        conv.append(dict(level=lev, n_elem=r["elements"], dofs=2 * r["nodes"], x_elem_nm=mm.L_SEG / nx * 1e9,
                         pi_rows=npi, epk_tied_um=r["epk_bonded"] * 1e6, epk_spring_um=ETA["spring"] * r["epk_bonded"] * 1e6,
                         zn_um=r["zn"] * 1e6, kappa_fe_over_kappa=1 - r["uniformity"], time_s=r["time_s"]))
        log(f"  level {lev}: {r['elements']} elements, z_n {r['zn']*1e6:.5f} um, eps/kappa (bonded) {r['epk_bonded']*1e6:.5f} um")
    fin = conv[-1]
    for c in conv:
        c["d_spring_pct"] = c["d_tied_pct"] = (c["epk_tied_um"] - fin["epk_tied_um"]) / fin["epk_tied_um"] * 100
    mech["convergence"] = conv
    mech["production_level"] = 3
    mm.STACKS["baseline"] = (base_layers, base_title)

    # ---- 2. structures (production mesh = level 3: 3400 x-elements, 8 substrate rows)
    log("structures")
    res = {n: run_slice(n) for n in ("baseline", "pi_12p5", "pet_2p5", "pmma_cap", "buried", "symmetric")}
    for n, r in res.items():
        log(f"  {n:10s} z_n {r['zn']*1e6:8.4f} um   eps/kappa bonded {r['epk_bonded']*1e6:9.5f} um   uniformity {r['uniformity']:.1e}")
    epk = {STRUCTS[0]: ETA["spring"] * res["baseline"]["epk_bonded"],
           STRUCTS[1]: ETA["tied"] * res["pmma_cap"]["epk_bonded"],
           STRUCTS[2]: ETA["spring"] * res["pi_12p5"]["epk_bonded"],
           STRUCTS[3]: ETA["spring"] * res["pet_2p5"]["epk_bonded"],
           STRUCTS[4]: ETA["tied"] * res["buried"]["epk_bonded"]}

    # rigid island (full 5 mm half strip)
    island_res = run_island()
    epk[STRUCTS[5]] = ETA["spring"] * island_res["epk_bonded"]
    mech["island_local_kappa_ratio"] = island_res["k_local_over_mean"]
    mech["island_ratio_to_baseline"] = epk[STRUCTS[5]] / epk[STRUCTS[0]]
    mech["island"] = island_res
    log(f"  rigid island: local/mean curvature {island_res['k_local_over_mean']:.4f}, "
        f"channel strain ratio to baseline {mech['island_ratio_to_baseline']:.4f}")
    epk[STRUCTS[6]] = epk[STRUCTS[0]] * PREBENT_RATIO

    # nu = 0 verification against 1D laminate theory
    saved = dict(mm.MAT)
    mm.MAT = {k: (E, 0.0, rho) for k, (E, nu_, rho) in saved.items()}
    mm.STACKS["check_nu0"] = (base_layers, "Verification: all Poisson ratios = 0")
    r0 = run_slice("check_nu0")
    mm.MAT = saved
    mech["nu0_check"] = dict(zn_m=r0["zn"], epk_tied_m=r0["epk_bonded"])

    mech.update(zn_m=res["baseline"]["zn"], epk={n: epk[n] for n in STRUCTS},
                sym_epk_m=res["symmetric"]["epk_bonded"], buried_zn_m=res["buried"]["zn"],
                baseline_tied_epk_m=res["baseline"]["epk_bonded"],
                structures={n: {k: float(v) for k, v in r.items()} for n, r in res.items()},
                eta_note="eta applied as a strain-transfer factor on the bonded MoS2 strain (0.8 ALD cap, 1.0 polymer/buried)",
                calib={"k_bare": None, "eta_bare": mm.P["eta_bare"], "k_ALD": None, "eta_ALD": ETA["spring"],
                       "eta_polymer": ETA["tied"]})
    RT["total_mech_s"] = time.perf_counter() - t_all
    mech["runtimes"] = dict(RT)
    json.dump({"mech": mech}, open(os.path.join(DATA, "raw_mech.json"), "w"), indent=1, default=float)

    # ---- 3. sensor chain with the Elmer strain per curvature, then shared post-processing
    import run_fem as rf                     # chain_exports: Stage 3-6 equations, same export format
    rf.DATA, rf.RT = DATA, RT
    rf.chain_exports(mech, None)
    raw = json.load(open(os.path.join(DATA, "raw.json")))
    raw["comsol_version"] = "Elmer FEM 9.0 (ElmerSolver 26.2), StressSolver, 2D plane strain"
    json.dump(raw, open(os.path.join(DATA, "raw.json"), "w"), indent=1, default=float)
    py = json.load(open(os.path.join(ROOT, "figures", "results_python.json")))
    assemble(DATA, HERE, python_results=py)
    subprocess.run([sys.executable, os.path.join(ROOT, "make_figures.py"), "--source", "elmer"], check=True, cwd=ROOT)
    log("done: elmer/results.json, elmer/comparison.csv, elmer/figures")


if __name__ == "__main__":
    main()
