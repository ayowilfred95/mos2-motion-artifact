"""
Additional Elmer runs for the reviewer's comments (pre-submission review, 2026-10-22).

  R7  mesh convergence of the 2.5 um PET stack (same 4 levels as the baseline) and of the rigid-island strip
      (mesh scale 0.5, 1, 2), with the convergence metric: relative change of eps/kappa against the finest level.
  R8  rigid island under a curvature imposed by the skin on the PDMS underside, bonded (no slip) and sliding,
      each with and without the island (island replaced by PDMS on the same mesh), next to the moment-loaded case.

Writes elmer/data/revision_runs.json. Usage: python elmer/revision_runs.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "comsol"),
                os.path.join(os.path.dirname(HERE), "fem")]
import make_models as mm   # noqa: E402
import run_elmer as re_    # noqa: E402

BASE_EPK = 11.556111380185293e-6     # baseline eps/kappa (bonded), production mesh, elmer/data/raw_mech.json


def pet_convergence():
    layers, title = mm.STACKS["pet_2p5"]
    rows = []
    for lev, (nx, npet) in enumerate([(1700, 4), (2400, 6), (3400, 8), (4800, 12)], start=1):
        mm.STACKS["pet_2p5"] = ([("PET", 2.5e-6, npet)] + mm.FILMS, title)
        r = re_.run_slice("pet_2p5", nx)
        rows.append(dict(level=lev, elements=r["elements"], x_elem_nm=mm.L_SEG / nx * 1e9, pet_rows=npet,
                         epk_bonded_um=r["epk_bonded"] * 1e6, zn_um=r["zn"] * 1e6))
        re_.log(f"PET level {lev}: {r['elements']} elements, eps/kappa {r['epk_bonded']*1e6:.6f} um")
    mm.STACKS["pet_2p5"] = (layers, title)
    for r in rows:
        r["change_vs_finest_pct"] = (r["epk_bonded_um"] / rows[-1]["epk_bonded_um"] - 1) * 100
    return rows


def island_convergence():
    rows = []
    for scale in (0.5, 1, 2):
        r = re_.run_island(scale=scale, name=f"rigid_island_conv{str(scale).replace('.', 'p')}")
        rows.append(dict(scale=scale, elements=r["elements"], ratio_to_baseline=r["epk_bonded"] * 0.8 / (0.8 * BASE_EPK),
                         k_local_over_mean=r["k_local_over_mean"]))
        re_.log(f"island scale {scale}: {r['elements']} elements, channel strain/baseline {rows[-1]['ratio_to_baseline']*100:.4f} %")
    for r in rows:
        r["change_vs_finest_pct"] = (r["ratio_to_baseline"] / rows[-1]["ratio_to_baseline"] - 1) * 100
    return rows


def skin_curvature():
    """Channel strain per imposed curvature at the channel centre (x = 0)."""
    import numpy as np
    out = {}
    k = 1 / mm.R_WRIST
    for loading in ("skin_bonded", "skin_sliding"):
        for island in (True, False):
            name = f"island_{loading}_{'with' if island else 'without'}"
            d, info = mm.model_island(name=name, loading=loading, island=island, kappa=k,
                                      title=f"Rigid island strip, {loading}, island={'yes' if island else 'no (PDMS)'}")
            re_.elmer(d, name)
            a = np.loadtxt(os.path.join(d, "results", "lines.dat"))
            x, y, e = a[:, 3], a[:, 4], a[:, 13]
            ts = info["t_sub"]
            thick = np.isclose(x, 1e-9, atol=1e-12) & (y >= 0) & (y <= ts)
            k0, c0 = np.polyfit(y[thick], e[thick], 1)
            e_m = np.polyval([k0, c0], ts + re_.H_M)
            out[name] = dict(loading=loading, island=island, kappa_imposed=k, kappa_local=k0,
                             eps_mos2=e_m, epk_um=e_m / k * 1e6, ratio_to_baseline=e_m / k / BASE_EPK)
            re_.log(f"{name}: local/imposed curvature {k0/k:.4f}, eps/kappa {e_m/k*1e6:.4f} um, "
                    f"vs baseline {e_m/k/BASE_EPK*100:.1f} %")
    return out


if __name__ == "__main__":
    res = dict(pet_convergence=pet_convergence(), island_convergence=island_convergence(),
               skin_curvature=skin_curvature(),
               solver="ElmerSolver 26.2 StressSolver, plane strain, linear system: direct (UMFPACK), "
                      "quadratic 8-node quadrilaterals; no iterative tolerance applies")
    json.dump(res, open(os.path.join(HERE, "data", "revision_runs.json"), "w"), indent=1)
    re_.log("wrote elmer/data/revision_runs.json")
