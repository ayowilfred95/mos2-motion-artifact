"""
Stepped bending motions solved in Elmer (21 "Scanning" steps per cycle) for the ParaView animations.

  baseline : wrist flexion, elbow flexion, walking
  PET 2.5 um : wrist flexion
  rigid island (100 um PDMS) : wrist flexion (load scaled so the strip-average curvature is the wrist value)

Check after each run: strain per curvature at the MoS2 mid-plane at every bent step must equal the static
Elmer value (the model is linear). Writes elmer/data/motions.json.
Usage: python elmer/run_motions.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "comsol"),
                os.path.join(os.path.dirname(HERE), "fem")]
import make_models as mm        # noqa: E402
import run_elmer as re_         # noqa: E402

N = 21
H_M = re_.H_M
mech = json.load(open(os.path.join(HERE, "data", "raw_mech.json")))["mech"]
KMEAN_STATIC = mech["island"]["k_mean"]          # strip-average curvature of the static island solve (1/m)

RUNS = [  # name, stack or "island", motion, title
    ("wrist_motion", "baseline", "wrist", "Elmer: baseline sensor, wrist flexion (flat -> R = 25 mm -> flat)"),
    ("elbow_motion", "baseline", "elbow", "Elmer: baseline sensor, elbow flexion (flat -> R = 40 mm -> flat)"),
    ("walking_motion", "baseline", "walking", "Elmer: baseline sensor, walking arm swing (R = 300 mm both ways, 1.8 Hz)"),
    ("pet_wrist_motion", "pet_2p5", "wrist", "Elmer: 2.5 um PET sensor, wrist flexion (flat -> R = 25 mm -> flat)"),
    ("island_wrist_motion", "island", "wrist", "Elmer: rigid-island sensor (100 um PDMS), wrist flexion (strip-average R = 25 mm)"),
]


def check(d, t_sub, motion):
    """MoS2-plane strain at each step; per-curvature values of the bent steps."""
    a = np.loadtxt(os.path.join(d, "results", "lines.dat"))
    mo = mm.MOTIONS[motion]
    k_peak = 1 / mo["R"]
    out = []
    for i in range(1, N + 1):
        m = (a[:, 0] == i) & np.isclose(a[:, 3], 1e-9, atol=1e-12) & (a[:, 4] >= 0) & (a[:, 4] <= t_sub)
        y, e = a[m, 4], a[m, 13]
        k, c = np.polyfit(y, e, 1)
        tau = (i - 1) / (N - 1)
        f = 0.5 * (1 - np.cos(2 * np.pi * tau)) if mo["kind"] == "flex" else np.sin(2 * np.pi * tau)
        e_m = np.polyval([k, c], t_sub + H_M)
        out.append(dict(step=i, t=tau * mo["T"], kappa_applied=k_peak * f, eps_mos2=float(e_m),
                        epk_um=float(e_m / (k_peak * f) * 1e6) if abs(f) > 1e-9 else None))
    return out


def main():
    res = {}
    for name, stack, motion, title in RUNS:
        t0 = time.perf_counter()
        if stack == "island":
            d, info = mm.model_island(name=name, title=title, steps=N, motion=motion,
                                      load_scale=(1 / mm.R_WRIST) / KMEAN_STATIC)
            t_sub = info["t_sub"]
        else:
            layers, _ = mm.STACKS[stack]
            d, info = mm.model(name, layers, title, steps=N, motion=motion)
            t_sub = layers[0][1]
        re_.elmer(d, name)
        steps = check(d, t_sub, motion)
        epks = [s["epk_um"] for s in steps if s["epk_um"] is not None]
        res[name] = dict(stack=stack, motion=motion, title=title, t_sub=t_sub, steps=steps,
                         epk_um_min=min(epks), epk_um_max=max(epks), time_s=time.perf_counter() - t0)
        print(f"{name:20s} {time.perf_counter()-t0:6.0f} s   eps/kappa at the MoS2 plane, bent steps: "
              f"{min(epks):.4f} ... {max(epks):.4f} um   peak MoS2 strain {max(abs(s['eps_mos2']) for s in steps)*100:.4f} %",
              flush=True)
    json.dump(res, open(os.path.join(HERE, "data", "motions.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
