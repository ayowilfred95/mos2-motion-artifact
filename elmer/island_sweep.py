"""Rigid-island sensitivity to the (unspecified) PDMS carrier thickness. Writes elmer/data/island_sweep.json."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "comsol"),
                os.path.join(os.path.dirname(HERE), "fem")]
import run_elmer as re_   # noqa: E402

base = json.load(open(os.path.join(HERE, "data", "raw_mech.json")))["mech"]
epk_base = base["epk"]["Baseline: 25 um PI, 5 nm Al2O3"]
rows = []
for t in (50e-6, 100e-6, 200e-6, 500e-6):
    r = re_.run_island(t_pdms=t, name=f"rigid_island_pdms{int(round(t * 1e6))}")
    ratio = re_.ETA["spring"] * r["epk_bonded"] / epk_base
    rows.append(dict(t_pdms_um=t * 1e6, elements=r["elements"], ratio_to_baseline=ratio,
                     k_island_over_mean=r["k_local_over_mean"], k_free_over_mean=r["k_unsupported"] / r["k_mean"],
                     zn_local_um=r["zn_local_um"], channel_offset_um=r["channel_offset_um"],
                     epk_um=re_.ETA["spring"] * r["epk_bonded"] * 1e6))
    print(f"PDMS {t*1e6:5.0f} um: {r['elements']:6d} elements  channel strain / baseline = {ratio*100:6.2f} %   "
          f"island curvature / mean = {r['k_local_over_mean']*100:5.2f} %   neutral plane under channel at "
          f"{r['zn_local_um']:+.2f} um (channel {r['channel_offset_um']:.2f} um above it)", flush=True)
json.dump(dict(published_ratio_pct=4.2, baseline_epk_um=epk_base * 1e6,
               baseline_channel_offset_um=(25e-6 + re_.H_M) * 1e6 - base["zn_m"] * 1e6, sweep=rows),
          open(os.path.join(HERE, "data", "island_sweep.json"), "w"), indent=1)
