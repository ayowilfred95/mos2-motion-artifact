"""
Convert the exports of the Java classes (comsol/data/java/) into the raw-export format used by
assemble.py, then run the post-processing stage (results.json, comparison.csv, figures, methods).

Usage: python comsol/import_java.py
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JDIR = os.path.join(HERE, "data", "java")
DATA = os.path.join(HERE, "data")


def read_arrays(path):
    out = {}
    with open(path) as f:
        for line in f:
            key, _, vals = line.rstrip("\n").partition("\t")
            if key:
                out[key] = np.array([float(v) for v in vals.split(",")]) if vals else np.array([])
    return out


def main():
    need = ["raw_mech.json", "raw_chain.json", "sweeps.txt", "time_baseline.txt", "time_thin.txt",
            "time_island.txt", "motion.txt"]
    missing = [n for n in need if not os.path.exists(os.path.join(JDIR, n))]
    if missing:
        sys.exit(f"missing Java exports in {JDIR}: {missing} (run MechModel and ChainModel first)")
    mech = json.load(open(os.path.join(JDIR, "raw_mech.json")))
    chain = json.load(open(os.path.join(JDIR, "raw_chain.json")))
    dpath = os.path.join(JDIR, "raw_debye.json")
    debye = json.load(open(dpath)) if os.path.exists(dpath) else None
    runtimes = {}
    runtimes.update(mech.get("runtimes", {}))
    runtimes.update(chain.pop("runtimes", {}))
    if debye:
        runtimes["debye"] = debye.pop("debye_s", None)
    raw = dict(mech=mech, chain=chain, debye=debye, runtimes=runtimes,
               comsol_version="COMSOL Java API (compiled classes, comsol batch)")
    json.dump(raw, open(os.path.join(DATA, "raw.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DATA, "raw_sweeps.npz"), **read_arrays(os.path.join(JDIR, "sweeps.txt")))
    for tag in ("baseline", "thin", "island"):
        np.savez_compressed(os.path.join(DATA, f"raw_time_{tag}.npz"), **read_arrays(os.path.join(JDIR, f"time_{tag}.txt")))
    np.savez_compressed(os.path.join(DATA, "raw_motion.npz"), **read_arrays(os.path.join(JDIR, "motion.txt")))
    print("converted Java exports -> comsol/data/raw.json + raw_*.npz")
    subprocess.run([sys.executable, os.path.join(HERE, "run_all.py"), "--stage", "post"], check=True)


if __name__ == "__main__":
    main()
