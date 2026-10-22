"""Figure: rigid-island channel strain vs PDMS carrier thickness (from elmer/data/island_sweep.json)."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "data", "island_sweep.json")))
t = [r["t_pdms_um"] for r in d["sweep"]]
ratio = [r["ratio_to_baseline"] * 100 for r in d["sweep"]]
kr = [r["k_island_over_mean"] * 100 for r in d["sweep"]]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.45, 2.6))
ax.plot(t, ratio, "o-", color="#c0392b", label="channel strain / baseline (Elmer)")
ax.plot(t, kr, "s--", color="#2980b9", label="island curvature / strip mean (Elmer)")
ax.axhline(d["published_ratio_pct"], color="k", ls=":", lw=0.8)
ax.text(t[-1], d["published_ratio_pct"] * 1.12, "stretching isolation ratio of ref. [14] (4.2 %)", fontsize=6.5, ha="right")
ax.axhline(100, color="#888", lw=0.6)
ax.text(t[0], 106, "no island (baseline)", fontsize=6.5, color="#555")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("PDMS carrier thickness (um)"); ax.set_ylabel("percent")
ax.set_xticks(t); ax.set_xticklabels([f"{v:g}" for v in t])
ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.93)); ax.grid(alpha=0.3, lw=0.4, which="both")
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "island_pdms_sweep.png"), dpi=1200)
print("wrote elmer/figures/island_pdms_sweep.png")
