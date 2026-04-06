"""
Generate all auto-generated figures for the paper.
Run from the project root:  python figures/generate_figures.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__))
os.makedirs(OUT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

PIXEL_LEVEL = [
    # (label, dJF_adv, dJF_auc)
    ("C1 (clip)", 0.016, 0.004),
    ("C2 (video)", 0.010, 0.004),
    ("C3 (full)", 0.005, 0.004),
]

FEATURE_TAGS = ["F0_B", "F0_CD_nm", "F0_CD"]
FEATURE_LABELS = ["F0-B\n(FPN)", "F0-CDₙₘ\n(C+D, no match)", "F0-CD\n(Full C+D)"]
results = {}
for tag in FEATURE_TAGS:
    p = f"results_v100/attack_frame0_{tag}/results.json"
    if os.path.exists(p):
        with open(p) as f:
            results[tag] = json.load(f)

# ── Figure 2: Main results bar chart ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))

all_labels = [x[0] for x in PIXEL_LEVEL] + FEATURE_LABELS
all_adv    = [x[1] for x in PIXEL_LEVEL] + [
    results[t]["mean_djf_adv_valid"] for t in FEATURE_TAGS
]
all_auc    = [x[2] for x in PIXEL_LEVEL] + [
    results[t]["mean_djf_attack_under_codec"] for t in FEATURE_TAGS
]

N = len(all_labels)
x = np.arange(N)
w = 0.36

bars_adv = ax.bar(x - w/2, all_adv, w, color="#4C8BB5", label="dJF (no codec)", zorder=3)
bars_auc = ax.bar(x + w/2, all_auc, w, color="#E07B35", label="dJF under CRF23 (primary)", zorder=3)

ax.axhline(0.05, color="#2ca02c", linestyle="--", lw=1.4, label="Proceed threshold (0.05)")
ax.axhline(0.03, color="#d62728", linestyle=":",  lw=1.4, label="Kill threshold (0.03)")
ax.axhline(0.00, color="black",   linestyle="-",  lw=0.8)

# Shade negative region
ax.axhspan(-0.12, 0.0, alpha=0.06, color="#d62728")

ax.set_xticks(x)
ax.set_xticklabels(all_labels, fontsize=9)
ax.set_ylabel("Tracking degradation (dJF, ↑ = more attack damage)", fontsize=9)
ax.set_title("Attack Effectiveness Before and After H.264 CRF23 Compression", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(-0.12, 0.20)
ax.grid(axis="y", alpha=0.3, zorder=0)

# Divider between pixel-level and feature-space
ax.axvline(2.5, color="gray", linestyle="-", lw=0.8, alpha=0.5)
ax.text(1.0, 0.18, "Pixel-level (g_θ CNN)", ha="center", fontsize=8, color="gray",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8))
ax.text(4.0, 0.18, "Feature-space (frame-0 direct opt.)", ha="center", fontsize=8, color="gray",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8))

# Bus annotation
bus_idx = PIXEL_LEVEL.__len__() + FEATURE_TAGS.index("F0_CD")
ax.annotate("bus: dJF_adv=0.87\n(truncated)",
            xy=(bus_idx - w/2, 0.095),
            xytext=(bus_idx - w/2 - 1.0, 0.14),
            fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8),
            ha="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_main_results.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "fig2_main_results.png"), bbox_inches="tight", dpi=200)
plt.close()
print("Saved fig2_main_results.pdf")

# ── Figure 4: Per-video scatter (dJF_adv vs dJF_auc) ─────────────────────────

fig, ax = plt.subplots(figsize=(5.5, 5.5))

colors = {"F0_B": "#4C8BB5", "F0_CD_nm": "#E07B35", "F0_CD": "#2ca02c"}
markers = {"F0_B": "o", "F0_CD_nm": "s", "F0_CD": "^"}

for tag in FEATURE_TAGS:
    per_video = results[tag]["per_video"]
    xs = [r["dJF_adv"] for r in per_video]
    ys = [r["dJF_attack_under_codec"] for r in per_video]
    label = {"F0_B": "F0-B (FPN)", "F0_CD_nm": "F0-CDₙₘ (C+D nm)", "F0_CD": "F0-CD (Full C+D)"}[tag]
    ax.scatter(xs, ys, c=colors[tag], marker=markers[tag], s=55, label=label,
               zorder=3, alpha=0.85)
    # Annotate bus video in F0_CD
    if tag == "F0_CD":
        for r in per_video:
            if r["video"] == "bus":
                ax.annotate("bus", (r["dJF_adv"], r["dJF_attack_under_codec"]),
                            textcoords="offset points", xytext=(6, 4), fontsize=7)

# Reference lines
x_range = np.linspace(-0.15, 0.92, 100)
ax.plot(x_range, x_range, "k--", lw=1.0, label="y = x (perfect codec robustness)", alpha=0.4)
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)

# Shade regions
ax.fill_between([-0.15, 0.92], [0.0, 0.0], [0.25, 0.25], alpha=0.04, color="#2ca02c",
                label="Positive attack (codec)")
ax.fill_between([-0.15, 0.92], [-0.25, -0.25], [0.0, 0.0], alpha=0.04, color="#d62728",
                label="Negative attack (codec)")

ax.set_xlabel("dJF without codec (attack effective in isolation)", fontsize=9)
ax.set_ylabel("dJF under CRF23 (primary metric)", fontsize=9)
ax.set_title("Per-Video: Attack Success vs. Codec Survival", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(-0.12, 0.94)
ax.set_ylim(-0.12, 0.12)
ax.grid(alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_scatter.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "fig4_scatter.png"), bbox_inches="tight", dpi=200)
plt.close()
print("Saved fig4_scatter.pdf")

# ── Figure 5 / Table 1: Per-video breakdown (saved as CSV for LaTeX) ──────────

import csv

videos_ordered = [
    "cows", "car-turn", "car-roundabout", "crossing",
    "bus", "blackswan", "classic-car", "color-run", "bike-packing"
]

rows = {}
for tag in FEATURE_TAGS:
    for r in results[tag]["per_video"]:
        v = r["video"]
        if v not in rows:
            rows[v] = {
                "jf_clean": r["jf_clean"],
                "ssim_b": None, "ssim_cd": None, "ssim_cd_nm": None,
            }
        if tag == "F0_B":
            rows[v]["dJF_adv_b"] = r["dJF_adv"]
            rows[v]["dJF_auc_b"] = r["dJF_attack_under_codec"]
            rows[v]["ssim_b"] = r["ssim_f0"]
        elif tag == "F0_CD_nm":
            rows[v]["dJF_adv_cd_nm"] = r["dJF_adv"]
            rows[v]["dJF_auc_cd_nm"] = r["dJF_attack_under_codec"]
            rows[v]["ssim_cd_nm"] = r["ssim_f0"]
        elif tag == "F0_CD":
            rows[v]["dJF_adv_cd"] = r["dJF_adv"]
            rows[v]["dJF_auc_cd"] = r["dJF_attack_under_codec"]
            rows[v]["ssim_cd"] = r["ssim_f0"]

# Write LaTeX table to file
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\caption{Per-video results for all three feature-space attack modes on 9 DAVIS validation videos (CRF 23). $\Delta\text{JF}_\text{adv}$: tracking degradation without codec (↑ better attack). $\Delta\text{JF}_\text{auc}$: tracking degradation under CRF23 (↑ better attack). All $\Delta\text{JF}_\text{auc}$ values are near zero, confirming the codec-kill hypothesis.}")
lines.append(r"\label{tab:per_video}")
lines.append(r"\begin{tabular}{lc|cc|cc|cc}")
lines.append(r"\toprule")
lines.append(r"& & \multicolumn{2}{c|}{F0-B (FPN)} & \multicolumn{2}{c|}{F0-CD$_\text{nm}$} & \multicolumn{2}{c}{F0-CD (Full)} \\")
lines.append(r"Video & JF$_\text{clean}$ & $\Delta$JF$_\text{adv}$ & $\Delta$JF$_\text{auc}$ & $\Delta$JF$_\text{adv}$ & $\Delta$JF$_\text{auc}$ & $\Delta$JF$_\text{adv}$ & $\Delta$JF$_\text{auc}$ \\")
lines.append(r"\midrule")
for v in videos_ordered:
    r = rows[v]
    bus_note = r" \dagger" if v == "bus" else ""
    jf_c = r["jf_clean"]
    lines.append(
        f"{v.replace('-', '-')}{bus_note} & {jf_c:.3f} "
        f"& {r.get('dJF_adv_b', 0):+.3f} & {r.get('dJF_auc_b', 0):+.3f} "
        f"& {r.get('dJF_adv_cd_nm', 0):+.3f} & {r.get('dJF_auc_cd_nm', 0):+.3f} "
        f"& {r.get('dJF_adv_cd', 0):+.3f} & {r.get('dJF_auc_cd', 0):+.3f} \\\\"
    )
lines.append(r"\midrule")
# Means
means_b_adv  = np.mean([rows[v].get("dJF_adv_b", 0) for v in videos_ordered])
means_b_auc  = np.mean([rows[v].get("dJF_auc_b", 0) for v in videos_ordered])
means_nm_adv = np.mean([rows[v].get("dJF_adv_cd_nm", 0) for v in videos_ordered])
means_nm_auc = np.mean([rows[v].get("dJF_auc_cd_nm", 0) for v in videos_ordered])
means_cd_adv = np.mean([rows[v].get("dJF_adv_cd", 0) for v in videos_ordered])
means_cd_auc = np.mean([rows[v].get("dJF_auc_cd", 0) for v in videos_ordered])
jf_mean = np.mean([rows[v]["jf_clean"] for v in videos_ordered])
lines.append(
    f"\\textbf{{Mean}} & {jf_mean:.3f} "
    f"& {means_b_adv:+.3f} & {means_b_auc:+.3f} "
    f"& {means_nm_adv:+.3f} & {means_nm_auc:+.3f} "
    f"& {means_cd_adv:+.3f} & {means_cd_auc:+.3f} \\\\"
)
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\begin{flushleft}")
lines.append(r"\footnotesize $\dagger$ Bus video: F0-CD achieves $\Delta$JF$_\text{adv}=+0.867$ (SAM2 tracking collapses from 93.5\% to 6.8\%), but $\Delta$JF$_\text{auc}=-0.002$ after CRF23.")
lines.append(r"\end{flushleft}")
lines.append(r"\end{table}")

with open(os.path.join(OUT, "table1_per_video.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Saved table1_per_video.tex")

# ── latex_includes.tex ────────────────────────────────────────────────────────

includes = r"""% Auto-generated figure includes — paste into LaTeX sections

% Figure 2: Main results bar chart
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{../figures/fig2_main_results.pdf}
\caption{Tracking degradation (dJF) before and after H.264 CRF23 compression across all six experiments. Blue bars: attack effectiveness without codec; orange bars: dJF under CRF23 (primary metric). Dashed lines mark the proceed (0.05) and kill (0.03) thresholds. All orange bars fall below the kill threshold.}
\label{fig:main_results}
\end{figure}

% Figure 4: Per-video scatter
\begin{figure}[t]
\centering
\includegraphics[width=0.85\linewidth]{../figures/fig4_scatter.pdf}
\caption{Per-video dJF without codec (x-axis) vs.\ dJF under CRF23 (y-axis) for the three feature-space attack modes (27 data points). A perfectly codec-robust attack would lie on $y=x$ (dashed line). All attacks collapse toward $y=0$ regardless of insertion point. The bus video (F0-CD, $x=0.87$) dramatically illustrates that optimizer success does not imply codec survival.}
\label{fig:scatter}
\end{figure}
"""

with open(os.path.join(OUT, "latex_includes.tex"), "w", encoding="utf-8") as f:
    f.write(includes)
print("Saved latex_includes.tex")
print("\nAll figures generated.")
