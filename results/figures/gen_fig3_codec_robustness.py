"""
Fig 3: Codec robustness — pre vs post codec δJF for each method,
on the two videos where attack is positive (bike-packing, bus).
Also shows codec survival ratio.
"""
import sys
sys.path.insert(0, "E:/PycharmProjects/pythonProject/sam2_privacy_preprocessor/results/figures")
from paper_plot_style import *
import numpy as np

METHODS = ['B1a UAP', 'B1b UAP+LPIPS', 'B2 Stage1', 'B3 Stage3']

DATA = {
    'bike-packing': {
        'adv':   [0.024, 0.051, 0.037, 0.050],
        'codec': [0.025, 0.032, -0.005, 0.026],
    },
    'bus': {
        'adv':   [0.003, 0.004, 0.015, 0.018],
        'codec': [0.002, 0.001, 0.008, 0.015],
    },
}

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

# ── Panel 1 & 2: pre vs post codec scatter per video ──────────────────────
for ax, (video, vals) in zip(axes[:2], DATA.items()):
    adv = vals['adv']
    codec = vals['codec']
    colors = [METHOD_COLORS[m] for m in METHODS]

    for i, (m, a, c, col) in enumerate(zip(METHODS, adv, codec, colors)):
        ax.scatter(a, c, color=col, s=90, zorder=5, label=m)
        ax.annotate(m.replace(' ', '\n'), (a, c),
                    textcoords='offset points', xytext=(6, 2),
                    fontsize=7, color=col)

    # Identity line y=x (perfect robustness)
    lim = max(max(abs(v) for v in adv + codec), 0.06)
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.8, alpha=0.4, label='y=x (perfect)')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.set_xlim(-lim * 1.3, lim * 1.5)
    ax.set_ylim(-lim * 1.3, lim * 1.5)
    ax.set_xlabel('δJF$_{adv}$ (pre-codec) ↑')
    ax.set_ylabel('δJF$_{codec}$ (post-H.264) ↑')
    ax.set_title(f'Video: {video}')

axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')

# ── Panel 3: Codec survival ratio on bike-packing ─────────────────────────
ax = axes[2]
adv_bp = DATA['bike-packing']['adv']
codec_bp = DATA['bike-packing']['codec']
ratios = [c / a if abs(a) > 1e-4 else 0.0 for a, c in zip(adv_bp, codec_bp)]

x = np.arange(len(METHODS))
bars = ax.bar(x, ratios,
              color=[METHOD_COLORS[m] for m in METHODS],
              alpha=0.88, edgecolor='white', linewidth=0.5)
ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5, label='100% survival')
ax.axhline(0.0, color='gray', linewidth=0.5, alpha=0.4)
for bar, r in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width() / 2,
            r + (0.03 if r >= 0 else -0.07),
            f'{r:.2f}', ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels([m.replace(' ', '\n') for m in METHODS], fontsize=8)
ax.set_ylabel('Codec Survival Ratio\n(δJF$_{codec}$ / δJF$_{adv}$)')
ax.set_title('Codec Robustness — bike-packing')
ax.legend(frameon=False, fontsize=8)

fig.tight_layout(pad=1.5)
save_fig(fig, 'fig3_codec_robustness')
