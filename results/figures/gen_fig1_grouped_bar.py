"""
Fig 1: Grouped bar chart — δJF_adv and δJF_codec (CRF18) across methods and videos.
"""
import sys
sys.path.insert(0, "E:/PycharmProjects/pythonProject/sam2_privacy_preprocessor/results/figures")
from paper_plot_style import *
import numpy as np

VIDEOS = ['bike-packing', 'blackswan', 'bmx-bumps', 'bus']
VIDEO_LABELS = ['bike-\npacking', 'blackswan', 'bmx-\nbumps', 'bus']

DATA = {
    'B1a UAP': {
        'adv':   [ 0.024,  0.030,  0.005,  0.003],
        'codec': [ 0.025, -0.001, -0.008,  0.002],
    },
    'B1b UAP+LPIPS': {
        'adv':   [ 0.051, -0.023,  0.007,  0.004],
        'codec': [ 0.032, -0.010, -0.010,  0.001],
    },
    'B2 Stage1': {
        'adv':   [ 0.037, -0.031, -0.058,  0.015],
        'codec': [-0.005, -0.018, -0.057,  0.008],
    },
    'B3 Stage3': {
        'adv':   [ 0.050, -0.028, -0.137,  0.018],
        'codec': [ 0.026, -0.010, -0.055,  0.015],
    },
}

METHODS = list(DATA.keys())
n_videos = len(VIDEOS)
n_methods = len(METHODS)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)

bar_width = 0.18
x = np.arange(n_videos)

for ax, key, title in zip(axes, ['adv', 'codec'], ['Pre-codec (δJF$_{adv}$)', 'Post-H.264 (δJF$_{codec}$, CRF=18)']):
    for i, method in enumerate(METHODS):
        offsets = (i - (n_methods - 1) / 2) * bar_width
        vals = DATA[method][key]
        color = METHOD_COLORS[method]
        bars = ax.bar(x + offsets, vals, bar_width,
                      label=method, color=color, alpha=0.88, edgecolor='white', linewidth=0.5)
        # Value labels
        for bar, v in zip(bars, vals):
            ha = 'center'
            va = 'bottom' if v >= 0 else 'top'
            offset_y = 0.002 if v >= 0 else -0.002
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + offset_y,
                    f'{v:.2f}', ha=ha, va=va, fontsize=7, rotation=90)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(VIDEO_LABELS)
    ax.set_ylabel('J&F Drop (δJF) ↑ better attack')
    ax.set_title(title)
    ax.set_xlim(-0.5, n_videos - 0.5)

axes[0].legend(frameon=False, loc='upper right', ncol=2)

fig.tight_layout(pad=1.5)
save_fig(fig, 'fig1_grouped_bar')
