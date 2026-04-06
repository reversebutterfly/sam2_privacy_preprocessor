"""
Fig 2: Summary table — methods x metrics, rendered as a clean matplotlib figure.
"""
import sys
sys.path.insert(0, "E:/PycharmProjects/pythonProject/sam2_privacy_preprocessor/results/figures")
from paper_plot_style import *
import numpy as np

METHODS = ['B1a UAP', 'B1b UAP+LPIPS', 'B2 Stage1', 'B3 Stage3']
METRICS = [u'\u03b4JF_adv \u2191', u'\u03b4JF_codec \u2191', 'SSIM \u2191', 'PSNR (dB)']

# Mean over 4 normal videos (bike-packing, blackswan, bmx-bumps, bus)
raw = [
    [ 0.0155,  0.0045, 0.885, 34.1],   # B1a
    [ 0.0098,  0.0033, 0.881, 33.9],   # B1b
    [-0.0093, -0.0135, 0.938, 32.3],   # B2
    [-0.0243, -0.0060, 0.928, 32.1],   # B3
]
data = np.array(raw)

cell_text = []
for i, method in enumerate(METHODS):
    row = []
    for j in range(len(METRICS)):
        v = data[i, j]
        if j < 2:
            row.append(f'{v:+.3f}')
        elif j == 2:
            row.append(f'{v:.3f}')
        else:
            row.append(f'{v:.1f}')
    cell_text.append(row)

# Per-column normalize for background coloring
normed = np.zeros_like(data)
for col in range(data.shape[1]):
    mn, mx = data[:, col].min(), data[:, col].max()
    normed[:, col] = (data[:, col] - mn) / (mx - mn + 1e-9)

cmap = matplotlib.colormaps.get_cmap('RdYlGn')
cell_colors = []
for i in range(len(METHODS)):
    row_c = []
    for j in range(len(METRICS)):
        rgba = cmap(normed[i, j])
        # Return as hex to avoid tuple length issues
        row_c.append(matplotlib.colors.to_hex(rgba))
    cell_colors.append(row_c)

row_colors = [METHOD_COLORS[m] for m in METHODS]
col_colors = ['#cccccc'] * len(METRICS)

fig, ax = plt.subplots(figsize=(7.5, 2.2))
ax.axis('off')

table = ax.table(
    cellText=cell_text,
    rowLabels=METHODS,
    colLabels=METRICS,
    cellColours=cell_colors,
    rowColours=row_colors,
    colColours=col_colors,
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.0, 2.2)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#888888')
    cell.set_linewidth(0.5)
    if col == -1:
        cell.set_text_props(fontweight='bold', color='white')
    if row == 0:
        cell.set_text_props(fontweight='bold')

fig.tight_layout()
save_fig(fig, 'fig2_summary_table')
