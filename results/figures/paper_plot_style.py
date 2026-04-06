import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

FIG_DIR = "E:/PycharmProjects/pythonProject/sam2_privacy_preprocessor/results/figures"
COLORS = ['#4878D0', '#EE854A', '#6ACC65', '#D65F5F']  # blue, orange, green, red
METHOD_COLORS = {
    'B1a UAP':       '#4878D0',
    'B1b UAP+LPIPS': '#EE854A',
    'B2 Stage1':     '#6ACC65',
    'B3 Stage3':     '#D65F5F',
}

def save_fig(fig, name):
    path = f"{FIG_DIR}/{name}.pdf"
    fig.savefig(path)
    path_png = f"{FIG_DIR}/{name}.png"
    fig.savefig(path_png)
    print(f"Saved: {name}.pdf / .png")
