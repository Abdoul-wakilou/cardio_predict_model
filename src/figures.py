"""Style commun des figures du mémoire et du README."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.config import FIGURES_DIR  # noqa: E402

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRID = "#e4e3df"
INK = "#52514e"

# Chaque modèle garde sa couleur et son style de trait d'une figure à l'autre.
MODEL_STYLES = {
    "regression_logistique": {"color": BLUE, "linestyle": "-"},
    "svm": {"color": ORANGE, "linestyle": "--"},
    "random_forest": {"color": AQUA, "linestyle": "-."},
}
CLASS_COLORS = {0: BLUE, 1: ORANGE}


def apply_style():
    """Applique un style sobre, lisible à l'impression."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "lines.linewidth": 2,
            "font.size": 11,
        }
    )


def save_figure(figure, file_name):
    """Enregistre une figure dans reports/figures puis libère la mémoire."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / file_name, dpi=200, bbox_inches="tight")
    plt.close(figure)
