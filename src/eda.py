"""Analyse exploratoire du jeu combiné, à partir des valeurs réellement mesurées.

Cette étape est descriptive : aucune décision de modélisation n'en dépend.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import CENTRE, FEATURE_ORDER, TARGET
from src.figures import BLUE, CLASS_COLORS, apply_style, save_figure
from src.preprocessing import load_dataset

logger = logging.getLogger(__name__)

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def plot_target_distribution(df):
    """Représente la répartition des patients sains et malades."""
    counts = df[TARGET].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    bars = axis.bar(
        ["Sain", "Malade"], counts.values, color=[CLASS_COLORS[0], CLASS_COLORS[1]], width=0.5
    )
    axis.bar_label(bars, labels=[f"{v} ({v / len(df):.1%})" for v in counts.values], padding=3)
    axis.set_ylabel("Nombre de patients")
    axis.grid(axis="x", visible=False)
    save_figure(figure, "distribution_cible.png")


def plot_numeric_distributions(df):
    """Trace histogrammes et boîtes à moustaches des variables continues."""
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.ravel()
    for axis, column in zip(axes, CONTINUOUS, strict=False):
        sns.histplot(df[column].dropna(), kde=True, ax=axis, color=BLUE)
        axis.set_ylabel("Effectif")
    axes[-1].set_visible(False)
    save_figure(figure, "histogrammes_numeriques.png")

    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.ravel()
    for axis, column in zip(axes, CONTINUOUS, strict=False):
        sns.boxplot(
            x=TARGET, y=column, hue=TARGET, data=df, ax=axis, palette=CLASS_COLORS, legend=False
        )
        axis.set_xticks([0, 1], ["Sain", "Malade"])
        axis.set_xlabel("")
    axes[-1].set_visible(False)
    save_figure(figure, "boxplots_numeriques.png")


def plot_correlations(df):
    """Trace les corrélations des variables continues et de ca avec la cible."""
    matrix = df[[*CONTINUOUS, "ca", TARGET]].corr()
    figure, axis = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, ax=axis)
    axis.grid(False)
    save_figure(figure, "matrice_correlation.png")


def plot_missing_by_centre(df):
    """Montre que les données manquantes dépendent fortement du centre."""
    rates = df.groupby(CENTRE)[FEATURE_ORDER].apply(lambda group: group.isna().mean() * 100)
    figure, axis = plt.subplots(figsize=(10, 3.8))
    sns.heatmap(
        rates,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        ax=axis,
        cbar_kws={"label": "% manquant"},
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.grid(False)
    save_figure(figure, "valeurs_manquantes_par_centre.png")


def log_group_comparison(df):
    """Compare sains et malades par test de Welch sur les valeurs observées."""
    rows = []
    for column in CONTINUOUS:
        healthy = df.loc[df[TARGET] == 0, column].dropna()
        sick = df.loc[df[TARGET] == 1, column].dropna()
        _, p_value = stats.ttest_ind(healthy, sick, equal_var=False)
        rows.append(
            {"variable": column, "sain": healthy.mean(), "malade": sick.mean(), "p": p_value}
        )
    logger.info("Comparaison des groupes :\n%s", pd.DataFrame(rows).round(4).to_string(index=False))


def log_prevalence(df):
    """Affiche la prévalence de la maladie pour chaque modalité observée."""
    for column in CATEGORICAL:
        prevalence = df.dropna(subset=[column]).groupby(column)[TARGET].agg(["size", "mean"])
        logger.info("Prévalence selon %s :\n%s", column, prevalence.round(3).to_string())


def main():
    """Produit les statistiques descriptives et les figures exploratoires."""
    apply_style()
    df = load_dataset()
    logger.info("Statistiques descriptives :\n%s", df[CONTINUOUS].describe().round(2).to_string())
    log_group_comparison(df)
    log_prevalence(df)
    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_correlations(df)
    plot_missing_by_centre(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
