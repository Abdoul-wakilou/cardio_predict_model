"""Évalue les modèles sur le jeu de test, mis de côté jusqu'à cette étape."""

import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from src.config import (
    FEATURE_ORDER,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    SELECTION_PATH,
    TEST_RESULTS_PATH,
)
from src.figures import BLUE, MODEL_STYLES, apply_style, save_figure
from src.modeling import CANDIDATES, MODEL_NAMES, build_pipeline, compute_metrics
from src.preprocessing import load_dataset, split_dataset

logger = logging.getLogger(__name__)

N_BOOTSTRAP = 2000
INTERVAL_METRICS = ("accuracy", "auc", "sensibilite", "specificite")


def fit_candidate(name, parameters, X_train, y_train):
    """Réentraîne un candidat sur le train avec ses paramètres sélectionnés."""
    pipeline = build_pipeline(clone(CANDIDATES[name][0]))
    pipeline.set_params(**{f"model__{key}": value for key, value in parameters.items()})
    return pipeline.fit(X_train, y_train)


def bootstrap_intervals(y_true, probabilities):
    """Estime les intervalles de confiance à 95 % par rééchantillonnage des patients."""
    y_true, probabilities = np.asarray(y_true), np.asarray(probabilities)
    rng = np.random.default_rng(RANDOM_STATE)
    samples = {metric: [] for metric in INTERVAL_METRICS}
    while len(samples["auc"]) < N_BOOTSTRAP:
        indices = rng.integers(0, y_true.size, y_true.size)
        if np.unique(y_true[indices]).size < 2:
            continue
        metrics = compute_metrics(y_true[indices], probabilities[indices])
        for metric in INTERVAL_METRICS:
            samples[metric].append(metrics[metric])
    return {
        metric: [float(value) for value in np.percentile(values, [2.5, 97.5])]
        for metric, values in samples.items()
    }


def plot_confusion_matrices(y_test, probabilities_by_model):
    """Trace une matrice de confusion par modèle."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, (name, probabilities) in zip(axes, probabilities_by_model.items(), strict=True):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            (probabilities >= 0.5).astype(int),
            display_labels=["Sain", "Malade"],
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(MODEL_NAMES[name])
        axis.set_xlabel("Classe prédite")
        axis.set_ylabel("Classe réelle")
        axis.grid(False)
    save_figure(figure, "matrices_confusion.png")


def plot_roc_curves(y_test, probabilities_by_model, results):
    """Compare les courbes ROC des trois modèles."""
    figure, axis = plt.subplots(figsize=(7.5, 6))
    for name, probabilities in probabilities_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        label = f"{MODEL_NAMES[name]} (AUC = {results[name]['auc']:.3f})"
        axis.plot(fpr, tpr, label=label, **MODEL_STYLES[name])
    axis.plot([0, 1], [0, 1], color="#9a9890", linewidth=1, label="Hasard (AUC = 0.5)")
    axis.set_xlabel("1 − spécificité")
    axis.set_ylabel("Sensibilité")
    axis.legend(loc="lower right", frameon=False)
    save_figure(figure, "courbe_roc_comparative.png")


def assess_calibration(y_test, probabilities, name):
    """Compare les probabilités annoncées aux fréquences observées."""
    observed, predicted = calibration_curve(y_test, probabilities, n_bins=5, strategy="quantile")
    figure, axis = plt.subplots(figsize=(6, 5.5))
    axis.plot([0, 1], [0, 1], color="#9a9890", linewidth=1, label="Calibration parfaite")
    axis.plot(predicted, observed, marker="o", markersize=8, label=MODEL_NAMES[name], color=BLUE)
    axis.set_xlabel("Probabilité moyenne prédite")
    axis.set_ylabel("Fréquence observée de la maladie")
    axis.legend(loc="upper left", frameon=False)
    save_figure(figure, "courbe_calibration.png")
    return {
        "probabilite_predite": [float(value) for value in predicted],
        "frequence_observee": [float(value) for value in observed],
    }


def assess_importance(model, X_test, y_test):
    """Mesure la perte d'AUC quand chaque variable clinique est permutée."""
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=RANDOM_STATE
    )
    order = np.argsort(result.importances_mean)
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    axis.barh(
        np.array(FEATURE_ORDER)[order],
        result.importances_mean[order],
        xerr=result.importances_std[order],
        color=BLUE,
        height=0.55,
        error_kw={"ecolor": "#52514e", "elinewidth": 1},
    )
    axis.set_xlabel("Baisse moyenne de l'AUC après permutation")
    axis.grid(axis="y", visible=False)
    save_figure(figure, "importance_permutation.png")
    ranking = sorted(
        zip(FEATURE_ORDER, result.importances_mean, result.importances_std, strict=True),
        key=lambda item: -item[1],
    )
    return [
        {"variable": variable, "baisse_auc": float(mean), "ecart_type": float(std)}
        for variable, mean, std in ranking
    ]


def main():
    """Produit les métriques finales, les figures et complète les métadonnées."""
    apply_style()
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selected = selection["modele_retenu"]
    X_train, X_test, y_train, y_test = split_dataset(load_dataset())

    models = {
        name: fit_candidate(name, summary["parametres"], X_train, y_train)
        for name, summary in selection["candidats"].items()
        if name != selected
    }
    # Le modèle retenu est relu depuis son artefact : je mesure ce que l'API sert.
    models[selected] = joblib.load(MODEL_PATH)

    probabilities = {
        name: models[name].predict_proba(X_test)[:, 1] for name in selection["candidats"]
    }
    results = {
        name: {
            **compute_metrics(y_test, values),
            "intervalle_confiance_95": bootstrap_intervals(y_test, values),
        }
        for name, values in probabilities.items()
    }
    for name, metrics in results.items():
        logger.info(
            "%-22s acc=%.4f auc=%.4f sens=%.4f spec=%.4f brier=%.4f %s",
            MODEL_NAMES[name],
            metrics["accuracy"],
            metrics["auc"],
            metrics["sensibilite"],
            metrics["specificite"],
            metrics["brier"],
            metrics["matrice_confusion"],
        )

    plot_confusion_matrices(y_test, probabilities)
    plot_roc_curves(y_test, probabilities, results)
    calibration = assess_calibration(y_test, probabilities[selected], selected)
    importance = assess_importance(models[selected], X_test, y_test)

    TEST_RESULTS_PATH.write_text(
        json.dumps(
            {
                "modele_retenu": selected,
                "modeles": results,
                "calibration": calibration,
                "importance_permutation": importance,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["resultats_test"] = results[selected]
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
