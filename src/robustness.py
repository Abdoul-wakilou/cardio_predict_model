"""Mesure la stabilité des modèles et leur transfert d'un centre à l'autre.

Les deux analyses refont l'optimisation à l'intérieur de chaque découpage. Elles
décrivent la méthode, sans modifier le modèle servi par l'API. Compter environ
30 minutes sur un ordinateur portable.
"""

import json
import logging

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.config import CENTRE, FEATURE_ORDER, RANDOM_STATE, ROBUSTNESS_PATH, TARGET
from src.modeling import CANDIDATES, MODEL_NAMES, compute_metrics, tune
from src.preprocessing import load_dataset

logger = logging.getLogger(__name__)

N_OUTER_FOLDS = 10
SUMMARY_METRICS = ("accuracy", "auc", "sensibilite", "specificite")


def nested_cross_validation(name, X, y):
    """Estime la performance par validation croisée imbriquée stratifiée à 10 plis."""
    outer = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for train_index, test_index in outer.split(X, y):
        search = tune(name, X.iloc[train_index], y.iloc[train_index])
        probabilities = search.predict_proba(X.iloc[test_index])[:, 1]
        folds.append(compute_metrics(y.iloc[test_index], probabilities))
    return {
        metric: {
            "moyenne": float(np.mean([fold[metric] for fold in folds])),
            "ecart_type": float(np.std([fold[metric] for fold in folds])),
        }
        for metric in SUMMARY_METRICS
    }


def leave_one_centre_out(name, df):
    """Entraîne sur trois centres et teste sur le quatrième, pour chaque centre."""
    results = {}
    for centre in df[CENTRE].unique():
        held_out = df[CENTRE] == centre
        train, test = df[~held_out], df[held_out]
        search = tune(name, train[FEATURE_ORDER], train[TARGET])
        probabilities = search.predict_proba(test[FEATURE_ORDER])[:, 1]
        results[centre] = compute_metrics(test[TARGET], probabilities)
    return results


def main():
    """Exécute les deux analyses pour les trois familles de modèles."""
    df = load_dataset()
    X, y = df[FEATURE_ORDER], df[TARGET]
    report = {"validation_croisee_imbriquee": {}, "validation_par_centre": {}}
    for name in CANDIDATES:
        logger.info("Analyse de robustesse : %s", MODEL_NAMES[name])
        report["validation_croisee_imbriquee"][name] = nested_cross_validation(name, X, y)
        report["validation_par_centre"][name] = leave_one_centre_out(name, df)

    ROBUSTNESS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Rapport écrit : %s", ROBUSTNESS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
