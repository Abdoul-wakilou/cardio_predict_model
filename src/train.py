"""Optimise les trois modèles et choisit le modèle final sur le train uniquement.

Le jeu de test n'est jamais lu ici : il sert une seule fois, dans evaluate.py.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime

import joblib
import sklearn

from src.config import (
    DECISION_THRESHOLD,
    FEATURE_ORDER,
    METADATA_PATH,
    MIN_SPECIFICITY,
    MODEL_PATH,
    SELECTION_PATH,
)
from src.modeling import CANDIDATES, MODEL_NAMES, rank_configuration, tune
from src.preprocessing import load_dataset, split_dataset

logger = logging.getLogger(__name__)

MODEL_VERSION = "4.0.0"
METRICS = ("sensibilite", "specificite", "accuracy", "auc")


def summarize_search(search):
    """Extrait les paramètres retenus et leurs scores moyens de validation croisée."""
    index = search.best_index_
    results = search.cv_results_
    return {
        "parametres": {
            key.replace("model__", ""): value for key, value in search.best_params_.items()
        },
        "validation_croisee": {
            metric: {
                "moyenne": float(results[f"mean_test_{metric}"][index]),
                "ecart_type": float(results[f"std_test_{metric}"][index]),
            }
            for metric in METRICS
        },
    }


def select_candidate(summaries):
    """Applique entre les familles de modèles la même règle qu'entre configurations."""
    names = list(summaries)
    scores = {
        metric: [summaries[name]["validation_croisee"][metric]["moyenne"] for name in names]
        for metric in ("sensibilite", "specificite", "auc")
    }
    return names[rank_configuration(scores["sensibilite"], scores["specificite"], scores["auc"])]


def main():
    """Lance la sélection puis sauvegarde le Pipeline retenu et ses métadonnées."""
    X_train, _, y_train, _ = split_dataset(load_dataset())

    searches = {name: tune(name, X_train, y_train) for name in CANDIDATES}
    summaries = {name: summarize_search(search) for name, search in searches.items()}
    selected = select_candidate(summaries)

    for name, summary in summaries.items():
        cv = summary["validation_croisee"]
        logger.info(
            "%-22s sens=%.4f spec=%.4f auc=%.4f %s",
            MODEL_NAMES[name],
            cv["sensibilite"]["moyenne"],
            cv["specificite"]["moyenne"],
            cv["auc"]["moyenne"],
            summary["parametres"],
        )
    logger.info("Modèle retenu : %s", MODEL_NAMES[selected])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(searches[selected].best_estimator_, MODEL_PATH, compress=3)

    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(
        json.dumps(
            {
                "regle": (
                    "Sensibilité moyenne maximale en validation croisée 5 plis sur le train, "
                    f"sous contrainte de spécificité >= {MIN_SPECIFICITY}, AUC en départage."
                ),
                "modele_retenu": selected,
                "candidats": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    METADATA_PATH.write_text(
        json.dumps(
            {
                "version": MODEL_VERSION,
                "date_entrainement": datetime.now(UTC).date().isoformat(),
                "modele": MODEL_NAMES[selected],
                "parametres": summaries[selected]["parametres"],
                "colonnes": FEATURE_ORDER,
                "seuil_decision": DECISION_THRESHOLD,
                "n_train": int(len(X_train)),
                "scikit_learn": sklearn.__version__,
                "sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                "validation_croisee_train": summaries[selected]["validation_croisee"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
