"""Modèles candidats, métriques et règle de sélection."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    make_scorer,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import DECISION_THRESHOLD, MIN_SPECIFICITY, RANDOM_STATE
from src.preprocessing import build_preprocessor

MODEL_NAMES = {
    "regression_logistique": "Régression logistique",
    "svm": "SVM",
    "random_forest": "Random Forest",
}

CANDIDATES = {
    "regression_logistique": (
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1, 10, 100, 1000], "model__class_weight": [None, "balanced"]},
    ),
    "svm": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.01, 0.1],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__class_weight": [None, "balanced"],
        },
    ),
}


def predict_from_probability(probabilities):
    """Applique le seuil de décision commun aux métriques et à l'API."""
    return (np.asarray(probabilities) >= DECISION_THRESHOLD).astype(int)


def _sensitivity(y_true, probabilities):
    return recall_score(y_true, predict_from_probability(probabilities), pos_label=1)


def _specificity(y_true, probabilities):
    return recall_score(y_true, predict_from_probability(probabilities), pos_label=0)


def _accuracy(y_true, probabilities):
    return accuracy_score(y_true, predict_from_probability(probabilities))


# Le SVM peut donner une classe différente via predict et via predict_proba :
# je base tous les scores sur la probabilité, comme le fait l'API.
SCORING = {
    "sensibilite": make_scorer(_sensitivity, response_method="predict_proba"),
    "specificite": make_scorer(_specificity, response_method="predict_proba"),
    "accuracy": make_scorer(_accuracy, response_method="predict_proba"),
    "auc": "roc_auc",
}


def build_pipeline(estimator):
    """Associe préprocesseur et modèle pour que chaque pli réajuste l'imputation."""
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])


def rank_configuration(sensitivity, specificity, auc):
    """Renvoie l'indice retenu : sensibilité maximale sous plancher de spécificité.

    L'AUC départage les égalités. Si aucune configuration n'atteint le plancher,
    je retiens la meilleure moyenne sensibilité/spécificité.
    """
    sensitivity, specificity, auc = map(np.asarray, (sensitivity, specificity, auc))
    eligible = np.flatnonzero(specificity >= MIN_SPECIFICITY)
    if eligible.size == 0:
        balanced = (sensitivity + specificity) / 2
        return int(np.lexsort((auc, balanced))[-1])
    best = np.lexsort((auc[eligible], sensitivity[eligible]))[-1]
    return int(eligible[best])


def select_index(cv_results):
    """Adapte la règle de sélection au format attendu par GridSearchCV."""
    return rank_configuration(
        cv_results["mean_test_sensibilite"],
        cv_results["mean_test_specificite"],
        cv_results["mean_test_auc"],
    )


def tune(name, X, y, n_splits=5):
    """Optimise un candidat par validation croisée stratifiée sur les données fournies."""
    estimator, grid = CANDIDATES[name]
    search = GridSearchCV(
        build_pipeline(estimator),
        grid,
        scoring=SCORING,
        refit=select_index,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    return search.fit(X, y)


def compute_metrics(y_true, probabilities):
    """Calcule les métriques de classification à partir des probabilités."""
    y_true = np.asarray(y_true)
    predictions = predict_from_probability(probabilities)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(y_true.size),
        "malades": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "auc": float(roc_auc_score(y_true, probabilities)),
        "sensibilite": float(tp / (tp + fn)),
        "specificite": float(tn / (tn + fp)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "matrice_confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
