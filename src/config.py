"""Constantes partagées par le pipeline scientifique et par l'API."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT_DIR / "data" / "raw"
DATASET_PATH = ROOT_DIR / "data" / "processed" / "heart_uci_combined.csv"
MODEL_PATH = ROOT_DIR / "models" / "cardio_pipeline.joblib"
METADATA_PATH = ROOT_DIR / "models" / "metadata.json"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SELECTION_PATH = REPORTS_DIR / "selection_modele.json"
TEST_RESULTS_PATH = REPORTS_DIR / "resultats_test.json"
ROBUSTNESS_PATH = REPORTS_DIR / "validation_robustesse.json"

RANDOM_STATE = 42
TEST_SIZE = 0.30

# Seuil unique : les métriques et l'API doivent décider exactement de la même façon.
DECISION_THRESHOLD = 0.5

# Plancher fixé avant la sélection : sans lui, classer tout le monde « malade »
# obtiendrait une sensibilité parfaite et gagnerait la comparaison.
MIN_SPECIFICITY = 0.70

TARGET = "target"
CENTRE = "centre"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]

# Ordre de référence des 13 variables, identique dans le jeu de données et l'API.
FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
