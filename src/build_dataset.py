"""Construit le jeu UCI Heart Disease combiné à partir des quatre centres.

Les valeurs manquantes restent vides dans le fichier produit : je les impute
uniquement dans le Pipeline d'entraînement, pour que la cible et le jeu de test
n'influencent jamais l'imputation.

Encodages après recodage :
  cp    : 0=typique, 1=atypique, 2=non-angineuse, 3=asymptomatique
  slope : 0=ascendante, 1=plate, 2=descendante
  thal  : 0=normal, 1=défaut fixe, 2=défaut réversible
  target: 0=sain, 1=malade
"""

import logging

import numpy as np
import pandas as pd

from src.config import CENTRE, DATASET_PATH, FEATURE_ORDER, RAW_DIR, TARGET

logger = logging.getLogger(__name__)

RAW_COLUMNS = [*FEATURE_ORDER, "num"]
SOURCES = {
    "processed.cleveland.data": "Cleveland",
    "processed.hungarian.data": "Hungarian",
    "processed.switzerland.data": "Switzerland",
    "processed.va.data": "Long Beach VA",
}
MAX_MISSING_FEATURES = 3


def load_raw_centres(raw_dir=RAW_DIR):
    """Charge et concatène les fichiers « processed » des quatre centres."""
    frames = []
    for file_name, centre in SOURCES.items():
        frame = pd.read_csv(
            raw_dir / file_name,
            header=None,
            names=RAW_COLUMNS,
            na_values=["?", "-9", "-9.0"],
        )
        frame[CENTRE] = centre
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def recode(df):
    """Binarise le diagnostic et ramène les catégories UCI à des codes 0-based."""
    df = df.copy()
    df[TARGET] = (df["num"] >= 1).astype(int)
    df["cp"] = df["cp"].map({1: 0, 2: 1, 3: 2, 4: 3})
    df["slope"] = df["slope"].map({1: 0, 2: 1, 3: 2})
    df["thal"] = df["thal"].map({3: 0, 6: 1, 7: 2})
    return df.drop(columns="num")


def filter_incomplete_patients(df):
    """Écarte les patients auxquels il manque au moins 4 variables sur 13."""
    return df[df[FEATURE_ORDER].isna().sum(axis=1) <= MAX_MISSING_FEATURES].reset_index(drop=True)


def mark_impossible_values_as_missing(df):
    """Traite les zéros physiologiquement impossibles comme des valeurs manquantes.

    Un cholestérol ou une tension à 0 est un code d'absence de mesure, concentré
    dans certains centres. Le garder revient à laisser le modèle reconnaître le
    centre d'origine au lieu de l'état du patient.
    """
    df = df.copy()
    for column in ("chol", "trestbps"):
        df.loc[df[column] == 0, column] = np.nan
    return df


def build_dataset(raw_dir=RAW_DIR):
    """Enchaîne les étapes de construction et renvoie le jeu final, non imputé."""
    df = recode(load_raw_centres(raw_dir))
    # Le filtre passe avant le traitement des zéros : je garde ainsi la cohorte
    # de 834 patients du mémoire et la comparaison porte sur les mêmes personnes.
    df = filter_incomplete_patients(df)
    df = mark_impossible_values_as_missing(df)
    return df[[*FEATURE_ORDER, TARGET, CENTRE]]


def main():
    """Écrit le jeu combiné et affiche un résumé de contrôle."""
    df = build_dataset()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)

    logger.info("Jeu de données : %d patients, %d malades", len(df), int(df[TARGET].sum()))
    logger.info("Patients par centre :\n%s", df[CENTRE].value_counts().to_string())
    logger.info("Valeurs manquantes :\n%s", df[FEATURE_ORDER].isna().sum().to_string())
    logger.info("Fichier écrit : %s", DATASET_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
