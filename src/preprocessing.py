"""Chargement, découpage et préprocesseur du jeu combiné."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def load_dataset(path=DATASET_PATH):
    """Charge le jeu combiné produit par build_dataset."""
    return pd.read_csv(path)


def split_dataset(df):
    """Découpe en train/test stratifié (70/30) avec une graine fixe."""
    return train_test_split(
        df[FEATURE_ORDER],
        df[TARGET],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )


def build_preprocessor():
    """Assemble imputation, normalisation et encodage, sans jamais lire la cible."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
        ],
        remainder="drop",
    )
