"""Tests garantissant la non-régression méthodologique du pipeline."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.build_dataset import build_dataset
from src.config import DATASET_PATH, FEATURE_ORDER, METADATA_PATH, MODEL_PATH, TARGET
from src.modeling import build_pipeline, rank_configuration
from src.preprocessing import load_dataset, split_dataset


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_published_dataset_matches_raw_files(dataset):
    rebuilt = build_dataset()
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), dataset, check_dtype=False)


def test_cohort_is_the_thesis_cohort(dataset):
    assert len(dataset) == 834
    assert int(dataset[TARGET].sum()) == 463


def test_missing_values_are_not_imputed_before_split(dataset):
    assert int(dataset["ca"].isna().sum()) == 525
    assert int(dataset["thal"].isna().sum()) == 405


def test_impossible_zeros_are_missing(dataset):
    assert not (dataset["chol"] == 0).any()
    assert not (dataset["trestbps"] == 0).any()


def test_split_is_stratified_and_sized_like_the_thesis(dataset):
    X_train, X_test, y_train, y_test = split_dataset(dataset)
    assert (len(X_train), len(X_test)) == (583, 251)
    assert (int(y_test.sum()), int((y_test == 0).sum())) == (139, 112)
    assert not set(X_train.index) & set(X_test.index)


def test_imputer_learns_from_training_data_only(dataset):
    from sklearn.linear_model import LogisticRegression

    X_train, _, y_train, _ = split_dataset(dataset)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    imputer = pipeline.named_steps["preprocessor"].named_transformers_["num"]["imputer"]
    expected = X_train[["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]].median()
    np.testing.assert_allclose(imputer.statistics_, expected.values)


def test_selection_prefers_sensitivity_above_specificity_floor():
    # La deuxième configuration a la meilleure sensibilité parmi celles admissibles.
    chosen = rank_configuration([0.99, 0.90, 0.85], [0.40, 0.75, 0.80], [0.9, 0.8, 0.95])
    assert chosen == 1


def test_selection_uses_auc_to_break_ties():
    assert rank_configuration([0.9, 0.9], [0.8, 0.8], [0.7, 0.85]) == 1


def test_selection_falls_back_when_no_configuration_is_eligible():
    assert rank_configuration([0.95, 0.70], [0.10, 0.60], [0.9, 0.9]) == 1


def test_model_artifact_matches_its_metadata():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == metadata["sha256"]
    assert metadata["colonnes"] == FEATURE_ORDER


def test_model_predicts_a_probability(dataset):
    model = joblib.load(MODEL_PATH)
    assert list(model.feature_names_in_) == FEATURE_ORDER
    probabilities = model.predict_proba(dataset[FEATURE_ORDER].head(5))[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_dataset_file_exists():
    assert DATASET_PATH.exists()
