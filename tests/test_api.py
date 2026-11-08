"""Tests de l'API : contrat de réponse, validation stricte et sécurité."""

import logging

import pytest

from api.app import DISCLAIMER, create_app
from api.validation import SCHEMA
from src.config import FEATURE_ORDER

PATIENT = {
    "age": 55,
    "sex": 1,
    "cp": 3,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 0,
    "thalach": 130,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 1,
    "thal": 2,
}


@pytest.fixture
def client():
    return create_app({"TESTING": True, "RATE_LIMIT": "1000 per minute"}).test_client()


def test_schema_follows_model_columns():
    assert list(SCHEMA) == FEATURE_ORDER


def test_valid_prediction_keeps_mobile_contract(client):
    response = client.post("/predict", json=PATIENT)
    body = response.get_json()
    assert response.status_code == 200
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability_disease"] <= 1
    assert body["risk_level"] in range(5)
    assert isinstance(body["recommendations"], list)
    assert body["disclaimer"] == DISCLAIMER
    assert "error" not in body


@pytest.mark.parametrize(
    "change",
    [
        {"age": "abc"},
        {"age": "55"},
        {"age": 10},
        {"age": 55.5},
        {"chol": 0},
        {"trestbps": 0},
        {"cp": None},
        {"cp": 1.5},
        {"cp": 4},
        {"thal": 0.5},
        {"sex": True},
        {"nom_patient": "X"},
    ],
)
def test_invalid_field_is_rejected(client, change):
    response = client.post("/predict", json={**PATIENT, **change})
    body = response.get_json()
    assert response.status_code == 400
    assert body["details"]
    assert body["disclaimer"] == DISCLAIMER


def test_integer_valued_float_is_accepted_for_categories(client):
    assert client.post("/predict", json={**PATIENT, "cp": 3.0}).status_code == 200


def test_missing_field_is_rejected(client):
    incomplete = {key: value for key, value in PATIENT.items() if key != "ca"}
    assert client.post("/predict", json=incomplete).status_code == 400


@pytest.mark.parametrize("token", ["NaN", "Infinity"])
def test_non_finite_number_is_rejected(client, token):
    body = f'{{"age": {token}}}'
    assert client.post("/predict", data=body, content_type="application/json").status_code == 400


def test_malformed_json_returns_400(client):
    response = client.post("/predict", data="{age:", content_type="application/json")
    assert response.status_code == 400


def test_wrong_content_type_returns_415(client):
    assert client.post("/predict", data="x=1", content_type="text/plain").status_code == 415


def test_non_object_body_returns_400(client):
    assert client.post("/predict", json=[1, 2]).status_code == 400


def test_oversized_request_returns_413():
    client = create_app({"TESTING": True, "MAX_CONTENT_LENGTH": 10}).test_client()
    assert client.post("/predict", json=PATIENT).status_code == 413


def test_rate_limit_returns_429():
    client = create_app({"TESTING": True, "RATE_LIMIT": "1 per minute"}).test_client()
    assert client.post("/predict", json=PATIENT).status_code == 200
    assert client.post("/predict", json=PATIENT).status_code == 429


def test_internal_error_stays_generic(monkeypatch):
    app = create_app({"TESTING": True, "RATE_LIMIT": "1000 per minute"})

    class BrokenModel:
        def predict_proba(self, _):
            raise ValueError("detail interne sensible")

    monkeypatch.setitem(app.extensions, "cardio_model", BrokenModel())
    response = app.test_client().post("/predict", json=PATIENT)
    assert response.status_code == 500
    assert "sensible" not in response.get_data(as_text=True)


def test_patient_values_never_reach_logs(client, caplog):
    with caplog.at_level(logging.DEBUG):
        client.post("/predict", json={**PATIENT, "chol": 317})
        client.post("/predict", json={**PATIENT, "chol": 9173})
    assert "317" not in caplog.text
    assert "9173" not in caplog.text
    assert "/predict" in caplog.text


def test_cors_is_closed_by_default(client):
    response = client.post("/predict", json=PATIENT, headers={"Origin": "https://exemple.org"})
    assert "Access-Control-Allow-Origin" not in response.headers


def test_cors_allows_only_declared_origin():
    client = create_app({"TESTING": True, "CORS_ORIGINS": {"http://localhost:8080"}}).test_client()
    allowed = client.get("/health", headers={"Origin": "http://localhost:8080"})
    refused = client.get("/health", headers={"Origin": "https://exemple.org"})
    assert allowed.headers["Access-Control-Allow-Origin"] == "http://localhost:8080"
    assert "Access-Control-Allow-Origin" not in refused.headers


@pytest.mark.parametrize("route", ["/", "/health"])
def test_every_route_returns_disclaimer(client, route):
    assert client.get(route).get_json()["disclaimer"] == DISCLAIMER
