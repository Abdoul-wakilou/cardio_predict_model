"""API Flask de CardioPredict : validation, prédiction et réponses JSON.

Encodage attendu :
  cp    : 0=typique, 1=atypique, 2=non-angineuse, 3=asymptomatique
  slope : 0=ascendante, 1=plate, 2=descendante
  thal  : 0=normal, 1=défaut fixe, 2=défaut réversible
"""

import hashlib
import json
import logging
import os
import time
import traceback

import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, g, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from api.recommendations import generate_recommendations, get_risk_level
from api.validation import validate_payload
from src.config import DECISION_THRESHOLD, FEATURE_ORDER, METADATA_PATH, MODEL_PATH, ROOT_DIR

logger = logging.getLogger("cardiopredict.api")

API_VERSION = "4.0.0"
DISCLAIMER = (
    "Outil d'aide à la décision à visée de recherche : ce résultat n'est pas un diagnostic "
    "et ne remplace pas l'avis d'un professionnel de santé."
)
ERROR_MESSAGES = {
    400: "Requête invalide.",
    404: "Ressource introuvable.",
    405: "Méthode non autorisée.",
    413: "Requête trop volumineuse.",
    415: "Le type de contenu doit être application/json.",
    429: "Trop de requêtes. Réessayez dans un instant.",
}


def load_model():
    """Charge le modèle du dépôt après vérification de son empreinte.

    Un fichier joblib peut exécuter du code : le chemin est donc fixe, jamais
    fourni par une requête ou par l'environnement.
    """
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise RuntimeError("Empreinte du modèle incorrecte : chargement refusé.")
    return joblib.load(MODEL_PATH), metadata


def json_response(payload, status=200):
    """Construit une réponse JSON portant toujours l'avertissement médical."""
    return jsonify({**payload, "disclaimer": DISCLAIMER}), status


def _read_settings(overrides):
    settings = {
        "MAX_CONTENT_LENGTH": int(os.getenv("CARDIO_MAX_CONTENT_LENGTH", "4096")),
        "RATE_LIMIT": os.getenv("CARDIO_RATE_LIMIT", "30 per minute"),
        "RATE_LIMIT_STORAGE": os.getenv("CARDIO_RATE_LIMIT_STORAGE", "memory://"),
        "TRUSTED_PROXIES": int(os.getenv("CARDIO_TRUSTED_PROXIES", "0")),
        "CORS_ORIGINS": {
            origin.strip()
            for origin in os.getenv("CARDIO_CORS_ORIGINS", "").split(",")
            if origin.strip()
        },
    }
    settings.update(overrides or {})
    return settings


def _register_observability(app):
    @app.before_request
    def start_timer():
        g.started_at = time.perf_counter()

    @app.after_request
    def finalize_response(response):
        # Seules des informations techniques sont journalisées, jamais le corps.
        duration_ms = (time.perf_counter() - g.get("started_at", time.perf_counter())) * 1000
        logger.info(
            "%s %s %s %.1fms", request.method, request.path, response.status_code, duration_ms
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"

        # Une application native n'envoie pas d'en-tête Origin : le CORS ne sert
        # qu'aux essais navigateur, et seulement pour les origines déclarées.
        origin = request.headers.get("Origin")
        if origin and origin in app.config["CORS_ORIGINS"]:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Vary"] = "Origin"
        return response


def _register_error_handlers(app):
    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        return json_response(
            {"error": ERROR_MESSAGES.get(error.code, "Erreur de la requête.")}, error.code
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        # Je garde le type et la position, sans le message qui pourrait citer une valeur.
        frame = traceback.extract_tb(error.__traceback__)[-1]
        logger.error(
            "Erreur interne %s (%s:%s)", type(error).__name__, frame.filename, frame.lineno
        )
        return json_response({"error": "Erreur interne du serveur."}, 500)


def create_app(overrides=None):
    """Crée l'application Flask avec sa configuration de sécurité."""
    load_dotenv(ROOT_DIR / ".env")
    app = Flask(__name__)
    app.config.update(_read_settings(overrides))
    if app.config["TRUSTED_PROXIES"]:
        count = app.config["TRUSTED_PROXIES"]
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=count, x_proto=count)

    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        storage_uri=app.config["RATE_LIMIT_STORAGE"],
    )
    pipeline, metadata = load_model()
    app.extensions["cardio_model"] = pipeline

    _register_observability(app)
    _register_error_handlers(app)

    @app.post("/predict")
    @limiter.limit(lambda: app.config["RATE_LIMIT"])
    def predict():
        clean, errors = validate_payload(request.get_json())
        if errors:
            return json_response({"error": "Données invalides.", "details": errors}, 400)

        frame = pd.DataFrame([clean], columns=FEATURE_ORDER)
        probability = float(app.extensions["cardio_model"].predict_proba(frame)[0][1])
        prediction = int(probability >= DECISION_THRESHOLD)
        risk_level, risk_label, risk_description = get_risk_level(prediction, probability)

        return json_response(
            {
                "success": True,
                "prediction": prediction,
                "prediction_label": "Malade" if prediction == 1 else "Sain",
                "probability_disease": round(probability, 4),
                "probability_no_disease": round(1 - probability, 4),
                "risk_level": risk_level,
                "risk_label": risk_label,
                "risk_description": risk_description,
                "recommendations": generate_recommendations(clean, prediction, probability),
            }
        )

    @app.get("/health")
    def health():
        return json_response({"status": "healthy", "model_version": metadata["version"]})

    @app.get("/")
    def index():
        return json_response(
            {
                "service": "CardioPredict API",
                "version": API_VERSION,
                "dataset": "UCI Heart Disease, 4 centres (834 patients)",
                "encoding": {
                    "cp": "0=typique, 1=atypique, 2=non-angineuse, 3=asymptomatique",
                    "slope": "0=ascendante, 1=plate, 2=descendante",
                    "thal": "0=normal, 1=défaut fixe, 2=défaut réversible",
                    "target": "0=sain, 1=malade",
                },
            }
        )

    return app


app = create_app()
