"""Validation stricte des données reçues par l'API.

Les messages citent le champ et la règle, jamais la valeur envoyée : une réponse
ou un journal ne doit pas pouvoir recopier une donnée de santé.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NumericField:
    """Variable mesurée, bornée à une plage médicalement plausible."""

    minimum: float
    maximum: float
    integer: bool = False


@dataclass(frozen=True)
class CategoricalField:
    """Variable codée, limitée aux modalités connues du modèle."""

    allowed: tuple[int, ...]


SCHEMA = {
    # Le modèle n'a vu que des adultes : je refuse de l'extrapoler aux enfants.
    "age": NumericField(18, 120, integer=True),
    "sex": CategoricalField((0, 1)),
    "cp": CategoricalField((0, 1, 2, 3)),
    "trestbps": NumericField(50, 250),
    "chol": NumericField(50, 700),
    "fbs": CategoricalField((0, 1)),
    "restecg": CategoricalField((0, 1, 2)),
    "thalach": NumericField(50, 250),
    "exang": CategoricalField((0, 1)),
    "oldpeak": NumericField(-5, 10),
    "slope": CategoricalField((0, 1, 2)),
    "ca": CategoricalField((0, 1, 2, 3)),
    "thal": CategoricalField((0, 1, 2)),
}


def _is_number(value):
    # bool hérite de int en Python : sans ce test, true passerait pour 1.
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _validate_field(name, value):
    """Renvoie (valeur normalisée, erreur) pour un champ."""
    rule = SCHEMA[name]
    if not _is_number(value):
        return None, f"{name} : un nombre est attendu."
    if isinstance(rule, CategoricalField):
        # 3.0 reste accepté (certains clients JSON sérialisent ainsi), pas 1.5.
        if value not in rule.allowed:
            allowed = ", ".join(str(item) for item in rule.allowed)
            return None, f"{name} : valeurs autorisées {allowed}."
        return int(value), None
    if rule.integer and value != int(value):
        return None, f"{name} : un nombre entier est attendu."
    if not rule.minimum <= value <= rule.maximum:
        return None, f"{name} : doit être compris entre {rule.minimum} et {rule.maximum}."
    return (int(value) if rule.integer else value), None


def validate_payload(payload):
    """Contrôle le corps de la requête et renvoie (données propres, erreurs)."""
    if not isinstance(payload, dict):
        return None, ["Le corps de la requête doit être un objet JSON."]

    errors = [f"{name} : champ non autorisé." for name in payload if name not in SCHEMA]
    errors += [f"{name} : champ obligatoire." for name in SCHEMA if name not in payload]

    clean = {}
    for name in SCHEMA:
        if name in payload:
            clean[name], error = _validate_field(name, payload[name])
            if error:
                errors.append(error)
    return (None, errors) if errors else (clean, [])
