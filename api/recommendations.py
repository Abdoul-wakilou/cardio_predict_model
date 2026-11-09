"""Messages de prévention associés à une prédiction.

Le modèle ne pose pas de diagnostic : les textes parlent de signaux à vérifier
et orientent vers un professionnel, sans nommer de traitement.

Encodage : cp 0=typique, 3=asymptomatique ; thal 1=défaut fixe, 2=défaut réversible.
"""

EMERGENCY_ADVICE = "En cas de douleur thoracique ou d'essoufflement brutal, appelez les urgences."


def _age_advice(age):
    if age > 65:
        return (
            f"Âge ({age} ans) : bilan annuel conseillé (ECG, lipides, glycémie). "
            "Marche 30 min/jour, fruits et légumes, moins de sel et de graisses saturées."
        )
    if age > 50:
        return (
            f"Âge ({age} ans) : bilan cardiovasculaire tous les 2 ans. "
            "Activité physique régulière, alimentation équilibrée, arrêt du tabac si fumeur."
        )
    return None


def _blood_pressure_advice(pressure):
    if pressure >= 160:
        return (
            f"Tension très élevée ({pressure} mmHg) : consultez rapidement un médecin. "
            "Réduisez le sel (moins de 5 g/jour) et mesurez votre tension matin et soir."
        )
    if pressure >= 140:
        return (
            f"Tension élevée ({pressure} mmHg) : consultez un médecin. "
            "Moins de sel, fruits et légumes, marche 30 min/jour, perte de poids si surpoids."
        )
    if pressure >= 130:
        return (
            f"Tension à surveiller ({pressure} mmHg) : contrôle tous les 6 mois. "
            "Moins de sel, activité physique régulière, contrôle du poids."
        )
    return None


def _cholesterol_advice(cholesterol):
    if cholesterol >= 280:
        return (
            f"Cholestérol très élevé ({cholesterol} mg/dL) : consultez pour un bilan lipidique. "
            "Limitez beurre, charcuterie et fritures ; privilégiez poissons gras et fibres."
        )
    if cholesterol >= 240:
        return (
            f"Cholestérol élevé ({cholesterol} mg/dL) : moins de graisses saturées, poisson gras "
            "2 fois par semaine, marche 30 min/jour. Bilan lipidique dans 3 mois."
        )
    if cholesterol >= 200:
        return (
            f"Cholestérol limite ({cholesterol} mg/dL) : surveillez l'alimentation, "
            "privilégiez fibres et poissons. Bilan lipidique annuel."
        )
    return None


def _heart_rate_advice(heart_rate):
    if heart_rate < 100:
        return (
            f"Fréquence cardiaque maximale très basse ({heart_rate} bpm) : "
            "avis cardiologique rapide conseillé (ECG, échocardiographie, test d'effort)."
        )
    if heart_rate < 120:
        return (
            f"Fréquence cardiaque maximale basse ({heart_rate} bpm) : avis cardiologique "
            "conseillé. Reprenez l'effort progressivement, en commençant par 15 min de marche."
        )
    return None


def _st_depression_advice(oldpeak):
    if oldpeak >= 3.0:
        return (
            f"Sous-décalage ST important ({oldpeak} mm) : signe possible d'ischémie, à confirmer. "
            f"Consultation cardiologique rapide, évitez tout effort intense. {EMERGENCY_ADVICE}"
        )
    if oldpeak >= 1.5:
        return (
            f"Sous-décalage ST notable ({oldpeak} mm) : ECG d'effort à discuter avec un médecin. "
            "Limitez les efforts intenses en attendant."
        )
    return None


def _vessels_advice(vessels):
    if vessels >= 2:
        return (
            f"{vessels} vaisseaux principaux colorés en fluoroscopie : ce résultat est à "
            "interpréter avec un cardiologue. Contrôlez tension, cholestérol et glycémie, "
            "arrêt total du tabac."
        )
    if vessels == 1:
        return (
            "1 vaisseau principal coloré en fluoroscopie : résultat à interpréter avec "
            "un cardiologue. Alimentation équilibrée et activité physique adaptée."
        )
    return None


def _thallium_advice(thal):
    if thal == 1:
        return (
            "Défaut fixe à la scintigraphie : séquelle possible, à interpréter par un "
            "cardiologue. Un bilan cardiologique complet est conseillé."
        )
    if thal == 2:
        return (
            "Défaut réversible à la scintigraphie : ischémie possible, à confirmer. "
            "Consultation cardiologique rapide conseillée, évitez tout effort intense."
        )
    return None


def _overall_advice(prediction, probability):
    if prediction == 0:
        if probability <= 0.15:
            return (
                "Le modèle ne détecte pas de signal de risque élevé. Gardez vos bonnes habitudes "
                "et faites un bilan de santé annuel."
            )
        return (
            "Risque faible selon le modèle : bilan de santé annuel conseillé. "
            "Corrigez les facteurs de risque identifiés, arrêt du tabac si fumeur."
        )
    if probability < 0.60:
        return (
            "Risque modéré selon le modèle : consultation médicale dans les 2 à 4 semaines "
            "(tension, lipides, glycémie, ECG)."
        )
    if probability < 0.80:
        return (
            f"Risque élevé selon le modèle : consultation cardiologique sous 8 jours. "
            f"{EMERGENCY_ADVICE}"
        )
    return (
        f"Risque très élevé selon le modèle : consultation médicale urgente (sous 48 h). "
        f"{EMERGENCY_ADVICE}"
    )


def generate_recommendations(patient, prediction, probability):
    """Assemble les conseils correspondant aux facteurs de risque du patient."""
    advice = [
        _age_advice(patient["age"]),
        _blood_pressure_advice(patient["trestbps"]),
        _cholesterol_advice(patient["chol"]),
        _heart_rate_advice(patient["thalach"]),
        _st_depression_advice(patient["oldpeak"]),
        _vessels_advice(patient["ca"]),
        _thallium_advice(patient["thal"]),
    ]
    if patient["fbs"] == 1:
        advice.append(
            "Glycémie à jeun élevée (> 120 mg/dL) : consultez pour un dosage HbA1c. "
            "Réduisez les sucres rapides et marchez 15 min après les repas."
        )
    if patient["exang"] == 1:
        advice.append(
            "Angine à l'effort signalée : consultation cardiologique sous 8 jours. "
            f"Arrêtez tout effort déclenchant la douleur. {EMERGENCY_ADVICE}"
        )
    if patient["cp"] == 0:
        advice.append(
            "Douleur angineuse typique signalée : bilan coronarien à discuter avec un cardiologue."
        )
    advice.append(_overall_advice(prediction, probability))
    return [item for item in advice if item]


def get_risk_level(prediction, probability):
    """Traduit la probabilité en niveau de risque (0 à 4), libellé et description."""
    if prediction == 0:
        if probability < 0.20:
            return (
                0,
                "Risque très faible",
                "Probabilité très faible selon le modèle. Bilan annuel conseillé.",
            )
        return 1, "Risque faible", "Probabilité faible selon le modèle. Suivi médical régulier."
    if probability < 0.60:
        return 2, "Risque modéré", "Risque modéré selon le modèle. Consultation dans le mois."
    if probability < 0.80:
        return (
            3,
            "Risque élevé",
            "Risque élevé selon le modèle. Consultation cardiologique sous 8 jours.",
        )
    return (
        4,
        "Risque très élevé",
        "Risque très élevé selon le modèle. Consultation médicale urgente sous 48 h.",
    )
