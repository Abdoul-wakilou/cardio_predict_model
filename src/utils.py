"""
utils.py - Fonctions utilitaires pour Flask
"""


def generate_recommendations(patient_data, prediction, probability):
    """
    Génère des recommandations personnalisées (version concise).
    """
    recommendations = []

    # Âge
    age = patient_data.get('age', 0)
    if age > 65:
        recommendations.append(f"Âge avancé ({age} ans) : bilan cardiovasculaire annuel recommandé.")
    elif age > 50:
        recommendations.append(f"Âge ({age} ans) : surveillance cardiovasculaire tous les 2 ans.")

    # Tension artérielle
    trestbps = patient_data.get('trestbps', 0)
    if trestbps > 140:
        recommendations.append(f"Tension élevée ({trestbps} mmHg) : réduisez le sel, activité physique régulière.")
    elif trestbps > 130:
        recommendations.append(f"Pré-hypertension ({trestbps} mmHg) : surveillez votre tension.")

    # Cholestérol
    chol = patient_data.get('chol', 0)
    if chol > 240:
        recommendations.append(f"Cholestérol élevé ({chol} mg/dL) : régime méditerranéen, évitez les graisses saturées.")
    elif chol > 200:
        recommendations.append(f"Cholestérol limite haut ({chol} mg/dL) : surveillez votre alimentation.")

    # Glycémie
    if patient_data.get('fbs', 0) == 1:
        recommendations.append("Glycémie élevée : réduisez les sucres rapides, activité physique après les repas.")

    # Fréquence cardiaque maximale
    thalach = patient_data.get('thalach', 0)
    if thalach < 120:
        recommendations.append(f"Fréquence cardiaque basse ({thalach} bpm) : consultation cardiologique recommandée.")

    # Angine à l'effort
    if patient_data.get('exang', 0) == 1:
        recommendations.append("Angine à l'effort détectée : consultation cardiologique rapide.")

    # Dépression ST
    oldpeak = patient_data.get('oldpeak', 0)
    if oldpeak > 1.5:
        recommendations.append(f"Dépression ST significative ({oldpeak} mm) : surveillance ECG nécessaire.")

    # Vaisseaux obstrués
    ca = patient_data.get('ca', 0)
    if ca >= 3:
        recommendations.append(f"{ca} vaisseaux obstrués : suivi cardiologique rapproché impératif.")
    elif ca >= 2:
        recommendations.append(f"{ca} vaisseaux obstrués : suivi cardiologique tous les 6 mois.")
    elif ca == 1:
        recommendations.append("Sténose coronaire unique : suivi cardiologique annuel.")

    # Thalassémie
    thal = patient_data.get('thal', 1)
    if thal == 2:
        recommendations.append("Défaut fixe à la scintigraphie : séquelle d'infarctus possible, bilan cardiologique.")
    elif thal == 3:
        recommendations.append("Défaut réversible à la scintigraphie : ischémie, consultation urgente.")

    # Recommandation finale selon le risque
    if prediction == 0:
        p_sain = 1 - probability
        if p_sain >= 0.85:
            recommendations.append("Santé cardiovasculaire excellente. Continuez vos bonnes habitudes.")
        else:
            recommendations.append("Risque faible. Bilan de santé annuel recommandé.")
    else:
        if probability < 0.6:
            recommendations.append("Risque modéré. Consultation médicale recommandée.")
        elif probability < 0.8:
            recommendations.append("Risque élevé. Consultation cardiologique rapide impérative.")
        else:
            recommendations.append("Risque très élevé. Consultation URGENTE recommandée.")

    # Supprimer les doublons
    unique = []
    for r in recommendations:
        if r not in unique:
            unique.append(r)

    return unique

def get_risk_level(prediction, probability):
    """
    Détermine le niveau de risque à partir de la prédiction et de P(malade).

    Args:
        prediction  (int)  : 0 = sain, 1 = malade
        probability (float): P(malade) — sortie de predict_proba()[:,1]

    Returns:
        tuple (niveau: int, libellé: str, description: str)
    """
    if prediction == 0:
        # P(malade) < 0.5 par définition — on utilise P(sain) pour graduer
        p_sain = 1 - probability
        if p_sain >= 0.8:
            return (0, "Aucun risque",  "Santé cardiovasculaire excellente.")
        else:
            return (1, "Risque faible", "Risque faible, surveillance recommandée.")
    else:
        # P(malade) >= 0.5 par définition
        if probability < 0.6:
            return (2, "Risque modéré",      "Risque modéré, consultation médicale conseillée.")
        elif probability < 0.8:
            return (3, "Risque élevé",       "Risque élevé, consultation cardiologique recommandée.")
        else:
            return (4, "Risque très élevé",  "RISQUE TRÈS ÉLEVÉ - Consultation URGENTE.")