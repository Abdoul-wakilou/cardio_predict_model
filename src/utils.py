"""
utils.py - Fonctions utilitaires pour Flask
- Génération des recommandations personnalisées
- Interprétation des probabilités
"""

def generate_recommendations(patient_data, prediction, probability):
    """
    Génère des recommandations personnalisées basées sur les données du patient.
    
    Args:
        patient_data (dict): Données cliniques du patient
        prediction (int): Prédiction (0 = sain, 1 = malade)
        probability (float): Probabilité d'être malade
    
    Returns:
        list: Liste de recommandations personnalisées
    """
    recommendations = []
    
    # Recommandations basées sur les valeurs anormales
    if patient_data.get('age', 0) > 65:
        recommendations.append("Âge avancé (> 65 ans) : surveillance cardiovasculaire annuelle recommandée.")
    
    if patient_data.get('trestbps', 0) > 140:
        recommendations.append(f"Tension artérielle élevée ({patient_data['trestbps']} mmHg). Limitez le sel, pratiquez une activité physique régulière.")
    
    if patient_data.get('trestbps', 0) > 160:
        recommendations.append("Tension artérielle très élevée : consultation médicale urgente recommandée.")
    
    if patient_data.get('chol', 0) > 240:
        recommendations.append(f"Cholestérol élevé ({patient_data['chol']} mg/dL). Réduisez les graisses saturées, privilégiez les fruits et légumes.")
    
    if patient_data.get('chol', 0) > 300:
        recommendations.append("Cholestérol très élevé : traitement médicamenteux probablement nécessaire.")
    
    if patient_data.get('fbs', 0) == 1:
        recommendations.append("Glycémie à jeun élevée. Contrôlez votre alimentation (sucres rapides), activité physique régulière.")
    
    if patient_data.get('thalach', 0) < 120:
        recommendations.append(f"Fréquence cardiaque maximale basse ({patient_data['thalach']} bpm). Consultation cardiologique pour test d'effort.")
    
    if patient_data.get('exang', 0) == 1:
        recommendations.append("Angine à l'effort détectée. Évaluation cardiologique complète recommandée.")
    
    if patient_data.get('oldpeak', 0) > 1.5:
        recommendations.append(f"Dépression ST significative ({patient_data['oldpeak']} mm). Surveillance ECG nécessaire.")
    
    if patient_data.get('ca', 0) > 1:
        recommendations.append(f"Présence de {patient_data['ca']} vaisseau(x) obstrué(s). Suivi cardiologique rapproché.")
    
    if patient_data.get('thal', 0) in [6, 7]:
        recommendations.append("Anomalie à la scintigraphie myocardique. Consultation spécialisée requise.")
    
    # Recommandations basées sur le niveau de risque global
    if prediction == 0 and probability < 0.8:
        recommendations.append("Risque faible détecté. Maintenez une bonne hygiène de vie (alimentation équilibrée, activité physique).")
    
    if prediction == 1 and probability < 0.6:
        recommendations.append("Risque modéré. Consultation médicale recommandée pour bilan cardiovasculaire.")
    
    if prediction == 1 and probability >= 0.6:
        recommendations.append("⚠️ RISQUE ÉLEVÉ. Consultation cardiologique rapide impérative.")
    
    if prediction == 1 and probability >= 0.8:
        recommendations.append("⚠️⚠️ RISQUE TRÈS ÉLEVÉ. Consultation URGENTE recommandée.")
    
    # Recommandations générales si aucune spécifique
    if not recommendations:
        recommendations.append("Aucun facteur de risque majeur détecté. Continuez vos bonnes habitudes de vie.")
        recommendations.append("Bilan de santé annuel recommandé.")
    
    return recommendations


def get_risk_level(prediction, probability):
    """
    Détermine le niveau de risque (0-4) à partir de la prédiction et de la probabilité.
    
    Args:
        prediction (int): 0 = sain, 1 = malade
        probability (float): Probabilité d'être malade (entre 0 et 1)
    
    Returns:
        tuple: (niveau, libelle, description)
    """
    if prediction == 0:  # Prédit sain
        if probability >= 0.8:
            return (0, "Aucun risque", "✅ Santé cardiovasculaire excellente.")
        else:
            return (1, "Risque faible", "⚠️ Risque faible, surveillance recommandée.")
    else:  # Prédit malade
        if probability < 0.6:
            return (2, "Risque modéré", "🔶 Risque modéré, consultation médicale conseillée.")
        elif probability < 0.8:
            return (3, "Risque élevé", "🚨 Risque élevé, consultation cardiologique recommandée.")
        else:
            return (4, "Risque très élevé", "💥 RISQUE TRÈS ÉLEVÉ - Consultation URGENTE.")