"""
utils.py - Fonctions utilitaires pour Flask
- Génération des recommandations personnalisées
- Interprétation des probabilités
"""

def generate_recommendations(patient_data, prediction, probability):
    """
    Génère des recommandations personnalisées basées sur les données du patient.
    """
    recommendations = []
    
    # ================================================================
    # RECOMMANDATIONS BASÉES SUR LES VALEURS ANORMALES (facteurs de risque)
    # ================================================================
    
    # Âge
    if patient_data.get('age', 0) > 65:
        recommendations.append("Âge avancé (> 65 ans) : surveillance cardiovasculaire annuelle recommandée.")
    
    # Tension artérielle
    trestbps = patient_data.get('trestbps', 0)
    if trestbps > 140:
        recommendations.append(f"Tension artérielle élevée ({trestbps} mmHg). Limitez le sel, pratiquez une activité physique régulière.")
    elif trestbps < 90:
        recommendations.append(f"Tension artérielle basse ({trestbps} mmHg). Surveillez les étourdissements.")
    else:
        recommendations.append(f"Tension artérielle normale ({trestbps} mmHg). Continuez à surveiller régulièrement.")
    
    # Cholestérol
    chol = patient_data.get('chol', 0)
    if chol > 240:
        recommendations.append(f"Cholestérol élevé ({chol} mg/dL). Réduisez les graisses saturées, privilégiez les fruits et légumes.")
    elif chol > 200:
        recommendations.append(f"Cholestérol limite haut ({chol} mg/dL). Surveillez votre alimentation.")
    else:
        recommendations.append(f"Cholestérol normal ({chol} mg/dL). Maintenez une alimentation équilibrée.")
    
    # Glycémie à jeun
    if patient_data.get('fbs', 0) == 1:
        recommendations.append("Glycémie à jeun élevée. Contrôlez votre alimentation (sucres rapides), activité physique régulière.")
    
    # Fréquence cardiaque maximale
    thalach = patient_data.get('thalach', 0)
    if thalach < 120:
        recommendations.append(f"Fréquence cardiaque maximale basse ({thalach} bpm). Consultation cardiologique pour test d'effort.")
    elif thalach > 180:
        recommendations.append(f"Fréquence cardiaque très élevée ({thalach} bpm). Évaluation cardiologique recommandée.")
    
    # Angine à l'effort
    if patient_data.get('exang', 0) == 1:
        recommendations.append("Angine à l'effort détectée. Évaluation cardiologique complète recommandée.")
    
    # Dépression ST
    oldpeak = patient_data.get('oldpeak', 0)
    if oldpeak > 1.5:
        recommendations.append(f"Dépression ST significative ({oldpeak} mm). Surveillance ECG nécessaire.")
    elif oldpeak > 0.5:
        recommendations.append(f"Légère dépression ST ({oldpeak} mm). Surveillance recommandée.")
    
    # Vaisseaux obstrués
    ca = patient_data.get('ca', 0)
    if ca > 1:
        recommendations.append(f"Présence de {ca} vaisseau(x) obstrué(s). Suivi cardiologique rapproché.")
    
    # Test thalassémie
    thal = patient_data.get('thal', 0)
    if thal in [6, 7]:
        recommendations.append("Anomalie à la scintigraphie myocardique. Consultation spécialisée requise.")
    
    # ================================================================
    # RECOMMANDATIONS BASÉES SUR LE NIVEAU DE RISQUE GLOBAL
    # ================================================================
    
    if prediction == 0:  # Patient sain
        if probability >= 0.8:
            recommendations.append("✅ Santé cardiovasculaire excellente. Continuez vos bonnes habitudes de vie.")
        else:
            recommendations.append("⚠️ Risque faible détecté. Maintenez une bonne hygiène de vie (alimentation équilibrée, activité physique régulière).")
        
        # Recommandations de prévention générales
        if not any("Bilan de santé" in r for r in recommendations):
            recommendations.append("📋 Bilan de santé annuel recommandé.")
    
    else:  # Patient malade
        if probability < 0.6:
            recommendations.append("🔶 Risque modéré. Consultation médicale recommandée pour bilan cardiovasculaire.")
        elif probability < 0.8:
            recommendations.append("⚠️ RISQUE ÉLEVÉ. Consultation cardiologique rapide impérative.")
        else:
            recommendations.append("💥 RISQUE TRÈS ÉLEVÉ. Consultation URGENTE recommandée.")
    
    # ================================================================
    # SI AUCUNE RECOMMANDATION N'A ÉTÉ AJOUTÉE (cas improbable)
    # ================================================================
    
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