"""
utils.py - Fonctions utilitaires pour Flask — CardioPredict
Recommandations médicales personnalisées par facteur de risque cardiovasculaire.
"""


def generate_recommendations(patient_data, prediction, probability):
    """
    Génère des recommandations médicales personnalisées et concises
    basées sur les facteurs de risque individuels du patient.
    """
    recommendations = []

    # =========================================================================
    # 1. ÂGE
    # =========================================================================
    age = patient_data.get('age', 0)
    if age > 65:
        recommendations.append(
            f"Âge avancé ({age} ans) : bilan annuel (ECG, lipides, glycémie). "
            f"Marche 30 min/jour, alimentation riche en fruits et légumes, "
            f"réduire le sel et les graisses saturées."
        )
    elif age > 50:
        recommendations.append(
            f"Âge ({age} ans) : bilan cardiovasculaire tous les 2 ans. "
            f"Activité physique régulière, alimentation équilibrée, "
            f"arrêt du tabac si fumeur."
        )

    # =========================================================================
    # 2. PRESSION ARTÉRIELLE
    # =========================================================================
    trestbps = patient_data.get('trestbps', 0)
    if trestbps >= 160:
        recommendations.append(
            f"Tension sévère ({trestbps} mmHg) : consultation URGENTE. "
            f"Régime sans sel (< 5g/jour), marche 30 min/jour, "
            f"automesure tensionnelle matin et soir."
        )
    elif trestbps >= 140:
        recommendations.append(
            f"Hypertension ({trestbps} mmHg) : consulter un médecin. "
            f"Réduire le sel, fruits et légumes, marche 30 min/jour, "
            f"perte de poids si surpoids."
        )
    elif trestbps >= 130:
        recommendations.append(
            f"Pré-hypertension ({trestbps} mmHg) : surveiller la tension tous les 6 mois. "
            f"Réduire le sel, activité physique régulière, contrôler le poids."
        )

    # =========================================================================
    # 3. CHOLESTÉROL
    # =========================================================================
    chol = patient_data.get('chol', 0)
    if chol >= 280:
        recommendations.append(
            f"Cholestérol sévère ({chol} mg/dL) : consultation pour bilan lipidique. "
            f"Supprimer graisses saturées (beurre, charcuterie, fritures), "
            f"adopter régime méditerranéen (huile d'olive, poissons gras, fibres)."
        )
    elif chol >= 240:
        recommendations.append(
            f"Cholestérol élevé ({chol} mg/dL) : régime méditerranéen, "
            f"réduire graisses saturées, poisson gras 2x/semaine, "
            f"marche 30 min/jour. Bilan lipidique dans 3 mois."
        )
    elif chol >= 200:
        recommendations.append(
            f"Cholestérol limite haut ({chol} mg/dL) : surveiller alimentation, "
            f"limiter graisses animales, privilégier fibres et poissons, "
            f"activité physique régulière. Bilan lipidique annuel."
        )

    # =========================================================================
    # 4. GLYCÉMIE (DIABÈTE)
    # =========================================================================
    if patient_data.get('fbs', 0) == 1:
        recommendations.append(
            f"Glycémie à jeun élevée (> 120 mg/dL) : consulter pour dosage HbA1c. "
            f"Réduire sucres rapides (sodas, pâtisseries, pain blanc), "
            f"marche 15 min après les repas, fractionner les repas."
        )

    # =========================================================================
    # 5. FRÉQUENCE CARDIAQUE MAXIMALE
    # =========================================================================
    thalach = patient_data.get('thalach', 0)
    if thalach < 100:
        recommendations.append(
            f"Fréquence cardiaque maximale très basse ({thalach} bpm) : "
            f"consultation cardiologique URGENTE. Bilan : ECG, échocardiographie, "
            f"test d'effort. Ne pas pratiquer d'exercice intense sans avis."
        )
    elif thalach < 120:
        recommendations.append(
            f"Fréquence cardiaque maximale basse ({thalach} bpm) : "
            f"consultation cardiologique recommandée. Reconditionnement progressif "
            f"à l'effort (débuter par 15 min de marche)."
        )

    # =========================================================================
    # 6. ANGINE À L'EFFORT
    # =========================================================================
    if patient_data.get('exang', 0) == 1:
        recommendations.append(
            f"Angine à l'effort détectée : consultation cardiologique RAPIDE "
            f"(sous 8 jours). Arrêter tout effort déclenchant la douleur. "
            f"Si douleur au repos ou aggravée : appel URGENCE (15)."
        )

    # =========================================================================
    # 7. DÉPRESSION ST (ISCHÉMIE)
    # =========================================================================
    oldpeak = patient_data.get('oldpeak', 0)
    if oldpeak >= 3.0:
        recommendations.append(
            f"Dépression ST sévère ({oldpeak} mm) : ischémie myocardique probable. "
            f"Consultation cardiologique URGENTE. Arrêt immédiat de tout effort. "
            f"Si douleur thoracique : appel URGENCE (15)."
        )
    elif oldpeak >= 1.5:
        recommendations.append(
            f"Dépression ST significative ({oldpeak} mm) : ECG d'effort recommandé. "
            f"Limiter les efforts physiques intenses, surveillance cardiologique."
        )

    # =========================================================================
    # 8. VAISSEAUX CORONAIRES OBSTRUÉS (CORONAROGRAPHIE)
    # =========================================================================
    ca = patient_data.get('ca', 0)
    if ca >= 3:
        recommendations.append(
            f"{ca} vaisseaux obstrués (tritronculaire) : suivi cardiologique tous les 3 mois. "
            f"Traitement strict (antiagrégants, statine). Réadaptation cardiaque. "
            f"Reconnaître les signes d'alerte : douleur thoracique, essoufflement brutal."
        )
    elif ca == 2:
        recommendations.append(
            f"2 vaisseaux obstrués (bitronculaire) : suivi cardiologique tous les 6 mois. "
            f"Traitement : statine, antiagrégant. Contrôle tension, cholestérol, glycémie. "
            f"Marche 30 min/jour (après validation médicale). Arrêt total du tabac."
        )
    elif ca == 1:
        recommendations.append(
            f"Sténose coronaire unique : suivi cardiologique annuel. "
            f"Traitement préventif, alimentation équilibrée, activité physique adaptée."
        )

    # =========================================================================
    # 9. THALASSÉMIE (SCINTIGRAPHIE)
    # =========================================================================
    thal = patient_data.get('thal', 0)
    if thal == 2:
        recommendations.append(
            f"Défaut fixe à la scintigraphie : séquelle d'infarctus probable. "
            f"Bilan cardiologique complet (échocardiographie). "
            f"Traitement préventif secondaire : statine, antiagrégant."
        )
    elif thal == 3:
        recommendations.append(
            f"Défaut réversible à la scintigraphie : ISCHEMIE ACTIVE. "
            f"Consultation cardiologique URGENTE (sous 48h). Coronarographie souvent indiquée. "
            f"Zéro tabac, arrêt de tout effort intense."
        )

    # =========================================================================
    # 10. TYPE DE DOULEUR THORACIQUE
    # =========================================================================
    cp = patient_data.get('cp', 0)
    if cp == 1:
        recommendations.append(
            f"Angine typique : bilan coronarien recommandé. "
            f"Éviter les efforts déclenchants, consulter un cardiologue."
        )

    # =========================================================================
    # 11. RECOMMANDATION GLOBALE SELON LE RISQUE
    # =========================================================================
    if prediction == 0:
        p_sain = 1 - probability
        if p_sain >= 0.85:
            recommendations.append(
                "Santé cardiovasculaire excellente : continuez vos bonnes habitudes. "
                "Marche 30 min/jour, 5 fruits/légumes par jour, sommeil 7-8h. "
                "Bilan de santé annuel."
            )
        else:
            recommendations.append(
                "Risque faible détecté : bilan de santé annuel recommandé. "
                "Corriger les facteurs de risque identifiés, activité physique régulière, "
                "alimentation équilibrée, arrêt du tabac si fumeur."
            )
    else:
        if probability < 0.6:
            recommendations.append(
                "Risque modéré : consultation médicale dans les 2 à 4 semaines. "
                "Bilan complet (tension, lipides, glycémie, ECG). "
                "Appliquer les recommandations spécifiques ci-dessus."
            )
        elif probability < 0.8:
            recommendations.append(
                "Risque élevé : consultation cardiologique RAPIDE (sous 8 jours) IMPERATIVE. "
                "ECG, échocardiographie, bilan biologique complet. "
                "Si douleur thoracique ou essoufflement : appel URGENCE (15)."
            )
        else:
            recommendations.append(
                "RISQUE TRES ELEVE : consultation URGENTE (sous 48h). "
                "Ne pas laisser le patient repartir sans avis cardiologique. "
                "Numéros d'urgence : 15, 18, 112."
            )

    # Supprimer les doublons (tout en gardant l'ordre)
    unique = []
    for r in recommendations:
        if r not in unique:
            unique.append(r)

    return unique


def get_risk_level(prediction, probability):
    """
    Détermine le niveau de risque clinique à partir de la prédiction et de P(malade).

    Niveaux :
        0 : prediction=0, P(malade) < 20%
        1 : prediction=0, P(malade) entre 20% et 50%
        2 : prediction=1, P(malade) entre 50% et 60%
        3 : prediction=1, P(malade) entre 60% et 80%
        4 : prediction=1, P(malade) >= 80%
    """
    if prediction == 0:
        if probability < 0.20:
            return (0, "Aucun risque", "Santé cardiovasculaire excellente. Bilan annuel recommandé.")
        else:
            return (1, "Risque faible", "Risque faible détecté. Surveillance médicale régulière.")
    else:
        if probability < 0.60:
            return (2, "Risque modéré", "Risque modéré. Consultation médicale dans le mois.")
        elif probability < 0.80:
            return (3, "Risque élevé", "Risque élevé. Consultation cardiologique rapide (sous 8 jours).")
        else:
            return (4, "Risque très élevé", "RISQUE TRES ELEVE - Consultation URGENTE (sous 48h).")