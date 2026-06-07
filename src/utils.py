"""
utils.py — Recommandations médicales CardioPredict v3.0
=========================================================
Encodage 0-based :
  cp    : 0=typique, 1=atypique, 2=non-angineuse, 3=asymptomatique
  slope : 0=ascendante, 1=plate, 2=descendante
  thal  : 0=normal, 1=défaut fixe, 2=défaut réversible
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
            "Marche 30 min/jour, alimentation riche en fruits et légumes, "
            "réduire le sel et les graisses saturées."
        )
    elif age > 50:
        recommendations.append(
            f"Âge ({age} ans) : bilan cardiovasculaire tous les 2 ans. "
            "Activité physique régulière, alimentation équilibrée, "
            "arrêt du tabac si fumeur."
        )

    # =========================================================================
    # 2. PRESSION ARTÉRIELLE
    # =========================================================================
    trestbps = patient_data.get('trestbps', 0)
    if trestbps >= 160:
        recommendations.append(
            f"Tension sévère ({trestbps} mmHg) : consultation URGENTE. "
            "Régime sans sel (< 5 g/jour), marche 30 min/jour, "
            "automesure tensionnelle matin et soir."
        )
    elif trestbps >= 140:
        recommendations.append(
            f"Hypertension ({trestbps} mmHg) : consulter un médecin. "
            "Réduire le sel, fruits et légumes, marche 30 min/jour, "
            "perte de poids si surpoids."
        )
    elif trestbps >= 130:
        recommendations.append(
            f"Pré-hypertension ({trestbps} mmHg) : surveiller la tension tous les 6 mois. "
            "Réduire le sel, activité physique régulière, contrôler le poids."
        )

    # =========================================================================
    # 3. CHOLESTÉROL
    # =========================================================================
    chol = patient_data.get('chol', 0)
    if chol >= 280:
        recommendations.append(
            f"Cholestérol sévère ({chol} mg/dL) : consultation pour bilan lipidique. "
            "Supprimer graisses saturées (beurre, charcuterie, fritures), "
            "adopter régime méditerranéen (huile d'olive, poissons gras, fibres)."
        )
    elif chol >= 240:
        recommendations.append(
            f"Cholestérol élevé ({chol} mg/dL) : régime méditerranéen, "
            "réduire graisses saturées, poisson gras 2×/semaine, "
            "marche 30 min/jour. Bilan lipidique dans 3 mois."
        )
    elif chol >= 200:
        recommendations.append(
            f"Cholestérol limite-haut ({chol} mg/dL) : surveiller l'alimentation, "
            "limiter graisses animales, privilégier fibres et poissons, "
            "activité physique régulière. Bilan lipidique annuel."
        )

    # =========================================================================
    # 4. GLYCÉMIE
    # =========================================================================
    if patient_data.get('fbs', 0) == 1:
        recommendations.append(
            "Glycémie à jeun élevée (> 120 mg/dL) : consulter pour dosage HbA1c. "
            "Réduire sucres rapides (sodas, pâtisseries, pain blanc), "
            "marche 15 min après les repas, fractionner les repas."
        )

    # =========================================================================
    # 5. FRÉQUENCE CARDIAQUE MAXIMALE
    # =========================================================================
    thalach = patient_data.get('thalach', 0)
    if thalach < 100:
        recommendations.append(
            f"Fréquence cardiaque maximale très basse ({thalach} bpm) : "
            "consultation cardiologique URGENTE. "
            "Bilan : ECG, échocardiographie, test d'effort."
        )
    elif thalach < 120:
        recommendations.append(
            f"Fréquence cardiaque maximale basse ({thalach} bpm) : "
            "consultation cardiologique recommandée. "
            "Reconditionnement progressif à l'effort (débuter par 15 min de marche)."
        )

    # =========================================================================
    # 6. ANGINE À L'EFFORT
    # =========================================================================
    if patient_data.get('exang', 0) == 1:
        recommendations.append(
            "Angine à l'effort détectée : consultation cardiologique RAPIDE (sous 8 jours). "
            "Arrêter tout effort déclenchant la douleur. "
            "Si douleur au repos ou aggravée : appel URGENCE (15)."
        )

    # =========================================================================
    # 7. DÉPRESSION ST
    # =========================================================================
    oldpeak = patient_data.get('oldpeak', 0)
    if oldpeak >= 3.0:
        recommendations.append(
            f"Dépression ST sévère ({oldpeak} mm) : ischémie myocardique probable. "
            "Consultation cardiologique URGENTE. Arrêt immédiat de tout effort. "
            "Si douleur thoracique : appel URGENCE (15)."
        )
    elif oldpeak >= 1.5:
        recommendations.append(
            f"Dépression ST significative ({oldpeak} mm) : ECG d'effort recommandé. "
            "Limiter les efforts physiques intenses, surveillance cardiologique."
        )

    # =========================================================================
    # 8. VAISSEAUX OBSTRUÉS
    # =========================================================================
    ca = patient_data.get('ca', 0)
    if ca >= 3:
        recommendations.append(
            f"{ca} vaisseaux obstrués (tritronculaire) : suivi cardiologique tous les 3 mois. "
            "Traitement strict (antiagrégants, statine). Réadaptation cardiaque. "
            "Reconnaître les signes d'alerte : douleur thoracique, essoufflement brutal."
        )
    elif ca == 2:
        recommendations.append(
            "2 vaisseaux obstrués (bitronculaire) : suivi cardiologique tous les 6 mois. "
            "Traitement : statine, antiagrégant. Contrôle tension, cholestérol, glycémie. "
            "Arrêt total du tabac."
        )
    elif ca == 1:
        recommendations.append(
            "Sténose coronaire unique détectée : suivi cardiologique annuel. "
            "Traitement préventif, alimentation équilibrée, activité physique adaptée."
        )

    # =========================================================================
    # 9. THALASSÉMIE — encodage 0-based
    #    0=normal, 1=défaut fixe, 2=défaut réversible
    # =========================================================================
    thal = patient_data.get('thal', 0)
    if thal == 1:
        recommendations.append(
            "Défaut fixe à la scintigraphie : séquelle d'infarctus probable. "
            "Bilan cardiologique complet (échocardiographie). "
            "Traitement préventif secondaire : statine, antiagrégant."
        )
    elif thal == 2:
        recommendations.append(
            "Défaut réversible à la scintigraphie : ISCHÉMIE ACTIVE. "
            "Consultation cardiologique URGENTE (sous 48 h). Coronarographie souvent indiquée. "
            "Zéro tabac, arrêt de tout effort intense."
        )

    # =========================================================================
    # 10. TYPE DE DOULEUR THORACIQUE — encodage 0-based
    #     0=typique, 1=atypique, 2=non-angineuse, 3=asymptomatique
    # =========================================================================
    cp = patient_data.get('cp', 3)
    if cp == 0:
        recommendations.append(
            "Angine typique confirmée : bilan coronarien recommandé. "
            "Éviter les efforts déclenchants, consulter un cardiologue."
        )

    # =========================================================================
    # 11. RECOMMANDATION GLOBALE SELON LE RISQUE
    # =========================================================================
    if prediction == 0:
        p_sain = 1 - probability
        if p_sain >= 0.85:
            recommendations.append(
                "Santé cardiovasculaire excellente : continuez vos bonnes habitudes. "
                "Marche 30 min/jour, 5 fruits/légumes par jour, sommeil 7–8 h. "
                "Bilan de santé annuel."
            )
        else:
            recommendations.append(
                "Risque faible détecté : bilan de santé annuel recommandé. "
                "Corriger les facteurs de risque identifiés, activité physique régulière, "
                "alimentation équilibrée, arrêt du tabac si fumeur."
            )
    else:
        if probability < 0.60:
            recommendations.append(
                "Risque modéré : consultation médicale dans les 2 à 4 semaines. "
                "Bilan complet (tension, lipides, glycémie, ECG). "
                "Appliquer les recommandations spécifiques ci-dessus."
            )
        elif probability < 0.80:
            recommendations.append(
                "Risque élevé : consultation cardiologique RAPIDE (sous 8 jours) IMPÉRATIVE. "
                "ECG, échocardiographie, bilan biologique complet. "
                "Si douleur thoracique ou essoufflement : appel URGENCE (15)."
            )
        else:
            recommendations.append(
                "RISQUE TRÈS ÉLEVÉ : consultation URGENTE (sous 48 h). "
                "Ne pas laisser le patient repartir sans avis cardiologique. "
                "Numéros d'urgence : 15 · 18 · 112."
            )

    # Dédoublonnage (conserve l'ordre)
    seen, unique = set(), []
    for r in recommendations:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def get_risk_level(prediction, probability):
    """
    Détermine le niveau de risque clinique.

    Niveaux :
      0 — prediction=0, P(malade) < 20 %    → Aucun risque
      1 — prediction=0, P(malade) 20–50 %   → Risque faible
      2 — prediction=1, P(malade) 50–60 %   → Risque modéré
      3 — prediction=1, P(malade) 60–80 %   → Risque élevé
      4 — prediction=1, P(malade) ≥ 80 %    → Risque très élevé
    """
    if prediction == 0:
        if probability < 0.20:
            return (0, "Aucun risque",
                    "Santé cardiovasculaire excellente. Bilan annuel recommandé.")
        else:
            return (1, "Risque faible",
                    "Risque faible détecté. Surveillance médicale régulière.")
    else:
        if probability < 0.60:
            return (2, "Risque modéré",
                    "Risque modéré. Consultation médicale dans le mois.")
        elif probability < 0.80:
            return (3, "Risque élevé",
                    "Risque élevé. Consultation cardiologique rapide (sous 8 jours).")
        else:
            return (4, "Risque très élevé",
                    "RISQUE TRÈS ÉLEVÉ — Consultation URGENTE (sous 48 h).")