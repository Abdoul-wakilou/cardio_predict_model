# app.py - Version corrigée avec le bon prétraitement
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

print("🔄 Chargement du modèle et du préprocesseur pour Flask...")

# Charger le modèle ET le préprocesseur
model = joblib.load('model/best_model.joblib')
preprocessor = joblib.load('model/preprocessor.joblib')

print("✅ Modèle et préprocesseur chargés!")
print(f"   Modèle: {type(model).__name__}")
print(f"   Préprocesseur: {type(preprocessor).__name__}")

@app.route('/')
def home():
    return jsonify({
        "message": "API CardioPredict - Cleveland Heart Disease",
        "status": "active",
        "model": "Régression Logistique",
        "features_required": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal: reçoit des données BRUTES, prétraite, prédit"""
    try:
        # 1. Recevoir les données brutes de Flutter
        data = request.get_json()
        
        # 2. Vérifier les features requises
        required = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({
                'error': f'Features manquantes: {missing}',
                'required_features': required
            }), 400
        
        # 3. Convertir en DataFrame (UNE SEULE LIGNE)
        # IMPORTANT: garder le même ordre que pendant l'entraînement
        input_df = pd.DataFrame([{
            'age': data['age'],
            'sex': data['sex'],
            'cp': data['cp'],
            'trestbps': data['trestbps'],
            'chol': data['chol'],
            'fbs': data['fbs'],
            'restecg': data['restecg'],
            'thalach': data['thalach'],
            'exang': data['exang'],
            'oldpeak': data['oldpeak'],
            'slope': data['slope'],
            'ca': data['ca'],
            'thal': data['thal']
        }])
        
        # 4. APPLIQUER LE MÊME PRÉTRAITEMENT que pendant l'entraînement
        # C'est la partie CRITIQUE qui résout votre problème
        processed_data = preprocessor.transform(input_df)
        
        # 5. Faire la prédiction
        prediction = model.predict(processed_data)[0]  # 0 ou 1
        probabilities = model.predict_proba(processed_data)[0]  # [prob_sain, prob_malade]
        
        # 6. Formater la réponse pour Flutter
        response = {
            'prediction': int(prediction),
            'prediction_label': 'Risque Élevé' if prediction == 1 else 'Risque Faible/Modéré',
            'probability_no_disease': float(probabilities[0]),
            'probability_disease': float(probabilities[1]),
            'risk_score_percent': float(probabilities[1] * 100),
            'status': 'success',
            'message': 'Prédiction effectuée avec succès'
        }
        
        # 7. Log pour debug
        print(f"📊 Prédiction: {prediction} (prob: {probabilities[1]:.3f})")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'message': 'Erreur lors du traitement'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "CardioPredict API"}), 200

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 API CardioPredict démarrée")
    print("   URL: http://localhost:5000")
    print("   Test: POST http://localhost:5000/predict")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)