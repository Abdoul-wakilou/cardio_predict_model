
# Configuration pour Flask - CardioPredict
# Généré automatiquement par train_and_save_pipeline.py

FEATURES_ORDER = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['cp', 'restecg', 'slope', 'thal']
BINARY_FEATURES = ['sex', 'fbs', 'exang']

MODEL_INFO = {
    'name': 'Régression Logistique',
    'accuracy': 0.8222,
    'model_file': 'best_model.joblib',
    'preprocessor_file': 'preprocessor.joblib'
}

# Exemple de requête Flask:

# curl -X POST http://localhost:5000/predict \
#   -H "Content-Type: application/json" \
#   -d '{'age': 67, 'sex': 1, 'cp': 4, 'trestbps': 120, 'chol': 237, 'fbs': 0, 'restecg': 0, 'thalach': 71, 'exang': 0, 'oldpeak': 1.0, 'slope': 2, 'ca': 0, 'thal': 3}'
