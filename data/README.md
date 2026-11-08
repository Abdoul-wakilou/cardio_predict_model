# UCI Heart Disease (4 Centers) - Combined Dataset with Documented Missing Values

## Important update: version 2 replaces version 1

**Please do not use version 1 to train or evaluate models.** Version 1 filled missing values with the median of each patient's diagnosis class, before any train/test split. This is target leakage: for the 525 patients with no recorded `ca` value (62.9%), the imputed `ca` revealed the diagnosis exactly (0 for healthy patients, 1 for diseased patients).

Version 1 also kept `chol = 0` and `trestbps = 0`, which are physiologically impossible and act as missing-value codes. Their frequency depends on the center and the diagnosis.

In a re-analysis on the same 251 test patients, a Random Forest fell from 92.83% accuracy and 0.980 AUC to 78.49% accuracy and 0.869 AUC once these two issues were corrected. If you used version 1, results obtained with `ca` are likely overestimated.

Version 2 contains the same 834 patients in the same order, **without imputation**. Missing values are left empty so that each user can impute them on training data only.

## Overview

| | |
|---|---|
| Patients | 834 (463 diseased, 371 healthy) |
| Columns | 13 clinical features, 1 binary target, 1 center identifier |
| Task | Binary classification of angiographic heart disease |
| Source | UCI Machine Learning Repository, Heart Disease (1988) |

## Centers

| Center | Patients in the UCI file | Patients kept | Diseased (kept) |
|---|---:|---:|---:|
| Cleveland Clinic Foundation (USA) | 303 | 303 | 45.9% |
| Hungarian Institute of Cardiology, Budapest | 294 | 272 | 37.9% |
| University Hospitals of Zurich and Basel (Switzerland) | 123 | 117 | 93.2% |
| V.A. Medical Center, Long Beach (USA) | 200 | 142 | 78.9% |
| **Total** | **920** | **834** | **55.5%** |

## Processing

1. Merge the four `processed.*.data` UCI files and read `?` and `-9` as missing.
2. Binarize the diagnosis: `target = 0` if `num == 0`, otherwise `target = 1`.
3. Recode `cp`, `slope` and `thal` to zero-based categories.
4. Remove the 86 patients with 4 or more missing features out of 13.
5. Mark `chol = 0` and `trestbps = 0` as missing. This step follows step 4 to keep the same patients as version 1.
6. Keep every remaining missing value empty. No imputation, scaling or encoding is applied.

## Columns

| Column | Description | Values | Missing |
|---|---|---|---:|
| `age` | Age | years | 0 |
| `sex` | Sex | 0 = female, 1 = male | 0 |
| `cp` | Chest pain type | 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic | 0 |
| `trestbps` | Resting blood pressure | mm Hg | 3 |
| `chol` | Serum cholesterol | mg/dL | 167 |
| `fbs` | Fasting blood sugar above 120 mg/dL | 0 = no, 1 = yes | 80 |
| `restecg` | Resting ECG | 0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy | 0 |
| `thalach` | Maximum heart rate achieved | bpm | 0 |
| `exang` | Exercise-induced angina | 0 = no, 1 = yes | 0 |
| `oldpeak` | ST depression induced by exercise relative to rest | mm | 2 |
| `slope` | Slope of the peak exercise ST segment | 0 = upsloping, 1 = flat, 2 = downsloping | 225 |
| `ca` | Number of major vessels colored by fluoroscopy | 0 to 3 | 525 |
| `thal` | Thallium stress test result | 0 = normal, 1 = fixed defect, 2 = reversible defect | 405 |
| `target` | Heart disease diagnosis | 0 = absent, 1 = present | 0 |
| `centre` | Center of origin | Cleveland, Hungarian, Switzerland, Long Beach VA | 0 |

## Recommended use

- **Split first, then impute**, with the imputer inside a pipeline fitted on training data only.
- **Never use `target` to impute** another column.
- **Exclude `centre` from the features** unless your study explicitly models it. Missingness and disease prevalence both depend strongly on the center.
- **Test across centers** (train on three, test on the fourth) to estimate how a model transfers.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("/kaggle/input/uci-heart-disease-clean-combined-dataset/heart_uci_combined.csv")
X = df.drop(columns=["target", "centre"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

numeric = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
categorical = ["cp", "restecg", "slope", "thal"]
binary = ["sex", "fbs", "exang"]

preprocessor = ColumnTransformer(
    [
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        (
            "cat",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            categorical,
        ),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
    ]
)

model = make_pipeline(preprocessor, LogisticRegression(max_iter=5000))
model.fit(X_train, y_train)
print(f"Test AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.3f}")
```

## Limitations

- The data are historical (published in 1988) and come from populations in the USA and Europe.
- Disease prevalence varies from 37.9% to 93.2% between centers.
- `ca` and `thal` are mostly missing outside Cleveland, and they require specialized examinations.
- This dataset is intended for research and education. It must not be used on its own for clinical decisions.

## Code and results

The full pipeline, model comparison and center-by-center validation are available at [github.com/Abdoul-wakilou/cardio_predict_model](https://github.com/Abdoul-wakilou/cardio_predict_model).

## Attribution

Please cite the original data:

> Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

Principal investigators: Andras Janosi, M.D. (Hungarian Institute of Cardiology), William Steinbrunn, M.D. (University Hospital, Zurich), Matthias Pfisterer, M.D. (University Hospital, Basel), Robert Detrano, M.D., Ph.D. (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation).

The UCI source data are distributed under the CC BY 4.0 license. Dataset compiled and maintained by Abdoul-Wakilou TIGA.
