from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

print("Loading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

print(f"Dataset shape: {X.shape}")
print(f"Missing values: {X.isnull().sum().sum()}")


# --------------------------------------------------
# 2. Train / Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# --------------------------------------------------
# 3. Define models
# --------------------------------------------------

models = {
    "Linear": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]
    ),

    "Ridge": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge())
        ]
    ),

    "Lasso": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=20000))
        ]
    ),

    "ElasticNet": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=20000))
        ]
    )
}


# --------------------------------------------------
# 4. Hyperparameter grids
# --------------------------------------------------

param_grids = {
    "Linear": {},

    "Ridge": {
        "model__alpha": [0.01, 0.1, 1, 10, 100]
    },

    "Lasso": {
        "model__alpha": [0.001, 0.01, 0.1, 1]
    },

    "ElasticNet": {
        "model__alpha": [0.001, 0.01, 0.1, 1],
        "model__l1_ratio": [0.2, 0.5, 0.8]
    }
}


# --------------------------------------------------
# 5. Cross-validation
# --------------------------------------------------

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 6. Hyperparameter tuning
# --------------------------------------------------

searches = {}
rows = []

print("\nStarting cross-validation and hyperparameter tuning...")

for name, pipeline in models.items():

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    searches[name] = search

    rows.append(
        {
            "Model": name,
            "CV_RMSE": -search.best_score_,
            "Best_Params": str(search.best_params_)
        }
    )

    print(f"\n{name}")
    print(f"Best CV RMSE: {-search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")


# --------------------------------------------------
# 7. Compare models
# --------------------------------------------------

cv_results = (
    pd.DataFrame(rows)
    .sort_values("CV_RMSE")
    .reset_index(drop=True)
)

print("\nModel comparison:")
print(cv_results.to_string(index=False))


# Save CV results
cv_results.to_csv(
    REPORTS_DIR / "cv_results.csv",
    index=False
)


# --------------------------------------------------
# 8. Select best model
# --------------------------------------------------

best_name = cv_results.iloc[0]["Model"]

best_model = searches[best_name].best_estimator_

print(f"\nSelected model: {best_name}")


# --------------------------------------------------
# 9. Final test evaluation
# --------------------------------------------------

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

test_results = pd.DataFrame(
    [
        {
            "Selected_Model": best_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    ]
)

print("\nFinal test results:")
print(test_results.to_string(index=False))


# Save test results
test_results.to_csv(
    REPORTS_DIR / "test_results.csv",
    index=False
)


# --------------------------------------------------
# 10. Error analysis
# --------------------------------------------------

error_analysis = X_test.copy()

error_analysis["actual"] = y_test
error_analysis["predicted"] = y_pred

error_analysis["error"] = (
    error_analysis["actual"]
    - error_analysis["predicted"]
)

error_analysis["absolute_error"] = (
    error_analysis["error"].abs()
)

error_analysis = error_analysis.sort_values(
    "absolute_error",
    ascending=False
)

print("\nTop 10 largest prediction errors:")
print(
    error_analysis.head(10).to_string()
)


# Save error analysis
error_analysis.to_csv(
    REPORTS_DIR / "error_analysis.csv",
    index=False
)


# --------------------------------------------------
# 11. Save best model
# --------------------------------------------------

model_path = MODELS_DIR / "housing_regression.joblib"

joblib.dump(
    best_model,
    model_path
)

print(f"\nModel saved to: {model_path}")


# --------------------------------------------------
# 12. Test saved model
# --------------------------------------------------

loaded_model = joblib.load(model_path)

sample_predictions = loaded_model.predict(
    X_test.iloc[:3]
)

print("\nSample predictions from saved model:")

for prediction in sample_predictions:
    print(f"{prediction:.4f}")


print("\nTraining pipeline completed successfully.")
