import pandas as pd
import numpy as np
import os

from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/train-test.csv")

df["date"] = pd.to_datetime(df["date"])


# ============================================================
# 2. CLEAN DATA
# ============================================================

# Negative weights are invalid
df.loc[df["weight"] < 0, "weight"] = np.nan


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear

# Route feature
df["route"] = df["pickup"] + " -> " + df["delivery"]


# ============================================================
# 4. FEATURES
# ============================================================

features = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "market_index",
    "quote_signal",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "route"
]

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]


# ============================================================
# 5. TIME-BASED TRAIN / VALIDATION SPLIT
# ============================================================

train_df = df[df["date"] < "2025-09-01"].copy()

valid_df = df[df["date"] >= "2025-09-01"].copy()

X_train = train_df[features]
X_valid = valid_df[features]

y_train = train_df["posted_rate"]
y_valid = valid_df["posted_rate"]


print("=" * 60)
print("FINAL FREIGHT RATE MODEL")
print("=" * 60)

print(f"Training rows:   {len(train_df)}")
print(f"Validation rows: {len(valid_df)}")

print(
    f"\nTraining period: "
    f"{train_df['date'].min().date()} → "
    f"{train_df['date'].max().date()}"
)

print(
    f"Validation period: "
    f"{valid_df['date'].min().date()} → "
    f"{valid_df['date'].max().date()}"
)


# ============================================================
# 6. LOG TRANSFORMATION
# ============================================================

# Helps CatBoost handle the strongly right-skewed target
y_train_log = np.log1p(y_train)


# ============================================================
# 7. TRAIN FINAL CATBOOST MODEL
# ============================================================

print("\nTraining final Log-Target CatBoost model...")

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    verbose=100,
    random_seed=42
)

model.fit(
    X_train,
    y_train_log,
    cat_features=categorical_features
)


# ============================================================
# 8. PREDICTION
# ============================================================

predicted_log = model.predict(X_valid)

# Convert log predictions back to dollar scale
y_pred = np.expm1(predicted_log)


# Prevent tiny negative values
y_pred = np.maximum(y_pred, 0)


# ============================================================
# 9. MODEL METRICS
# ============================================================

mae = mean_absolute_error(
    y_valid,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        y_pred
    )
)

r2 = r2_score(
    y_valid,
    y_pred
)


print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE:  ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R²:   {r2:.4f}")


# ============================================================
# 10. PREDICTION ACCURACY
# ============================================================

errors = np.abs(
    y_valid.values - y_pred
)

print("\nPrediction Accuracy:")

for threshold in [50, 100, 200, 500, 1000]:

    percentage = (
        (errors <= threshold).mean() * 100
    )

    print(
        f"Within ${threshold}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# 11. HIGH-RATE ANALYSIS
# ============================================================

HIGH_THRESHOLD = 10000

high_mask = y_valid.values > HIGH_THRESHOLD

print("\n" + "=" * 60)
print("HIGH-RATE VALIDATION ANALYSIS")
print("=" * 60)

print(
    f"Loads above ${HIGH_THRESHOLD}: "
    f"{high_mask.sum()}"
)

if high_mask.sum() > 0:

    high_actual = y_valid.values[high_mask]
    high_predicted = y_pred[high_mask]

    high_mae = mean_absolute_error(
        high_actual,
        high_predicted
    )

    print(
        f"High-rate MAE: "
        f"${high_mae:.2f}"
    )


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.get_feature_importance()
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance.to_string(index=False)
)


# ============================================================
# 13. SAVE VALIDATION PREDICTIONS
# ============================================================

output = valid_df.copy()

output["predicted_rate"] = y_pred

output["error"] = (
    output["posted_rate"] -
    output["predicted_rate"]
)

output["absolute_error"] = np.abs(
    output["error"]
)

output.to_csv(
    "data/final_model_predictions.csv",
    index=False
)

print(
    "\nValidation predictions saved to:"
    " data/final_model_predictions.csv"
)


# ============================================================
# 14. SAVE FEATURE IMPORTANCE
# ============================================================

importance.to_csv(
    "data/final_feature_importance.csv",
    index=False
)

print(
    "Feature importance saved to:"
    " data/final_feature_importance.csv"
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model.save_model(
    "models/final_freight_rate_catboost.cbm"
)

print(
    "Model saved to:"
    " models/final_freight_rate_catboost.cbm"
)


# ============================================================
# 16. SHOW SAMPLE PREDICTIONS
# ============================================================

sample = output[
    [
        "load_id",
        "posted_rate",
        "predicted_rate",
        "absolute_error"
    ]
].head(10)

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(
    sample.to_string(index=False)
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPLETE")
print("=" * 60)