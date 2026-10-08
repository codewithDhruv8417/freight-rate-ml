import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. LOAD FINAL MODEL RESULTS
# ============================================================

predictions = pd.read_csv(
    "data/final_model_predictions.csv"
)

importance = pd.read_csv(
    "data/final_feature_importance.csv"
)

os.makedirs("plots", exist_ok=True)


# ============================================================
# ACTUAL VS PREDICTED - LOG SCALE
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    predictions["posted_rate"],
    predictions["predicted_rate"],
    alpha=0.4
)

max_value = max(
    predictions["posted_rate"].max(),
    predictions["predicted_rate"].max()
)

plt.plot(
    [1, max_value],
    [1, max_value],
    linestyle="--"
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Actual Freight Rate ($)")
plt.ylabel("Predicted Freight Rate ($)")
plt.title("Actual vs Predicted Freight Rates")

plt.tight_layout()

plt.savefig(
    "plots/actual_vs_predicted_log.png",
    dpi=300
)

plt.close()

# ============================================================
# 3. ERROR / RESIDUAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    predictions["error"],
    bins=50
)

plt.xlabel("Prediction Error ($)")
plt.ylabel("Number of Loads")
plt.title("Prediction Error Distribution")

plt.axvline(
    0,
    linestyle="--"
)

plt.tight_layout()

plt.savefig(
    "plots/error_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. FEATURE IMPORTANCE
# ============================================================

importance_sorted = importance.sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(9, 7))

plt.barh(
    importance_sorted["feature"],
    importance_sorted["importance"]
)

plt.xlabel("Importance (%)")
plt.ylabel("Feature")
plt.title("CatBoost Feature Importance")

plt.tight_layout()

plt.savefig(
    "plots/feature_importance.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. POSTED RATE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    predictions["posted_rate"],
    bins=60
)

plt.xlabel("Posted Freight Rate ($)")
plt.ylabel("Number of Loads")
plt.title("Freight Rate Distribution")

plt.tight_layout()

plt.savefig(
    "plots/rate_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. PREDICTION ACCURACY
# ============================================================

thresholds = [
    50,
    100,
    200,
    500,
    1000
]

accuracy = []

for threshold in thresholds:

    percentage = (
        predictions["absolute_error"]
        <= threshold
    ).mean() * 100

    accuracy.append(percentage)


plt.figure(figsize=(8, 6))

plt.bar(
    [str(x) for x in thresholds],
    accuracy
)

plt.xlabel("Maximum Prediction Error ($)")
plt.ylabel("Predictions Within Limit (%)")
plt.title("Prediction Accuracy")

for i, value in enumerate(accuracy):

    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.ylim(0, 110)

plt.tight_layout()

plt.savefig(
    "plots/prediction_accuracy.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. FINISHED
# ============================================================

print("\n" + "=" * 50)
print("ALL GRAPHS CREATED")
print("=" * 50)

print("\nSaved in:")

print("plots/actual_vs_predicted.png")
print("plots/error_distribution.png")
print("plots/feature_importance.png")
print("plots/rate_distribution.png")
print("plots/prediction_accuracy.png")
import pandas as pd
import matplotlib.pyplot as plt

predictions = pd.read_csv(
    "data/final_model_predictions.csv"
)

thresholds = [50, 100, 200, 500, 1000]

accuracy = []

for threshold in thresholds:
    percentage = (
        predictions["absolute_error"] <= threshold
    ).mean() * 100

    accuracy.append(percentage)


plt.figure(figsize=(8, 6))

plt.bar(
    [str(x) for x in thresholds],
    accuracy
)

plt.xlabel("Maximum Prediction Error ($)")
plt.ylabel("Predictions Within Limit (%)")
plt.title("Prediction Accuracy")

for i, value in enumerate(accuracy):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.ylim(0, 110)

plt.tight_layout()

plt.savefig(
    "plots/prediction_accuracy.png",
    dpi=300
)

plt.close()

print("Prediction accuracy graph created!")