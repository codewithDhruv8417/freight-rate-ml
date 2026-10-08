import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/train-test.csv")

print("First 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 2. BASIC STATISTICS
# ============================================================

print("\nDescriptive Statistics:")
print(df.describe(include="all").T)


# ============================================================
# 3. TARGET VARIABLE
# ============================================================

target = "posted_rate"

print("\nTarget Statistics:")
print(df[target].describe())


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(df["posted_rate"], bins=50)

plt.xlabel("Posted Rate")
plt.ylabel("Number of Loads")
plt.title("Distribution of Freight Rates")

plt.show()


# ============================================================
# 5. DATE CONVERSION
# ============================================================

df["date"] = pd.to_datetime(df["date"])

print("\nDate Range:")
print("Minimum:", df["date"].min())
print("Maximum:", df["date"].max())


# ============================================================
# 6. DATA QUALITY CHECKS
# ============================================================

print("\nNegative Weights:")

print(
    df[df["weight"] < 0][
        ["load_id", "weight", "distance", "equipment", "posted_rate"]
    ]
)

print("\nNumber of Negative Weights:")
print((df["weight"] < 0).sum())


# Fix negative weights
# Negative weight is invalid, so treat it as missing
df.loc[df["weight"] < 0, "weight"] = np.nan


# ============================================================
# 7. DUPLICATE CHECK
# ============================================================

print("\nDuplicate Complete Rows:")
print(df.duplicated().sum())

print("\nDuplicate Load IDs:")
print(df["load_id"].duplicated().sum())


# ============================================================
# 8. CATEGORICAL INFORMATION
# ============================================================

print("\nEquipment Values:")
print(df["equipment"].value_counts())

print("\nNumber of Pickup Locations:")
print(df["pickup"].nunique())

print("\nNumber of Delivery Locations:")
print(df["delivery"].nunique())


print("\nTop Pickup Locations:")
print(df["pickup"].value_counts().head(10))

print("\nTop Delivery Locations:")
print(df["delivery"].value_counts().head(10))


# ============================================================
# 9. LOADS PER MONTH
# ============================================================

print("\nLoads Per Month:")

monthly_loads = (
    df["date"]
    .dt.to_period("M")
    .value_counts()
    .sort_index()
)

print(monthly_loads)


# ============================================================
# 10. NUMERICAL CORRELATION
# ============================================================

numeric_columns = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate"
]

print("\nCorrelation with Posted Rate:")

correlation = (
    df[numeric_columns]
    .corr()["posted_rate"]
    .sort_values(ascending=False)
)

print(correlation)


# ============================================================
# 11. DISTANCE VS POSTED RATE
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["distance"],
    df["posted_rate"],
    alpha=0.10
)

plt.xlabel("Distance")
plt.ylabel("Posted Rate")
plt.title("Distance vs Posted Rate")

plt.show()


# ============================================================
# 12. DISTANCE TREND LINE
# ============================================================

x = df["distance"]
y = df["posted_rate"]

coefficient = np.polyfit(x, y, 1)
trend = np.poly1d(coefficient)

x_sorted = x.sort_values()

plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    alpha=0.10
)

plt.plot(
    x_sorted,
    trend(x_sorted),
    linewidth=2
)

plt.xlabel("Distance")
plt.ylabel("Posted Rate")
plt.title("Distance vs Posted Rate with Trend Line")

plt.show()


# ============================================================
# 13. RATE PER MILE - EDA ONLY
# ============================================================
# IMPORTANT:
# DO NOT USE rate_per_mile AS A MODEL FEATURE.
#
# rate_per_mile = posted_rate / distance
# Therefore it contains information from the target variable.
# Using it for prediction would cause DATA LEAKAGE.

df["rate_per_mile"] = (
    df["posted_rate"] / df["distance"]
)

print("\nRate Per Mile Statistics:")
print(df["rate_per_mile"].describe())


plt.figure(figsize=(10, 6))

plt.hist(
    df["rate_per_mile"],
    bins=100
)

plt.xlabel("Rate per Mile")
plt.ylabel("Number of Loads")
plt.title("Rate per Mile Distribution")

plt.show()


# ============================================================
# 14. POSTED RATE BY EQUIPMENT
# ============================================================

print("\nPosted Rate by Equipment:")

equipment_stats = (
    df.groupby("equipment")["posted_rate"]
    .agg(["count", "mean", "median", "std"])
)

print(equipment_stats)


df.boxplot(
    column="posted_rate",
    by="equipment",
    figsize=(10, 6)
)

plt.title("Posted Rate by Equipment")
plt.suptitle("")
plt.xlabel("Equipment")
plt.ylabel("Posted Rate")

plt.show()


# ============================================================
# 15. MONTHLY RATE TREND
# ============================================================

monthly_rate = (
    df.groupby(df["date"].dt.to_period("M"))["posted_rate"]
    .agg(["mean", "median"])
)

print("\nMonthly Freight Rate:")
print(monthly_rate)


plt.figure(figsize=(10, 5))

plt.plot(
    monthly_rate.index.astype(str),
    monthly_rate["mean"],
    marker="o",
    label="Mean"
)

plt.plot(
    monthly_rate.index.astype(str),
    monthly_rate["median"],
    marker="o",
    label="Median"
)

plt.xlabel("Month")
plt.ylabel("Posted Rate")
plt.title("Monthly Freight Rate Trend")

plt.xticks(rotation=45)
plt.legend()
plt.grid(axis="y")

plt.show()


# ============================================================
# 16. MARKET INDEX AND QUOTE SIGNAL
# ============================================================

print("\nMarket Index / Quote Signal Correlation:")

print(
    df[
        ["market_index", "quote_signal", "posted_rate"]
    ].corr()
)


# Market Index vs Posted Rate

plt.figure(figsize=(10, 5))

plt.scatter(
    df["market_index"],
    df["posted_rate"],
    alpha=0.10
)

plt.xlabel("Market Index")
plt.ylabel("Posted Rate")
plt.title("Market Index vs Posted Rate")

plt.show()


# ============================================================
# 17. FEATURE ENGINEERING
# ============================================================

# Date features

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear


# Route feature

df["route"] = (
    df["pickup"].astype(str)
    + "_"
    + df["delivery"].astype(str)
)


# ============================================================
# 18. MODEL FEATURES
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


target = "posted_rate"


print("\nModel Features:")
print(features)

print("\nNumber of Features:")
print(len(features))


# ============================================================
# 19. TIME-BASED TRAIN / VALIDATION SPLIT
# ============================================================

train_df = df[
    df["date"] < "2025-09-01"
].copy()


valid_df = df[
    (df["date"] >= "2025-09-01") &
    (df["date"] <= "2025-10-31")
].copy()


print("\n==============================")
print("TRAIN / VALIDATION SPLIT")
print("==============================")


print("\nTraining rows:")
print(len(train_df))


print("\nValidation rows:")
print(len(valid_df))


print("\nTraining period:")

print(
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)


print("\nValidation period:")

print(
    valid_df["date"].min(),
    "to",
    valid_df["date"].max()
)


# ============================================================
# 20. CREATE X AND y
# ============================================================

X_train = train_df[features]
y_train = train_df[target]

X_valid = valid_df[features]
y_valid = valid_df[target]


print("\nX_train shape:")
print(X_train.shape)

print("\ny_train shape:")
print(y_train.shape)

print("\nX_valid shape:")
print(X_valid.shape)

print("\ny_valid shape:")
print(y_valid.shape)
X_train = train_df[features]
y_train = train_df[target]

X_valid = valid_df[features]
y_valid = valid_df[target]
categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]
from catboost import CatBoostRegressor

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]

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
    y_train,
    cat_features=categorical_features
)
y_pred = model.predict(X_valid)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_valid, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_valid, y_pred)
)

r2 = r2_score(y_valid, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)
print("Training rows:", len(train_df))
print("Validation rows:", len(valid_df))
# =========================
# MODEL TRAINING
# =========================

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X_train = train_df[features]
y_train = train_df[target]

X_valid = valid_df[features]
y_valid = valid_df[target]

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]

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
    y_train,
    cat_features=categorical_features
)

# Predictions
y_pred = model.predict(X_valid)

# Evaluation
mae = mean_absolute_error(y_valid, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_valid, y_pred)
)

r2 = r2_score(y_valid, y_pred)

print("\n====================")
print("MODEL RESULTS")
print("====================")

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)
results = valid_df[["load_id", "posted_rate"]].copy()

results["predicted_rate"] = y_pred

results["error"] = (
    results["posted_rate"] - results["predicted_rate"]
)

results["absolute_error"] = (
    results["error"].abs()
)

print("\nLargest prediction errors:")

print(
    results
    .sort_values("absolute_error", ascending=False)
    .head(20)
)
for threshold in [50, 100, 200, 500, 1000]:
    percentage = (
        results["absolute_error"] <= threshold
    ).mean() * 100

    print(
        f"Within ${threshold}: {percentage:.2f}%"
    )
    plt.figure(figsize=(8, 8))

plt.scatter(
    y_valid,
    y_pred,
    alpha=0.15
)

minimum = min(y_valid.min(), y_pred.min())
maximum = max(y_valid.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Posted Rate")
plt.ylabel("Predicted Posted Rate")
plt.title("Actual vs Predicted Freight Rate")

plt.show()
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)
# ==========================================
# ERROR ANALYSIS
# ==========================================

results = valid_df[["load_id", "posted_rate"]].copy()

results["predicted_rate"] = y_pred

results["error"] = (
    results["posted_rate"] - results["predicted_rate"]
)

results["absolute_error"] = (
    results["error"].abs()
)

print("\nLargest prediction errors:")

print(
    results
    .sort_values("absolute_error", ascending=False)
    .head(20)
)


# Prediction accuracy by error threshold

print("\nPrediction accuracy:")

for threshold in [50, 100, 200, 500, 1000]:

    percentage = (
        results["absolute_error"] <= threshold
    ).mean() * 100

    print(
        f"Within ${threshold}: {percentage:.2f}%"
    )


# Actual vs predicted graph

plt.figure(figsize=(8, 8))

plt.scatter(
    y_valid,
    y_pred,
    alpha=0.15
)

minimum = min(y_valid.min(), y_pred.min())
maximum = max(y_valid.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Posted Rate")
plt.ylabel("Predicted Posted Rate")
plt.title("Actual vs Predicted Freight Rate")

plt.show()


# Feature importance

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)
# ==========================================
# INVESTIGATE EXTREME ERRORS
# ==========================================

error_analysis = valid_df.copy()

error_analysis["predicted_rate"] = y_pred

error_analysis["absolute_error"] = (
    error_analysis["posted_rate"]
    - error_analysis["predicted_rate"]
).abs()

print("\nExtreme error records:")

print(
    error_analysis[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "market_index",
            "quote_signal",
            "posted_rate",
            "predicted_rate",
            "absolute_error"
        ]
    ]
    .sort_values("absolute_error", ascending=False)
    .head(20)
)
print("\nValidation target statistics:")

print(
    valid_df["posted_rate"].describe()
)
print("\nVery high rate loads:")

print(
    valid_df[
        valid_df["posted_rate"] > 10000
    ][
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "market_index",
            "quote_signal",
            "posted_rate"
        ]
    ]
    .sort_values("posted_rate", ascending=False)
    .head(30)
)# ==========================================
# STEP 8: INSPECT EXTREME HIGH-RATE LOADS
# ==========================================

extreme = valid_df[
    valid_df["posted_rate"] > 10000
].copy()

print("\nNumber of validation loads above $10,000:")
print(len(extreme))

print("\nExtreme high-rate loads:")

print(
    extreme[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "market_index",
            "quote_signal",
            "posted_rate"
        ]
    ]
    .sort_values("posted_rate", ascending=False)
    .head(30)
)
# ==========================================
# STEP 9: COMPARE HIGH-RATE LOADS
# TRAINING vs VALIDATION
# ==========================================

print("\n==============================")
print("HIGH RATE COMPARISON")
print("==============================")

print("\nTraining loads above $10,000:")
print(
    (train_df["posted_rate"] > 10000).sum()
)

print("\nValidation loads above $10,000:")
print(
    (valid_df["posted_rate"] > 10000).sum()
)

print("\nTraining loads above $5,000:")
print(
    (train_df["posted_rate"] > 5000).sum()
)

print("\nValidation loads above $5,000:")
print(
    (valid_df["posted_rate"] > 5000).sum()
)
print("\nHighest training rates:")

print(
    train_df[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "market_index",
            "quote_signal",
            "posted_rate"
        ]
    ]
    .sort_values("posted_rate", ascending=False)
    .head(20)
)
print("\nHighest validation rates:")

print(
    valid_df[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "market_index",
            "quote_signal",
            "posted_rate"
        ]
    ]
    .sort_values("posted_rate", ascending=False)
    .head(20)
)
# ==========================================
# STEP 10: RATE PER MILE FOR EXTREME LOADS
# ==========================================

print("\n==============================")
print("EXTREME RATE PER MILE")
print("==============================")

extreme_train = train_df[train_df["posted_rate"] > 10000].copy()
extreme_valid = valid_df[valid_df["posted_rate"] > 10000].copy()

print("\nTraining extreme loads:")
print(
    extreme_train[
        ["load_id", "distance", "equipment", "weight",
         "market_index", "quote_signal", "posted_rate", "rate_per_mile"]
    ]
    .sort_values("rate_per_mile", ascending=False)
    .head(20)
)

print("\nValidation extreme loads:")
print(
    extreme_valid[
        ["load_id", "distance", "equipment", "weight",
         "market_index", "quote_signal", "posted_rate", "rate_per_mile"]
    ]
    .sort_values("rate_per_mile", ascending=False)
)
print("\n==============================")
print("HIGH RATE ROUTE ANALYSIS")
print("==============================")

# Create rate per mile
df["rate_per_mile"] = df["posted_rate"] / df["distance"]

# Routes with extreme rates
high_rate = df[df["posted_rate"] > 10000].copy()

print("\nNumber of high-rate loads:")
print(len(high_rate))

print("\nHigh-rate loads by route:")
route_analysis = (
    high_rate
    .groupby("route")
    .agg(
        count=("posted_rate", "size"),
        avg_rate=("posted_rate", "mean"),
        max_rate=("posted_rate", "max"),
        avg_rpm=("rate_per_mile", "mean"),
        max_rpm=("rate_per_mile", "max")
    )
    .sort_values("count", ascending=False)
)

print(route_analysis.head(30))
print("\n==============================")
print("EXTREME VALIDATION ROUTES")
print("==============================")

valid_extreme = valid_df[valid_df["posted_rate"] > 10000].copy()

print(
    valid_extreme[
        [
            "load_id",
            "pickup",
            "delivery",
            "route",
            "distance",
            "equipment",
            "posted_rate",
            "rate_per_mile"
        ]
    ]
    .sort_values("rate_per_mile", ascending=False)
)
# STEP 12

print("\n==============================")
print("HIGH RATE LOADS BY MONTH")
print("==============================")

df["date"] = pd.to_datetime(df["date"])
df["month_period"] = df["date"].dt.to_period("M")

monthly_high = (
    df.groupby("month_period")
      .agg(
          total_loads=("posted_rate", "size"),
          high_rate_loads=("posted_rate",
                           lambda x: (x > 10000).sum()),
          very_high_rate_loads=("posted_rate",
                                lambda x: (x > 15000).sum()),
          max_rate=("posted_rate", "max")
      )
)

monthly_high["high_rate_pct"] = (
    monthly_high["high_rate_loads"]
    / monthly_high["total_loads"] * 100
)

print(monthly_high)
# Save validation predictions
validation_output = valid_df.copy()

validation_output["predicted_rate"] = y_pred
validation_output["absolute_error"] = (
    validation_output["posted_rate"] -
    validation_output["predicted_rate"]
).abs()

validation_output.to_csv(
    "data/model_predictions.csv",
    index=False
)

print("\nValidation predictions saved!")