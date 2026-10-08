
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