import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# 1. Load
df = pd.read_csv("kc_house_data.csv")

# 2. Features
df["date"] = pd.to_datetime(df["date"])
df["sale_year"] = df["date"].dt.year
df["sale_month"] = df["date"].dt.month
df["house_age"] = df["sale_year"] - df["yr_built"]
df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
df = df[df["bedrooms"] < 15]
df = df.drop(columns=["id", "date"])

# 3. Split
X = df.drop(columns=["price"])
y = np.log1p(df["price"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train
model = GradientBoostingRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
print("Test R2:", round(r2_score(y_test, model.predict(X_test)), 3))

# 5. Save
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "columns.pkl")
print("Saved model.pkl and columns.pkl")