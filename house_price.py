import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from xgboost import XGBRegressor
import joblib


df = pd.read_csv("house.csv")

print("\nDATASET SHAPE")
print(df.shape)

print("\nFIRST 5 ROWS")
print(df.head())

print("\nMISSING VALUES")
print(df.isnull().sum())




if "date" in df.columns:
    df.drop("date", axis=1, inplace=True)


if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)


target = "price"

X = df.drop(target, axis=1)
y = df[target]


X = pd.get_dummies(X, drop_first=True)


print("\nCORRELATION WITH PRICE")

temp_df = pd.concat([X, y], axis=1)

print(
    temp_df.corr(numeric_only=True)["price"]
    .sort_values(ascending=False)
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
}



best_model = None
best_score = -999
best_name = ""

print("\nMODEL RESULTS")
print("-" * 40)

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    score = r2_score(y_test, pred)

    print(f"{name}: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = model
        best_name = name



print("\nBEST MODEL")
print(best_name)

print("BEST R2 SCORE")
print(round(best_score, 4))


pred = best_model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": pred
})

print("\nSAMPLE PREDICTIONS")
print(comparison.head(10))


if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    print("\nTOP 15 FEATURES")

    print(
        importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(15)
    )

joblib.dump(best_model, "house_price_model.pkl")