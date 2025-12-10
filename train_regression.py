import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

CLEAN_CSV = Path("raw_data/clean_player_stats.csv")
MODEL_PATH = Path("models/linear_regression_pts_next.pkl")


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading cleaned data from {path}...")
    df = pd.read_csv(path)
    print("Data shape:", df.shape)
    return df


def build_feature_target(df: pd.DataFrame):
    df = df.dropna(subset=["PTS_NEXT"])

    candidate_features = [
        "PTS_ROLL5",
        "REB_ROLL5",
        "AST_ROLL5",
        "MIN_ROLL5",
        "PTS_PER_MIN",
        "MIN",
        "PTS",   
        "REB",
        "AST",
    ]

    features = [f for f in candidate_features if f in df.columns]
    print("Using features:", features)

    X = df[features].values
    y = df["PTS_NEXT"].values

    return X, y, features


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]
    )

    print("Training Linear Regression model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Evaluation ---")
    print(f"MSE : {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    return pipeline, (mse, rmse, r2)


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"\nModel saved to {path}")


if __name__ == "__main__":
    df = load_data(CLEAN_CSV)
    X, y, features = build_feature_target(df)
    model, metrics = train_and_evaluate(X, y)
    save_model(model, MODEL_PATH)

    print("\nDone.")
