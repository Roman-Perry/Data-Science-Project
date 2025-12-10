import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CLEAN_CSV = Path("raw_data/clean_player_stats.csv")
MODEL_PATH = Path("models/linear_regression_pts_next.pkl")
SAVE_DIR = Path("visuals/regression")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path):
    return pd.read_csv(path)


def build_feature_target(df: pd.DataFrame):
    features = [
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
    features = [f for f in features if f in df.columns]

    X = df[features].values
    y = df["PTS_NEXT"].values
    return X, y, features


def main():
    print("Loading data and model...")
    df = load_data(CLEAN_CSV)
    model = joblib.load(MODEL_PATH)

    X, y, features = build_feature_target(df)

    y_pred = model.predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.4)
    plt.xlabel("Actual Points Next Game")
    plt.ylabel("Predicted Points Next Game")
    plt.title("Predicted vs Actual (Linear Regression)")
    plt.grid(True)

    save_path = SAVE_DIR / "predicted_vs_actual.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

    residuals = y_pred - y

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Points")
    plt.ylabel("Residuals")
    plt.title("Residual Plot (Linear Regression)")
    plt.grid(True)

    save_path = SAVE_DIR / "residuals.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
