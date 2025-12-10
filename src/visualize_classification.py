import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

CLEAN_CSV = Path("raw_data/clean_player_stats.csv")
MODEL_PATH = Path("models/logreg_high_scorer_next.pkl")
SAVE_DIR = Path("visuals/classification")
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

    df = df.dropna(subset=["HIGH_SCORER_NEXT"])
    df["HIGH_SCORER_NEXT"] = df["HIGH_SCORER_NEXT"].astype(int)

    X = df[features].values
    y = df["HIGH_SCORER_NEXT"].values
    return X, y, features


def main():
    print("Loading data and model...")
    df = load_data(CLEAN_CSV)
    model = joblib.load(MODEL_PATH)

    X, y, features = build_feature_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Low", "Pred High"],
                yticklabels=["Actual Low", "Actual High"])
    plt.title("Confusion Matrix (Logistic Regression)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    save_path = SAVE_DIR / "confusion_matrix.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(y_prob, bins=20, alpha=0.7)
    plt.xlabel("Predicted Probability of High Scoring Game")
    plt.ylabel("Frequency")
    plt.title("Probability Distribution (Logistic Regression)")
    plt.grid(True)

    save_path = SAVE_DIR / "prob_distribution.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
