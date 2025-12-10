import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

CLEAN_CSV = Path("raw_data/clean_player_stats.csv")
MODEL_PATH = Path("models/logreg_high_scorer_next.pkl")


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading cleaned data from {path}...")
    df = pd.read_csv(path)
    print("Data shape:", df.shape)
    return df


def build_feature_target(df: pd.DataFrame):
    df = df.dropna(subset=["HIGH_SCORER_NEXT"])

    df["HIGH_SCORER_NEXT"] = df["HIGH_SCORER_NEXT"].astype(int)

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
    y = df["HIGH_SCORER_NEXT"].values

    return X, y, features


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    print("Training Logistic Regression model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Evaluation (Classification) ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)

    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return pipeline, (acc, prec, rec, f1, cm)


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
