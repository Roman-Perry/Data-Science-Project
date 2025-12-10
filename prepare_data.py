import pandas as pd
import numpy as np
from pathlib import Path

RAW_CSV = Path("raw_data/raw_player_stats_23_24.csv")  
CLEAN_CSV = Path("raw_data/clean_player_stats.csv")


def load_data(csv_path: Path) -> pd.DataFrame:
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print("Raw shape:", df.shape)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "PLAYER_NAME", "Game_ID", "GAME_DATE", "MATCHUP", "WL",
        "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT"
    ]
    df = df[keep_cols].copy()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")

    num_cols = [
        "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["MIN", "PTS"])

    df = df[df["MIN"] > 0]

    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"])

    print("After basic cleaning:", df.shape)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df["PTS_ROLL5"] = (
        df.groupby("PLAYER_NAME")["PTS"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    df["REB_ROLL5"] = (
        df.groupby("PLAYER_NAME")["REB"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    df["AST_ROLL5"] = (
        df.groupby("PLAYER_NAME")["AST"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    df["MIN_ROLL5"] = (
        df.groupby("PLAYER_NAME")["MIN"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )

    df["PTS_PER_MIN"] = df["PTS"] / df["MIN"].replace(0, np.nan)

    df["PTS_NEXT"] = (
        df.groupby("PLAYER_NAME")["PTS"]
        .shift(-1)
    )

    df = df.dropna(subset=["PTS_NEXT"])

    df["HIGH_SCORER_NEXT"] = (df["PTS_NEXT"] >= 20).astype(int)

    print("After feature engineering:", df.shape)
    return df


def save_data(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path}")


if __name__ == "__main__":
    df_raw = load_data(RAW_CSV)
    df_clean = basic_clean(df_raw)
    df_features = add_features(df_clean)
    print(df_features.head())
    print(df_features[["PTS", "PTS_ROLL5", "PTS_NEXT", "HIGH_SCORER_NEXT"]].head())

    save_data(df_features, CLEAN_CSV)
