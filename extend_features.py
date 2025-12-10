import pandas as pd
from pathlib import Path
import numpy as np

INPUT_FILE = Path("raw_data/clean_player_stats.csv")
OUTPUT_FILE = Path("raw_data/enhanced_player_stats.csv")


def extend_features(df: pd.DataFrame) -> pd.DataFrame:
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

 
    df["REST_DAYS"] = (
        df.groupby("PLAYER_NAME")["GAME_DATE"]
        .diff()
        .dt.days
        .fillna(3)  
    )

    df["HOME_GAME"] = df["MATCHUP"].apply(
        lambda x: 1 if "vs." in str(x) else 0
    )

    df["OPPONENT"] = df["MATCHUP"].astype(str).str.split().str[-1]

    opp_stats = (
        df.groupby("OPPONENT")["PTS"]
        .mean()
        .rename("OPP_DEF_PTS_ALLOWED")
    )
    df = df.merge(opp_stats, on="OPPONENT", how="left")

    df["PTS_ROLL10"] = df.groupby("PLAYER_NAME")["PTS"].transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )
    df["REB_ROLL10"] = df.groupby("PLAYER_NAME")["REB"].transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )
    df["AST_ROLL10"] = df.groupby("PLAYER_NAME")["AST"].transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    print("Loaded:", df.shape)
    df = extend_features(df)
    print("After extending features:", df.shape)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Extended feature dataset saved → {OUTPUT_FILE}")
