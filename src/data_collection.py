import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import time

SEASON = "2023-24"
MAX_PLAYERS = None   
SLEEP_SECONDS = 0.5  


def fetch_active_players(max_players=None):
    all_players = players.get_players()            
    active_players = [p for p in all_players if p.get("is_active")]  

    if max_players is not None:
        active_players = active_players[:max_players]

    print(f"Found {len(active_players)} active players to download.")
    return active_players


def fetch_player_logs(season: str, max_players=None) -> pd.DataFrame:
    active_players = fetch_active_players(max_players)
    logs = []

    for idx, player in enumerate(active_players, start=1):
        pid = player["id"]
        name = player["full_name"]

        print(f"[{idx}/{len(active_players)}] Fetching logs for {name} ({pid})...")

        try:
            log = playergamelog.PlayerGameLog(player_id=pid, season=season)
            df = log.get_data_frames()[0]

            if not df.empty:
                df["PLAYER_NAME"] = name
                logs.append(df)

        except Exception as e:
            print(f"  -> Skipped {name} due to error: {e}")

        time.sleep(SLEEP_SECONDS)  

    if logs:
        return pd.concat(logs, ignore_index=True)

    return pd.DataFrame()


if __name__ == "__main__":
    df = fetch_player_logs(SEASON, max_players=MAX_PLAYERS)

    if df.empty:
        print("No data collected.")
    else:
        output_path = "raw_data/raw_player_stats_23_24.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} game logs to {output_path}")
