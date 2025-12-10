import pandas as pd
from pathlib import Path
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATA = Path("raw_data/enhanced_player_stats.csv")
df = pd.read_csv(DATA)

X = df[["PTS_ROLL10", "PTS_ROLL5", "REB_ROLL5", "AST_ROLL5",
        "PTS_PER_MIN", "REST_DAYS", "HOME_GAME", "OPP_DEF_PTS_ALLOWED"]]
y = df["PTS_NEXT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for model in [Lasso(alpha=0.05), Ridge(alpha=0.5)]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(type(model).__name__, "MSE →", mean_squared_error(y_test, pred))
