import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("raw_data/enhanced_player_stats.csv")
df = pd.read_csv(DATA)

X = df[["PTS_ROLL5", "PTS_ROLL10", "PTS_PER_MIN", "REST_DAYS"]].values
y = df["PTS_NEXT"].values.reshape(-1,1)


X = (X - X.mean(axis=0)) / X.std(axis=0)
X = np.hstack([np.ones((X.shape[0],1)), X])

w = np.zeros((X.shape[1],1))
lr = 0.001
epochs = 5000

for i in range(epochs):
    predictions = X.dot(w)
    error = predictions - y
    gradient = (2/len(X)) * X.T.dot(error)
    w -= lr * gradient
    if i % 500 == 0:
        mse = np.mean(error**2)
        print(f"Epoch {i} | MSE: {mse:.4f}")

print("\nFinal weights:\n", w)
