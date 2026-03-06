import pandas as pd
import numpy as np

def build_features(df):
    X = df[[
        'age',
        'sleep_duration',
        'rem_sleep_percentage',
        'deep_sleep_percentage',
        'light_sleep_percentage',
        'awakenings',
        'caffeine_consumption',
        'alcohol_consumption',
        'exercise_frequency',
        'smoking_status'
    ]].copy()

    X['smoking_status'] = X['smoking_status'].map({'Yes': 1, 'No': 0})

    y = pd.cut(df['sleep_efficiency'],
               bins=[0, 0.70, 0.85, 1.0],
               labels=[0, 1, 2]).astype(int)

    return X.values, y.values

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned.csv")
    X, y = build_features(df)
    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Labels distribution: {np.bincount(y)}")