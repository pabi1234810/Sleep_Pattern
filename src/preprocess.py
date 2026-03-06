import pandas as pd
import numpy as np
import os

def load_and_clean():
    df = pd.read_csv("data/raw/Sleep_Efficiency.csv")
    
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())

    # Drop rows with missing values
    df = df.dropna()

    # Rename for convenience
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned.csv", index=False)
    print(f"\n✅ Cleaned data saved — {len(df)} rows remaining")
    return df

if __name__ == "__main__":
    df = load_and_clean()
    print(df.head())