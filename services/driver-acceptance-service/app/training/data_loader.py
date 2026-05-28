import pandas as pd
from pathlib import Path


def load_data():

    base_dir = Path(__file__).resolve().parents[4]

    data_path = base_dir / "data" / "raw" / "rides.csv"

    df = pd.read_csv(data_path)

    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())