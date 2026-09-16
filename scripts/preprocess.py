"""
preprocess.py
-------------
Loads the raw CV removal dataset, performs the 70/30 train/test split
(fixed seed for reproducibility, per thesis section 3.1.3), and
standardises the 5 input features using scikit-learn's StandardScaler
fitted ONLY on the training data (to avoid data leakage into the test
set, exactly as described in the methodology).

Outputs (data/processed/):
    X_train.csv, X_test.csv   - scaled feature matrices
    y_train.csv, y_test.csv   - target vectors (Removal_Efficiency_percent)
    scaler.joblib             - fitted StandardScaler (needed to scale
                                 any new/real data the same way later)
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RANDOM_SEED = 42
TEST_SIZE = 0.30  # 30% test (81 of 270), 70% train (189 of 270)

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "cv_removal_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

FEATURE_COLS = [
    "Initial_Concentration_mgL",
    "Adsorbent_Dosage_gL",
    "pH",
    "Contact_Time_min",
    "Temperature_C",
]
TARGET_COL = "Removal_Efficiency_percent"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_PATH}. Run scripts/generate_data.py "
            f"first, or place the real dataset CSV at this path with columns: "
            f"{FEATURE_COLS + [TARGET_COL]}"
        )

    df = pd.read_csv(RAW_PATH)
    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {missing}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Fit scaler on training data ONLY, then transform both sets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS, index=X_test.index)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_scaled.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    print(f"Train set: {X_train_scaled.shape[0]} rows (70%)")
    print(f"Test set:  {X_test_scaled.shape[0]} rows (30%)")
    print(f"Saved scaled features/targets to {PROCESSED_DIR}")
    print(f"Saved fitted scaler to {MODELS_DIR / 'scaler.joblib'}")


if __name__ == "__main__":
    main()