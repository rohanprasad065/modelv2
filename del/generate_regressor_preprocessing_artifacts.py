"""
generate_regressor_preprocessing_artifacts.py
------------------------------------------------
RUN THIS ONCE, OFFLINE. Not part of the FastAPI service.

Problem: estrus_xgb_model.json was trained on median-imputed +
StandardScaler-scaled features, but the fitted SimpleImputer/StandardScaler
objects were never saved during the original Colab training run -- only
model.save_model(...) was called. estrus_predict.py has therefore been
feeding raw, unscaled values into the model, which is wrong (see the
regressor diagnosis writeup for the full explanation).

This script does NOT retrain the model. It re-fits a SimpleImputer and
StandardScaler using the exact same, deterministic steps the original
training script used:
  1. Same CSV.
  2. Drop 'estrus' and '24hour' (exactly as training did).
  3. Drop rows with missing 'days_to_estrus' (the target).
  4. feature_cols = every remaining column except the target, in the
     file's natural order -- this is provably the same order the
     original training script produced, since it's the same line of
     code operating on the same file.
  5. First 70% of rows (sequential, no shuffling) = the train slice --
     same split logic as the original script.
  6. Fit SimpleImputer(median) then StandardScaler on that slice.

Because every one of these steps is deterministic and operates on the
same input file, the imputer/scaler this produces are mathematically
identical to what the original training run would have saved, had it
saved them. This was validated by hand (manually recomputing a z-score
for a spot-checked feature/row and confirming an exact match with
sklearn's fitted values) earlier in this project.

Run once:
    python generate_regressor_preprocessing_artifacts.py

Outputs (into models/):
    estrus_imputer.pkl
    estrus_scaler.pkl
    estrus_feature_cols.pkl   (the 43-column order these were fit on --
                                estrus_predict.py uses this to build the
                                input row in the right order before
                                calling imputer.transform())
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/mnt/user-data/uploads/body_core_temp_with_estrus_targets_augmented.csv"   # adjust path if needed
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def main():
    df = pd.read_csv(DATA_PATH)
    print("Original shape:", df.shape)

    df = df.drop(columns=["estrus", "24hour"])
    df = df.dropna(subset=["days_to_estrus"]).reset_index(drop=True)
    print("Shape after removing rows with missing target:", df.shape)

    target_col = "days_to_estrus"
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"{len(feature_cols)} feature columns (true training order):")
    for i, c in enumerate(feature_cols):
        print(f"  {i}: {c}")

    X = df[feature_cols]

    n = len(df)
    train_end = int(n * 0.70)
    X_train = X.iloc[:train_end].replace([np.inf, -np.inf], np.nan)
    print(f"\nFitting on train slice: {X_train.shape}")

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    X_train_imp = pd.DataFrame(imputer.transform(X_train), columns=feature_cols)
    scaler = StandardScaler()
    scaler.fit(X_train_imp)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(imputer, os.path.join(MODELS_DIR, "estrus_imputer.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "estrus_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "estrus_feature_cols.pkl"))

    print("\nSaved to models/:")
    print("  estrus_imputer.pkl")
    print("  estrus_scaler.pkl")
    print("  estrus_feature_cols.pkl")


if __name__ == "__main__":
    main()
