from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "uci_heart" / "heart.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tabular"

SEED = 42


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id", "dataset"], errors="ignore")
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    df = df.dropna(subset=["num"]).copy()
    df["num"] = (df["num"] > 0).astype(int)

    X = df.drop(columns=["num"])
    y = df["num"]

    categorical_cols = X.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    # Fit only on training data to avoid data leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train_processed).to_csv(
        OUTPUT_DIR / "X_train.csv", index=False
    )
    pd.DataFrame(X_test_processed).to_csv(
        OUTPUT_DIR / "X_test.csv", index=False
    )
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")

    print("Tabular preprocessing completed.")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Processed feature count:", X_train_processed.shape[1])
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()