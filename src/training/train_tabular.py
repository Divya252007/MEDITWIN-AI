import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone

# Paths
DATA_PATH = "data/raw/uci_heart/heart.csv"
CHECKPOINT_PATH = "models/checkpoints/tabular_model.pth"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Load data
df = pd.read_csv(DATA_PATH)

# Remove identifier and dataset-source columns
df = df.drop(columns=["id", "dataset"], errors="ignore")

# Convert target to binary:
# 0 = no disease; 1–4 = disease present
df = df.dropna(subset=["num"]).copy()
df["num"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
df = df.dropna(subset=["num"])

X = df.drop(columns=["num"])
y = df["num"].astype(int)

# Identify column types
categorical_cols = X.select_dtypes(
    include=["object", "bool", "category"]
).columns.tolist()

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Impute missing values and scale numerical data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# Split before fitting preprocessing to avoid leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train_tensor = torch.tensor(
    X_train_processed, dtype=torch.float32, device=DEVICE
)
X_test_tensor = torch.tensor(
    X_test_processed, dtype=torch.float32, device=DEVICE
)
y_train_tensor = torch.tensor(
    y_train.to_numpy(), dtype=torch.float32, device=DEVICE
).view(-1, 1)

# Small tabular neural network
class TabularClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = TabularClassifier(X_train_tensor.shape[1]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
EPOCHS = 40

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    logits = model(X_train_tensor)
    loss = criterion(logits, y_train_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    probabilities = torch.sigmoid(model(X_test_tensor)).cpu().numpy().ravel()

predictions = (probabilities >= 0.5).astype(int)

print("\nTest accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save model and preprocessing pipeline
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

torch.save({
    "model_state_dict": model.cpu().state_dict(),
    "input_dim": X_train_processed.shape[1],
    "preprocessor": preprocessor,
    "target_classes": ["no disease", "disease present"],
    "feature_columns": X.columns.tolist(),
    "model_name": "TabularClassifier",
    "seed": SEED
}, CHECKPOINT_PATH)

print("\nTabular checkpoint saved:", CHECKPOINT_PATH)