
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def load_similarity_matrix(filepath):
    return pd.read_csv(filepath, index_col=0)

def evaluate_with_similarity(pred_ids, true_ids, similarity_matrix):
    total_score = 0.0
    for pred, true in zip(pred_ids, true_ids):
        if str(pred) in similarity_matrix.columns and str(true) in similarity_matrix.index:
            score = similarity_matrix.loc[str(true), str(pred)]
        else:
            score = 0.0
        total_score += score
    return total_score / len(pred_ids)

with open("blearn_log1.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss", "test_accuracy"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
similarity_matrix = load_similarity_matrix("response_similarity.csv")
df = pd.read_csv("BLearn1.csv")

base_cols = ["target", "subject", "object", "intent", "tense"]
strength_cols = [col + "_strength" for col in base_cols]
extra_cols = ["emotion", "formality"]
label_col = "correct_id"

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[()]", "", regex=True)
X_categorical = df[base_cols + extra_cols].fillna("不明").astype(str)
X_strength = df[strength_cols].fillna(0).astype(float)
y = df[label_col].astype(str)

encoders = {col: LabelEncoder() for col in base_cols + extra_cols}
for col in base_cols + extra_cols:
    X_categorical[col] = encoders[col].fit_transform(X_categorical[col])

X_final = pd.concat([X_categorical, X_strength], axis=1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_final.values, y_encoded, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

class StrengthNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

model = StrengthNet(input_dim=X_train.shape[1], output_dim=len(label_encoder.classes_)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(6000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        pred_labels = label_encoder.inverse_transform(preds.cpu().numpy())
        true_labels = label_encoder.inverse_transform(y_test.cpu().numpy())
        acc = evaluate_with_similarity(pred_labels, true_labels, similarity_matrix)

    with open("blearn_log1.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, loss.item(), acc])

model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    pred_labels = label_encoder.inverse_transform(preds.cpu().numpy())
    true_labels = label_encoder.inverse_transform(y_test.cpu().numpy())
    acc = evaluate_with_similarity(pred_labels, true_labels, similarity_matrix)
    print(f"\nTest Accuracy (Similarity-based): {acc:.4f}")

    sample_idx = 0
    input_features = X_test[sample_idx].cpu().numpy()
    print(f"\n入力ベクトル: {input_features}")
    print(f"予測ID: {pred_labels[sample_idx]} / 正解ID: {true_labels[sample_idx]}")
