
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine

def safe_cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return 1 - cosine(vec1, vec2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("BLearn2.csv")

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

num_classes = len(label_encoder.classes_)

# リセット方式のトレーニングループ
for total_epochs in range(1000, 12001, 1000):
    # モデル初期化
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

    model = StrengthNet(input_dim=X_train.shape[1], output_dim=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(total_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 評価とログ保存
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1).cpu().numpy()
        true_ids = y_test.cpu().numpy()
        acc = (preds == true_ids).mean()

        filename = f"blearn2_reset_epoch{total_epochs}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "accuracy", "pred", "true", "euclidean", "cosine_similarity"])
            for pred, true in zip(preds, true_ids):
                pred_vec = np.eye(num_classes)[pred]
                true_vec = np.eye(num_classes)[true]
                euclidean = np.linalg.norm(pred_vec - true_vec)
                cosine_sim = safe_cosine_similarity(pred_vec, true_vec)
                writer.writerow([total_epochs, loss.item(), acc, pred, true, euclidean, cosine_sim])
