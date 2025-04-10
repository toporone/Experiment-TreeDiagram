import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_with_numeric_distance(pred_ids, true_ids, max_id):
    total_score = 0.0
    for pred, true in zip(pred_ids, true_ids):
        dist = abs(int(pred) - int(true))
        score = 1 - (dist / max_id)  # 距離が遠いほど低スコア
        score = max(score, 0.0)
        total_score += score
    return total_score / len(pred_ids)


# GPU対応
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSV読み込み
df = pd.read_csv("BLearn2.csv")

# 特徴量とラベル（+ 強度 + 新特徴）
base_cols = ["target", "subject", "object", "intent", "tense"]
strength_cols = [col + "_strength" for col in base_cols]
extra_cols = ["emotion", "formality"]
feature_cols = base_cols + strength_cols + extra_cols
label_col = "correct_id"

# 欠損埋めと文字列化
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[()]", "", regex=True)
X_categorical = df[base_cols + extra_cols].fillna("不明").astype(str)
X_strength = df[strength_cols].fillna(0).astype(float)
y = df[label_col].astype(str)

# エンコード（カテゴリ → 数値）
encoders = {col: LabelEncoder() for col in base_cols + extra_cols}
for col in base_cols + extra_cols:
    X_categorical[col] = encoders[col].fit_transform(X_categorical[col])

# 最終入力 = エンコード値 + 強度
X_final = pd.concat([X_categorical, X_strength], axis=1)

# ラベルエンコード
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 学習・テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X_final.values, y_encoded, test_size=0.2, random_state=42
)

# Tensorに変換
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# モデル定義
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

# 学習ループ
for epoch in range(50000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    # 評価
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1).cpu().numpy()
        true_ids = y_test.cpu().numpy()
        acc = (preds == true_ids).mean()

    # CSV書き込み（appendモードだけ！）
    with open("blearn2_50klog.csv", "a", newline="") as f:
        writer = csv.writer(f)
        for pred, true in zip(preds, true_ids):
            writer.writerow([epoch, loss.item(), acc, pred, true])




# 評価
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean()
    print(f"\nTest Accuracy: {acc:.2f}")

    # 可視化用：1件だけ表示
    sample_idx = 0
    input_features = X_test[sample_idx].cpu().numpy()
    pred_label = label_encoder.inverse_transform([preds[sample_idx].cpu().item()])[0]
    true_label = label_encoder.inverse_transform([y_test[sample_idx].cpu().item()])[0]
    print(f"\n入力ベクトル: {input_features}")
    print(f"予測ID: {pred_label} / 正解ID: {true_label}")
