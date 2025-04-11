import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("../data/BLearn2.csv").fillna("不明")

# 特徴カラムとラベル
base_cols = ["target", "subject", "object", "intent", "tense"]
strength_cols = [col + "_strength" for col in base_cols]
extra_cols = ["emotion", "formality"]
feature_cols = base_cols + strength_cols + extra_cols
label_col = "correct_id"

# カテゴリ変数エンコード
encoders = {col: LabelEncoder() for col in base_cols + extra_cols}
for col in base_cols + extra_cols:
    df[col] = encoders[col].fit_transform(df[col].astype(str))

# 精度が足りずdouble
df[strength_cols] = df[strength_cols].astype(float)

# 最終特徴ベクトル
X_full = df[base_cols + extra_cols + strength_cols]
y = df[label_col].astype(str)

# ラベルエンコード
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# PCAによる意味ベクトル化
pca = PCA(n_components=12)#今の特徴量最大値
X_pca = pca.fit_transform(X_full)

# 保存（任意）
joblib.dump(pca, "tdc_pca.pkl")
joblib.dump(label_encoder, "tdc_label_encoder.pkl")
joblib.dump(encoders, "tdc_encoders.pkl")

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_encoded, test_size=0.2, random_state=42
)

# Tensor変換
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# モデル定義（シンプル2層）
class TDCResponseNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# モデル・損失関数・最適化
model = TDCResponseNet(input_dim=X_train.shape[1], output_dim=len(label_encoder.classes_)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
for epoch in range(8100):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 評価
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean()
    print(f"\nTest Accuracy: {acc:.2f}")

    # サンプル出力
    idx = 0
    pred_id = label_encoder.inverse_transform([preds[idx].cpu().item()])[0]
    true_id = label_encoder.inverse_transform([y_test[idx].cpu().item()])[0]
    print(f"予測ID: {pred_id} / 正解ID: {true_id}")

# モデル保存
torch.save(model.state_dict(), "tdc_response_model.pth")

# 評価（ベクトル距離付き）
model.eval()
results = []

# IDごとのPCA平均ベクトル作成
df["correct_id_encoded"] = y_encoded
df["pca_vector"] = list(X_pca)
id_to_vec = df.groupby("correct_id_encoded")["pca_vector"].apply(lambda x: np.mean(np.stack(x), axis=0))

# 類似度関数
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# テスト全件について評価
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean()
    print(f"\nTest Accuracy: {acc:.2f}")

    for i in range(len(y_test)):
        true_id = y_test[i].item()
        pred_id = preds[i].item()

        # PCAベクトルを取得（安全に）
        true_vec = id_to_vec.get(true_id)
        pred_vec = id_to_vec.get(pred_id)

        if true_vec is None or pred_vec is None:
            continue

        euclid_dist = np.linalg.norm(pred_vec - true_vec)
        cos_sim = cosine_similarity(pred_vec, true_vec)

        results.append({
            "index": i,
            "true_id": label_encoder.inverse_transform([true_id])[0],
            "pred_id": label_encoder.inverse_transform([pred_id])[0],
            "euclidean_distance": euclid_dist,
            "cosine_similarity": cos_sim
        })

# CSV出力
results_df = pd.DataFrame(results)
results_df.to_csv("tdc_similarity_report.csv", index=False, encoding="utf-8-sig")

print("✅ 意味的類似度レポートを保存しました：tdc_similarity_report.csv")
