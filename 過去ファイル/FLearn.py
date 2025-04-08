import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# GPU対応
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSV読み込み
df = pd.read_csv("conte.csv")

# 特徴量とラベル
feature_cols = ["target", "subject", "object", "intent", "tense"]
label_col = "correct_id"

# 欠損埋め
X = df[feature_cols].fillna("不明")
y = df[label_col].astype(str)

# エンコード（カテゴリ → 数値）
encoders = {col: LabelEncoder() for col in feature_cols}
for col in feature_cols:
    X[col] = encoders[col].fit_transform(X[col].astype(str))

# 型を明示的に float32 に変換（← 重要！）
X = X.astype("float32")

# ラベルもエンコード
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 学習・テストに分割
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

# Tensorに変換してGPUに転送
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# モデル定義
class ResponseNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResponseNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# モデル・ロス・最適化
model = ResponseNet(X_train.shape[1], len(label_encoder.classes_)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
for epoch in range(1800):
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
    preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean()
    print(f"\nTest Accuracy: {acc:.2f}")

    # サンプル確認
    sample_idx = 0
    input_features = X_test[sample_idx].cpu().numpy()
    pred_label = label_encoder.inverse_transform([preds[sample_idx].cpu().item()])[0]
    true_label = label_encoder.inverse_transform([y_test[sample_idx].cpu().item()])[0]

    print(f"\n入力ベクトル: {input_features}")
    print(f"予測ID: {pred_label}, 正解ID: {true_label}")

# 予測 vs 実際のラベルを全部見る
for i in range(len(y_test)):
    true_label = label_encoder.inverse_transform([y_test[i].cpu().item()])[0]
    pred_label = label_encoder.inverse_transform([preds[i].cpu().item()])[0]
    print(f"{i+1}: 正解: {true_label} / 予測: {pred_label}")
