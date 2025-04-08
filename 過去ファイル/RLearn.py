import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# GPU対応
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSV読み込みと前処理
df = pd.read_csv("context.csv").fillna("不明")

# 特徴量とラベル
feature_cols = ["target", "subject", "object", "intent", "tense"]
label_col = "correct_id"

# カテゴリ変数を数値化
encoders = {col: LabelEncoder() for col in feature_cols}
for col in feature_cols:
    df[col] = encoders[col].fit_transform(df[col])

label_encoder = LabelEncoder()
df[label_col] = label_encoder.fit_transform(df[label_col].astype(str))

# 入力と出力
X = df[feature_cols].values
y = df[label_col].values

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tensor化
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# モデル定義
class ResponseNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = ResponseNet(X_train.shape[1], len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
for epoch in range(3000):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train)
    loss = criterion(preds, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 精度評価
model.eval()
with torch.no_grad():
    test_preds = model(X_test).argmax(dim=1)
    acc = (test_preds == y_test).float().mean()
    print(f"\nTest Accuracy: {acc:.2f}")

    # サンプルの可視化
    sample_idx = 0
    input_vector = X_test[sample_idx]
    pred_label = label_encoder.inverse_transform([test_preds[sample_idx].item()])[0]
    true_label = label_encoder.inverse_transform([y_test[sample_idx].item()])[0]

    print(f"\n入力ベクトル: {input_vector.cpu().numpy()}")
    print(f"予測ID: {pred_label} / 正解ID: {true_label}")

torch.save(model.state_dict(), "response_model.pth")
