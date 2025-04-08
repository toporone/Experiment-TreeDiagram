import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv("context.csv", dtype=str).fillna("不明")

feature_cols = ["target", "subject", "object", "intent", "tense"]
encoders = {col: LabelEncoder() for col in feature_cols}
X = pd.DataFrame()
for col in feature_cols:
    X[col] = encoders[col].fit_transform(df[col])

pca = PCA(n_components=3)
Y = pca.fit_transform(X)

# データ分割
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.2, random_state=42)

class MeaningDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

train_loader = torch.utils.data.DataLoader(MeaningDataset(X_train, Y_train), batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(MeaningDataset(X_test, Y_test), batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StreamNet(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)


model = StreamNet(in_dim=X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2000):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/100 - Loss: {total_loss:.4f}")

model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        all_preds.append(pred)
        all_true.append(yb.numpy())

mse = np.mean((np.vstack(all_preds) - np.vstack(all_true))**2)
print(f"\nTest MSE: {mse:.4f}")
