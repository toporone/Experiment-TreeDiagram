import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

df = pd.read_csv("context.csv", dtype={"correct_id": str})

feature_cols = ["target", "subject", "object", "intent", "tense"]
label_col = "correct_id"

X = df[feature_cols].fillna("不明").copy()
y = df[label_col].astype(str)

encoders = {col: LabelEncoder() for col in feature_cols}
for col in feature_cols:
    X[col] = encoders[col].fit_transform(X[col])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    max_depth=4,
    n_estimators=100,
    learning_rate=0.1,
    verbosity=0
)

model.fit(X, y_encoded)

y_pred = model.predict(X)
acc = accuracy_score(y_encoded, y_pred)

print(f"Train Accuracy: {acc:.2f}")
print("↓ 予測結果（入力 → 正解ID / 予測ID）")
for i in range(len(X)):
    input_row = df.iloc[i][feature_cols].to_dict()
    correct = df.iloc[i][label_col]
    predicted = label_encoder.inverse_transform([y_pred[i]])[0]
    print(f"{input_row} → 正解: {correct}, 予測: {predicted}")

print("\n★ 特徴量の重要度")
for i, col in enumerate(feature_cols):
    print(f"{col:10s}: {model.feature_importances_[i]:.4f}")

joblib.dump(model, "tree_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(encoders, "encoders.pkl")