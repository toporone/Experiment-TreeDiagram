import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

model = joblib.load("tree_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
encoders = joblib.load("encoders.pkl")

response_df = pd.read_csv("responses.csv", dtype={"id": str})
response_dict = dict(zip(response_df["id"], response_df["response"]))


def process_input(text):
    text = text.strip()

    if len(text) < 3:
        return "もう少し詳しく教えてくれない？"

    if "怒" in text:
        return {"target": "怒った", "subject": "彼", "object": "不明", "intent": "陳述", "tense": "過去"}
    elif "泣" in text:
        if "け" in text or "て" in text:
            return {"target": "泣いた", "subject": "不明", "object": "不明", "intent": "命令", "tense": "現在"}
        else:
            return {"target": "泣いた", "subject": "彼", "object": "不明", "intent": "陳述", "tense": "過去"}
    elif "見" in text:
        return {"target": "見た", "subject": "彼", "object": "空", "intent": "陳述", "tense": "過去"}
    elif "取" in text or "撮" in text:
        return {"target": "取った", "subject": "彼", "object": "リンゴ", "intent": "陳述", "tense": "過去"}
    else:
        return {"target": "怒った", "subject": "不明", "object": "不明", "intent": "命令", "tense": "現在"}

# 対話ループ
print("TreeDiagram Chat 起動！ 話しかけてみてね（終了するには 'exit'）")
while True:
    user_input = input("あなた: ")
    if user_input.lower() == "exit":
        print("AI: またね！🌳")
        break

    parsed = process_input(user_input)

    X_input = pd.DataFrame([parsed])
    for col in X_input.columns:
        if col in encoders:
            X_input[col] = encoders[col].transform(X_input[col])

    
    pred_id = label_encoder.inverse_transform(model.predict(X_input))[0]
    response = response_dict.get(pred_id, "うまく理解できませんでした…")

    print(f"AI: {response}")
