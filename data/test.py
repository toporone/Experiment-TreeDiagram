import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("ALearn1.csv")
print(df.columns.tolist())

# dictionary.csv（応答IDと文）の読み込み
df = pd.read_csv("dictionary.csv")  # ID, token（または text）
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["token"])
similarity_matrix = cosine_similarity(tfidf_matrix)
