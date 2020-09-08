from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json

df = pd.read_csv('./data/label_processed.csv')
df.dropna(inplace=True)
X_train,y = df['text'].values, df['label'].values
vectorizer = TfidfVectorizer()
vectorizer.fit(df['text'].values)

X = vectorizer.transform(X_train)

clf = RandomForestClassifier(max_depth=10, random_state=42)
clf.fit(X, y)
yhat = [clf.predict(vectorizer.transform([a])) for a in X_train]

acc = accuracy_score(yhat, y)
with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc}, outfile)
