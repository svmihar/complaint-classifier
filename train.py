from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.model_pytorch import lstm
import pandas as pd
import json

def model_sklearn(data_path='./data/label_processed.csv'): 
        df = pd.read_csv(data_path)
        X_train,y = df['text'].values, df['label'].values
        vectorizer = TfidfVectorizer()
        vectorizer.fit(df['text'].values)
        X = vectorizer.transform(X_train)
        clf = RandomForestClassifier(max_depth=100, random_state=42)
        clf.fit(X, y)
        yhat = [clf.predict(vectorizer.transform([a])) for a in X_train]
        return yhat, y


if __name__ == "__main__":
        yhat,y = model_sklearn()
        acc = accuracy_score(yhat, y)
        with open("metrics.json", 'w') as outfile:
                json.dump({ "accuracy": acc}, outfile)
