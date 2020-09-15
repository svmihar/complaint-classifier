import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


def load_data(data_path="./data/label_processed.csv"):
    df = pd.read_csv(data_path)
    X_train, y = df["text"].values, df["label"].values
    return X_train, y


def model_sklearn(df):
    X_train, y = load_data()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df["text"].values)
    X = vectorizer.transform(X_train)
    clf = RandomForestClassifier(max_depth=100, random_state=42)
    clf.fit(X, y)
    yhat = [clf.predict(vectorizer.transform([a])) for a in X_train]
    return yhat, y


def model_xgboost(trial):
    import optuna

    X_train, y = load_data()
    # vectorizer = TfidfVectorizer()
    # vectorizer.fit(X_train)
    # joblib.dump(vectorizer, "models/tfidf.pkl")
    vectorizer = joblib.load('./models/tfidf.pkl')
    X = vectorizer.transform(X_train)
    x_train, x_test, y_train, y_test = train_test_split(
        X.toarray(), y, test_size=0.2, random_state=1337
    )
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    param = {
        "silent": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "nthread": 4,
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, "validation-auc"
    )
    bst = xgb.train(
        param, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback]
    )
    pred = np.rint(bst.predict(dtest))
    accuracy = accuracy_score(y_test, pred)
    bst.save_model(f"models/xgb_{trial.number}.model")

    return accuracy


def load_xgboost_model(model_path):
    model = xgb.Booster({"nthread": 4})
    model.load_model(model_path)
    vectorizer = joblib.load("./models/tfidf.pkl")

    return model, vectorizer


def predict(m, v, query):
    q = v.transform([query])
    q_d = xgb.DMatrix(q)
    p = m.predict(q_d)
    return 1 if p[0]>.5 else 0


def optimizer():
    import optuna
    s = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    n_trial = 500
    s.optimize(model_xgboost, n_trials=n_trial)
    n_best = s.best_trial.number
    all_models = ["models/" + a for a in os.listdir("models") if 'xgb' in a]
    all_models.remove(f"models/xgb_{n_best}.model")
    for n_model in all_models:
        os.system(f"rm {n_model}")


if __name__ == "__main__":
    # optimizer()
    m, v = load_xgboost_model(model_path="models/xgb_58.model")
    x, y = load_data()
    y_pred = [predict(m, v, a) for a in x]
    acc = accuracy_score(y_pred, y)

    with open("metrics.json", "w") as f:
        json.dumps({"accuracy": acc}, f)
