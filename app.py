from fastapi import FastAPI
import uvicorn
from train import predict, load_xgboost_model

m,v = load_xgboost_model()
app = FastAPI()

@app.get('/')
def home():
    return {'foo': 'bar', 'message': 'predict, and serve'}

@app.get("/predict/{query}")
def pred(query: str):
    result = predict(m,v,query.lower())
    return {'label': result}

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8080)