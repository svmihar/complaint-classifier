from fastapi import FastAPI
import uvicorn
from model import classify, load_model

model = load_model()
app = FastAPI()

@app.get('/')
def home():
    return {'foo': 'bar', 'message': 'predict, and serve'}

@app.get("/predict/{query}")
def pred(query: str):
    result = classify(model,query)
    return {'label': result}

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8080)