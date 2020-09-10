from flair.models import TextClassifier
from flair.data import Sentence
from pathlib import Path


def download_model():
    pass


def load_model(model_path='./models/'):
    mp = Path(model_path)
    if not mp.is_dir():
        mp.mkdir(exist_ok=True)
    if not Path(mp/'best-lm.pt').is_file()
    download_model()

    return TextClassifier.load(mp/'best-lm.pt')

def classify(model, query):
    sentence = Sentence(query)
    model.predict(sentence)
    return sentence.labels