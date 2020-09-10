from flair.models import TextClassifier
from flair.data import Sentence
from pathlib import Path
from gdown import download


def download_model():
    download('https://drive.google.com/uc?id=1oiNAxhhtRV-NRMSeSj-bIXHPhNpCOjbR&export=download', output='./models/best-lm.pt')
    return True


def load_model(model_path='./models/'):
    mp = Path(model_path)
    if not mp.is_dir():
        mp.mkdir(exist_ok=True)
    if not Path(mp/'best-lm.pt').is_file():
        download_model()

    return TextClassifier.load(mp/'best-lm.pt')

def classify(model, query):
    sentence = Sentence(query)
    model.predict(sentence)
    return sentence.labels