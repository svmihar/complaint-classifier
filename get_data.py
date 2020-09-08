# from pymongo import MongoClient
import pandas as pd
# import os


# mdb = MongoClient(os.environ["MONGODB_URI"])
# coll = mdb["koinwork"]
# raw_data = coll["label"]

df = pd.read_csv('./data/eda.csv')

df.to_csv("./data/label.csv")
