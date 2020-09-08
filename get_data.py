from pymongo import MongoClient
import pandas as pd
import os


mdb = MongoClient(os.environ["MONGODB_URI"])
coll = mdb["koinworks"]
raw_data = coll["label"]

df = pd.DataFrame(list(raw_data.find({}, {'_id':0})))

df.to_csv("./data/label.csv")
