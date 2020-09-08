import pandas as pd

df = pd.read_csv('./data/label.csv')
df.dropna(inplace=True)
df = df[['cleaned', 'complaint']]
df.columns = ['text', 'label']
df['label'].apply(lambda x: 'complaint' if x>0 else 'not complaint')

df.to_csv('./data/label_processed.csv')
