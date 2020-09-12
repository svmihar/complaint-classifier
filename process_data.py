import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('./data/label.csv')
df.dropna(inplace=True)
df = df[['complaint','cleaned']]
df.columns = ['label','text']
X, y= train_test_split(df)
y_test, y_valid = train_test_split(y)

X.to_csv('./data/train.csv', index=False)
y_test.to_csv('./data/test.csv', index=False)
y_valid.to_csv('./data/valid.csv', index=False)


df['label'].apply(lambda x: 'complaint' if x>0 else 'not complaint')
df.to_csv('./data/label_processed.csv', index=False)
