import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


data = pd.read_csv("data.csv", index_col='date', parse_dates=['date']).dropna()
data.drop(['Unnamed: 0'], axis=1, inplace=True)
data.drop(['index'], axis=1, inplace=True)
data.drop(data[data['age'] < 0].index, inplace=True)
data.drop(data[data['exp'] < 0].index, inplace=True)
data.drop(data[data['note'] > 100].index, inplace=True)


data[['age', 'note']] /= 100
cat = ['cheveux', 'sexe', 'diplome', 'specialite', 'dispo']
data = pd.get_dummies(data, columns=cat)
