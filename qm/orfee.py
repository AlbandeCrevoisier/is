import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


cat = ['cheveux', 'sexe', 'diplome', 'specialite', 'dispo']


def load_data():
	d = pd.read_csv("data.csv", index_col='date', parse_dates=['date']).dropna()
	d.drop(['Unnamed: 0'], axis=1, inplace=True)
	d.drop(['index'], axis=1, inplace=True)
	d.drop(d[d['age'] < 0].index, inplace=True)
	d.drop(d[d['exp'] < 0].index, inplace=True)
	d.drop(d[d['note'] > 100].index, inplace=True)
	return d


def pp(d):
	d[['age', 'note']] /= 100
	d = pd.get_dummies(d, columns=cat)
