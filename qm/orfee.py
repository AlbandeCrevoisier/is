import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.plotting.register_matplotlib_converters()
sns.set()


def load_data():
    d = pd.read_csv("data.csv", index_col='index', parse_dates=['date'])
    d.dropna(inplace=True)
    d.drop('Unnamed: 0', axis=1, inplace=True)
    d.drop(d[d['age'] < 18].index, inplace=True)
    d.drop(d[d['exp'] < 0].index, inplace=True)
    d.drop(d[d['note'] > 100].index, inplace=True)
    d.sort_values('date', inplace=True)
    return d


def plots(d):
    print("Taux d'embauche : ", d['embauche'].mean())
    sns.lineplot(data=d[['date', 'embauche']].groupby('date').sum().cumsum())
    sns.jointplot('age', 'exp', kind='kde', data=d)
    sns.kdeplot(d['salaire'], shade=True)
    sns.scatterplot('salaire', 'note', 'embauche', data=d)
    sns.countplot('diplome', hue='specialite', data=d)
    sns.barplot('diplome', 'embauche', 'specialite', data=d)
    sns.barplot('sexe', 'note', 'embauche', data=d)


def pp(X):
    X['day'] = X['date'].transform(lambda x: x.day)
    X['month'] = X['date'].transform(lambda x: x.month)
    X['year'] = X['date'].transform(lambda x: x.year)
    X.drop('date', axis=1, inplace=True)
    cat = ['cheveux', 'sexe', 'diplome', 'specialite', 'dispo',
           'day', 'month', 'year']
    X = pd.get_dummies(X, columns=cat)
    X[['age', 'note']] /= 100
    return X
