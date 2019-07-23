import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
pd.plotting.register_matplotlib_converters()
sns.set()


def load_data():
    """Read data.csv and return a pandas DataFrame."""
    d = pd.read_csv("data.csv", index_col='index', parse_dates=['date'])
    d.dropna(inplace=True)
    d.drop('Unnamed: 0', axis=1, inplace=True)
    d.drop(d[d['age'] < 18].index, inplace=True)
    d.drop(d[d['exp'] < 0].index, inplace=True)
    d.drop(d[d['note'] > 100].index, inplace=True)
    d.sort_values('date', inplace=True)
    return d


def plots(d):
    """Relevant Seaborn plots."""
    print("Taux d'embauche :", d['embauche'].mean())
    sns.lineplot(data=d[['date', 'embauche']].groupby('date').sum().cumsum())
    sns.jointplot('age', 'exp', kind='kde', data=d)
    sns.kdeplot(d['salaire'], shade=True)
    sns.scatterplot('salaire', 'note', 'embauche', data=d)
    sns.countplot('diplome', hue='specialite', data=d)
    sns.barplot('diplome', 'embauche', 'specialite', data=d)
    sns.barplot('sexe', 'note', 'embauche', data=d)


def pp(d):
    """Preprocess the DataFrame d and return X, y."""
    y = d.pop('embauche')
    X = d
    X['day'] = X['date'].transform(lambda x: x.day)
    X['month'] = X['date'].transform(lambda x: x.month)
    X['year'] = X['date'].transform(lambda x: x.year)
    X.drop('date', axis=1, inplace=True)
    cat = ['cheveux', 'sexe', 'diplome', 'specialite', 'dispo',
           'day', 'month', 'year']
    X = pd.get_dummies(X, columns=cat)
    X[['age', 'note']] /= 100
    X['salaire'] = (X['salaire'] - X['salaire'].mean()) / X['salaire'].std()
    X['exp'] = (X['exp'] - X['exp'].mean()) / X['exp'].std()
    return X, y


if __name__ == "__main__":
    data = load_data()
    X, y = pp(data)
    erd = ExtraTreesClassifier()
    score = cross_val_score(erd, X, y)
    print(score.mean())
