import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
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


def feature_importance(X, y):
    """Get the feature importance."""
    ert = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
    ert.fit(X, y)
    fi = ert.feature_importances_
    idx = np.argsort(fi)[::-1]
    plt.xticks(rotation=30)
    sns.barplot(X.columns.values[idx[:10]], fi[idx[:10]])


def compare_clf(X, y):
    """Compare the standard methods."""
    nb = GaussianNB()
    nb_score = cross_val_score(nb, X, y, cv=10, verbose=1, n_jobs=-1)
    print('Naive Bayes', nb_score.mean(), nb_score.std())
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial',
        n_jobs=-1)
    lr_score = cross_val_score(lr, X, y, cv=10, verbose=1, n_jobs=-1)
    print('Logistic Regression', lr_score.mean(), lr_score.std())
    ert = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
    ert_score = cross_val_score(ert, X, y, cv=10, verbose=1, n_jobs=-1)
    print('Random Forest ', ert_score.mean(), ert_score.std())
    svm = SVC(gamma='scale')
    svm_score = cross_val_score(svm, X, y, cv=10, verbose=1, n_jobs=-1)
    print('SVM', svm_score.mean(), svm_score.std())
    # Multi-Layer Perceptron


if __name__ == "__main__":
    X, y = pp(load_data())
    compare_clf(X, y)
