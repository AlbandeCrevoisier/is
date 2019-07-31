from importlib import reload

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import learning_curve, RandomizedSearchCV

from scipy.stats import randint as sp_randint
from scipy.stats import uniform

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns

pd.plotting.register_matplotlib_converters()
sns.set(style='white')


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


def pp(data, test_size=0.25, inplace=True):
    """Preprocess data and return X_train, X_test,  y_train, y_test."""
    data.drop('date', axis=1, inplace=inplace)
    if inplace:
        y = data.pop('embauche')
        X = data
    else:
        y = data['embauche']
        X = data.drop('embauche')
    cat = ['cheveux', 'sexe', 'diplome', 'specialite', 'dispo']
    X = pd.get_dummies(X, columns=cat)
    X[['age', 'note']] /= 100
    X['salaire'] = (X['salaire'] - X['salaire'].mean()) / X['salaire'].std()
    X['exp'] = (X['exp'] - X['exp'].mean()) / X['exp'].std()
    return train_test_split(X, y, test_size=test_size)


def feature_importance(X, y):
    """Get the feature importance."""
    ert = ExtraTreesClassifier(n_estimators=200, n_jobs=-1)
    ert.fit(X, y)
    fi = ert.feature_importances_
    idx = np.argsort(fi)[::-1]
    plt.xticks(rotation=30)
    sns.barplot(X.columns.values[idx], fi[idx])


def make_clfs():
    """Make classifiers following the standard methods."""
    nb = GaussianNB()
    lr = LogisticRegressionCV(cv=5, solver='lbfgs', multi_class='multinomial',
        n_jobs=-1)
    ert = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
    svm = SVC(gamma='scale')
    gbt = GradientBoostingClassifier()
    mlp = MLPClassifier((60, 60, 60, 60), solver='lbfgs')
    return {
        'Naive Bayes': nb,
        'Logistic Regression': lr,
        'Extremely Randomized Trees': ert,
        'Gradient Boosted Trees': gbt,
        'SVM': svm,
        'Multi-layer Perceptron': mlp
        }


def compare_clfs(clfs, X, y):
    """Compare the given classifiers."""
    for name, clf in clfs.items():
        train_size, train, test = learning_curve(clf, X, y, cv=5, verbose=1,
            shuffle=True, n_jobs=-1)
        trainmean = np.mean(train, axis=1)
        trainstd = np.std(train, axis=1)
        testmean = np.mean(test, axis=1)
        teststd = np.std(test, axis=1)
        f = plt.figure()
        plt.title(name)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.plot(train_size, trainmean, 'o-', color='b', label='Training')
        plt.fill_between(train_size, trainmean - trainstd,
            trainmean + trainstd, alpha=0.1, color='b')
        plt.plot(train_size, testmean, 'o-', color='r',
            label='Cross-validation')
        plt.fill_between(train_size, testmean - teststd,
            testmean + teststd, alpha=0.1, color='r')
        plt.legend(loc='best')
        sns.despine()
        f.savefig(name + '.pdf', bbox_inches='tight')


def main():
    """Do stuff."""
    X_train, X_test, y_train, y_test = pp(load_data())
    clfs = make_clfs()
    compare_clfs(clfs, X_train, y_train)


if __name__ == "__main__":
    main()
