import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

class TLEL:

    def __init__(self, n_learner=10, n_tree=10):
        self.n_learner = n_learner
        self.n_tree = n_tree
        self.learners = []

    def fit(self, X_train, y_train):
        df = pd.concat([X_train, y_train], axis=1)
        dfs = split_df(df, self.n_learner)
        for i in range(self.n_learner):
            dfs[i] = random_undersampling(dfs[i])
            X, y = dfs[i].iloc[:, :-1], dfs[i].iloc[:, -1]
            learner = RandomForestClassifier(n_estimators=self.n_tree, random_state=42)
            learner.fit(X, y)
            self.learners.append(learner)

    def predict_proba(self, X_test):
        y_pred = []
        for i in range(self.n_learner):
            y_pred.append(self.learners[i].predict_proba(X_test))
        return np.mean(y_pred, axis=0)


def split_df(df, num_subdf):
    df = df.sample(frac=1).reset_index(drop=True)
    sub_dfs = np.array_split(df, num_subdf)
    return sub_dfs


def random_undersampling(df):
    majority_class = df[df['label'] == 0]
    minority_class = df[df['label'] == 1]
    if len(majority_class) < len(minority_class):
        majority_class, minority_class = minority_class, majority_class

    n_samples = len(minority_class)
    majority_undersampled = resample(majority_class,
                                     replace=False,
                                     n_samples=n_samples,
                                     random_state=42)
    return pd.concat([minority_class, majority_undersampled])