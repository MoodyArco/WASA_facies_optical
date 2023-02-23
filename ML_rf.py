import numpy as np
import pandas as pd
from csv import reader
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time

start = time.process_time()

def train_test_split(y, groups, random_state = 24):
    """
    This function is to directly generate indices for the training/
    test set, which follows the strategy of OnegroupOnefacies_cv.
    """
    np.random.seed(random_state) 

    # pick up one section from each facies to test set
    sections_test = []
    for fa in np.unique(y):
        sections_test = np.hstack([sections_test, np.random.choice(np.unique(groups[y == fa]), 1)])
        
    # build the indices for data points
    test_idxs = []
    for section in np.unique(sections_test):
        test_idxs = np.hstack([test_idxs, np.where(groups == section)[0]])
    test_idxs = test_idxs.astype(int)
        
        # the training indices are the rest of indices
    train_idxs = np.array(
        list(
            set(np.arange(0, len(y), 1)) - set(test_idxs)
        )
    )
        
    return train_idxs, test_idxs

def OnegrupOnefacies_cv(y, groups, n_splits = 5, random_state = 24):
        """
        This function is for integrating with sklearn.GridSearchCV.
        It picks up one section in each facies as the test set randomly 
        while the rest are as training set.
        """
        np.random.seed(random_state) 

        for _ in range(n_splits):
            # pick up one section from each facies to test set
            sections_test = []
            for fa in np.unique(y):
                sections_test = np.hstack([sections_test, 
                                           np.random.choice(np.unique(groups[y == fa]), 1)])

            # build the indices for data points
            test_idxs = []
            for section in np.unique(sections_test):
                test_idxs = np.hstack([test_idxs, np.where(groups == section)[0]])
            test_idxs = test_idxs.astype(int)

            # the training indices are the rest of indices
            train_idxs = np.array(
                list(
                    set(np.arange(0, len(y), 1)) - set(test_idxs)
                )
            )

            yield train_idxs, test_idxs

df = pd.read_csv("N:/vendor126/database_reclass.csv")
print(df.head)
print("read finished")
#split
X = df.iloc[:, 2:].values
Y = df.iloc[:,1].values
groups = df.iloc[:,0].values
print(type(X))
print(type(Y))

print("step2")
train_idx, test_idx = train_test_split(Y, groups)
X_train = X[train_idx]
Y_train = Y[train_idx]
groups_train = groups[train_idx]
print(type(X_train))
print(type(Y_train))

print("split finished")

#ML

pipe = Pipeline([
    ('scaling', StandardScaler()),
    ('pca', PCA(n_components=1)), 
    ('rf', RandomForestClassifier(class_weight='balanced', random_state = 24))])
param_grid = [
    {'pca':[PCA(whiten=True)],
    'rf__max_depth': [5, 10, 15],
    'rf__n_estimators': [100, 1000, 5000]}
]

mycv = OnegrupOnefacies_cv(Y_train, groups_train, n_splits = 5, random_state=24)
grid = GridSearchCV(pipe, param_grid=param_grid, cv = mycv, scoring='balanced_accuracy', n_jobs=45)
grid.fit(X_train, Y_train)

print("Best score on validation set: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)
end = time.process_time()
print("Run time = %f seconds" % (end-start))