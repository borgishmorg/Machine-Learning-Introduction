import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator


def entropy(y):    
    unique_y = set(y)
    res = 0
    for u in unique_y:
        p = (y == u).mean()
        res -= p*np.log2(p)
    return res


def gini(y):
    unique_y = set(y)
    res = 1
    for u in unique_y:
        p = (y == u).mean()
        res -= p*p
    return res


def variance(y):
    return np.var(y)


def mad_median(y):
    return np.abs(y-np.median(y)).mean()


regression_criterions = ['variance', 'mad_median']
criterions = {
    'entropy': entropy,
    'gini': gini,
    'variance': variance,
    'mad_median': mad_median
}


class DecisionTree(BaseEstimator):
    
    def __init__(
        self, 
        max_depth=np.inf, 
        min_samples_split=2, 
        criterion='gini',
        k=None, 
        debug=False
    ):
        self.k = k
        self.left: DecisionTree = None
        self.right: DecisionTree = None
        self.classes = None
        self.j = None
        self.t = None
        self.Q = -np.inf
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.criterion_func = criterions[criterion]
        self.regression = criterion in regression_criterions
        self.debug = debug
    
    def fit(self, X, y):
        k = len(np.unique(y))
        if self.k is None:
            self.k = k

        rows, features = X.shape
        if rows < self.min_samples_split or self.max_depth < 1 or k == 1:
            self.classes = y
            return self
        
        H = self.criterion_func

        for j in range(features):
            for t in np.unique(X[..., j]):
                l_index = X[..., j] <= t
                l = np.sum(l_index)
                
                if l == 0 or l == rows:
                    continue

                r_index = ~l_index

                Q = H(y) - l*H(y[l_index])/rows - (rows - l)*H(y[r_index])/rows
                if Q > self.Q:
                    self.Q = Q
                    self.j = j
                    self.t = t
        
        if self.Q == -np.inf:
            self.classes = y
            return self

        l_index = X[..., self.j] <= self.t
        r_index = ~l_index

        self.left = DecisionTree(
            max_depth=self.max_depth-1,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            debug=self.debug,
            k=self.k
        )
        self.left.fit(X[l_index], y[l_index])
        self.right = DecisionTree(
            max_depth=self.max_depth-1,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            debug=self.debug,
            k=self.k
        )
        self.right.fit(X[r_index], y[r_index])
        return self

    
    def predict_one(self, X):
        if self.classes is not None:
            if self.regression:
                return np.mean(self.classes)
            else:
                return scipy.stats.mode(self.classes)[0][0]
        if X[self.j] <= self.t:
            return self.left.predict_one(X)
        return self.right.predict_one(X)

    def predict(self, X):
        return np.apply_along_axis(
            self.predict_one,
            1,
            X
        )

    def predict_proba_one(self, X):
        if self.classes is not None:
            return np.array([ np.mean(self.classes == i) for i in range(self.k)])
        if X[self.j] <= self.t:
            return self.left.predict_proba_one(X)
        return self.right.predict_proba_one(X)

    def predict_proba(self, X):
        return np.apply_along_axis(
            self.predict_proba_one,
            1,
            X
        )
