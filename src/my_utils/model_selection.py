import numpy as np


class BlockingTimeSeriesSplit():
    def __init__(self, n_splits, test_size=0.25):
        self.n_splits = n_splits
        self.test_size = test_size

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        k_fold_size_x = int(k_fold_size * (1 - self.test_size))
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = k_fold_size_x + start
            yield indices[start: mid], indices[mid + margin: stop]
