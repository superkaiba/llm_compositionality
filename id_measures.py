import sklearn
import numpy as np

class ParticipationRatio:
    def __init__(self):
        self.pca = sklearn.decomposition.PCA()
    
    def fit_transform(self, data):
        self.pca.fit(data)
        explained_variances = self.pca.explained_variance_
        pr_id = sum(explained_variances)**2 / sum(explained_variances ** 2)

        return pr_id

class PCA:
    def __init__(self):
        self.pca = sklearn.decomposition.PCA()

    def fit_transform(self, data):
        var_thresh = 0.99
        self.pca.fit(data)
        pca_id = int(np.sum(np.cumsum(self.pca.explained_variance_ratio_) < var_thresh))

        return pca_id