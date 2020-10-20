import numpy as np
import sklearn as sk
from sklearn.base import TransformerMixin, BaseEstimator

class NormalizeScaler(TransformerMixin, BaseEstimator):
    """ A more flexible scaler for features and data, that mimics 
    sklearn.preprocessing.StandardScaler but makes the mean norm of the rows
    equal to one. """
    
    
    def __init__(self, *, with_mean=True, with_norm=True, per_feature=False):
        """ Initialize NormalizeScaler. Defines whether it will subtract mean
        (with_mean=True), apply normalization (with_norm=True) and whether it will
        normalize each feature separately (per_feature=True). """
        
        self.__with_mean = with_mean
        self.__with_norm = with_norm
        self.__per_feature = per_feature
        self.n_samples_seen_ = 0
        
    def fit(self, X, y=None):
        """ Compute mean and scaling to be applied for subsequent normalization. """
        
        
        self.n_samples_seen_, self.n_features_ = X.shape
        self.mean_ = np.zeros(self.n_features_)        
        
        if self.__with_mean:
            self.mean_ = X.mean(axis=0)
            centred_X = X - self.mean_
        else:
            centred_X = X
        
        self.scale_ = 1.0
        if self.__with_norm:
            var = (centred_X**2).mean(axis=0)
            
            if self.__per_feature:
                if np.any(var==0):
                    raise ValueError("Cannot normalize a feature with zero variance")
                self.scale_ = np.sqrt(1.0/(self.n_features_*var))                
            else:
                self.scale_ = 1.0/np.sqrt(var.sum())
                
        return self

    def transform(self, X, y=None):
        """ Normalize a vector based on previously computed mean and scaling. """
        
        if self.n_samples_seen_ == 0 :
            raise sk.exceptions.NotFittedError("This "+type(self).__name__+" instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return self.scale_*(X-self.mean_)
