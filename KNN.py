'''
given a data point:
1] calculate its distance from all other points in the dataset
2] Get the closet K points
3] Regression: Get the average of their values
4] Classification: Get the most common class among them
'''
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum(x1-x2)**2)
    return distance


class KNN:
    def __init__(self,k=3):
        self.k = k
        
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        
    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self,x):
        # compute the distances between x and all examples in the training set
        # Euclidean distance
        distance = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_label = [self.y_train[i] for i in k_indices]
        
        # majority vote, most common class label
        most_common = Counter(k_nearest_label).most_common()
        return most_common[0][0]