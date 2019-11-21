import numpy as np
from sklearn.model_selection import train_test_split

class Preprocessing:
    def __init__(self, dataFilePath, labelFilePath, testset_ratio, validset_ratio):
        """
        dataFilePath: the path of data.npy
        labelFilePath: the path of labels.npy
        testset_ratio: a float between 0 to 1. This ratio determines how many data is used in test set. The rest will be
                       used for train set and validation set.
        validset_ratio:  a float between 0 to 1. This ratio determines how many data is used in validation set.
                         The rest will be used for traing set.
        """
        X = np.load(dataFilePath)
        y = np.load(labelFilePath)
        X_train, X_test, self.X_test, self.y_test = train_test_split(X, y, test_size=testset_ratio, random_state=1)
        self.X_train, self.y_train, self.X_val, self.y_val = train_test_split(X_train, X_test, test_size=validset_ratio, random_state=1)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_val_data(self):
        return self.X_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.y_test
