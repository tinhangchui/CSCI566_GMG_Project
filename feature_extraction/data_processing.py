import numpy as np
from sklearn.model_selection import train_test_split

class Preprocessing:
    def __init__(self, dataFilePath, labelFilePath, testset_ratio, validset_ratio):
        """
        dataFilePath: a list of path of data.npy (You need to select with data.npy to be used for training)
        labelFilePath: a list of path of labels.npy (You need to select with label.npy to be used for training)
        testset_ratio: a float between 0 to 1. This ratio determines how many data is used in test set. The rest will be
                       used for train set and validation set.
        validset_ratio:  a float between 0 to 1. This ratio determines how many data is used in validation set.
                         The rest will be used for traing set.
        """
        if not isinstance(dataFilePath, list) and isinstance(dataFilePath, str):
            dataFilePath = [dataFilePath]
        else:
            raise TypeError('dataFilePath must be a list or string'.)

        if not isinstance(labelFilePath) and isinstance(labelFilePath, str):
            labelFilePath = [labelFilePath]
        else:
            raise TypeError('labelFilePath must be a list or string.')

        dataList = [np.load(dataFile) for dataFile in dataFilePath]
        labelList = [np.load(labelFile) for labelFile in labelFilePath]

        X = np.concatenate(dataList)
        y = np.concatenate(labelList)
        X_train, X_test, self.X_test, self.y_test = train_test_split(X, y, test_size=testset_ratio, random_state=1)
        self.X_train, self.y_train, self.X_val, self.y_val = train_test_split(X_train, X_test, test_size=validset_ratio, random_state=1)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_val_data(self):
        return self.X_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.y_test
