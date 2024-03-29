import numpy as np
from sklearn.model_selection import train_test_split


def merge_labels(predicted_tse, predicted_bpm, predicted_energy):
    """
    Example:
    Input:
    predicted_tse: [0,0,0]
    predicted_bpm: [1,1,1]
    predicted_energy: [2,2,2]

    Expected output:
    [
        [0,1,2],
        [0,1,2],
        [0,1,2],
    ]
    """
    num_preds = len(predicted_tse)
    assert len(predicted_bpm) == num_preds
    assert len(predicted_energy) == num_preds
    default_tse_set = np.array([4] * num_preds).reshape(-1,1)
    predicted_tse = predicted_tse.reshape(-1,1)
    predicted_bpm = predicted_bpm.reshape(-1,1)
    predicted_energy = predicted_energy.reshape(-1,1)
    return np.concatenate((default_tse_set, predicted_tse, predicted_bpm, predicted_energy),axis=1)


def generate_wb_config(predicted_labels):
    """
    Example:
    Input:
    predicted_labels: []
    """
    pass


def load_predict_data(data_path, num_prediction):
    if isinstance(data_path, str):
        data_path = [data_path]
    elif not isinstance(data_path, list):
        raise TypeError('data_path must be a list or string.')
    dataList = [np.load(dataFile) for dataFile in data_path]
    X = np.array(np.concatenate(dataList)).astype(np.float)
    shape = X.shape
    if len(shape) is 5:
        X = X.reshape(shape[0] * shape[1], shape[2], shape[3], shape[4])
    return X[:num_prediction]


class Preprocessing:
    def __init__(self, dataFilePath, labelFilePath, testset_ratio, validset_ratio, sample_per_section=9):
        """
        dataFilePath: a list of path of data.npy (You need to select with data.npy to be used for training)
        labelFilePath: a list of path of labels.npy (You need to select with label.npy to be used for training)
        testset_ratio: a float between 0 to 1. This ratio determines how many data is used in test set. The rest will be
                       used for train set and validation set.
        validset_ratio:  a float between 0 to 1. This ratio determines how many data is used in validation set.
                         The rest will be used for traing set.
        sample_per_section: the number of samples in one video section, label will be repeated according to this number to match with image data.
        """
        if isinstance(dataFilePath, str):
            dataFilePath = [dataFilePath]
        elif not isinstance(dataFilePath, list):
            raise TypeError('dataFilePath must be a list or string.')

        if isinstance(labelFilePath, str):
            labelFilePath = [labelFilePath]
        elif not isinstance(labelFilePath, list):
            raise TypeError('labelFilePath must be a list or string.')

        dataList = [np.load(dataFile) for dataFile in dataFilePath]
        labelList = [np.load(labelFile) for labelFile in labelFilePath]

        X = np.array(np.concatenate(dataList)).astype(np.float)
        y = np.array(np.concatenate(labelList)).astype(np.float)

        shape = X.shape
        X = X.reshape(shape[0] * shape[1], shape[2], shape[3], shape[4])
        y = np.repeat(y, sample_per_section, axis=0)

        # convert bpm
        y[:,2] = (60 * 1000000) / (y[:,2] * 4)

        print(X.shape)
        print(y.shape)

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=testset_ratio, random_state=1)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, test_size=validset_ratio, random_state=1)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_val_data(self):
        return self.X_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.y_test
