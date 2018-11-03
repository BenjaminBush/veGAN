import numpy as np

def preprocess_data(X_train, X_test):
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    return X_train, X_test