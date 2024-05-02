from .. import np


def categorical_accuracy(y_act, y_pred):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_act, axis=1))

def sparse_accuracy(y_act, y_pred):
    # this might actually just work for all cases
    return np.mean(np.round(y_pred) == y_act)