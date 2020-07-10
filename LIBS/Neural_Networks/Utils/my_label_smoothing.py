import numpy as np


def smooth_labels(y, label_smoothing):
    # https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.

    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        label_smoothing: label smoothing factor (between 0 and 1)

    # Returns
        A matrix of smoothed labels.
    '''

    y1 = y.copy()
    y1 = y1.astype(np.float32)
    assert len(y1.shape) == 2 and 0 <= label_smoothing <= 1

    y1 *= 1 - label_smoothing
    y1 += label_smoothing / y1.shape[1]

    # y1[y1 == 1] = 1-label_smoothing + label_smoothing / num_classes
    # y1[y1 == 0] = label_smoothing / num_classes

    # np.multiply(y1, 1-label_smoothing, out=y1, casting="unsafe")

    return y1