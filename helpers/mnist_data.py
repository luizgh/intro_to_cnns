import numpy as np
import urllib
import os
import gzip
import cPickle

def load_data():
    dataset_file = 'mnist.pkl.gz'

    #Download dataset if not yet done:
    if not os.path.isfile(dataset_file):
        print('Downloading dataset')
        urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', dataset_file)

    #Load the dataset
    f = gzip.open(dataset_file, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    #Convert the dataset to the shape we want
    x_train, y_train = train_set
    x_valid, y_valid = valid_set
    x_test, y_test = test_set

    x_train = x_train.reshape(-1, 1, 28, 28)
    x_valid = x_valid.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)
    
    y_train = y_train.astype(np.int32)
    y_valid = y_valid.astype(np.int32)
    y_test = y_test.astype(np.int32)
    
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)