import numpy as np
import h5py
from keras.utils import np_utils

def _generatorFactory(filepath,dimensions_minus_batch,x_label='train_images',y_label='train_labels'):
    """
        Produces a generator function.
        Give a filepath and data dimensions:
          HDF5 data is loaded from the filepath
          batchsize is prepended to the data dimensions for reshaping final output
        x and y labels need to match the HDF5 file's columns
    """
    def _generator(batchsize):
        dimensions = (batchsize,)+dimensions_minus_batch
        while 1:
            with h5py.File(filepath, "r") as f:
                filesize = len(f['train_labels'])
                n_entries = 0
                while n_entries < (filesize - batchsize):
                    x_train= f[x_label][n_entries : n_entries + batchsize]
                    x_train= np.reshape(x_train, dimensions).astype('float32')

                    y_train = f[y_label][n_entries:n_entries+batchsize]
                    y_train_onecoding = np_utils.to_categorical(y_train, 2)

                    n_entries += batchsize

                    # Shuffle
                    p = np.random.permutation(len(y_train_onecoding))
                    yield (x_train[p], y_train_onecoding[p])
                f.close()
    
    return _generator


_filepath = '../../data/Adience/hdf5/adience-100.h5'
_dimensions = (100,100,3)
adience_train_generator = _generatorFactory(_filepath,_dimensions)

"""
Produce three data samples from adience set:
    _generator = adience_train_generator(batchsize=3)
    _generator.next()     
"""
