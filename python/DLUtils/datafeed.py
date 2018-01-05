import numpy as np
import h5py
from keras.utils import np_utils

from DLUtils import configs

def _generatorFactory(filepath,x_label='train_images',y_label='train_labels'):
    """
        Produces a generator function.
        Give a filepath and column labels:
            HDF5 data is loaded from the filepath
            x and y labels need to match the HDF5 file's columns
    """
    def _generator(batchsize):
        while 1:
            with h5py.File(filepath, "r") as f:
                filesize = len(f['train_labels'])
                n_entries = 0
                while n_entries < (filesize - batchsize):
                    x_train= f[x_label][n_entries : n_entries + batchsize]
                    #x_train= x_train.astype('float32')

                    y_train = f[y_label][n_entries:n_entries+batchsize]
                    y_train_onecoding = np_utils.to_categorical(y_train)

                    n_entries += batchsize

                    # Shuffle
                    p = np.random.permutation(len(y_train_onecoding))
                    yield (x_train[p], y_train_onecoding[p])
                f.close()
    
    return _generator


_filepath = configs.get_section_dict('adience')['data_path']
_adience_factory = _generatorFactory(_filepath)
adience_train_generator = _adience_factory(batchsize=3)




