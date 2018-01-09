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
    def _generator(dimensions,nbclasses,batchsize):
        while 1:
            with h5py.File(filepath, "r") as f:
                filesize = len(f['train_labels'])
                n_entries = 0
                while n_entries < (filesize - batchsize):
                    x_train= f[x_label][n_entries : n_entries + batchsize]
                    x_train= np.reshape(x_train, dimensions).astype('float32')

                    y_train = f[y_label][n_entries:n_entries+batchsize]
                    # data-specific formatting should be done elsewhere later, even onecoding
                    # if dimensions is needed, can be gotten from x_train.shape
                    y_train_onecoding = np_utils.to_categorical(y_train, nbclasses)

                    n_entries += batchsize

                    # Shuffle
                    p = np.random.permutation(len(y_train_onecoding))
                    yield (x_train[p], y_train_onecoding[p])
                f.close()
    
    return _generator


if __name__ == '__main__':
    config = configs.Config()
    _filepath = config.get_section_dict('adience')['data_path']
    _adience_factory = _generatorFactory(_filepath)
    adience_train_generator = _adience_factory(batchsize=3)




