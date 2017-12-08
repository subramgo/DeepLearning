import h5py
import numpy as np
import pickle as pkl
from keras.utils import np_utils





hdf5_path = '../../data/Adience/hdf5/adience.h5'
hdf5_file = h5py.File(hdf5_path, mode = 'r')

x_train = hdf5_file['train_images']
y_train = hdf5_file['train_labels']

y_train_1 = y_train[:,0]
y_train_2 = y_train[:,1]

y_train_1_onecoding = np_utils.to_categorical(y_train_1, 8)
y_train_2_onecoding = np_utils.to_categorical(y_train_2, 2)

y_train_onecoding = np.concatenate((y_train_2_onecoding, y_train_1_onecoding), axis = 1)


#to save it
with open("../../data/Adience/adience.pkl", "wb") as f:
    pkl.dump([x_train, y_train_onecoding], f,2)

