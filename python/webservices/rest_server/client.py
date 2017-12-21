import requests
import h5py
import numpy as np
import pandas as pd
from keras.utils import np_utils
import sys,os
sys.path.append(os.getcwd())

def get_predictions(X_input,dimensions):
    """Get predictions from a rest backend for your input."""
    print("Requesting prediction for image\n")
    r = requests.post("http://localhost:7171/predict", json={'X_input': X_input,'dimensions': dimensions})
    print(r.status_code, r.reason)
    resp = r.json()
    prediction = resp['pred_val'][0]
    return prediction

def adience_datagenerator(filepath, batchsize,dimensions):
    while 1:

        f = h5py.File(filepath, "r")
        filesize = len(f['train_labels'])

        n_entries = 0
        while n_entries < (filesize - batchsize):
            x_train= f['train_images'][n_entries : n_entries + batchsize]
            #x_train= np.reshape(x_train, dimensions).astype('float32')

            y_train = f['train_labels'][n_entries:n_entries+batchsize]
            y_train_1 = y_train[:,0]
            y_train_2 = y_train[:,1]
            y_train_1_onecoding = np_utils.to_categorical(y_train_1, 8)
            y_train_2_onecoding = np_utils.to_categorical(y_train_2, 2)
            y_train_onecoding = np.concatenate((y_train_2_onecoding, y_train_1_onecoding), axis = 1)

            n_entries += batchsize
            yield (x_train, y_train_onecoding)
        f.close()

hdf5_path = '../../data/Adience/hdf5/adience.h5'
batch_size = 4
dimensions = (batch_size, 256, 256,3)

vals = adience_datagenerator(hdf5_path, batch_size, dimensions)
x_test, y_test = vals.next()

_json = pd.Series(np.reshape(x_test,-1)).to_json(orient='values')

prediction = get_predictions(_json,dimensions)

prediction




