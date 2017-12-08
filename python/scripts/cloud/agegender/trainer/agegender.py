import numpy as np
import h5py
#import matplotlib.pyplot as plt
import random
from keras.utils import np_utils
from keras.layers import Input, Conv2D, Dense,MaxPooling2D, Flatten, Activation,Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.backend import tf as ktf
from keras import optimizers
from keras.callbacks import History, EarlyStopping
import argparse
from tensorflow.python.lib.io import file_io
import pickle as pkl

import numpy as np
np.random.seed(123)


pickle_path = 'gs://subramgo/data/adience.pkl'
f_stream = file_io.FileIO(pickle_path ,mode="r")
data  = pkl.load(f_stream)

(x_train, y_train_onecoding)  = data

batch_size = 256
nb_class = 10






def age_gender_model():
    

    input_shape = (256, 256, 3)
    
    x_input = Input(input_shape)
    
    # Conv Layer 1
    
    x = Conv2D(filters = 96, kernel_size = (7,7), strides = (1,1), \
               padding = "valid", kernel_initializer='glorot_uniform')(x_input)
    
    x = Activation("relu")(x)
    
    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    
    x = BatchNormalization()(x)
    
    # Conv Layer 2
    
    x = Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1), 
               padding = "valid",kernel_initializer='glorot_uniform')(x)
    
    x = Activation("relu")(x)


    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    
    x = BatchNormalization()(x)

    # Conv Layer 3
    
    x = Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), 
               padding = "valid",kernel_initializer='glorot_uniform')(x)
    
    x = Activation("relu")(x)


    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    
    x = BatchNormalization()(x)
    
    
    x = Flatten()(x)
    
    x = Dense(512, activation = "relu")(x)
    
    x = Dropout(rate = 0.5)(x)
    
    x = Dense(512, activation ="relu")(x)
    
    x = Dropout(rate = 0.5)(x)

    
    predictions = Dense(10, activation="softmax")(x)
    
    model = Model(inputs = x_input, outputs = predictions)

    
    return model





# Optimizer
sgd = optimizers.SGD(lr=0.001, momentum=0, decay=1e-6, nesterov=False)
# Callbacks
early_stop_th = 10**-7
callbacks = [EarlyStopping(monitor='acc', min_delta=early_stop_th, patience=5, verbose=0, mode='auto')]
#
model = age_gender_model()
batch_size = 64
epochs = 100

model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ["accuracy"])


hist = model.fit(x_train, y_train_onecoding, batch_size=32,validation_split = 0.1,
                     epochs=epochs, callbacks = callbacks, verbose = 1)



