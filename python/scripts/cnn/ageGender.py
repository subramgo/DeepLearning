"""
Classification model for Age and Gender

Uses Adience data for training.
"""

import numpy as np
import h5py
#import matplotlib.pyplot as plt
import random
from keras.utils import np_utils
from keras.layers import Input, Conv2D, Dense,MaxPooling2D, Flatten, Activation,Dense, Dropout, BatchNormalization,GlobalAveragePooling2D
from keras.models import Model
from keras.backend import tf as ktf
from keras import optimizers
from keras.callbacks import History, EarlyStopping
import keras.backend as K
import sys,os
sys.path.append(os.getcwd())
from utils.evaluate import Evaluate
from utils.DataGenerator import adience_datagenerator
from utils.memoryreqs import get_model_memory_usage

K.set_image_data_format('channels_last')

np.random.seed(123)








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
sgd = optimizers.SGD(lr=0.005, momentum=0, decay=1e-6, nesterov=False)


# Callbacks
early_stop_th = 10**-7
callbacks = [EarlyStopping(monitor='acc', min_delta=early_stop_th, patience=5, verbose=0, mode='auto')]



model = age_gender_model()
batch_size = 4

print(get_model_memory_usage(batch_size, model))
epochs = 10
hdf5_path = '../data/Adience/hdf5/adience.h5'


model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ["accuracy"])

hist = model.fit_generator(adience_datagenerator(hdf5_path, batch_size), steps_per_epoch = 1000,  epochs = epochs)

model_path = '../models/age_gender_model.h5'
model.save(model_path)
del model 


evals = adience_datagenerator(hdf5_path, batch_size)
x_test, y_test = next(evals)
eval = Evaluate(model_path, x_test, y_test, batch_size = 1)
eval.process()


## TODO: evaluate insample test (`hist` object) 
## TODO: serialize model for domain transfer testing

