"""
Classification model for Age and Gender

Uses Adience data for training.
"""

import numpy as np
import h5py
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
from DLUtilsEvaluation import DemographicClassifier
from DLUtilsDataGenerator import adience_datagenerator,adience_datagenerator_16classes
from DLUtilsmemoryreqs import get_model_memory_usage,model_memory_params

K.set_image_data_format('channels_last')


np.random.seed(123)



def age_gender_model():
    

    input_shape = (100, 100, 3)
    
    x_input = Input(input_shape)
    
    # Conv Layer 1
    
    x = Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), \
               padding = "valid", kernel_initializer='glorot_uniform')(x_input)
    
    x = Activation("relu")(x)
    
    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    
    x = BatchNormalization()(x)
    
    # Conv Layer 2
    
    x = Conv2D(filters = 100, kernel_size = (3,3), strides = (1,1), 
               padding = "valid",kernel_initializer='glorot_uniform')(x)
    
    x = Activation("relu")(x)


    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    
    x = BatchNormalization()(x)

    # Conv Layer 3
    x = Conv2D(filters = 200, kernel_size = (3,3), strides = (1,1), 
               padding = "valid",kernel_initializer='glorot_uniform')(x)
    
    x = Activation("relu")(x)


    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    
    x = BatchNormalization()(x)



    x = GlobalAveragePooling2D()(x)        
    #x = Flatten()(x)
    
    x = Dense(512, activation = "relu")(x)
    
    x = Dropout(rate = 0.5)(x)
    
    x = Dense(512, activation ="relu")(x)
    
    x = Dropout(rate = 0.5)(x)

    
    predictions = Dense(16, activation="softmax")(x)
    
    model = Model(inputs = x_input, outputs = predictions)

    
    return model


def build_model(model, batch_size):

    # Optimizer
    sgd = optimizers.SGD(lr=0.001, momentum=0.95, decay=1e-6, nesterov=False)


    # Callbacks
    early_stop_th = 10**-7
    callbacks = [EarlyStopping(monitor='acc', min_delta=early_stop_th, patience=5, verbose=0, mode='auto')]




    epochs = 2500
    hdf5_path = '../data/Adience/hdf5/adience-100.h5'


    model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ["accuracy"])

    hist = model.fit_generator(adience_datagenerator_16classes(hdf5_path, batch_size), steps_per_epoch = 1000,  epochs = epochs)

    model_path = '../models/age_gender_model-0.3.h5'
    model.save(model_path)
    del model 


    evals = adience_datagenerator_16classes(hdf5_path, batch_size)
    x_test, y_test = next(evals)
    eval = DemographicClassifier(model_path)
    eval.process(x_test, y_test, batch_size = 1)


## TODO: evaluate insample test (`hist` object) 
## TODO: serialize model for domain transfer testing



if __name__ == '__main__':
    model = age_gender_model()
    batch_size = 8
    model_memory_params(batch_size, model)
    build_model(model, batch_size)





