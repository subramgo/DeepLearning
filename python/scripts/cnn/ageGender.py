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
from DLUtils.evaluate import DemographicClassifier
from DLUtils.DataGenerator import adience_datagenerator
from DLUtils.MemoryReqs import get_model_memory_usage,model_memory_params
from keras.preprocessing.image import ImageDataGenerator
import configparser
from ast import literal_eval

K.set_image_data_format('channels_last')
np.random.seed(123)

def get_configs():
    Config = configparser.ConfigParser()
    Config.read('../settings/models.ini')
    section = 'agegender'
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = literal_eval(Config.get(section, option))
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    print (dict1)
    return dict1

def age_gender_model(input_shape, nb_classes):
    """
    """
    x_input = Input(input_shape)
    # Conv Layer 1
    x = Conv2D(filters = 96, kernel_size = (7,7), strides = (1,1), \
               padding = "valid", name = 'conv-1',kernel_initializer='glorot_uniform')(x_input)
    
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    x = BatchNormalization()(x)
    
    # Conv Layer 2
    x = Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1), 
               padding = "valid",name= 'conv-2',kernel_initializer='glorot_uniform')(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    x = BatchNormalization()(x)

    # Conv Layer 3
    x = Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), 
               padding = "valid",name= 'conv-3',kernel_initializer='glorot_uniform')(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    x = BatchNormalization()(x)
        
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(512, activation = "relu",name='dense-1')(x)
    x = Dropout(rate = 0.5)(x)
    x = Dense(512, activation ="relu",name='dense-2')(x)
    x = Dropout(rate = 0.5)(x)

    predictions = Dense(nb_classes, activation="softmax",name="softmax")(x)
    
    model = Model(inputs = x_input, outputs = predictions)

    
    return model


def build_model(model, config_dict):

    #hdf5_path = config_dict['hdf5_path']


    # Optimizer
    sgd = optimizers.SGD(lr= config_dict['learning_rate'] , momentum = config_dict['momentum']
        , decay=1e-6, nesterov=False)
   
    # Callbacks
    callbacks = [EarlyStopping(monitor='acc', min_delta=config_dict['early_stop_th'], patience=5, verbose=0, mode='auto')]

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            config_dict['train_path'],
            target_size=config_dict['target_size'],
            batch_size=config_dict['batch_size'],
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            config_dict['eval_path'],
            target_size=config_dict['target_size'],
            batch_size=config_dict['batch_size'],
            class_mode='categorical')


    model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ["accuracy"])
    hist = model.fit_generator(train_generator, steps_per_epoch=config_dict['steps_per_epoch'],verbose = 2,
        epochs=config_dict['epochs'], validation_data=validation_generator,validation_steps=config_dict['validation_steps'])

    model.save(config_dict['model_path'])
    #del model 

if __name__ == '__main__':
    config_dict = get_configs()
    model = age_gender_model(config_dict['input_shape'], config_dict['nb_classes'])
    model_memory_params(config_dict['batch_size'], model)
    build_model(model, config_dict)





