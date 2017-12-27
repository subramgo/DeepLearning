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
from keras.callbacks import History, EarlyStopping,ReduceLROnPlateau,CSVLogger,ModelCheckpoint
import keras.backend as K
import sys,os
from DLUtils.evaluate import DemographicClassifier
from DLUtils.DataGenerator import adience_datagenerator
from DLUtils.MemoryReqs import get_model_memory_usage,model_memory_params
from keras.preprocessing.image import ImageDataGenerator
from DLUtils.configs import get_configs

K.set_image_data_format('channels_last')
np.random.seed(123)



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



    # Optimizer
    sgd = optimizers.SGD(lr= config_dict['learning_rate'] , momentum = config_dict['momentum']
        , decay=1e-6, nesterov=False)
   
 # Callbacks
    callbacks = [EarlyStopping(monitor='acc', min_delta=config_dict['early_stop_th'], patience=5, verbose=0, mode='auto'), 
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0), 
    CSVLogger(config_dict['log_path'], separator=',', append=False),
    ModelCheckpoint(config_dict['check_path'], monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]


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

if __name__ == '__main__':
    config_dict = get_configs('gender')
    model = age_gender_model(config_dict['input_shape'], config_dict['nb_classes'])
    model_memory_params(config_dict['batch_size'], model)
    build_model(model, config_dict)





