"""
Gender Classification using Resnet50
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
from DLUtils.DataGenerator import adience_train_generator,adience_eval_generator
from DLUtils.MemoryReqs import get_model_memory_usage,model_memory_params
from keras.preprocessing.image import ImageDataGenerator
from DLUtils.configs import get_configs
from DLUtils.h5Utils import create_h5_file
from DLUtils.datafeed import _generatorFactory
from keras.applications import ResNet50

K.set_image_data_format('channels_last')
np.random.seed(123)



def age_gender_model(input_shape, nb_classes):
    """
    """

    resnet_base = ResNet50(include_top = False, weights = 'imagenet',input_tensor = None, input_shape = input_shape ,pooling = 'max')

    for layer in resnet_base.layers:
    	layer.trainable = False
    
        
    #x = Flatten()(x)
    x = Dense(1024, activation = "relu",name='dense-1')(resnet_base.output)
    x = Dropout(rate = 0.5)(x)
    x = Dense(512, activation = "relu",name='dense-2')(x)
    x = Dropout(rate = 0.5)(x)
    x = Dense(512, activation ="relu",name='dense-3')(x)
    x = Dropout(rate = 0.5)(x)
    x = Dense(256, activation ="relu",name='dense-4')(x)
    x = Dropout(rate = 0.5)(x)    

    predictions = Dense(nb_classes, activation="softmax",name="softmax")(x)
    
    model = Model(inputs = resnet_base.input, outputs = predictions)

    print(model.summary())
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


    #train_generator = adience_train_generator(config_dict['batch_size'])
    #validation_generator = adience_eval_generator(config_dict['batch_size'])
    _adience_train_factory = _generatorFactory(config_dict['h5_input'],x_label='train_images',y_label='train_labels')
    train_generator = _adience_train_factory(dimensions=config_dict['target_size'] ,nbclasses=config_dict['nb_classes'],batchsize=config_dict['batch_size'])

    _adience_eval_factory = _generatorFactory(config_dict['h5_input'],x_label='eval_images',y_label='eval_labels')
    validation_generator = _adience_eval_factory(dimensions=config_dict['target_size'] ,nbclasses=config_dict['nb_classes'],batchsize=config_dict['batch_size'])


    model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"])
    
    hist = model.fit_generator(train_generator, steps_per_epoch=config_dict['steps_per_epoch'],verbose = 2,callbacks = callbacks,
        epochs=config_dict['epochs'], validation_data=validation_generator,validation_steps=config_dict['validation_steps'])


    model.save_weights(config_dict['weights_path'])
    model.save(config_dict['model_path'])

if __name__ == '__main__':

    config_dict = get_configs('gender_resnet')


    if len(sys.argv) >= 2:
        if sys.argv[1] == 'create_h5':
            create_h5_file(config_dict['label_csv'], config_dict['h5_input'],config_dict['target_size'][0],config_dict['target_size'][0],'gender_class','image_loc','type')


    model = age_gender_model(config_dict['input_shape'], config_dict['nb_classes'])
    print(model.summary())
    model_memory_params(config_dict['batch_size'], model)
    build_model(model, config_dict)
