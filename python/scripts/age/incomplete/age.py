"""
Classification model for Age and Gender

Train a model using Adience data
Save model to HDF5 file:
  * architecture
  * weights

Define preprocessing:
  * ???

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
from DLUtils import datafeed
from DLUtils.MemoryReqs import get_model_memory_usage,model_memory_params
from keras.preprocessing.image import ImageDataGenerator
from DLUtils.configs import get_configs
from DLUtils.h5Utils import create_h5_file

K.set_image_data_format('channels_last')
np.random.seed(123)


def train_model():
    """  """

    # Instantiate model architecture
    model = age_model(config_dict['input_shape'], config_dict['nb_classes'])
    print(model.summary())

    # Optimizer
    sgd_opt = optimizers.SGD(lr= config_dict['learning_rate'] , momentum = config_dict['momentum'], decay=1e-6, nesterov=False)

    # Callbacks
    callbacks = [EarlyStopping(monitor='acc', min_delta=config_dict['early_stop_th'], patience=5, verbose=0, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
    CSVLogger(config_dict['log_path'], separator=',', append=False),
    ModelCheckpoint(config_dict['check_path'], monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]

    
    model.compile(optimizer = sgd_opt, loss = "categorical_crossentropy", metrics = ["accuracy"])
    hist = model.fit_generator(train_generator, steps_per_epoch=config_dict['steps_per_epoch'],verbose = 2,callbacks = callbacks,
        epochs=config_dict['epochs'], validation_data=validation_generator,validation_steps=config_dict['validation_steps'])


def serialize_model(keras_model,weights_path,model_path):
    keras_model.save_weights(weights_path)
    keras_model.save(model_path)

def main():
    config_dict = get_configs('gender')

    train_generator = datafeed.adience_train_generator(config_dict['batch_size'])
    validation_generator = datafeed.adience_eval_generator(config_dict['batch_size'])
    
    trained_model = train_model(train_generator,validation_generator)



    serialize_model(keras_model,config_dict['weights_path'],config_dict['model_path'])


def validation():
    pass
    #TODO
    #model_memory_params(config_dict['batch_size'], model)
    #


def age_model(input_shape, nb_classes):
    """ Implementation of XXX architecture """
    x_input = Input(input_shape)
    
    # Conv Layer 1
    x = Conv2D(filters = 96, kernel_size = (5,5), strides = (1,1), \
               padding = "valid", name = 'conv-1',kernel_initializer='glorot_uniform')(x_input)

    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)
    x = BatchNormalization()(x)

    # Conv Layer 2
    x = Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1),
               padding = "valid",name= 'conv-2',kernel_initializer='glorot_uniform')(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)
    x = BatchNormalization()(x)

    # Conv Layer 3
    x = Conv2D(filters = 512, kernel_size = (5,5), strides = (1,1),
               padding = "valid",name= 'conv-4',kernel_initializer='glorot_uniform')(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)
    x = BatchNormalization()(x)

    # Conv Layer 4
    x = Conv2D(filters = 1024, kernel_size = (1,1), strides = (1,1),
               padding = "valid",name= 'conv-3',kernel_initializer='glorot_uniform')(x)
    x = Activation("relu")(x)


    x = Flatten()(x)
    x = Dense(1024, activation = "relu",name='dense-1')(x)
    x = Dropout(rate = 0.5)(x)
    x = Dense(512, activation = "relu",name='dense-2')(x)
    x = Dropout(rate = 0.5)(x)
    x = Dense(512, activation ="relu",name='dense-3')(x)
    x = Dropout(rate = 0.5)(x)

    predictions = Dense(nb_classes, activation="softmax",name="softmax")(x)

    model = Model(inputs = x_input, outputs = predictions)


    return model



if __name__ == '__main__':

    if len(sys.argv) >= 2:
        if sys.argv[1] == 'create_h5':
            create_h5_file(config_dict['label_csv'], config_dict['h5_input'],config_dict['target_size'][0],config_dict['target_size'][0],'gender_class','image_loc','type')

    main()






