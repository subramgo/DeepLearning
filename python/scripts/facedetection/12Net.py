"""
Face Detection 

"""

import numpy as np
import h5py
import random
from keras.utils import np_utils
from keras.layers import Input, Conv2D, Dense,MaxPooling2D, Reshape,Flatten, Activation,Dense, Dropout, BatchNormalization,GlobalAveragePooling2D
from keras.models import Model
from keras.backend import tf as ktf
from keras import optimizers
from keras.callbacks import History, EarlyStopping,ReduceLROnPlateau,CSVLogger,ModelCheckpoint
import keras.backend as K
from keras.engine import Layer
import sys,os
from DLUtils.evaluate import DemographicClassifier
from DLUtils.DataGenerator import face_12net_train_generator, face_12net_eval_generator
from DLUtils.MemoryReqs import get_model_memory_usage,model_memory_params
from DLUtils.configs import get_configs

K.set_image_data_format('channels_last')


np.random.seed(123)





def net12_model():

    input_shape = (12, 12, 3)
    nb_classes = 2

    x_input = Input(input_shape)
    # Conv Layer 1 # Input (12,12,3)
    x = Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), \
               padding = "valid", kernel_initializer='glorot_uniform')(x_input)
    print(x.shape)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)
   
    print(x.shape)

    # Conv Layer 2 # Input ()
    x = Conv2D(filters = 16, kernel_size = (4,4), strides = (1,1), 
               padding = "valid",kernel_initializer='glorot_uniform')(x)
    x = Activation("relu")(x)

    print(x.shape)
    # Conv Layer 3
    #x= Conv2D(filters = 2, kernel_size = (1,1), strides = (1,1), 
    #          padding = "valid",kernel_initializer='glorot_uniform', activation = "softmax")(x)
    x = Flatten()(x)
    x = Dense(16,activation = "relu")(x)

    print(x.shape)

    predictions = Dense(2, activation="softmax")(x)
   # x = Dense(16, activation = "relu")(x)

    #predictions = Dense(nb_classes, activation="softmax")(x)
    
    model = Model(inputs = x_input, outputs = predictions)

    
    return model




def build_model(model, config_dict):



    # Optimizer
    sgd = optimizers.SGD(lr= config_dict['learning_rate'] , momentum = config_dict['momentum']
        , decay=1e-6, nesterov=False)

    train_generator = face_12net_train_generator(config_dict['batch_size'])
    eval_generator = face_12net_eval_generator(config_dict['batch_size'])
   
    # Callbacks
    callbacks = [#EarlyStopping(monitor='acc', min_delta=config_dict['early_stop_th'], patience=5, verbose=0, mode='auto'), 
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0), 
    CSVLogger(config_dict['log_path'], separator=',', append=False),
    ModelCheckpoint(config_dict['check_path'], monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]

    # compile model
    model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ["accuracy"])

    # train model
    hist = model.fit_generator(train_generator, steps_per_epoch=config_dict['steps_per_epoch'], callbacks = callbacks,
        epochs=config_dict['epochs'], validation_data=eval_generator,validation_steps=config_dict['validation_steps'],verbose=2)

    """
    for i in range(config_dict['epochs']):
        avg_acc = 0
        for j in range(config_dict['steps_per_epoch']):
            x, y = next(train_generator)
            mets = model.train_on_batch(x, y)
            avg_acc+=mets[1]
        print("Epoch {} Accuracy {}".format(i+1, avg_acc / (1.0 * config_dict['steps_per_epoch']) ))
    """
    model.save(config_dict['model_path'])
    #del model 

if __name__ == '__main__':
    config_dict = get_configs('faces12net')
    model = net12_model()
    model_memory_params(config_dict['batch_size'], model)
    build_model(model, config_dict)