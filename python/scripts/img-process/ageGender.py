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

import numpy as np
np.random.seed(123)


hdf5_path = '../../data/Adience/hdf5/adience.h5'
hdf5_file = h5py.File(hdf5_path, mode = 'r')


data_size = hdf5_file['train_images'].shape[0]
batch_size = 256
nb_class = 10

print("Total Training Images {}".format(data_size))


x_train = hdf5_file['train_images']
y_train = hdf5_file['train_labels']


y_train_1 = y_train[:,0]
y_train_2 = y_train[:,1]



y_train_1_onecoding = np_utils.to_categorical(y_train_1, 8)
y_train_2_onecoding = np_utils.to_categorical(y_train_2, 2)

y_train_onecoding = np.concatenate((y_train_2_onecoding, y_train_1_onecoding), axis = 1)






def age_gender_model():
    

    input_shape = (256, 256, 3)
    
    x_input = Input(input_shape)
    
    # Conv Layer 1
    
    x = Conv2D(filters = 50, kernel_size = (7,7), strides = (1,1), \
               padding = "valid", kernel_initializer='glorot_uniform')(x_input)
    
    x = Activation("relu")(x)
    
    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    
    #x = BatchNormalization()(x)
    
    # Conv Layer 2
    
    x = Conv2D(filters = 100, kernel_size = (5,5), strides = (1,1), 
               padding = "valid",kernel_initializer='glorot_uniform')(x)
    
    x = Activation("relu")(x)


    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    
    #x = BatchNormalization()(x)

    # Conv Layer 3
    
    x = Conv2D(filters = 200, kernel_size = (3,3), strides = (1,1), 
               padding = "valid",kernel_initializer='glorot_uniform')(x)
    
    x = Activation("relu")(x)


    x = MaxPooling2D(pool_size = (3,3), strides = (1,1))(x)
    
    x = BatchNormalization()(x)
    
    
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(120, activation = "relu")(x)
    
    x = Dropout(rate = 0.5)(x)
    
    x = Dense(120, activation ="relu")(x)
    
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
batch_size = 64
epochs = 100

model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ["accuracy"])


hist = model.fit(x_train, y_train_onecoding, batch_size=32,validation_split = 0.1,
                     epochs=epochs, callbacks = callbacks, verbose = 1)



