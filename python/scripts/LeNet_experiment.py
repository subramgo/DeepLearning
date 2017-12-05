from keras.layers import Input, Conv2D, Dense,AveragePooling2D, Flatten, Activation,Dense, Dropout
from keras.models import Model
from keras.backend import tf as ktf
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(123)



nb_classes = 10


def get_data():
    """
    Return the MNIST data set reshaped to fit a ConvNet
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape to add the channel
    # input image dimensions
    img_rows, img_cols = 28, 28
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test 


def lenet_model(**kwargs):
    """
    Modified LeNet Model
    """
    
    input_shape = kwargs['input_shape']
    nb_classes  = kwargs['nb_classes']
    
    optimizer = kwargs['optimizer']
    loss      = kwargs['loss']
    mets      = kwargs['metrics']
    
    
    x_input = Input(input_shape)
    
    x = Conv2D(filters = 6, kernel_size = (5,5), strides = (1,1), \
               padding = "valid", kernel_initializer='glorot_uniform')(x_input)
    
    x = Activation("relu")(x)
    
    x = AveragePooling2D(pool_size = (2,2), strides = (2,2))(x)
    
    x = Conv2D(filters = 16, kernel_size = (5,5), strides = (1,1), 
               padding = "valid", name = "conv_1",kernel_initializer='glorot_uniform')(x)
    
    x = Activation("relu")(x)


    x = AveragePooling2D(pool_size = (2,2), strides = (2,2))(x)
    
    x = Flatten()(x)
    
    x = Dense(120, activation = "relu")(x)
    
    x = Dropout(rate = 0.5)(x)
    
    x = Dense(120, activation ="relu")(x)
    
    x = Dropout(rate = 0.5)(x)

    
    predictions = Dense(nb_classes, activation="softmax")(x)
    
    model = Model(inputs = x_input, outputs = predictions)
    model.compile(optimizer = optimizer, loss= loss, metrics = mets)

    
    return model




    



    


