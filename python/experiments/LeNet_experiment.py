from keras.layers import Input, Conv2D, Dense,AveragePooling2D, Flatten, Activation,Dense, Dropout
from keras.models import Model
from keras.backend import tf as ktf
from keras import optimizers
from keras.datasets import mnist
from keras.utils import np_utils
import sys,os
sys.path.append(os.getcwd())
from utils.memoryreqs import get_model_memory_usage,model_memory_params

import numpy as np
np.random.seed(123)

"""
Hyperas Debugging:
via https://github.com/maxpumperla/hyperas#typeerror-generator-object-is-not-subscriptable
    This is currently a known issue.
    Just `pip install networkx==1.11`
"""

def mnist_data():
    """
    Return the MNIST data set reshaped to fit a ConvNet
    """
    nb_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape to add the channel
    # input image dimensions
    img_rows, img_cols = 28, 28
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test ,nb_classes


def lenet_model(x_train, y_train, x_test, y_test,nb_classes):
    """
    Modified LeNet Model
    """
    input_shape = x_train.shape[1:]
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

    return model 

def train_model(model,x_train, y_train, x_test, y_test,batch_size):

    sgd = optimizers.SGD(lr=0.01, momentum=0, decay=0, nesterov=False)
    epochs = 2
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x=x_train,y= y_train, validation_split = 0.1, batch_size = batch_size ,epochs = epochs, verbose = 1)
    metrics = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=0, sample_weight=None, steps=None)


    accuracy = metrics[1]
    print("loss {}".format( (1-accuracy)*100 ) )

    # Save the model
    model.save('../models/lenet.h5')

    


if __name__ == "__main__":
    x_train, y_train, x_test, y_test,nb_classes = mnist_data()
    model = lenet_model(x_train, y_train, x_test, y_test,nb_classes)
    print("Model size {} GB".format(get_model_memory_usage(32,model)))
    batch_size = 32

    model_memory_params(batch_size, model)
    #train_model(model,x_train, y_train, x_test, y_test,batch_size)


    

    



    


