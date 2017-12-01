from keras import layers
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.callbacks import History, EarlyStopping
from keras.models import Model 
from keras import optimizers
from keras.datasets import mnist
from keras.utils import np_utils
from hyperas.distributions import uniform
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from keras.constraints import max_norm


import numpy as np
np.random.seed(123)




def mnist_data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes = 10
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test

# build model
def create_model(x_train, y_train, x_test, y_test):
    
    batch_size = 256
    epochs = 1
    learning_rate = 0.8713270582626444
    momentum = 0.8671876498073315
    decay = 0.0
    early_stop_th = 10**-5
    input_dim = (784,)

    dropout_1 = 0.026079803111884514
    dropout_2 = 0.4844455237320119

    # Stop the training if the accuracy is not moving more than a delta
    # keras.callbacks.History is by default added to all keras model
    # callbacks = [EarlyStopping(monitor='acc', min_delta=early_stop_th, patience=5, verbose=0, mode='auto')]

    # Code up the network
    x_input = Input(input_dim)
    x = Dropout(dropout_1)(x_input)
    x = Dense(1024, activation='relu', name ="dense1",kernel_constraint=max_norm( {{uniform(0.9, 5)}} ) )(x)
    x = Dropout(dropout_2)(x)
    x = Dense(1024, activation='relu', name = "dense2",kernel_constraint=max_norm( {{uniform(0.9,5)}} ) )(x)
    predictions = Dense(10, activation='softmax')(x)

    # Optimizer
    sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=0, nesterov=False)


    # Create and train model
    model = Model(inputs = x_input, outputs = predictions)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train,y= y_train, validation_split = 0.1, batch_size = batch_size ,epochs = epochs, verbose = 1)
    metrics = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=0, sample_weight=None, steps=None)


    accuracy = metrics[1]
    return {'loss': 1-accuracy, 'status': STATUS_OK, 'model': model}




if __name__ == "__main__":
    
    x_train, y_train, x_test, y_test = mnist_data()
    best_run, best_model = optim.minimize(model=create_model,
                                      data = mnist_data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials())

    
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)