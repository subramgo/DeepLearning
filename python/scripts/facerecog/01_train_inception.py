"""
Train a siamese network made up of inception network to generate a 128 bit encoding
for a given face.

The inception model and the pre-trained weights are taken from
https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/utils.py


"""

from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate,Dropout, merge
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.models import load_model
import tensorflow as tf
import os
from numpy import genfromtxt
import numpy as np
from keras.callbacks import History, EarlyStopping,ReduceLROnPlateau,CSVLogger,ModelCheckpoint
from DLUtils.configs import get_configs
import cv2


config_dict = get_configs('facerecog')


def LRN2D(x):
  import tensorflow as tf
  return tf.nn.lrn(x, alpha=1e-4, beta=0.75)


#trainable_layers=[  'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3','dense_layer']
trainable_layers=[  'dense_layer']


weights = [
  'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
  'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
  'inception_3a_pool_conv', 'inception_3a_pool_bn',
  'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
  'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
  'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
  'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
  'inception_3b_pool_conv', 'inception_3b_pool_bn',
  'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
  'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
  'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
  'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
  'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
  'inception_4a_pool_conv', 'inception_4a_pool_bn',
  'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
  'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
  'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
  'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
  'inception_5a_pool_conv', 'inception_5a_pool_bn',
  'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
  'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
  'inception_5b_pool_conv', 'inception_5b_pool_bn',
  'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
  'dense_layer'
]

conv_shape = {
  'conv1': [64, 3, 7, 7],
  'conv2': [64, 64, 1, 1],
  'conv3': [192, 64, 3, 3],
  'inception_3a_1x1_conv': [64, 192, 1, 1],
  'inception_3a_pool_conv': [32, 192, 1, 1],
  'inception_3a_5x5_conv1': [16, 192, 1, 1],
  'inception_3a_5x5_conv2': [32, 16, 5, 5],
  'inception_3a_3x3_conv1': [96, 192, 1, 1],
  'inception_3a_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_3x3_conv1': [96, 256, 1, 1],
  'inception_3b_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_5x5_conv1': [32, 256, 1, 1],
  'inception_3b_5x5_conv2': [64, 32, 5, 5],
  'inception_3b_pool_conv': [64, 256, 1, 1],
  'inception_3b_1x1_conv': [64, 256, 1, 1],
  'inception_3c_3x3_conv1': [128, 320, 1, 1],
  'inception_3c_3x3_conv2': [256, 128, 3, 3],
  'inception_3c_5x5_conv1': [32, 320, 1, 1],
  'inception_3c_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_3x3_conv1': [96, 640, 1, 1],
  'inception_4a_3x3_conv2': [192, 96, 3, 3],
  'inception_4a_5x5_conv1': [32, 640, 1, 1,],
  'inception_4a_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_pool_conv': [128, 640, 1, 1],
  'inception_4a_1x1_conv': [256, 640, 1, 1],
  'inception_4e_3x3_conv1': [160, 640, 1, 1],
  'inception_4e_3x3_conv2': [256, 160, 3, 3],
  'inception_4e_5x5_conv1': [64, 640, 1, 1],
  'inception_4e_5x5_conv2': [128, 64, 5, 5],
  'inception_5a_3x3_conv1': [96, 1024, 1, 1],
  'inception_5a_3x3_conv2': [384, 96, 3, 3],
  'inception_5a_pool_conv': [96, 1024, 1, 1],
  'inception_5a_1x1_conv': [256, 1024, 1, 1],
  'inception_5b_3x3_conv1': [96, 736, 1, 1],
  'inception_5b_3x3_conv2': [384, 96, 3, 3],
  'inception_5b_pool_conv': [96, 736, 1, 1],
  'inception_5b_1x1_conv': [256, 736, 1, 1],
}

def load_weights():
  # Set weights path
  print("Loading weights for the inception model")
  dirPath = '../models/facerecog/weights'
  fileNames = filter(lambda f: not f.startswith('.'), os.listdir(dirPath))
  paths = {}
  weights_dict = {}

  for n in fileNames:
    paths[n.replace('.csv', '')] = dirPath + '/' + n

  for name in weights:
    if 'conv' in name:
      conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
      conv_w = np.reshape(conv_w, conv_shape[name])
      conv_w = np.transpose(conv_w, (2, 3, 1, 0))
      conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
      weights_dict[name] = [conv_w, conv_b]     
    elif 'bn' in name:
      bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
      bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
      bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
      bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
      weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]
    elif 'dense' in name:
      dense_w = genfromtxt(dirPath+'/dense_w.csv', delimiter=',', dtype=None)
      dense_w = np.reshape(dense_w, (128, 736))
      dense_w = np.transpose(dense_w, (1, 0))
      dense_b = genfromtxt(dirPath+'/dense_b.csv', delimiter=',', dtype=None)
      weights_dict[name] = [dense_w, dense_b]

  return weights_dict

def conv2d_bn(
  x,
  layer=None,
  cv1_out=None,
  cv1_filter=(1, 1),
  cv1_strides=(1, 1),
  cv2_out=None,
  cv2_filter=(3, 3),
  cv2_strides=(1, 1),
  padding=None,
):
  num = '' if cv2_out == None else '1'
  tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, name=layer+'_conv'+num)(x)
  tensor = BatchNormalization(axis=3, epsilon=0.00001, name=layer+'_bn'+num)(tensor)
  tensor = Activation('relu')(tensor)
  if padding == None:
    return tensor
  tensor = ZeroPadding2D(padding=padding)(tensor)
  if cv2_out == None:
    return tensor
  tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, name=layer+'_conv'+'2')(tensor)
  tensor = BatchNormalization(axis=3, epsilon=0.00001, name=layer+'_bn'+'2')(tensor)
  tensor = Activation('relu')(tensor)
  return tensor


def load_model_weights(model):
  weights_dict = load_weights()

  # Set layer weights of the model
  for name in weights:
    if model.get_layer(name) != None:
      try:
        model.get_layer(name).set_weights(weights_dict[name])
      except ValueError:
        print(name)
        print(model.get_layer(name).input_shape)
        print(model.get_layer(name).output_shape)
#    elif model.get_layer(name) != None:
#     model.get_layer(name).set_weights(weights_dict[name])
  return model 


def faceRecoModel(input_shape):
  myInput = Input(input_shape)

  x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
  x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
  x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
  x = Activation('relu')(x)
  x = ZeroPadding2D(padding=(1, 1))(x)
  x = MaxPooling2D(pool_size=3, strides=2)(x)
  x = Lambda(LRN2D, name='lrn_1')(x)
  x = Conv2D(64, (1, 1), name='conv2')(x)
  x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
  x = Activation('relu')(x)
  x = ZeroPadding2D(padding=(1, 1))(x)
  x = Conv2D(192, (3, 3), name='conv3')(x)
  x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
  x = Activation('relu')(x)
  x = Lambda(LRN2D, name='lrn_2')(x)
  x = ZeroPadding2D(padding=(1, 1))(x)
  x = MaxPooling2D(pool_size=3, strides=2)(x)

  # Inception3a
  inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
  inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
  inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
  inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
  inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
  inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
  inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

  inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
  inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
  inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
  inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
  inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
  inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
  inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

  inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
  inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
  inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
  inception_3a_pool = Activation('relu')(inception_3a_pool)
  inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

  inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
  inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
  inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

  inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

  # Inception3b
  inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
  inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
  inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
  inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
  inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
  inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
  inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

  inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
  inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
  inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
  inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
  inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
  inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
  inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

  inception_3b_pool = Lambda(lambda x: x**2, name='power2_3b')(inception_3a)
  inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
  inception_3b_pool = Lambda(lambda x: x*9, name='mult9_3b')(inception_3b_pool)
  inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
  inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
  inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
  inception_3b_pool = Activation('relu')(inception_3b_pool)
  inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

  inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
  inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
  inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

  inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

  # Inception3c
  inception_3c_3x3 = conv2d_bn(inception_3b,
                                     layer='inception_3c_3x3',
                                     cv1_out=128,
                                     cv1_filter=(1, 1),
                                     cv2_out=256,
                                     cv2_filter=(3, 3),
                                     cv2_strides=(2, 2),
                                     padding=(1, 1))

  inception_3c_5x5 = conv2d_bn(inception_3b,
                                     layer='inception_3c_5x5',
                                     cv1_out=32,
                                     cv1_filter=(1, 1),
                                     cv2_out=64,
                                     cv2_filter=(5, 5),
                                     cv2_strides=(2, 2),
                                     padding=(2, 2))

  inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
  inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

  inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

  #inception 4a
  inception_4a_3x3 = conv2d_bn(inception_3c,
                                     layer='inception_4a_3x3',
                                     cv1_out=96,
                                     cv1_filter=(1, 1),
                                     cv2_out=192,
                                     cv2_filter=(3, 3),
                                     cv2_strides=(1, 1),
                                     padding=(1, 1))
  inception_4a_5x5 = conv2d_bn(inception_3c,
                                     layer='inception_4a_5x5',
                                     cv1_out=32,
                                     cv1_filter=(1, 1),
                                     cv2_out=64,
                                     cv2_filter=(5, 5),
                                     cv2_strides=(1, 1),
                                     padding=(2, 2))

  inception_4a_pool = Lambda(lambda x: x**2, name='power2_4a')(inception_3c)
  inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
  inception_4a_pool = Lambda(lambda x: x*9, name='mult9_4a')(inception_4a_pool)
  inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)
  inception_4a_pool = conv2d_bn(inception_4a_pool,
                                     layer='inception_4a_pool',
                                     cv1_out=128,
                                     cv1_filter=(1, 1),
                                     padding=(2, 2))
  inception_4a_1x1 = conv2d_bn(inception_3c,
                                     layer='inception_4a_1x1',
                                     cv1_out=256,
                                     cv1_filter=(1, 1))
  inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

  #inception4e
  inception_4e_3x3 = conv2d_bn(inception_4a,
                                     layer='inception_4e_3x3',
                                     cv1_out=160,
                                     cv1_filter=(1, 1),
                                     cv2_out=256,
                                     cv2_filter=(3, 3),
                                     cv2_strides=(2, 2),
                                     padding=(1, 1))
  inception_4e_5x5 = conv2d_bn(inception_4a,
                                     layer='inception_4e_5x5',
                                     cv1_out=64,
                                     cv1_filter=(1, 1),
                                     cv2_out=128,
                                     cv2_filter=(5, 5),
                                     cv2_strides=(2, 2),
                                     padding=(2, 2))
  inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
  inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

  inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

  #inception5a
  inception_5a_3x3 = conv2d_bn(inception_4e,
                                     layer='inception_5a_3x3',
                                     cv1_out=96,
                                     cv1_filter=(1, 1),
                                     cv2_out=384,
                                     cv2_filter=(3, 3),
                                     cv2_strides=(1, 1),
                                     padding=(1, 1))

  inception_5a_pool = Lambda(lambda x: x**2, name='power2_5a')(inception_4e)
  inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
  inception_5a_pool = Lambda(lambda x: x*9, name='mult9_5a')(inception_5a_pool)
  inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)
  inception_5a_pool = conv2d_bn(inception_5a_pool,
                                     layer='inception_5a_pool',
                                     cv1_out=96,
                                     cv1_filter=(1, 1),
                                     padding=(1, 1))
  inception_5a_1x1 = conv2d_bn(inception_4e,
                                     layer='inception_5a_1x1',
                                     cv1_out=256,
                                     cv1_filter=(1, 1))

  inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

  #inception_5b
  inception_5b_3x3 = conv2d_bn(inception_5a,
                                     layer='inception_5b_3x3',
                                     cv1_out=96,
                                     cv1_filter=(1, 1),
                                     cv2_out=384,
                                     cv2_filter=(3, 3),
                                     cv2_strides=(1, 1),
                                     padding=(1, 1))
  inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
  inception_5b_pool = conv2d_bn(inception_5b_pool,
                                     layer='inception_5b_pool',
                                     cv1_out=96,
                                     cv1_filter=(1, 1))
  inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

  inception_5b_1x1 = conv2d_bn(inception_5a,
                                     layer='inception_5b_1x1',
                                     cv1_out=256,
                                     cv1_filter=(1, 1))
  inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

  av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
  reshape_layer = Flatten()(av_pool)
  dense_layer = Dense(128, name='dense_layer')(reshape_layer)
  norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)


  # Final Model
  model = Model(inputs=myInput, outputs=norm_layer)
  model = load_model_weights(model)

  # Set trainable
  for layer in model.layers:
    layer.trainable=False

  for layer_name in trainable_layers:
    model.get_layer(layer_name).trainable = True
  return model


def identity_loss(y_true, y_pred):
  return K.mean(y_pred -  0 * y_true)


def triplet_loss(y_true, y_pred):
  import tensorflow as tf
  anchor = y_pred[:,0]
  positive = y_pred[:,1]
  negative = y_pred[:,2]
  #anchor, positive, negative = y_pred
  alpha = 0.2
  pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
  neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
  basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
  loss = tf.maximum(tf.reduce_mean(basic_loss), 0.0)
  return loss

def build_siamese_model():

  inception_model = faceRecoModel((96,96,3))

  anchor_input = Input((96,96,3),name="anchor_input")
  anchor_output = inception_model(anchor_input)

  positive_input = Input((96,96,3), name="positive_input")
  positive_output = inception_model(positive_input)

  negative_input = Input((96,96,3), name="negative_input")
  negative_output = inception_model(negative_input)

  #loss = merge([anchor_output, positive_output, negative_output], mode =triplet_loss,name='loss', output_shape= (1,) )
 # final = concatenate([anchor_output, positive_output, negative_output], axis=-1)
  final = Lambda( 
              lambda vects: K.stack(vects, axis=1),
              name='final')([anchor_output, positive_output,negative_output])

  final_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=final)

  return final_model
  


def make_triplets():

  register_path = '../data/facerecog/ProcessedNamedFaces/'

  anchors   = {}
  positives   = {}
  negatives   = {}

  for entry in os.scandir(register_path):
    if entry.is_dir() and entry.name != 'General':
      name = entry.name 
      addedAnchor = False
      addedPos =False
      for img in os.scandir(entry.path):
        if not img.name.startswith('.') :
          if not addedAnchor:
            anchors[name] = img.path
            addedAnchor = True
          elif not addedPos:
            positives[name] = img.path
            addedPos = True
  prev = None
  First = None
  for k,v in anchors.items():
    if prev == None:
      First = k
    if prev != None:
      negatives[k] = anchors[prev]
    prev = k
  negatives[First] = anchors[k] 



  X_anchor = []
  X_positive =[]
  X_negative =[]
  y = []

  # Create the Input vectors
  for name, image in anchors.items():
    anchor  =  cv2.imread(image,1)
    positive  =  cv2.imread(positives[name],1)
    negative  = cv2.imread(negatives[name],1)
    X_anchor.append(anchor)
    X_positive.append(positive)
    X_negative.append(negative)

  X_anchor = np.asarray(X_anchor)
  X_positive = np.asarray(X_positive)
  X_negative = np.asarray(X_negative)

  y = np.ones((11,3,128))  
  print(y[:,0].shape)
  print(X_anchor.shape)
  return X_anchor, X_positive, X_negative, y

def build_model():

  final_model = build_siamese_model()
  final_model.compile(loss=triplet_loss, optimizer="adam")

   
  # Callbacks
  callbacks = [EarlyStopping(monitor='acc', min_delta=config_dict['early_stop_th'], patience=5, verbose=0, mode='auto'), 
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0), 
    CSVLogger(config_dict['log_path'], separator=',', append=False),
    ModelCheckpoint(config_dict['check_path'], monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]

  print(final_model.summary())

  X_anchor, X_positive, X_negative,y = make_triplets()
  final_model.fit([X_anchor, X_positive, X_negative], y, batch_size = 5, epochs = 25, callbacks=[CSVLogger('log.csv', separator=',', append=False)])
  final_model.save('../models/facerecog/facerecog_2.h5')

"""

    train_generator = adience_train_generator(config_dict['batch_size'])
    validation_generator = adience_eval_generator(config_dict['batch_size'])

    model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ["accuracy"])

    hist = model.fit_generator(train_generator, steps_per_epoch=config_dict['steps_per_epoch'],verbose = 2,callbacks = callbacks,
        epochs=config_dict['epochs'], validation_data=validation_generator,validation_steps=config_dict['validation_steps'])


    model.save_weights(config_dict['weights_path'])
    model.save(config_dict['model_path'])
"""


if __name__ == '__main__':

  build_model()
