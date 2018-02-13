import numpy as np
import tensorflow as tf
import random as rn



############# Begin code for reproducable results ##############

"""
https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
Steps to obtain reproducable results.
"""

import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(12345)

rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

############# End code for reproducable results ##############

from keras.models import load_model
import glob
import cv2 
import pickle
from keras.layers import  Input,Dense,Dropout
from keras.models import Model


register_path = '../data/facerecog/ProcessedNamedFaces/'


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



def image_decode(filepath, model):

  # Subdetectors are located in https://github.com/davisking/dlib/blob/master/dlib/image_processing/frontal_face_detector.h
  image = cv2.imread(filepath, )
  resized = np.expand_dims(image, axis =0)
  vector = model.predict_on_batch(resized)
  return vector


def load_facerecog_model():
  print("Loading Model....")
  full_model = load_model('../models/facerecog/facerecog_2.h5',custom_objects={'triplet_loss': triplet_loss})
  #model.compile(loss=identity_loss, optimizer="sgd")

  #print(full_model.summary())

  print("Extract inception model")
  inception_model = full_model.get_layer('model_1')
  #print(inception_model.summary())
  return inception_model

def make_registry(register_path, model):
  register_dict = {}
  for entry in os.scandir(register_path):
    if entry.is_dir() and entry.name != 'General' and entry.name != 'Others':
      name = entry.name 
      for img in os.scandir(entry.path):
        if not img.name.startswith('.') :
          person_vector = image_decode(img.path, model)
          if person_vector is not None:
            if name in register_dict:
              register_dict[name].append(person_vector)
            else:
              register_dict[name] = []
              register_dict[name].append(person_vector)
  pickle.dump(register_dict, open('../models/facerecog/registry.pkl','wb'))
  return register_dict

if __name__ == '__main__':
  model = load_facerecog_model()
  make_registry(register_path, model)
