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

import pickle
from keras.layers import  Input,Dense,Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers

def face_reco_classifier():

  from sklearn.preprocessing import label_binarize
  from sklearn import svm
  from sklearn.multiclass import OneVsRestClassifier
  from sklearn.metrics import confusion_matrix

  registry = pickle.load(open('../models/facerecog/registry.pkl','rb'))
  X = []
  Y = []

  for key, value in registry.items():
    for v in value:
      X.append(v[0])
      Y.append(key)
  class_labels = list(set(Y))
  print(class_labels)

  Y_labels = label_binarize(Y, classes = class_labels)
  X = np.array(X)
  Y_labels = np.array(Y_labels)

  print(X.shape)

  x_input = Input((128,))
  
  """ 0 validation accuracy
  x = Dense(100, activation='relu', name='fc0')(x_input)
  x = Dense(90, activation='relu', name='fc01')(x)
  x = Dropout(0.5)(x)
  x = Dense(80, activation='relu', name='fc1')(x)
  x = Dense(60, activation='relu', name='fc2')(x)
  x = Dropout(0.4)(x)
  x = Dense(50, activation='relu', name='fc3')(x)
  x = Dropout(0.1)(x)

  x = Dense(40, activation='relu', name='fc4')(x)
  x = Dropout(0.3)(x)
  x = Dense(30, activation='relu', name='fc5')(x)
  x = Dense(20, activation='relu', name='fc6')(x)
  x = Dropout(0.1)(x)
  x = Dense(len(class_labels), activation='softmax', name='predictions')(x)
  """
  x = Dense(128, activation='relu',name = 'fc0')(x_input)
  x = Dropout(0.5)(x)
  x = Dense(64, activation='relu',name = 'fc00')(x)
  x = Dense(32, activation='relu', name = 'fc1')(x)
  x = Dense(16, activation='relu', name = 'fc2')(x)
  x = Dense(len(class_labels), activation='softmax', name='predictions')(x)

  
  model = Model(input=x_input, output=x)
  model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
  model.fit(X, Y_labels, epochs=250, batch_size=5, validation_split =0.1, shuffle=False)
  scores = model.evaluate(X, Y_labels)
  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

  model.save('../models/facerecog/final_classifier.h5')
  pickle.dump(class_labels, open('../models/facerecog/class_labels.pkl','wb'))



if __name__ == '__main__':
  #model = load_facerecog_model()
  #make_registry(register_path, model)
  face_reco_classifier()


