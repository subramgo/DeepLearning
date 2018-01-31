from keras.models import load_model
import os
import glob
import tensorflow as tf
import cv2 
import numpy as np
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
    if entry.is_dir() and entry.name != 'General':
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


def face_reco_classifier():

  from sklearn.preprocessing import label_binarize
  from sklearn import svm
  from sklearn.multiclass import OneVsRestClassifier
  from sklearn.metrics import confusion_matrix

  np.random.seed(7)
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
  x = Dense(100, activation='relu', name='fc0')(x_input)
  x = Dense(90, activation='relu', name='fc01')(x)

  x = Dense(80, activation='relu', name='fc1')(x)
  x = Dense(60, activation='relu', name='fc2')(x)
  x = Dense(50, activation='relu', name='fc3')(x)

  x = Dense(40, activation='relu', name='fc4')(x)
  x = Dense(30, activation='relu', name='fc5')(x)

  x = Dropout(0.1)(x)
  x = Dense(len(class_labels), activation='softmax', name='predictions')(x)
  
  model = Model(input=x_input, output=x)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(X, Y_labels, epochs=2500, batch_size=25)
  scores = model.evaluate(X, Y_labels)
  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




  model.save('../models/facerecog/final_classifier.h5')

  pickle.dump(class_labels, open('../models/facerecog/class_labels.pkl','wb'))



if __name__ == '__main__':
  model = load_facerecog_model()
  make_registry(register_path, model)
  face_reco_classifier()


