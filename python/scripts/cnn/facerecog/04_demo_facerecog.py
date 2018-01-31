import pickle
import os
from keras.models import load_model
import glob
import cv2
import numpy as np


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

print("Load Registry")
registry = pickle.load(open('../models/facerecog/registry.pkl','rb'))
class_labels = pickle.load(open('../models/facerecog/class_labels.pkl','rb'))
print("Load Final Face Recog Model")
model = load_model('../models/facerecog/final_classifier.h5')
print("Loading Full Siamese Model....")
full_model = load_model('../models/facerecog/facerecog_2.h5',custom_objects={'triplet_loss': triplet_loss})
print("Extract inception model")
inception_model = full_model.get_layer('model_1')

test_path = os.path.expanduser('../data/facerecog/TestFaces/')
register_path = os.path.expanduser('../data/facerecog/ProcessedNamedFaces/')



# Make a dictionary of original faces
# to be used by the GUI
# to show the original user
original_faces = {}
for entry in os.scandir(register_path):
  if entry.is_dir():
    name = entry.name 
    for img in os.scandir(entry.path):
      original_faces[name] = img.path 
      break


# Face Recognition Test
file_list = glob.glob(test_path + '/*.jpg')
print(file_list)
for img in file_list:
  encoding = image_decode(img, inception_model)
  if encoding is None:
    print('Cant get encoding')
    continue

  # Predict the face
  ps = model.predict(encoding)
  p = np.argmax(ps, axis = 1)
  prob = ps[0][p]
  print(prob, class_labels[p[0]])
  if prob > 0.98:
    predicted_name = class_labels[p[0]]
  else:
    predicted_name = 'Others'


  g_image = cv2.imread(img)
  p_image = None
  if predicted_name != 'Others':
    p_image = cv2.imread(original_faces[predicted_name])
  else:
    p_image = cv2.imread(original_faces['General'])

  g_image = cv2.resize(g_image, (150,150))

  p_image = cv2.resize(p_image, (150,150))

  numpy_horizontal = np.hstack((g_image, p_image))
  cv2.imshow('Given            vs           Registry', numpy_horizontal)
  cv2.waitKey()
