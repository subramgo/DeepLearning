from keras.models import load_model
import os
import glob
import tensorflow as tf
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

test_path = '../data/facerecog/ProcessedNamedFaces/Andy/'

print("Loading Model....")
full_model = load_model('../models/facerecog/facerecog_2.h5',custom_objects={'triplet_loss': triplet_loss})
#model.compile(loss=identity_loss, optimizer="sgd")

print(full_model.summary())

inception_model = full_model.get_layer('model_1')
print(inception_model.summary())


file_list = glob.glob(test_path + '/*.jpg')

for img in file_list:
	image = cv2.imread(img, )
	resized = np.expand_dims(image, axis =0)
	vector = inception_model.predict_on_batch(resized)
	print(vector)
