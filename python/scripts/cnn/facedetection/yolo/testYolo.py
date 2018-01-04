from keras.models import load_model
from keras.engine import InputLayer
import cv2
import sys
import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import keras.backend as K
from DLUtils.configs import get_configs
from DLUtils.DataGenerator import face_12net_train_generator, face_12net_eval_generator
import tensorflow as tf

K.set_image_data_format('channels_last')


model_path = sys.argv[1]


# Load the model
model = load_model(model_path)

#Get an input image from commandline
image_path = sys.argv[2]

image_w = int(sys.argv[3])
image_h = int(sys.argv[4])

class_dict = {0:'aeroplane',
1:'bicycle',
2:'bird',
3:'boat',
4:'bottle',
5:'bus',
6:'car',
7:'cat',
8:'chair',
9:'cow',
10:'diningtable',
11:'dog',
12:'horse',
13:'motorbike',
14:'person',
15:'pottedplant',
16:'sheep',
17:'sofa',
18:'train',
19:'tvmonitor'
}




def simple_test(image_path):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	height = image.shape[1]
	width = image.shape[0]
	image = cv2.resize(image, (image_w,image_h))
	image = image.reshape((1,image_w,image_h,3))
	prediction = model.predict(image, batch_size=1)
	print(prediction.shape)
	# 1, 13, 13, 125
	# Reshape it to 1,13,13,5,25
	# 5 anchor boxes at every grid in 13 x 13
	# 25 elements of reach anchorbox
	# probabiliity if an object is present, bx, by, w, h, 20 dim vector for each class
	p_resh = prediction.reshape(1, 13, 13, 5, 25)
	print(p_resh.shape)

	for box_i in range(5):
		box = p_resh[0][0][0][box_i]
		pc   = box[0]
		c_scores = box[5:]
		res = pc * c_scores
		idx = np.argmax(res)
		p = class_dict[idx]
		print("Box No {} score {} box {},{},{},{} class {} ".format(box_i, res[idx], box[1],box[2],box[3],box[4], p))

	box_confidence = p_resh[:,:,:,:,0]
	box_confidence = box_confidence.reshape(1,13,13,5,1)
	boxes = p_resh[:,:,:,:,1:5]
	boxes = boxes.reshape(1,13,13,5,4)
	box_class_prob = p_resh[:,:,:,:,5:]
	box_class_prob = box_class_prob.reshape(1,13,13,5,20)

	# Filter the boxes
	threshold = 0.6
	box_scores = np.multiply(box_confidence, box_class_prob)
	print(box_scores.shape)
	box_class = K.argmax(box_scores, axis =-1)
	box_class_scores = K.max(box_scores, axis=-1)
	# Filtering mask
	filtering_mask = K.greater_equal(box_class_scores, threshold)
	with K.get_session() as test:
		scores = tf.boolean_mask(box_class_scores, filtering_mask).eval()
		boxes = tf.boolean_mask(boxes, filtering_mask).eval()
		classes = tf.boolean_mask(box_class, filtering_mask).eval()
	
		print(boxes.shape)
		print(classes.shape)
		print(scores.shape)


		max_boxes = 5
		iou_threshold = 0.6


		max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
		test.run(tf.variables_initializer([max_boxes_tensor]))# initialize variable max_boxes_tensor
	# Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep


		nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
		scores = K.gather(scores, nms_indices).eval()
		boxes = K.gather(boxes, nms_indices).eval()
		classes = K.gather(classes, nms_indices).eval()

		print(boxes.shape)
		print(classes.shape)
		print(scores.shape)

		# scale the boxes
		image_dims = K.stack([height, width, height, width])
		image_dims = K.reshape(image_dims, [1, 4])
		boxes = boxes * image_dims

		print(boxes.eval())
#	image = cv2.imread(image_path, cv2.IMREAD_COLOR)

#	image = cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
#	cv2.imshow("image",image)
#	cv2.waitKey(0)




if __name__ == "__main__":
	simple_test(image_path)
