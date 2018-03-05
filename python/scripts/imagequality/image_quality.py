"""
Given an image, give a list of attribute of the image to depict the quaility fo the images
Image quality attribtues are used from widerface detection benchmark dataset
http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

"""

import re
import argparse
from PIL import Image as pil_image
from random import shuffle
import numpy as np
import keras.backend as K
from keras.layers import Input, Conv2D, Dense,MaxPooling2D, Flatten, Activation,Dense, Dropout, BatchNormalization,GlobalAveragePooling2D
from keras.models import Model
from keras.backend import tf as ktf
from keras import optimizers
from keras.callbacks import History, EarlyStopping,ReduceLROnPlateau,CSVLogger,ModelCheckpoint
from keras.applications import ResNet50


TARGET_HEIGHT=416
TARGET_WIDTH=416
NO_CHANNELS=3
NB_CLASSES=14

#### Read the Label file in widerface format #############
class Quality():
	""" Class to hold the image quality parameters """
	def __init__(self):
		self.blur = None
		self.blur_vector = None
		self.blur_type = ['clear','normal','heavy']

		self.expression = None
		self.expression_vector =None
		self.expression_type = ['typical', 'exaggerate']

		self.illuminaiton = None
		self.illuminaiton_vector =None 
		self.illuminaiton_type=['normal','extreme']

		self.occulsion = None
		self.occulsion_vector = None 
		self.occulsion_type =['no','partial','heavy']

		self.pose = None
		self.pose_vector = None 
		self.pose_type=['typical', 'atypical']

		self.invalid = None 
		self.invalid_vector = None 
		self.invalid_type=['valid image','invalid image']

		self.x1 	= None
		self.y1 = None
		self.width = None
		self.height = None	

	def showme(self):
		print("######## Image Quality ########")
		print("Blur {} Expression {} Illuminaiton {} Occulsion {} Pose {} Invalid {}"
			.format(self.blur, self.expression, self.illuminaiton, self.occulsion, self.pose, self.invalid))

class Label():
	""" Class to hold the label details """
	def __init__(self):
		self.image_file       =  None
		self.complete_path = None
		self.class_type       =  None
		self.no_bbxs         =  None

		self.qual_metrix    =  []
		self.image_quality = None
		self.image_quality_no = None


	def showme(self):
		print("######## A single Label ##########")
		print("Image File :{}  Class:{} No_BBx:{}".format(self.image_file, self.class_type, self.no_bbxs))
		print("Image Quality Vector", self.image_quality)
		print("Image Quality No", self.image_quality_no)


	def _make_vector(self,size, indx):
		vector = np.zeros(size)
		vector[indx] = 1
		return vector

	def _image_quality(self):
		"""
		The data set has image quaility metric for each bounding box
		Using that we find a single image quality vector for the whole image
		We find the most frequent value in each column and use it 
		as the overall image quality metric
		"""
		q =[[q.blur, q.expression, q.illuminaiton, q.invalid, q.occulsion, q.pose] for q in self.qual_metrix]
		q = np.asarray(q, dtype=int)
		freq_q=np.apply_along_axis(lambda x: np.argmax(np.bincount(x)) , axis=0, arr=q)
		self.image_quality_no = freq_q
		final_array = np.zeros(14)

		# Blur
		final_array[0:3][freq_q[0]] = 1
		# Expression
		final_array[3:5][freq_q[1]] = 1
		# illumination
		final_array[5:7][freq_q[2]] =1
		# invalid
		final_array[7:9][freq_q[3]] = 1
		# occulsion
		final_array[9:12][freq_q[4]] =1
		#pose
		final_array[12:][freq_q[5]] = 1

		return final_array




	def parse(self, lines):
		for i, line in enumerate(lines):
			if i == 0:
				""" Image file"""
				self.image_file = line.rstrip()
				c = line.split("--")
				self.complete_path = c[1]
				self.class_type = int(c[0])


			elif i == 1:
				"Number of bounding boxes"
				self.no_bbxs = int(line)
			else:
				contents = line.split(' ')
				if len(contents) == 11:
					quality = Quality()
					quality.x1 = contents[0]
					quality.y1 = contents[1]
					quality.width = contents[2]
					quality.height = contents[3]
					
					quality.blur = int(contents[4])
					quality.blur_vector = self._make_vector(len(quality.blur_type), quality.blur)

					quality.expression = int(contents[5])
					quality.expression_vector = self._make_vector(len(quality.expression_type), quality.expression)

					quality.illuminaiton = int(contents[6])
					quality.illuminaiton_vector = self._make_vector(len(quality.illuminaiton_type), quality.illuminaiton)

					quality.occulsion = int(contents[8])
					quality.occulsion_vector = self._make_vector(len(quality.occulsion_type), quality.occulsion)

					quality.pose = int(contents[9])
					quality.pose_vector = self._make_vector(len(quality.pose_type), quality.pose)
					
					quality.invalid = int(contents[7])
					quality.invalid_vector = self._make_vector(len(quality.invalid_type), quality.invalid)

					self.qual_metrix.append(quality)
		self.image_quality = self._image_quality()


def read_labels(file_path):
	"""
	Create a list of Label Objects, reading label file
	"""
	with open(file_path,'r') as f:
		contents = f.readlines()

	p =r'^\d--*'
	patches = []
	labels = []

	for line in contents:
		m = re.match(p ,line)
		if m is not None and  len(m.group()) == 3:
			if len(patches) > 0:
				label = Label()
				label.parse(patches)
				patches = []
				labels.append(label)
		patches.append(line)
	return labels


def generator(img_root_dir, label_file, batch_size=4):
	"""
	Batch generates images and labels.
	"""
	labels = read_labels(label_file)
	path = img_root_dir + '/'

	while True:
		rnd_indx = np.random.randint(0, high=len(labels),size=batch_size)
		X = []
		y = []

		for i in rnd_indx:
			#labels[i].showme()
			image_path = labels[i].image_file
			image = pil_image.open(path + image_path)
			image = image.resize((TARGET_WIDTH,TARGET_HEIGHT))
			x = np.asarray(image, dtype=K.floatx())
			X.append(x)
			y.append(labels[i].image_quality)

		X = np.asarray(X).reshape((batch_size, TARGET_WIDTH, TARGET_HEIGHT, NO_CHANNELS))
		Y = np.asarray(y).reshape((batch_size,14))


		yield (X,Y)



def image_quality_model(input_shape, nb_classes):

    resnet_base = ResNet50(include_top = False, weights = 'imagenet',input_tensor = None, input_shape = input_shape ,pooling = 'max')

    for layer in resnet_base.layers:
    	layer.trainable = False
    
    x = Dense(1024, activation = "relu",name='dense-1')(resnet_base.output)
    x = Dense(512, activation = "relu",name='dense-2')(x)
    x = Dense(512, activation ="relu",name='dense-3')(x)
    x = Dense(256, activation ="relu",name='dense-4')(x)
    predictions = Dense(nb_classes, activation="softmax",name="softmax")(x)
    model = Model(inputs = resnet_base.input, outputs = predictions)

    return model


def build_model( traindir, trainlabel,modeldir,input_shape,epochs,batch_size):

	epochs = 10
	input_shape = (TARGET_WIDTH, TARGET_HEIGHT, NO_CHANNELS)
	model = image_quality_model(input_shape, NB_CLASSES)
	model.compile(optimizer = 'sgd', loss = "categorical_crossentropy", metrics = ["accuracy"])

	#for i in range(epochs):


	train_generator = generator(traindir, trainlabel, batch_size=4)
	hist = model.fit_generator(train_generator, steps_per_epoch=10, epochs=epochs)
	


	model.save_weights(modeldir + '/img_quality_weights.h5')
	model.save(modeldir + '/img_quality_model.h5')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-trainlabel", help="Widerface train label file", type=str, action="store", dest="trainlabel")
	parser.add_argument("-traindir", help="Widerface train image directory", type=str, action="store", dest="traindir")
	parser.add_argument('-modeldir',help="model directory",type=str,action="store",dest="modeldir")
	parser.add_argument('-epochs',help="Number of epochs",type=int,action="store",dest="epochs")
	parser.add_argument('-batchsize',help="batchsize",type=int,action="store",dest="batchsize")




	args = parser.parse_args()

	build_model(args.traindir, args.trainlabel,args.modeldir,(TARGET_WIDTH, TARGET_HEIGHT),epochs=args.epochs,batch_size=args.batchsize)

	#train_generator = generator(args.traindir, args.trainlabel, batch_size=4)

	#for i in range(10):
	#	x,y = next(train_generator)





	#main()

		
















