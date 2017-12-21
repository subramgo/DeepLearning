import numpy as np
from keras.models import load_model
from keras.datasets import mnist 
from keras.utils import np_utils


gender_filter = {
	0:'Female' 
	,1:'Male' 
          ,2:'Female' 
          ,3:'Male'
           , 4:'Female'
           , 5:'Male'
           , 6:'Female'
           ,7:'Male'
           ,8:'Female'
           ,9:'Male'
           ,10:'Female'
           , 11:'Male'
           ,12:'Female'
           ,13:'Male'
           , 14:'Female'
           ,15:'Male'
           }

age_gender_filter = {
	0:'f-(0, 2)' 
	,1:'m-(0, 2)' 
          ,2:'f-(4, 6)' 
          ,3:'m-(4, 6)'
           , 4:'f-(8, 12)'
           , 5:'m-(8, 12)'
           , 6:'f-(15, 20)'
           ,7:'m-(15, 20)'
           ,8:'f-(25, 32)'
           ,9:'m-(25, 32)'
           ,10:'f-(38, 43)'
           , 11:'m-(38, 43)'
           ,12:'f-(48, 53)'
           ,13:'m-(48, 53)'
           , 14:'f-(60, 100)'
           ,15:'m-(60, 100)'
           }

class Evaluate():
	def __init__(self, model_path):
		self.model_path = model_path
		self.model = None
		self.scores = None
		self.predictions = None
		self.__load_model()

	def __load_model(self):
		self.model = load_model(self.model_path)

	def __evaluate(self, x_test, y_test, batch_size):
		self.x_test  = x_test
		self.y_test  = y_test 
		self.batch_size = batch_size

		self.scores = self.model.evaluate(self.x_test, self.y_test, batch_size = self.batch_size)
	
	def __predict(self, x_test, batch_size):
		self.x_test  = x_test
		self.y_test  = y_test 
		self.batch_size = batch_size

		self.predictions = self.model.predict(self.x_test, batch_size=self.batch_size)

	def __cleanup(self):
		del self.model

	def process(self, x_test, y_test, batch_size, eval = True, predict = True):
		if eval:
			self.__evaluate(x_test, y_test,batch_size)
			print("Score {}".format(self.scores[1]))
			return None

		if predict:
			self.__predict(x_test, batch_size)
			print(self.predictions)
			idx = np.argmax(self.predictions)
			return gender_filter[idx]

		self.__cleanup()




if __name__ == '__main__':
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Reshape to add the channel
	# input image dimensions
	img_rows, img_cols = 28, 28

	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	y_test = np_utils.to_categorical(y_test, 10)

	model_path = '../../models/lenet.h5'
	obj = Evaluate(model_path, x_test, y_test, 1)
	obj.process()






