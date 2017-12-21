from keras.models import load_model
from keras.datasets import mnist 
from keras.utils import np_utils

class Evaluate():
	def __init__(self, model_path, x_test, y_test, batch_size):
		self.model_path = model_path
		self.x_test  = x_test
		self.y_test  = y_test 
		self.batch_size = batch_size
		self.model = None
		self.scores = None
		self.predictions = None
		self.__load_model()

	def __load_model(self):
		self.model = load_model(self.model_path)

	def __evaluate(self):
		self.scores = self.model.evaluate(self.x_test, self.y_test, batch_size = self.batch_size)
	
	def __predict(self):
		self.predictions = self.model.predict(self.x_test, batch_size=self.batch_size)

	def __cleanup(self):
		del self.model

	def process(self, eval = True, predict = True):
		self.__load_model()
		if eval:
			self.__evaluate()
			print("Score {}".format(self.scores[1]))

		if predict:
			self.__predict()
			print(self.predictions)

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






