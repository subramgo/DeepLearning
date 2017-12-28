import numpy as np
from keras.models import load_model
from keras.datasets import mnist 
from keras.utils import np_utils
import cv2



class  GenderClassifier():
    gender_filter = {
              0:'Female' 
            , 1:'Male' 
            , 2:'Female' 
            , 3:'Male'
            , 4:'Female'
            , 5:'Male'
            , 6:'Female'
            , 7:'Male'
            , 8:'Female'
            , 9:'Male'
            ,10:'Female'
            ,11:'Male'
            ,12:'Female'
            ,13:'Male'
            ,14:'Female'
            ,15:'Male'
            }
    def __init__(self):
      self.model_path = '../cellar/gender_150x150.h5'
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
        self.batch_size = batch_size
        self.predictions = self.model.predict(self.x_test, batch_size=self.batch_size)

    def __cleanup(self):
        del self.model

    def set_model(self,new_model_path):
        self.__cleanup()
        self.model_path = model_path

    def input_preprocessing(self,image_arr):
        """ Preprocessing to match the training conditions for this model. 
        Apply resize, reshape, other scaling/whitening effects.
        x_test can be any image size greater than 100x100 and it will be resized
        """
        resized = cv2.resize(image_arr, (150, 150)) 
        resized = resized.reshape(1,150,150,3)
        return resized

    def process(self, x_test, y_test=None, batch_size=1):
        preprocessed = self.input_preprocessing(x_test)

        if y_test is not None:
            self.__evaluate(preprocessed, y_test,batch_size)
            print("Score {}".format(self.scores[1]))
            return None

        else:
            self.__predict(preprocessed, batch_size)
            print(self.predictions)
            idx = np.argmax(self.predictions)
            return self.gender_filter[idx]



class DemographicClassifier():
    """Given a trained face classification model, apply it to some data."""
    gender_filter = {
              0:'Female' 
            , 1:'Male' 
            , 2:'Female' 
            , 3:'Male'
            , 4:'Female'
            , 5:'Male'
            , 6:'Female'
            , 7:'Male'
            , 8:'Female'
            , 9:'Male'
            ,10:'Female'
            ,11:'Male'
            ,12:'Female'
            ,13:'Male'
            ,14:'Female'
            ,15:'Male'
            }

    age_gender_filter = {
              0:'f-(0, 2)' 
            , 1:'m-(0, 2)' 
            , 2:'f-(4, 6)' 
            , 3:'m-(4, 6)'
            , 4:'f-(8, 12)'
            , 5:'m-(8, 12)'
            , 6:'f-(15, 20)'
            , 7:'m-(15, 20)'
            , 8:'f-(25, 32)'
            , 9:'m-(25, 32)'
            ,10:'f-(38, 43)'
            ,11:'m-(38, 43)'
            ,12:'f-(48, 53)'
            ,13:'m-(48, 53)'
            ,14:'f-(60, 100)'
            ,15:'m-(60, 100)'
            }

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
        self.batch_size = batch_size

        self.predictions = self.model.predict(self.x_test, batch_size=self.batch_size)

    def __cleanup(self):
        del self.model

    def set_model(self,new_model_path):
        self.__cleanup()
        self.model_path = model_path

    def input_preprocessing(self,image_arr):
        """ Preprocessing to match the training conditions for this model. 
        Apply resize, reshape, other scaling/whitening effects.
        x_test can be any image size greater than 100x100 and it will be resized
        """
        resized = cv2.resize(image_arr, (100, 100)) 
        resized = resized.reshape(1,100,100,3)
        return resized

    def process(self, x_test, y_test=None, batch_size=1):
        preprocessed = self.input_preprocessing(x_test)

        if y_test is not None:
            self.__evaluate(preprocessed, y_test,batch_size)
            print("Score {}".format(self.scores[1]))
            return None

        else:
            self.__predict(resized, batch_size)
            print(self.predictions)
            idx = np.argmax(self.predictions)
            return self.gender_filter[idx]

    def mnist_demo(self):
      (x_train, y_train), (x_test, y_test) = mnist.load_data()

      # Reshape to add the channel
      # input image dimensions
      img_rows, img_cols = 28, 28

      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      y_test = np_utils.to_categorical(y_test, 10)

      print("testing MNIST data:")
      self.process(x_test,y_test,1)


if __name__ == '__main__':
  model_path = '../../cellar/gender_100.h5'
  obj = DemographicClassifier(model_path)
  obj.mnist_demo()





