import numpy as np
from tensorflow import Graph
import tensorflow as tf
from keras.models import load_model
from keras.datasets import mnist 
from keras.utils import np_utils
import cv2
from DLUtils.configs import get_configs
import pickle



class FaceRecog:

  def __init__(self):

    self.registry = pickle.load(open('../models/facerecog/registry.pkl','rb'))
    self.class_labels = pickle.load(open('../models/facerecog/class_labels.pkl','rb'))


    self.model = load_model('../models/facerecog/final_classifier.h5')

    self.frmodel = load_model('../models/facerecog/facerecog_1.h5')

    self.image_w = 96
    self.image_h = 96
    print("Loaded facerecog model")

  def predict(self, img):
    encoding = self.image_decode(img)
    ps = self.model.predict(encoding)
    p = np.argmax(ps, axis = 1)
    prob = ps[0][p]
    if prob > 0.90:
      predicted_name = self.class_labels[p[0]]
    else:
      predicted_name = 'Unknown'
    return predicted_name

  def image_decode(self,image):

    image = self.centeredCrop(image, self.image_w, self.image_h)
    image = image[...,::-1]
    image = np.around(np.transpose(image, (0,1,2))/255.0, decimals=12)
    resized = np.expand_dims(image, axis =0)
    vector = self.frmodel.predict_on_batch(resized)
    return vector

  def centeredCrop(self,img, new_height, new_width):

     width =  np.size(img,1)
     height =  np.size(img,0)

     left = np.ceil((width - new_width)/2.)
     top = np.ceil((height - new_height)/2.)
     right = np.floor((width + new_width)/2.)
     bottom = np.floor((height + new_height)/2.)


     top = int(top)
     bottom = int(bottom)
     left = int(left)
     right = int(right)
     cImg = img[top:bottom, left:right]
     #cImg = img.crop((left, top, right, bottom))
     return cImg

class AgeClassifier:
    # age class labels
    _filter = {'(0, 2)' :0, '(4, 6)':1 , '(8, 12)':2, '(15, 20)':3,'(25, 32)':4,'(38, 43)':5, '(48, 53)':6, '(60, 100)':7}

    def __init__(self):
        self.config_dict = get_configs('age')
        self.x_test = None
        self.preprocessed = None
        self.y_test = None
        self.batch_size = 1
        self.model = None
        self.scores = None
        self.predictions = None
        target_size = self.config_dict['target_size']
        self.image_w = target_size[0]
        self.image_h = target_size[1]


        self.__load_model()

    def __load_model(self):
        self.model = load_model(self.config_dict['model_path'])

    def __evaluate(self):
        self.scores = self.model.evaluate(self.preprocessed, self.y_test, batch_size = self.batch_size)
    
    def __predict(self):

        self.predictions = self.model.predict(self.preprocessed, batch_size=self.batch_size)

    def __cleanup(self):
        del self.model


    def input_preprocessing(self):
        """ Preprocessing to match the training conditions for this model. 
        Apply resize, reshape, other scaling/whitening effects.
        x_test can be any image size greater than 100x100 and it will be resized
        """
        resized = cv2.resize(self.x_test, (self.image_w, self.image_h)) 
        self.preprocessed = resized.reshape(1,self.image_w,self.image_h,3)

    def process(self, x_test, y_test , batch_size):
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.input_preprocessing()

        if y_test is not None:
            self.__evaluate()
            #print("Score {}".format(self.scores[1]))
            return None

        else:
            self.__predict()
            #print(self.predictions)
            idx = np.argmax(self.predictions)
            return self._filter[idx]


class  GenderClassifier():
    gender_filter = {0:'male',1:'female'}


    def __init__(self):
        self.config_dict = get_configs('gender')
        self.x_test = None
        self.preprocessed = None
        self.y_test = None
        self.batch_size = 1
        self.model = None
        self.scores = None
        self.predictions = None
        target_size = self.config_dict['target_size']
        self.image_w = target_size[0]
        self.image_h = target_size[1]
        self.gender_session = None
        self.gender_graph = Graph()
        self.__load_model()

    def __load_model(self):
      self.model = load_model(self.config_dict['model_path'])
      print("Gender Model Loaded")

    def __evaluate(self):
        self.scores = self.model.evaluate(self.preprocessed, self.y_test, batch_size = self.batch_size)
    
    def __predict(self):
        print(self.preprocessed.shape)
        #with self.gender_session.as_default():
        self.predictions = self.model.predict(self.preprocessed, batch_size=self.batch_size)

    def __cleanup(self):
        del self.model


    def input_preprocessing(self):
        """ Preprocessing to match the training conditions for this model. 
        Apply resize, reshape, other scaling/whitening effects.
        x_test can be any image size greater than 100x100 and it will be resized
        """
        resized = cv2.resize(self.x_test, (self.image_w, self.image_h)) 
        self.preprocessed = resized.reshape(1,self.image_w,self.image_h,3)



    def process(self, x_test, y_test , batch_size):
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.input_preprocessing()

        if y_test is not None:
            self.__evaluate()
            #print("Score {}".format(self.scores[1]))
            return None

        else:
            self.__predict()
            #print(self.predictions)
            idx = np.argmax(self.predictions)

            return self.gender_filter[idx]

    def process1(self, x_test, y_test , batch_size):
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.input_preprocessing()

        if y_test is not None:
            self.__evaluate()
            #print("Score {}".format(self.scores[1]))
            return None

        else:
            self.__predict()
            #print(self.predictions)
            idx = np.argmax(self.predictions)

            return self.gender_filter[idx],self.predictions

class DemographicClassifier():
    """Given a trained face classification model, apply it to some data."""
    age_filter = {'(0, 2)' :0, '(4, 6)':1 , '(8, 12)':2, '(15, 20)':3,'(25, 32)':4,'(38, 43)':5, '(48, 53)':6, '(60, 100)':7}

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


#if __name__ == '__main__':
    #model_path = '../../cellar/gender_100.h5'
    #obj = DemographicClassifier(model_path)
    #obj.mnist_demo()
    #del obj





