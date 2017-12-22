import cv2
import base64
import numpy as np

# encode all as 
unit_dtype = np.int64

def file2base64(filepath):
    return array2base64(file2array(filepath))

def file2array(filepath):
    with open(filepath, "rb") as image_file:
        return cv2.imread(filepath, cv2.IMREAD_COLOR)

def base642file(img64,filepath):
    print('TODO')

def array2file(np_array,filepath):
    print('TODO')

def array2base64(np_array):
    """convert a numpy array to a base-64 encoded image"""
    # save array shape, flatten array, convert to base64
    shape = np_array.shape
    dtype = np_array.dtype
    return base64.b64encode(np_array.flatten()),shape,dtype

def base642array(img64,shape,dtype):
    """convert base-64 encoded image to a numpy array"""
    return np.frombuffer(base64.decodestring(img64),dtype=dtype).reshape(shape)


############################################################
############################################################


def _demo():
  arimg = file2array('test.jpg')
  enc,shape,dtype = array2base64(arimg)
  dec = base642array(enc,shape,dtype)
  if (arimg == dec).all():
    print("numpy array <-> base64 conversion parity successful")
  else:
    print("It's broken")

