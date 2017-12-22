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
    shape = np_array.shape
    return base64.encodestring(np_array),shape

def base642array(img64,shape):
    """convert base-64 encoded image to a numpy array"""
    return np.frombuffer(base64.decodestring(img64))

np_array = file2array('test.jpg')
enc,shape = array2base64(np_array)
dec = base642array(enc,shape)

# Problem:
# enc != dec 
