import argparse
import glob
import logging
import multiprocessing as mp
import os
import time

import cv2

from DLUtils.align_dlib import AlignDlib

logger = logging.getLogger(__name__)
align_dlib = AlignDlib()

input_dir = '../data/facerecog/TestFaces'
output_dir ='../data/facerecog/TestOutFaces/'
crop_dim = (96,96)

files = glob.glob(input_dir + '/*.jpg')

for filename in files:
    basename = os.path.basename(filename)
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    bb = align_dlib.getLargestFaceBoundingBox(image)
    aligned = align_dlib.align(crop_dim, image)
    if aligned is not None:
    	aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    	cv2.imwrite(output_dir + '/' + basename, aligned)




