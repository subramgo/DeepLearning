import io
import os
import glob
from time import sleep
import requests



gender_url = "http://10.38.12.63:5000/gender"
face_url = "http://10.38.12.63:5000/facerecog"

#gender_url = "http://0.0.0.0:5000/gender"
#face_url = "http://0.0.0.0:5000/facerecog"

headers={'Content-Type': 'application/octet-stream'}

imagePath = './data/facerecog/bocacafe/20180201/*.jpg'


imageList = glob.glob(imagePath)
for file_name in imageList:
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()


        response = requests.post(url=gender_url, data = content, json = None, headers = headers)
        gender = response.json()['gender']
        response = requests.post(url=face_url, data = content, json = None, headers = headers)

        name = response.json()['person']

        print(file_name, gender, name)




