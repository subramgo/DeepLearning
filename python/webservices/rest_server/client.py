import requests
import json
from DLUtils import convert

full_url = "http://10.38.4.84:5000/gender"
headers={'Content-Type': 'application/octet-stream'}

def post_image_json(image_path = './test.jpg'):
    json_payload = convert.file2json(image_path)
    response = requests.post(url=full_url, data=json_payload, headers=headers)    
    return response.json()


def post_image(image_path):
    img = open(image_path, 'rb').read()
    response = requests.post(url=full_url, data = img, json = None, headers = headers)
    return response.json() 


if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:
        res = post_image(sys.argv[1])
        print(res)
        print(res['gender'])
    else:
        post_image()
