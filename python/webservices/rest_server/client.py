import requests
import json
from DLUtils import convert

model_id = "6i"
image_id = "35"
url_path = "/demographics/{}/image/{}".format(model_id,image_id)
full_url = "http://10.38.4.84:9001"+url_path
headers={'Content-Type': 'application/octet-stream'}

def get_prediction(model_id,image_id,json_payload):
    """Get predictions from a rest backend for your input."""
    print("Requesting prediction for image\n")
    url_path = "/demographics/{}/image/{}".format(model_id,image_id)


    res = requests.post(url="http://localhost:7171"+url_path,
                        data=json_payload,
                        headers={'Content-Type': 'application/octet-stream'})
    
    print(res.status_code, res.reason)
    resp = res.json()
    return resp

def _adience():
    hdf5_path = '../../data/Adience/hdf5/adience.h5'
    batch_size = 4
    dimensions = (batch_size, 256, 256,3)

    vals = adience_datagenerator(hdf5_path, batch_size, dimensions)
    x_test, y_test = vals.next()


def send_image(image_path = './test.jpg'):
    json_payload = convert.file2json(image_path)
    get_prediction(model_id,image_id,json_payload)


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
