import requests
from DLUtils import convert

model_id = "6i"
image_id = "35"

def get_prediction(model_id,image_id,img64,shape,dtype):
    """Get predictions from a rest backend for your input."""
    print("Requesting prediction for image\n")
    _path = "/demographics/{}/image/{}".format(model_id,image_id)


    res = requests.post(url="http://localhost:7171"+_path,
                        data=img64,
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


def _test():
    img64,shape,dtype = convert.file2base64('test.jpg')
    get_prediction(model_id,image_id,img64,shape,dtype)

