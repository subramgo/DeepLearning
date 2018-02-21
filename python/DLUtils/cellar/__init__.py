### DLUtils.cellar

# import all submodules which are cellar models
from . import yolo

# imports used here
import requests as _requests
from keras.models import load_model as _load_model
from .. import configs as _configs

def load_h5(cellar_path):
    """ return hdf5 model loaded from cellar path """
    if not cellar_path[0] == '/':
        cellar_path = '/'+cellar+path

    config = _configs.Config()
    return _load_model(config.resolve_paths(cellar_path))



def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = _requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)
