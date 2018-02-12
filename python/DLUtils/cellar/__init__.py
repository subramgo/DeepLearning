import h5py
from keras.models import load_model

from DLUtils import configs

def load_h5(cellar_path):
    """ return hdf5 model loaded from cellar path """
    if not cellar_path[0] == '/':
        cellar_path = '/'+cellar+path

    config = configs.Config()
    return load_model(config.resolve_paths(cellar_path))

if __name__ == '__main__':
    config = configs.Config()
    _filepath = config.get_section_dict('gender_resnet')['h5_path']

