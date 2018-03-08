# imports used here
from keras.models import load_model as _load_model
from .. import configs as _configs

# import all submodules which are cellar models
from . import yolo

def load_h5(cellar_path):
    """ return hdf5 model loaded from cellar path """
    if not cellar_path[0] == '/':
        cellar_path = '/'+cellar+path

    config = _configs.Config()
    return _load_model(config.resolve_paths(cellar_path))

if __name__ == '__main__':
    config = _configs.Config()
    _filepath = config.get_section_dict('gender_resnet')['h5_path']
