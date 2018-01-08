import configparser
import csv
import os.path
from ast import literal_eval

my_path = os.path.abspath(os.path.dirname(__file__))
config_path = '../settings/models.ini'
path = os.path.join(my_path, config_path)

_config = configparser.ConfigParser()
_config.read(path)


def get_section_dict(section):
    """ Return dictionary with literal_eval values of the given section """
    return {k:literal_eval(val) for k,val in dict(_config[section]).items()}

def get_configs(section):
    return get_section_dict(section)
