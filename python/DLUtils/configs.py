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
    return {k:literal_eval(val) for k,val in dict(_config[section]).iteritems()}

