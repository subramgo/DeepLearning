import configparser
import csv
import os.path
from ast import literal_eval

my_path = os.path.abspath(os.path.dirname(__file__))
config_path = '../settings/models.ini'
path = os.path.join(my_path, config_path)

def get_configs(section):
    Config = configparser.ConfigParser()
    Config.read(path)
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = literal_eval(Config.get(section, option))
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    print (dict1)
    return dict1