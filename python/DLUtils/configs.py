"""
    Default config file is located in "settings/models.ini"
    Can load another one using `load_configs(...)`
"""

import configparser
import csv
import os.path
from ast import literal_eval

class Config:

    def __init__(self):
        self.my_path = os.path.abspath(os.path.dirname(__file__))
        self.config_path = '../settings/models.ini'
        
        path = os.path.join(self.my_path, self.config_path)
        self.load_configs(path)

    def load_configs(self,filepath):
        """ Return dictionary for the given section and filepath """
        self._config = configparser.ConfigParser()
        self._config.read(filepath)

    def get_section_dict(self,section=None):
        """ Return dictionary with literal_eval values of the given section """
        if not section:
            section = self._config.sections()[0]
            print("loading section '{}'".format(section))

        return {k:self._eval(val) for k,val in dict(self._config[section]).iteritems()}

    def _resolve_paths(self,path):
        """ check for paths relative to the package installation """
        path_to_data = '../../data/'
        if path.startswith('data:'):
            path = path[5:]
            path = os.path.join(self.my_path,path_to_data,path.strip('/'))
            path = os.path.abspath(path)
        elif os.path.isabs(path):
            """ check if path is absolute from default data location """
            _test = os.path.join(self.my_path,'../..',path[1:])
            if os.path.exists(os.path.abspath(_test)):
                path = os.path.abspath(_test)

        return path

    def _eval(self,value):
        """ Parse into Python objects using literal_eval. Resolve paths. """
        value = literal_eval(value)
        if isinstance(value,basestring):
            value = self._resolve_paths(value)
        return value

