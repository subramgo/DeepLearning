"""
    Default config file is located in "settings/models.ini"
    Can load another one using `load_configs(...)`
    Absolute paths are checked relative to the package location.
      e.g. '/data/stuff' will try '../../data/stuff'
"""

import configparser as _cfgparser
import os as _os
from ast import literal_eval as _literal_eval

def resolve_paths(path):
    """ Check for paths relative to the package installation.
        If given path begins with '/', search in '.' and '..' """
    my_path = _os.path.abspath(_os.path.dirname(__file__))
        
    if _os.path.isabs(path):
        _test = _os.path.join(my_path,'../..',path[1:])
        if _os.path.exists(_os.path.abspath(_test)):
            path = _os.path.abspath(_test)

        _test = _os.path.join(my_path,'..',path[1:])
        if _os.path.exists(_os.path.abspath(_test)):
            path = _os.path.abspath(_test)

    return path     

class Config:
    def __init__(self):
        self.my_path = _os.path.abspath(_os.path.dirname(__file__))
        self.config_path = resolve_paths('/settings/models.ini')

        path = _os.path.join(self.my_path, self.config_path)
        self.load_configs(path)

    def __call__(self,kwargs):
        return self.get_section_dict(kwargs)

    def load_configs(self,filepath):
        """ Return dictionary for the given section and filepath """
        self._config = _cfgparser.ConfigParser()
        self._config.read(filepath)

    def get_section_dict(self,section=None):
        """ Return dictionary with literal_eval values of the given section """
        if not section:
            section = self._config.sections()[0]
            print("loading section '{}'".format(section))

        return {k:self._eval(val) for k,val in dict(self._config[section]).items()}

    def _eval(self,value):
        """ Parse into Python objects using literal_eval. Resolve paths. """
        value = _literal_eval(value)
        try:
            basestring
        except NameError:
            basestring = str
        if isinstance(value,basestring):
            value = resolve_paths(value)
        return value

def get_configs(section_name = None):
    """ shorthand default way to grab configs and support other code """
    if section_name:
        return Config()(section_name)
    else:
        return Config()(kwargs={})
