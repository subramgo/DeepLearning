'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='agegender_model',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='age gender model',
      author='Gopi Subramanian',
      author_email='gopi.Subramanian@gmail.com',
      license='MIT',
      install_requires=[
         'tensorflow',
          'keras',
          'h5py'],
      zip_safe=False)