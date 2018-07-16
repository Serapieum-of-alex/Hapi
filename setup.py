# -*- coding: utf-8 -*-
"""
@author: Mostafa
"""
from setuptools import setup, find_packages

#with open('requirement.txt') as f:
#    Dependencies = f.readlines()

setup(
      name='Hapi',
      version='0.1.0',
      description='Distributed Hydrological model',
      author='Mostafa Farrag',
      author_email='moah.farag@gmail.come',
      url='https://github.com/MAfarrag/HAPI',
      License="MIT" ,
      #install_requires=Dependencies,
      packages=['Hapi'],
      zip_safe=False
     )
