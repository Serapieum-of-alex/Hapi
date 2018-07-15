# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:38:45 2018

@author: Mostafa
"""
from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


setup(
      name='Hapi',
      version='0.1.0',
      description='Distributed Hydrological model',
      long_description=readme,
      author='Mostafa Farrag',
      author_email='moah.farag@gmail.come',
      url='https://github.com/MAfarrag/HAPI',
      License=license ,
      packages=find_packages(exclude=('tests','docs')),
     )







