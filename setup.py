#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='src',
      version='0.0.1',
      description='implementation of ppo',
      author='',
      author_email='',
     install_requires=[
            'torch'
      ],
      packages=find_packages()
      )

