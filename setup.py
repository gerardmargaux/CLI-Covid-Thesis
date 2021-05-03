"""
Setup script for the Covid prediction model
=========================================
Call from command line as::
    python setup.py --help
to see the options available.
"""
from setuptools import setup
from pkg_resources import parse_requirements


with open('requirements.txt') as f:
    install_requires = []
    for req in parse_requirements(f.read()):
        install_requires.append(str(req).replace('==', '>='))

kwargs = {'install_requires':  install_requires}

setup(**kwargs)