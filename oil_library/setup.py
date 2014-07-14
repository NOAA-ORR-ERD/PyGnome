#!/usr/bin/env python
import sys, os
from subprocess import call

from setuptools import setup, find_packages
from setuptools.command.install import install

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.txt')).read()


if "cleanall" in "".join(sys.argv[1:]):
    print "Deleting files .."
    os.system(r'find . -name \*.pyc -exec rm -v {} \;')
    os.system('rm -rv OilLibrary.egg-info')
    os.system('rm -v OilLibrary.db')
    sys.argv[1] = 'clean'   # this is what distutils understands

requires = [
    'SQLAlchemy',
    'transaction',
    'zope.sqlalchemy',
    ]

s=setup(name='OilLibrary',
      version='0.0',
      description='OilLibrary',
      long_description=README, 
      author='ADIOS/GNOME team at NOAA ORR',
      author_email='orr.gnome@noaa.gov',
      url='',
      keywords='adios weathering oilspill modeling',
      packages=find_packages(),
      include_package_data=True,
      install_requires=requires,
      tests_require=requires,
      package_data={'oil_library': ['OilLib']},
      entry_points = {
              'console_scripts': [
                'initialize_OilLibrary_db = oil_library.initializedb:make_db',
                      ],
              }
      )

# make database post install - couldn't call this directly so used
# console script
#===================
#import oil_library
#inst_loc = os.path.dirname(oil_library.__file__)
#if not os.path.exists(os.path.join(inst_loc, 'OilLib.db')):
#===================
call("initialize_OilLibrary_db")
print 'OilLibrary database successfully generated from file!'
