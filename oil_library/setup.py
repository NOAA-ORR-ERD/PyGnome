#!/usr/bin/env python
import os
import glob
import shutil
from subprocess import call

from setuptools import setup, find_packages
from distutils.command.clean import clean

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.txt')).read()
pkg_name = 'OilLibrary'


class cleandev(clean):
    def run(self):
        clean.run(self)

        src = os.path.join(here, r'oil_library')
        to_rm = glob.glob(os.path.join(src, r'*.pyc'))
        to_rm.extend([os.path.join(here, '{0}.egg-info'.format(pkg_name)),
                      os.path.join(here, 'build'),
                      os.path.join(here, 'dist'),
                      os.path.join(src, 'OilLib.db')])
        for f in to_rm:
            try:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
            except:
                pass

            print "Deleting {0} ..".format(f)

requires = [
    'SQLAlchemy >= 0.9.1',
    'transaction',
    'zope.sqlalchemy',
    'awesome-slugify',
    'hazpy.unit_conversion',
    'pytest',
    'numpy'
    ]

s = setup(name=pkg_name,
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
          package_data={'oil_library': ['OilLib',
                                        'tests/*.py',
                                        'tests/sample_data/*']},
          cmdclass={'cleandev': cleandev},
          entry_points={
                  'console_scripts': [('initialize_OilLibrary_db = '
                                       'oil_library.initializedb:make_db'),
                                      ],
                        }
          )

# make database post install - couldn't call this directly so used
# console script
if 'install' in s.script_args:
    call("initialize_OilLibrary_db")
elif 'develop' in s.script_args:
    if os.path.exists(os.path.join(here, 'oil_library', 'OilLib.db')):
        print 'OilLibrary database exists - do not remake!'
    else:
        call("initialize_OilLibrary_db")
        print 'OilLibrary database successfully generated from file!'
