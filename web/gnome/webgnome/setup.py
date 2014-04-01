import sys
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.txt')).read()
CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()

if "cleanall" in "".join(sys.argv[1:]):
    print "Deleting files .."
    os.system(r'find . -name \*.pyc -exec rm -v {} \;')
    os.system('rm -rv webgnome.egg-info')
    os.system('rm -v OilLibrary.db')
    sys.argv[1] = 'clean'   # this is what distutils understands

requires = [
    'pyramid',
    'SQLAlchemy',
    'transaction',
    'pyramid_tm',
    'pyramid_debugtoolbar',
    'zope.sqlalchemy',
    'waitress'
]

setup(name='webgnome',
      version='0.0',
      description='webgnome',
      long_description=README + '\n\n' + CHANGES,
      classifiers=[
        "Programming Language :: Python",
        "Framework :: Pyramid",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Internet :: WWW/HTTP :: WSGI :: Application",
        ],
      author='',
      author_email='',
      url='',
      keywords='web pyramid pylons',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=requires,
      tests_require=requires,
      test_suite="webgnome",
      entry_points="""\
      [paste.app_factory]
      main = webgnome:main
      [console_scripts]
      initialize_webgnome_db = webgnome.scripts.initialize_oil_db:main
      create_location_file = webgnome.scripts.create_location_file:main
      clean_model_images = webgnome.scripts.clean_model_images:main
      clean_file_uploads = webgnome.scripts.clean_file_uploads:main
      """,
      )
