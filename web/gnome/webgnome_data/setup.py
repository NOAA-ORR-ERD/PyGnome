"""
    Setup file.
"""

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.rst')) as f:
    README = f.read()

requires = [
            'waitress',
            'WebTest',
            'webhelpers2',
            'pyramid_redis_sessions',
            'cornice',
            ]


setup(name='webgnome_data',
      version=0.1,
      description='webgnome_data',
      long_description=README,
      classifiers=["Programming Language :: Python",
                   "Framework :: Pylons",
                   "Topic :: Internet :: WWW/HTTP",
                   "Topic :: Internet :: WWW/HTTP :: WSGI :: Application"
                   ],
      keywords="web services",
      author='',
      author_email='',
      url='',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=requires,
      tests_require=requires,
      test_suite='webgnome_data',
      entry_points="""\
      [paste.app_factory]
      main = webgnome_data:main
      """,
      paster_plugins=['pyramid'],
)
