from setuptools import setup, find_packages

# This setup is suitable for "python setup.py develop".  It does not
# install the scripts in the 'bin' directory.

setup(
    name = "Hazpy",
    version = "1.7",
    package_dir = {"": "lib"},
    packages = find_packages("lib"),
    scripts = ["scripts/Verdat2Poly.py",
               "scripts/Poly2Verdat.py",
               "scripts/thin_bna.py"
               ],
    entry_points = """\
[console_scripts]
ldap-lookup = hazpy.scripts.ldap_lookup:main
sitestats-summarize-month = hazpy.sitestats.summarize_month:main
sitestats-raw-referers = hazpy.sitestats.make_raw_referers:main
"""
    )

