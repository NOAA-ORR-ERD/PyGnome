from pathlib import Path
import os
import sys
import pytest
import glob
import tempfile
import contextlib
import json
from gnome.utilities.save_updater import (extract_zipfile,
                                          update_savefile,
                                          remember_cwd,
                                          v0tov1,
                                          v1tov2,
                                          v3tov4,
                                          )

'''
Tests the save file update system.

Consists of two parts:
1. Generic fixes that get applied in the 'extract_zipfile' function
2. Update patches that update the extracted files from one version to another

Because this system involves many file manipulations, zip extraction, file creation/deletion, etc
it is very important that the provided context managers are used.
'''

mac_zips = all_saves = pure_saves = all_names = None
pth = os.path.dirname(__file__)
all_saves = glob.glob(os.path.join(pth, '*.zip')) + glob.glob(os.path.join(pth, '*.gnome'))
all_names = [os.path.basename(s) for s in all_saves]
pure_saves = [f for f in all_saves if 'mac' not in os.path.basename(f)]
mac_saves =  [f for f in all_saves if 'mac' in os.path.basename(f)]


@contextlib.contextmanager
def setup_workspace(savename):
    '''
    extracts a save file into a temporary folder and moves the cwd to that
    folder.
    '''
    tempdir = tempfile.mkdtemp()
    with remember_cwd(os.path.dirname(__file__)):
        extract_zipfile(savename, tempdir)
    curdir= os.getcwd()
    os.chdir(tempdir)
    try:
        yield tempdir
    finally:
        os.chdir(curdir)


def check_files(func):
    files = glob.glob('*.json')
    for f in files:
        with open(f, 'r') as fp:
            json_ = json.load(fp)
            found = func(json_)
            if found:
                return found
    return False


def test_extract_zipfile():
    # note test changed -- we shouldn't have filenames
    # that are illegal on Windows anywhere -- it could be a zip file
    # that gets moved between platforms.
    #Ensure the *GENERIC DIESEL.json gets renamed as a file and as a reference
    with setup_workspace('v1_double_diesel.gnome'):
        files = glob.glob('*.json')
        assert "GENERIC DIESEL.json" in files
        assert check_files(
            lambda js: 'spill.Spill' in js['obj_type']
                       and js.get('substance', None) == "GENERIC DIESEL.json"
        )

    # Ensure that the __MACOSX folder is ignored
    # And the zip structure is flat
    with setup_workspace('v0_diesel_mac.zip') :
        files = os.listdir('.')
        assert len(list(filter(lambda f: '__MACOSX' in f, files))) == 0
        assert 'Model.json' in files


# Should automate this in future
def test_v0_to_v1():
    with setup_workspace('v0_diesel_mac.zip'):
        errs = []
        msgs = []
        v0tov1(msgs, errs)
        assert len(errs) == 0
        assert check_files(
            lambda js: 'gnome.spills.substance.GnomeOil' in js['obj_type'] and
                js.get('name', None) == "*GENERIC DIESEL"
        )


def test_v1_to_v2():
    """
    fixme: this is only checking 'gnome.spills.substance.GnomeOil'
    not any others, including: gnome.spills.spill.Spill

    check_files should probably be updated to do more than one check
    """
    with setup_workspace('v0_diesel_mac.zip'):
        errs = []
        msgs = []
        # fixme: we should probably have some v1 data to use for tests!
        # have to run v0tov1 first for this old data
        v0tov1(msgs, errs)
        v1tov2(msgs, errs)
        assert len(errs) == 0

        def checker(js):
            return ('gnome.spills.substance.GnomeOil' in js['obj_type']
                    and js.get('name', None) == "*GENERIC DIESEL")

        assert check_files(checker)


def test_v3_to_v4():
    with setup_workspace('v0_diesel_mac.zip'):
        errs = []
        msgs = []
        # fixme: we should probably have some v1 data to use for tests!
        # have to run v0tov1 first for this old data
        v0tov1(msgs, errs)
        v1tov2(msgs, errs)
        v3tov4(msgs, errs)
        assert len(errs) == 0

        def checker(js):
            return ('gnome.weatherers.weathering_data.WeatheringData' not in js['obj_type'])

        assert check_files(checker)