# Updates a save loaded using pygnome.Model to the latest version

import json
import logging
import glob
import sys
import contextlib
import os
import re
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

errortypes = [
    'Save file version not compatible with this updater. Version: {0} Updater: {1}',
    'Save file does not have a version.txt',
    'Failed to remove old file: {0} Error: {1}',
]

NON_WEATHERING_DICT = {
 "obj_type": "gnome.spill.substance.NonWeatheringSubstance",
 "id": "dummy-id-from-save_updater",
 "name": "NonWeatheringSubstance_99",
 "initializers": [
  "InitWindages_99.json"
 ],
 "is_weatherable": False,
 "standard_density": 1000.0
}

# if needed:
# WINDAGE_INIT_DICT = {
#  "windage_range": [
#   0.01,
#   0.04
#  ],
#  "windage_persist": 900,
#  "obj_type": "gnome.spill.initializers.InitWindages",
#  "id": "1d05bcd6-fa73-11e9-a559-0242ac120006",
#  "name": "InitWindages_99"
# }

@contextlib.contextmanager
def remember_cwd(new_wd):
    curdir = os.getcwd()
    os.chdir(new_wd)
    try:
        yield
    finally:
        os.chdir(curdir)


def update_savefile(save_directory):

    if not isinstance(save_directory, str) or not os.path.isdir(save_directory):
        raise ValueError('Must unzip save to directory in order to upgrade it to '
                         'the latest version')

    messages = []
    errors = []

    with remember_cwd(save_directory):
        # get current save file version
        allfiles = glob.glob('*')
        if 'version.txt' in allfiles:
            with open('version.txt') as fp:
                v = int(fp.readline())
        else:
            v = 0

        for i in range(v, len(all_update_steps)):
            # execute update
            step = all_update_steps[i]
            messages, errors = step(messages, errors)

            if len(errors) > 0:
                for e in errors:
                    sys.stderr.write(e+'\n')
                raise ValueError('Errors occurred during save update process')

        if len(messages) > 0:
            for m in messages:
                log.info(m)
        return True


def v0tov1(messages, errors):
    '''
    Takes a zipfile containing no version.txt and up-converts it to 'version 1'.
    This functions purpose is to upgrade save files to maintain compatibility
    after the SpillRefactor upgrades.
    '''
    def Substance_from_ElementType(et_json, water):
        '''
        Takes element type cstruct with a substance, creates an appropriate
        GnomeOil cstruct
        '''
        inits = et_json.get('initializers', [])
        for init in inits:
            if isinstance(init, dict):
                init['obj_type'] = init['obj_type'].replace('.elements.', '.')
        if 'substance' not in et_json:
            '''
            Note the id of the new cstructs. The ID IS required at this stage, because
            the load process will use it later to establish references between objects
            '''
            substance = NON_WEATHERING_DICT
            substance["initializers"] = inits
        else:
            substance = {
                "obj_type": "gnome.spills.substance.GnomeOil",
                "name": et_json.get('substance', 'Unknown Oil'),
                "initializers": et_json.get('initializers', []),
                "is_weatherable": True,
                "water": water,
                "id": "v0-v1-update-id-1"
            }
            if isinstance(et_json.get('substance', None), dict):
                substance.update(et_json.get('substance'))

        return substance

    jsonfiles = glob.glob('*.json')

    log.debug('updating save file from v0 to v1 (Spill Refactor)')
    water_json = element_type_json = None
    spills = []
    inits = []
    for fname in jsonfiles:
        with open(fname, 'r') as fn:
            json_ = json.load(fn)
            if 'obj_type' in json_:
                if ('Water' in json_['obj_type']
                        and 'environment' in json_['obj_type']
                        and water_json is None):
                    water_json = (fname, json_)

                if ('element_type.ElementType' in json_['obj_type']
                        and element_type_json is None):
                    element_type_json = (fname, json_)

                if 'gnome.spills.spill.Spill' in json_['obj_type']:
                    spills.append((fname, json_))

                if 'initializers' in json_['obj_type']:
                    inits.append((fname, json_))

    # Generate new substance object
    if water_json is None:
        water_json = (None, None)

    substance = None
    if element_type_json is not None:
        substance = Substance_from_ElementType(element_type_json[1],
                                               water_json[1])
        substance_fn = sanitize_filename(substance['name'] + '.json')
        # Delete .json for deprecated objects (element_type)
        fn = element_type_json[0]
        try:
            os.remove(fn)
        except Exception as e:
            err = errortypes[2].format(fn, e)
            errors.append(err)
            return messages, errors

    # Write modified and new files
    if substance is not None:
        with open(substance_fn, 'w') as subs_file:
            json.dump(substance, subs_file, indent=True)
    for spill in spills:
        fn, sp = spill
        del sp['element_type']
        sp['substance'] = substance_fn
        with open(fn, 'w') as fp:
            json.dump(sp, fp, indent=True)
    for init in inits:
        fn, init = init
        init['obj_type'] = init['obj_type'].replace('.elements.', '.')
        with open(fn, 'w') as fp:
            json.dump(init, fp, indent=True)
    with open('version.txt', 'w') as vers_file:
        vers_file.write('1')

    messages.append('**Update from v0 to v1 successful**')
    return messages, errors


def v1tov2(messages, errors):
    '''
    Takes a zipfile containing version 1 and up-converts it
    to 'version 2'.

    This function's purpose is to upgrade save files to maintain compatibility
    after the grand renaming:
    [link to commit here]
    '''
    log.debug('updating save file from v1 to v2 (Renaming)')

    jsonfiles = glob.glob('*.json')

    files_to_remove = []
    # GnomeOil update
    oils = []
    for fname in jsonfiles:
        with open(fname, 'r') as fn:
            json_ = json.load(fn)
            if 'obj_type' in json_:
                if json_['obj_type'] == "gnome.spill.substance.GnomeOil":
                    oils.append(fname)
                    # See if it can be used with the current GnomeOil
                    try:
                        GnomeOil(**json_)
                    except Exception:
                        # can't be used with GnomeOIl: replace with NonWeathering
                        log.info(f"Oil: {json_['name']} is not longer valid\n"
                                 "You will need re-load an oil, which can be "
                                 "obtained from The ADIOS Oil Database:\n"
                                 "https://adios.orr.noaa.gov/")
                        nws = NON_WEATHERING_DICT
                        # this will catch the windages info
                        nws['initializers'] = json_['initializers']
                        # write out the new file
                        with open(fname, 'w') as fn:
                            json.dump(nws, fn)


    # updating the name of spills
    spills = []  # things with a "gnome.spill" in the path
    for fname in jsonfiles:
        with open(fname, 'r') as fn:
            json_ = json.load(fn)
            if 'obj_type' in json_:
                if 'gnome.spill.' in json_['obj_type']:
                    spills.append((fname, json_))
    for fn, sp in spills:
        sp['obj_type'] = sp['obj_type'].replace('gnome.spill.', 'gnome.spills.')
        with open(fn, 'w') as fp:
            json.dump(sp, fp, indent=True)

    movers = []  # current_movers, GridCurrentMover
    for fname in jsonfiles:
        with open(fname, 'r') as fn:
            json_ = json.load(fn)
            if 'obj_type' in json_:
                if 'gnome.movers.current_movers.' in json_['obj_type']:
                    movers.append((fname, json_))
    for fn, mv in movers:
        mv['obj_type'] = mv['obj_type'].replace('current_movers.',
                                                'c_current_movers.')
        mv['obj_type'] = mv['obj_type'].replace('GridCurrentMover',
                                                'c_GridCurrentMover')
        with open(fn, 'w') as fp:
            json.dump(mv, fp, indent=True)

    # remove InitWindages
    for fname in jsonfiles:
        with open(fname, 'r') as fn:
            json_ = json.load(fn)
            if 'obj_type' in json_ and 'initializers' in json_:
                # this is assuming only one
                for wind_init in json_.pop('initializers'):
                    init_js = json.load(open(wind_init, 'r'))
                    if "InitWindages" in init_js["obj_type"]:
                        json_['windage_range'] = init_js['windage_range']
                        json_['windage_persist'] = init_js['windage_persist']
                        json.dump(json_, open(fname, 'w'))
                        files_to_remove.append(wind_init)

    with open('version.txt', 'w') as vers_file:
        vers_file.write('2')

    for fname in files_to_remove:
        Path(fname).unlink()

    messages.append('**Update from v1 to v2 successful**')
    return messages, errors


def extract_zipfile(zip_file, to_folder='.'):
    def work(zf):
        folders = [name for name in zf.namelist()
                   if name.endswith('/') and not name.startswith('__MACOSX')]
        prefix = None
        if len(folders) == 1:
            # we allow our model content to be in a single top-level folder
            prefix = folders[0]

        fn_edits = {}
        for name in zf.namelist():
            if (prefix and name.find(prefix) != 0) or name.endswith('/'):
                # ignores the __MACOSX files
                pass
            else:
                orig = os.path.basename(name)
                fn = sanitize_filename(orig)
                if orig != fn:
                    log.info('Invalid filename found: {0}'.format(orig))
                    fn_edits[orig] = fn

                target = os.path.join(to_folder, fn)
                with open(target, 'wb') as f:
                    f.write(zf.read(name))
        if len(fn_edits) > 0:
            log.info('Save file contained invalid names. '
                     'Editing extracted json to maintain save file integrity.')
            for jsonfile in glob.glob(os.path.join(to_folder, '*.json')):
                # if any file name edits were made, references may need to be updated too
                # otherwise the .json file won't be found
                contents = None
                replaced = False
                with open(jsonfile, 'r') as jf:
                    contents = jf.read()
                    for k, v in fn_edits.items():
                        if k in contents:
                            contents = contents.replace(k, v)
                            replaced = True
                if replaced:
                    with open(jsonfile, 'w') as jf:
                        jf.write(contents)

    if isinstance(zip_file, zipfile.ZipFile):
        work(zip_file)
    else:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            work(zf)


def sanitize_filename(fname):
    '''
    make filename legal on all systems (windows is pickier)
    '''
    return re.sub(r'[\\\\/*?:"<>|]', "", fname)



# note these should be indexed by version number
all_update_steps = [v0tov1, v1tov2]
