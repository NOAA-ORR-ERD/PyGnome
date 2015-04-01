'''
Save/load gnome objects
'''
import os
import shutil
import json
import zipfile
import logging

import gnome

log = logging.getLogger(__name__)


class References(object):
    '''
    PyGnome objects like the WindMover contain other objects, eg Wind object.
    When persisting a Model, the Wind object is not created by the WindMover,
    it is merely referenced by the WindMover. When persisting a Model, the
    referenced objects are saved in their own file and a reference is stored
    for it. This class manages these references.
    '''
    def __init__(self):
        self._refs = {}

    def __contains__(self, obj):
        if self.get_reference(obj):
            return True
        else:
            return False

    @property
    def files(self):
        return self._refs.keys()

    def get_reference(self, obj):
        '''
        return key if obj already exists in references list
        else return None
        '''
        for key, item in self._refs.iteritems():
            if item is obj:
                return key
        return None

    def _add_reference_with_name(self, obj, name):
        '''
        add an object reference specified by 'name'
        '''
        if self.retrieve(name):
            if self.retrieve(name) is not obj:
                raise ValueError('a different object is referenced by '
                    '{0}'.format(name))
        else:
            # make sure object doesn't already exist
            if self.get_reference(obj):
                raise ValueError('this object is already referenced by '
                    '{0}'.format(self.get_reference(obj)))

            else:
                self._refs[name] = obj

    def reference(self, obj, name=None):
        '''
        Get a unique reference to the object. By default this string is the
        filename in which the json for the object is stored
        If a reference to obj already exists, then it is returned

        :param obj: object for which a reference must be added
        :param name=None: add an object reference specified by 'name' for
            filename
        '''
        if name is not None:
            return self._add_reference_with_name(obj, name)

        key = self.get_reference(obj)
        if key is not None:
            return key

        key = "{0}_{1}.json".format(obj.__class__.__name__, len(self._refs))

        self._refs[key] = obj
        return key

    def retrieve(self, ref):
        '''
        retrieve the object associated with the reference
        '''
        try:
            return self._refs[ref]
        except KeyError:
            return None


def class_from_objtype(obj_type):
    '''
    object type must be a string in the gnome namespace:
        gnome.xxx.xxx
    '''
    if len(obj_type.split('.')) == 1:
        return

    try:
        # call getattr recursively
        obj = reduce(getattr, obj_type.split('.')[1:], gnome)
        return obj
    except AttributeError:
        log.warning("{0} is not part of gnome namespace".format(obj_type))


def load(saveloc, fname='Model.json', references=None):
    '''
    read json from file and load the appropriate object
    This is a general purpose load method that looks at the json['obj_type']
    and invokes json['obj_type'].loads(json) method

    :param saveloc: path to zipfile that contains data files and json files.
        It could also be a directory containing files - keep original support
        for location files.
    :param fname: .json file to load. Default is 'Model.json'. zipfile/dir
        must contain 'fname'
    :param references=None: References object that keeps track of objects
        in a dict as they are constructed, using the filename as the key

    :returns: object constructed from the json

    .. note:: Function first assumes saveloc is a directory and looks for
        saveloc/fname. If this fails, it checks if saveloc is a zipfile. If
        this fails, it checks if saveloc is a file and loads this assuming its
        json for a gnome object. If none of these work, it just returns None.
    '''
    # is a directory, look for Model.json in directory
    if os.path.isdir(saveloc):
        fd = open(os.path.join(saveloc, fname), 'r')

    elif zipfile.is_zipfile(saveloc):
        z = zipfile.ZipFile(saveloc)
        fd = z.open(fname, 'r')

    elif os.path.isfile(saveloc):
        fd = open(saveloc, 'r')
        saveloc, fname = os.path.split(saveloc)

    else:
        # nothing to do, saveloc is not zipfile or a directory
        return

    # load json data from file descriptor
    json_data = json.loads("".join([l.rstrip('\n') for l in fd]))
    fd.close()

    # create a reference to the object being loaded
    cls = class_from_objtype(json_data.pop('obj_type'))
    obj = cls.loads(json_data, saveloc, references)

    if obj is None:
        # object failed to load - look in log messages for clues
        return

    # after loading, add the object to references
    if references:
        references.reference(obj, fname)
    return obj


'''
Define general purpose functions for checking and rejecting bad zipfiles
'''


class Savable(object):
    '''
    Create a mixin for save/load options for saving and loading serialized
    gnome objects. Mix this in with the Serializable class so all gnome objects
    can save/load themselves
    '''
    _allowzip64 = False

    def _ref_in_saveloc(self, saveloc, ref):
        '''
        returns true if reference is found in saveloc, false otherwise
        '''
        if zipfile.is_zipfile(saveloc):
            with zipfile.ZipFile(saveloc, 'r') as z:
                if ref in z.namelist():
                    return True
                else:
                    return False
        else:
            return os.path.exists(os.path.join(saveloc, ref))

    def _update_and_save_refs(self, json_, saveloc, references):
        '''
        for attributes that are stored as references - ensure the references
        are moved to saveloc, and update its value in the json_
        '''
        for field in self._state:
            if (field.save_reference and
                getattr(self, field.name) is not None):
                '''
                attribute is stored as a reference
                json_ will not contain a key for the referenced objects
                Add a key with the reference
                '''
                obj = getattr(self, field.name)
                ref = references.reference(obj)
                json_[field.name] = ref
                if not self._ref_in_saveloc(saveloc, ref):
                    obj.save(saveloc, references, name=ref)
        return json_

    def _json_to_saveloc(self,
                         json_,
                         saveloc,
                         references=None,
                         name=None):
        '''
        save json_ to saveloc

        -. first save and update references if any attributes are stored as
            references
        -. add self to references
        -. move any datafiles to saveloc that need to be persisted
        -. finally, write json_ directly to zip if saveloc is a zip or dump
            to *.json of saveloc is a directory

        :param json_: json data after serialization. Default is the output of
            self.serialize('save')
        :param saveloc: Either zipfile to which object's json can be appended.
            Or a valid path where object's *.json can be dumped to a file
        :param references: References object to keep track of objects that
            have been constructed. If a referenced object has been saved,
            no need to add it again to saveloc - don't want multiple copies
        :param name: json is stored in 'name.json' in zip archive/specified
            directory. Default is self.__class__.__name__. If references object
            contains self.__class__.__name__, then let
        '''
        references = (references, References())[references is None]
        json_ = self._update_and_save_refs(json_, saveloc, references)

        f_name = \
            (name, '{0}.json'.format(self.__class__.__name__))[name is None]

        # add yourself to references
        try:
            references.reference(self, f_name)
        except ValueError:
            # f_name already assigned to an object, so let References() assign
            # a different filename
            f_name = references.reference(self)

        # move datafiles to saveloc
        json_ = self._move_data_file(saveloc, json_)
        if zipfile.is_zipfile(saveloc):
            self._write_to_zip(saveloc, f_name, json.dumps(json_, indent=True))
        else:
            # make last leaf of save location if it doesn't exist
            self._write_to_file(saveloc, f_name, json_)

        return references

    def _write_to_file(self, saveloc, f_name, json_):
        full_name = os.path.join(saveloc, f_name)
        with open(full_name, 'w') as outfile:
            json.dump(json_, outfile, indent=True)

    def _write_to_zip(self, saveloc, f_name, s_data):
        '''
        general function for writing string data directly to zipfile.

        f_name is the archive name and s_data is the corresponding string,
        added to zipfile
        '''
        with zipfile.ZipFile(saveloc, 'a',
                             compression=zipfile.ZIP_DEFLATED,
                             allowZip64=self._allowzip64) as z:
            z.writestr(f_name, s_data)

    def save(self, saveloc, references=None, name=None):
        """
        save object state as json to user specified saveloc

        :param saveloc: zip archive or a valid directory. Model files are
            either persisted here or a new model is re-created from the files
            stored here. The files are clobbered when save() is called.
        :type saveloc: A path as a string or unicode
        :param name=None: filename to store json. If None, default name is:
            "{0}.json".format(self.__class__.__name__). If saveloc is zipfile,
            this is the name of archive in which json for self is stored.
        :type name: str
        :param references: dict of references mapping 'id' to a string used for
            the reference. The value could be a unique integer or it could be
            a filename. It is upto the creator of the reference list to decide
            how to reference a nested object.
        """

        json_ = self.serialize('save')
        return self._json_to_saveloc(json_, saveloc, references=references,
                                     name=name)

    def _move_data_file(self, saveloc, json_):
        """
        Look at _state attribute of object. Find all fields with 'isdatafile'
        attribute as True. If there is a key in json_ corresponding with
        'name' of the fields with True 'isdatafile' attribute then move that
        datafile and update the key in the json_ to point to new location
        """
        fields = self._state.get_field_by_attribute('isdatafile')

        for field in fields:
            if field.name not in json_:
                continue

            # data filename
            d_fname = os.path.split(json_[field.name])[1]

            if zipfile.is_zipfile(saveloc):
                # add datafile to zip archive
                with zipfile.ZipFile(saveloc, 'a',
                                     compression=zipfile.ZIP_DEFLATED,
                                     allowZip64=self._allowzip64) as z:
                    if d_fname not in z.namelist():
                        z.write(json_[field.name], d_fname)
            else:
                # move datafile to saveloc
                if json_[field.name] != os.path.join(saveloc, d_fname):
                    shutil.copy(json_[field.name], saveloc)

            # always want to update the reference so it is relative to saveloc
            json_[field.name] = d_fname

        return json_

    @classmethod
    def _load_refs(cls, json_data, saveloc, references):
        '''
        loads up references. First looks for object in references, if not found
        then it creates object from json and adds a reference to it. If found
        in references, it just uses/returns the object
        '''
        ref_fields = cls._state.get_field_by_attribute('save_reference')

        # pop references from json_data, create objects for them
        ref_dict = {}
        if ref_fields:
            for field in ref_fields:
                if field.name in json_data:
                    i_ref = json_data.pop(field.name)
                    ref_obj = references.retrieve(i_ref)
                    if not ref_obj:
                        ref_obj = load(saveloc, i_ref, references)

                    ref_dict[field.name] = ref_obj

        return ref_dict

    @classmethod
    def _update_datafile_path(cls, json_data, saveloc):
        '''
        update path to attributes that use a datafile
        if saveloc is a zipfile, then extract the datafile to same location
        as zipfile and upate path in json_data.
        '''
        datafiles = cls._state.get_field_by_attribute('isdatafile')
        if len(datafiles) == 0:
            return True

        iszip = False

        if zipfile.is_zipfile(saveloc):
            iszip = True
            z = zipfile.ZipFile(saveloc, 'r')
            saveloc = os.path.split(saveloc)[0]

        # fix datafiles path from relative to absolute so we can load datafiles
        for field in datafiles:
            if field.name in json_data:
                # ZipCheck: path must only be defined relative to saveloc
                # currently, all datafiles stored at same level in saveloc,
                # no subdirectories.
                # Also, the datafiles are extracted to saveloc/.
                # For zip files coming from the web, is_savezip_valid() tests
                # filenames in archive do not contain paths with '..'
                # In here, we just extract datafile to saveloc/.
                if iszip:
                    z.extract(json_data[field.name], saveloc)

                json_data[field.name] = os.path.join(saveloc,
                                                     json_data[field.name])
        if iszip:
            z.close()

        return True

    @classmethod
    def loads(cls, json_data, saveloc=None, references=None):
        '''
        loads object from json_data

        - load json for references from files
        - update paths of datafiles if needed
        - deserialize json_data
        - and create object with new_from_dict()

        json_data: dict containing json data. It has been parsed through the
            json.loads() command. The json will be valided here when it gets
            deserialized. Its references and datafile paths will be recreated
            here prior to calling new_from_dict()

        Optional parameter

        :param saveloc: location of data files or *.json files for objects
            stored as references. If object requires no datafiles and does not
            need to read references from a *.json file in saveloc, then this
            can be None.
        :param references: references object - if this is called by the Model,
            it will pass a references object. It is not required.

        :returns: object constructed from json_data.
        '''
        references = (references, References())[references is None]
        ref_dict = cls._load_refs(json_data, saveloc, references)
        sucess = cls._update_datafile_path(json_data, saveloc)
        if not sucess:
            return None

        # deserialize after removing references
        _to_dict = cls.deserialize(json_data)

        if ref_dict:
            _to_dict.update(ref_dict)

        obj = cls.new_from_dict(_to_dict)

        return obj


# max json filesize is 1MegaByte
# max compression ratio: uncompressed/compressed = 3
_max_json_filesize = 1024 * 1024
_max_compress_ratio = 3


def is_savezip_valid(savezip):
    '''
    some basic checks on validity of zipfile. Primarily for checking save
    zipfiles loaded from the Web. Following are the types of errors it checks:

    :returns: True if zip is valid, False otherwise

    1. Failed to open zipfile
    2. CRC failed for a file in the archive - rejecting zip
    3. Found a *.json with size > _max_json_filesize - rejecting
    4. Reject - found a file with:
        uncompressed_size/compressed_size > _max_compress_ratio.
    5. Found a file in archive that has path outside of saveloc - rejecting
        rejects zipfile if it contains an archive with '..'

    .. note:: can change _max_json_filesize, _max_compress_ratio if required.
    '''
    if not zipfile.is_zipfile(savezip):
        log.warning("{0} is not a valid zipfile".format(savezip))
        return False

    with zipfile.ZipFile(savezip, 'r') as z:
        # 1) Failed to open zipfile
        try:
            badfile = z.testzip()
        except:
            msg = "Failed to open or run testzip() on {0}".format(savezip)
            log.warning(msg)
            return False

        # 2) CRC failed for a file in the archive - rejecting zip
        if badfile is not None:
            # log the bad zipfile and return False
            log.warning("{0} is corrupt. rejecting zipfile".format(badfile))
            return False

        for zi in z.filelist:
            if (os.path.splitext(zi.filename)[1] == '.json' and
                zi.file_size > _max_json_filesize):
                # 3) Found a *.json with size > _max_json_filesize. Rejecting.
                msg = ("Filesize of {0} is {1}. It must be less than "
                       "_max_json_filesize: {2}. "
                       "Rejecting zipfile.".format(zi.filename, zi.file_size,
                                                   _max_json_filesize))
                log.warning(msg)
                return False

        # integer division - it will floor
        if zi.file_size/zi.compress_size > _max_compress_ratio:
            # 4) Found a file with
            #    uncompressed_size/compressed_size > _max_compress_ratio.
            #    Rejecting.
            msg = ("uncompressed filesize is {0} time compressed filesize."
                   "_max_compress_ratio must be less than {1}. "
                   "Rejecting zipfile".format(zi.file_size/zi.compress_size,
                                              _max_compress_ratio))
            log.warning(msg)
            return False

        if '..' in zi.filename:
            # 5) reject zipfile if it contains .. anywhere in filename.
            #    currently, all datafiles stored at same level in saveloc,
            #    no subdirectories. Even if we start using subdirectories,
            #    there should never be a need to do '..'
            msg = ("Found '..' in {0}. Rejecting zipfile".format(zi.filename))
            log.warning(msg)
            return False

    # all checks pass - so we can load zipfile
    return True
