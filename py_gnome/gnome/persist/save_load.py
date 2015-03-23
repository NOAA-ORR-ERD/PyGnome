'''
Save/load gnome objects
'''
import os
import shutil
import json
import zipfile

import gnome    # used in eval to 'load' gnome object from json


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

        key = "{0}_{1}.json".format(obj.__class__.__name__,
                len(self._refs))

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


def load(fname, references=None):
    '''
    read json from file and load the appropriate object
    This is a general purpose load method that looks at the json['obj_type']
    and invokes json['obj_type'].loads(json) method

    :param fname: .json file to load. If this is a directory, then look for a
        default name of 'Model.json' to load
    :param references=None:
    '''
    # is a directory, look for Model.json in directory
    if os.path.isdir(fname):
        fname = os.path.join(fname, 'Model.json')

    with open(fname, 'r') as fd:
        json_data = json.loads("".join([l.rstrip('\n') for l in fd]))
        #json_data = json.load(infile)

    # create a reference to the object being loaded
    saveloc, name = os.path.split(fname)
    obj_type = json_data.pop('obj_type')
    obj = eval(obj_type).loads(json_data, saveloc, references)

    # after loading, add the object to references
    if references:
        references.reference(obj, name)
    return obj


class Savable(object):
    '''
    Create a mixin for save/load options for saving and loading serialized
    gnome objects. Mix this in with the Serializable class so all gnome objects
    can save/load themselves
    '''
    def _json_to_saveloc(self,
                         json_,
                         saveloc,
                         references=None,
                         name=None):
        '''
        break up save into two steps so child classes can modify json if
        desired before invoking this method
        '''

        references = (references, References())[references is None]
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
                if not os.path.exists(os.path.join(saveloc, ref)):
                    obj.save(saveloc, references, name=ref)

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
            self._write_to_zip(saveloc, json_)
        else:
            # make last leaf of save location if it doesn't exist
            #self._make_saveloc(saveloc)
            self._write_to_file(os.path.join(saveloc, f_name), json_)

        return references

    def _write_to_file(self, full_name, json_):
        with open(full_name, 'w') as outfile:
            json.dump(json_, outfile, indent=True)

    def _write_to_zip(self):
        pass

    def _make_saveloc(self, saveloc):
        '''
        Create the last leaf in saveloc path if it doesn't exist
        '''
        path_, savedir = os.path.split(saveloc)
        if path_ == '':
            path_ = '.'

        if not os.path.exists(path_):
            raise ValueError('"{0}" does not exist. \nCannot create "{1}"'
                             .format(path_, savedir))

        if not os.path.exists(saveloc):
            os.mkdir(saveloc)

    def save(self, saveloc, references=None, name=None):
        """
        save object serialized to json format to user specified saveloc

        :param saveloc: A valid directory. Model files are either persisted
            here or a new model is re-created from the files stored here.
            The files are clobbered when save() is called.
        :type saveloc: A path as a string or unicode
        :param name=None: filename to store json. If None, default name is:
            "{0}.json".format(self.__class__.__name__)
        :type name: str
        :param references: dict of references mapping 'id' to a string used for
            the reference. The value could be a unique integer or it could be
            a filename. It is upto the creator of the reference list to decide
            how a reference a nested object.
        """

        json_ = self.serialize('save')
        return self._json_to_saveloc(json_, saveloc, references=references,
                                     name=name)

    def _move_data_file(self, saveloc, json_):
        """
        Look at _state attribute of object. Find all fields with 'isdatafile'
        attribute as True. If there is a key in to_json corresponding with
        'name' of the fields with True 'isdatafile' attribute then move that
        datafile and update the key in the to_json to point to new location

        TODO: maybe this belongs in serializable base class? Revisit this
        """
        fields = self._state.get_field_by_attribute('isdatafile')

        for field in fields:
            if field.name not in json_:
                continue

            # data filename
            d_fname = os.path.split(json_[field.name])[1]

            if zipfile.is_zipfile(saveloc):
                # add datafile to zip archive
                with zipfile.ZipFile(saveloc, 'a') as z:
                    if d_fname not in z.namelist():
                        z.write(d_fname)
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
                        ref_filename = os.path.join(saveloc, i_ref)
                        ref_obj = load(ref_filename, references)

                    ref_dict[field.name] = ref_obj

        return ref_dict

    @classmethod
    def _update_datafile_path(cls, json_data, saveloc):
        'update path to attributes that use a datafile'
        datafiles = cls._state.get_field_by_attribute('isdatafile')
        # fix datafiles path from relative to absolute so we can load datafiles
        for field in datafiles:
            if field.name in json_data:
                json_data[field.name] = os.path.join(saveloc,
                                                     json_data[field.name])

    @classmethod
    def loads(cls, json_data, saveloc=None, references=None):
        '''
        loads object from json_data

        - load json for references from files
        - update paths of datafiles if needed
        - deserialize json_data
        - and create object with new_from_dict()

        Optional parameter

        :param saveloc: location of data files or *.json files for objects
            stored as references. If object requires no datafiles and does not
            need to read references from a *.json file in saveloc, then this
            can be None.
        :param references: references object - if this is called by the Model,
            it will pass a references object. It is not required.
        '''
        references = (references, References())[references is None]
        ref_dict = cls._load_refs(json_data, saveloc, references)
        cls._update_datafile_path(json_data, saveloc)

        # deserialize after removing references
        _to_dict = cls.deserialize(json_data)

        if ref_dict:
            _to_dict.update(ref_dict)

        obj = cls.new_from_dict(_to_dict)

        return obj
