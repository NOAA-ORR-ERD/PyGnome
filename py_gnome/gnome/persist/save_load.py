'''
Save/load gnome objects
'''
import os
import shutil
import json


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

    def reference(self, obj):
        '''
        Get a unique reference to the object. By default this string is the
        filename in which the json for the object is stored
        '''
        # return key if obj already exists in references list
        for key, item in self._refs.iteritems():
            if item is obj:
                return key

        ref = "{0}_{1}.json".format(obj.__class__.__name__, len(self._refs))
        self._refs[ref] = obj
        return ref

    def retrieve(self, ref):
        '''
        retrieve the object associated with the reference
        '''
        try:
            return self._refs[ref]
        except KeyError:
            return None


def load(fname, references):
    '''
    read json from file and load the appropriate object
    This is a general purpose load method that look at the json['obj_type']
    and invokes json['obj_type'].load(json) method
    '''
    with open(fname, 'r') as infile:
        json_data = json.load(infile)

    # create a reference to the object being loaded
    obj_type = json_data.pop('obj_type')
    to_eval = ('{0}.load(saveloc, json_data, references)'.format(obj_type))
    obj = eval(to_eval)
    references[fname] = obj
    return obj


class Savable(object):
    '''
    Create a mixin for save/load options for saving and loading serialized
    gnome objects. Mix this in with the Serializable class so all gnome objects
    can save/load themselves
    '''
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
        if references is None:
            refs = References()

        for field in self._state:
            if (field.save_reference and
                getattr(self, field.name) is not None):
                '''
                attribute is stored as a reference
                json_ will not contain a key for the referenced objects
                Add a key with the reference
                '''
                obj = getattr(self, field.name)
                ref = refs.reference(obj)
                json_[field.name] = ref
                obj.save(saveloc, references=references, name=ref)

        f_name = \
            (name, '{0}.json'.format(self.__class__.__name__))[name is None]
        self._save_json_to_file(saveloc, json_, f_name)

    def _save_json_to_file(self, saveloc, data, name):
        """
        write json data to file

        :param dict data: JSON data to be saved
        :param obj: gnome object corresponding w/ data
        """

        fname = os.path.join(saveloc, name)
        data = self._move_data_file(saveloc, data)  # if there is a

        with open(fname, 'w') as outfile:
            json.dump(data, outfile, indent=True)

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

            value = json_[field.name]

            if os.path.exists(value) and os.path.isfile(value):
                shutil.copy(value, saveloc)
                json_[field.name] = os.path.split(json_[field.name])[1]

        return json_

    @classmethod
    def load(cls, saveloc, json_data, references=None):
        '''
        load json file, deserialize, and create object with new_from_dict()

        :param saveloc: location of data files
        '''
        if references is None:
            references = References()

        datafiles = cls._state.get_field_by_attribute('isdatafile')
        ref_fields = cls._state.get_field_by_attribute('save_reference')

        for field in datafiles:
            if field.name in json_data:
                json_data[field.name] = os.path.join(saveloc,
                                                     json_data[field.name])

        # pop references from json_data, create objects for them
        if ref_fields:
            ref_dict = {}
            for field in ref_fields:
                ref_filename = os.path.join(saveloc, json_data.pop(field.name))
                ref_obj = load(ref_filename, references)
                ref_dict[field.name] = ref_obj

        # deserialize after removing references
        to_eval = ('{0}.deserialize(json_data)'.format(json_data['obj_type']))
        _to_dict = eval(to_eval)

        if ref_fields:
            _to_dict.update(ref_fields)

        return _to_dict
