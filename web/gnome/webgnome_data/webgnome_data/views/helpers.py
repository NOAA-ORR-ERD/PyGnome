'''
Helper functions to be used by views.

Over time, these could be generalized into decorators, I suppose.
'''
from itertools import izip_longest

import numpy
np = numpy
from numpy import array, ndarray, void


def FQNamesToIterList(names):
    '''
       Turn a list of fully qualified object names into an iterlist of
       [<object_name>, <namespace>] items.
    '''
    for t in names:
        yield list(reversed(t.rsplit('.', 1))) if t.find('.') >= 0 else [t, '']


def FQNamesToList(names):
    return list(FQNamesToIterList(names))


def FilterFQNamesToIterList(names, name=None, namespace=None):
    for i in FQNamesToIterList(names):
        if ((name and i[0].find(name) >= 0) or
            (namespace and i[1].find(namespace) >= 0)):
            yield i


def FQNamesToDict(names):
    '''
        Takes a list of fully qualified names and turns it into a dict
        Where the object names are the keys.
        (note: dunno if this more useful than the plain dict() method.)
    '''
    return dict(FQNamesToIterList(names))


def JSONImplementedType(json_obj, obj_types):
    '''
        Here we determine if our JSON request payload implements an object
        contained within a set of implemented object types.

        I think this is a good place to implement our schema validators,
        but for right now let's just validate that it refers to an object
        type that is implementable.
        The convention we will use is this:
        - Our JSON will be a dictionary
        - This dictionary will contain a key called 'obj_type'
        - Key 'obj_type' will be in the format '<namespace>.<object_name>',
          where:
            - <namespace> refers to the python module namespace where
              the python class definition lives.
            - <object_name> refers to the name of the python class that
              implements the object.
        - This is not currently enforced, but It is understood that all
          other keys of the dictionary will conform to the referred object's
          construction method(s)

        :param json_obj: JSON payload
        :param obj_types: list of fully qualified object names.
    '''
    if not type(json_obj) == dict:
        raise ValueError('JSON needs to be a dict')

    if not 'obj_type' in json_obj:
        raise ValueError('JSON object needs to contain an obj_type')

    name, scope = FQNamesToList((json_obj['obj_type'],))[0]
    if name in FQNamesToDict(obj_types):
        return PyClassFromName(name, scope)

    return None


def JSONImplementsOneOf(json_obj, obj_types):
    try:
        return not JSONImplementedType(json_obj, obj_types) == None
    except:
        return False


def ObjectImplementsOneOf(model_object, obj_types):
    '''
        Here we determine if our python object type is contained within a set
        of implemented object types.

        :param model_obj: python object
        :param obj_types: list of fully qualified object names.
    '''
    requested_name = model_object.__class__.__name__

    if requested_name in FQNamesToDict(obj_types):
        return True

    return False


def PyClassFromName(name, scope):
    my_module = __import__(scope, globals(), locals(), [str(name)], -1)
    return getattr(my_module, name)


def CreateObject(json_obj, all_objects):
    '''
        Here we create a python object from our JSON payload
    '''
    name, scope = FQNamesToList((json_obj['obj_type'],))[0]
    py_class = PyClassFromName(name, scope)

    obj_dict = py_class.deserialize(json_obj)

    LinkObjectChildren(obj_dict, all_objects)

    return py_class.new_from_dict(obj_dict)


def LinkObjectChildren(obj_dict, all_objects):
    for k, v in obj_dict.items():
        if (ValueIsJsonObject(v)
            and v['id'] in all_objects):
            obj_dict[k] = all_objects[v['id']]


def UpdateObject(obj, json_obj):
    '''
        Here we update our python object with a JSON payload

        For now, I don't think we will be too fancy about this.
        We will grow more sophistication as we need it.
    '''
    name, scope = FQNamesToList((json_obj['obj_type'],))[0]
    py_class = PyClassFromName(name, scope)

    dict_ = py_class.deserialize(json_obj)

    return UpdateObjectAttributes(obj, dict_.iteritems())


def UpdateObjectAttributes(obj, items):
    #for k, v in items:
    #    UpdateObjectAttribute(obj, k, v)
    return all([UpdateObjectAttribute(obj, k, v) for k, v in items])


def UpdateObjectAttribute(obj, attr, value):
    if attr in ('id', 'obj_type', 'json_'):
        return False

    if (not ValueIsJsonObject(value)
        and not ObjectAttributesAreEqual(getattr(obj, attr), value)):
        setattr(obj, attr, value)
        return True
    else:
        return False


def ValueIsJsonObject(value):
    return (isinstance(value, dict)
            and 'id' in value
            and 'obj_type' in value)


def ObjectAttributesAreEqual(attr1, attr2):
    '''
        Recursive equality which includes sequence objects
        (not really dicts yet though)
    '''
    if not type(attr1) == type(attr2):
        return False

    if isinstance(attr1, (list, tuple, ndarray, void)):
        for x, y in izip_longest(attr1, attr2):
            if not ObjectAttributesAreEqual(x, y):
                # we want to short-circuit our iteration
                return False
        return True
    else:
        return attr1 == attr2
