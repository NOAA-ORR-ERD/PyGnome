'''
Helper functions to be used by views.

Over time, these could be generalized into decorators, I suppose.
'''
import json


def FQNamesToIterList(names):
    '''
       Turn a list of fully qualified object names into an iterlist of
       [<object_name>, <namespace>] items.
    '''
    for t in names:
        yield list(reversed(t.rsplit('.', 1))) if t.find('.') >= 0 else [t, '']


def FQNamesToList(names):
    return list(FQNamesToIterList(names))


def FilterFQNamesToIterList(names,
                            name=None,
                            namespace=None):
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


def JSONImplementsOneOf(json_obj, obj_types):
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
        return False

    if not 'obj_type' in json_obj:
        return False

    requested_name = FQNamesToList((json_obj['obj_type'],))[0][0]

    if requested_name in FQNamesToDict(obj_types):
        return True

    # TODO: We will maybe want to further validate our JSON object, maybe
    #       by using our schemas.  And we could do it here.

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
    print 'PyClassFromName(): name, scope = {0}'.format((name, scope))
    my_module = __import__(scope, globals(), locals(), [str(name)], -1)
    return getattr(my_module, name)


def CreateObject(json_obj):
    '''
        Here we create a python object from our JSON payload
    '''
    name, scope = FQNamesToList((json_obj['obj_type'],))[0]
    py_class = PyClassFromName(name, scope)

    print 'dir(py_class) = ', dir(py_class)
    dict_ = py_class.deserialize(json_obj)
    print 'dict_ = ', dict_

    return py_class.new_from_dict(dict_)


def UpdateObject(model_object, json_obj):
    '''
        Here we update our python object with a JSON payload

        For now, I don't think we will be too fancy about this.
        We will grow more sophistication as we need it.
    '''
    pass
