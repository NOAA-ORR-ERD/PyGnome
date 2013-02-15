'''
Created on Feb 15, 2013
'''

class Serializable(object):
    """
    Designed as a mix-in for adding to Gnome objects
    if they should have to_dict and from_dict methods
    """
    serializable_fields = []

    def to_dict(self):
        """
        Return a dictionary containing the serialized representation of this
        object, using `self.serializable_fields` to look up fields on the object
        that the dictionary should contain.

        For every field, if there is a method defined on the object such that
        the method name is `{field_name}_to_dict`, use the return value of that
        method as the field value.

        Note: any field in `self.serializable_fields` that does not exist on the
        object and does not have a to_dict method will raise an AttributeError.
        """
        data = {}

        for key in self.serializable_fields:
            to_dict_fn_name = '%s_to_dict' % key

            if hasattr(self, to_dict_fn_name):
                value = getattr(self, to_dict_fn_name)()
            else:
                value = getattr(self, key)

            if hasattr(value, 'to_dict'):
                value = value.to_dict()

            data[key] = value
        return data

    def from_dict(self, data):
        """
        Set the state of this object using the dictionary ``data`` by looking up
        the value of each key in ``data`` that is also in
        `self.serializable_fields`.

        For every field, the choice of how to set the field is as follows:

        If there is a method defined on the object such that the method name is
        `{field_name}_from_dict`, call that method with the field's data.

        If the field on the object has a ``from_dict`` method, then use that
        method instead.

        If neither method exists, then set the field with the value from
        ``data`` directly on the object.
        """
        for key in self.serializable_fields:
            if not key in data:
                continue

            from_dict_fn_name = '%s_from_dict' % key
            field = getattr(self, key)
            value = data[key]

            if hasattr(self, from_dict_fn_name):
                getattr(self, from_dict_fn_name)(value)
            elif hasattr(field, 'from_dict'):
                field.from_dict(value)
            else:
                setattr(self, key, value)

        return self
