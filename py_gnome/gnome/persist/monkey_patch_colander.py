'''
Created on Feb 27, 2013

Patching the result of the following 4 functions, to generate proper JSON output:

Boolean.serialize()
Int.serialize()
Float.serialize()
MappingSchema.serialize()

Obtained from: https://github.com/Pylons/colander/issues/80
'''
import colander

def apply():
    # Recover boolean values which were coerced into strings.
    serialize_boolean = getattr(colander.Boolean, 'serialize')
    def patched_boolean_serialization(*args, **kwds):
        result = serialize_boolean(*args, **kwds)
        if result is not colander.null:
            result = result == 'true'
        return result
    setattr(colander.Boolean, 'serialize', patched_boolean_serialization)

    # Recover float values which were coerced into strings.
    serialize_float = getattr(colander.Float, 'serialize')
    def patched_float_serialization(*args, **kwds):
        result = serialize_float(*args, **kwds)
        if result is not colander.null:
            result = float(result)
        return result
    setattr(colander.Float, 'serialize', patched_float_serialization)

    # Recover integer values which were coerced into strings.
    serialize_int = getattr(colander.Int, 'serialize')
    def patched_int_serialization(*args, **kwds):
        result = serialize_int(*args, **kwds)
        if result is not colander.null:
            result = int(result)
        return result
    setattr(colander.Int, 'serialize', patched_int_serialization)

    # Remove optional mapping keys which were associated with 'colander.null'.
    serialize_mapping = getattr(colander.MappingSchema, 'serialize')
    def patched_mapping_serialization(*args, **kwds):
        result = serialize_mapping(*args, **kwds)
        if result is not colander.null:
            result = {k:v for k,v in result.iteritems() if v is not colander.null}
        return result
    setattr(colander.MappingSchema, 'serialize', patched_mapping_serialization)
