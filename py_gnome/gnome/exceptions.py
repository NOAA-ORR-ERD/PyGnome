'''
PyGnome custom exceptions
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class GnomeRuntimeError(Exception):
    def __init__(self, *args):
        '''
        *args can contains a message, and other arguments
        '''
        super(GnomeRuntimeError, self).__init__(*args)


class ReferencedObjectNotSet(Exception):
    '''
    *args can contains a message, and other arguments
    '''
    def __init__(self, *args):
        super(ReferencedObjectNotSet, self).__init__(*args)
