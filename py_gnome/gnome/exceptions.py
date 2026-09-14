'''
PyGnome custom exceptions
'''


class GnomeRuntimeError(Exception):
    def __init__(self, *args):
        '''
        *args can contain a message, and other arguments
        '''
        super().__init__(*args)


class ReferencedObjectNotSet(Exception):
    '''
    *args can contain a message, and other arguments
    '''
    def __init__(self, *args):
        super().__init__(*args)
