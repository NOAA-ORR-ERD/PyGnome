"""
__init__.py for teh utilities package

a few small things here, 'cause why not?

"""

def get_mem_use(units='MB'):
    """
    returns the total memory use of the current python process

    :param units='MB': the units you want the reslut in. Options are:
                       'GB', 'MB', 'KB'

    This may not work on Windows
    """
    import resource
    useage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    div = {'GB': 1024*1024*1024,
           'MB': 1024*1024,
           'KB': 1024,
           }
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / float(div[units])