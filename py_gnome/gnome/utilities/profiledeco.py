'''

Purpose of this is to allow fast implementation of profiling during development by adding
decorators where profiling is desired.

@profile
def foo():
    sleep(1)
    
this should profile foo whenever it is called and add it to the global profile stats.

'''

import cProfile
import pstats
import io

profiler = cProfile.Profile()

def profile(func):
    def wrap(*args, **kwargs):
        profiler.enable()
        result = func(*args,**kwargs)
        profiler.disable()
        return result
    return wrap

def print_func_profile(*args,**kwargs):
    ordering = 'cumulative'
    if 'ordering' in kwargs:
        ordering = kwargs['ordering']
    ps = pstats.Stats(profiler).strip_dirs().sort_stats(ordering)
    ps.print_stats(*args)
    
def print_stats(*args, **kwargs):
    print_func_profile(*args, **kwargs)
    
def clear_stats():
    '''
    Just rebuild the global object
    '''
    global profiler
    profiler = cProfile.Profile()
