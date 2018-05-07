"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""
import copy

from gnome.utilities import serializable


class EnvironmentMeta(type):
    def __init__(cls, _name, _bases, _dct):
        cls._subclasses = []
        for c in cls.__mro__:
            if hasattr(c, '_subclasses') and c is not cls:
                c._subclasses.append(cls)


class Environment(object):
    """
    A base class for all classes in environment module

    This is primarily to define a dtype such that the OrderedCollection
    defined in the Model object requires it.
    """
    _subclasses = []
    _state = copy.deepcopy(serializable.Serializable._state)

    # env objects referenced by others using this attribute name
    # eg: For Wind objects, set to 'wind', for Water object set to 'water'
    # so we have a way to identify all wind objects without relying on
    # insinstance() checks. Used by model to automatically hook up objects that
    # reference environment objects
    _ref_as = 'environment'

    __metaclass__ = EnvironmentMeta

    def __init__(self, name=None, make_default_refs=True):
        '''
        base class for environment objects

        :param name=None:
        '''
        self.name = None if name is None else name

        self.make_default_refs = make_default_refs

    @property
    def data_start(self):
        raise NotImplementedError

    @data_start.setter
    def data_start(self, value):
        raise NotImplementedError

    @property
    def data_stop(self):
        raise NotImplementedError

    @data_stop.setter
    def data_stop(self, value):
        raise NotImplementedError

    def prepare_for_model_run(self, model_time):
        """
        Override this method if a derived environment class needs to perform
        any actions prior to a model run
        """
        pass

    def prepare_for_model_step(self, model_time):
        """
        Override this method if a derived environment class needs to perform
        any actions prior to a model run
        """
        pass


def env_from_netCDF(filename=None, dataset=None,
                    grid_file=None, data_file=None, _cls_list=None,
                    **kwargs):
    '''
        Returns a list of instances of environment objects that can be produced
        from a file or dataset.  These instances will be created with a common
        underlying grid, and will interconnect when possible.
        For example, if an IceAwareWind can find an existing IceConcentration,
        it will use it instead of instantiating another. This function tries
        ALL gridded types by default. This means if a particular subclass
        of object is possible to be built, it is likely that all it's parents
        will be built and included as well.

        If you wish to limit the types of environment objects that will
        be used, pass a list of the types using "_cls_list" kwarg
    '''
    def attempt_from_netCDF(cls, **klskwargs):
        obj = None
        try:
            obj = c.from_netCDF(**klskwargs)
        except Exception as e:
            import logging
            logging.warn('''Class {0} could not be constituted from netCDF file
                                    Exception: {1}'''.format(c.__name__, e))
        return obj

    from gnome.environment.gridded_objects_base import Variable, VectorVariable
    from gridded.utilities import get_dataset
    from gnome.environment import PyGrid, Environment

    new_env = []

    if filename is not None:
        data_file = filename
        grid_file = filename

    ds = None
    dg = None
    if dataset is None:
        if grid_file == data_file:
            ds = dg = get_dataset(grid_file)
        else:
            ds = get_dataset(data_file)
            dg = get_dataset(grid_file)
    else:
        if grid_file is not None:
            dg = get_dataset(grid_file)
        else:
            dg = dataset
        ds = dataset
    dataset = ds

    grid = kwargs.pop('grid', None)
    if grid is None:
        grid = PyGrid.from_netCDF(filename=filename, dataset=dg, **kwargs)
        kwargs['grid'] = grid

    if _cls_list is None:
        scs = copy.copy(Environment._subclasses)
    else:
        scs = _cls_list

    for c in scs:
        if (issubclass(c, (Variable, VectorVariable)) and
                not any([isinstance(o, c) for o in new_env])):
            clskwargs = copy.copy(kwargs)
            obj = None

            try:
                req_refs = c._req_refs
            except AttributeError:
                req_refs = None

            if req_refs is not None:
                for ref, klass in req_refs.items():
                    for o in new_env:
                        if isinstance(o, klass):
                            clskwargs[ref] = o

                    if ref in clskwargs.keys():
                        continue
                    else:
                        obj = attempt_from_netCDF(c,
                                                  filename=filename,
                                                  dataset=dataset,
                                                  grid_file=grid_file,
                                                  data_file=data_file,
                                                  **clskwargs)
                        clskwargs[ref] = obj

                        if obj is not None:
                            new_env.append(obj)

            obj = attempt_from_netCDF(c,
                                      filename=filename,
                                      dataset=dataset,
                                      grid_file=grid_file,
                                      data_file=data_file,
                                      **clskwargs)

            if obj is not None:
                new_env.append(obj)

    return new_env


def ice_env_from_netCDF(filename=None, **kwargs):
    '''
        A short function to generate a list of all the 'ice_aware' classes
        for use in env_from_netCDF (this excludes GridCurrent, GridWind,
        GridTemperature, etc.)
    '''
    from gnome.environment import Environment
    cls_list = Environment._subclasses
    ice_cls_list = [c for c in cls_list
                    if (hasattr(c, '_ref_as') and 'ice_aware' in c._ref_as)]

    return env_from_netCDF(filename=filename, _cls_list=ice_cls_list, **kwargs)


def get_file_analysis(filename):
    env = env_from_netCDF(filename=filename)
    # classes = copy.copy(Environment._subclasses)

    if len(env) > 0:
        report = ['Can create {0} types of environment objects'
                  .format(len([env.__class__ for e in env]))]
        report.append('Types are: {0}'.format(str([e.__class__ for e in env])))

    report = report + grid_detection_report(filename)

    return report


def grid_detection_report(filename):
    from gnome.environment.gridded_objects_base import PyGrid

    topo = PyGrid._find_topology_var(filename)
    report = ['Grid report:']

    if topo is None:
        report.append('    A standard grid topology was not found in the file')
        report.append('    topology breakdown future feature')
    else:
        report.append('    A grid topology was found in the file: {0}'
                      .format(topo))

    return report
