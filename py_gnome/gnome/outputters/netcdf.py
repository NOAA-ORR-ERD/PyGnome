'''
NetCDF outputter - follows the interface defined by gnome.Outputter for a
NetCDF output writer
'''

import copy
import os
from datetime import datetime
from collections import OrderedDict

import netCDF4 as nc

import numpy
np = numpy
from colander import SchemaNode, String, drop, Int, Bool

from gnome import __version__, array_types
from gnome.basic_types import oil_status, world_point_type

from gnome.utilities.serializable import Serializable, Field
from gnome.utilities.time_utils import round_time

from . import Outputter, BaseSchema


# Big dict that stores the attributes for the standard data arrays
# in the output - these are constants. The instance var_attributes are stored
# with the NetCDFOutput object
var_attributes = {
    'time': {'long_name': 'time since the beginning of the simulation',
             'standard_name': 'time',
             'calendar': 'gregorian',
             'standard_name': 'time',
             'comment': 'unspecified time zone',
             #units will get set based on data
             },
    'particle_count': {'units': '1',
                       'long_name': 'number of particles in a given timestep',
                       'ragged_row_count': 'particle count at nth timestep',
                       },
    'longitude': {'long_name': 'longitude of the particle',
                  'standard_name': 'longitude',
                  'units': 'degrees_east',
              },
    'latitude': {'long_name': 'latitude of the particle',
                 'standard_name': 'latitude',
                 'units': 'degrees_north',
                },
    'depth': {'long_name': 'particle depth below sea surface',
              'standard_name': 'depth',
              'units': 'meters',
              'axis': 'z positive down',
              },
    'mass': {'long_name': 'mass of particle',
             'units': 'grams',
             },
    'age': {'long_name': 'age of particle from time of release',
            'units': 'seconds',
            },
    'status_codes': {'long_name': 'particle status code',
                     'flag_values': " ".join(["%i" % i for i in oil_status._int]),
                     'flag_meanings': " ".join(["%i: %s," % pair for pair in sorted(zip(oil_status._int,
                                        oil_status._attr))])
                    },
    'id': {'long_name': 'particle ID',
          },
    'droplet_diameter': {'long_name': 'diameter of oil droplet class',
                         'units': 'meters'
                        },
    'rise_vel': {'long_name': 'rise velocity of oil droplet class',
                              'units': 'm s-1'},
    'next_positions': {},
    'last_water_positions': {},
    }


class NetCDFOutputSchema(BaseSchema):
    'colander schema for serialize/deserialize object'
    netcdf_filename = SchemaNode(String(), missing=drop)
    all_data = SchemaNode(Bool(), missing=drop)
    compress = SchemaNode(Bool(), missing=drop)
    _start_idx = SchemaNode(Int(), missing=drop)
    _middle_of_run = SchemaNode(Bool(), missing=drop)


class NetCDFOutput(Outputter, Serializable):
    """
    A NetCDFOutput object is used to write the model's data to a NetCDF file.
    It inherits from Outputter class and implements the same interface.

    This class is meant to be used within the Model, to be added to list of
    outputters.

    >>> model = gnome.model.Model(...)
    >>> model.outputters += gnome.netcdf_outputter.NetCDFOutput(
                os.path.join(base_dir,'sample_model.nc'), which_data='most')

    `which_data` flag is used to set which data to add to the netcdf file:
    'standard' : the basic stuff most people would want
    'most': everything the model is tracking except the internal-use-only
        arrays
    'all': everything tracked by the model (mostly used for diagnostics of
        save files)


    .. note::
       cf_attributes is a class attribute: a dict
       that contains the global attributes per CF convention

       The attribute: `.arrays_to_output` is a set of the data arrays that
       will be added to the netcdf file. array names may be added to or removed
       from this set before a model run to customize what gets output:
            `the_netcdf_outputter.arrays_to_output.add['rise_vel']`

       Since some of the names of the netcdf variables are different from the
       names in the SpillContainer data_arrays, this list uses the netcdf names
    """
    which_data_lu = {'standard', 'most', 'all'}
    compress_lu = {True, False}

    cf_attributes = {'comment': 'Particle output from the NOAA PyGnome model',
                     'source': 'PyGnome version {0}'.format(__version__),
                     'references': 'TBD',
                     'feature_type': 'particle_trajectory',
                     'institution': 'NOAA Emergency Response Division',
                     'conventions': 'CF-1.6',
                     }

    ## the set of arrays we usually output -- i.e. the default
    standard_arrays = ['latitude',
                       'longitude',  # pulled from the 'positions' array
                       'depth',
                       'status_codes',
                       'spill_num',
                       'id',
                       'mass',
                       'age',
                       ]

    ## the list of arrays that we usually don't want -- i.e. for internal use
    ## these will get skipped if "most" is asked for
    ## "all" will output everything.
    usually_skipped_arrays = ['next_positions',
                              'last_water_positions',
                              'windages',
                              'windage_range',
                              'windage_persist',
                              'mass_components',
                              'half_lives',
                              ]

    # define _state for serialization
    _state = copy.deepcopy(Outputter._state)

    # data file should not be moved to save file location!
    _state.add_field([Field('netcdf_filename', save=True, update=True,
                            test_for_eq=False),
                     Field('which_data', save=True, update=True),
                     #Field('netcdf_format', save=True, update=True),
                     Field('compress', save=True, update=True),
                     Field('_start_idx', save=True),
                     Field('_middle_of_run', save=True),
                     ])
    _schema = NetCDFOutputSchema

    def __init__(self, netcdf_filename, which_data='standard', compress=True,
                 **kwargs):
        """
        Constructor for Net_CDFOutput object. It reads data from cache and
        writes it to a NetCDF4 format file using the CF convention

        :param netcdf_filename: Required parameter. The filename in which to
            store the NetCDF data.
        :type netcdf_filename: str. or unicode

        :param which_data: If 'all', write all data to NetCDF.
            If 'standard', write only standard data.
            Not sure what 'most' means.
            Default is 'standard'.
        :type which_data: {'standard', 'most', 'all'}

        Optional arguments passed on to base class (kwargs):

        :param cache: sets the cache object from which to read data. The model
            will automatically set this param

        :param output_timestep: default is None in which case every time the
            write_output is called, output is written. If set, then output is
            written every output_timestep starting from model_start_time.
        :type output_timestep: timedelta object

        :param output_zero_step: default is True. If True then output for
            initial step (showing initial release conditions) is written
            regardless of output_timestep
        :type output_zero_step: boolean

        :param output_last_step: default is True. If True then output for
            final step is written regardless of output_timestep
        :type output_last_step: boolean

        use super to pass optional kwargs to base class __init__ method
        """
        self._check_netcdf_filename(netcdf_filename)
        self._netcdf_filename = netcdf_filename
        self._uncertain = False
        self._u_netcdf_filename = None
        self.name = os.path.split(netcdf_filename)[1]

        # flag to keep track of _state of the object - is True after calling
        # prepare_for_model_run
        self._middle_of_run = False

        if which_data.lower() in self.which_data_lu:
            self._which_data = which_data.lower()
        else:
            raise ValueError('which_data must be one of: '
                             '{"standard", "most", "all"}')

        self.arrays_to_output = set(self.standard_arrays)

        # this is only updated in prepare_for_model_run if which_data is
        # 'all' or 'most'
        #self.arr_types = None
        self._format = 'NETCDF4'

        if compress in self.compress_lu:
            self._compress = compress
        else:
            raise ValueError('compress must be one of: {True, False}')


        # 1k is about right for 1000LEs and one time step.
        # up to 0.5MB tested better for large datasets, but
        # we don't want to have far-too-large files for the
        # smaller ones
        # The default in netcdf4 is 1 -- which works really badly
        self._chunksize = 1024

        # need to keep track of starting index for writing data since variable
        # number of particles are released
        self._start_idx = 0

        # spill_num's attributes are instance attributes.
        # 'spill_names' is set based on the names of spill's as defined by user
        self._var_attributes = {
            'spill_num':
                        {'long_name': 'spill to which the particle belongs',
                         'spills_map': ''}
                         }
        super(NetCDFOutput, self).__init__(**kwargs)

    @property
    def middle_of_run(self):
        return self._middle_of_run

    @property
    def netcdf_filename(self):
        return self._netcdf_filename

    @netcdf_filename.setter
    def netcdf_filename(self, new_name):
        if self.middle_of_run:
            raise AttributeError('This attribute cannot be changed in the '
                                 'middle of a run')
        else:
            self._check_netcdf_filename(new_name)
            self._netcdf_filename = new_name

    @property
    def which_data(self):
        return self._which_data

    @which_data.setter
    def which_data(self, value):
        if self.middle_of_run:
            raise AttributeError('This attribute cannot be changed in the '
                                 'middle of a run')

        if value in self.which_data_lu:
            self._which_data = value
        else:
            raise ValueError('which_data must be one of: '
                             '{"standard", "most", "all"}')

    @property
    def chunksize(self):
        return self._chunksize

    @chunksize.setter
    def chunksize(self, value):
        if self.middle_of_run:
            raise AttributeError('chunksize can not be set '
                                 'in the middle of a run')
        else:
            self._chunksize = value

    @property
    def compress(self):
        return self._compress

    @compress.setter
    def compress(self, value):
        if self.middle_of_run:
            raise AttributeError('This attribute cannot be changed in the '
                                 'middle of a run')

        if value in self.compress_lu:
            self._compress = value
        else:
            raise ValueError('compress must be one of: {True, False}')

    @property
    def netcdf_format(self):
        return self._format

    def _check_netcdf_filename(self, netcdf_filename):
        'basic checks to make sure the netcdf_filename is valid'
        if os.path.isdir(netcdf_filename):
            raise ValueError('netcdf_filename must be a file not a directory.')

        if not os.path.exists(os.path.realpath(os.path.dirname(netcdf_filename)
                                               )):
            raise ValueError('{0} does not appear to be a valid path'
                             .format(os.path.dirname(netcdf_filename)))

    def _nc_file_exists_error(self, file_):
        """
        invoked by prepare_for_model_run. If file already exists, it will raise
        this error.

        Do this in prepare_for_model_run, because user may want to define the
        model and run it in batch mode. This will allow netcdf_outputter to be
        created, but the first time it tries to write this file, it will check
        and raise an error if file exists
        """
        if os.path.exists(file_):
            raise ValueError('{0} file exists. Enter a filename that '
                             'does not exist in which to save data.'
                             .format(file_))

    def _update_spill_names(self, spills):
        '''
        update spill_names list in NC attribute 'spill_num'
        '''
        names = " ".join(["{0}: {1}, ".format(ix, spill.name)
                                        for ix, spill in enumerate(spills)])
        self._var_attributes['spill_num']['spills_map'] = names

    def prepare_for_model_run(self, model_start_time,
                              spills, uncertain=False,
                              **kwargs):
        """
        .. function:: prepare_for_model_run(model_start_time,
                                            cache=None,
                                            uncertain=False,
                                            spills=None,
                                            **kwargs)

        Write global attributes and define dimensions and variables for NetCDF
        file. This must be done in prepare_for_model_run because if model _state
        changes, it is rewound and re-run from the beginning.

        This takes more than standard 'cache' argument. Some of these are
        required arguments - they contain None for defaults because non-default
        argument cannot follow default argument. Since cache is already 2nd
        positional argument for Renderer object, the required non-default
        arguments must be defined following 'cache'.

        If uncertainty is on, then UncertainSpillPair object contains
        identical _data_arrays in both certain and uncertain SpillContainer's,
        the data itself is different, but they contain the same type of data
        arrays.

        :param bool uncertain: Default is False. Model automatically sets this
                               based on whether uncertainty is on or off.
                               If this is True then an uncertain data is
                               written to netcdf_filename + '_uncertain.nc'

        :param spills: If 'which_data' flag is set to 'all' or 'most', then
            model must provide the model.spills object
            (SpillContainerPair object) so NetCDF variables can be
            defined for the remaining data arrays.
            If spills is None, but which_data flag is 'all' or
            'most', a ValueError will be raised.
            It does not make sense to write 'all' or 'most' but not
            provide 'model.spills'.
        :type spills: gnome.spill_container.SpillContainerPair object.

        .. note::
            Does not take any other input arguments; however, to keep the
            interface the same for all outputters, define kwargs in case
            future outputters require different arguments.

        use super to pass model_start_time, cache=None and
        remaining kwargs to base class method
        """
        super(NetCDFOutput, self).prepare_for_model_run(model_start_time,
                                                        **kwargs)
        self._uncertain = uncertain

        self._update_spill_names(spills)
        if self._uncertain:
            name, ext = os.path.splitext(self.netcdf_filename)
            self._u_netcdf_filename = '{0}_uncertain{1}'.format(name, ext)
            filenames = (self.netcdf_filename, self._u_netcdf_filename)
        else:
            filenames = (self.netcdf_filename,)

        for file_ in filenames:
            self._nc_file_exists_error(file_)
            ## create the netcdf files and write the standard stuff:
            with nc.Dataset(file_, 'w', format=self._format) as rootgrp:
                ## fixme: why remove the "T" ??
                self.cf_attributes['creation_date'] = \
                            round_time(datetime.now(), roundTo=1).isoformat()
                rootgrp.setncatts(self.cf_attributes)
                    #rootgrp.comment = self.cf_attributes['comment']
                    #rootgrp.source = self.cf_attributes['source']
                    #rootgrp.references = self.cf_attributes['references']
                    #rootgrp.feature_type = self.cf_attributes['feature_type']
                    #rootgrp.institution = self.cf_attributes['institution']
                    #rootgrp.convention = self.cf_attributes['conventions']

                # create the dimensions we need
                # not sure if it's a convention or if dimensions
                # need to be names...
                rootgrp.createDimension('time')  # unlimited
                rootgrp.createDimension('data')  # unlimited
                rootgrp.createDimension('two', 2)
                rootgrp.createDimension('three', 3)
                rootgrp.createDimension('four', 4)
                rootgrp.createDimension('five', 5)

                # create the time variable
                time_ = rootgrp.createVariable('time',
                                               np.float64,
                                               ('time', ),
                                               zlib=self._compress,
                                               chunksizes=(self._chunksize,))
                time_.setncatts(var_attributes['time'])
                time_.units = 'seconds since {0}'.format(
                        self._model_start_time.isoformat())

                # create the particle count variable
                pc_ = rootgrp.createVariable('particle_count',
                                            np.int32,
                                            ('time', ),
                                            zlib=self._compress,
                                            chunksizes=(self._chunksize,))
                pc_.setncatts(var_attributes['particle_count'])

                ## create the list of variables that we want to put in the file
                if self.which_data in ('all', 'most'):
                    for var_name in spills.items()[0].array_types:
                        if var_name != 'positions':
                            # handled by latitude, longitude, depth
                            self.arrays_to_output.add(var_name)

                    if self.which_data == 'most':
                        # remove the ones we don't want
                        for var_name in self.usually_skipped_arrays:
                            self.arrays_to_output.discard(var_name)

                for var_name in self.arrays_to_output:
                    # the special cases:
                    if var_name in ('latitude', 'longitude', 'depth'):
                        # these don't  map directly to an array_type
                        dt = world_point_type
                        shape = ('data', )
                        chunksizes = (self._chunksize,)
                    else:
                        at = getattr(array_types, var_name)
                        dt = at.dtype

                        # total kludge for the cases we happen to have...
                        if at.shape == (5,):
                            shape = ('data', 'five')
                            chunksizes = (self._chunksize, 5)
                        elif at.shape == (4,):
                            shape = ('data', 'four')
                            chunksizes = (self._chunksize, 4)
                        elif at.shape == (3,):
                            shape = ('data', 'three')
                            chunksizes = (self._chunksize, 3)
                        elif at.shape == (2,):
                            shape = ('data', 'two')
                            chunksizes = (self._chunksize, 2)
                        else:
                            shape = ('data',)
                            chunksizes = (self._chunksize,)

                    var = rootgrp.createVariable(var_name,
                                                 dt,
                                                 shape,
                                                 zlib=self._compress,
                                                 chunksizes=chunksizes,
                                                 )

                    # add attributes
                    if var_name in var_attributes:
                        var.setncatts(var_attributes[var_name])
                    elif var_name in self._var_attributes:
                        var.setncatts(self._var_attributes[var_name])

        # need to keep track of starting index for writing data since variable
        # number of particles are released
        self._start_idx = 0
        self._middle_of_run = True

    def write_output(self, step_num, islast_step=False):
        """
        Write NetCDF output at the end of the step

        :param int step_num: the model step number you want rendered.
        :param bool islast_step: Default is False.
                                 Flag that indicates that step_num is
                                 last step.
                                 If 'output_last_step' is True then this is
                                 written out

        Use super to call base class write_output method
        """
        super(NetCDFOutput, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        for sc in self.cache.load_timestep(step_num).items():
            if sc.uncertain and self._u_netcdf_filename is not None:
                file_ = self._u_netcdf_filename
            else:
                file_ = self.netcdf_filename

            time_stamp = sc.current_time_stamp

            with nc.Dataset(file_, 'a') as rootgrp:
                curr_idx = len(rootgrp.variables['time'])
                rootgrp.variables['time'][curr_idx] = nc.date2num(time_stamp,
                                                                  rootgrp.variables['time'].units,
                                                                  rootgrp.variables['time'].calendar)
                pc = rootgrp.variables['particle_count']
                pc[curr_idx] = len(sc)

                _end_idx = self._start_idx + pc[curr_idx]

                # add the data:
                for var_name in self.arrays_to_output:
                    ## special case positions:
                    if var_name == 'longitude':
                        rootgrp.variables['longitude'][self._start_idx:_end_idx] = \
                                    sc['positions'][:, 0]
                    elif var_name == 'latitude':
                        rootgrp.variables['latitude'][self._start_idx:_end_idx] = \
                                    sc['positions'][:, 1]
                    elif var_name == 'depth':
                        rootgrp.variables['depth'][self._start_idx:_end_idx] = \
                                    sc['positions'][:, 2]
                    else:
                        rootgrp.variables[var_name][self._start_idx:_end_idx] = \
                                    sc[var_name]

        self._start_idx = _end_idx  # set _start_idx for the next timestep

        # update self._next_output_time if data is successfully written
        self._update_next_output_time(step_num, sc.current_time_stamp)

        return {'step_num': step_num,
                'netcdf_filename': (self.netcdf_filename,
                                    self._u_netcdf_filename),
                'time_stamp': time_stamp}

    def rewind(self):
        '''
        If rewound, delete both the files and expect prepare_for_model_run to
        be called since rewind means start from beginning.

        Also call base class rewind to reset internal variables. Using super
        '''
        super(NetCDFOutput, self).rewind()

        if os.path.exists(self.netcdf_filename):
            os.remove(self.netcdf_filename)

        if (self._u_netcdf_filename is not None
            and os.path.exists(self._u_netcdf_filename)):
            os.remove(self._u_netcdf_filename)

        self._middle_of_run = False
        self._start_idx = 0

    @classmethod
    def read_data(klass, netcdf_file, time=None, index=None,
                  which_data='standard'):
        """
        Read and create standard data arrays for a netcdf file that was created
        with NetCDFOutput class. Make it a class method since it is
        independent of an instance of the Outputter. The method is put with
        this class because the NetCDF functionality for PyGnome data with CF
        standard is captured here.

        :param str netcdf_file: Name of the NetCDF file from which to read
            the data
        :param datetime time: timestamp at which the data is desired. Looks in
            the netcdf data's 'time' array and finds the closest time to this
            and outputs this data. If both 'time' and 'index' are None, return
            data if file only contains one 'time' else raise an error
        :param int index: Index of the 'time' variable (or time_step). This is
            only used if 'time' is None. If both 'time' and 'index' are None,
            return data if file only contains one 'time' else raise an error
        :param which_data='standard': Which data arrays are desired options are
            ('standard', 'most', 'all', [list_of_array_names])
        :type which_data: string or sequence of strings.

        :return: A dict containing standard data closest to the indicated
            'time'. Standard data is defined as follows:

        Standard data arrays are numpy arrays of size N, where N is number of
        particles released at time step of interest. They are defined by the
        class attribute "standard_arrays", currently:

            'current_time_stamp': datetime object associated with this data
            'positions'         : NX3 array. NetCDF variables: 'longitude', 'latitude', 'depth'
            'status_codes'      : NX1 array. NetCDF variable :'status_codes'
            'spill_num'         : NX1 array. NetCDF variable: 'spill_num'
            'id'                : NX1 array of particle id. NetCDF variable 'id'
            'mass'              : NX1 array showing 'mass' of each particle

        standard_arrays = ['latitude',
                           'longitude', # pulled from the 'positions' array
                           'depth',
                           'status_codes',
                           'spill_num',
                           'id',
                           'mass',
                           'age',
                           ]

        """
        if not os.path.exists(netcdf_file):
            raise IOError('File not found: {0}'.format(netcdf_file))

        arrays_dict = {}
        with nc.Dataset(netcdf_file) as data:
            _start_ix = 0

            # first find the index of index in which we are interested
            time_ = data.variables['time']
            if time == None and index == None:
                # there should only be 1 time in file. Read and
                # return data associated with it
                if len(time_) > 1:
                    raise ValueError('More than one time found in netcdf '
                                     'file. Please specify time/index for '
                                     'which data is desired')
                else:
                    index = 0
            else:
                if time is not None:
                    time_offset = nc.date2num(time, time_.units,
                                              calendar=time_.calendar)
                    if time_offset < 0:
                        'desired time is before start of model'
                        index = 0
                    else:
                        index = abs(time_[:] - time_offset).argmin()

            for idx in range(index):
                _start_ix += data.variables['particle_count'][idx]

            _stop_ix = _start_ix + data.variables['particle_count'][index]
            elem = data.variables['particle_count'][index]

            c_time = nc.num2date(time_[index], time_.units,
                                 calendar=time_.calendar)

            arrays_dict['current_time_stamp'] = np.array(c_time)

            ## figure out what arrays to read in:
            if which_data == 'standard':
                data_arrays = set(klass.standard_arrays)
                # swap out positions:
                [data_arrays.discard(x)
                 for x in ('latitude', 'longitude', 'depth')]
                data_arrays.add('positions')
            elif which_data == 'all':
                # pull them from the nc file
                data_arrays = set(data.variables.keys())
                ## remove the irrelevant ones:
                [data_arrays.discard(x) for x in ('time',
                                                  'particle_count',
                                                  'latitude',
                                                  'longitude',
                                                  'depth')]
                data_arrays.add('positions')
            else:  # should be list of data arrays
                data_arrays = set(which_data)

            # get the data
            for array_name in data_arrays:
                #special case time and positions:
                if array_name == 'positions':
                    positions = np.zeros((elem, 3), dtype=world_point_type)
                    positions[:, 0] = data.variables['longitude'][_start_ix:_stop_ix]
                    positions[:, 1] = data.variables['latitude'][_start_ix:_stop_ix]
                    positions[:, 2] = data.variables['depth'][_start_ix:_stop_ix]

                    arrays_dict['positions'] = positions
                else:
                    arrays_dict[array_name] = (data.variables[array_name][_start_ix:_stop_ix])

        return arrays_dict

    def save(self, saveloc, references=None, name=None):
        '''
        See baseclass :meth:`~gnome.persist.Savable.save`

        update netcdf_filename to point to saveloc, then call base class save
        using super
        '''
        json_ = self.serialize('save')
        fname = os.path.split(json_['netcdf_filename'])[1]
        json_['netcdf_filename'] = os.path.join('./', fname)
        return self._json_to_saveloc(json_, saveloc, references, name)

    @classmethod
    def load(cls, saveloc, json_data, references=None):
        '''
        update path to 'netcdf_filename' in json_data, then finsih loading
        by calling super class' load method
        '''
        json_data['netcdf_filename'] = os.path.join(saveloc,
                                        json_data['netcdf_filename'])

        return super(NetCDFOutput, cls).load(saveloc, json_data, references)
