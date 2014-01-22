'''
NetCDF outputter - follows the interface defined by gnome.Outputter for a
NetCDF output writer
'''

import copy
import os
from datetime import datetime
from collections import OrderedDict

import netCDF4 as nc
import numpy as np

import gnome
from gnome.basic_types import oil_status
from gnome.outputter import Outputter
from gnome.utilities import serializable, time_utils

## Big dict that stores the attributes for the standard data arrays in the output
var_attributes = {'time': {'long_name':'time since the beginning of the simulation',
                           'standard_name':'time',
                           'calendar':'gregorian',
                           'standard_name':'time',
                           'comment':'unspecified time zone',
                           #units will get set based on data
                           },
                  'particle_count': {'units':'1',
                                     'long_name':'number of particles in a given timestep',
                                     'ragged_row_count':'particle count at nth timestep',
                                     },
                  'longitude': {'long_name':'longitude of the particle',
                                'standard_name':'longitude',
                                'units':'degrees_east',
                            },
                  'latitude': {'long_name':'latitude of the particle',
                               'standard_name':'latitude',
                               'units':'degrees_north',
                              },
                  'depth': {'long_name':'particle depth below sea surface',
                            'standard_name':'depth',
                            'units':'meters',
                            'axis':'z positive down',
                            },
                  'mass': {'long_name':'mass of particle',
                           'units':'grams',
                           },
                  'age': {'long_name':'age of particle from time of release',
                          'units':'seconds',
                          },
                  'status_codes': {'long_name':'particle status code',
                                   'flag_values': " ".join([ "%i" for i in oil_status._int]),
                                   'flag_meanings': " ".join ( [ "%i: %s"%pair for pair in sorted(zip(basic_types.oil_status._int,
                                                      basic_types.oil_status._attr) ) ] )
                                  },
                  'id': {'long_name':'particle ID',
                        },
                  'spill_num': {'long_name':'spill to which the particle belongs',

                  'droplet_diameter': {'long_name': 'diameter of oil droplet class',
                                       'units': 'meters'
                                      }
                  'rise_vel': {'long_name': 'rise velocity of oil droplet class',
                                            'units': 'm s-1'}
                  'next_positions':{},
                  'last_water_positions':{},
                  }
}


class NetCDFOutput(Outputter, serializable.Serializable):

    """
    A NetCDFOutput object is used to write the model's data to a NetCDF file.
    It inherits from Outputter class and implements the same interface.

    This class is meant to be used within the Model, to be added to list of
    outputters.

    >>> model = gnome.model.Model(...)
    >>> model.outputters += gnome.netcdf_outputter.NetCDFOutput(
                os.path.join(base_dir,'sample_model.nc'), which_data='most')

    `which_data` flag is used to set which data to add to the netcdf file:
        'standard' : teh basic stuff most people would want
        'most': everything the model is tracking except the for-internal-use only arrays
        'all': eveything tracked by teh model (mostly used for diagnosticts of save files)


    .. note::
       cf_attributes is a class attribute: a dict
       that contains the global attributes per CF convention
       
       The attribute: `.arrays_to_output` is a list of the data arrays that will be
       added to the netcdf file. array names may be added to or removed from
       this list before a model run to customize what gets output:
           `the_netcdf_outputter.arrays_to_output.append['rise_vel']`
       
       Since some of the names of the netcdf variables are different from the
       names in the SpillContainer data_arrays, this list uses the netcdf names

    """

    cf_attributes = {'comment': 'Particle output from the NOAA PyGnome model',
                     'source': 'PyGnome version %s'%gnome.__version__,
                     'references': 'TBD',
                     'feature_type': 'particle_trajectory',
                     'institution': 'NOAA Emergency Response Division',
                     'conventions': 'CF-1.6',
                     }

    ## the set of arrays we usually output -- i.e. the default
    standard_arrays = ['latitude',
                       'longitude', # these are pulled from the 'positions' array
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
                              ]

    # define state for serialization

    state = copy.deepcopy(serializable.Serializable.state)

    # data file should not be moved to save file location!
    state.add_field([ serializable.Field('netcdf_filename',
                                          create=True,
                                          update=True),
                      serializable.Field('which_data', create=True, update=True),
                      #serializable.Field('netcdf_format', create=True, update=True),
                      serializable.Field('compress', create=True, update=True),
                      serializable.Field('_start_idx', create=True),
                      serializable.Field('_middle_of_run', create=True),
                      ])

    @classmethod
    def new_from_dict(cls, dict_):
        """
        creates a new object from dictionary
        """

        _middle_of_run = dict_.pop('_middle_of_run', None)
        _start_idx = dict_.pop('_start_idx', None)
        obj = cls(**dict_)

        if _middle_of_run is None or _start_idx is None:
            raise KeyError('Expected netcdf_outputter to contain keys'
                           ' _middle_of_run and _start_idx')

        # If prepare_for_model_run is called, these will be reset
        obj._middle_of_run = _middle_of_run
        obj._start_idx = _start_idx
        return obj

    def __init__(self,
                 netcdf_filename,
                 which_data='standard', # options are 'standard', 'most', 'all'
                 compress=True,
                 id=None,
                 **kwargs
                 ):
        """

        Constructor for Net_CDFOutput object. It reads data from cache and
        writes it to a NetCDF4 format file using the CF convention

        :param netcdf_filename: Required parameter. The filename in which to
            store the NetCDF data.
        :type netcdf_filename: str. or unicode
        
        :param which_data: If true, write all data to NetCDF, otherwise write
            only standard data. Default is False.
        :type which_data: string, one of: 'standard', 'most', 'all'
        
        :param id: Unique Id identifying the newly created object (a UUID as a
            string). This is used when loading an object from a persisted
            state. User should never have to set this.

        Optional arguments passed on to base class (kwargs):

        :param cache: sets the cache object from which to read data. The model
            will automatically set this param

        :param output_timestep: default is None in which case everytime the
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

        use super to pass optional **kwargs to base class __init__ method
        """

        self._check_netcdf_filename(netcdf_filename)
        self._netcdf_filename = netcdf_filename
        self._uncertain = False
        self._u_netcdf_filename = None

        # flag to keep track of state of the object - is True after calling
        # prepare_for_model_run
        self._middle_of_run = False

        self._which_data = which_data
        self.arrays_to_output = copy.copy(self.standard_arrays)

        # this is only updated in prepare_for_model_run if all_data is True
        self.arr_types = None
        self._format = 'NETCDF4'
        self._compress = compress
        self._chunksize = 1024 # 1k is about right for 1000LEs and one time step.
                               # up to 0.5MB tested better for large datasets, but
                               # we don't want to have far-too-large files for the
                               # smaller ones
                               # The default in netcdf4 is 1 -- which works really badly

        # need to keep track of starting index for writing data since variable
        # number of particles are released
        self._start_idx = 0

        self._gnome_id = gnome.GnomeId(id)
        super(NetCDFOutput, self).__init__(**kwargs)

    @property
    def id(self):
        """
        Function returns the unique id to identify the object,
        """

        return self._gnome_id.id

    @property
    def middle_of_run(self):
        return self._middle_of_run

    @property
    def netcdf_filename(self):
        return self._netcdf_filename

    @netcdf_filename.setter
    def netcdf_filename(self, new_name):
        if self.middle_of_run:
            raise AttributeError('This attribute cannot be changed in the'
                                 ' middle of a run')
        else:
            self._check_netcdf_filename(new_name)
            self._netcdf_filename = new_name

    @property
    def which_data(self):
        return self._which_data

    @which_data.setter
    def which_data(self, value):
        if self.middle_of_run:
            raise AttributeError('This attribute cannot be changed in the'
                                 ' middle of a run')
        else:
            if value not in ('standard', 'most', 'all'):
                raise ValueError("which_data must be one of: 'standard', 'most', 'all'")
            else:
                self._which_data = value

    @property
    def compress(self):
        return self._compress

    @compress.setter
    def compress(self, value):
        if self.middle_of_run:
            raise AttributeError('This attribute cannot be changed in the'
                                 ' middle of a run')
        else:
            self._compress = value

    @property
    def netcdf_format(self):
        return self._format

    def _check_netcdf_filename(self, netcdf_filename):
        """ basic checks to make sure the netcdf_filename is valid """

        if os.path.isdir(netcdf_filename):
            raise ValueError('netcdf_filename must be a file not a directory.')

        if not os.path.exists(os.path.realpath(
                                            os.path.dirname(netcdf_filename))):
            raise ValueError('{0} does not appear to be a valid'
                             ' path'.format(os.path.dirname(netcdf_filename)))

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
            raise ValueError('{0} file exists. Enter a filename that does not'
                ' exist in which to save data.'.format(file_))

    def prepare_for_model_run(
        self,
        model_start_time,
        cache=None,
        uncertain=False,
        spills=None,
        **kwargs
        ):
        """
        .. function:: prepare_for_model_run(model_start_time,
                cache=None, uncertain=False, spills=None,
                **kwargs)

        Write global attributes and define dimensions and variables for NetCDF
        file. This must be done in prepare_for_model_run because if model state
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

        :param uncertain: Default is False. Model automatically sets this based
            on whether uncertainty is on or off. If this is True then a
            uncertain data is written to netcdf_filename + '_uncertain.nc'
        :type uncertain: bool
        :param spills: If 'which_data' flag is set to 'all' or 'most', then model
            must provide the model.spills object (SpillContainerPair object) so
            NetCDF variables can be defined for the remaining data arrays. If
            spills is None, but which_data flag is 'all' or 'most', a ValueError
            will be raised. It does not make sense to write 'all' or 'most' but
            not provide 'model.spills'.
        :type spills: gnome.spill_container.SpillContainerPair object.

        .. note::
        Does not take any other input arguments; however, to keep the interface
            the same for all outputters, define **kwargs incase future
            outputters require different arguments.

        use super to pass model_start_time, cache=None and
        remaining **kwargs to base class method
        """

        super(NetCDFOutput, self).prepare_for_model_run(model_start_time,
                cache, **kwargs)

        if ( self.which_data in ('all', 'most') ) and spills is None:
            raise ValueError("'which_data' flag is '%s', however spills is None."
                " Please provide valid model.spills so we know which"
                " additional data to write."%self.which_data)

        self._uncertain = uncertain

        if self._uncertain:
            (name, ext) = os.path.splitext(self.netcdf_filename)
            self._u_netcdf_filename = '{0}_uncertain{1}'.format(name,
                    ext)
            filenames = (self.netcdf_filename, self._u_netcdf_filename)
        else:
            filenames = (self.netcdf_filename, )

        for file_ in filenames:
            self._nc_file_exists_error(file_)
            with nc.Dataset(file_, 'w', format=self._format) as rootgrp:
                rootgrp.comment = self.cf_attributes['comment']
                rootgrp.creation_date = time_utils.round_time(datetime.now(),
                                                              roundTo=1).isoformat().replace('T', ' ')
                rootgrp.source = self.cf_attributes['source']
                rootgrp.references = self.cf_attributes['references']
                rootgrp.feature_type = self.cf_attributes['feature_type'
                        ]
                rootgrp.institution = self.cf_attributes['institution']
                rootgrp.convention = self.cf_attributes['conventions']

                rootgrp.createDimension('time', 0)
                rootgrp.createDimension('data', 0)

                time_ = rootgrp.createVariable('time',
                                               np.double,
                                               ('time', ),
                                               zlib=self._compress,
                                               chunksizes=(self._chunksize,) )
                time_.units = 'seconds since {0}'.format(
                        self._model_start_time.isoformat().replace('T', ' '))
                time_.long_name = 'time'
                time_.standard_name = 'time'
                time_.calendar = 'gregorian'
                time_.comment = 'unspecified time zone'

                pc = rootgrp.createVariable('particle_count',
                                            np.int32,
                                            ('time', ),
                                            zlib=self._compress,
                                            chunksizes=(self._chunksize,) )
                pc.units = '1'
                pc.long_name = 'number of particles in a given timestep'
                pc.ragged_row_count = 'particle count at nth timestep'

                for (key, val) in self.data_vars.iteritems():
                    # don't pop since it maybe required twice
                    var = rootgrp.createVariable(key,
                                                 val.get('dtype'),
                                                 ('data', ),
                                                 zlib=self._compress,
                                                 chunksizes=(self._chunksize,) )

                    # iterate over remaining attributes

                    [setattr(var, key2, val2) for (key2, val2) in
                     val.iteritems() if key2 != 'dtype']

                if self.which_data in ('all', 'most'):
                    rootgrp.createDimension('world_point', 3)
                    self.arr_types = dict()

                    at = spills.items()[0].array_types
                    [self.arr_types.update({key: atype}) for (key,
                     atype) in at.iteritems() if key
                     not in self.arr_types and key
                     not in self.standard_data]

                    # create variables

                    for (key, val) in self.arr_types.iteritems():
                        if len(val.shape) == 0:
                            rootgrp.createVariable(key,
                                                   val.dtype,
                                                   'data',
                                                   zlib=self._compress,
                                                   chunksizes=(self._chunksize,),
                                                   )
                        elif val.shape[0] == 3:
                            rootgrp.createVariable(key,
                                                   val.dtype,
                                                   ('data', 'world_point'),
                                                   zlib=self._compress,
                                                   chunksizes=(self._chunksize, 3),
                                                   )
                        else:
                            raise ValueError('{0} has an undefined dimension:'
                                             ' {1}'.format(key, val.shape))

        # need to keep track of starting index for writing data since variable
        # number of particles are released
        self._start_idx = 0
        self._middle_of_run = True

    def write_output(self, step_num, islast_step=False):
        """
        write NetCDF output at the end of the step

        :param step_num: the model step number you want rendered.
        :type step_num: int

        :param islast_step: default is False. Flag that indicates that step_num
            is last step. If 'output_last_step' is True then this is written
            out
        :type islast_step: bool

        use super to call base class write_output method
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
                rootgrp.variables['time'][curr_idx] = \
                    nc.date2num(time_stamp, rootgrp.variables['time'
                                ].units, rootgrp.variables['time'
                                ].calendar)
                pc = rootgrp.variables['particle_count']
                pc[curr_idx] = len(sc['status_codes'])

                _end_idx = self._start_idx + pc[curr_idx]
                rootgrp.variables['longitude'][self._start_idx:_end_idx] = \
                    sc['positions'][:, 0]
                rootgrp.variables['latitude'][self._start_idx:_end_idx] = \
                    sc['positions'][:, 1]
                rootgrp.variables['depth'][self._start_idx:_end_idx] = \
                    sc['positions'][:, 2]
                rootgrp.variables['status'][self._start_idx:_end_idx] = \
                    sc['status_codes'][:]

                for key in self._same_keynames:
                    rootgrp.variables[key][self._start_idx:_end_idx] = \
                        sc[key][:]

                # write remaining data

                if self.which_data in ('all', 'most'):
                    for (key, val) in self.arr_types.iteritems():
                        if len(val.shape) == 0:
                            rootgrp.variables[key][self._start_idx:
                                    _end_idx] = sc[key]
                        else:
                            rootgrp.variables[key][self._start_idx:
                                    _end_idx, :] = sc[key]

        self._start_idx = _end_idx  # set _start_idx for the next timestep

        # update self._next_output_time if data is successfully written
        self._update_next_output_time(step_num, sc.current_time_stamp)

        return {'step_num': step_num,
                'netcdf_filename': (self.netcdf_filename,
                self._u_netcdf_filename), 'time_stamp': time_stamp}

    def rewind(self):
        """
        if rewound, delete both the files and expect prepare_for_model_run to
        be called since rewind means start from beginning.

        Also call base class rewind to reset internal variables. Using super
        """

        super(NetCDFOutput, self).rewind()
        if os.path.exists(self.netcdf_filename):
            os.remove(self.netcdf_filename)

        if self._u_netcdf_filename is not None \
            and os.path.exists(self._u_netcdf_filename):
            os.remove(self._u_netcdf_filename)

        self._middle_of_run = False
        self._start_idx = 0

    def write_output_post_run(self,
        model_start_time,
        num_time_steps,
        cache=None,
        uncertain=False,
        spills=None,
        **kwargs):
        """
        Define all the positional input arguments. Pass these onto baseclass
        write_output_post_run as correct kwargs
        """
        super(NetCDFOutput, self).write_output_post_run(model_start_time,
                                                        num_time_steps,
                                                        cache,
                                                        uncertain=uncertain,
                                                        spills=spills,
                                                        **kwargs)

    @staticmethod
    def read_data(netcdf_file, time=None, all_data=False):
        """
        Read and create standard data arrays for a netcdf file that was created
        with NetCDFOutput class. Make it a static method since it is
        independent of an instance of the Outputter. The method is put with
        this class because the NetCDF functionality for PyGnome data with CF
        standard is captured here.

        :param netcdf_file: Name of the NetCDF file from which to read the data
        :type netcdf_file: str
        :param time: Index of the 'time' variable (or time_step) for which
            data is desired. Default is 0 so it returns data associated with
            first timestamp.
        :type time: datetime

        :returns: a dict containing standard data closest to the indicated
            'time'. Standard data is defined as follows:

        Standard data arrays are numpy arrays of size N, where N is number of
        particles released at time step of interest:
            'current_time_stamp': datetime object associated with this data
            'positions'         : NX3 array. Corresponds with NetCDF variables
                                  'longitude', 'latitude', 'depth'
            'status_codes'      : NX1 array. Corresponds with NetCDF variable
                                  'status'
            'spill_num'         : NX1 array. Corresponds with NetCDF variable
                                  'spill_num'
            'id'                : NX1 array showing particle id. Corresponds
                                  with NetCDF variable 'id'
            'mass'              : NX1 array showing 'mass' of each particle
        """

        if not os.path.exists(netcdf_file):
            raise IOError('File not found: {0}'.format(netcdf_file))

        arrays_dict = dict()
        with nc.Dataset(netcdf_file) as data:
            _start_ix = 0

            # first find the index of timestep in which we are interested
            time_ = data.variables['time']
            if time == None:
                # there should only be 1 time in file. Read and
                # return data associated with it
                if len(time_) > 1:
                    raise ValueError("More than one times found in netcdf"
                                     " file. Please specify time for which"
                                     " data is desired")
                else:
                    index = 0
            else:
                time_offset = nc.date2num(time, time_.units,
                                          calendar=time_.calendar)
                if time_offset < 0:
                    """ desired time is before start of model """
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

            positions = np.zeros((elem, 3),
                                 dtype=gnome.basic_types.world_point_type)

            positions[:, 0] = data.variables['longitude'][_start_ix:_stop_ix]
            positions[:, 1] = data.variables['latitude'][_start_ix:_stop_ix]
            positions[:, 2] = data.variables['depth'][_start_ix:_stop_ix]

            arrays_dict['positions'] = positions
            arrays_dict['status_codes'] = (data.variables['status']
                [_start_ix:_stop_ix])

            for key in NetCDFOutput._same_keynames:
                arrays_dict[key] = data.variables[key][_start_ix:_stop_ix]

            if all_data:  # append remaining data arrays
                excludes = NetCDFOutput.data_vars.keys()
                excludes.extend(['time', 'particle_count'])

                for key in data.variables.keys():
                    if key not in excludes:
                        if key in arrays_dict.keys():
                            raise ValueError('Error in read_data. {0} is'
                                ' already added to arrays_dict - trying to'
                                ' add it again'.format(key))

                        arrays_dict[key] = \
                            (data.variables[key])[_start_ix:_stop_ix]

        return arrays_dict
