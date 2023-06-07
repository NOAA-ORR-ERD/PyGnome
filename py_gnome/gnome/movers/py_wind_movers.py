'''
Wind Movers and associated helper functions.
This module has no compiled C dependency
'''

from . import movers

import numpy as np
import os
import warnings

from colander import (SchemaNode,
                      Bool, Float, String, Sequence, drop)

from gnome.basic_types import oil_status
from gnome.array_types import gat

from gnome.utilities import rand
from gnome.utilities.projections import FlatEarthProjection

from gnome.environment import GridWind

from gnome.movers.movers import TimeRangeSchema, PyMoverSchema

from gnome.persist.validators import convertible_to_seconds
from gnome.persist.extend_colander import LocalDateTime, FilenameSchema
from gnome.persist.base_schema import GeneralGnomeObjectSchema
from gnome.environment.gridded_objects_base import Grid_U, VectorVariableSchema


class WindMoverSchema(PyMoverSchema):
    '''
    Schema for WindMover object
    '''
    wind = GeneralGnomeObjectSchema(save=True, update=True,
                                    save_reference=True,
                                    acceptable_schemas=[VectorVariableSchema,
                                                        GridWind._schema])
    scale_value = SchemaNode(Float(), save=True, update=True, missing=drop)
    #time_offset = SchemaNode(Float(), save=True, update=True, missing=drop)
    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)
    uncertain_duration = SchemaNode(Float())
    uncertain_time_delay = SchemaNode(Float())
    uncertain_speed_scale = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    uncertain_angle_scale = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
PyWindMoverSchema = WindMoverSchema

class WindMover(movers.PyMover):
    '''
    WindMover implemented in Python. Uses the .wind attribute to move particles.
    The .at() interface is expected on the .wind attribute
    '''
    _schema = WindMoverSchema

    _ref_as = 'py_wind_movers'

    _req_refs = {'wind': GridWind}

    def __init__(self,
                 wind=None,
                 time_offset=0,
                 uncertain_duration=3.* 3600,
                 uncertain_time_delay=0,
                 uncertain_speed_scale=2.,
                 uncertain_angle_scale=0.4,
                 scale_value=1,
                 default_num_method='RK2',
                 filename=None,
                 **kwargs):
        """
        Initialize a WindMover
        :param wind: Environment object representing wind to be
                        used.
        :type wind: Any Wind or Wind-like that implements the .at() function

        :param active_range: Range of datetimes for when the mover should be
                             active
        :type active_range: 2-tuple of datetimes

        :param scale_value: Value to scale wind data
        :param uncertain_duration: how often does a given uncertain element
                                   get reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_speed_scale: Scale for uncertainty of wind speed
        :param uncertain_angle_scale: Scale for uncertainty of wind angle
        :param time_offset: Time zone shift if data is in GMT
        :param num_method: Numerical method for calculating movement delta.
                           Choices:('Euler', 'RK2', 'RK4')
                           Default: RK2

        """

        (super(WindMover, self).__init__(default_num_method=default_num_method, **kwargs))
        self.wind = wind
        self.make_default_refs = False

        if self.wind is None:
            raise ValueError("Must provide a wind object")
        if isinstance(self.wind, (str, os.PathLike)):
            warnings.warn("The behavior of providing a filename to a WindMover __init__ is deprecated. "
                          "Please pass a wind or use a helper function", DeprecationWarning)
            self.wind = GridWind.from_netCDF(filename=self.wind,
                                                 **kwargs)
        if filename is not None:
            warnings.warn("The behavior of providing a filename to a WindMover __init__ is deprecated. "
                          "Please pass a wind or use a helper function", DeprecationWarning)

        self.uncertain_duration = uncertain_duration
        self.uncertain_time_delay = uncertain_time_delay
        self.uncertain_speed_scale = uncertain_speed_scale
        self.uncertain_angle_scale = uncertain_angle_scale
        self.uncertain_diffusion = 0

        self.scale_value = scale_value
        #self.time_offset = time_offset

        self.sigma_theta = 0
        self.sigma2 = 0
        self.is_first_step = False
        self.time_uncertainty_was_set = 0
        self.shape = (2,)
        self.uncertainty_list = np.zeros((0,)+self.shape, dtype=np.float64)

        self.array_types.update({'windages': gat('windages'),
                                 'windage_range': gat('windage_range'),
                                 'windage_persist': gat('windage_persist')})

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    time_offset=0,
                    scale_value=1,
                    uncertain_duration=3 * 3600,
                    uncertain_time_delay=0,
                    uncertain_speed_scale=2.,
                    uncertain_angle_scale=.4,
                    default_num_method='RK2',
                    **kwargs):
        warnings.warn("WindMover.from_netCDF is deprecated. "
                      "Please create the wind separately or use a helper function", DeprecationWarning)

        wind = GridWind.from_netCDF(filename, **kwargs)

        return cls(wind=wind,
                   filename=filename,
                   time_offset=time_offset,
                   scale_value=scale_value,
                   uncertain_speed_scale=uncertain_speed_scale,
                   uncertain_angle_scale=uncertain_angle_scale,
                   default_num_method=default_num_method)

    @property
    def data_start(self):
        return self.wind.data_start

    @property
    def data_stop(self):
        return self.wind.data_stop

    def prepare_for_model_run(self):
        """
        reset uncertainty
        """
        self.is_first_step = True
        self.uncertainty_list = np.zeros((0,)+self.shape, dtype=np.float64)
        self.time_uncertainty_was_set = 0

        return

    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
        """
        Call base class method using super
        Also updates windage for this timestep

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of model as a date time object
        """
        super(WindMover, self).prepare_for_model_step(sc, time_step,
                                                        model_time_datetime)

        # if no particles released, then no need for windage
        # TODO: revisit this since sc.num_released shouldn't be None
        if sc.num_released is None or sc.num_released == 0:
            return

        if self.active:
            rand.random_with_persistance(sc['windage_range'][:, 0],
                                    sc['windage_range'][:, 1],
                                    sc['windages'],
                                    sc['windage_persist'],
                                    time_step)

            seconds = self.datetime_to_seconds(model_time_datetime)
            if self.is_first_step:
                self.model_start_time = seconds	#check units on this

            if sc.uncertain:
                elapsed_time = seconds - self.model_start_time
                eddy_diffusion = 1000000.	#this is fixed, should it be an input?
                self.update_uncertainty(sc.num_released, elapsed_time)
                self.uncertain_diffusion = np.sqrt(6 * (eddy_diffusion / 10000.) / time_step)

        return

    def model_step_is_done(self, sc):
        """
        remove any off map les
        """
        if not self.active or not self.on:
            return

        self.is_first_step = False

        if sc.uncertain:
            to_be_removed = np.where(sc['status_codes'] ==
                                     oil_status.to_be_removed)[0]

            if len(to_be_removed) > 0:
                new_uncertainty = np.copy(self.uncertainty_list)
                self.uncertainty_list = np.delete(new_uncertainty, to_be_removed, axis=0)

    def get_bounds(self):
        '''
            Return a bounding box surrounding the grid data.
            This function exists because it is part of the top level Mover API
        '''
        if hasattr(self.wind, 'get_bounds'):
            return self.wind.get_bounds
        else:
            return super(WindMover, self).get_bounds()

    def update_uncertainty(self, num_les, elapsed_time):
        """
        update uncertainty

        :param num_les: the number released so far
        :param elapsed_time: time in seconds since model run started
        """
        need_to_reinit = False
        need_to_reallocate = False

        add_uncertainty = elapsed_time >= self.uncertain_time_delay

        if not add_uncertainty:
            #I'm not sure if we have to do anything here # we will not be adding uncertainty
            #self.uncertainty_list = np.zeros((0,)+self.shape, dtype=np.float64)
            #self.time_uncertainty_was_set = 0
            return

        uncertain_list_size = len(self.uncertainty_list)

        if uncertain_list_size==0:
            need_to_reinit = True

        if elapsed_time < self.time_uncertainty_was_set:
            # this shouldn't happen
            need_to_reinit = True

        if num_les > uncertain_list_size:
            need_to_reallocate = True

        if num_les < uncertain_list_size: # this shouldn't happen unless a reset was missed
            need_to_reinit = True

        if need_to_reallocate and uncertain_list_size!=0:
            a_append = np.zeros((num_les-uncertain_list_size,)+self.shape,dtype=np.float64)
            cos_arg = np.zeros((num_les-uncertain_list_size,)+self.shape,dtype=np.float64)
            srt = np.zeros((num_les-uncertain_list_size,)+self.shape,dtype=np.float64)
            cos_arg = 2. * np.pi * np.random.uniform(0,1, size=(num_les-uncertain_list_size,))
            srt = np.sqrt(-2. * np.log(np.random.uniform(0.001,.999, size=(num_les-uncertain_list_size,))))
            # need a loop to check TermsLessThanMax fabs(self.sigma_theta * sinTerm/rndv2) <= angleMax (60)
            for i in range(num_les-uncertain_list_size):
                for j in range(10):
                    if np.abs(self.sigma_theta * srt[i] * np.sin(cos_arg[i])) <= 60.:
                        break
                        cos_arg[i] = 2. * np.pi * np.random.uniform(0,1)
                        srt[i] = np.sqrt(-2. * np.log(np.random.uniform(0.001,.999)))

            a_append[:,0] = srt * np.cos(cos_arg) #cos term
            a_append[:,1] = srt * np.sin(cos_arg) #sin term
            self.uncertainty_list = np.r_[self.uncertainty_list, a_append]

        # question - should self.sigma2 change only when the duration value is exceeded ??
        # or every step as it does now ??
        self.sigma2 = self.uncertain_speed_scale * .315 * np.power(elapsed_time - self.uncertain_time_delay, .147)
        self.sigma2 = self.sigma2 * self.sigma2 / 2.

        self.sigma_theta = self.uncertain_angle_scale * 2.73 * np.sqrt(np.sqrt(elapsed_time - self.uncertain_time_delay))

        if need_to_reinit:
            self.allocate_uncertainty(num_les)
            self.update_uncertainty_values(elapsed_time)
        elif elapsed_time >= self.time_uncertainty_was_set + self.uncertain_duration:
            self.update_uncertainty_values(elapsed_time)

        return


    def update_uncertainty_values(self, elapsed_time):
        """
        update uncertainty values

        :param elapsed_time: time in seconds since model run started
        """
        self.time_uncertainty_was_set = elapsed_time
        num_les = len(self.uncertainty_list)
        if num_les==0:
            return

        cos_arg = np.zeros((num_les,)+self.shape,dtype=np.float64)
        srt = np.zeros((num_les,)+self.shape,dtype=np.float64)
        cos_arg = 2. * np.pi * np.random.uniform(0,1, size=(num_les,))
        srt = np.sqrt(-2. * np.log(np.random.uniform(0.001,.999, size=(num_les,))))
        # need a loop to check TermsLessThanMax: fabs(self.sigma_theta * sinTerm/rndv2) <= angleMax (60)
        for i in range(0,num_les):
            for j in range(10):
                if np.abs(self.sigma_theta * srt[i] * np.sin(cos_arg[i])) <= 60.:
                    break
                    cos_arg[i] = 2. * np.pi * np.random.uniform(0,1)
                    srt[i] = np.sqrt(-2. * np.log(np.random.uniform(0.001,.999)))

        self.uncertainty_list[:,0] = srt * np.cos(cos_arg) #cos term
        self.uncertainty_list[:,1] = srt * np.sin(cos_arg) #sin term


    def allocate_uncertainty(self, num_les):
        """
        add uncertainty

        :param num_les: the number of les released so far
        """
        shape = (2,)
        self.uncertainty_list = np.zeros((num_les,)+shape, dtype=np.float64)

        return


    def add_uncertainty(self, deltas, time_step):
        """
        add uncertainty

        :param deltas: the movement for the current time step
        """
        if self.uncertainty_list is None:
            return deltas # this is our clue to not add uncertainty

        num_les = len(self.uncertainty_list)
        if len(self.uncertainty_list)>0:
            #make a copy of deltas
            new_deltas=deltas.copy()
            unrec=self.uncertainty_list
            u = new_deltas[:,0] / time_step
            v = new_deltas[:,1] / time_step
            norm = np.sqrt(u*u + v*v)

            s = norm * norm - self.sigma2
            s = np.clip(s,a_min=0,a_max=None)
            sqs = np.sqrt(s)
            m = np.sqrt(sqs)
            s2 = np.sqrt(norm - sqs)
            rand_cos = unrec[:,0]	#cos term
            rand_sin = unrec[:,1]	#sin term
            x = rand_cos * s2 + m
            w = x * x
            dtheta = rand_sin * self.sigma_theta * np.pi / 180.
            cos_theta = np.cos(dtheta)
            sin_theta = np.sin(dtheta)
            #w = w / (cos_theta < .001 ? .001 : costheta) # compensate for projection vector effect
            norm_clip = np.clip(norm,a_min=0.001,a_max=None)
            cos_theta_clip = np.clip(cos_theta,a_min=0.001,a_max=None)
            w = w / cos_theta_clip  # compensate for projection vector effect

            # Scale pattern velocity to have norm w
            t = w / norm_clip
            u *= t
            v *= t

            # Rotate velocity by dtheta
            deltas[:,0] = (u * cos_theta - v * sin_theta) * time_step
            deltas[:,1] = (v * cos_theta + u * sin_theta) * time_step

            for i in range(0,num_les):
                if norm[i] < 1:
                    rand1 = np.random.uniform(-1., 1.)
                    rand2 = np.random.uniform(-1., 1.)
                    deltas[i,0] = u[i] * self.uncertain_diffusion * rand1
                    deltas[i,1] = v[i] * self.uncertain_diffusion * rand2

        else:
            raise ValueError("something wrong with uncertainty")

        return deltas


    def get_move(self, sc, time_step, model_time_datetime, num_method=None):
        """
        Compute the move in (long,lat,z) space. It returns the delta move
        for each element of the spill as a numpy array of size
        (number_elements X 3) and dtype = gnome.basic_types.world_point_type

        Base class returns an array of numpy.nan for delta to indicate the
        get_move is not implemented yet.

        Each class derived from Mover object must implement it's own get_move

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current model time as datetime object

        All movers must implement get_move() since that's what the model calls
        """
        positions = sc['positions']

        if self.active and len(positions) > 0:
            status = sc['status_codes'] != oil_status.in_water
            pos = positions[:]

            deltas = self.delta_method(num_method)(sc, time_step, model_time_datetime, pos, self.wind)

            if sc.uncertain:
                deltas = self.add_uncertainty(deltas, time_step)

            deltas[:, 0] *= sc['windages'] * self.scale_value
            deltas[:, 1] *= sc['windages'] * self.scale_value

            deltas = FlatEarthProjection.meters_to_lonlat(deltas, positions)

            deltas[status] = (0, 0, 0)
        else:
            deltas = np.zeros_like(positions)

        return deltas
PyWindMover = WindMover

def grid_wind_mover(filename, wind_kwargs=None, *args, **kwargs):
    '''
    Helper function to load a gridded wind from a file and create a WindMover

    :param filename: File to create the GridWind object from
    :type filename: string or Path-like'

    :param wind_kwargs: keyword arguments for the GridWind object. OPTIONAL
    :type wind_kwargs: dict
    '''
    wind = GridWind.from_netCDF(filename=filename, **wind_kwargs)
    return WindMover(wind=wind, **kwargs)
