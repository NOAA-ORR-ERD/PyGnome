from . import movers

import numpy as np
import os
import warnings

from colander import (SchemaNode, Bool, Float, drop)

from gnome.basic_types import oil_status
# from gnome.basic_types import (world_point_type,
#                                status_code_type)

from gnome.utilities.projections import FlatEarthProjection

from gnome.environment import GridCurrent
from gnome.environment.gridded_objects_base import Grid_U, VectorVariableSchema

from gnome.movers.movers import TimeRangeSchema, PyMoverSchema

from gnome.persist.validators import convertible_to_seconds
from gnome.persist.extend_colander import LocalDateTime, FilenameSchema
from gnome.persist.base_schema import GeneralGnomeObjectSchema


class CurrentMoverSchema(PyMoverSchema):
    '''
    Schema for CurrentMover object
    '''
    current = GeneralGnomeObjectSchema(save=True, update=True,
                                       save_reference=True,
                                       acceptable_schemas=[VectorVariableSchema,
                                                           GridCurrent._schema]
                                       )
    scale_value = SchemaNode(Float(), save=True, update=True,
                               missing=drop)
    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)
    uncertain_duration = SchemaNode(Float())
    uncertain_time_delay = SchemaNode(Float())
    uncertain_along = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    uncertain_cross = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
PyCurrentMoverSchema = CurrentMoverSchema
class CurrentMover(movers.PyMover):
    '''
    CurrentMover implemented in Python. Uses the .current attribute to move particles.
    The .at() interface is expected on the .current attribute
    '''

    _schema = CurrentMoverSchema

    _ref_as = 'py_current_movers'

    _req_refs = {'current': GridCurrent}

    def __init__(self,
                 current=None,
                 time_offset=0,
                 scale_value=1,
                 uncertain_duration= 24 * 3600,
                 uncertain_time_delay=0,
                 uncertain_along=.5,
                 #uncertain_across=.25,
                 uncertain_cross=.25,
                 default_num_method='RK2',
                 filename=None,
                 **kwargs
                 ):
        """
        Initialize a CurrentMover

        :param current: Environment object representing ocean currents to be
                        used.
        :type current: Any Current or Current-like that implements the .at() function

        :param active_range: Range of datetimes for when the mover should be
                             active
        :type active_range: 2-tuple of datetimes

        :param scale_value: Value to scale current data
        :param uncertain_duration: how often does a given uncertain element
                                   get reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_cross: Scale for uncertainty perpendicular to the flow
        :param uncertain_along: Scale for uncertainty parallel to the flow
        :param time_offset: Time zone shift if data is in GMT
        :param default_num_method: Numerical method for calculating movement delta.
                                   Choices:('Euler', 'RK2', 'RK4')
                                   Default: RK2
        """

        (super(CurrentMover, self).__init__(default_num_method=default_num_method,
                                              **kwargs))
        self.filename = filename
        self.current = current


        if self.current is None:
            raise ValueError("Must provide a current object")
        if isinstance(self.current, (str, os.PathLike)):
            warnings.warn("The behavior of providing a filename to a CurrentMover __init__ is deprecated. Please pass a current or use a helper function", DeprecationWarning)
            self.current = GridCurrent.from_netCDF(filename=self.current,
                                                 **kwargs)
        if filename is not None:
            warnings.warn("The behavior of providing a filename to a CurrentMover __init__ is deprecated. Please pass a current or use a helper function", DeprecationWarning)

        self.scale_value = scale_value

        self.uncertain_along = uncertain_along
        self.uncertain_cross = uncertain_cross
        self.uncertain_duration = uncertain_duration
        self.uncertain_time_delay = uncertain_time_delay

        self.model_time = 0

        # either a 1, or 2 depending on whether spill is certain or not
        self.spill_type = 0

        self.is_first_step = False
        self.time_uncertainty_was_set = 0
        self.shape = (2,)
        self.uncertainty_list = np.zeros((0,)+self.shape, dtype=np.float64)

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    name=None,
                    time_offset=0,
                    scale_value=1,
                    uncertain_duration=24 * 3600,
                    uncertain_time_delay=0,
                    uncertain_along=.5,
                    #uncertain_across=.25,
                    uncertain_cross=.25,
                    **kwargs):
        """
        Function for specifically creating a CurrentMover from a file
        """
        warnings.warn("CurrentMover.from_netCDF is deprecated. Please create the current separately or use a helper function", DeprecationWarning)
        current = GridCurrent.from_netCDF(filename, **kwargs)

        return cls(name=name,
                   current=current,
                   filename=filename,
                   time_offset=time_offset,
                   scale_value=scale_value,
                   uncertain_along=uncertain_along,
                   #uncertain_across=uncertain_across,
                   uncertain_cross=uncertain_cross,
                   **kwargs)

    @property
    def filename(self):
        if hasattr(self, '_filename'):
            if self._filename is None and self.current is not None:
                return self.current.data_file
            else:
                return self._filename
        else:
            return None

    @filename.setter
    def filename(self, fn):
        self._filename = fn

    @property
    def data_start(self):
        return self.current.data_start

    @property
    def data_stop(self):
        return self.current.data_stop

    def get_bounds(self):
        '''
            Return a bounding box surrounding the grid data.
            This function exists because it is part of the top level Mover API
        '''
        if hasattr(self.current, 'get_bounds'):
            return self.current.get_bounds
        else:
            return super(CurrentMover, self).get_bounds()

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

            res = self.delta_method(num_method)(sc, time_step,
                                                model_time_datetime,
                                                pos,
                                                self.current)

            if res.shape[1] == 2:
                deltas = np.zeros_like(positions)
                deltas[:, 0:2] = res
            else:
                deltas = res

            deltas *= self.scale_value
            deltas = FlatEarthProjection.meters_to_lonlat(deltas, positions)

            if sc.uncertain:
                deltas = self.add_uncertainty(deltas)

            deltas[status] = (0, 0, 0)
        else:
            deltas = np.zeros_like(positions)

        return deltas


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
            need_to_reinit = True

        if num_les>uncertain_list_size:
            need_to_reallocate = True

        if num_les<uncertain_list_size: # this shouldn't happen unless a reset was missed
            need_to_reinit = True

        if need_to_reallocate and uncertain_list_size!=0:
            a_append = np.zeros((num_les-uncertain_list_size,)+self.shape,dtype=np.float64)
            a_append[:,0] = np.random.uniform(-self.uncertain_along, self.uncertain_along, size=(num_les-uncertain_list_size,))
            a_append[:,1] = np.random.uniform(-self.uncertain_cross, self.uncertain_cross, size=(num_les-uncertain_list_size,))
            self.uncertainty_list = np.r_[self.uncertainty_list, a_append]
#             for i in range(uncertain_list_size,num_les):
#                 self.uncertainty_list[i:,0] = np.random.uniform(-self.uncertain_along, self.uncertain_along)
#                 self.uncertainty_list[i:,1] = np.random.uniform(-self.uncertain_cross, self.uncertain_cross)

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

        self.uncertainty_list[:,0] = np.random.uniform(-self.uncertain_along, self.uncertain_along, size=(num_les,))
        self.uncertainty_list[:,1] = np.random.uniform(-self.uncertain_cross, self.uncertain_cross, size=(num_les,))


    def allocate_uncertainty(self, num_les):
        """
        add uncertainty

        :param num_les: the number of les released so far
        """
        shape = (2,)
        self.uncertainty_list = np.zeros((num_les,)+shape, dtype=np.float64)

        return


    def add_uncertainty(self, deltas):
        """
        add uncertainty

        :param deltas: the movement for the current time step
        """
        if self.uncertainty_list is None:
            return deltas # this is our clue to not add uncertainty

        if len(self.uncertainty_list)>0:
            #make a copy of deltas
            new_deltas=deltas.copy()
            unrec=self.uncertainty_list
            u = new_deltas[:,0]
            v = new_deltas[:,1]
            lengthS = np.sqrt(u*u + v*v)

            #alpha = unrec.downStream
            #beta = unrec.crossStream
            alpha = unrec[:,0]	#downstream
            beta = unrec[:,1]	#crossstream

            # Gnome had a minimum value case for lengthS - uncertMinimumInMPS
            deltas[:,0] = u*(1+alpha)+v*beta
            deltas[:,1] = v*(1+alpha)-u*beta

        else:
            raise ValueError("something wrong with uncertainty")

        return deltas


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
        add uncertainty
        """
        super(CurrentMover, self).prepare_for_model_step(sc, time_step,
                                                           model_time_datetime)

        if not self.active:
            return

        seconds = self.datetime_to_seconds(model_time_datetime)
        if self.is_first_step:
            self.model_start_time = seconds	#check units on this

        if sc.uncertain:
            elapsed_time = seconds - self.model_start_time
            self.update_uncertainty(sc.num_released, elapsed_time)

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
                #self.uncertainty_list = np.delete(self.uncertainty_list, to_be_removed, axis=0)
                self.uncertainty_list = np.delete(new_uncertainty, to_be_removed, axis=0)
PyCurrentMover = CurrentMover

def grid_current_mover(filename, current_kwargs=None, *args, **kwargs):
    '''
    Helper function to load a gridded current from a file and create a CurrentMover

    :param filename: File to create the GridCurrent object from
    :type filename: string or Path-like'

    :param current_kwargs: keyword arguments for the GridCurrent object. OPTIONAL
    :type current_kwargs: dict
    '''
    current = GridCurrent.from_netCDF(filename=filename, **current_kwargs)
    return CurrentMover(current=current, **kwargs)