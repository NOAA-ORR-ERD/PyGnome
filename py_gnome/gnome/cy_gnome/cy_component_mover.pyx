import os

cimport numpy as cnp
import numpy as np

from type_defs cimport *
from movers cimport ComponentMover_c, CATSMover_c, Mover_c
from gnome import basic_types
from gnome.cy_gnome.cy_ossm_time cimport CyOSSMTime
#from gnome.cy_gnome.cy_shio_time cimport CyShioTime
from gnome.cy_gnome cimport cy_mover
from gnome.cy_gnome.cy_helpers cimport to_bytes

"""
Dynamic casts are not currently supported in Cython - define it here instead.
Since this function is custom for each mover, just keep it with the definition for each mover
"""
cdef extern from *:
    ComponentMover_c* dynamic_cast_ptr "dynamic_cast<ComponentMover_c *>" (Mover_c *) except NULL
    #CATSMover_c* dynamic_cast_ptr "dynamic_cast<CATSMover_c *>" (Mover_c *) except NULL


cdef class CyComponentMover(cy_mover.CyMover):

    cdef ComponentMover_c *component
    #cdef CATSMover_c *cats_pattern1
    #cdef CATSMover_c *cats_pattern2
    #cdef CATSMover_c *catsmover1
    #cdef CATSMover_c *catsmover2
    
    def __cinit__(self):
        self.mover = new ComponentMover_c()
        self.component = dynamic_cast_ptr(self.mover)
    
    def __dealloc__(self):
        del self.mover  # since this is allocated in this class, free memory here as well
        self.component = NULL
    
    def text_read(self, cats_file1, cats_file2=None):
        """
        .. function::text_read

        """
        cdef OSErr err
        cdef bytes pattern1, pattern2

        cats_file1 = os.path.normpath(cats_file1)
        pattern1 = to_bytes(unicode(cats_file1))

        if cats_file2 is None:
            err = self.component.TextRead(pattern1, '')
        else:
            cats_file2 = os.path.normpath(cats_file2)
            pattern2 = to_bytes(unicode(cats_file2))
            err = self.component.TextRead(pattern1, pattern2)

        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors
            are defined and enumerated
            """
            raise OSError("ComponentMover_c.TextRead returned an error.")

    def __init__(self, pat1_angle=0, pat1_speed=10, pat1_speed_units = 2, pat1_scale_to_value=.1):
        """
        Initialize the CyComponentMover which sets the properties for the underlying C++ ComponentMover_c object

        :param pat1_angle=0: The angle for pattern 1
        :param pat1_speed=10: 
        :param pat1_speed_units=m/s: Speed units
        :param pat1_scale_to_value=0.1: 

		Optional parameters:
        :param pat2_angle=0: The angle for pattern 2
        :param pat2_speed=10: 
        :param pat2_speed_units=m/s: Speed units
        :param pat2_scale_to_value=0.1: 

        """
        cdef WorldPoint p
        self.component.pat1Angle = pat1_angle
        self.component.pat1Speed = pat1_speed
        self.component.pat1SpeedUnits = pat1_speed_units
        self.component.pat1ScaleToValue = pat1_scale_to_value
        #self.catsmover1 = NULL
        #self.catsmover2 = NULL

    property pat1_angle:
        def __get__(self):
            return self.component.pat1Angle
        
        def __set__(self,value):
            self.component.pat1Angle = value
    
    property pat1_speed:
        def __get__(self):
            return self.component.pat1Speed
        
        def __set__(self,value):
            self.component.pat1Speed = value
            
    property pat1_speed_units:
        def __get__(self):
            return self.component.pat1SpeedUnits
        
        def __set__(self,value):
            self.component.pat1SpeedUnits = value
            
    property pat1_scale_to_value:
        def __get__(self):
            return self.component.pat1ScaleToValue
        
        def __set__(self,value):
            self.component.pat1ScaleToValue = value    
    
    property pat2_angle:
        def __get__(self):
            return self.component.pat2Angle
        
        def __set__(self,value):
            self.component.pat2Angle = value
    
    property pat2_speed:
        def __get__(self):
            return self.component.pat2Speed
        
        def __set__(self,value):
            self.component.pat2Speed = value
            
    property pat2_speed_units:
        def __get__(self):
            return self.component.pat2SpeedUnits
        
        def __set__(self,value):
            self.component.pat2SpeedUnits = value
            
    property pat2_scale_to_value:
        def __get__(self):
            return self.component.pat2ScaleToValue
        
        def __set__(self,value):
            self.component.pat2ScaleToValue = value    
    
    property ref_point:
        def __get__(self):
            """
            returns the tuple containing (long, lat) of reference point if it is defined
            by the user otherwise it returns None 
            
            """
            return (self.component.refP.pLong/1.e6, self.component.refP.pLat/1.e6)
    
        def __set__(self,ref_point):
            """
            accepts a list or a tuple
            will not work with a numpy array since indexing assumes a list or a tuple
            
            takes only (long, lat), if length is bigger than 2, it uses the 1st 2 datapoints
            
            """
            ref_point = np.asarray(ref_point)   # make it a numpy array
            cdef WorldPoint p
            p.pLong = ref_point[0]*10**6    # should this happen in C++?
            p.pLat = ref_point[1]*10**6
            self.component.SetRefPosition(p)
         
    def __repr__(self):
        """
        Return an unambiguous representation of this object so it can be recreated
        
        Probably want to return filename as well  
        """
        repr_ = '{0}(pat1_angle={1.pat1_angle}, pat1_speed={1.pat1_speed}, pat1_speed_units={1.pat1_speed_units}, pat1_scale_to_value={1.pat1_scale_to_value})'.format(self.__class__.__name__, self)
        return repr_
      
    def __str__(self):
        """Return string representation of this object"""
        
        info  = "{0} object - see attributes for more info\n".format(self.__class__.__name__)
        info += "  pattern angle = {0.pat1_angle}\n".format(self)
        info += "  pattern speed = {0.pat1_speed}\n".format(self)
        info += "  pattern speed units = {0.pat1_speed_units}\n".format(self)
        info += "  pattern scale to = {0.pat1_scale_to_value}\n".format(self)
        
        return info
         
    def set_ossm(self, CyOSSMTime ossm):
        """
        Takes a CyOSSMTime object as input and sets C++ Component mover properties from the OSSM object.
        """
        self.component.SetTimeFile(ossm.time_dep)
        #self.cats.bTimeFileActive = True   # What is this?
        return True
            
#     def text_read(self, cats_pattern1, cats_pattern2=None):
#         """
#         read the current files
#         """
#         cdef OSErr err
#         cdef bytes path1_
#         cdef bytes path2_
#         
#         fname1 = os.path.normpath(cats_pattern1)
#         path1_ = to_bytes( unicode(cats_pattern1))
#         
#         if os.path.exists(path1_):
#             self.catsmover1 = new CATSMover_c()
#             #self.cats_pattern1 = dynamic_cast_ptr(self.catsmover1)
#             #err = self.cats_pattern1.TextRead(path1_)
#             err = self.catsmover1.TextRead(path1_)
#             if err != False:
#                 raise ValueError("CATSMover.text_read(..) returned an error. OSErr: {0}".format(err))
#         else:
#             raise IOError("No such file: " + path1_)
#         
#         if os.path.exists(cats_pattern2):
#             fname2 = os.path.normpath(cats_pattern2)
#             path2_ = to_bytes( unicode(cats_pattern2))
#             if os.path.exists(path2_):
#                 self.catsmover2 = new CATSMover_c()
#                 #self.cats_pattern2 = dynamic_cast_ptr(self.catsmover2)
#                 #err = self.cats_pattern2.TextRead(path2_)
#                 err = self.catsmover2.TextRead(path2_)
#                 if err != False:
#                     raise ValueError("CATSMover.text_read(..) returned an error. OSErr: {0}".format(err))
#             else:
#                 raise IOError("No such file: " + path2_)
# 
#         return True
#     

    def get_move(self, 
                 model_time, 
                 step_len, 
                 cnp.ndarray[WorldPoint3D, ndim=1] ref_points, 
                 cnp.ndarray[WorldPoint3D, ndim=1] delta, 
                 cnp.ndarray[short] LE_status, 
                 LEType spill_type):
        """
        .. function:: get_move(self,
                 model_time,
                 step_len,
                 cnp.ndarray[WorldPoint3D, ndim=1] ref_points,
                 cnp.ndarray[WorldPoint3D, ndim=1] delta,
                 cnp.ndarray[cnp.npy_double] windages,
                 cnp.ndarray[short] LE_status,
                 LEType LE_type)
                 
        Invokes the underlying C++ ComponentMover_c.get_move(...)
        
        :param model_time: current model time
        :param step_len: step length over which delta is computed
        :param ref_points: current locations of LE particles
        :type ref_points: numpy array of WorldPoint3D
        :param delta: the change in position of each particle over step_len
        :type delta: numpy array of WorldPoint3D
        :param LE_windage: windage to be applied to each particle
        :type LE_windage: numpy array of numpy.npy_int16
        :param le_status: status of each particle - movement is only on particles in water
        :param spill_type: LEType defining whether spill is forecast or uncertain 
        :returns: none
        """
        cdef OSErr err
            
        N = len(ref_points)
 
        err = self.component.get_move(N, model_time, step_len, &ref_points[0], &delta[0], &LE_status[0], spill_type, 0)
        if err == 1:
            raise ValueError("Make sure numpy arrays for ref_points and deltas are defined")
        
        """
        Can probably raise this error before calling the C++ code - but the C++ also throwing this error
        """
        if err == 2:
            raise ValueError("The value for spill type can only be 'forecast' or 'uncertainty' - you've chosen: " + str(spill_type))
        
            
