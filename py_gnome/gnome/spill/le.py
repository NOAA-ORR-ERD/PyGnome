import numpy as np

from gnome.gnomeobject import GnomeId
from gnome.array_types import _default_values, default_array_types

class LEData(dict):
    """

    """
    def __init__(self, *args, **kwargs):
        super(LEData, self).__init__(*args, **kwargs)
        self._array_types = {}
        self._array_types.update(default_array_types)
        self._bufs = {}
        self._len = 0

    def rewind(self):
        self.clear()
        self._array_types = {}
        self._array_types.update(default_array_types)
        self._bufs = {}
        self._len = 0

    def __eq__(self, other):
        'Compare equality of two LEData objects'
        # check key/val that are not dicts
        return all(self._array_types == other._array_types) and super(LEData, self).__eq__(other)

    def __ne__(self, other):
        return not (self == other)

    def __len__(self):
        """
        The "length" of a spill container is the number of elements in it.
        The first dimension of any ndarray in our data_arrays
        will always be the number of elements that are contained in a
        SpillContainer.
        """
        return 0 if len(self.values()) is 0 else len(self.values()[0])

    @property
    def num_released(self):
        """
        The number of elements currently in the SpillContainer

        If SpillContainer is initialized, all data_arrays exist as ndarrays
        even if no elements are released.  So this will always return a valid
        int >= 0.
        """
        return len(self)

    def prepare_for_model_run(self, array_types, num_oil_components):
        self._array_types.update(array_types)
        self.num_oil_components = num_oil_components
        self.initialize_data_arrays()

    def initialize_data_arrays(self):
        """
        initialize_data_arrays() is called without input data during rewind
        and prepare_for_model_run to define all data arrays.
        At this time the arrays are empty.
        """
        for name, atype in self._array_types.items():
            if atype.shape is None:
                shape = (self.num_oil_components, )
            else:
                shape = atype.shape
            self._bufs[name] = np.zeros((100,) + shape, dtype=atype.dtype)
            self._bufs[name][:] = atype.initial_value
            self[name] = self._bufs[name][0:0]

    def _extend_data_arrays(self, num_new_elements):
        """
        initialize data arrays once spill has spawned particles
        Data arrays are set to their initial_values

        :param int num_released: number of particles released

        """
        for name, atype in self._array_types.items():
            # initialize all arrays even if 0 length
            if atype.shape is None:
                shape = (self.num_oil_components, )
            else:
                shape = atype.shape
            buf = self._bufs[name]
            new_buflen = len(self) + num_new_elements
            if len(buf) < new_buflen:
                #need to resize. Use double the new_buflen
                oldbuf = self._bufs[name]
                self._bufs[name] = np.resize(oldbuf, (2*new_buflen, ) + oldbuf.shape[1:])
                self._bufs[name][len(oldbuf):] = atype.initial_value

            self[name] = self._bufs[name][0:new_buflen]
