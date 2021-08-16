
import numpy as np
from collections.abc import MutableMapping
import warnings

from gnome.gnomeobject import AddLogger
from gnome.array_types import default_array_types


class LEData(MutableMapping, AddLogger, dict):
    # Fixme: we need a docstring here!

    def __init__(self, name=None, *args, **kwargs):
        super(LEData, self).__init__(*args, **kwargs)
        self.mass_balance = {}
        self._array_types = {}
        self._array_types.update(default_array_types)
        self._bufs = {}
        self._arrs = {}
        self._arrs.update(dict(*args, **kwargs))
        self._initialized = False
        if not name:
            name = 'LEData'
        self.name = name

    def rewind(self):
        self.mass_balance.clear()
        self._array_types.clear()
        self._array_types.update(default_array_types)
        self._bufs.clear()
        self._arrs.clear()
        self._initialized = False

    clear = rewind

    def __eq__(self, other):
        'Compare equality of two LEData objects'
        # check key/val that are not dicts
        t1 = self._array_types == other._array_types
        t2 = super(LEData, self).__eq__(other)
        return t1 and t2

    def __ne__(self, other):
        return not (self == other)

    def __len__(self):
        """
        The "length" of a spill container is the number of elements in it.
        The first dimension of any ndarray in our data_arrays
        will always be the number of elements that are contained in a
        SpillContainer.
        """
        return 0 if len(self._arrs) == 0 else len(next(iter(self._arrs.values())))

    def __getitem__(self, key):
        return self._arrs[key]

    def __setitem__(self, key, value):
        if key in self._arrs:
            warnings.warn('Replacing existing LEData array references directly'
                          'is dangerous! If this *really* is what you want to do, '
                          'please use the _set_existing_LEData(key, value) '
                          'function instead')
        self._arrs[key] = value

    def _set_existing_LEData(self, key, value):
        self._arrs[key] = value

    def __delitem__(self, key):
        del self._arrs[key]

    def __iter__(self):
        return iter(self._arrs)

    def __repr__(self):
        return self.name

    __str__ = __repr__

    @property
    def num_released(self):
        """
        The number of elements currently in the SpillContainer

        If SpillContainer is initialized, all data_arrays exist as ndarrays
        even if no elements are released.  So this will always return a valid
        int >= 0.
        """
        return len(self)

    def prepare_for_model_run(self, array_types, substance):
        self._array_types.update(array_types)
        if hasattr(substance, 'num_oil_components'):
            self.num_oil_components = substance.num_oil_components
        else:
            self.num_oil_components = 1
        self.initialize_data_arrays()

    def initialize_data_arrays(self):
        """
        initialize_data_arrays() is called without input data during rewind
        and prepare_for_model_run to prepare the buffers and define the data
        arrays.
        """
        if self._initialized:
            warnings.warn('{0} is already initialized.'.format(self))
        for name, atype in self._array_types.items():
            if atype.shape is None:
                shape = (self.num_oil_components, )
            else:
                shape = atype.shape
            self._bufs[name] = np.zeros((100,) + shape, dtype=atype.dtype)
            self._bufs[name][:] = atype.initial_value
            self[name] = self._bufs[name][0:0]
        self._initialized = True

    def extend_data_arrays(self, num_new_elements):
        """
        Adds num_new_elements to the data arrays. Resizes the arrays as
        necessary. Data arrays are set to their initial_values

        :param int num_new_elements: number of particles released

        """
        if not self._initialized:
            raise ValueError("Must initialize spill data arrays before extending is possible")
        for name, atype in self._array_types.items():
            # initialize all arrays even if 0 length
            if atype.shape is None:
                shape = (self.num_oil_components, )
            else:
                shape = atype.shape
            buf = self._bufs[name]
            new_buflen = len(self[name]) + num_new_elements
            if len(buf) < new_buflen:
                #need to resize. Use double the new_buflen
                oldbuf = self._bufs[name]
                self._bufs[name] = np.resize(oldbuf, (2*new_buflen, ) + oldbuf.shape[1:])
                self._bufs[name][len(oldbuf):] = atype.initial_value

            self._set_existing_LEData(name, self._bufs[name][0:new_buflen])
