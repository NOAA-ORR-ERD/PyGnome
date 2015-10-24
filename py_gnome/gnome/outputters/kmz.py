"""
kmz  outputter
"""

import copy
import os
from glob import glob

import numpy as np
from datetime import timedelta


from colander import SchemaNode, String, drop, Int, Bool

from gnome.utilities.time_utils import date_to_sec
from gnome.utilities.serializable import Serializable, Field

from gnome.persist import class_from_objtype
from gnome.basic_types import oil_status

from .outputter import Outputter, BaseSchema
from . import kmz_templates


class KMZSchema(BaseSchema):
    '''
    Nothing is required for initialization
    '''
#    round_data = SchemaNode(Bool(), missing=drop)
#    round_to = SchemaNode(Int(), missing=drop)
    output_dir = SchemaNode(String(), missing=drop)


class KMZOutput(Outputter, Serializable):
    '''
    class that outputs GNOME results in a kmz format.

    Suitable for Google Earth, and semi-suitable for MarPlot

    '''
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _state += [Field('output_dir', update=True, save=True),]
    _schema = KMZSchema

    def __init__(self, filename, **kwargs):
        '''
        :param str output_dir=None: output directory for kmz files.

        uses super to pass optional \*\*kwargs to base class __init__ method
        '''
        ## a little check:
        self._check_filename(filename)
        self.filename = filename if filename[-4:] == ".kml" else filename + ".kml"

        super(KMZOutput, self).__init__(**kwargs)

    def prepare_for_model_run(self,
                              model_start_time,
                              spills,
                              **kwargs):
        """
        .. function:: prepare_for_model_run(model_start_time,
                                            cache=None,
                                            uncertain=False,
                                            spills=None,
                                            **kwargs)

        Write the headers, png files, etc for the KMZ file

        This must be done in prepare_for_model_run because if model _state
        changes, it is rewound and re-run from the beginning.

        If there are existing output files, they are deleted here.

        This takes more than standard 'cache' argument. Some of these are
        required arguments - they contain None for defaults because non-default
        argument cannot follow default argument. Since cache is already 2nd
        positional argument for Renderer object, the required non-default
        arguments must be defined following 'cache'.

        If uncertainty is on, then SpillContainerPair object contains
        identical _data_arrays in both certain and uncertain SpillContainer's,
        the data itself is different, but they contain the same type of data
        arrays. If uncertain, then data arrays for uncertain spill container
        are written to the KMZ file.

        .. note::
            Does not take any other input arguments; however, to keep the
            interface the same for all outputters, define kwargs in case
            future outputters require different arguments.
        """
        super(KMZOutput, self).prepare_for_model_run(model_start_time,
                                                     spills,
                                                     **kwargs)

        self.delete_output_files()
        # shouldn't be required if the above worked!
        self._file_exists_error(self.filename)

        # create the kmz files and write the standard stuff:
        #Here's the real work!
        with open(self.filename, 'wb') as kmz_file: # note: file is closed when done -- must be re-opened to write each timestep
            #just the header
            kmz_file.write(kmz_templates.header_template)
        # # netcdf outputter has this --  not sure why
        # self._middle_of_run = True

    def write_output(self, step_num, islast_step=False):
        """dump a timestep's data into the kmz file"""

        super(KMZOutput, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        # Open the file for appending..
        with open(self.filename, 'a') as kmz_file: # note: file is closed after header is written
            for sc in self.cache.load_timestep(step_num).items(): # loop through uncertain and certain LEs
                ## extract the data
                start_time = sc.current_time_stamp
                if self.output_timestep is None:
                    end_time = start_time + timedelta(seconds = self.model_timestep)
                else:
                    end_time = start_time + self.output_timestep
                start_time = start_time.isoformat()
                end_time = end_time.isoformat()

                positions = sc['positions']
                water_positions = positions[sc['status_codes']   == oil_status.in_water]
                beached_positions = positions[sc['status_codes'] == oil_status.on_land]

                data_dict = {'certain' : "Uncertainty"if sc.uncertain else "Best Guess",
                            }
                kmz_file.write(kmz_templates.build_one_timestep(water_positions,
                                                                beached_positions,
                                                                start_time,
                                                                end_time,
                                                                sc.uncertain
                                                                ))
            if islast_step: # close out the file
                kmz_file.write(kmz_templates.footer)






        # output_filename = self.output_to_file(geojson, step_num)
        output_info = {'time_stamp': sc.current_time_stamp.isoformat(),
                       'output_filename': self.filename}

        return output_info


    def rewind(self):
        '''
        reset a few parameter and call base class rewind to reset
        internal variables.
        '''
        super(KMZOutput, self).rewind()

        self._middle_of_run = False
        self._start_idx = 0

    def delete_output_files(self):
        '''
        deletes ouput files that may be around

        called by prepare_for_model_run

        here in case it needs to be called from elsewhere
        '''
        try:
            os.remove(self.filename)
        except OSError:
            pass # it must not be there




