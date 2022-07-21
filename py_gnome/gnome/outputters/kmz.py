"""
kmz  outputter
"""

import os
from datetime import timedelta, datetime
import zipfile
import base64

from gnome.basic_types import oil_status

from .outputter import Outputter, OutputterFilenameMixin, BaseOutputterSchema
from gnome.persist import drop, FilenameSchema

from . import kmz_templates


class KMZSchema(BaseOutputterSchema):
    '''
    Nothing is required for initialization
    '''
#    round_data = SchemaNode(Bool(), missing=drop)
#    round_to = SchemaNode(Int(), missing=drop)
    filename = FilenameSchema(
        missing=drop, save=True, update=True, test_equal=False
    )


class KMZOutput(OutputterFilenameMixin, Outputter):
    '''
    class that outputs GNOME results in a kmz format.

    Suitable for Google Earth, and semi-suitable for MarPlot

    '''
    _schema = KMZSchema

    time_formatter = '%m/%d/%Y %H:%M'

    def __init__(self, filename, **kwargs):
        '''
        :param str output_dir=None: output directory for kmz files.

        uses super to pass optional ``**kwargs`` to base class ``__init__`` method
        '''
        # a little check:
        self._check_filename(filename)

        # strip off the .kml or .kmz
        filename = filename[:-4] if filename.endswith(".kml") else filename
        filename = filename[:-4] if filename.endswith(".kmz") else filename
        filename += ".kmz"

        self.kml_name = os.path.split(filename)[-1] + ".kml"

        super(KMZOutput, self).__init__(filename=filename,
                                        **kwargs)

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
        if not self.on:
            return

        super(KMZOutput, self).prepare_for_model_run(model_start_time,
                                                     spills,
                                                     **kwargs)

        # shouldn't be required if the above worked!
        self._file_exists_error(self.filename)

        # create a list to hold what will be the contents of the kml
        self.kml = [kmz_templates.header_template
                    .format(caveat=kmz_templates.caveat,
                            kml_name=self.kml_name,
                            valid_timestring=model_start_time.strftime(self.time_formatter),
                            issued_timestring=datetime.now().strftime(self.time_formatter),
                            )
                    ]

        # netcdf outputter has this --  not sure why
        # self._middle_of_run = True

    def write_output(self, step_num, islast_step=False):
        """dump a timestep's data into the kmz file"""

        super(KMZOutput, self).write_output(step_num, islast_step)

        #if not self.on or not self._write_step:
        if not self.on:
            return None

        # add to the kml list:
        if self._write_step:
            for sc in self.cache.load_timestep(step_num).items():
                # loop through uncertain and certain LEs
                # extract the data
                start_time = sc.current_time_stamp

                if self.output_timestep is None:
                    end_time = start_time + timedelta(seconds=self.model_timestep)
                else:
                    end_time = start_time + self.output_timestep

                start_time = start_time.isoformat()
                end_time = end_time.isoformat()

                positions = sc['positions']
                water_positions = positions[sc['status_codes'] == oil_status.in_water]
                beached_positions = positions[sc['status_codes'] == oil_status.on_land]

                self.kml.append(kmz_templates.build_one_timestep(water_positions,
                                                                 beached_positions,
                                                                 start_time,
                                                                 end_time,
                                                                 sc.uncertain
                                                                 ))

        if islast_step:  # now we really write the file:
            self.kml.append(kmz_templates.footer)

            with zipfile.ZipFile(self.filename, 'w',
                                 compression=zipfile.ZIP_DEFLATED) as kmzfile:
                kmzfile.writestr('dot.png', base64.b64decode(DOT))
                kmzfile.writestr('x.png', base64.b64decode(X))
                kmzfile.writestr(self.kml_name,
                                 "".join(self.kml).encode('utf8'))


        if not self._write_step:
            return None

        output_info = {'time_stamp': sc.current_time_stamp.isoformat(),
                       'output_filename': self.filename}

        return output_info

    def rewind(self):
        '''
        reset a few parameter and call base class rewind to reset
        internal variables.
        '''
        super(KMZOutput, self).rewind()

        self._start_idx = 0

    # def clean_output_files(self):
    #     # fixme: this could use the FileNameMixin
    #     '''
    #     deletes ouput files that may be around

    #     called by prepare_for_model_run

    #     here in case it needs to be called from elsewhere
    #     '''
    #     try:
    #         os.remove(self.filename)
    #     except OSError:
    #         pass  # it must not be there


# These icons were encoded by the "build_icons" script
# (they are base64 encoded 3-pixel sized dots in a 32x32 transparent PNG)
# Fixme: Static values built by a tool?  Maybe we should make the generation
#        of these icons a dynamic process.
DOT = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAJOgAACToB8GSSSgAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAEASURBVFiF7ZY7DsIwEEQfET09Ej11lFtwK06Re3ANlCoFPQpnoGJoHClCXpOPg10wUhonnnlyvF5vJJFSRdL0P0AOANsZcwqgAkrg6MZuQANcgdckN0ljn52kWlInW537ZjfWd2z4SVIbCP5U6+ZEAThLek4I7/V0cxcBnGaGDyGCK/Htn09ZdkutAnsiBFBHCO9VWzkb+XtBAdyB/Ywy9ekBHPCUqHUQVRHDcV6V74UFUEYMD3paAEdjfIm8nsl7gQVwWyHL62kBNCsAeD2zLcMXcIkUjvPyt+nASZj8KE7ejLJox1lcSIZ7IvqVzCrDkKJeSucARFW2veAP8DO9AXV74Qmb/4vgAAAAAElFTkSuQmCC"
X = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAN1wAADdcBQiibeAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAHKSURBVFiFrdXLq01hGMfx12HMzMCUU4zFQEYiROYkEkkpHbeTXI5LSDqHtomBEJGY+RMMGBlKKWVmaiDXzvExsN7admuv9azLU89k7ef5fb/ruhMSVuIy3uEOVhXH++w1mMEbnMFSpITl+Ob/+oOpHuFHiszh+oIVCbPGVx8Sh0vguaYT3lcIdJU4VAGHtwm3agTaShysgcMgYUNAoKnEgQAcVueFqR4l9mMhkHVJ8RbkPt6DxL4g/EreGQ3oIrE3CL86vFd2FidaSOzBfGDn+ihv3KU82UBidxB+o4xV9TBFJSKX/eY4Tt0TfSooUVWzVYzIO326A3yuLj/6YWkjcTuSHRVImG4AH0RzJ1K8PqSUFoKzn8KpQdNd+N3wFoT+OyLwnfjVEB6WqIPv6AAPSVTBt+NnR3itxDj4tiD8Hs52kSiDb8WPQOB9LCp2WkuMwrcE4Q8xMbJ7ro3EcMBmfA8EPCqBt5bIi5uC8McV8Nznm0gkLMPXwMKTADz3haDExoRjgcGnWByEN5EYJLyuGXrWAp57pib7Y8K1ioHnHeC5L1bkP0iYHPPjCyzpCK+SmMdkHliLl8XBVzjaIzz3Ov++H59xF+uR/gJmOo2+fdNArAAAAABJRU5ErkJggg=="
