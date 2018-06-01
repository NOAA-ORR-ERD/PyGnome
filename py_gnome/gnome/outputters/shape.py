"""
shapefile  outputter
"""
import copy
import os
import zipfile

from colander import SchemaNode, String, Boolean, drop
import shapefile as shp
from gnome.persist.extend_colander import FilenameSchema


from .outputter import Outputter, BaseOutputterSchema


class ShapeSchema(BaseOutputterSchema):
    filename = FilenameSchema(
        missing=drop, save=True, update=True, test_equal=False
    )
    zip_output = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )


class ShapeOutput(Outputter):
    '''
    class that outputs GNOME results (particles) in a shapefile format.

    '''
    _schema = ShapeSchema

    time_formatter = '%m/%d/%Y %H:%M'

    def __init__(self, filename, zip_output=True, surface_conc="kde", **kwargs):
        '''
        :param str output_dir=None: output directory for shape files

        :param zip_output=True: whether to zip up the ouput shape files
        '''
        # # a little check:
        self._check_filename(filename)

        filename = filename.split(".zip")[0].split(".shp")[0]

        self.filename = filename
        self.filedir = os.path.dirname(filename)

        self.zip_output = zip_output

        super(ShapeOutput, self).__init__(surface_conc=surface_conc,**kwargs)

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

        Write the headers, png files, etc for the shape file file

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
        super(ShapeOutput, self).prepare_for_model_run(model_start_time,
                                                       spills,
                                                       **kwargs)

        if not self.on:
            return

        self.delete_output_files()

        # shouldn't be required if the above worked!
        self._file_exists_error(self.filename + '.zip')

        # info for prj file
        self.epsg = ('GEOGCS["WGS 84",'
                     'DATUM["WGS_1984",'
                     'SPHEROID["WGS 84",6378137,298.257223563]]'
                     ',PRIMEM["Greenwich",0],'
                     'UNIT["degree",0.0174532925199433]]')

        for sc in self.sc_pair.items():
            w = shp.Writer(shp.POINT)
            w.autobalance = 1

            w.field('Time', 'C')
            w.field('LE id', 'N')
            w.field('Depth', 'N')
            w.field('Mass', 'N')
            w.field('Age', 'N')
            w.field('Surf_Conc','F',10,5)
            w.field('Status_Code', 'N')

            if sc.uncertain:
                self.w_u = w
            else:
                self.w = w

    def write_output(self, step_num, islast_step=False):
        """dump a timestep's data into the kmz file"""

        super(ShapeOutput, self).write_output(step_num, islast_step)

        if not self.on or not self._write_step:
            return None

        uncertain = False

        for sc in self.cache.load_timestep(step_num).items():
            curr_time = sc.current_time_stamp

            if sc.uncertain:
                uncertain = True

                for k, p in enumerate(sc['positions']):
                    self.w_u.point(p[0], p[1])
                    self.w_u.record(curr_time.strftime('%Y-%m-%dT%H:%M:%S'),
                                    sc['id'][k],
                                    p[2],
                                    sc['mass'][k],
                                    sc['age'][k],
                                    0.0,
                                    sc['status_codes'][k])
            else:
                for k, p in enumerate(sc['positions']):
                    self.w.point(p[0], p[1])
                    self.w.record(curr_time.strftime('%Y-%m-%dT%H:%M:%S'),
                                  sc['id'][k],
                                  p[2],
                                  sc['mass'][k],
                                  sc['age'][k],
                                  sc['surface_concentration'][k],
                                  sc['status_codes'][k])

        if islast_step:  # now we really write the files:
            if uncertain:
                shapefilenames = [self.filename, self.filename + '_uncert']
            else:
                shapefilenames = [self.filename]

            for fn in shapefilenames:
                if uncertain:
                    self.w_u.save(fn)
                else:
                    self.w.save(fn)

                prj_file = open("%s.prj" % fn, "w")
                prj_file.write(self.epsg)
                prj_file.close()

                if self.zip_output is True:
                    zfilename = fn + '.zip'
                    zipf = zipfile.ZipFile(zfilename, 'w')

                    for suf in ['shp', 'prj', 'dbf', 'shx']:
                        f = os.path.split(fn)[-1] + '.' + suf
                        zipf.write(os.path.join(self.filedir, f), arcname=f)
                        os.remove(fn + '.' + suf)

                    zipf.close()

        if self.zip_output is True:
            output_filename = self.filename + '.zip'
        else:
            output_filename = self.filename

        output_info = {'time_stamp': sc.current_time_stamp.isoformat(),
                       'output_filename': output_filename}

        return output_info

    def rewind(self):
        '''
        reset a few parameter and call base class rewind to reset
        internal variables.
        '''
        super(ShapeOutput, self).rewind()

        self._middle_of_run = False
        self._start_idx = 0

    def delete_output_files(self):
        '''
        deletes ouput files that may be around

        called by prepare_for_model_run

        here in case it needs to be called from elsewhere
        '''
        try:
            os.remove(self.filename + '.zip')
            os.remove(self.filename + '_uncert.zip')
        except OSError:
            pass  # it must not be there
