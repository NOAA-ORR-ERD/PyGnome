"""Shapefile Outputter"""

from colander import SchemaNode, Boolean, drop, Float, Int
import os
import pathlib
import shutil
import tempfile
import zipfile

from gnome.persist.extend_colander import FilenameSchema
from gnome.utilities.shapefile_builder import ParticleShapefileBuilder, BoundaryShapefileBuilder
from .outputter import Outputter, BaseOutputterSchema


class ShapeSchema(BaseOutputterSchema):
    filename = FilenameSchema(
        missing=drop, save=True, update=True, test_equal=False
    )
    zip_output = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    include_certain_boundary = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    certain_boundary_separate_by_spill = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    certain_boundary_hull_ratio = SchemaNode(Float(), save=True, update=True)
    certain_boundary_hull_allow_holes = SchemaNode(Boolean(), save=True, update=True)
    include_uncertain_boundary = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    include_certain_in_uncertain_boundary = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    uncertain_boundary_separate_by_spill = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    uncertain_boundary_hull_ratio = SchemaNode(Float(), save=True, update=True)
    uncertain_boundary_hull_allow_holes = SchemaNode(Boolean(), save=True, update=True)
    # Currently we do not support timezone, but instead stick with static
    # time offset.
    # timezone = SchemaNode(String(), save=True, update=True)
    timeoffset = SchemaNode(Float(), save=True, update=True)


class ShapeOutput(Outputter):
    """class that outputs GNOME results (particles) in a shapefile format."""
    _schema = ShapeSchema
    time_formatter = '%m/%d/%Y %H:%M'

    def __init__(self, filename, zip_output=True,
                 include_certain_boundary=False,
                 certain_boundary_separate_by_spill=True,
                 certain_boundary_hull_ratio=0.5, certain_boundary_hull_allow_holes=False,
                 include_uncertain_boundary=True,
                 include_certain_in_uncertain_boundary=True,
                 uncertain_boundary_separate_by_spill=True,
                 uncertain_boundary_hull_ratio=0.5, uncertain_boundary_hull_allow_holes=False,
                 timeoffset=None,
                 surface_conc="kde", **kwargs):
        """
        :param filename: Full path and basename of the shape file.
        :param zip_output=True: Whether to zip up the output shape files.
        :param surface_conc="kde": Method to use to compute surface concentration
                                   current options are: 'kde' and None
        """
        super(ShapeOutput, self).__init__(surface_conc=surface_conc, **kwargs)
        pathlib_path = pathlib.Path(filename)
        # If zip is requested... force .zip, else we return .shp.
        # Later we also check if uncertain is on... and if so we force .zip
        if zip_output or include_certain_boundary or include_uncertain_boundary:
            self.filename = pathlib_path.with_suffix('.zip')
        else:
            # Else we return .shp
            self.filename = pathlib_path.with_suffix('.shp')
        self.filenamestem = pathlib_path.stem
        self.filedir = str(pathlib_path.parent)
        # A temp dir used to do our work...
        self.tempdir = tempfile.TemporaryDirectory(prefix='gnome.')
        # We will be building shapefiles, so come up with names in the temp dir
        # These are names without ext... as those get added when deciding to zip or not
        base_shapefile_name = os.path.join(self.tempdir.name, self.filenamestem)
        self.shapefile_name_certain = base_shapefile_name+'_certain'
        self.shapefile_name_certain_boundary = base_shapefile_name+'_certain_boundary'
        self.shapefile_name_uncertain = base_shapefile_name+'_uncertain'
        self.shapefile_name_uncertain_boundary = base_shapefile_name+'_uncertain_boundary'
        self.timeoffset = timeoffset
        # Should we be zipping the output
        self.zip_output = zip_output
        self.include_certain_boundary = include_certain_boundary
        self.certain_boundary_separate_by_spill = certain_boundary_separate_by_spill
        self.certain_boundary_hull_ratio = certain_boundary_hull_ratio
        self.certain_boundary_hull_allow_holes = certain_boundary_hull_allow_holes
        self.include_uncertain_boundary = include_uncertain_boundary
        self.include_certain_in_uncertain_boundary = include_certain_in_uncertain_boundary
        self.uncertain_boundary_separate_by_spill = uncertain_boundary_separate_by_spill
        self.uncertain_boundary_hull_ratio = uncertain_boundary_hull_ratio
        self.uncertain_boundary_hull_allow_holes = uncertain_boundary_hull_allow_holes

        # Our shapefile builders
        self.shapefile_builder_certain = ParticleShapefileBuilder(self.shapefile_name_certain,
                                                                  zip_output=zip_output,
                                                                  timeoffset=self.timeoffset)
        self.shapefile_builder_certain_boundary = BoundaryShapefileBuilder(self.shapefile_name_certain_boundary,
                                                                           zip_output=zip_output,
                                                                           timeoffset=self.timeoffset)
        self.shapefile_builder_uncertain = ParticleShapefileBuilder(self.shapefile_name_uncertain,
                                                                    zip_output=zip_output,
                                                                    timeoffset=self.timeoffset)
        self.shapefile_builder_uncertain_boundary = BoundaryShapefileBuilder(self.shapefile_name_uncertain_boundary,
                                                                             zip_output=zip_output,
                                                                             timeoffset=self.timeoffset)


    def __del__(self):
        self.tempdir.cleanup()

    def prepare_for_model_run(self,
                              model_start_time,
                              spills,
                              uncertain = False,
                              **kwargs):
        """
        If uncertainty is on, then SpillContainerPair object contains
        identical _data_arrays in both certain and uncertain SpillContainer's,
        the data itself is different, but they contain the same type of data
        arrays. If uncertain, then data arrays for uncertain spill container
        are written.
        """
        super(ShapeOutput, self).prepare_for_model_run(model_start_time,
                                                       spills,
                                                       **kwargs)
        if not self.on:
            return
        self.model_start_time = model_start_time
        self.spills = spills
        self.uncertain = uncertain
        if uncertain:
            self.filename = self.filename.with_suffix('.zip')

    def write_output(self, step_num, islast_step=False):
        """Dump a timestep's data into the shapefile"""

        super(ShapeOutput, self).write_output(step_num, islast_step)
        if not self.on:
            return None

        sp = self.cache.load_timestep(step_num).items()
        for sc in sp:
            if self._write_step:
                # If this is just a step... append the data
                if sc.uncertain:
                    self.shapefile_builder_uncertain.append(sc)
                    if self.include_uncertain_boundary:
                        karg = {'separate_by_spill': self.uncertain_boundary_separate_by_spill,
                                'hull_ratio': self.uncertain_boundary_hull_ratio,
                                'hull_allow_holes': self.uncertain_boundary_hull_allow_holes
                                }
                        if self.include_certain_in_uncertain_boundary:
                            # If we want to include certain in the uncertain boundary
                            # we need to pass in the spill pair, not just the uncertain
                            # spill container.
                            self.shapefile_builder_uncertain_boundary.append(sp, **karg)
                        else:
                            self.shapefile_builder_uncertain_boundary.append(sc, **karg)
                else:
                    self.shapefile_builder_certain.append(sc)
                    if self.include_certain_boundary:
                        karg = {'separate_by_spill': self.certain_boundary_separate_by_spill,
                                'hull_ratio': self.certain_boundary_hull_ratio,
                                'hull_allow_holes': self.certain_boundary_hull_allow_holes
                                }
                        self.shapefile_builder_certain_boundary.append(sc, **karg)

        if not self._write_step:
            return None

        output_info = {'time_stamp': sc.current_time_stamp.isoformat(),
                       'output_filename': str(self.filename)}
        return output_info

    def post_model_run(self):
        """ The final step to wrap everything up """
        # We first try to write the shapefile
        try:
            self.shapefile_builder_certain.write()
            if self.include_certain_boundary:
                self.shapefile_builder_certain_boundary.write()
            if self.uncertain:
                self.shapefile_builder_uncertain.write()
                if self.include_uncertain_boundary:
                    self.shapefile_builder_uncertain_boundary.write()
        except Exception as exception:
            self.logger.debug(f'Could not write shapefile: {exception=}')
            data_string = (f'{self.on=}, '
                           f'{self.model_start_time=}, '
                           f'{self.output_single_step=}, '
                           f'{self.output_zero_step=}, '
                           f'{self.output_last_step=}, '
                           f'{self.output_timestep=}, '
                           f'{self.output_start_time=}')
            self.logger.debug(f'Model state: {data_string}')
            raise

        # Finally wrap it all up... bundling if both certain and uncertain
        if self.uncertain is True:
            # If we have uncertain, we bundle and rename as filename
            shutil.copy(self.create_bundle(uncertain=True), self.filename)
        else:
            if self.zip_output:
                if self.include_certain_boundary:
                    shutil.copy(self.create_bundle(uncertain=False), self.filename)
                else:
                    shutil.copy(self.shapefile_builder_certain.filename, self.filename)
            else:
                # If its just a single shapefile (no zip), glob all the files
                # and rename them... removing the _certain
                fn = self.shapefile_builder_certain.filename
                for f in fn.parent.glob(fn.with_suffix('.*').name):
                    pathlib_path = pathlib.Path(f)
                    stem = pathlib_path.stem.removesuffix('_certain')
                    suffix = pathlib_path.suffix
                    shutil.copy(f, pathlib.Path(self.filedir, stem + suffix))
                fn = self.shapefile_builder_certain_boundary.filename
                for f in fn.parent.glob(fn.with_suffix('.*').name):
                    pathlib_path = pathlib.Path(f)
                    stem = pathlib_path.stem.removesuffix('_certain_boundary') + '_boundary'
                    suffix = pathlib_path.suffix
                    shutil.copy(f, pathlib.Path(self.filedir, stem + suffix))


    def create_bundle(self, uncertain):
        """Create a shapefile bundle including both certain and uncertain shapefiles"""
        zipf_filename = os.path.join(self.tempdir.name, self.filenamestem)+'_bundle.zip'
        zipf = zipfile.ZipFile(zipf_filename, 'w')
        # Add the main forcast file
        dir, file_to_zip = os.path.split(self.shapefile_builder_certain.filename)
        zipf.write(self.shapefile_builder_certain.filename,
                   arcname=file_to_zip)
        if self.include_certain_boundary:
            dir, file_to_zip = os.path.split(self.shapefile_builder_certain_boundary.filename)
            zipf.write(self.shapefile_builder_certain_boundary.filename,
                       arcname=file_to_zip)
        if uncertain:
            dir, file_to_zip = os.path.split(self.shapefile_builder_uncertain.filename)
            zipf.write(self.shapefile_builder_uncertain.filename,
                       arcname=file_to_zip)
            if self.include_uncertain_boundary:
                dir, file_to_zip = os.path.split(self.shapefile_builder_uncertain_boundary.filename)
                zipf.write(self.shapefile_builder_uncertain_boundary.filename,
                           arcname=file_to_zip)
        zipf.close()
        return zipf_filename

    def rewind(self):
        '''
        reset a few parameter and call base class rewind to reset
        internal variables.
        '''
        super(ShapeOutput, self).rewind()
        self._middle_of_run = False
        self._start_idx = 0

    def clean_output_files(self):
        '''
        deletes ouput files that may be around
        called by prepare_for_model_run
        here in case it needs to be called from elsewhere
        '''
        super(ShapeOutput, self).clean_output_files()
        # pathlib.Path(self.filename).unlink(missing_ok=True)
