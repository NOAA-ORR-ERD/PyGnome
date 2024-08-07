"""ERMA Data Package outputter"""
from colander import SchemaNode, Boolean, drop, String, Int, Float
import copy
import geopandas as gpd
import itertools
import json
import nucos as uc
import numpy as np
import os
import pandas as pd
import pathlib
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
import shutil
import tempfile
import zipfile

from gnome.persist.extend_colander import FilenameSchema
from gnome.utilities.geometry.polygons import PolygonSet
from gnome.utilities.shapefile_builder import ParticleShapefileBuilder
from gnome.utilities.shapefile_builder import BoundaryShapefileBuilder
from gnome.utilities.shapefile_builder import ContourShapefileBuilder
from gnome.utilities.hull import calculate_hull
from gnome.utilities import convert_mass_to_mass_or_volume
from .outputter import Outputter, BaseOutputterSchema


erma_data_package_data_dir = pathlib.Path(__file__).parent / "erma_data_package_data"

class ERMADataPackageSchema(BaseOutputterSchema):
    filename = FilenameSchema(
        missing=drop, save=True, update=True, test_equal=False
    )
    # ERMA folder
    base_folder_name = SchemaNode(String(), save=True, update=True)
    model_folder_name = SchemaNode(String(), save=True, update=True)
    # Default styling
    default_erma_styling = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    # Certain layer
    certain_layer_name = SchemaNode(String(), save=True, update=True)
    include_certain_particles = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    # Uncertain layer
    uncertain_layer_name = SchemaNode(String(), save=True, update=True)
    include_uncertain_particles = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    # Certain boundary
    certain_boundary_layer_name = SchemaNode(String(), save=True, update=True)
    include_certain_boundary = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    certain_boundary_separate_by_spill = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    certain_boundary_hull_ratio = SchemaNode(Float(), save=True, update=True)
    certain_boundary_hull_allow_holes = SchemaNode(Boolean(), save=True, update=True)
    certain_boundary_color = SchemaNode(String(), save=True, update=True)
    certain_boundary_size = SchemaNode(Int(), save=True, update=True)
    # Certain contours
    certain_contours_layer_name = SchemaNode(String(), save=True, update=True)
    include_certain_contours = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    certain_contours_hull_ratio = SchemaNode(Float(), save=True, update=True)
    certain_contours_hull_allow_holes = SchemaNode(Boolean(), save=True, update=True)
    certain_contours_size = SchemaNode(Int(), save=True, update=True)
    # Uncertain boundary
    uncertain_boundary_layer_name = SchemaNode(String(), save=True, update=True)
    include_uncertain_boundary = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    uncertain_boundary_separate_by_spill = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    uncertain_boundary_hull_ratio = SchemaNode(Float(), save=True, update=True)
    uncertain_boundary_hull_allow_holes = SchemaNode(Boolean(), save=True, update=True)
    uncertain_boundary_color = SchemaNode(String(), save=True, update=True)
    uncertain_boundary_size = SchemaNode(Int(), save=True, update=True)
    # Map Bounds
    map_bounds_layer_name = SchemaNode(String(), save=True, update=True)
    include_map_bounds = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    map_bounds_color = SchemaNode(String(), save=True, update=True)
    map_bounds_size = SchemaNode(Int(), save=True, update=True)
    # Spillable Area
    spillable_area_layer_name = SchemaNode(String(), save=True, update=True)
    include_spillable_area = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    spillable_area_color = SchemaNode(String(), save=True, update=True)
    spillable_area_size = SchemaNode(Int(), save=True, update=True)
    # Land Polys
    land_polys_layer_name = SchemaNode(String(), save=True, update=True)
    include_land_polys = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    land_polys_color = SchemaNode(String(), save=True, update=True)
    land_polys_size = SchemaNode(Int(), save=True, update=True)
    # Spill location
    include_spill_location = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    # Legend collapse
    disable_legend_collapse = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    # Override the ERMA timestep info
    time_step_override = SchemaNode(
        Int(), missing=drop, save=True, update=True
    )
    time_unit_override = SchemaNode(String(), save=True, update=True)
    timezone = SchemaNode(String(), save=True, update=True)

class ERMADataPackageOutput(Outputter):
    '''
    Outputs the GNOME results as shapefile, and then wraps in an
    ERMA data package.
    '''
    _schema = ERMADataPackageSchema

    time_formatter = '%m/%d/%Y %H:%M'

    def __init__(self, filename, base_folder_name='', model_folder_name='',
                 default_erma_styling=True,
                 # Certain layer
                 certain_layer_name=None, include_certain_particles=True,
                 # Uncertain layer
                 uncertain_layer_name=None, include_uncertain_particles=True,
                 # Certain boundary
                 certain_boundary_layer_name=None, include_certain_boundary=False,
                 certain_boundary_separate_by_spill=True,
                 certain_boundary_hull_ratio=0.5, certain_boundary_hull_allow_holes=False,
                 certain_boundary_color=None, certain_boundary_size=None,
                 # Certain contours
                 certain_contours_layer_name=None, include_certain_contours=False,
                 certain_contours_hull_ratio=0.5, certain_contours_hull_allow_holes=False,
                 certain_contours_size=None,
                 # Uncertain boundary
                 uncertain_boundary_layer_name=None, include_uncertain_boundary=True,
                 uncertain_boundary_separate_by_spill=True,
                 uncertain_boundary_hull_ratio=0.5, uncertain_boundary_hull_allow_holes=False,
                 uncertain_boundary_color=None, uncertain_boundary_size=None,
                 # Map bounds
                 map_bounds_layer_name=None, include_map_bounds=False,
                 map_bounds_color=None, map_bounds_size=None,
                 # Spillable area
                 spillable_area_layer_name=None, include_spillable_area=False,
                 spillable_area_color=None, spillable_area_size=None,
                 # Land polys
                 land_polys_layer_name=None, include_land_polys=False,
                 land_polys_color=None, land_polys_size=None,
                 #Spill location
                 include_spill_location=True,
                 # Legend collapse
                 disable_legend_collapse=False,
                 # Time settings
                 time_step_override=None, time_unit_override=None,
                 timezone='',
                 # Other
                 surface_conc="kde", **kwargs):
        '''
        :param filename: Full path and filename (with ext) of the desired outout file
        :param surface_conc="kde": method to use to compute surface concentration
                                   current options are: 'kde' and None
        '''
        super(ERMADataPackageOutput, self).__init__(surface_conc=surface_conc, **kwargs)
        pathlib_path = pathlib.Path(filename)
        # ERMA data packages are always zip files... so force that
        self.test_var = True
        self.filename = pathlib_path.with_suffix('.zip')
        self.filenamestem = pathlib_path.stem
        self.filedir = str(pathlib_path.parent)
        # A temp dir used to do our work...
        self.tempdir = tempfile.TemporaryDirectory(prefix='gnome.')
        # We dont know the model_start_time right off the bat, but we will fill in
        # when the model run starts.
        self.model_start_time = None

        # Default some vars if they are null
        if not base_folder_name:
            self.base_folder_name =  "Testing Layers > Trajectories > " + self.filenamestem + '-' + self.id
        else:
            self.base_folder_name = base_folder_name
        self.model_folder_name = model_folder_name
        self.folder_name = self.base_folder_name + ' > ' + self.model_folder_name
        # Default styling
        self.default_erma_styling = default_erma_styling
        # Certain layer
        self.certain_layer_name = certain_layer_name
        self.include_certain_particles = include_certain_particles
        # Uncertain layer
        self.uncertain_layer_name = uncertain_layer_name
        self.include_uncertain_particles = include_uncertain_particles

        # Certain boundary
        self.certain_boundary_layer_name = certain_boundary_layer_name
        self.include_certain_boundary = include_certain_boundary
        self.certain_boundary_separate_by_spill = certain_boundary_separate_by_spill
        self.certain_boundary_hull_ratio = certain_boundary_hull_ratio
        self.certain_boundary_hull_allow_holes = certain_boundary_hull_allow_holes
        self.certain_boundary_color = certain_boundary_color
        self.certain_boundary_size = certain_boundary_size

        # Certain contours
        self.certain_contours_layer_name = certain_contours_layer_name
        self.include_certain_contours = include_certain_contours
        self.certain_contours_hull_ratio = certain_contours_hull_ratio
        self.certain_contours_hull_allow_holes = certain_contours_hull_allow_holes
        self.certain_contours_size = certain_contours_size

        # Uncertain boundary
        self.uncertain_boundary_layer_name = uncertain_boundary_layer_name
        self.include_uncertain_boundary = include_uncertain_boundary
        self.uncertain_boundary_separate_by_spill = uncertain_boundary_separate_by_spill
        self.uncertain_boundary_hull_ratio = uncertain_boundary_hull_ratio
        self.uncertain_boundary_hull_allow_holes = uncertain_boundary_hull_allow_holes
        self.uncertain_boundary_color = uncertain_boundary_color
        self.uncertain_boundary_size = uncertain_boundary_size
        # Map bounds
        self.map_bounds_layer_name = map_bounds_layer_name
        self.include_map_bounds = include_map_bounds
        self.map_bounds_color = map_bounds_color
        self.map_bounds_size = map_bounds_size
        # Spillable area
        self.spillable_area_layer_name = spillable_area_layer_name
        self.include_spillable_area = include_spillable_area
        self.spillable_area_color = spillable_area_color
        self.spillable_area_size = spillable_area_size
        # Land polys
        self.land_polys_layer_name = land_polys_layer_name
        self.include_land_polys = include_land_polys
        self.land_polys_color = land_polys_color
        self.land_polys_size = land_polys_size
        # Spill location
        self.include_spill_location = include_spill_location
        # Legend collapse
        self.disable_legend_collapse=disable_legend_collapse
        # Time settings
        self.time_step_override = time_step_override
        self.time_unit_override = time_unit_override
        self.timezone = timezone
        # We will be building shapefiles, so come up with names in the temp dir
        # These are names without ext... as those get added when deciding to zip or not
        base_shapefile_name = os.path.join(self.tempdir.name, self.filenamestem)
        self.shapefile_name_certain = base_shapefile_name+'_certain'
        self.shapefile_name_certain_boundary = base_shapefile_name+'_certain_boundary'
        self.shapefile_name_certain_contours = base_shapefile_name+'_certain_contours'
        self.shapefile_name_uncertain = base_shapefile_name+'_uncertain'
        self.shapefile_name_uncertain_boundary = base_shapefile_name+'_uncertain_boundary'
        # Our shapefile builders
        self.shapefile_builder_certain = ParticleShapefileBuilder(self.shapefile_name_certain)
        self.shapefile_builder_certain_boundary = BoundaryShapefileBuilder(self.shapefile_name_certain_boundary)
        self.shapefile_builder_certain_contours = ContourShapefileBuilder(self.shapefile_name_certain_contours)
        self.shapefile_builder_uncertain = ParticleShapefileBuilder(self.shapefile_name_uncertain)
        self.shapefile_builder_uncertain_boundary = BoundaryShapefileBuilder(self.shapefile_name_uncertain_boundary)

        # Build some mappings for styling
        self.default_unit_map = {'Mass':{'column': 'mass',
                                         'unit':'kilograms'},
                                 'Surface Concentration':{'column': 'surf_conc',
                                                          'unit':'kg/m^2'},
                                 'Age': {'column': 'age',
                                         'unit': 'seconds'},
                                 'Viscosity': {'column': 'viscosity',
                                               'unit': 'm^2/s'}
                                 }

    def __del__(self):
        self.tempdir.cleanup()

    def prepare_for_model_run(self,
                              model_start_time,
                              spills,
                              uncertain = False,
                              **kwargs):
        """ Setup before we run the model. """
        if not self.on:
            return
        super(ERMADataPackageOutput, self).prepare_for_model_run(model_start_time,
                                                                 spills,
                                                                 **kwargs)
        self.model_start_time = model_start_time
        # By default we want to name the folder for the output based on the model
        # start time.  Since we are just finding that out now (if it was not
        # explicitily set by the user), we need to set the folder_name here as well
        if not self.model_folder_name:
            self.model_folder_name =  self.model_start_time.strftime(self.time_formatter)
            self.folder_name = self.base_folder_name + ' > ' + self.model_folder_name
        self.spills = spills
        # This generates a cutoff struct for contours based
        # on our spills
        self.cutoff_struct = self.generate_cutoff_struct()
        self.uncertain = uncertain
        self.hull_info = []

    def write_output(self, step_num, islast_step=False):
        """Dump a timestep's data into the shapefile """
        if not self.on:
            return None
        super(ERMADataPackageOutput, self).write_output(step_num, islast_step)
        self.logger.debug(f'erma_data_package step_num: {step_num}')
        for sc in self.cache.load_timestep(step_num).items():
            if self._write_step:
                # If this is just a step... append the data
                if sc.uncertain:
                    self.shapefile_builder_uncertain.append(sc)
                    if self.include_uncertain_boundary:
                        karg = {'separate_by_spill': self.uncertain_boundary_separate_by_spill,
                                'hull_ratio': self.uncertain_boundary_hull_ratio,
                                'hull_allow_holes': self.uncertain_boundary_hull_allow_holes
                                }
                        self.shapefile_builder_uncertain_boundary.append(sc, **karg)
                else:
                    self.shapefile_builder_certain.append(sc)
                    if self.include_certain_boundary:
                        karg = {'separate_by_spill': self.certain_boundary_separate_by_spill,
                                'hull_ratio': self.certain_boundary_hull_ratio,
                                'hull_allow_holes': self.certain_boundary_hull_allow_holes
                                }
                        self.shapefile_builder_certain_boundary.append(sc, **karg)
                    if self.include_certain_contours:
                        karg = {'cutoff_struct': self.cutoff_struct,
                                'hull_ratio': self.certain_contours_hull_ratio,
                                'hull_allow_holes': self.certain_contours_hull_allow_holes
                                }
                        self.shapefile_builder_certain_contours.append(sc, **karg)
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
            if self.include_certain_contours:
                self.shapefile_builder_certain_contours.write()
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

        # Finally wrap it all up... Build the ERMA data package
        self.build_package()

    def build_package(self):
        id = itertools.count()
        layer_json = []
        if self.include_spill_location:
            layer_json.append(self.make_spill_location_package_layer(next(id)))
        if self.include_certain_particles:
            layer_name = self.certain_layer_name if self.certain_layer_name else 'Certain Particles'
            layer_json.append(self.make_particle_package_layer(next(id), layer_name, False,
                                                               self.shapefile_builder_certain.filename))
        if self.include_uncertain_particles and self.uncertain:
            layer_name = self.uncertain_layer_name if self.uncertain_layer_name else 'Uncertain Particles'
            layer_json.append(self.make_particle_package_layer(next(id), layer_name, True,
                                                               self.shapefile_builder_uncertain.filename))
        mp = {}
        if self.map:
            mp=self.map.get_polygons()
        if self.include_spillable_area and 'spillable_area' in mp and mp['spillable_area'] is not None:
            layer_name = self.spillable_area_layer_name if self.spillable_area_layer_name else 'Spillable Area'
            layer_color = self.spillable_area_color if self.spillable_area_color else '#0000FF'
            layer_size = self.spillable_area_size if self.spillable_area_size else 2
            layer_json.append(self.make_map_polygon_package_layer(next(id), 'spillable_area', 'spillable_area',
                                                                  layer_name, 'Spillable Area',
                                                                  layer_color, layer_size))
        if self.include_land_polys and 'land_polys' in mp and mp['land_polys'] is not None:
            layer_name = self.land_polys_layer_name if self.land_polys_layer_name else 'Land Polygons'
            layer_color = self.land_polys_color if self.land_polys_color else '#000000'
            layer_size = self.land_polys_size if self.land_polys_size else 2
            layer_json.append(self.make_map_polygon_package_layer(next(id), 'land_polys', 'land_polys',
                                                                  layer_name, 'Land Polygons',
                                                                  layer_color, layer_size))
        if self.include_map_bounds and 'map_bounds' in mp and mp['map_bounds'] is not None:
            layer_name = self.map_bounds_layer_name if self.map_bounds_layer_name else 'Map Bounds'
            layer_color = self.map_bounds_color if self.map_bounds_color else '#FF0000'
            layer_size = self.map_bounds_size if self.map_bounds_size else 2
            layer_json.append(self.make_map_polygon_package_layer(next(id), 'map_bounds', 'map_bounds',
                                                                  layer_name, 'Map Bounds',
                                                                  layer_color, layer_size))
        if self.include_certain_boundary:
            layer_name = self.certain_boundary_layer_name if self.certain_boundary_layer_name else 'Certain Particles Boundary'
            layer_color = self.certain_boundary_color if self.certain_boundary_color else '#0000FF'
            layer_size = self.certain_boundary_size if self.certain_boundary_size else 3
            layer_json.append(self.make_boundary_polygon_package_layer(next(id), False,
                                                                       self.shapefile_builder_certain_boundary.filename,
                                                                       layer_name, 'Best Estimate Boundary',
                                                                       layer_color, layer_size))
        if self.include_certain_contours:
            layer_name = self.certain_contours_layer_name if self.certain_contours_layer_name else 'Certain Particles Contours'
            layer_size = self.certain_contours_size if self.certain_contours_size else 3
            layer_json.append(self.make_contour_polygon_package_layer(next(id),
                                                                      self.shapefile_builder_certain_contours.filename,
                                                                      layer_name, 'Best Estimate Contours',
                                                                      layer_size))
        if self.include_uncertain_boundary and self.uncertain:
            layer_name = self.uncertain_boundary_layer_name if self.uncertain_boundary_layer_name else 'Uncertain Particles Boundary'
            layer_color = self.uncertain_boundary_color if self.uncertain_boundary_color else '#FF0000'
            layer_size = self.uncertain_boundary_size if self.uncertain_boundary_size else 3
            layer_json.append(self.make_boundary_polygon_package_layer(next(id), True,
                                                                       self.shapefile_builder_uncertain_boundary.filename,
                                                                       layer_name, 'Uncertainty Boundary',
                                                                       layer_color, layer_size))
        # Now we can zip it all up
        zipf = zipfile.ZipFile(self.filename, 'w')
        for layer in layer_json:
            dir, file_to_zip = os.path.split(layer['shapefile_filename'])
            zipf.write(layer['shapefile_filename'],
                       arcname='source_files/'+file_to_zip)
            dir, file_to_zip = os.path.split(layer['json_filename'])
            zipf.write(layer['json_filename'],
                       arcname='layers/'+file_to_zip)
        # Write a readme with the basics of the output
        zipf.writestr('README', f'{self.folder_name}')
        # Write out some empty dirs for now for attachments etc.
        zipf.writestr(zipfile.ZipInfo('attachment/'), '')
        # Needed font file
        font_path = erma_data_package_data_dir / 'SHAPES.TTF'
        zipf.write(font_path,
                   arcname='support_files/fonts/SHAPES.TTF')
        zipf.close()

    def make_contour_polygon_package_layer(self, id, shapefile_filename,
                                           layer_title, style_name,
                                           style_width):
        dir, basefile = os.path.split(shapefile_filename)
        output_path = os.path.join(self.tempdir.name, str(id)+".json")
        generic_name = 'contour_certain'
        generic_description = 'Contour Certain'
        layer_template = None

        layer_template_path = erma_data_package_data_dir / 'layer_template.json'
        contour_template = None
        contour_template_path = erma_data_package_data_dir / 'default_contour_template.json'

        with open(layer_template_path) as f:
            layer_template = json.load(f)
        with open(contour_template_path) as f:
            contour_template = json.load(f)
        if layer_template and contour_template:
            # Check the timestep and set the time override for ERMA time slider
            if self.time_step_override and self.time_unit_override:
                layer_template['time_step_override'] = self.time_step_override
                layer_template['time_unit_override'] = self.time_unit_override
            else:
                timestepseconds = None
                if self.output_timestep:
                    timestepseconds = int(self.output_timestep.seconds)
                else:
                    timestepseconds = int(self.model_timestep)
                minute = 60
                hour = minute*60
                day = hour*24
                month = day*30
                year = day*365
                if timestepseconds < hour:
                    # Less than an hour... so use minutes
                    layer_template['time_step_override'] = int(timestepseconds / minute)
                    layer_template['time_unit_override'] = 'minute'
                elif timestepseconds < day:
                    # Less than a day, use hours
                    layer_template['time_step_override'] = int(timestepseconds / hour)
                    layer_template['time_unit_override'] = 'hour'
                elif timestepseconds < month:
                    # Less than a month, use day
                    layer_template['time_step_override'] = int(timestepseconds / day)
                    layer_template['time_unit_override'] = 'day'
                elif timestepseconds < year:
                    # Less than a year, use month
                    layer_template['time_step_override'] = int(timestepseconds / month)
                    layer_template['time_unit_override'] = 'month'
                else:
                    # More than a year, use year
                    layer_template['time_step_override'] = int(timestepseconds / year)
                    layer_template['time_units_override'] = 'year'
            # Now build the layer file
            # Folder name
            layer_template['folder_path'] = self.folder_name
            layer_template['title'] = layer_title
            layer_template['mapfile_layer']['layer_type'] = 'line'
            layer_template['mapfile_layer']['shapefile']['name'] = generic_name + '_shapefile'
            layer_template['mapfile_layer']['shapefile']['description'] = generic_description + ' Shapefile'
            layer_template['mapfile_layer']['shapefile']['file'] = "file://source_files/" + basefile
            # If we have a timezone, write that into the timezone_fields
            if self.timezone:
                layer_template['mapfile_layer']['shapefile']['timezone_fields'] = {"time": self.timezone}
            layer_template['mapfile_layer']['layer_name'] = generic_name
            layer_template['mapfile_layer']['layer_desc'] = generic_description
            layer_template['mapfile_layer']['classitem'] = 'cutoff_id'
            layer_template['mapfile_layer']['labelitem'] = 'label'
            # Modify the style object
            # Loop through self.cutoff_struct and build classes...
            #{0: {'param': 'surf_conc', 'cutoffs': [{'cutoff': 0.0005280305158615599, 'label': 'Low'}, {'cutoff': 0.001544090616505013, 'label': 'Medium'}, {'cutoff': 0.0038150665966212825, 'label': 'High'}]}}
            classcounter = itertools.count()
            for spill_num, spill in enumerate(self.spills):
                if spill_num in self.cutoff_struct:
                    cutoff_element = self.cutoff_struct[spill_num]
                    param = cutoff_element['param']
                    for cutoff in cutoff_element['cutoffs']:
                        contour_template_solid = copy.deepcopy(contour_template)
                        contour_template_solid['name'] = style_name+f'_{spill_num}_{next(classcounter)}'
                        contour_template_solid['expression'] = cutoff['cutoff_id']
                        contour_template_solid['styles'][0]['outlinesymbol'] = None
                        contour_template_solid['styles'][0]['color'] = cutoff['color']
                        contour_template_solid['styles'][0]['style_width'] = style_width
                        contour_template_solid['labels'][0]['color'] = cutoff['color']
                        layer_template['mapfile_layer']['layer_classes'].append(contour_template_solid)
        else:
            raise ValueError("Can not write ERMA Data Package without template!!!")

        with open(output_path, "w") as o:
            json.dump(layer_template, o, indent=4)
        return {'shapefile_filename': shapefile_filename,
                'json_filename': output_path}

    def make_boundary_polygon_package_layer(self, id, uncertain, shapefile_filename,
                                            layer_title, style_name,
                                            color, style_width):
        dir, basefile = os.path.split(shapefile_filename)
        output_path = os.path.join(self.tempdir.name, str(id)+".json")
        #shz_name = os.path.join(self.tempdir.name, shapefile_name+'.shz')
        #shapefile_pathlib_path = pathlib.Path(shz_name)
        generic_name = f'boundary_{"uncertain" if uncertain else "certain"}'
        generic_description = f'Boundary {"Uncertain" if uncertain else "Certain"}'
        layer_template = None
        layer_template_path = erma_data_package_data_dir / 'layer_template.json'
        polygon_template = None
        polygon_template_path = erma_data_package_data_dir / 'default_polygon_cartoline_template.json'
        with open(layer_template_path) as f:
            layer_template = json.load(f)
        with open(polygon_template_path) as f:
            polygon_template = json.load(f)
        if layer_template and polygon_template:
            # Check the timestep and set the time override for ERMA time slider
            if self.time_step_override and self.time_unit_override:
                layer_template['time_step_override'] = self.time_step_override
                layer_template['time_unit_override'] = self.time_unit_override
            else:
                timestepseconds = None
                if self.output_timestep:
                    timestepseconds = int(self.output_timestep.seconds)
                else:
                    timestepseconds = int(self.model_timestep)
                minute = 60
                hour = minute*60
                day = hour*24
                month = day*30
                year = day*365
                if timestepseconds < hour:
                    # Less than an hour... so use minutes
                    layer_template['time_step_override'] = int(timestepseconds / minute)
                    layer_template['time_unit_override'] = 'minute'
                elif timestepseconds < day:
                    # Less than a day, use hours
                    layer_template['time_step_override'] = int(timestepseconds / hour)
                    layer_template['time_unit_override'] = 'hour'
                elif timestepseconds < month:
                    # Less than a month, use day
                    layer_template['time_step_override'] = int(timestepseconds / day)
                    layer_template['time_unit_override'] = 'day'
                elif timestepseconds < year:
                    # Less than a year, use month
                    layer_template['time_step_override'] = int(timestepseconds / month)
                    layer_template['time_unit_override'] = 'month'
                else:
                    # More than a year, use year
                    layer_template['time_step_override'] = int(timestepseconds / year)
                    layer_template['time_units_override'] = 'year'
            # Now build the layer file
            # Folder name
            layer_template['folder_path'] = self.folder_name
            layer_template['title'] = layer_title
            layer_template['mapfile_layer']['layer_type'] = 'polygon'
            layer_template['mapfile_layer']['shapefile']['name'] = generic_name + '_shapefile'
            layer_template['mapfile_layer']['shapefile']['description'] = generic_description + ' Shapefile'
            layer_template['mapfile_layer']['shapefile']['file'] = "file://source_files/" + basefile
            # If we have a timezone, write that into the timezone_fields
            if self.timezone:
                layer_template['mapfile_layer']['shapefile']['timezone_fields'] = {"time": self.timezone}
            layer_template['mapfile_layer']['layer_name'] = generic_name
            layer_template['mapfile_layer']['layer_desc'] = generic_description
            # Get rid of a few things we dont want
            layer_template['mapfile_layer']['classitem'] = None
            # Modify the style object
            polygon_template_cartoline = copy.deepcopy(polygon_template)
            polygon_template_cartoline['name'] = style_name
            polygon_template_cartoline['expression'] = None
            polygon_template_cartoline['expression_type'] = None
            polygon_template_cartoline['styles'][0]['outlinecolor'] = color
            polygon_template_cartoline['styles'][0]['outlinesymbol'] = 'dashedcartoline'
            polygon_template_cartoline['styles'][0]['style_width'] = style_width
            layer_template['mapfile_layer']['layer_classes'].append(polygon_template_cartoline)
        else:
            raise ValueError("Can not write ERMA Data Package without template!!!")

        with open(output_path, "w") as o:
            json.dump(layer_template, o, indent=4)
        return {'shapefile_filename': shapefile_filename,
                'json_filename': output_path}

    def make_map_polygon_package_layer(self, id, map_polygon_name, shapefile_name,
                                       layer_title, style_name,
                                       color, style_width):
        output_path = os.path.join(self.tempdir.name, str(id)+".json")
        shz_name = os.path.join(self.tempdir.name, shapefile_name+'.shz')
        shapefile_pathlib_path = pathlib.Path(shz_name)
        layer_template = None
        layer_template_path = erma_data_package_data_dir / 'layer_template.json'
        polygon_template = None
        polygon_template_path = erma_data_package_data_dir / 'default_polygon_template.json'

        with open(layer_template_path) as f:
            layer_template = json.load(f)
        with open(polygon_template_path) as f:
            polygon_template = json.load(f)
        if layer_template and polygon_template:
            # Write the shapefile
            mp=self.map.get_polygons()
            if isinstance(mp[map_polygon_name], np.ndarray):
                # Just an array... we make a geo data frame
                gdf = gpd.GeoDataFrame([], crs='epsg:4326', geometry=[Polygon(mp[map_polygon_name])])
            elif isinstance(mp[map_polygon_name], PolygonSet):
                # The PolygonSet's have metadata... grab that
                columns = [f'meta{id}' for id in range(len(mp[map_polygon_name][0].metadata))]
                df=pd.DataFrame((x.metadata for x in mp[map_polygon_name]), columns=columns)
                gdf = gpd.GeoDataFrame(df, crs='epsg:4326',
                                       geometry=[Polygon(sa.points) for sa in mp[map_polygon_name]])
            else:
                raise ValueError(f'{map_polygon_name} is not a supported map polygon!!!')
            # Write out the zipped shapefile
            gdf.to_file(shz_name, driver='ESRI Shapefile',
                             engine="pyogrio")
            shutil.copy(shz_name, shapefile_pathlib_path.with_suffix('.zip'))
            # Now build the layer file
            # Folder name
            layer_template['folder_path'] = self.folder_name
            layer_template['title'] = layer_title
            layer_template['mapfile_layer']['layer_type'] = 'polygon'
            layer_template['mapfile_layer']['shapefile']['name'] = 'map_polygon_shapefile'
            layer_template['mapfile_layer']['shapefile']['description'] = 'Map Polygon Shapefile'
            layer_template['mapfile_layer']['shapefile']['file'] = "file://source_files/" + shapefile_name + '.zip'
            layer_template['mapfile_layer']['layer_name'] = 'map_polygon'
            layer_template['mapfile_layer']['layer_desc'] = 'Map Polygon'
            # Get rid of a few things we dont want
            layer_template['mapfile_layer']['shapefile']['timezone_fields'] = None
            layer_template['mapfile_layer']['classitem'] = None
            layer_template['mapfile_layer']['time_column'] = None
            # Modify the style object
            polygon_template['name'] = style_name
            polygon_template['styles'][0]['outlinecolor'] = color
            polygon_template['styles'][0]['style_width'] = style_width
            layer_template['mapfile_layer']['layer_classes'].append(polygon_template)
        else:
            raise ValueError("Can not write ERMA Data Package without template!!!")

        with open(output_path, "w") as o:
            json.dump(layer_template, o, indent=4)
        return {'shapefile_filename': shapefile_pathlib_path.with_suffix('.zip'),
                'json_filename': output_path}

    def make_spill_location_package_layer(self, id):
        shapefile_name = 'spill_location'
        output_path = os.path.join(self.tempdir.name, str(id)+".json")
        shz_name = os.path.join(self.tempdir.name, shapefile_name+'.shz')
        shapefile_pathlib_path = pathlib.Path(shz_name)
        layer_template = None
        layer_template_path = erma_data_package_data_dir / 'layer_template.json'
        default_spill_location_template = None
        default_spill_location_template_path = erma_data_package_data_dir / 'default_spill_location_template.json'
        with open(layer_template_path) as f:
            layer_template = json.load(f)
        with open(default_spill_location_template_path) as f:
            default_spill_location_template = json.load(f)
        if layer_template and default_spill_location_template:
            # Make a quick shapefile of the spill location(s)
            spill_ids = []
            spill_locations = []
            for spill_id, spill in enumerate(self.spills):
                spill_ids.append(spill_id)
                spill_locations.append(Point(spill.release.centroid))
            frame_data = {'Spill_id': spill_ids,
                          'Position': spill_locations
                          }
            gdf = gpd.GeoDataFrame(frame_data, crs='epsg:4326', geometry='Position')
            # Write out the zipped shapefile
            gdf.to_file(shz_name, driver='ESRI Shapefile',
                             engine="pyogrio")
            shutil.copy(shz_name, shapefile_pathlib_path.with_suffix('.zip'))
            # Now build the layer file
            # Folder name
            layer_template['folder_path'] = self.folder_name
            layer_template['title'] = 'Spill Location'
            layer_template['mapfile_layer']['shapefile']['name'] = 'spill_location_shapefile'
            layer_template['mapfile_layer']['shapefile']['description'] = 'Spill Location Shapefile'
            layer_template['mapfile_layer']['shapefile']['file'] = "file://source_files/" + shapefile_name + '.zip'
            layer_template['mapfile_layer']['layer_name'] = 'spill_location'
            layer_template['mapfile_layer']['layer_desc'] = 'Spill Location'
            # Get rid of a few things we dont want
            layer_template['mapfile_layer']['shapefile']['timezone_fields'] = None
            layer_template['mapfile_layer']['classitem'] = None
            layer_template['mapfile_layer']['time_column'] = None
            layer_template['mapfile_layer']['layer_classes'].append(default_spill_location_template)
        else:
            raise ValueError("Can not write ERMA Data Package without template!!!")

        with open(output_path, "w") as o:
            json.dump(layer_template, o, indent=4)
        return {'shapefile_filename': shapefile_pathlib_path.with_suffix('.zip'),
                'json_filename': output_path}

    def spills_match_style(self, spill1, spill2):
        appearance1 = appearance2 = colormap1 = colormap2 = None
        # Try to grab appearance and colormap data for the spills
        if spill1._appearance and spill1._appearance.colormap:
            appearance1 = spill1._appearance.to_dict()
            colormap1 = spill1._appearance.colormap.to_dict()
        if spill2._appearance and spill2._appearance.colormap:
            appearance2 = spill2._appearance.to_dict()
            colormap2 = spill2._appearance.colormap.to_dict()
        # If one has appearance/colormap and the other does not
        # we return False
        if ((appearance1 and not appearance2) or
            (colormap1 and not colormap2)):
            return False
        # If they both dont have appearance/colormap
        # we return True
        if ((not appearance1 and not appearance2) and
            (not colormap1 and not colormap2)):
            return True
        # Now we know we have appearance and colormap for both
        # spills, we can compare the important bits
        if ((appearance1['data'] == appearance2['data']) and
            (appearance1['units'] == appearance2['units']) and
            (colormap1['numberScaleDomain'] == colormap2['numberScaleDomain']) and
            (colormap1['colorScaleDomain'] == colormap2['colorScaleDomain']) and
            (colormap1['colorBlockLabels'] == colormap2['colorBlockLabels']) and
            (appearance1['scale'] == appearance2['scale'])):
            return True
        else:
            return False


    # Return a list of grouped ids
    def group_like_spills(self):
        spills_with_no_style = []
        spills_grouped_by_style = []
        for spill_id, spill in enumerate(self.spills):
            # If we have an appearance, we use that
            if spill._appearance and spill._appearance.colormap:
                appearance = spill._appearance.to_dict()
                colormap = spill._appearance.colormap.to_dict()
                found = False
                for group in spills_grouped_by_style:
                    # Look to see if it matches the first element in
                    # the group
                    if self.spills_match_style(self.spills[group[0]], spill):
                        group.append(spill_id)
                        found = True
                        break
                if not found:
                    # If it did not match a current group, we
                    # add a new group
                    spills_grouped_by_style.append([spill_id])
            else:
                spills_with_no_style.append(spill_id)
        if spills_with_no_style:
            spills_grouped_by_style.append(spills_with_no_style)
        return spills_grouped_by_style

    # Generate a cutoff stuct for contours if needed
    def generate_cutoff_struct(self):
        cutoff_struct = {}
        for spill_idx, spill in enumerate(self.spills):
            if spill._appearance and spill._appearance.colormap:
                appearance = spill._appearance.to_dict()
                colormap = spill._appearance.colormap.to_dict()
                requested_display_param = appearance['data']
                requested_display_unit = appearance['units']
                unit_map = {}
                if requested_display_param in self.default_unit_map:
                    unit_map = self.default_unit_map[requested_display_param]
                else:
                    raise ValueError(f'Style requested is not supported!!! {requested_display_param}')
                data_column = unit_map['column']
                # Looks like the domains are always in the base units...
                number_scale_domain = [val for val in colormap['numberScaleDomain']]
                cutoff_array = [val for val in colormap['colorScaleDomain']]
                cutoff_array.append(number_scale_domain[-1])
                cutoffs = []
                for idx, val in enumerate(cutoff_array):
                    colorblocklabel = colormap['colorBlockLabels'][idx]
                    color = colormap['colorScaleRange'][idx]
                    cutoffs.append({'cutoff': val,
                                    'cutoff_id': idx,
                                    'color': color,
                                    'label': colorblocklabel})
                cutoff_struct[spill_idx] = {'param': data_column,
                                            'cutoffs': cutoffs}
        return cutoff_struct

    def make_particle_package_layer(self, id, layer_name, uncertain, shapefile_filename):
        dir, basefile = os.path.split(shapefile_filename)
        output_path = dir+"/"+str(id)+".json"
        layer_template_path = erma_data_package_data_dir / 'layer_template.json'
        default_floating_template_path = erma_data_package_data_dir / 'default_floating_template.json'
        default_beached_template_path = erma_data_package_data_dir / 'default_beached_template.json'
        layer_template = None
        default_floating_template = default_beached_template = None
        with open(layer_template_path) as f:
            layer_template = json.load(f)
        with open(default_floating_template_path) as f:
            default_floating_template = json.load(f)
        with open(default_beached_template_path) as f:
            default_beached_template = json.load(f)

        if layer_template:
            # Folder name
            layer_template['folder_path'] = self.folder_name
            layer_template['title'] = layer_name
            if uncertain:
                layer_template['opacity'] = '0.75'

            layer_template['mapfile_layer']['shapefile']['name'] = basefile
            layer_template['mapfile_layer']['shapefile']['description'] = basefile
            layer_template['mapfile_layer']['shapefile']['file'] = "file://source_files/" + basefile
            # If we have a timezone, write that into the timezone_fields
            if self.timezone:
                layer_template['mapfile_layer']['shapefile']['timezone_fields'] = {"time": self.timezone}
            # Check the timestep and set the time override for ERMA time slider
            if self.time_step_override and self.time_unit_override:
                layer_template['time_step_override'] = self.time_step_override
                layer_template['time_unit_override'] = self.time_unit_override
            else:
                timestepseconds = None
                if self.output_timestep:
                    timestepseconds = int(self.output_timestep.seconds)
                else:
                    timestepseconds = int(self.model_timestep)
                minute = 60
                hour = minute*60
                day = hour*24
                month = day*30
                year = day*365
                if timestepseconds < hour:
                    # Less than an hour... so use minutes
                    layer_template['time_step_override'] = int(timestepseconds / minute)
                    layer_template['time_unit_override'] = 'minute'
                elif timestepseconds < day:
                    # Less than a day, use hours
                    layer_template['time_step_override'] = int(timestepseconds / hour)
                    layer_template['time_unit_override'] = 'hour'
                elif timestepseconds < month:
                    # Less than a month, use day
                    layer_template['time_step_override'] = int(timestepseconds / day)
                    layer_template['time_unit_override'] = 'day'
                elif timestepseconds < year:
                    # Less than a year, use month
                    layer_template['time_step_override'] = int(timestepseconds / month)
                    layer_template['time_unit_override'] = 'month'
                else:
                    # More than a year, use year
                    layer_template['time_step_override'] = int(timestepseconds / year)
                    layer_template['time_units_override'] = 'year'
            if not self.default_erma_styling:
                classorder = itertools.count()
                spills_grouped_by_style = []
                if self.disable_legend_collapse:
                    # In the case we dont want to collapse, we just group
                    # each spill on its own
                    for s_id, s in enumerate(self.spills):
                        spills_grouped_by_style.append([s_id])
                else:
                    # We can try to group spills with like styling...
                    spills_grouped_by_style = self.group_like_spills()
                # Now we loop through the spill groups and style them
                for spill_group in spills_grouped_by_style:
                    # Since we are styling by group, we grab the first spill
                    # from this group
                    spill_id = spill_group[0]
                    spill = self.spills[spill_id]
                    # Build a string indicating the ids in this group for
                    # the mapserver expression
                    spill_group_string = (',').join([str(s) for s in spill_group])
                    # For the legend, we group the names
                    name_string_list = []
                    for s_id in spill_group:
                        spill_name = (f'{self.spills[s_id].name} '
                                      f'({self.spills[s_id].amount}'
                                      f'{self.spills[s_id].units})')
                        name_string_list.append(spill_name)
                    # In ERMA the pipe puts them on separate lines in the legend
                    spill_names = ('|').join(name_string_list)
                    # If we have an appearance, we use that
                    if spill._appearance and spill._appearance.colormap and not self.default_erma_styling:
                        appearance = spill._appearance.to_dict()
                        colormap = spill._appearance.colormap.to_dict()
                        requested_display_param = appearance['data']
                        requested_display_unit = appearance['units']
                        unit_map = {}
                        if requested_display_param in self.default_unit_map:
                            unit_map = self.default_unit_map[requested_display_param]
                        else:
                            raise ValueError(f'Style requested is not supported!!! {requested_display_param}')
                        # Looks like the domains are always in the base units...
                        number_scale_domain = [val for val in colormap['numberScaleDomain']]
                        color_scale_domain = [val for val in colormap['colorScaleDomain']]
                        #number_scale_domain = [uc.convert(requested_display_unit,
                        #                                  unit_map['unit'], val) for val in colormap['numberScaleDomain']]
                        #color_scale_domain = [uc.convert(requested_display_unit,
                        #                                 unit_map['unit'], val) for val in colormap['colorScaleDomain']]
                        min = float(number_scale_domain[0])
                        max = None
                        data_column = unit_map['column']
                        style_size = 8 * appearance['scale']
                        if uncertain:
                            uncertain_color = '#FF0000'
                            # In the uncertain case, we just color everything red
                            floating_class_template = copy.deepcopy(default_floating_template)
                            beached_class_template = copy.deepcopy(default_beached_template)
                            floating_class_template['expression'] = ('[statuscode] = 2 AND '
                                                                     f'[spill_id] IN "{spill_group_string}"')
                            floating_class_template['styles'][0]['color'] = uncertain_color
                            floating_class_template['styles'][0]['style_size'] = style_size
                            floating_class_template['name'] = f'{spill_names}|Floating Uncertain'
                            beached_class_template['expression'] = ('[statuscode] = 3 AND '
                                                                    f'[spill_id] IN "{spill_group_string}"')
                            beached_class_template['styles'][0]['color'] = uncertain_color
                            beached_class_template['styles'][0]['outlinecolor'] = uncertain_color
                            beached_class_template['styles'][0]['style_size'] = style_size
                            beached_class_template['name'] = f'{spill_names}|Beached Uncertain'
                            layer_template['mapfile_layer']['layer_classes'].append(floating_class_template)
                            layer_template['mapfile_layer']['layer_classes'].append(beached_class_template)
                        else:
                            # Loop through the color scale and make classes
                            for idx, color in enumerate(colormap['colorScaleRange']):
                                if (idx+1) > len(color_scale_domain):
                                    max = float(number_scale_domain[1])
                                else:
                                    max = float(color_scale_domain[idx])
                                # Convert our min/max with unit conversion for labeling
                                if requested_display_param == 'Mass':
                                    converted_min = convert_mass_to_mass_or_volume(unit_map['unit'],
                                                                                   requested_display_unit,
                                                                                   spill.substance.standard_density,
                                                                                   min)
                                    converted_max = convert_mass_to_mass_or_volume(unit_map['unit'],
                                                                                   requested_display_unit,
                                                                                   spill.substance.standard_density,
                                                                                   max)
                                elif requested_display_param == 'Surface Concentration':
                                    converted_min = uc.convert('oil concentration',
                                                               unit_map['unit'],
                                                               requested_display_unit,
                                                               min)
                                    converted_max = uc.convert('oil concentration',
                                                               unit_map['unit'],
                                                               requested_display_unit,
                                                               max)
                                elif requested_display_param == 'Viscosity':
                                    converted_min = uc.convert('kinematic viscosity',
                                                               unit_map['unit'],
                                                               requested_display_unit,
                                                               min)
                                    converted_max = uc.convert('kinematic viscosity',
                                                               unit_map['unit'],
                                                               requested_display_unit,
                                                               max)
                                else:
                                    converted_min = uc.convert(unit_map['unit'],
                                                               requested_display_unit,
                                                               min)
                                    converted_max = uc.convert(unit_map['unit'],
                                                               requested_display_unit,
                                                               max)
                                # A mapserver class per color
                                floating_class_template = copy.deepcopy(default_floating_template)
                                beached_class_template = copy.deepcopy(default_beached_template)
                                units = unit_map['unit']
                                class_label = ''
                                if idx == 0 and len(colormap['colorScaleRange']) == 1:
                                    # Special case that we only have one range... so we show ALL
                                    # particles, but still label it with the range
                                    floating_class_template['expression'] = ('[statuscode] = 2 AND '
                                                                             f'[spill_id] IN "{spill_group_string}"')
                                    beached_class_template['expression'] = ('[statuscode] = 3 AND '
                                                                            f'[spill_id] IN "{spill_group_string}"')
                                    class_label = f'<{converted_min:#.4g} - {data_column} - {converted_max:#.4g}+ ({requested_display_unit})'
                                elif idx == 0:
                                    # First one... open ended lower range
                                    floating_class_template['expression'] = ('[statuscode] = 2 AND '
                                                                             f'[spill_id] IN "{spill_group_string}" AND '
                                                                             f'[{data_column}] <= {max}')
                                    beached_class_template['expression'] = ('[statuscode] = 3 AND '
                                                                            f'[spill_id] IN "{spill_group_string}" AND '
                                                                            f'[{data_column}] <= {max}')
                                    class_label = f'<{converted_min:#.4g} - {data_column} - {converted_max:#.4g} ({requested_display_unit})'
                                elif idx == len(colormap['colorScaleRange'])-1:
                                    # Last one... open ended upper range
                                    floating_class_template['expression'] = ('[statuscode] = 2 AND '
                                                                             f'[spill_id] IN "{spill_group_string}" AND '
                                                                             f'{min} <= [{data_column}]')
                                    beached_class_template['expression'] = ('[statuscode] = 3 AND '
                                                                            f'[spill_id] IN "{spill_group_string}" AND '
                                                                            f'{min} <= [{data_column}]')
                                    class_label = f'{converted_min:#.4g} - {data_column} - {converted_max:#.4g}+ ({requested_display_unit})'
                                else:
                                    floating_class_template['expression'] = ('[statuscode] = 2 AND '
                                                                             f'[spill_id] IN "{spill_group_string}" AND '
                                                                             f'({min} < [{data_column}] AND '
                                                                             f'[{data_column}] <= {max})')
                                    beached_class_template['expression'] = ('[statuscode] = 3 AND '
                                                                            f'[spill_id] IN "{spill_group_string}" AND '
                                                                            f'({min} < [{data_column}] AND '
                                                                            f'[{data_column}] <= {max})')
                                    class_label = f'{converted_min:#.4g} - {data_column} - {converted_max:#.4g} ({requested_display_unit})'
                                colorblocklabel = colormap['colorBlockLabels'][idx]
                                floating_class_template['name'] = f'{spill_names}|{colorblocklabel if colorblocklabel else class_label}'
                                floating_class_template['ordering'] = next(classorder)
                                floating_class_template['styles'][0]['color'] = color
                                floating_class_template['styles'][0]['style_size'] = style_size
                                beached_class_template['name'] = f'{spill_names}|Beached'
                                beached_class_template['ordering'] = next(classorder)
                                beached_class_template['styles'][0]['color'] = color
                                beached_class_template['styles'][0]['outlinecolor'] = color
                                beached_class_template['styles'][0]['style_size'] = style_size
                                min = max
                                layer_template['mapfile_layer']['layer_classes'].append(floating_class_template)
                                layer_template['mapfile_layer']['layer_classes'].append(beached_class_template)
                            ## # Loop through the classes... and set "ordering" in reverse order
                            ## for idx, layer_class in enumerate(list(reversed(layer_template['mapfile_layer']['layer_classes']))):
                            ##     layer_class['ordering'] = idx
                    else:
                        # No appearance data... use a default
                        floating_class_template = copy.deepcopy(default_floating_template)
                        beached_class_template = copy.deepcopy(default_beached_template)
                        floating_class_template['expression'] = ('[statuscode] = 2 AND '
                                                                 f'[spill_id] IN "{spill_group_string}"')
                        beached_class_template['expression'] = ('[statuscode] = 3 AND '
                                                                f'[spill_id] IN "{spill_group_string}"')
                        if uncertain:
                            uncertain_color = '#FF0000'
                            floating_class_template['styles'][0]['color'] = uncertain_color
                            floating_class_template['styles'][0]['outlinecolor'] = uncertain_color
                            floating_class_template['name'] = f'{spill_names}|Floating'
                            beached_class_template['styles'][0]['color'] = uncertain_color
                            beached_class_template['styles'][0]['outlinecolor'] = uncertain_color
                            beached_class_template['name'] = f'{spill_names}|Beached'
                        else:
                            floating_class_template['name'] = f'{spill_names}|Floating'
                            beached_class_template['name'] = f'{spill_names}|Beached'
                        layer_template['mapfile_layer']['layer_classes'].append(floating_class_template)
                        layer_template['mapfile_layer']['layer_classes'].append(beached_class_template)
                # Loop through the classes... and set "ordering" in reverse order
                for idx, layer_class in enumerate(list(reversed(layer_template['mapfile_layer']['layer_classes']))):
                    layer_class['ordering'] = idx
                with open(output_path, "w") as o:
                    json.dump(layer_template, o, indent=4)
                return {'shapefile_filename': shapefile_filename,
                        'json_filename': dir+"/"+str(id)+".json"}
            else:
                # Default styling was selected... collapse everything to simple styling
                # No appearance data... use a default
                floating_class_template = copy.deepcopy(default_floating_template)
                beached_class_template = copy.deepcopy(default_beached_template)
                floating_class_template['expression'] = '[statuscode] = 2'
                beached_class_template['expression'] = '[statuscode] = 3'
                if uncertain:
                    uncertain_color = '#FF0000'
                    floating_class_template['styles'][0]['color'] = uncertain_color
                    floating_class_template['styles'][0]['outlinecolor'] = uncertain_color
                    floating_class_template['name'] = 'Floating'
                    beached_class_template['styles'][0]['color'] = uncertain_color
                    beached_class_template['styles'][0]['outlinecolor'] = uncertain_color
                    beached_class_template['name'] = 'Beached'
                else:
                    floating_class_template['name'] = 'Floating'
                    beached_class_template['name'] = 'Beached'
                    layer_template['mapfile_layer']['layer_classes'].append(floating_class_template)
                    layer_template['mapfile_layer']['layer_classes'].append(beached_class_template)
                with open(output_path, "w") as o:
                    json.dump(layer_template, o, indent=4)
                return {'shapefile_filename': shapefile_filename,
                        'json_filename': dir+"/"+str(id)+".json"}
        else:
            raise ValueError("Can not write ERMA Data Package without template!!!")

    def rewind(self):
        '''
        reset a few parameter and call base class rewind to reset
        internal variables.
        '''
        super(ERMADataPackageOutput, self).rewind()

        self._middle_of_run = False
        self._start_idx = 0

    def clean_output_files(self):
        '''
        deletes ouput files that may be around
        called by prepare_for_model_run
        here in case it needs to be called from elsewhere
        '''
        super(ERMADataPackageOutput, self).clean_output_files()
