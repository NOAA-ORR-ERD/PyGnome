
"""
renderer_gd.py

module to hold all the map rendering code.

This one used the new map_canvas_gd, which uses the gd rendering lib.

"""
import os
from os.path import basename
import glob
import copy
import zipfile
import numpy as np

from colander import SchemaNode, String, drop

from gnome.persist import base_schema, class_from_objtype

from . import Outputter, BaseSchema
from gnome.utilities.map_canvas_gd import MapCanvas
from gnome.utilities.serializable import Field
from gnome.utilities.file_tools import haz_files
from gnome.utilities import projections

from gnome.basic_types import oil_status


class RendererSchema(BaseSchema):

    # not sure if bounding box needs defintion separate from LongLatBounds
    viewport = base_schema.LongLatBounds()

    # following are only used when creating objects, not updating -
    # so missing=drop
    map_filename = SchemaNode(String(), missing=drop)
    projection = SchemaNode(String(), missing=drop)
    image_size = base_schema.ImageSize(missing=drop)
    output_dir = SchemaNode(String())
    draw_ontop = SchemaNode(String())


class Renderer(Outputter, MapCanvas):

    """
    Map Renderer

    class that writes map images for GNOME results:
        writes the frames for the LE "movies", etc.

    """

    # This defines the colors used for the map
    #   -- they can then be referenced by name in the rest of the code.
    map_colors = [('background', (255, 255, 255)), # white
                  ('lake', (255, 255, 255)), # white
                  ('land', (255, 204, 153)), # brown
                  ('LE', (0, 0, 0)), # black
                  ('uncert_LE', (255, 0, 0)), # red
                  ('map_bounds', (175, 175, 175)), # grey
                  ('spillable_area', (255, 0, 0)), #  red
                  ('raster_map', (51, 102, 0)), # dark green
                  ('raster_map_outline', (0, 0, 0)), # black
                  ]

    background_map_name = 'background_map.png'
    foreground_filename_format = 'foreground_{0:05d}.png'
    foreground_filename_glob = 'foreground_?????.png'

    # todo: how should output_dir be saved? Absolute? Currently, it is relative
    # no, it's not.... see issue:
    _update = ['viewport', 'map_BB', 'image_size', 'draw_ontop']
    _create = ['image_size', 'projection', 'draw_ontop']

    _create.extend(_update)
    _state = copy.deepcopy(Outputter._state)
    _state.add(save=_create, update=_update)
    _state.add_field(Field('map_filename',
                           isdatafile=True,
                           save=True,
                           read=True,
                           test_for_eq=False))
    _state += Field('output_dir', save=True, update=True, test_for_eq=False)
    _schema = RendererSchema

    @classmethod
    def new_from_dict(cls, dict_):
        """
        change projection_type from string to correct type for loading from
        save file
        """
        viewport = dict_.pop('viewport')
        if 'projection' in dict_:
            # assume dict_ is from a save file since only the save file stores
            # the 'projection' (why does this matter??)
            # todo:
            # The 'projection' isn't stored as a nested object - should
            # revisit this and see if we can make it consistent with nested
            # objects .. but this works!

            # creates an instance of the projection class
            proj = class_from_objtype(dict_.pop('projection'))()
            # then creates the object
            obj = cls(projection=proj, **dict_)
            obj.viewport = viewport
        else:
            obj = super(Renderer, cls).new_from_dict(dict_)
            obj.viewport = viewport
        return obj

    def __init__(
        self,
        map_filename=None,
        output_dir='./',
        image_size=(800, 600),
        projection=None,
        viewport=None,
        map_BB=None,
        draw_back_to_fore=True,
        draw_map_bounds=False,
        draw_spillable_area=False,
        cache=None,
        output_timestep=None,
        output_zero_step=True,
        output_last_step=True,
        draw_ontop='forecast',
        name=None,
        on=True,
        **kwargs
        ):
        """
        Init the image renderer.

        :param map_filename=None: name of file for basemap (BNA)
        :type map_filename: string

        :param output_dir='./': directory to output the images
        :type output_dir: string

        :param image_size=(800, 600): size of images to output
        :type image_size: 2-tuple of integers

        :param projection = projections.FlatEarthProjection(): projection to use
        :type projection: a gnome.utilities.projection.Projection instance

        :param viewport: viewport of map -- what gets drawn and on what
                         scale. Default is full globe: (((-180, -90), (180, 90)))
        :type viewport: pair of (lon, lat) tuples ( lower_left, upper right )

        :param map_BB=None: bounding box of map if None, it will use the
                            bounding box of the mapfile.

        :param draw_back_to_fore=True: draw the background (map) to the
                                       foregound image when outputting the images
                                       each time step.
        :type draw_back_to_fore: boolean

        Following args are passed to base class Outputter's init:

        :param cache: sets the cache object from which to read data. The model
            will automatically set this param

        :param output_timestep: default is None in which case everytime the
            write_output is called, output is written. If set, then output is
            written every output_timestep starting from model_start_time.
        :type output_timestep: timedelta object

        :param output_zero_step: default is True. If True then output for
            initial step (showing initial release conditions) is written
            regardless of output_timestep
        :type output_zero_step: boolean

        :param output_last_step: default is True. If True then output for
            final step is written regardless of output_timestep
        :type output_last_step: boolean

        :param draw_ontop: draw 'forecast' or 'uncertain' LEs on top. Default
            is to draw 'forecast' LEs, which are in black on top
        :type draw_ontop: str


        Remaining kwargs are passed onto baseclass's __init__ with a direct
        call: Outputter.__init__(..)

        """
        projection = projections.FlatEarthProjection() if projection is None else projection
        # set up the canvas
        self.map_filename = map_filename

        if map_filename is not None:
            self.land_polygons = haz_files.ReadBNA(map_filename, 'PolygonSet')
        else:
            self.land_polygons = [] # empty list so we can loop thru it

        self.last_filename = ''
        self.draw_ontop = draw_ontop
        self.draw_back_to_fore = draw_back_to_fore

        Outputter.__init__(self,
                           cache,
                           on,
                           output_timestep,
                           output_zero_step,
                           output_last_step,
                           name,
                           output_dir
                           )

        if map_BB is None:
            if not self.land_polygons:
                map_BB = ((-180, -90), (180, 90))
            else:
                map_BB = self.land_polygons.bounding_box
        self.map_BB = map_BB

        MapCanvas.__init__(self,
                           image_size,
                           projection=projection,
                           viewport=self.map_BB)

        # assorted rendering flags:
        self.draw_map_bounds=draw_map_bounds
        self.draw_spillable_area=draw_spillable_area
        self.raster_map = None
        self.raster_map_fill=True
        self.raster_map_outline=False

        # initilize the images:
        self.add_colors(self.map_colors)
        self.background_color='background'


    @property
    def map_filename(self):
        return basename(self._filename)
    @map_filename.setter
    def map_filename(self, name):
        self._filename = name

    @property
    def draw_ontop(self):
        return self._draw_ontop
    @draw_ontop.setter
    def draw_ontop(self, val):
        if val not in ['forecast', 'uncertain']:
            raise ValueError("'draw_ontop' must be either 'forecast' or"
                             "'uncertain'. {0} is invalid.".format(val))
        self._draw_ontop = val

    def output_dir_to_dict(self):
        return os.path.abspath(self.output_dir)

    def prepare_for_model_run(self, *args, **kwargs):
        """
        prepares the renderer for a model run.

        Parameters passed to base class (use super): model_start_time, cache

        Does not take any other input arguments; however, to keep the interface
        the same for all outputters, define **kwargs and pass into base class

        In this case, it draws the background image and clears the previous
        images. If you want to save the previous images, a new output dir
        should be set.
        """
        super(Renderer, self).prepare_for_model_run(*args, **kwargs)

        self.clean_output_files()

        self.draw_background()
        self.save_background(os.path.join(self.output_dir,
                                          self.background_map_name)
                             )

    def clean_output_files(self):

        # clear out the output dir:
        try:
            os.remove(os.path.join(self.output_dir,
                                   self.background_map_name))
        except OSError:
            # it's not there to delete..
            pass

        foreground_filenames = glob.glob(os.path.join(self.output_dir,
                self.foreground_filename_glob))
        for name in foreground_filenames:
            os.remove(name)

    def draw_background(self):
        """
        Draws the background image -- just land for now

        This should be called whenever the scale changes
        """
        # create a new background image
        self.clear_background()
        self.draw_land()
        if self.raster_map is not None:
            self.draw_raster_map()

    def draw_land(self):
        """
        Draws the land map to the internal background image.
        """

        for poly in self.land_polygons:
            if poly.metadata[1].strip().lower() == 'map bounds':
                if self.draw_map_bounds:
                    self.draw_polygon(poly,
                                       line_color='map_bounds',
                                       fill_color=None,
                                       line_width=2,
                                       background=True)
            elif poly.metadata[1].strip().lower().replace(' ','') == 'spillablearea':
                if self.draw_spillable_area:
                    self.draw_polygon(poly,
                                       line_color='spillable_area',
                                       fill_color=None,
                                       line_width=2,
                                       background=True)

            elif poly.metadata[2] == '2':
                # this is a lake
                self.draw_polygon(poly, fill_color='lake', background=True)
            else:
                self.draw_polygon(poly,
                                  fill_color='land', background=True)
        return None

    def draw_elements(self, sc):
        """
        Draws the individual elements to a foreground image

        :param sc: a SpillContainer object to draw

        """
        # TODO: add checks for the other status flags!

        if sc.num_released > 0:  # nothing to draw if no elements
            if sc.uncertain:
                color = 'uncert_LE'
            else:
                color = 'LE'

            positions = sc['positions']

            # which ones are on land?
            on_land = sc['status_codes'] == oil_status.on_land
            self.draw_points(positions[on_land],
                             diameter=2,
                             color='black',
                             #color=color,
                             shape="x")
            # draw the four pixels for the elements not on land and
            # not off the map
            self.draw_points(positions[~on_land],
                             diameter=2,
                             color=color,
                             shape="round")

    def draw_raster_map(self):
        """
        draws the raster map used for beaching to the image.

        draws a grid for the pixels

        this is pretty slow, but only used for diagnostics.
        (not bad for just the lines)
        """
        if self.raster_map is not None:
            raster_map = self.raster_map
            w, h = raster_map.bitmap.shape
            if self.raster_map_outline:
                # vertical lines
                for i in range(w):
                    coords = raster_map.projection.to_lonlat( np.array( ( (i, 0.0), (i, h) ), dtype=np.float64) )
                    self.draw_polyline(coords, background=True, line_color='raster_map_outline')
                # horizontal lines
                for i in range(h):
                    coords = raster_map.projection.to_lonlat( np.array( ( (0.0, i), (w, i) ), dtype=np.float64) )
                    self.draw_polyline(coords, background=True, line_color='raster_map_outline')

            if self.raster_map_fill:
                print "drawing filled rects..."
                for i in range(w):
                    for j in range(h):
                        if raster_map.bitmap[i,j] == 1:
                            rect = raster_map.projection.to_lonlat( np.array( ( (i, j), (i+1, j), (i+1, j+1), (i, j+1)), dtype=np.float64) )
                            self.draw_polygon(rect, fill_color='raster_map', background=True)


    def write_output(self, step_num, islast_step=False):
        """
        Render the map image, according to current parameters.

        :param step_num: the model step number you want rendered.
        :type step_num: int

        :param islast_step: default is False. Flag that indicates that step_num
            is last step. If 'output_last_step' is True then this is written
            out
        :type islast_step: bool

        :returns: A dict of info about this step number if this step
            is to be output, None otherwise.
            'step_num': step_num
            'image_filename': filename
            'time_stamp': time_stamp # as ISO string

        use super to call base class write_output method

        If this is last step, then data is written; otherwise
        prepare_for_model_step determines whether to write the output for
        this step based on output_timestep
        """

        super(Renderer, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        image_filename = os.path.join(self.output_dir,
                                      self.foreground_filename_format
                                      .format(step_num))

        self.clear_foreground
        if self.draw_back_to_fore:
            self.copy_back_to_fore()

        # draw data for self.draw_ontop second so it draws on top
        scp = self.cache.load_timestep(step_num).items()
        if len(scp) == 1:
            self.draw_elements(scp[0])
        else:
            if self.draw_ontop == 'forecast':
                self.draw_elements(scp[1])
                self.draw_elements(scp[0])
            else:
                self.draw_elements(scp[0])
                self.draw_elements(scp[1])

        time_stamp = scp[0].current_time_stamp.isoformat()
        self.save_foreground(image_filename)
        self.last_filename = image_filename

        return {'image_filename': image_filename,
                'time_stamp': time_stamp}

    def _draw(self, step_num):
        """
        create a small function so data arrays are garbage collected from
        memory after this function exits - it returns current_time_stamp
        """

        # draw data for self.draw_ontop second so it draws on top
        scp = self.cache.load_timestep(step_num).items()
        if len(scp) == 1:
            self.draw_elements(scp[0])
        else:
            if self.draw_ontop == 'forecast':
                self.draw_elements(scp[1])
                self.draw_elements(scp[0])
            else:
                self.draw_elements(scp[0])
                self.draw_elements(scp[1])

        return scp[0].current_time_stamp

    def projection_to_dict(self):
        """
        store projection class as a string for now since that is all that
        is required for persisting
        todo: This may not be the case for all projection classes, but keep
        simple for now so we don't have to make the projection classes
        serializable
        """
        return '{0}.{1}'.format(self.projection.__module__,
                                self.projection.__class__.__name__)

    def serialize(self, json_='webapi'):
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()

        if json_ == 'save':
            toserial['map_filename'] = self._filename

        return schema.serialize(toserial)

    def save(self, saveloc, references=None, name=None):
        '''
        update the 'output_dir' key in the json_ to point to directory
        inside saveloc, then save the json - do not copy image files or
        image directory over
        '''
        json_ = self.serialize('save')
        out_dir = os.path.split(json_['output_dir'])[1]
        # store output_dir relative to saveloc
        json_['output_dir'] = os.path.join('./', out_dir)

        return self._json_to_saveloc(json_, saveloc, references, name)

    @classmethod
    def loads(cls, json_data, saveloc, references=None):
        '''
        loads object from json_data

        prepend saveloc path to 'output_dir' and create output_dir in saveloc,
        then call super to load object
        '''
        if zipfile.is_zipfile(saveloc):
            saveloc = os.path.split(saveloc)[0]

        os.mkdir(os.path.join(saveloc, json_data['output_dir']))
        json_data['output_dir'] = os.path.join(saveloc,
                                               json_data['output_dir'])

        return super(Renderer, cls).loads(json_data, saveloc, references)
