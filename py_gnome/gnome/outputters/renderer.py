"""
renderer_gd.py

module to hold all the map rendering code.

This one used the new map_canvas, which uses the gd rendering lib.

"""
import os

import glob

import numpy as np
import py_gd

from colander import SchemaNode, String, drop

from gnome.basic_types import oil_status

from gnome.utilities.file_tools import haz_files
from gnome.utilities.map_canvas import MapCanvas

from gnome.utilities import projections
from gnome.utilities.projections import ProjectionSchema

from gnome.environment.gridded_objects_base import Grid_S

from gnome.persist import base_schema
from gnome.persist.extend_colander import FilenameSchema

from . import Outputter, BaseOutputterSchema



class RendererSchema(BaseOutputterSchema):
    # not sure if bounding box needs defintion separate from LongLatBounds
    viewport = base_schema.LongLatBounds(save=True, update=True)

    # following are only used when creating objects, not updating -
    # so missing=drop
    map_filename = FilenameSchema(save=True, update=True,
                                  isdatafile=True, test_equal=False,
                                  missing=drop,)

    projection = ProjectionSchema(save=True, update=True, missing=drop)
    image_size = base_schema.ImageSize(save=True, update=False, missing=drop)
    output_dir = SchemaNode(String(), save=True, update=True, test_equal=False)
    draw_ontop = SchemaNode(String(), save=True, update=True)


class Renderer(Outputter, MapCanvas):
    """
    Map Renderer

    class that writes map images for GNOME results.

    Writes the frames for the LE "movies", etc.
    """
    # This defines the colors used for the map
    #   -- they can then be referenced by name in the rest of the code.
    map_colors = [('background', (255, 255, 255)),  # white
                  ('lake', (255, 255, 255)),  # white
                  ('land', (255, 204, 153)),  # brown
                  ('LE', (0, 0, 0)),  # black
                  ('uncert_LE', (255, 0, 0)),  # red
                  ('map_bounds', (175, 175, 175)),  # grey
                  ('spillable_area', (255, 0, 0)),  # red
                  ('raster_map', (51, 102, 0)),  # dark green
                  ('raster_map_outline', (0, 0, 0)),  # black
                  ('grid_1', (51, 78, 0)),
                  ('grid_2', (175, 175, 175)),
                  ]

    background_map_name = 'background_map.'
    foreground_filename_format = 'foreground_{0:05d}.'
    foreground_filename_glob = 'foreground_?????.*'

    _schema = RendererSchema

    def __init__(self,
                 map_filename=None,
                 output_dir='./',
                 image_size=(800, 600),
                 projection=None,
                 viewport=None,
                 map_BB=None,
                 land_polygons=None,
                 draw_back_to_fore=True,
                 draw_map_bounds=False,
                 draw_spillable_area=False,
                 formats=['png', 'gif'],
                 draw_ontop='forecast',
                 cache=None,
                 output_timestep=None,
                 output_zero_step=True,
                 output_last_step=True,
                 output_single_step=False,
                 output_start_time=None,
                 on=True,
                 timestamp_attrib={},
                 **kwargs
                 ):
        """
        Init the image renderer.

        :param map_filename=None: GnomeMap or name of file for basemap (BNA)
        :type map_filename: GnomeMap or PathLike (str or Path)

        :param str output_dir='./': directory to output the images

        :param 2-tuple image_size=(800, 600): size of images to output

        :param projection=None: projection instance to use: If None,
                                set to projections.FlatEarthProjection()
        :type projection: a gnome.utilities.projection.Projection instance

        :param viewport: viewport of map -- what gets drawn and on what scale.
                         Default is full globe: (((-180, -90), (180, 90)))
                         If not specifies, it will be set to the map's bounds.

        :type viewport: pair of (lon, lat) tuples ( lower_left, upper right )

        :param map_BB=None: bounding box of map if None, it will use the
                            bounding box of the mapfile.

        :param draw_back_to_fore=True: draw the background (map) to the
                                       foregound image when outputting
                                       the images each time step.
        :type draw_back_to_fore: boolean

        :param formats=['gif']: list of formats to output.
        :type formats: string or list of strings. Options are:
                       ['bmp', 'jpg', 'jpeg', 'gif', 'png']

        :param draw_ontop: draw 'forecast' or 'uncertain' LEs on top. Default
            is to draw 'forecast' LEs, which are in black on top
        :type draw_ontop: str

        Following args are passed to base class Outputter's init:

        :param cache: sets the cache object from which to read prop. The model
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

        Remaining kwargs are passed onto baseclass's __init__ with a direct
        call: Outputter.__init__(..)

        """
        projection = (projections.FlatEarthProjection()
                      if projection is None
                      else projection)

        # set up the canvas
        # self.map_filename = map_filename
        self.output_dir = output_dir

        self.map_filename = map_filename
        if map_filename is not None and land_polygons is None:
            try:
                # check to see if this is a map object
                self.land_polygons = map_filename.get_polygons()['land_polys']
                self.map_filename = map_filename.filename
            except AttributeError: # assume it's a filename
                self.land_polygons = haz_files.ReadBNA(map_filename, 'PolygonSet')
        elif land_polygons is not None:
            self.land_polygons = land_polygons
        else:
            self.land_polygons = []  # empty list so we can loop thru it

        self.last_filename = ''
        self.draw_ontop = draw_ontop
        self.draw_back_to_fore = draw_back_to_fore

        Outputter.__init__(self,
                           cache,
                           on,
                           output_timestep,
                           output_zero_step,
                           output_last_step,
                           output_single_step,
                           output_start_time,
                           output_dir,
                           **kwargs)

        if map_BB is None:
            if not self.land_polygons:
                map_BB = ((-180, -90), (180, 90))
            else:
                map_BB = self.land_polygons.bounding_box

        self.map_BB = map_BB

        viewport = self.map_BB if viewport is None else viewport

        MapCanvas.__init__(self, image_size, projection=projection,
                           viewport=viewport)

        # assorted rendering flags:
        self.draw_map_bounds = draw_map_bounds
        self.draw_spillable_area = draw_spillable_area

        self.raster_map = None
        self.raster_map_fill = True
        self.raster_map_outline = False

        # initilize the images:
        self.add_colors(self.map_colors)
        self.background_color = 'background'

        if self.map_filename is not None:
            file_prefix = os.path.splitext(self.map_filename)[0]
            sep = '_'
        else:
            file_prefix = sep = ''

        fn = '{}{}anim.gif'.format(file_prefix, sep)
        self.anim_filename = os.path.join(output_dir, fn)

        self.formats = formats
        self.delay = 50
        self.repeat = True
        self.timestamp_attribs = {}

        self.set_timestamp_attrib(**timestamp_attrib)

        self.grids = []
        self.props = []

    @property
    def delay(self):
        return self._delay if 'gif' in self.formats else -1

    @delay.setter
    def delay(self, d):
        self._delay = d

    @property
    def repeat(self):
        return self._repeat if 'gif' in self.formats else False

    @repeat.setter
    def repeat(self, r):
        self._repeat = r

    @property
    def map_filename(self):
        return os.path.basename(self._filename) if self._filename is not None else None

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

    @property
    def formats(self):
        return self._formats

    @formats.setter
    def formats(self, val):
        if isinstance(val, str):
            val = (val,)
        self._formats = val

    def output_dir_to_dict(self):
        return os.path.abspath(self.output_dir)

    def start_animation(self, filename):
        self.animation = py_gd.Animation(filename, self.delay)
        looping = 0 if self.repeat else -1

        self.logger.info('Starting Animation')
        self.animation.begin_anim(self.back_image, looping)

    def prepare_for_model_run(self, *args, **kwargs):
        """
        prepares the renderer for a model run.

        Parameters passed to base class (use super): model_start_time, cache

        Does not take any other input arguments; however, to keep the interface
        the same for all outputters, define ``**kwargs`` and pass into the
        base class

        In this case, it draws the background image and clears the previous
        images. If you want to save the previous images, a new output dir
        should be set.
        """
        super(Renderer, self).prepare_for_model_run(*args, **kwargs)

        self.clean_output_files()
        self.draw_background()

        for ftype in self.formats:
            if ftype == 'gif':
                self.start_animation(self.anim_filename)
            else:
                self.save_background(os.path.join(self.output_dir,
                                                  self.background_map_name + ftype),
                                     file_type=ftype)

    def set_timestamp_attrib(self, **kwargs):
        """
        Function to set details of the timestamp's appearance when printed.
        These details are stored as a dict.

        Recognized attributes:

        :param on: Turn the draw function on or off
        :type on: Boolean

        :param dt_format: Format string for strftime to format the timestamp
        :type dt_format: String

        :param background: Color of the text background.
                           Color must be present in foreground palette
        :type background: str

        :param color: Color of the font. Note that the color must be present
                      in the foreground palette
        :type color: str

        :param size: Size of the font, one of {'tiny', 'small', 'medium',
                                               'large', 'giant'}
        :type size: str

        :param position: x, y pixel coordinates of where to draw the timestamp.
        :type position: tuple

        :param align: The reference point of the text bounding box.
                      One of:
                      {'lt'(left top), 'ct', 'rt', 'l', 'r', 'lb', 'cb', 'rb'}
        :type align: str
        """
        self.timestamp_attribs.update(kwargs)

    def draw_timestamp(self, time):
        """
        Function that draws the timestamp to the foreground.
        Uses self.timestamp_attribs to determine it's appearance.

        :param time: the datetime object representing the timestamp
        :type time: datetime
        """
        d = self.timestamp_attribs
        on = d['on'] if 'on' in d else True

        if not on:
            return

        dt_format = d.get('format', '%c')

        background = d.get('background', 'white')

        color = d.get('color', 'black')

        size = d.get('size', 'small')

        default_position = (self.fore_image.width / 2, self.fore_image.height)
        position = d['position'] if 'position' in d else default_position

        align = d.get('alignment', 'cb')

        self.fore_image.draw_text(time.strftime(dt_format),
                                  position, size, color, align, background)

    def clean_output_files(self):

        # clear out the output dir:
        # get the files (could have different extensions)
        files = glob.glob(os.path.join(self.output_dir,
                                       self.background_map_name) + "*")
        files += glob.glob(os.path.join(self.output_dir,
                                        self.foreground_filename_glob))
        files += glob.glob(os.path.join(self.output_dir,
                                        self.anim_filename) + "*")

        for name in files:
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

        self.draw_graticule()
        self.draw_tags()
        self.draw_grids()

    def add_grid(self, grid, on=True, color='grid_1', width=2):
        layer = GridVisLayer(grid, self.projection, on, color, width)

        self.grids.append(layer)

    def draw_grids(self):
        for grid in self.grids:
            grid.draw_to_image(self.back_image)

    def add_vec_prop(self, prop, on=True,
                     color='LE', mask_color='uncert_LE',
                     size=3, width=1, scale=1000):
        layer = GridPropVisLayer(prop, self.projection, on,
                                 color, mask_color, size, width, scale)
        self.props.append(layer)

    def draw_props(self, time):
        for prop in self.props:
            prop.draw_to_image(self.fore_image, time)

    def draw_masked_nodes(self, grid, time):
        if grid.appearance['on'] and grid.appearance['mask'] is not None:
            var = grid.appearance['mask']
            masked_nodes = grid.masked_nodes(time, var)
            dia = grid.appearance['n_size']

            unmasked_nodes = np.ascontiguousarray(masked_nodes
                                                  .compressed().reshape(-1, 2))

            self.draw_points(unmasked_nodes, dia, 'black')

            masked = np.ascontiguousarray(masked_nodes[masked_nodes.mask]
                                          .prop.reshape(-1, 2))

            self.draw_points(masked, dia, 'uncert_LE')

    def draw_land(self):
        """
        Draws the land map to the internal background image.
        """
        for poly in self.land_polygons:
            metadata = poly.metadata

            if metadata[1].strip().lower() == 'map bounds':
                if self.draw_map_bounds:
                    self.draw_polygon(poly,
                                      line_color='map_bounds',
                                      fill_color=None,
                                      line_width=2,
                                      background=True)
            elif metadata[1].strip().lower().replace(' ', '') == 'spillablearea':
                if self.draw_spillable_area:
                    self.draw_polygon(poly,
                                      line_color='spillable_area',
                                      fill_color=None,
                                      line_width=2,
                                      background=True)
            elif metadata[2] == '2':
                # this is a lake
                try:
                    self.draw_polygon(poly, fill_color='lake', background=True)
                except ValueError:
                    pass  # py_gd can't handle 2pt polygons
            else:
                try:
                    self.draw_polygon(poly, fill_color='land', background=True)
                except ValueError:
                    pass  # py_gd can't handle 2pt polygons

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
                             # color=color,
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
            projection = raster_map.projection
            w, h = raster_map.raster.shape

            if self.raster_map_outline:
                # vertical lines
                for i in range(w):
                    coords = projection.to_lonlat(np.array(((i, 0.0),
                                                            (i, h)),
                                                           dtype=np.float64))
                    self.draw_polyline(coords, background=True,
                                       line_color='raster_map_outline')
                # horizontal lines
                for i in range(h):
                    coords = projection.to_lonlat(np.array(((0.0, i),
                                                            (w, i)),
                                                           dtype=np.float64))
                    self.draw_polyline(coords, background=True,
                                       line_color='raster_map_outline')

            if self.raster_map_fill:
                for i in range(w):
                    for j in range(h):
                        if raster_map.raster[i, j] == 1:
                            rect = (projection
                                    .to_lonlat(np.array(((i, j),
                                                         (i + 1, j),
                                                         (i + 1, j + 1),
                                                         (i, j + 1)),
                                                        dtype=np.float64)))
                            self.draw_polygon(rect, fill_color='raster_map',
                                              background=True)

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

        If this is last step, then prop is written; otherwise
        prepare_for_model_step determines whether to write the output for
        this step based on output_timestep
        """
        super(Renderer, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        image_filename = os.path.join(self.output_dir,
                                      self.foreground_filename_format.format(step_num))

        self.clear_foreground()

        if self.draw_back_to_fore:
            self.copy_back_to_fore()

        # draw prop for self.draw_ontop second so it draws on top
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

        time_stamp = scp[0].current_time_stamp

        self.draw_timestamp(time_stamp)
        self.draw_props(time_stamp)

        for ftype in self.formats:
            if ftype == 'gif':
                self.animation.add_frame(self.fore_image, self.delay)
            else:
                image_filename += ftype
                self.save_foreground(image_filename, file_type=ftype)

        self.last_filename = image_filename

        return {'image_filename': image_filename,
                'time_stamp': time_stamp}

    def post_model_run(self):
        """
        Override this method if a derived class needs to perform
        any actions after a model run is complete (StopIteration triggered)
        """
        if 'gif' in self.formats:
            self.animation.close_anim()

    def _draw(self, step_num):
        """
        create a small function so prop arrays are garbage collected from
        memory after this function exits - it returns current_time_stamp
        """

        # draw prop for self.draw_ontop second so it draws on top
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

    def to_dict(self, json_=None):
        dict_ = Outputter.to_dict(self, json_=json_)
        dict_['map_filename'] = self._filename
        if json_ == 'save':
            dict_['output_dir'] = os.path.join('./', dict_['output_dir'])
        return dict_


class GridVisLayer(object):
    def __init__(self, grid, projection, on=True,
                 color='grid_1', width=1):
        self.grid = grid
        self.projection = projection
        self.on = on

        self.lines = self._get_lines(grid)
        self.color = color
        self.width = width

    def _get_lines(self, grid):
        if isinstance(grid, Grid_S):
            name = 'node'

            lons = getattr(grid, name + '_lon')
            lats = getattr(grid, name + '_lat')

            return np.ma.dstack((lons[:], lats[:]))
        else:
            if grid.edges is None:
                grid.build_edges()

            return grid.nodes[grid.edges]

    def draw_to_image(self, img):
        '''
        Draws the grid to the image
        '''
        if not self.on:
            return

        lines = self.projection.to_pixel_multipoint(self.lines, asint=True)

        for l in lines:
            img.draw_polyline(l, line_color=self.color, line_width=self.width)

        if len(lines[0]) > 2:
            # curvilinear grid; ugrids never have line segments greater than
            # 2 points
            for l in lines.transpose((1, 0, 2)).copy():
                img.draw_polyline(l, line_color=self.color,
                                  line_width=self.width)


class GridPropVisLayer(object):

    def __init__(self, prop, projection, on=True,
                 color='LE', mask_color='uncert_LE',
                 size=3, width=1, scale=1000):
        self.prop = prop
        self.projection = projection
        self.on = on

        self.color = color
        self.mask_color = mask_color

        self.size = size
        self.width = width
        self.scale = scale

    def draw_to_image(self, img, time):
        if not self.on:
            return

        t0 = self.prop.time.index_of(time, extrapolate=True) - 1

        data_u = self.prop.variables[0].data[t0]
        data_v = self.prop.variables[1].data[t0]

        if len(self.prop.time) > 1:
            data_u2 = self.prop.variables[0].data[t0 + 1]
            data_v2 = self.prop.variables[1].data[t0 + 1]
        else:
            data_u2 = data_u
            data_v2 = data_v

        t_alphas = self.prop.time.interp_alpha(time, extrapolate=True)

        data_u = data_u + t_alphas * (data_u2 - data_u)
        data_v = data_v + t_alphas * (data_v2 - data_v)

        data_u = data_u.reshape(-1)
        data_v = data_v.reshape(-1)

        start = None

        try:
            start = self.prop.grid.nodes.copy().reshape(-1, 2)
        except AttributeError:
            start = np.column_stack((self.prop.grid.node_lon,
                                     self.prop.grid.node_lat))

        if self.prop.grid.infer_location(data_u) == 'faces':
            if self.prop.grid.face_coordinates is None:
                self.prop.grid.build_face_coordinates()
            start = self.prop.grid.face_coordinates

        if hasattr(data_u, 'mask'):
            start[data_u.mask] = [0., 0.]

        data_u *= self.scale * 8.9992801e-06
        data_v *= self.scale * 8.9992801e-06
        data_u /= np.cos(np.deg2rad(start[:, 1]))

        end = start.copy()
        end[:, 0] += data_u
        end[:, 1] += data_v

        if hasattr(data_u, 'mask'):
            end[data_u.mask] = [0., 0.]

        bounds = self.projection.image_box

        pt1 = ((bounds[0][0] <= start[:, 0]) * (start[:, 0] <= bounds[1][0]) *
               (bounds[0][1] <= start[:, 1]) * (start[:, 1] <= bounds[1][1]))

        pt2 = ((bounds[0][0] <= end[:, 0]) * (end[:, 0] <= bounds[1][0]) *
               (bounds[0][1] <= end[:, 1]) * (end[:, 1] <= bounds[1][1]))

        start = start[pt1 * pt2]
        end = end[pt1 * pt2]

        start = self.projection.to_pixel_multipoint(start, asint=True)
        end = self.projection.to_pixel_multipoint(end, asint=True)

        img.draw_dots(start, diameter=self.size, color=self.color)

        line = np.array([[0., 0.], [0., 0.]])

        for i in range(0, len(start)):
            line[0] = start[i]
            line[1] = end[i]
            img.draw_polyline(line,
                              line_color=self.color,
                              line_width=self.width)
