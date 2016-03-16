"""
image outputters

These will output images for use in the Web client / OpenLayers

"""
import os
import copy
import collections

import numpy as np

from gnome.utilities.serializable import Serializable, Field
from gnome.utilities.time_utils import date_to_sec

from gnome.utilities.map_canvas import MapCanvas

from gnome.persist import class_from_objtype, References
from gnome.persist.base_schema import CollectionItemsList

from . import Outputter, BaseSchema

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2, width=120)


class IceImageSchema(BaseSchema):
    '''
    Nothing is required for initialization
    '''


class IceImageOutput(Outputter):
    '''
        Class that outputs ice data as an image for each ice mover.

        The image is PNG encoded, then Base64 encoded to include in a
        JSON response.
    '''
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _state.add_field(Field('ice_movers',
                           save=True, update=True, iscollection=True))

    _schema = IceImageSchema

    def __init__(self, ice_movers=None,
                 image_size=(800, 600),
                 projection=None,
                 viewport=None,
                 **kwargs):
        '''
            :param ice_movers: ice_movers associated with this outputter.
            :type ice_movers: An ice_mover object or sequence of ice_mover
                              objects.

            Use super to pass optional \*\*kwargs to base class __init__ method
        '''
        # this is a place where we store our gradient color infomration
        self.gradient_lu = {}

        self.map_canvas = MapCanvas(image_size,
                                    projection=projection,
                                    viewport=viewport,
                                    preset_colors='transparent')
        self.map_canvas.add_colors([('black', (0, 0, 0))])

        self.set_gradient_colors('thickness',
                                 color_range=((0, 0, 0x7f),  # dark blue
                                              (0, 0xff, 0xff)),  # cyan
                                 scale=(0.0, 10.0),
                                 num_colors=16)
        self.set_gradient_colors('concentration',
                                 color_range=((0, 0x40, 0x60),  # dark blue
                                              (0x80, 0xc0, 0xd0)),  # sky blue
                                 scale=(0.0, 10.0),
                                 num_colors=16)

        super(IceImageOutput, self).__init__(**kwargs)

        if (isinstance(ice_movers, collections.Iterable) and
                not isinstance(ice_movers, str)):
            self.ice_movers = ice_movers
        elif ice_movers is not None:
            self.ice_movers = (ice_movers,)
        else:
            self.ice_movers = tuple()

    def set_gradient_colors(self, gradient_name,
                            color_range=((0, 0, 0x7f),  # dark blue
                                         (0, 0xff, 0xff)),  # cyan
                            scale=(0.0, 10.0),
                            num_colors=16):
        '''
            Add a color gradient to our palette representing the colors we
            will use for our ice thickness

            :param gradient_name: The name of the gradient.
            :type gradient_name: str

            :param color_range: The colors we will build our gradient with.
            :type color_range: A 2 element sequence of 3-tuples containing
                               8-bit RGB values.

            :param scale: A range of values representing the low and high end
                          of our gradient.
            :type scale: A 2 element sequence of float

            :param num_colors: The number of colors to use for the gradient.
            :type num_colors: int
        '''
        color_names = self.add_gradient_to_canvas(color_range,
                                                  gradient_name, num_colors)

        self.gradient_lu[gradient_name] = (scale, np.array(color_names))

    def add_gradient_to_canvas(self, color_range, color_prefix, num_colors):
        '''
            Add a color gradient to our palette

            NOTE: Probably not the most efficient way to do this.
        '''
        low, high = color_range[:2]
        r_grad = np.linspace(low[0], high[0], num_colors).round().astype(int)
        g_grad = np.linspace(low[1], high[1], num_colors).round().astype(int)
        b_grad = np.linspace(low[2], high[2], num_colors).round().astype(int)

        new_colors = []
        for i, (r, g, b) in enumerate(zip(r_grad, g_grad, b_grad)):
            new_colors.append(('{}{}'.format(color_prefix, i), (r, g, b)))

        self.map_canvas.add_colors(new_colors)

        return [c[0] for c in new_colors]

    def lookup_gradient_color(self, gradient_name, values):
        try:
            (low_val, high_val), color_names = self.gradient_lu[gradient_name]
        except IndexError:
            return None

        scale_range = high_val - low_val
        q_step_range = scale_range / len(color_names)

        idx = np.floor(values / q_step_range).astype(int)

        return color_names[idx]

    def write_output(self, step_num, islast_step=False):
        """
            Generate image from data
        """
        # I don't think we need this for this outputter:
        #   - it does stuff with cache initialization
        super(IceImageOutput, self).write_output(step_num, islast_step)

        if (self.on is False or
                not self._write_step or
                len(self.ice_movers) == 0):
            return None

        # fixme -- doing all this cache stuff just to get the timestep..
        # maybe timestep should be passed in.
        for sc in self.cache.load_timestep(step_num).items():
            pass

        model_time = date_to_sec(sc.current_time_stamp)

        thick_image, conc_image = self.render_images(model_time)

        # info to return to the caller
        output_dict = {'step_num': step_num,
                       'time_stamp': sc.current_time_stamp.isoformat(),
                       'thickness_image': thick_image,
                       'concentration_image': conc_image,
                       'bounding_box': self.map_canvas.viewport,
                       'projection': ("EPSG:3857"),
                       }

        return output_dict

    def get_sample_image(self):
        """
            This returns a base 64 encoded PNG image for testing,
            just so we have something

            This should be removed when we have real functionality
        """
        # hard-coding the base64 really confused my editor..
        image_file_file_path = os.path.join(os.path.split(__file__)[0],
                                            'sample.b64')

        return open(image_file_file_path).read()

    def render_images(self, model_time):
        """
            render the actual images
            This uses the MapCanvas code to do the actual rendering

            returns: thickness_image, concentration_image
        """
        canvas = self.map_canvas

        # Here is where we render....
        for mover in self.ice_movers:
            mover_grid = mover.get_grid_data()

            print 'mover_grid (shape = {}):'.format(mover_grid.shape)
            pp.pprint(mover_grid)

            print 'bounding rectangle:'
            pp.pprint(mover.get_grid_bounding_rect(mover_grid))

            ice_coverage, ice_thickness = mover.get_ice_fields(model_time)

            thickness_colors = self.lookup_gradient_color('thickness',
                                                          ice_thickness)

            dtype = mover_grid.dtype.descr
            unstructured_type = dtype[0][1]
            new_shape = mover_grid.shape + (len(dtype),)
            mover_grid = (mover_grid
                          .view(dtype=unstructured_type)
                          .reshape(*new_shape))

            for poly, color in zip(mover_grid, thickness_colors):
                canvas.draw_polygon(poly, fill_color=color)

            # self.get_coverage_fc(ice_coverage, mover_triangles)
            # self.get_thickness_fc(ice_thickness, mover_triangles)

        # diagnostic so we can see what we have rendered.
        canvas.draw_graticule(False)
        canvas.save_background('background.png')
        canvas.save_foreground('foreground.png')

        return ("data:image/png;base64,{}".format(self.get_sample_image()),
                "data:image/png;base64,{}".format(self.get_sample_image()))

    def get_rounded_ice_values(self, coverage, thickness):
        return np.vstack((coverage.round(decimals=2),
                          thickness.round(decimals=1))).T

    def get_unique_ice_values(self, ice_values):
        '''
            In order to make numpy perform this function fast, we will use a
            contiguous structured array using a view of a void type that
            joins the whole row into a single item.
        '''
        dtype = np.dtype((np.void,
                          ice_values.dtype.itemsize * ice_values.shape[1]))
        voidtype_array = np.ascontiguousarray(ice_values).view(dtype)

        _, idx = np.unique(voidtype_array, return_index=True)

        return ice_values[idx]

    def get_matching_ice_values(self, ice_values, v):
        return np.where((ice_values == v).all(axis=1))

    def rewind(self):
        'remove previously written files'
        super(IceImageOutput, self).rewind()

    def serialize(self, json_='webapi'):
        """
        Serialize this outputter to JSON
        """
        dict_ = self.to_serialize(json_)
        schema = self.__class__._schema()
        json_out = schema.serialize(dict_)

        if self.ice_mover is not None:
            json_out['ice_mover'] = self.ice_mover.serialize(json_)
        else:
            json_out['ice_mover'] = None

        return json_out

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for current mover
        """
        schema = cls._schema()
        _to_dict = schema.deserialize(json_)

        if 'ice_mover' in json_ and json_['ice_mover'] is not None:
            cm_cls = class_from_objtype(json_['ice_mover']['obj_type'])
            cm_dict = cm_cls.deserialize(json_['ice_mover'])
            _to_dict['ice_mover'] = cm_dict
        return _to_dict
