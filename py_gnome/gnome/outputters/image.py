"""
image outputters

These will output images for use in the Web client / OpenLayers

NOTE: doesn't seem to be tested -- and may not be used anyway.

"""

import os
import tempfile
import base64

from collections.abc import Iterable

from colander import SequenceSchema
from gnome.persist.base_schema import GeneralGnomeObjectSchema

import numpy as np

from gnome.utilities.time_utils import date_to_sec

from gnome.utilities.map_canvas import MapCanvas

from . import Outputter, BaseOutputterSchema
from gnome.movers.c_current_movers import IceMoverSchema


class IceImageSchema(BaseOutputterSchema):
    ice_movers =  SequenceSchema(
        GeneralGnomeObjectSchema(
            acceptable_schemas=[IceMoverSchema]
        ),
        save=True, update=True, save_reference=True
    )
    '''
    Nothing is required for initialization
    '''


class IceImageOutput(Outputter):
    '''
        Class that outputs ice data as an image for each ice mover.

        The image is PNG encoded, then Base64 encoded to include in a
        JSON response.
    '''
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

            Use super to pass optional kwargs to base class __init__ method
        '''
        # this is a place where we store our gradient color infomration
        self.gradient_lu = {}

        self.map_canvas = MapCanvas(image_size,
                                    projection=projection,
                                    viewport=viewport,
                                    preset_colors='transparent')
        self.map_canvas.add_colors([('black', (0, 0, 0))])

        self.set_gradient_colors('thickness',
                                 color_range=((0, 0, 0x7f, 0x7f),  # dark blue
                                              (0, 0, 0x7f, 0x3f),  # dark blue
                                              (0, 0, 0x7f, 0x00),  # dark blue
                                              (0xff, 0, 0, 0x00)),  # red
                                 scale=(0.0, 6.0),
                                 num_colors=64)

        self.set_gradient_colors('concentration',
                                 color_range=((0x80, 0xc0, 0xd0, 0x7f),  # sky blue
                                              (0, 0x40, 0x60, 0x00)),  # dark blue
                                 scale=(0.0, 1.0),
                                 num_colors=64)

        super(IceImageOutput, self).__init__(**kwargs)

        if (isinstance(ice_movers, Iterable)
            and not isinstance(ice_movers, str)):
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
            :type num_colors: Number
        '''
        color_names = self.add_gradient_to_canvas(color_range,
                                                  gradient_name, num_colors)

        self.gradient_lu[gradient_name] = (scale, np.array(color_names))

    def add_gradient_to_canvas(self, color_range, color_prefix, num_colors):
        '''
            Add a color gradient to our palette

            NOTE: Probably not the most efficient way to do this.

            :param color_range: The colors that we would like to use to
                                generate our gradient
            :type color_range: A sequence of 2 or more 3-tuples

            :param color_prefix: The prefix that will be used in the naming
                                 of the colors in the gradient
            :type color_prefix: str

            :param num_colors: The number of gradient colors to generate
            :type num_colors: Number
        '''
        color_range_idx = list(range(len(color_range)))
        color_space = np.linspace(color_range_idx[0], color_range_idx[-1],
                                  num=num_colors)

        r_grad = np.interp(color_space, color_range_idx,
                           [c[0] for c in color_range])
        g_grad = np.interp(color_space, color_range_idx,
                           [c[1] for c in color_range])
        b_grad = np.interp(color_space, color_range_idx,
                           [c[2] for c in color_range])

        if all([len(c) >= 4 for c in color_range]):
            a_grad = np.interp(color_space, color_range_idx,
                               [c[3] for c in color_range])
        else:
            a_grad = np.array([0.] * num_colors)

        new_colors = []
        for i, (r, g, b, a) in enumerate(zip(r_grad, g_grad, b_grad, a_grad)):
            new_colors.append(('{}{}'.format(color_prefix, i), (r, g, b, a)))

        self.map_canvas.add_colors(new_colors)

        return [c[0] for c in new_colors]

    def lookup_gradient_color(self, gradient_name, values):
        try:
            (low_val, high_val), color_names = self.gradient_lu[gradient_name]
        except IndexError:
            return None

        scale_range = high_val - low_val
        q_step_range = scale_range / len(color_names)

        idx = ((values // q_step_range).astype(int)
               .clip(0, len(color_names) - 1))

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
            model_time = date_to_sec(sc.current_time_stamp)
            iso_time = sc.current_time_stamp.isoformat()

        thick_image, conc_image, bb = self.render_images(model_time)

        # web_mercator = 'EPSG:3857'
        equirectangular = 'EPSG:32662'

        # info to return to the caller
        output_dict = {'step_num': step_num,
                       'time_stamp': iso_time,
                       'thickness_image': thick_image,
                       'concentration_image': conc_image,
                       'bounding_box': bb,
                       'projection': equirectangular,
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

        # We kinda need to figure our our bounding box before doing the
        # rendering.  We will try to be efficient about it mainly by not
        # grabbing our grid data twice.
        mover_grid_bb = None
        mover_grids = []

        for mover in self.ice_movers:
            mover_grids.append(mover.get_grid_data())
            mover_grid_bb = mover.get_grid_bounding_box(mover_grids[-1],
                                                        mover_grid_bb)

        canvas.viewport = mover_grid_bb
        canvas.clear_background()

        # Here is where we draw our grid data....
        for mover, mover_grid in zip(self.ice_movers, mover_grids):
            mover_grid_bb = mover.get_grid_bounding_box(mover_grid,
                                                        mover_grid_bb)

            concentration, thickness = mover.get_ice_fields(model_time)

            thickness_colors = self.lookup_gradient_color('thickness',
                                                          thickness)
            concentration_colors = self.lookup_gradient_color('concentration',
                                                              concentration)

            dtype = mover_grid.dtype.descr
            unstructured_type = dtype[0][1]
            new_shape = mover_grid.shape + (len(dtype),)
            mover_grid = (mover_grid
                          .view(dtype=unstructured_type)
                          .reshape(*new_shape))

            for poly, tc, cc in zip(mover_grid,
                                    thickness_colors, concentration_colors):
                canvas.draw_polygon(poly, fill_color=tc)
                canvas.draw_polygon(poly, fill_color=cc, background=True)

        # py_gd does not currently have the capability to generate a .png
        # formatted buffer in memory. (libgd can be made to do this, but
        # the wrapper is yet to be written)
        # So we will just write to a tempfile and then read it back.
        # If we ever have to do this anywhere else, a context manager would be good.
        tempdir = tempfile.mkdtemp()
        tempfilename = os.path.join(tempdir, "gnome_temp_image_file.png")

        canvas.save_foreground(tempfilename)
        thickness_image = base64.b64encode(open(tempfilename, 'rb').read())

        canvas.save_background(tempfilename)
        coverage_image = base64.b64encode(open(tempfilename, 'rb').read())

        os.remove(tempfilename)
        os.rmdir(tempdir)

        return ("data:image/png;base64,{}".format(thickness_image),
                "data:image/png;base64,{}".format(coverage_image),
                mover_grid_bb)

    def ice_movers_to_dict(self):
        '''
        a dict containing 'obj_type' and 'id' for each object in
        list/collection
        '''
        return self._collection_to_dict(self.ice_movers)

