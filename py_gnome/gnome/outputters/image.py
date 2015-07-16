"""
image outputters

These will output images for use in the Web client / OpenLayers

"""
import copy

from gnome.utilities.serializable import Serializable, Field
from gnome.utilities.time_utils import date_to_sec

from gnome.utilities.map_canvas import MapCanvas

from . import Outputter, BaseSchema



class IceImageSchema(BaseSchema):
    '''
    Nothing is required for initialization
    '''


class IceImageOutput(Outputter, Serializable):
    '''
    Class that outputs ice data as an image for each ice mover.

    The image is PNG encodes, then Base64 encoded to include in a JSON response
    '''
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _state.add_field(Field('ice_movers',
                           save=True, update=True, iscollection=True))

    _schema = IceImageSchema

    def __init__(self, ice_movers, **kwargs):
        '''
        :param list current_movers: A list or collection of current grid mover
                                    objects.

        use super to pass optional \*\*kwargs to base class __init__ method
        '''
        self.ice_movers = ice_movers

        super(IceImageOutput, self).__init__(**kwargs)

        ## used to do the rendering
        image_size = (1000, 1000) # just to start
        self.map_canvas = MapCanvas(image_size)

    def write_output(self, step_num, islast_step=False):
        """
        Generate image from data
        """
        super(IceImageOutput, self).write_output(step_num, islast_step)

        if self.on is False or not self._write_step:
            return None

        for sc in self.cache.load_timestep(step_num).items():
            pass

        model_time = date_to_sec(sc.current_time_stamp)

        ## fixme: Can we really loop through the movers?
        ##        or should there be one IceImage outputter for each Ice Mover.
        for mover in self.ice_movers:
            mover_triangles = self.get_triangles(mover)
            ## this shojuld be separate, yes?
            ice_coverage, ice_thickness = mover.get_ice_fields(model_time)

            ## here is where we render....
            # do something with self.get_coverage_fc(ice_coverage, mover_triangles))
            # do somethign with self.get_thickness_fc(ice_thickness, mover_triangles))

        # info to return to the caller
        output_dict = {'step_num': step_num,
                       'time_stamp': sc.current_time_stamp.isoformat(),
                       'image': "data:image/png;base64,%s"%self.get_sample_image(),# dummy image
                       'bounding_box': ((-85.0, 29.0),(-55.0, 45.0)),
                       'projection': ("EPSG:3857"),
                       }
        print "returning output_dict"
        print output_dict.keys()
        return output_dict

    def get_sample_image(self):
        """
        this returns a base 64 encoded PNG image for testing -- just so we have something

        This should be removed when we have real functionality
        """
        ## hard-coding the base64 really confused my editor..
        image_file_file_path = os.path.join(os.path.split(__file__)[0], 'sample.b64')
        return open(image_file_file_path).read()

    def get_coverage_fc(self, coverage, triangles):
        return self.get_grouped_fc_from_1d_array(coverage, triangles,
                                                 'coverage',
                                                 decimals=2)

    def get_thickness_fc(self, thickness, triangles):
        return self.get_grouped_fc_from_1d_array(thickness, triangles,
                                                 'thickness',
                                                 decimals=1)

    def get_grouped_fc_from_1d_array(self, values, triangles,
                                     property_name, decimals):
        rounded = values.round(decimals=decimals)
        unique = np.unique(rounded)

        features = []
        for u in unique:
            matching = np.where(rounded == u)
            matching_triangles = (triangles[matching])

            dtype = matching_triangles.dtype.descr
            shape = matching_triangles.shape + (len(dtype),)

            coordinates = (matching_triangles.view(dtype='<f8')
                           .reshape(shape).tolist())

            prop_fmt = '{{:.{}f}}'.format(decimals)
            properties = {'{}'.format(property_name): prop_fmt.format(u)}

            feature = Feature(id="1",
                              properties=properties,
                              geometry=MultiPolygon(coordinates=coordinates
                                                    ))
            features.append(feature)

        return FeatureCollection(features)

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

    def get_triangles(self, mover):
        '''
        The triangle data that we get from the mover is in the form of
        indices into the points array.
        So we get our triangle data and points array, and then build our
        triangle coordinates by reference.
        '''
        triangle_data = self.get_triangle_data(mover)
        points = self.get_points(mover)

        dtype = triangle_data[0].dtype.descr
        unstructured_type = dtype[0][1]
        unstructured = (triangle_data.view(dtype=unstructured_type)
                        .reshape(-1, len(dtype))[:, :3])

        triangles = points[unstructured]

        return triangles

    def get_triangle_data(self, mover):
        return mover.mover._get_triangle_data()

    def get_points(self, mover):
        points = (mover.mover._get_points()
                  .astype([('long', '<f8'), ('lat', '<f8')]))
        points['long'] /= 10 ** 6
        points['lat'] /= 10 ** 6

        return points

    def rewind(self):
        'remove previously written files'
        super(IceImageOutput, self).rewind()

    def serialize(self, json_='webapi'):
        """
            Serialize our current velocities outputter to JSON
        """
        dict_ = self.to_serialize(json_)
        schema = self.__class__._schema()
        json_out = schema.serialize(dict_)

        json_out['ice_movers'] = []

        for cm in self.ice_movers:
            json_out['ice_movers'].append(cm.serialize(json_))

        return json_out

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for current mover
        """
        schema = cls._schema()
        _to_dict = schema.deserialize(json_)

        if 'ice_movers' in json_:
            _to_dict['ice_movers'] = []
            for i, cm in enumerate(json_['ice_movers']):
                cm_cls = class_from_objtype(cm['obj_type'])
                cm_dict = cm_cls.deserialize(json_['ice_movers'][i])

                _to_dict['ice_movers'].append(cm_dict)

        return _to_dict
