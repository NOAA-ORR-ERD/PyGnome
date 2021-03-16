"""
rasterer.py

module for results as raster of spill thickness.

author: Matt Hodge (HWR-llc)

"""
# temp imports from hodge
import pdb
from matplotlib import pyplot as plt
# permanent imports from hodge
from shapely import geometry, errors
from scipy.spatial import Voronoi, voronoi_plot_2d, distance
from scipy.interpolate import griddata

from pyproj import Proj, transform
import os
from os.path import basename
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



class RastererSchema(BaseOutputterSchema):
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


class Rasterer(Outputter):
    """
    class that write ASCII rasters for GNOME results.
    ...

    """


    _schema = RastererSchema

    def __init__(self,
                 map_filename=None,
                 output_dir='./',
                 projection=None,
                 map_BB=None,
                 land_polygons=None,
                 draw_map_bounds=False,
                 epsg_out=None,
                 epsg_in=None,
                 model_domain=None,
                 cache=None,
                 output_timestep=None,
                 output_zero_step=True,
                 output_last_step=True,
                 output_start_time=None,
                 on=True,
                 timestamp_attrib={},
                 **kwargs
                 ):
        """
        Init the results rasterer.

        :param str map_filename=None: name of file for basemap (BNA)
        :type map_filename: str

        :param str output_dir='./': directory to output the images

        :param projection=None: projection instance to use: If None,
                                set to projections.FlatEarthProjection()
        :type projection: a gnome.utilities.projection.Projection instance

        :param map_BB=None: bounding box of map if None, it will use the
                            bounding box of the mapfile.

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

        # set up boundaries
        self.map_filename = map_filename

        if map_filename is not None and land_polygons is None:
            self.land_polygons = haz_files.ReadBNA(map_filename, 'PolygonSet')
        elif land_polygons is not None:
            self.land_polygons = land_polygons
        else:
            self.land_polygons = []  # empty list so we can loop thru it

        # set up projections
        self.epsg_out = Proj(init=epsg_out)
        self.epsg_in = Proj(init='epsg:4326')
        Outputter.__init__(self,
                           cache,
                           on,
                           output_timestep,
                           output_zero_step,
                           output_last_step,
                           output_start_time,
                           output_dir,
                           **kwargs)

        
    def prepare_for_model_run(self, *args, **kwargs):
        """
        prepares the rasterer for a model run.

        Parameters passed to base class (use super): model_start_time, cache

        Does not take any other input arguments; however, to keep the interface
        the same for all outputters, define ``**kwargs`` and pass into the
        base class

        """
        super(Rasterer, self).prepare_for_model_run(*args, **kwargs)
       
        print('rasterer model run prep')
        self.model_domain = self.get_model_domain_geo_poly()

    def post_model_run(self):
        print('rasterer post model run actions')
    
        # support functions for processing
    def diagnostic_plot(self, vor_polys, **kwargs):       
        voronoi_plot_2d(vor_polys, show_points=True, show_vertices=False, line_colors='b')
        m_d_o_x, m_d_o_y = self.model_domain.exterior.xy  
        plt.plot(m_d_o_x, m_d_o_y, color='g', linewidth=1)
        if 'point' in kwargs.keys():
            # point must be an integer representing the splot point of interest
            point_index = kwargs['point']
            region_index = vor_polys.point_region[point_index]
            region_verts = vor_polys.regions[region_index]
            region_verts_coord_x = []
            region_verts_coord_y = []
            for vert in (region_verts):
                region_verts_coord_x.append(vor_polys.vertices[vert][0])
                region_verts_coord_y.append(vor_polys.vertices[vert][1])
            if -1 not in region_verts:
                region_verts_coord_x.append(vor_polys.vertices[region_verts[0]][0])
                region_verts_coord_y.append(vor_polys.vertices[region_verts[0]][1])
            font = {'family': 'serif',
                    'color': 'g',
                    'size': 12}       
            plt.plot(region_verts_coord_x, region_verts_coord_y, color='r', marker='o', linewidth=2)
            for vert in range(0,len(region_verts)):
                plt.text(region_verts_coord_x[vert], region_verts_coord_y[vert], region_verts[vert], fontdict=font)
        if 'extent' in kwargs.keys():
            # extent must be NumPy array [[minX, maxX][minY, maxY]]
            extents = kwargs['extent']
            plt.xlim(extents[0][0], extents[0,1])
            plt.ylim(extents[1][0], extents[1][1])
            
        if 'save' in kwargs.keys():
            # save must be a path to a directory where the file will be saved
            save_path = kwargs['save']
            # check for existing diagnostics
            diag_files_cnt = len([f for f in os.listdir(save_path) if 'diag_' in f])
            save_name = '/diag_' + str(diag_files_cnt).zfill(3) + '.png'
            plt.savefig(save_path + save_name)       
        plt.show()
        # example code snippets for kwargs
        # save
        # diagnostic_plot(vor_polys, self.model_domain, save=r'D:\Hodge.WaterResources\Development\oil\gnomeVerter\files\input\adcirc\gnome\output')
        # point
        # diagnostic_plot(vor_polys, self.model_domain, point=40)
        # extent
        # extent_window = np.array([[233100, 233300],[900300, 900400]])
        # diagnostic_plot(vor_polys, self.model_domain, extent=extent_window) 
    
    def get_thick(self, splots_xyms, **kwargs):
        if len(kwargs.keys()) == 0:
            diagnostic = False
        elif 'diagnostic' in kwargs.keys():
            diagnostic = kwargs['diagnostic']
        else:
            diagnostic = False
        splots_cnt = len(splots_xyms)
        pdb.set_trace()
        splots_xy_arr = splots_xyms[['x','y']].view((float,len(splots_xyms[['x','y']].dtype.names)))  
        # reshape???
        vor_polys = Voronoi(splots_xy_arr, furthest_site=False)
        # diagnostic_plot(vor_polys, self.model_domain, point=40)
        pdb.set_trace()    
        splots_thick = np.zeros((splots_cnt), dtype=[('x','float64'),('y','float64'),('kg_m2','float64')]) 
        # note: polygon order in vor_polys is the same order as they are entered in splots_xy_arr, splots_xyms
        # get max distance to point locations
        for point in range(0,len(vor_polys.points)):
            splots_thick[point]['x'] = vor_polys.points[point][0]
            splots_thick[point]['y'] = vor_polys.points[point][1]
            act_region_index = vor_polys.point_region[point]
            region_verts = vor_polys.regions[act_region_index]
            if splots_xyms['status_code'][point] == 3:
                if -1 in region_verts:
                    out_vert = region_verts.index(-1)
                    if out_vert == len(region_verts) - 1:
                        first_vert_index = 0
                        last_vert_index = out_vert - 1
                    elif out_vert == 0:
                        first_vert_index = out_vert
                        last_vert_index = len(region_verts) - 2 # one for end and one for removed -1
                    else:
                        last_vert_index = out_vert
                        first_vert_index = out_vert - 1            
                    region_verts = [vert for vert in region_verts if vert != -1]
                    poly_line_xys = [vor_polys.vertices[ind] for ind in region_verts]
                    first_point = geometry.Point(poly_line_xys[first_vert_index][0], poly_line_xys[first_vert_index][1])
                    last_point = geometry.Point(poly_line_xys[last_vert_index][0], poly_line_xys[last_vert_index][1])
                    if self.model_domain.contains(first_point) and self.model_domain.contains(last_point):
                        #first vertex
                        origin_vert = region_verts[first_vert_index]
                        new_xy_for_polyline = get_new_external_xy(point, origin_vert, act_region_index, vor_polys, self.model_domain, True)
                        if first_vert_index + 1 == last_vert_index:
                            poly_line_xys.insert(first_vert_index + 1, new_xy_for_polyline)
                        else:
                            poly_line_xys.insert(first_vert_index, new_xy_for_polyline)
                        # last vertex 
                        origin_vert = region_verts[last_vert_index]
                        new_xy_for_polyline = get_new_external_xy(point, origin_vert, act_region_index, vor_polys, self.model_domain, False)
                        if first_vert_index + 1 == last_vert_index:
                            poly_line_xys.insert(last_vert_index + 1,new_xy_for_polyline)
                        elif len(region_verts) - 1 == last_vert_index:
                            poly_line_xys.append(new_xy_for_polyline)
                        else:
                            # this is a catch for an unconsidered polygon definition
                            print 'case not considered, may need to debug'
                            pdb.set_trace()
                    elif self.model_domain.contains(first_point):
                        origin_vert = region_verts[first_vert_index]
                        new_xy_for_polyline = get_new_external_xy(point, origin_vert, act_region_index, vor_polys, self.model_domain, True)
                        if first_vert_index + 1 == last_vert_index:
                            poly_line_xys.insert(first_vert_index + 1, new_xy_for_polyline)
                        else:
                            poly_line_xys.insert(first_vert_index, new_xy_for_polyline)
                    elif self.model_domain.contains(last_point):
                        origin_vert = region_verts[last_vert_index]
                        new_xy_for_polyline = get_new_external_xy(point, origin_vert, act_region_index, vor_polys, self.model_domain, False)
                        if first_vert_index + 1 == last_vert_index:
                            poly_line_xys.insert(last_vert_index, new_xy_for_polyline)
                        elif len(region_verts) - 1 == last_vert_index:
                            poly_line_xys.append(new_xy_for_polyline) 
                else:
                    poly_line_xys = [vor_polys.vertices[ind] for ind in region_verts]
                try:
                    poly = geometry.Polygon(poly_line_xys)
                    poly = poly.intersection(self.model_domain)
                except errors.TopologicalError:
                    # this is a catch for a faulty voronoi diagram
                    pdb.set_trace()                              
                if poly.is_empty:
                    print 'entire polygon outside'
                    splots_thick[point]['kg_m2'] = splots_xyms[point]['mass']
                else:
                    new_poly_cent_x = poly.centroid.xy[0][0]
                    new_poly_cent_y = poly.centroid.xy[1][0]
                    splots_thick[point]['x'] = new_poly_cent_x
                    splots_thick[point]['y'] = new_poly_cent_y
                    if poly.area < 1:
                        splots_thick[point]['kg_m2'] = splots_xyms[point]['mass']
                    else:
                        splots_thick[point]['kg_m2'] = splots_xyms[point]['mass'] / poly.area 
            elif -1 in region_verts:
                splots_thick[point]['kg_m2'] = 0
            else:
                poly_line_xys = [vor_polys.vertices[ind] for ind in region_verts]
                poly = geometry.Polygon(poly_line_xys)
                try:
                    poly = poly.intersection(self.model_domain)
                except errors.TopologicalError:
                    # this is a catch for a faulty voronoi diagram
                    pdb.set_trace()                    
                splots_thick[point]['kg_m2'] = splots_xyms[point]['mass'] / poly.area
        if diagnostic == True:
            return splots_thick, [vor_polys]
        else:
            return splots_thick, []


    def transform_poly_points(self, points):
        sp_x, sp_y = transform(self.epsg_in, self.epsg_out, 
                               points[:,0],
                               points[:,1])
        ret_arr = []
        for i in range(0, len(sp_x)):
            ret_arr.append((sp_x[i], sp_y[i]))
        return ret_arr
    def get_spillable(self):
        sa_bool = False
        for i in range(0, len(self.land_polygons)):
            if 'SpillableArea' in self.land_polygons[i].metadata:
                sa_i = i
                sa_bool = True
                break
        if sa_bool == False:
            raise ValueError("rasterer requires 'SpillableArea'"
                             "in land_polygons")
        else:
            land_only = []
            for i in range(0, len(self.land_polygons)):
                if i != sa_i:
                    land_only.append(self.land_polygons[i])
            spillable = self.land_polygons[sa_i]
            return spillable, land_only             
            
    def get_model_domain_geo_poly(self):
        # find spillable area
        spill_poly, land_polys = self.get_spillable()
        sap_xy = self.transform_poly_points(spill_poly.points)
        sap_geo_poly = geometry.Polygon(sap_xy)
        inner_geo_poly = []
        edge_geo_poly = []
        for poly in land_polys:
            lp_xy = self.transform_poly_points(poly.points) 
            tmp_inner = geometry.Polygon(lp_xy)
            if sap_geo_poly.contains(tmp_inner):
                inner_geo_poly.append(tmp_inner.exterior.coords)
            else:
                edge_geo_poly.append(tmp_inner)
        tmp_mod_dom = geometry.Polygon(sap_geo_poly.exterior.coords, inner_geo_poly)
        for poly in edge_geo_poly:
            tmp_mod_dom = tmp_mod_dom.difference(poly)
            if isinstance(tmp_mod_dom, geometry.MultiPolygon):
                big_ind = 0
                big_area = 0
                for i in range(0, len(tmp_mod_dom)):
                    if tmp_mod_dom[i].area > big_area:
                        big_area = tmp_mod_dom[i].area
                        big_ind = i
                tmp_mod_dom = tmp_mod_dom[big_ind]
        model_domain = tmp_mod_dom

        # code for looking at model domain, move to diagnostic
#        x_sa, y_sa = sap_geo_poly.exterior.xy
#        plt.plot(x_sa, y_sa, 'k', linewidth=2)
#        x_md, y_md = model_domain.exterior.xy
#        plt.plot(x_md, y_md)
#        for i in range (len(model_domain.interiors)):
#            x_int, y_int = model_domain.interiors[i].xy
#            plt.plot(x_int, y_int)
#        plt.show()
        return model_domain
    def get_xyms(self, timestep):
            splots_cnt = len(timestep._data_arrays['id'])
            xyms = np.zeros((splots_cnt),
                            dtype=[('x', 'float64'), ('y','float64'), 
                                   ('mass','float64'), ('status_code', 'int32')])
            sc_x, sc_y = transform(self.epsg_in, self.epsg_out, 
                                   timestep._data_arrays['positions'][:,0],
                                   timestep._data_arrays['positions'][:,1])
            xyms['x'] = sc_x
            xyms['y'] = sc_y
            xyms['mass'] = timestep._data_arrays['mass']
            xyms['status_code'] = timestep._data_arrays['status_codes']
            return xyms





        
    def write_output(self, step_num, islast_step=False):
        """
        generate ascii raster file, according to current parameters.

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
        super(Rasterer, self).write_output(step_num, islast_step)
        if not self._write_step:
            return None # if _write_step False, do not write

        raster_filename = os.path.join(self.output_dir, str(step_num), '.asc')
        
        # pull relevant for processing
        scp = self.cache.load_timestep(step_num).items()
        if len(scp)  < 3: # place holder to consider forecasting
            spill_cert = scp[0]
            xyms = self.get_xyms(spill_cert)

            xyth, grp_voronoi = self.get_thick(xyms, diagnostic=True)
            pdb.set_trace()


            
        if (step_num == 30):
            scp = self.cache.load_timestep(step_num).items()
            pdb.set_trace()
        print('rasterer output files')

#    def get_splot_array(self, scp):
        

    def test_hi(self):
        print('hey, I added an outputter --> rasterer')

