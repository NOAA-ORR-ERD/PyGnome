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
from gnome.utilities.time_utils import asdatetime

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
                 diagnostic=None,
                 centers=None,
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

        self.output_dir = output_dir

        # set up projections
        self.epsg_out = Proj(init=epsg_out)
        self.epsg_in = Proj(init='epsg:4326')
        self.centers = centers
        self.diagnostic = diagnostic
        for center in self.centers:
            center['start_time'] = asdatetime(center['start_time'])
            center['end_time'] = asdatetime(center['end_time'])
            center['centers'] = np.array(center['centers'], dtype=[('lon', 'float64'), ('lat', 'float64')])

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
        # model domain outer boundary
        m_d_o_x, m_d_o_y = self.model_domain.exterior.xy
        plt.plot(m_d_o_x, m_d_o_y, color='g', linewidth=1)
        # model domain inner boundary
        for i in range (len(self.model_domain.interiors)):
            x_int, y_int = self.model_domain.interiors[i].xy
            plt.plot(x_int, y_int)
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
    
    def get_unit_vector(self, pt_1, pt_2):
        # each point to be xy pair as list or numpy array
        x_del = pt_2[0] - pt_1[0]
        y_del = pt_2[1] - pt_1[1]
        mag = (x_del**2 + y_del**2)**0.5
        unit_vector = np.asarray([x_del/mag, y_del/mag])
        return unit_vector
    def get_new_external_xy(self, point, origin_vert, act_region_index, vor_polys, first):
        # need perpendicular unit vector that goes between 2 splot points
        # loop to find next adjacent region
        for region in range(0,len(vor_polys.regions)):
            if region != act_region_index:                            
                if (origin_vert in vor_polys.regions[region]) and (-1 in vor_polys.regions[region]):
                    adj_region_ind = region
                    adj_point_ind = np.where(vor_polys.point_region == adj_region_ind)[0][0]
                    break
        cur_point_xy = vor_polys.points[point]
        adj_point_xy = vor_polys.points[adj_point_ind]
        # build unit vector
        origin_xy = vor_polys.vertices[origin_vert]
        origin_point = geometry.Point(origin_xy)
        cur_adj_vector = self.get_unit_vector(cur_point_xy, adj_point_xy)
        perp_vector = np.asarray([-cur_adj_vector[1], cur_adj_vector[0]])
        dist_model_bound = origin_point.distance(self.model_domain.exterior)
        new_xy_for_polyline = np.asarray(origin_xy) + perp_vector * dist_model_bound * 2
        new_xy_for_pl_point = geometry.Point(new_xy_for_polyline)
        # check for direcitonality of unit vector
        test_line = geometry.LineString([origin_xy, new_xy_for_polyline])
        region_close_list = []
        for region in range(0, len(vor_polys.regions)):
            if region != act_region_index:
                if (origin_vert in vor_polys.regions[region]) and (-1 not in vor_polys.regions[region]):
                    region_close_list.append(vor_polys.regions[region])
        for region in range(0,len(region_close_list)):
            region_xys = [vor_polys.vertices[ind] for ind in region_close_list[region]]
            tmp_poly = geometry.Polygon(region_xys)
            if not test_line.touches(tmp_poly):
    #            print 'flipping perp_vector'
                perp_vector = perp_vector * -1
                new_xy_for_polyline = np.asarray(origin_xy) + perp_vector * dist_model_bound * 2
                new_xy_for_pl_point = geometry.Point(new_xy_for_polyline)
                break   
        while self.model_domain.contains(new_xy_for_pl_point):
            dist_model_bound = dist_model_bound * 2
            new_xy_for_polyline = np.asarray(origin_xy) + perp_vector * dist_model_bound * 2
            new_xy_for_pl_point = geometry.Point(new_xy_for_polyline)
        return new_xy_for_polyline
    def get_thick(self, splots_xyms, **kwargs):
        if len(kwargs.keys()) == 0:
            diagnostic = False
        elif 'diagnostic' in kwargs.keys():
            diagnostic = kwargs['diagnostic']
        else:
            diagnostic = False
        splots_cnt = len(splots_xyms)
        splots_xy = splots_xyms[['x','y']].reshape(-1, splots_xyms.size)
        splots_xy_arr = np.array(splots_xy.tolist()[0])

        vor_polys = Voronoi(splots_xy_arr, furthest_site=False)
        extent_window = np.array([[240000, 280000],[880000, 920000]])
        if self.diagnostic == True:
            self.diagnostic_plot(vor_polys, extent=extent_window)        
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
                        new_xy_for_polyline = self.get_new_external_xy(point, origin_vert, act_region_index, vor_polys, True)
                        if first_vert_index + 1 == last_vert_index:
                            poly_line_xys.insert(first_vert_index + 1, new_xy_for_polyline)
                        else:
                            poly_line_xys.insert(first_vert_index, new_xy_for_polyline)
                        # last vertex 
                        origin_vert = region_verts[last_vert_index]
                        new_xy_for_polyline = self.get_new_external_xy(point, origin_vert, act_region_index, vor_polys, False)
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
                        new_xy_for_polyline = self.get_new_external_xy(point, origin_vert, act_region_index, vor_polys, True)
                        if first_vert_index + 1 == last_vert_index:
                            poly_line_xys.insert(first_vert_index + 1, new_xy_for_polyline)
                        else:
                            poly_line_xys.insert(first_vert_index, new_xy_for_polyline)
                    elif self.model_domain.contains(last_point):
                        origin_vert = region_verts[last_vert_index]
                        new_xy_for_polyline = self.get_new_external_xy(point, origin_vert, act_region_index, vor_polys, False)
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
            return splots_thick


    def reproj_points(self, xs, ys):
        sp_x, sp_y = transform(self.epsg_in, self.epsg_out, 
                               xs,
                               ys)
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
        sap_xy = self.reproj_points(spill_poly.points[:, 0], spill_poly.points[:, 1])
        sap_geo_poly = geometry.Polygon(sap_xy)
        inner_geo_poly = []
        edge_geo_poly = []
        for poly in land_polys:
            lp_xy = self.reproj_points(poly.points[:, 0], poly.points[:, 1]) 
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


    def get_centers(self, ts_datetime):
        ret_centers = None
        for timespan in self.centers:
            if ((ts_datetime >= timespan['start_time']) and (ts_datetime <= timespan['end_time'])):
                ret_centers = timespan['centers']
                break
        if isinstance(ret_centers, np.ndarray):
            return ret_centers
        else:
            print('timestep outside of range of centers')
            return np.array([])

    def subspill(self, splots_xyms, centers_xy):
        splots_xy = splots_xyms[['x','y']].reshape(-1, splots_xyms.size)
        splots_xy_arr = np.array(splots_xy.tolist()[0])
        centers_xy_arr = np.array(centers_xy.tolist())
        dist_2_centers = distance.cdist(splots_xy_arr, centers_xy_arr, 'euclidean')
        center_groups = dist_2_centers.argmin(axis=1)
        center_uniq = np.unique(center_groups, return_counts=True)
        splots_xyms_grpd = []
        splots_xyms_grpd_index = []
        for group in range(0,len(center_uniq[0])):
            splots_xyms_grpd.append(np.zeros(center_uniq[1][center_uniq[0][group]], dtype=splots_xyms.dtype))
            splots_xyms_grpd_index.append(0)
        for splot in range(0,len(splots_xyms)):
            act_group = center_groups[splot]
            splots_xyms_grpd[act_group][splots_xyms_grpd_index[act_group]] = splots_xyms[splot]
            splots_xyms_grpd_index[act_group] += 1
#            print 'act_group: ' + str(act_group) + ' - ' + str(splots_xyms_grpd_index) 
        return splots_xyms_grpd
    
    def create_raster_ascii(self, file_name, splots_xyth):
        print 'creating raster with ascii'
        buff_perc = 0.05
        int_perc = 0.01
        min_x = np.min(splots_xyth['x'])
        max_x = np.max(splots_xyth['x'])
        min_y = np.min(splots_xyth['y'])
        max_y = np.max(splots_xyth['y'])
        del_x = max_x - min_x
        del_y = max_y - min_y
        min_x_buff = np.around(min_x - (del_x * buff_perc), decimals=0)
        min_y_buff = np.around( min_y - (del_y * buff_perc), decimals=0)
        max_x_buff = np.around( max_x + (del_x * buff_perc), decimals=0)
        max_y_buff = np.around( max_y + (del_y * buff_perc), decimals=0)
        del_x_buff = max_x_buff - min_x_buff
        del_y_buff = max_y_buff - min_y_buff
        if del_x_buff <= del_y_buff:
            xy_int = np.around(del_x_buff * int_perc, decimals=0)
        else:
            xy_int = np.around(del_y_buff * int_perc, decimals=0)
        
        grid_x, grid_y = np.mgrid[min_x_buff: max_x_buff: xy_int, min_y_buff: max_y_buff: xy_int]
        
        splots_xy = splots_xyth[['x','y']].reshape(-1, splots_xyth.size)
        splots_xy_arr = np.array(splots_xy.tolist()[0])
        grid_thick = griddata(splots_xy_arr, splots_xyth['kg_m2'], (grid_x, grid_y), method='linear')
        for x in range(0, len(grid_thick)):
            for y in range(len(grid_thick[0]) - 1, -1, -1):
                cur_grid_point = geometry.Point(grid_x[x][y], grid_y[x][y])
                if not self.model_domain.contains(cur_grid_point):
                    if not np.isnan(grid_thick[x][y]):
                        grid_thick[x][y] = np.nan
        self.save_2_asc(file_name, grid_thick, min_x_buff, min_y_buff, xy_int)
    
    def save_2_asc(self, file_name, grid_thick, min_x, min_y, csize):
        f = open(os.path.join(self.output_dir, file_name), 'w')
        f.write('NCOLS ' + str(len(grid_thick)) + '\n')
        f.write('NROWS ' + str(len(grid_thick[0])) + '\n')
        f.write('XLLCORNER ' + str(min_x) + '\n')
        f.write('YLLCORNER ' + str(min_y) + '\n')
        f.write('CELLSIZE ' + str(csize) + '\n')
        f.write('NODATA_VALUE -999\n')
        for row in range(len(grid_thick[0]) - 1, -1, -1 ):
            for col in range(0, len(grid_thick)):
                if row != 0:
                    f.write(' ')
                if np.isnan(grid_thick[col][row]):
                    f.write('-999')
                else:
                    f.write(str(np.around(grid_thick[col][row], decimals=5)))
            f.write('\n')
        f.close() 

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
        # pull relevant results for processing
        scp = self.cache.load_timestep(step_num).items()
        if len(scp)  < 3: # place holder to consider forecasting
            spill_cert = scp[0]
            xyms = self.get_xyms(spill_cert)
            ts_centers = self.get_centers(scp[0].current_time_stamp)
            ts_centers_cnt = len(ts_centers)
            if ts_centers_cnt < 2:
                if self.diagnostic == True:
                    xyth, grp_voronoi = self.get_thick(xyms, diagnostic=True)
                else:
                    xyth = self.get_thick(xyms, diagnostic=False)
            else:
                print('developing multiple spill centers')
                cur_centers_xy = np.zeros((ts_centers_cnt),dtype=[('x','float64'), ('y','float64')])
                for center in range(0, ts_centers_cnt):
                    cur_centers_xy[center] = transform(self.epsg_in, self.epsg_out, 
                                           ts_centers[center]['lon'],
                                           ts_centers[center]['lat'])
                xyms_grpd = self.subspill(xyms, cur_centers_xy)
                for group in range(0,len(xyms_grpd)):
                    if self.diagnostic == True:
                        xyth_grpd, grp_voronoi = self.get_thick(xyms_grpd[group], diagnostic=True)
                    else:
                        xyth_grpd = self.get_thick(xyms_grpd[group], diagnostic=False)
                    if group == 0:
                        xyth = xyth_grpd
                        all_voronoi = []
                        all_voronoi.append(grp_voronoi)
                    else:
                        xyth = np.concatenate((xyth, xyth_grpd), axis=0)
                        all_voronoi.append(grp_voronoi)
            file_name = 'ras_' + '{:0>4d}'.format(step_num) + '.asc'
            self.create_raster_ascii(file_name, xyth)    
        print('rasterer output files')

#    def get_splot_array(self, scp):
        

    def test_hi(self):
        print('hey, I added an outputter --> rasterer')

