#!/usr/bin/env python

from kiva import agg
from hazpy.file_tools import haz_files
from numpy import array

class agg_drawing:

	def __init__(self, dimension=(1000,1000), mode='rgba32'):
		self.gc = agg.GraphicsContextArray(dimension, mode)

	def draw_polygon(self, polygon, land_color=array((255.,204.,153.))):
		poly_array = polygon[0]
		gc = self.gc
		gc.set_stroke_color(land_color)
		gc.set_fill_color(land_color)
		gc.set_line_width(10)
		initial_point = tuple(poly_array[0])
		gc.move_to(*initial_point)
		for point in poly_array:
			gc.line_to(*tuple(point))
		gc.close_path()
		gc.fill_path()		

	def write_file(self, filename):
		self.gc.save(filename)

if __name__=="__main__":
	drawing = agg_drawing()
	polygons = haz_files.ReadBNA("utilities/LongIslandSoundMap.bna")
	for polygon in polygons:
		drawing.draw_polygon(polygon)
	drawing.write_file("test.bmp")
