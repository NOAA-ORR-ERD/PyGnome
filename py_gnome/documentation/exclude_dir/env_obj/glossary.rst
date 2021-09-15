Glossary
================

Cells
  This is generally synonymous with faces, although it is not a term used in the code or API
Faces
  A collection of group of integer indices that define the polygons of the grid. Generally used for unstructured triangular grids to describe the triangulation of the nodes. 
Grid
  A conceptual 2D surface that is represented by at minimum a collection of nodes and optionally collections of faces, edges, and/or neighbors. 
Gridded data
  Up to 4-dimensional data (time, depth, lon, lat) that shares the shape of the nodes or faces in the last dimension(s). For example, if the nodes have a shape of 100x100, and the data’s final two dimensions are also 100x100, the data can be referred to as being ‘on the nodes’ If the data has a shape of 99x99, it can be referred to as being ‘on the faces/centers’
Grid topology
  Data that describes associations between parts of a grid and data structures. At minimum it must define what the nodes of a grid are if the grid is structured, and must define the nodes and faces of a grid if it is unstructured. 
Nodes
  A collection of 2D points which define the corners of the cells in a grid. Generally consists of two collections of variables, often referred to as ‘node_lon’ and ‘node_lat’
