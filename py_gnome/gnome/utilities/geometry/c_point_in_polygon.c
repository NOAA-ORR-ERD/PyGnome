/*
C code for point in polygon

from:
    
This is a C version from:
http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html

it said to be consistant about points on lines on the web page.

i.e. with floating point errors a p oint very near a segment could be
determined to be either inside or outside the ppolygon, but it will always
evaluate the same, so if it is inside one polygone, it will be outside the one
next to it (defined by exactly the same coordiantes, anyway)

Multiple Components and Holes

    The polygon may contain multiple separate components, and/or holes,
    provided that you separate the components and holes with a (0,0) vertex,
    as follows.

        First, include a (0,0) vertex.

        Then include the first component' vertices, repeating its first vertex
        after the last vertex.

        Include another (0,0) vertex.

        Include another component or hole, repeating its first vertex after
        the last vertex.

        Repeat the above two steps for each component and hole.

        Include a final (0,0) vertex.

    For example, let three components' vertices be A1, A2, A3, B1, B2, B3, and
     C1, C2, C3. Let two holes be H1, H2, H3, and I1, I2, I3. Let O be the
     point (0,0). List the vertices thus:

    O, A1, A2, A3, A1, O, B1, B2, B3, B1, O, C1, C2, C3, C1, O, H1, H2, H3,
    H1, O, I1, I2, I3, I1, O.

    Each component or hole's vertices may be listed either clockwise or
    counter-clockwise.

    If there is only one connected component, then it is optional to repeat
    the first vertex at the end. It's also optional to surround the component
    with zero vertices.


Another option here: http://softsurfer.com/Archive/algorithm_0103/algorithm_0103.htm

And one more: 
    http://alienryderflex.com/polygon/
    (this looks, at a glance to be the same)

*/

// Version that takes x and y in one array.    
char c_point_in_poly1(int nvert, double *vertices, double *point)
/*  nvert      Number of vertices in the polygon.
                   Whether to repeat the first vertex at the end is discussed above.
    vertices  Array containing the (x, y) coordinates of the polygon's vertices,
              aranged as a Nx2 array in classic C order
    point  double pointer to x and y-coordinate of the test point. x=point[0], y=point[1]
*/
    {
    int i, j = 0;
    char c = 0; /*really need a bool here...*/
    for (i = 0, j = nvert-1; i < nvert; j = i++) {
        if ( ((vertices[2*i+1]>point[1]) != (vertices[2*j+1]>point[1])) &&
            (point[0] < (vertices[2*j]-vertices[2*i]) * (point[1]-vertices[2*i+1]) / (vertices[2*j+1]-vertices[2*i+1]) + vertices[2*i]) )
            c = !c;
    }
    return c;
}

// Version that takes x and y in separate arrays.    
// int c_point_in_poly1(int nvert, double *vertx, double *verty, double testx, double testy)
//   nvert 	   Number of vertices in the polygon.
//                    Whether to repeat the first vertex at the end is discussed above.
//     vertx, verty  Arrays containing the x- and y-coordinates of the polygon's vertices.
//     testx, testy  X- and y-coordinate of the test point. 

//     {
//     int i, j, c = 0;
//     for (i = 0, j = nvert-1; i < nvert; j = i++) {
//         if ( ((verty[i]>testy) != (verty[j]>testy)) &&
//             (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
//             c = !c;
//     }
//     return c;
// }

