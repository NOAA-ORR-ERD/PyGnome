#ifndef __BUILDLIST__
#define __BUILDLIST__
//#include "DagTri.h"
//#include "DagStruct.h"
#include "DagTree.h"
					
typedef struct {
			double pLong;
			double pLat;
		} WorldPointD, *WORLDPOINTDP, **WORLDPOINTDH;

typedef struct	Test				// Only used in picking the best node to create the dag tree.
		{	long nodes;				// The test node
			long countLeft;		// 		test node statistics
			long countRight;
			long countBoth;
			long countTotal;
		} Test, tN;


typedef struct Side_List
{
	long			p1;					//starting point of ordered seg 
	long			p2;					//  ending point  "		"
	long			triLeft;				//number of triangle on the left of seg
	long			triRight;			//"		"				"		 right of seg
	long			triLeftP3;      	//index for third point of triangle to left of segment
	long			triRightP3;	  		//  	"			"		"		"	 	"	"	right 	"
	long			topoIndex;			// Index of p1 into full triangle topology list
													// for use with new indexed dag structure.
} Side_List;

Boolean maketriangles(TopologyHdl *topoHdl, LongPointHdl ptsH, long nv, LONGH boundarySegs, long nbounds); 
Boolean maketriangles2(TopologyHdl *topoHdl, LongPointHdl ptsH, long nv, LONGH boundarySegs, long nbounds, LONGH ptrVerdatToNetCDFH, long numCols_ext); 
//////////////////////////////////////////////////////////////////////////
//																								//
// CJ Beegle-Krause																		//
//																								//
// 						Directional Acyclic Graph (DAG) Structure				//
//																								//
//   Given:																					//
//				sideList,							From the Build_Side function	//
//				number of Sides					"			"			"				//
//				number of boundary sides   	"			"			"				//
//				pointList,							Lat & long for all the pionts.//
//	  Create:																				//
//				DAGtree structure															//
//					ordered segment list by points									//
//					branch/triangle on left list										//
//					branch/triangle on right list										//
//				the number of elements in the dag list								//
//																								//
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
//																								//
//  Increment the DagTree length in memory as we write dag					//
//																								//
//////////////////////////////////////////////////////////////////////////
void IncrementGDagTree(void);

//////////////////////////////////////////////////////////////////////////
// 																							//
// Here we pick a node to split another branch of the DAG tree				//
//		Select the new node randomly among the list of sides.					//
//		(Picking a non boundary side first will make the DAG tree flatter)//
// 																							//
//////////////////////////////////////////////////////////////////////////
long newNodexx(Side_List **sides_node);

//////////////////////////////////////////////////////////////////////////
// 																		//
// Here we find the original node number from the side list				//
// 																		//
//////////////////////////////////////////////////////////////////////////
long oldNodeNum(long pt1, long pt2);

//////////////////////////////////////////////////////////////////////////
// 																							//
// Here we split the segments into the DAG tree									//
// 																							//
//////////////////////////////////////////////////////////////////////////
void split(Side_List **sideListH, long nodeNum);

/////////////////////////////////////////////////////////////////////////////////////////
// 																												//
// Right_or_Left decides if a segement is to the right or left									//
//		of a reference segment, or is neither.															//
//																													//
//	Given: ref_seg_pt1, ref_seg_pt2, test_seg_pt1, test_seg_pt2									//
//	Return:																										//
//			location																								//
//				+1 = Right means the entire segment is to the right of the reference			//
//				0  = Neither.  The segment is either along a line created by extending		//
//							the reference segment infinitly OR the segment intersects a 		//
//							line created by extending the reference segment infinitely			//
//				-1 = Left means the entire segment is to the left of the reference			//
//			NOTE: A Segment that has only one point on the line created by extending		//
//							the reference segment infinitely will be determined as right		//
//							or left, not both																	//
// 																												//
/////////////////////////////////////////////////////////////////////////////////////////
int	Right_or_Left(long ref_p1,long ref_p2, long test_seg_p1, long test_seg_p2);

#ifdef __cplusplus
extern "C" {
#endif
int SideListCompare(void const *x1, void const *x2);
int NodeTestCompare(void const *x1, void const *x2);
#ifdef __cplusplus
}
#endif

Side_List**  BuildSideList(TopologyHdl topoHdl, char *errStr);
//Side_List** xx();
///////////////////////////////////////////////////////////////////////////////////
// CJ Beegle-Krause 																					//
//																											//
//									Build_Side															//
//																											//
// This subroutine will take a given triangle topology and return						//
// 	a list of sements and the triagles located to the right and left				//
//		of the segment for use in the DAG (Directional Acyclic Graph) structure.	//
//																											//
//	The topology is defined by a set of x,y points that are the locations 			//
//		of the triangle vertices.																	//
//																											//
//								n(i) = pointer for vertex(i) located at x(i), y(i)			//
//																											//
//	Triangles within the topology are described by the three vertices and 			//
//		the three neighboring triagles.  The order relates triangle and the 			//
//		opposite vertex, e.g. Triangle1 is located opposite to Vertex1 and 			//
//		adjacent to the side created by vertex2 and vertex3.								//
//																											//
// topology(i) = vertex1(i), vertex2(i), vertex3(i), adjTri1(i), adjTri2(i),     //							adjTri3(i)																	//
//																											//
// The product is a complete and counted list of tri sides within the topology. 	//
//		The vertices used to create each side will be referenced.						//
//		The triangle to the left and the triangle to the right of each side will be//
//		referenced.																						//
//																											//
//				 side(i) =	point1(i), point2(i), triLeft(i), triRight(i)				//
//																											//
//				Ordering is important because of the right and left references.		//
//				The segment "points" from point1 to point2.									//
//																											//
///////////////////////////////////////////////////////////////////////////////////

#endif
