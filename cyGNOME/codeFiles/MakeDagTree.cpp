// this file could be included in GNOME to deal with DagTree if not included in Ptcur files
// just uncomment out the makeDagTree call in TextRead
#include "Cross.h"
#include "DagTreeIO.h"
#include "GridVel.h"
#include "my_build_list.h"


DAGTreeStruct gDagTree;	
WORLDPOINTDH gCoord;
Side_List **gSidesList;
long gAllocatedDagLength = 0;
char gErrStr[256];
#define NUMTOALLOCATE  500
#define NUMTESTS	300
#define SEED	5

///////////////////////////////////////////////////////////////////////////////////
//																				 //
//	Given a triangle number and two vertices, return the point number of the 	 //
//			third vertex  (Note: go around triangle in direction of a->b, 		 //
//								this is not always counterclockwise ....		 //
//																				 //
///////////////////////////////////////////////////////////////////////////////////
long FindThirdVertex(long triNum, TopologyHdl THdl, long pointA, long pointB)
{
	long pointC;
	
	if (triNum == -1) {pointC = -1;}
	else if ((*THdl)[triNum].vertex1 == pointA && (*THdl)[triNum].vertex2 == pointB) 
		{pointC = (*THdl)[triNum].vertex3;}
	else if ((*THdl)[triNum].vertex2 == pointA && (*THdl)[triNum].vertex3 == pointB) 
		{pointC = (*THdl)[triNum].vertex1;}
	else if ((*THdl)[triNum].vertex3 == pointA && (*THdl)[triNum].vertex1 == pointB) 
		{pointC = (*THdl)[triNum].vertex2;}
	else if ((*THdl)[triNum].vertex1 == pointB && (*THdl)[triNum].vertex2 == pointA) 
		{pointC = (*THdl)[triNum].vertex3;}
	else if ((*THdl)[triNum].vertex2 == pointB && (*THdl)[triNum].vertex3 == pointA) 
		{pointC = (*THdl)[triNum].vertex1;}
	else if ((*THdl)[triNum].vertex3 == pointB && (*THdl)[triNum].vertex1 == pointA) 
		{pointC = (*THdl)[triNum].vertex2;}
	return (pointC);
}
///////////////////////////////////////////////////////////////////////////////////
//																				 //
//			Check to see if the test triangle numbers are lower or higher        //
//					to work up the list for the sort comparison					 //
//																				 //
///////////////////////////////////////////////////////////////////////////////////
int SideListCompare(void const *x1, void const *x2)
{
	Side_List *p1,*p2;
	p1 = (Side_List*)x1;
	p2 = (Side_List*)x2;
	
	if (p1->triRight < p2->triRight)
		return -1;  
	else if (p1->triRight > p2->triRight)
		return 1;
	else return 0;
		
}
///////////////////////////////////////////////////////////////////////////////////


Side_List**  BuildSideList(TopologyHdl topoHdl, char *errStr) 
{
	
	Side_List **sideListHdl;		// What we want to get.
	//Side_List tempSide;				// Temporary storage for a side during sorting.
	long sideCount;					// The total number of sides
	long numTriangle = (_GetHandleSize((Handle)topoHdl))/(sizeof(Topology));	// Total number of triangles in the topology
	long maxSides = 2*numTriangle + 1 + 1;
								// The maximum number of sides for a given number of triangles.
								// The first +1 is for the single triangle case, the second +1 is
								// for the array index (I do not define the zero segment)
	//long numBoundarySeg;				// The total number of boundary segments in the domain.					
	long i;								// index & counter

	sideListHdl = (Side_List**)_NewHandleClear((maxSides+1)*sizeof(Side_List));
	if(!sideListHdl) 
	{
		strcpy(errStr,"Out of memory in BuildSideList");
		return (nil);
	}

	sideCount = 0;	
								//  The triangle (triangle[0]) is outside.  This case
								//		does not need to go through the loop.
	for (i=0;i<numTriangle;i++)					// increment after using
	{	 
		if ((*topoHdl)[i].adjTri3 < i)
		{
			// safety check 
			if(sideCount > maxSides)
			{
				strcpy(errStr,"MaxSides exceeded in BuildSideList");
				if(sideListHdl) DisposeHandle((Handle)sideListHdl); 
				return (nil);
			}
			(*sideListHdl)[sideCount].p1 = (*topoHdl)[i].vertex1;  // Move around always in the same direction
			 								// as the numbering scheme: 1->2->3->1->2 ...
			(*sideListHdl)[sideCount].p2 = (*topoHdl)[i].vertex2;
			(*sideListHdl)[sideCount].triLeft = i;	//By definition in our geometry, the triangle to
												// the left of our segment will always be the 
												// the triangle that we are in.
			(*sideListHdl)[sideCount].triRight = (*topoHdl)[i].adjTri3; // By our geometry, the triangle adj on
												// the right will be the traiangle across from the
												// vertex not on the segment.//
			(*sideListHdl)[sideCount].triLeftP3 = (*topoHdl)[i].vertex3;
			(*sideListHdl)[sideCount].triRightP3 = FindThirdVertex((*topoHdl)[i].adjTri3,topoHdl,(*topoHdl)[i].vertex1,(*topoHdl)[i].vertex2);
			(*sideListHdl)[sideCount].topoIndex = (i)*6;
												
			sideCount++;								//Increment the counter for a new side.
		}
		if ((*topoHdl)[i].adjTri1 < i)
		{
			// safety check 
			if(sideCount > maxSides){strcpy(errStr,"MaxSides exceeded in BuildSideList");
						if(sideListHdl) DisposeHandle((Handle)sideListHdl); return (nil);}
			(*sideListHdl)[sideCount].p1 = (*topoHdl)[i].vertex2;
			(*sideListHdl)[sideCount].p2 = (*topoHdl)[i].vertex3;
			(*sideListHdl)[sideCount].triLeft = i;
			(*sideListHdl)[sideCount].triRight = (*topoHdl)[i].adjTri1;
			(*sideListHdl)[sideCount].triLeftP3 = (*topoHdl)[i].vertex1;
			(*sideListHdl)[sideCount].triRightP3 = FindThirdVertex((*topoHdl)[i].adjTri1,topoHdl,(*topoHdl)[i].vertex2,(*topoHdl)[i].vertex3);
			(*sideListHdl)[sideCount].topoIndex = (i)*6+1;
			sideCount++;
		}
		if ((*topoHdl)[i].adjTri2 < i)
		{
			// safety check 
			if(sideCount > maxSides)
			{
				strcpy(errStr,"MaxSides exceeded in BuildSideList");
				if(sideListHdl) DisposeHandle((Handle)sideListHdl); 
				return (nil);
			}
			(*sideListHdl)[sideCount].p1 = (*topoHdl)[i].vertex3;
			(*sideListHdl)[sideCount].p2 = (*topoHdl)[i].vertex1;
			(*sideListHdl)[sideCount].triLeft = i;
			(*sideListHdl)[sideCount].triRight = (*topoHdl)[i].adjTri2;
			(*sideListHdl)[sideCount].triLeftP3 = (*topoHdl)[i].vertex2;
			(*sideListHdl)[sideCount].triRightP3 = FindThirdVertex((*topoHdl)[i].adjTri2,topoHdl,(*topoHdl)[i].vertex3,(*topoHdl)[i].vertex1);
			(*sideListHdl)[sideCount].topoIndex = (i)*6+2;
			sideCount++;
		}
	}
// Count the number of boundary segments.
//	numBoundarySeg = 0;
//	for (i=0;i<sideCount;i++) 
//	{
//		if ((*sideListHdl)[i].triRight == -1) (numBoundarySeg++);
//	}
	
	// Sort The List so that boundary segments are first.
	_HLock((Handle)sideListHdl);
	qsort((*sideListHdl),sideCount,sizeof(Side_List),SideListCompare);
	_HUnlock((Handle)sideListHdl);

	
	_SetHandleSize((Handle)sideListHdl,(sideCount)*sizeof(Side_List));
	return(sideListHdl);
}

OSErr LatLongTransform(LongPointHdl vertices)// may need to read in the doubles then convert later
{
	long i,npoints = _GetHandleSize((Handle)vertices)/sizeof(LongPoint);	
	double deg2rad = 3.14159/180.;
	double R = 8000,dLat,dLong,Height,Width,scalefactor,tx,ty;
	double xmin=1e6,xmax=-1e6,ymin=1e6,ymax=-1e6;
	WORLDPOINTDH coord=0;
	
	if(!(coord = (WORLDPOINTDH) _NewHandleClear(sizeof(WorldPointD) * npoints)))return -1;
	for(i=0;i<npoints;i++)
	{
		(*coord)[i].pLat = (*vertices)[i].v / 1000000.;
		(*coord)[i].pLong = (*vertices)[i].h / 1000000.;
	}
	//GetVertexRange(vertices, &xmin,&xmax,&ymin,&ymax);	// defined in Iocats.c
	for(i=0;i<npoints;i++)
	{
		if((*coord)[i].pLat < ymin) ymin = (*coord)[i].pLat;
		if((*coord)[i].pLat > ymax) ymax = (*coord)[i].pLat;
		if((*coord)[i].pLong < xmin) xmin = (*coord)[i].pLong;
		if((*coord)[i].pLong > xmax) xmax = (*coord)[i].pLong;
	}
	dLat = ymax - ymin;
	dLong = xmax - xmin;
	Height = dLat * deg2rad * R;
	Width = dLong * deg2rad * R * cos((ymax + ymin)*deg2rad/2.0);
	scalefactor = (Height > Width) ? 1/Height : 1/Width;
	tx = scalefactor * Width/dLong;
	ty = scalefactor * Height/dLat;
	
	for(i= 0; i < npoints ; i++)
	{
		(*coord)[i].pLong = tx * ((*coord)[i].pLong - xmin);
		 (*coord)[i].pLat = ty * ((*coord)[i].pLat - ymin);
	}
	gCoord = coord;
	return 0;
}

void ConvertSegNoToTopIndex(Side_List **sl, DAGTreeStruct dagTree)
{
	long nnodes = dagTree.numBranches,i;
	DAG** dagHdl;	
	
	dagHdl = dagTree.treeHdl;
	
	for(i=0;i<nnodes;i++)
	{
		(*dagHdl)[i].topoIndex = (*sl)[(*dagHdl)[i].topoIndex].topoIndex;
	}
}

DAGTreeStruct  MakeDagTree(TopologyHdl topoHdl, LongPoint **pointList, char *errStr)
{
	Side_List **sidesList = 0;
	DAGTreeStruct  dagTree;
	long numSidesInList;		
	long nodeNum;					// The index in the current side list for the node (segment)
									//			to split across.
	strcpy(errStr,"");
	dagTree.treeHdl = 0;
	dagTree.numBranches = 0;
	sidesList=BuildSideList(topoHdl, errStr);

	//input checking
	if(sidesList == nil) 
	{ 
		strcpy(errStr,"sideList is nil in MakeDagTree");
		goto done;
	}
	
	LatLongTransform(pointList);
	gSidesList = sidesList;

	numSidesInList = (_GetHandleSize((Handle)sidesList))/(sizeof(Side_List));
	if (numSidesInList < 3)
	{
		sprintf(errStr,"Triangles have 3 sides; the data has only %ld sides.", numSidesInList);
		goto done;
	}

	// make a better guess of size here
	gErrStr[0] = 0;	
	gAllocatedDagLength = 2*numSidesInList;

	gDagTree.numBranches=0;	
	gDagTree.treeHdl = (DAG**)_NewHandleClear(gAllocatedDagLength*sizeof(DAG));
	if(!gDagTree.treeHdl)  
	{ 
		strcpy(errStr,"Out of memory in MakeDagTree");
		goto done;
	}
	
	srand(SEED);
	//srand(SEED+1);
	
	nodeNum = newNodexx(sidesList);
	split(sidesList, nodeNum);
	//	Nothing should come back until DAG is completed.

	//gDagTree.numBranches--;  	// Had one too many increments in the split recursion.
	
	if(gErrStr[0])
	{
		strcpy(errStr,gErrStr);
		gErrStr[0] = 0;
		if(gDagTree.treeHdl) DisposeHandle((Handle) gDagTree.treeHdl);
		gDagTree.treeHdl = nil;
		gDagTree.numBranches = 0;
		goto done;
	}
	
	ConvertSegNoToTopIndex(sidesList,gDagTree);

	dagTree = gDagTree;
	gDagTree.treeHdl = nil;// the handle is now their responsibility
	gDagTree.numBranches = 0;
	
done:
	if(sidesList)
	{
		{DisposeHandle((Handle)sidesList);sidesList=0;}
	}
	return dagTree;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//																		//
//  Use this function to compare the nodes in the list of possible		//
//		nodes.  We are looking for a minimum of the function in 		//
//		NewNodexx														//
//																		//
//////////////////////////////////////////////////////////////////////////

int NodeTestCompare(void const *x1, void const *x2)
{
	Test *p1,*p2;	
	p1 = (Test*)x1;
	p2 = (Test*)x2;
	
	if (p1->countTotal < p2->countTotal) 
		return -1;  // first less than second
	else if (p1->countTotal > p2->countTotal)
		return 1;
	else return 0;// same,equivalent
	
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//																		//
//  Increment the DagTree length in memory as we write dag				//
//																		//
//////////////////////////////////////////////////////////////////////////
void IncrementGDagTree(void)
{
	// increment the counter and allocate more memory if needed
	OSErr err = noErr;
	gDagTree.numBranches ++;
	if(gDagTree.numBranches  >= gAllocatedDagLength)
	{									// then allocate more memory
		long newNumber = gAllocatedDagLength + NUMTOALLOCATE;
		_SetHandleSize((Handle)gDagTree.treeHdl,newNumber* sizeof(DAG));
		
		err = _MemError();	
		if (err)
		{
			strcpy(gErrStr,"Not enough memory to expand the DAG tree structure");
		}
		else
		{
			gAllocatedDagLength = newNumber;
		}
	}
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// 																		//
// Here we pick a node to split another branch of the DAG tree			//
//		Select the new node randomly among the list of sides.			//
//																		//
//////////////////////////////////////////////////////////////////////////

long newNodexx(Side_List **sides_node)
// The side to serve as a new node for the DAG tree.
{
	//long new_node;					// The new branch point chosen randomly
	long numSides_node = (_GetHandleSize((Handle)sides_node))/(sizeof(Side_List));
	long numBoundSides_node = 0;
	Test tN[NUMTESTS];
	long numTests = NUMTESTS;
	long location;					// result of right or left routine
	long testNodeP1, testNodeP2;	// points in test node segment
	long i,j;						// Counters
	//char str[512];
	//long	start, end;		
	
	for (i=0; i<numSides_node; i++)
	{
		if ((*sides_node)[i].triRight == -1) numBoundSides_node++;
	}
	
	// Old Way
	//if (numSides_node == numBoundSides_node){
	//	new_node = (rand() % numSides_node)+1;
	//	return (new_node);
	//	}
	//new_node = (rand() % (numSides_node - numBoundSides_node)+1);
	//	return (numBoundSides_node+new_node);

	//start = TickCount();
	
	if (numSides_node < numTests) numTests = numSides_node;  // Don't do more tests than the total
															// number of sides to test.
	for (i=0;i<numTests;i++)
	{
		if (numSides_node == numBoundSides_node){
			tN[i].nodes = (rand() % numSides_node);
		}												//Want to use boundary sides last...
		else (tN[i].nodes = numBoundSides_node + (rand() % (numSides_node - numBoundSides_node)));
		testNodeP1 = (*sides_node)[tN[i].nodes].p1;
		testNodeP2 = (*sides_node)[tN[i].nodes].p2;
		tN[i].countBoth = tN[i].countRight = tN[i].countLeft = 0;
		for	(j=0;j<numSides_node;j++)				//Test the remaining sides against the
		{										//  current node to see how it divides the domain
			if (tN[i].nodes == j) continue;
			location = Right_or_Left(testNodeP1,testNodeP2,(*sides_node)[j].p1,(*sides_node)[j].p2);
			switch(location)
			{
				case -1:// left
					tN[i].countLeft++;
					break;
				case 1: // right
					tN[i].countRight++;
					break;
				case 0: // both
					tN[i].countBoth++;
					break;
			}

		}
		tN[i].countTotal = abs(tN[i].countLeft-tN[i].countRight)+2*(tN[i].countBoth); //Selection
	}										// algorithm for best node - minimize this function
	qsort(tN,numTests,sizeof(Test),NodeTestCompare);
	//end = TickCount();
	//sprintf(str,"%f seconds for %f tries to optimize node selection", (end-start)/60.0, (float)numTests);
	//MESSAGE(str);
	return(tN[0].nodes);
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// 																		//
// Here we find the original node number from the side list				//
// 																		//
//////////////////////////////////////////////////////////////////////////
long oldNodeNum(long pt1, long pt2)
{
	long segNum = -999;	// The number that we are looking for
	long j=0; 			// An index
	
	while (segNum == -999)
	{
		if ((*gSidesList)[j].p1 == pt1 && (*gSidesList)[j].p2 == pt2) segNum = j;
		else j++;
	}
	return (segNum);
}
//////////////////////////////////////////////////////////////////////////
// 																		//
// Here we split the segments into the DAG tree							//
// 																		//
//////////////////////////////////////////////////////////////////////////

void split(Side_List **sideListH, long nodeNum)
{
	// **sideListH;				 Listing of the sides in the domain for splitting DAG tree
	// nodeNum;						 The index for the segment from sideList to use for pivot/split in DAG tree
	long numSidesL =0;							// the number of sides that are split off to the left
	long numSidesR =0;							// the number of sides that are split off to the right
	Side_List **sidesLeftH = nil;				// pointer to area of heap where sides are left
	Side_List **sidesRightH = nil;				// pointer to area of hear where sides are left
	long numSideInList = _GetHandleSize((Handle)sideListH)/(sizeof(Side_List));	// The number of sides for splitting
	long location,i;
	long nodeP1,nodeP2;
	Side_List sideListForNode;
	long bCounterL =0, bCounterR =0;					// counting the number of boundary seg in a side list
	long branchNum;
	static long nsx = 0;

	nsx++;
	if(nsx == 964)
	{
		nsx = 0;
	}

	if(gErrStr[0] ) goto done;
	if(sideListH == nil) 
	{
		strcpy(gErrStr,"Nil handle passed to split");
		goto done;
	}

	branchNum = gDagTree.numBranches;
	IncrementGDagTree();
				
	(*gDagTree.treeHdl)[branchNum].topoIndex = oldNodeNum(((*sideListH)[nodeNum]).p1,((*sideListH)[nodeNum]).p2);

	sidesLeftH = (Side_List **)_NewHandleClear((numSideInList)*sizeof(Side_List));
	if(!sidesLeftH){strcpy(gErrStr,"Out of memory in split");goto done;}

	sidesRightH = (Side_List **)_NewHandleClear((numSideInList)*sizeof(Side_List));
	if(!sidesRightH) {strcpy(gErrStr,"Out of memory in split");goto done;}

	// find the node's points for determining relative right and left
	nodeP1 = (*sideListH)[nodeNum].p1;
	nodeP2 = (*sideListH)[nodeNum].p2;

	// Do the actual splitting
	for (i=0;i<numSideInList; i++) {
		if(i == nodeNum) continue; // skip the case that nodeNum = i
		
		// find the points associated with the test side
		location = Right_or_Left(nodeP1,nodeP2,(*sideListH)[i].p1,(*sideListH)[i].p2);

		// Distribute the sides to the two R/L areas of the heap
		switch(location)
		{
			case -1:// left
				(*sidesLeftH)[numSidesL++] = (*sideListH)[i];// copy the structure
				break;
			case 1: // right
				(*sidesRightH)[numSidesR++] = (*sideListH)[i];// copy the structure
				break;
			case 0: // both
				(*sidesLeftH)[numSidesL++] = (*sideListH)[i];// copy the structure
				(*sidesRightH)[numSidesR++] = (*sideListH)[i];// copy the structure
				break;
		}
	}
	
	
// trim sidesL and sidesR to numSidesL and numSidesR, respectively
	_SetHandleSize((Handle)sidesLeftH,(numSidesL)*sizeof(Side_List));
	_SetHandleSize((Handle)sidesRightH,(numSidesR)*sizeof(Side_List));
	
	sideListForNode = (*sideListH)[nodeNum];
	
	// Left Hand Side
	if (numSidesL ==0) {
		(*gDagTree.treeHdl)[branchNum].branchLeft = -8;
		// Fake a split
		if (sideListForNode.triLeft == -1) 
		{
			strcpy(gErrStr,"Boundary not set up correctly - outside is on LHS");
			goto done;
		}

	}
	else {
	// fTreeH in GNOME (a DAGHdl), member of fDagTree (part of GridVel, owned by the mover)
		(*gDagTree.treeHdl)[branchNum].branchLeft = gDagTree.numBranches;
		if(gErrStr[0]) goto done;
		// set call for correct area for sides
		nodeNum = newNodexx(sidesLeftH);
		MySpinCursor(); // JLM 8/4/99
		split(sidesLeftH,nodeNum); 			// recursively for LHS
		//sidesLeftH = nil; // split disposed of this hdl
	}


	// Right Hand Side
	if (numSidesR == 0) 
	{
		// Fake a split
		(*gDagTree.treeHdl)[branchNum].branchRight = -8;  // place holder for steps in dag to all tri
	}
	else 
	{
		(*gDagTree.treeHdl)[branchNum].branchRight = gDagTree.numBranches;
		if(gErrStr[0]) goto done;
		// set call for correct area for sides
		nodeNum = newNodexx(sidesRightH);
		split(sidesRightH,nodeNum); 			// recursively for RHS
		//sidesRightH = nil; // split disposed of this hdl
	}
	
done:

	if(sidesRightH) DisposeHandle((Handle) sidesRightH);
	sidesRightH = nil;
	if(sidesLeftH) DisposeHandle((Handle) sidesLeftH);
	sidesLeftH = nil;
	
	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// 																								//
// Right_or_Left decides if a segment is to the right or left									//
//		of a reference segment, or is neither.													//
//																								//
//	Given: ref_seg_pt1, ref_seg_pt2, test_seg_pt1, test_seg_pt2									//
//	Return:																						//
//			location																			//
//				+1 = Right means the entire segment is to the right of the reference			//
//				0  = Neither.  The segment is either along a line created by extending			//
//							the reference segment infinitely OR the segment intersects a 		//
//							line created by extending the reference segment infinitely			//
//				-1 = Left means the entire segment is to the left of the reference				//
//			NOTE: A Segment that has only one point on the line created by extending			//
//							the reference segment infinitely will be determined as right		//
//							or left, not both													//
// 																								//
//////////////////////////////////////////////////////////////////////////////////////////////////
int	Right_or_Left(long ref_p1,long ref_p2, long test_seg_p1, long test_seg_p2)
{
	//long ref_p1_h, ref_p1_v, ref_p2_h, ref_p2_v;	
	//long test_p1_h, test_p1_v, test_p2_h, test_p2_v;	
													
	double ref_h, ref_v;								
	double test1_h, test1_v;							
	double test2_h, test2_v;							
	
	double cp_1;						
	double cp_2;						
	int location;


	ref_h = ((*gCoord)[ref_p2].pLong - (*gCoord)[ref_p1].pLong);	
	ref_v = ((*gCoord)[ref_p2].pLat - (*gCoord)[ref_p1].pLat);
	test1_h = ((*gCoord)[test_seg_p1].pLong - (*gCoord)[ref_p1].pLong);
	test1_v = ((*gCoord)[test_seg_p1].pLat - (*gCoord)[ref_p1].pLat);
	test2_h = ((*gCoord)[test_seg_p2].pLong - (*gCoord)[ref_p1].pLong);
	test2_v = ((*gCoord)[test_seg_p2].pLat - (*gCoord)[ref_p1].pLat);
	
	cp_1 = test1_h* ref_v - test1_v * ref_h;
	cp_2 = test2_h* ref_v - test2_v * ref_h;
	
	//Test for segments that share one point
	if (ref_p1 == test_seg_p1)
		cp_1 = 0.;
	else if (ref_p1 == test_seg_p2)
		cp_2 = 0.;
		
	if (ref_p2 == test_seg_p1)
		cp_1 = 0.;
	else if (ref_p2 == test_seg_p2)
		cp_2 = 0.;

	// decide right, left or neither
	if (cp_1 > 0.)
		{
		if (cp_2 >= 0.) location=1;
		else if (cp_2 < 0.) location=0;
		}
	else if (cp_1 == 0.)
		{
		if (cp_2 > 0.) location=1;
		else if (cp_2 == 0.)  location = 0.;
		// This could mean that the segments are the same.....
		else if (cp_2 < 0.) location=-1;
		}	
	else if (cp_1 < 0.)
		{
		if (cp_2 > 0.) location = 0;
		else if (cp_2 <= 0.) location=-1;
		}

	return (location);
}
////////////////////////////////////////////////////////////////////////
