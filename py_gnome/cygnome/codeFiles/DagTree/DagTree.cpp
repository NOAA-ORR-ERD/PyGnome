
#include "Cross.h"
#include "DagTree/DagTree.h"
#include <iostream>
using namespace std;

long WhatTriIsPtIn(DAGHdl treeH,TopologyHdl topH, LongPointHdl ptsH,LongPoint pt);
int	Right_or_Left_of_Segment(LongPointHdl ptsH,long ref_p1,long ref_p2, LongPoint test_p1);
long FindTriThirdPoint(long **longH,long p1, long p2, long index);

void TDagTree::GetVelocity(long ntri,VelocityRec *r)
{
	if(fVelH) {
		r->u = (*fVelH)[ntri].u;
		r->v = (*fVelH)[ntri].v;
	}
	else
	{
		r->u = 0;
		r->v = 0;
	}
};

void TDagTree::GetVelocity(LongPoint lp,VelocityRec *r)
{
	long ntri = WhatTriAmIIn(lp);
	if(ntri > -1 && fVelH)
	{
		r->u = (*fVelH)[ntri].u;
		r->v = (*fVelH)[ntri].v;
	}
	else 
	{
		r->u = r->v = 0.0;
	}
};

TDagTree::TDagTree(LongPoint **ptsHdl, Topology **topHdl, DAG** tree,VelocityFH velocityH,long nBranches)
{
	fPtsH = ptsHdl;
	fTopH = topHdl;
	fTreeH = tree; 

	fVelH = velocityH;

	fNumBranches = nBranches;
}

void TDagTree::Dispose ()
{
	if (fPtsH)
	{
		DisposeHandle((Handle)fPtsH);
		fPtsH = nil;
	}
	if (fTopH)
	{
		DisposeHandle((Handle)fTopH);
		fTopH = nil;
	}
	if (fTreeH)
	{
		DisposeHandle((Handle)fTreeH);
		fTreeH = nil;
	}
	if (fVelH)
	{
		DisposeHandle((Handle)fVelH);
		fVelH = nil;
	}
}

/////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////

// Find the third point of the triangle to the right of the segment

//////////////////////////////////////////////////////////////////////

long TDagTree::FindThirdPoint(long p1, long p2, long index)

{
	return FindTriThirdPoint((long**)(fTopH),p1,p2,index);
}


/////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Navigate the DAG tree.  Given a point on the surface of the earth
// locate the triangle that the point falls within or identify if
// the point is outside of the boundary (the infinite triangle).
// RETURNS a negative number if there is an error
//////////////////////////////////////////////////////////////////////
long TDagTree::WhatTriAmIIn(LongPoint pt)
{
	return WhatTriIsPtIn(fTreeH,fTopH,fPtsH,pt);
}

/////////////////////////////////////////////////////////////////////////////////////////
// 																												//
// Right_or_Left decides if a test point is to the right or left								//
//		of a reference segment, or is neither.															//
//																													//
//	Given: ref_seg_pt1, ref_seg_pt2, testp1															//
//	Return:																										//
//			location																								//
//				+1 = Right means the test point is to the right of the reference				//
//				0  = Neither.  The test pt is along a line created by extending				//
//							the reference segment infinitly 												//
//				-1 = Left means the test point is to the left of the reference					//
// 																												//
/////////////////////////////////////////////////////////////////////////////////////////

int	TDagTree::Right_or_Left_Point(long ref_p1,long ref_p2, LongPoint test_p1)
{
	return Right_or_Left_of_Segment(fPtsH,ref_p1,ref_p2,test_p1);
}

////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////

// Find the third point of the triangle to the right of the segment

//////////////////////////////////////////////////////////////////////

long FindTriThirdPoint(long **longH,long p1, long p2, long index)

{
	if ((p1 == (*longH)[index]) || (p2 == (*longH)[index+1]))
		
		return (*longH)[index+2];
	
	else if ((p1 == (*longH)[index+1]) || (p2 == (*longH)[index+2]))
		
		return (*longH)[index];
	
	else if ((p1 == (*longH)[index+2]) || (p2 == (*longH)[index]))
		
		return (*longH)[index+1];
	
	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////
// 																												//
// Right_or_Left decides if a test point is to the right or left								//
//		of a reference segment, or is neither.															//
//																													//
//	Given: ref_seg_pt1, ref_seg_pt2, testp1															//
//	Return:																										//
//			location																								//
//				+1 = Right means the test point is to the right of the reference				//
//				0  = Neither.  The test pt is along a line created by extending				//
//							the reference segment infinitely 												//
//				-1 = Left means the test point is to the left of the reference					//
// 																												//
/////////////////////////////////////////////////////////////////////////////////////////
int	Right_or_Left_of_Segment(LongPointHdl ptsH,long ref_p1,long ref_p2, LongPoint test_p1)
{
	long ref_p1_h, ref_p1_v, ref_p2_h, ref_p2_v;	// lat (h) and long (v) for ref points
	long test_p1_h, test_p1_v;	
	// lat (h) and long (v) for test point
	double ref_h, ref_v;								// reference vector components
	double test1_h, test1_v;							// first test vector components (test p1 - ref p1)
	//double pi = 3.1415926;
	//double deg2Rad = (2.*pi/360.);						// convert deg to rad
	
	double cp_1;						// result of (ref1,test1) X (ref1,ref2)
	short location;
	
	// Make sure this code matches the code that generated the triangles !!!
	// (Right now that other code is in CATS)
	
	// Find the lat and lon associated with each point
	ref_p1_h = (*ptsH)[ref_p1].h;
	ref_p1_v = (*ptsH)[ref_p1].v;
	ref_p2_h = (*ptsH)[ref_p2].h;
	ref_p2_v = (*ptsH)[ref_p2].v;
	test_p1_h = test_p1.h;
	test_p1_v = test_p1.v;
	
	// Create the vectors by subtracting (p2 - p1)
	// Change the integers back into floating points by dividing by 1000000.
	
	ref_h = (ref_p2_h - ref_p1_h)/1000000.;
	ref_v = (ref_p2_v - ref_p1_v)/1000000.;
	test1_h = (test_p1_h - ref_p1_h)/1000000.;
	test1_v = (test_p1_v - ref_p1_v)/1000000.;
	
	// create  cross product
	/////////////////////////////////////////////////
	//cp_1 = (test1_h * ref_v * sin(ref_p1_v*deg2Rad/1000000.));
	//cp_1 = cp_1 - (test1_v * ref_h * sin(test_p1_v*deg2Rad/1000000.) );
	//cp_1 = cp_1 / sin( .5 * (ref_p1_v + test_p1_v)*deg2Rad/1000000. );
	// JLM , code goes here -- we don't have to mess with sin's right ???
	cp_1 = test1_h* ref_v - test1_v * ref_h;
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	
	
	// decide right, left or neither
	if (cp_1 < 0.)
	{
		location=-1;		// left
	}
	else if (cp_1 > 0.)
	{
		location = 1;	// right
	}
	else //if (cp_1 == 0.)
	{
		location = 0;	// The point lies on the line of the reference segment.
	}	
	
	return (location);
}

////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Navigate the DAG tree.  Given a point on the surface of the earth
// locate the triangle that the point falls within or identify if
// the point is outside of the boundary (the infinite triangle).
// RETURNS a negative number if there is an error
//////////////////////////////////////////////////////////////////////
long WhatTriIsPtIn(DAGHdl treeH,TopologyHdl topH, LongPointHdl ptsH,LongPoint pt)
{
	long i;							// an index
	short direction;				// the index for the direction of pt from the reference segment
	long firstPoint, secondPoint; // the point numbers for the test segment in the side list
	long thirdPoint;					// the 3rd point of 1 of the triangles including the test segment.
	//Boolean inNeighborTriangle;	// Is the testpoint in the triangle next to the segment?
	
	long triNum;					//  The triangle number computed from the topology index
	//			also is the number of the triangle to the left of the segment.
	long triNumIndex;				//  The index to start the topology record from triangle number triNum
	long secondPointIndex;		//  The index to the segment second point in the topo array.
	long thirdPointIndex;		//			"			"				third      " 			"
	long triRight;					//	 The number of the triangle to the right of the segment
	long triRightIndex;			//  The index of triRight into the topo array.
	long dagIndex;					//  The current topology index referred to in the dag tree.
	
	long** longH = (long**)(topH);  
	
	i=0;
	//while ((*treeH)[i].branchLeft >= 0)
	while ((*treeH)[i].branchLeft != -1 && (*treeH)[i].branchRight != -1)
	{
		dagIndex = ((*treeH)[i].topoIndex);
		triNum = dagIndex/6;
		triNumIndex = (triNum)*6;
		firstPoint = (*longH)[dagIndex];
		secondPointIndex = triNumIndex + ((dagIndex+1)%3);
		secondPoint = (*longH)[secondPointIndex];
		thirdPointIndex = triNumIndex + ((dagIndex+2)%3);
		direction =Right_or_Left_of_Segment(ptsH,firstPoint,secondPoint,pt);
		if (direction == -1) // left
		{
			// Am I in the triangle directly to the left of the segment?
			// if the tri on the left is infinity, we can't check it
			thirdPoint = (*longH)[thirdPointIndex];
			if(thirdPoint>=0)
			{
				direction =Right_or_Left_of_Segment(ptsH,secondPoint, thirdPoint, pt);
				if(direction == -1)
				{
					direction =Right_or_Left_of_Segment(ptsH,thirdPoint, firstPoint, pt);
					if(direction == -1)
					{
						return (triNum); //Start numbering triangles at zero
					}
				}
			}
			// if we get here, the point was not in the triangle on the left
			// Guess I'd better step on down the Dag tree...
			i = (*treeH)[i].branchLeft;
		}
		//else if (direction == 1) // right
		else // else it is to the right or on the line (in which case it is in both and we can use either side) JLM 10/15/99
		{
			// Am I in the triangle directly to the right of the segment?
			// if the tri on the left is infinity, we can't check it...should get it up at the top.
			triRight = (*longH)[thirdPointIndex+3];
			
			if(triRight == -1)
			{
				//i=-8;
				//goto checkTriPts;	// this caused large runs to grind to a halt when LEs beached
				return -1;
			}
			triRightIndex = (triRight)*6;
			// The order will reverse because the triangles are defined counterclockwise and 
			//			our segment is defined clockwise.
			thirdPoint = FindTriThirdPoint(longH,secondPoint,firstPoint,triRightIndex);
			if(thirdPoint>=0)
			{
				direction = Right_or_Left_of_Segment(ptsH,thirdPoint, secondPoint, pt);
				if(direction == -1) 
				{
					direction = Right_or_Left_of_Segment(ptsH,firstPoint, thirdPoint, pt);
					if(direction == -1)  
					{
						//ptsH = nil;
						return (triRight); //Start numbering triangles at zero
					}
				}
			}
			// Guess I'd better step on down the Dag tree...
			i = (*treeH)[i].branchRight;
		}
		
		/////////////////////////////////////////////////
		/////////////////////////////////////////////////
	checkTriPts:
		if (i== -8)
		{	// occasionally lost right on a vertex - this came up in dispersed oil Gnome
			if (pt.h == (*ptsH)[firstPoint].h && pt.v == (*ptsH)[firstPoint].v 
				|| pt.h == (*ptsH)[secondPoint].h && pt.v == (*ptsH)[secondPoint].v 
				|| pt.h == (*ptsH)[thirdPoint].h && pt.v == (*ptsH)[thirdPoint].v)	
				return (triNum);
			/////////////////////////////////////////////////
			// check all triangles that include any of the original vertices in case we're close
			long numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);
			long testPt1,testPt2,testPt3,triIndex;
			for (triIndex=0;triIndex<numTri;triIndex++)
			{
				testPt1 = (*longH)[triIndex*6];
				testPt2 = (*longH)[triIndex*6+1];
				testPt3 = (*longH)[triIndex*6+2];
				
				if(firstPoint==testPt1 || firstPoint==testPt2 || firstPoint==testPt3 ||
				   secondPoint==testPt1 || secondPoint==testPt2 || secondPoint==testPt3 ||
				   thirdPoint==testPt1 || thirdPoint==testPt2 || thirdPoint==testPt3
				   && triIndex != triNumIndex)	// already checked main triangle
				{
					direction = Right_or_Left_of_Segment(ptsH,testPt1,testPt2,pt);
					if (direction == -1) // left
					{
						// Am I in the triangle directly to the left of the segment?
						direction = Right_or_Left_of_Segment(ptsH,testPt2,testPt3,pt);
						if(direction == -1)
						{
							direction = Right_or_Left_of_Segment(ptsH,testPt3,testPt1,pt);
							if(direction == -1)
							{
								return (triIndex); 
							}
						}
					}
				}
			}
			/////////////////////////////////////////////////
			
			return(-8); 				// This is a special case caused by not being able
			// to confirm that a point is in the infinite triangle.
										// To see the change, have the function return -8 for triNum and
										// give that triangle a unique color for plotting.		
		}
	}
	return -1; // JLM, we already checked it was not in the triangle 
}

