
#include "DagTreeIO.h"
#include "DagTree.h"
#include "MemUtils.h"

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
