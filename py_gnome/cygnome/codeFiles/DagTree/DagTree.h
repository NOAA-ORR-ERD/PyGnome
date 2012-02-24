
#ifndef __DAGTREE__
#define __DAGTREE__

#include "Earl.h"
#include "TypeDefs.h"
#include "RectUtils.h"
#include "Geometry.h"

typedef struct	Topology
{
		long	vertex1;
		long	vertex2;
		long	vertex3;
		long	adjTri1;				// adjTriN is opposite vertexN
		long	adjTri2;
		long	adjTri3;

} Topology, *TopologyPtr, **TopologyHdl;

typedef struct DAG
{
	long				topoIndex;		// Index into topology array;
	long				branchLeft;		// index in node_or_triPtr;
	long				branchRight;	// index in node_or_triPtr;
} DAG,*DAGPtr,**DAGHdl;

typedef struct DAGTreeStruct	
{
	long numBranches;					// number of elements in the DAG tree
	DAG ** treeHdl;						// handle for the DAG tree
//	Side_List ** sideHdl;				// handle to the sidelist
} DAGTreeStruct;

class TDagTree
{
	public:
		long				fNumBranches;
		DAGHdl				fTreeH;
		LongPointHdl 		fPtsH;
		TopologyHdl			fTopH;
		VelocityFH			fVelH;
		//long**				longH;

		int Right_or_Left_Point(long ref_p1, long ref_p2, LongPoint test_p1);
		long FindThirdPoint(long p1, long p2, long index);

	public:
						TDagTree(LongPointHdl points, TopologyHdl topo,DAG **tree, VelocityFH velocityH,long numbranches);
		virtual		   ~TDagTree(){Dispose();}
		virtual void 	Dispose();

		long			WhatTriAmIIn(LongPoint pt);
		LongPointHdl	GetPointsHdl(){return fPtsH;};
		TopologyHdl		GetTopologyHdl(){return fTopH;};
		VelocityFH		GetVelocityHdl(){return fVelH;};
		DAGHdl			GetDagTreeHdl(){return fTreeH;};
		void			GetVelocity(long ntri,VelocityRec *r);
		void			GetVelocity(LongPoint lp,VelocityRec *r);
};

#endif 
