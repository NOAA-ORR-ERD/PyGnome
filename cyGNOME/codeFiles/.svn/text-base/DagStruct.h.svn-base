
#ifndef __DAGSTRUCT__
#define __DAGSTRUCT__
 
typedef struct	Topology
{
		long	vertex1;
		long	vertex2;
		long	vertex3;
		long	adjTri1;		// adjTriN is opposite vertexN
		long	adjTri2;
		long	adjTri3;

} Topology, *TopologyPtr, **TopologyHdl;


typedef struct DAG
{
	//long				segNumber;
	long				topoIndex;    // Index into topology array;
	long				branchLeft;   // index in node_or_triPtr;
	long				branchRight;  // index in node_or_triPtr;
} DAG,*DAGPtr,**DAGHdl;

typedef struct DAGTreeStruct	
{
	long numBranches;						// number of elements in the DAG tree
	DAG ** treeHdl;						// handle for the DAG tree
	//Side_List ** sideHdl;				// handle to the sidelist
} DAGTreeStruct;


		
#endif

