
#ifndef __DAGTREEIO__
#define __DAGTREEIO__

#include "DagTree.h"
#include "RectUtils.h"

OSErr ReadTIndexedDagTree(CHARH fileBUFH,long *line,DAGTreeStruct *dagTree,char* errmsg);
Boolean IsTIndexedDagTreeHeaderLine(char *s, long* numRecs);
OSErr ReadTIndexedDagTreeBody(CHARH fileBUFH,long *line,DAGTreeStruct *dagTree,char* errmsg,long numRecs);

OSErr ReadTTopology(CHARH fileBUFH,long *line,TopologyHdl *topH, VelocityFH *velH,char* errmsg);
Boolean IsTTopologyHeaderLine(char *s, long* numPts);
OSErr ReadTTopologyBody(CHARH fileBufH,long *line,TopologyHdl *topH,VelocityFH *velocityH,char* errmsg,long numRecs,Boolean wantVelData);

OSErr ReadTVertices(CHARH fileBUFH,long *line,LongPointHdl *ptsH,FLOATH *depthsH,char* errmsg);
Boolean IsTVerticesHeaderLine(char *s, long* numPts);
OSErr ReadTVerticesBody(CHARH fileBufH,long *line,LongPointHdl *pointsH,FLOATH *depthsH,char* errmsg,long numPoints,Boolean wantDepths);

long FindTriThirdPoint(long **longH,long p1, long p2, long index);
int	Right_or_Left_of_Segment(LongPointHdl ptsH,long ref_p1,long ref_p2, LongPoint test_p1);
long WhatTriIsPtIn(DAGHdl treeH,TopologyHdl topH, LongPointHdl ptsH,LongPoint pt);

Boolean IsPtCurVerticesHeaderLine(char *s, long* numPts, long* numLandPts);

Boolean IsWaterBoundaryHeaderLine(char *s, long* numWaterBoundaries, long* numBoundaryPts);
Boolean IsBoundarySegmentHeaderLine(char *s, long* numBoundarySegs);
Boolean IsBoundaryPointsHeaderLine(char *s, long* numBoundaryPts);

OSErr ReadBoundarySegs(CHARH fileBufH,long *line,LONGH *boundarySegs,long numBoundarySegs,char* errmsg);
OSErr ReadWaterBoundaries(CHARH fileBufH,long *line,LONGH *waterBoundaries,long numWaterBoundaries,long numBoundaryPts,char* errmsg);
OSErr ReadBoundaryPts(CHARH fileBufH,long *line,LONGH *boundaryPts,long numBoundaryPts,char* errmsg);

Boolean IsTransposeArrayHeaderLine(char *s, long* numPts);
OSErr ReadTransposeArray(CHARH fileBufH,long *line,LONGH *transposeArray,long numPts,char* errmsg);

#endif
