
#ifndef __DAGTREEIO__
#define __DAGTREEIO__

#include <vector>
#include "DagTree.h"
#include "RectUtils.h"

OSErr ReadTIndexedDagTree(CHARH fileBUFH,long *line,DAGTreeStruct *dagTree,char* errmsg);

bool IsTIndexedDagTreeHeaderLine(const std::string &strIn, long &numRecs);
Boolean IsTIndexedDagTreeHeaderLine(const char *s, long *numRecs);

OSErr ReadTIndexedDagTreeBody(std::vector<std::string> &linesInFile, long *line,
							  DAGTreeStruct *dagTree,
							  char *errmsg, long numRecs);
OSErr ReadTIndexedDagTreeBody(CHARH fileBUFH,long *line,DAGTreeStruct *dagTree,char* errmsg,long numRecs);

bool IsTTopologyHeaderLine(const std::string &strIn, long &numPts);
Boolean IsTTopologyHeaderLine(char *s, long *numPts);

OSErr ReadTTopology(std::vector<std::string> &linesInFile, long *line,
					TopologyHdl *topH, VelocityFH *velocityH, char *errmsg);
OSErr ReadTTopology(CHARH fileBUFH, long *line,
					TopologyHdl *topH, VelocityFH *velocityH, char *errmsg);

OSErr ReadTTopologyBody(std::vector<std::string> &linesInFile, long *line,
						TopologyHdl *topH, VelocityFH *velocityH,
						char *errmsg, long numRecs, Boolean wantVelData);
OSErr ReadTTopologyBody(CHARH fileBufH,long *line,TopologyHdl *topH,VelocityFH *velocityH,char* errmsg,long numRecs,Boolean wantVelData);

OSErr ReadTVertices(std::vector<std::string> &linesInFile, long *line,
					LongPointHdl *ptsH, FLOATH *depthsH,
					char *errmsg);
OSErr ReadTVertices(CHARH fileBUFH, long *line,
					LongPointHdl *ptsH, FLOATH *depthsH,
					char *errmsg);

bool IsTVerticesHeaderLine(const std::string &strIn, long &numPts);
Boolean IsTVerticesHeaderLine(const char *s, long *numPts);

OSErr ReadTVerticesBody(std::vector<std::string> &linesInFile, long *line,
						LongPointHdl *pointsH, FLOATH *depthsH, char *errmsg,
						long numPoints, bool wantDepths);
OSErr ReadTVerticesBody(CHARH fileBufH,long *line,
						LongPointHdl *pointsH, FLOATH *depthsH, char *errmsg,
						long numPoints,Boolean wantDepths);

long FindTriThirdPoint(long **longH,long p1, long p2, long index);
int	Right_or_Left_of_Segment(LongPointHdl ptsH,long ref_p1,long ref_p2, LongPoint test_p1);
long WhatTriIsPtIn(DAGHdl treeH,TopologyHdl topH, LongPointHdl ptsH,LongPoint pt);

bool IsPtCurVerticesHeaderLine(const std::string &strIn, long &numPts, long &numLandPts);
Boolean IsPtCurVerticesHeaderLine(const char *s, long* numPts, long* numLandPts);

bool IsWaterBoundaryHeaderLine(const std::string &strIn, long &numWaterBoundaries, long &numBoundaryPts);
Boolean IsWaterBoundaryHeaderLine(const char *s, long *numWaterBoundaries, long *numBoundaryPts);

bool IsBoundarySegmentHeaderLine(const std::string &strIn, long &numBoundarySegs);
Boolean IsBoundarySegmentHeaderLine(const char *s, long* numBoundarySegs);

bool IsBoundaryPointsHeaderLine(const std::string &strIn, long &numBoundaryPts);
Boolean IsBoundaryPointsHeaderLine(const char *s, long *numBoundaryPts);

OSErr ReadBoundarySegs(std::vector<std::string> &linesInFile, long *line,
					   LONGH *boundarySegs, long numSegs, char *errmsg);
OSErr ReadBoundarySegs(CHARH fileBufH,long *line,LONGH *boundarySegs,long numBoundarySegs,char* errmsg);

OSErr ReadWaterBoundaries(std::vector<std::string> &linesInFile, long *line,
						  LONGH *waterBoundaries,
						  long numWaterBoundaries,
						  long numBoundaryPts, char *errmsg);
OSErr ReadWaterBoundaries(CHARH fileBufH,long *line,LONGH *waterBoundaries,long numWaterBoundaries,long numBoundaryPts,char* errmsg);

OSErr ReadBoundaryPts(std::vector<std::string> &linesInFile, long *line,
					  LONGH *boundaryPts, long numBoundaryPts,
					  char *errmsg);
OSErr ReadBoundaryPts(CHARH fileBufH, long *line,
					  LONGH *boundaryPts, long numBoundaryPts,
					  char *errmsg);

bool IsTransposeArrayHeaderLine(const std::string &strIn, long &numPts);
Boolean IsTransposeArrayHeaderLine(const char *s, long *numPts);

OSErr ReadTransposeArray(std::vector<std::string> &linesInFile, long *line,
						 LONGH *transposeArray, long numPts, char *errmsg);
OSErr ReadTransposeArray(CHARH fileBufH, long *line,
						 LONGH *transposeArray, long numPts, char *errmsg);

#endif
