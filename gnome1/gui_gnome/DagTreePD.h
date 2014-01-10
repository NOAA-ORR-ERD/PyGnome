/*
 *  DagTreePD.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "DagTree.h"
#include "GuiTypeDefs.h"

OSErr ReadIndexedDagTree(BFPB *bfpb,DAGHdl *treeH,char* errmsg);
OSErr WriteIndexedDagTree(BFPB *bfpb,DAGHdl theTree,char* errmsg);

OSErr ReadVertices(BFPB *bfpb,LongPointHdl *ptsH,char* errmsg);
OSErr WriteVertices(BFPB *bfpb, LongPointHdl ptsH, char *errmsg);

OSErr ReadTopology(BFPB *bfpb,TopologyHdl *topH, VelocityFH *velH,char* errmsg);
OSErr WriteTopology(BFPB *bfpb,TopologyHdl topH,VelocityFH velocityH, char *errmsg);
