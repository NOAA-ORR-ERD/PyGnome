/*
 *  DagTreePD.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "DagTreePD.h"
#include "MemUtils.h"
#include "StringFunctions.h"
#include "CROSS.H"

/////////////////////////////////////////////////////////////////
OSErr WriteIndexedDagTree(BFPB *bfpb,DAGHdl theTree,char* errmsg)
{
	OSErr	err = noErr;
	long	i;
	long	numRecs;
	DAG		thisDag;
	
	numRecs = _GetHandleSize ((Handle) theTree) / sizeof (DAG);
	if (err = WriteMacValue(bfpb, numRecs)) goto done;
	
	for(i=0;i<numRecs;i++)
	{
		thisDag = (*theTree)[i];
		if (err = WriteMacValue(bfpb, thisDag.topoIndex)) goto done;
		if (err = WriteMacValue(bfpb, thisDag.branchLeft)) goto done;
		if (err = WriteMacValue(bfpb, thisDag.branchRight)) goto done;
	}	
	
done:
	
	if (err)
	{
		strcpy (errmsg, "Error while writing topology to file.");
	}
	
	return err;
}
/////////////////////////////////////////////////////////////////
OSErr ReadIndexedDagTree(BFPB *bfpb,DAGHdl *treeH,char* errmsg)
{
	OSErr	err = noErr;
	long	i;
	long	numRecs;
	DAG		thisDag;
	
	if (err = ReadMacValue(bfpb, &numRecs)) goto done;
	*treeH = (DAGHdl)_NewHandle(sizeof(DAG)*(numRecs));
	
	for(i=0;i<numRecs;i++)
	{
		if (err = ReadMacValue(bfpb, &thisDag.topoIndex)) goto done;
		if (err = ReadMacValue(bfpb, &thisDag.branchLeft)) goto done;
		if (err = ReadMacValue(bfpb, &thisDag.branchRight)) goto done;
		(**treeH)[i] = thisDag;
	}	
	
done:
	
	if (err)
	{
		strcpy (errmsg, "Error while writing topology to file.");
	}

	return err;
}

/////////////////////////////////////////////////////////////////
OSErr WriteVertices(BFPB *bfpb, LongPointHdl ptsH, char *errmsg)
{
	long		numPoints, i;
	LongPoint	thisLPoint;
	OSErr		err = noErr;
	
	numPoints = _GetHandleSize((Handle)ptsH)/sizeof(LongPoint);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	
	for(i=0;i<numPoints;i++)
	{
		thisLPoint.h = (*ptsH)[i].h;
		thisLPoint.v = (*ptsH)[i].v;
		
		if (err = WriteMacValue(bfpb, thisLPoint.h)) goto done;
		if (err = WriteMacValue(bfpb, thisLPoint.v)) goto done;		
	}
	
done:
	
	if(err) 
	{
		strcpy (errmsg, "Error while writing vertices to file.");
	}
	
	return err;		
}
/////////////////////////////////////////////////////////////////
OSErr ReadVertices(BFPB *bfpb, LongPointHdl *ptsH, char *errmsg)
{
	long		numPoints, i;
	LongPoint	thisLPoint;
	OSErr		err = noErr;
	
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;
	*ptsH = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numPoints));
	if (!*ptsH)
	{
		err = memFullErr;
		goto done;
	}
	
	for(i=0;i<numPoints;i++)
	{
		if (err = ReadMacValue(bfpb, &thisLPoint.h)) goto done;
		if (err = ReadMacValue(bfpb, &thisLPoint.v)) goto done;
		(**ptsH)[i].h = thisLPoint.h;
		(**ptsH)[i].v = thisLPoint.v;	
	}
	
done:
	
	if(err) 
	{
		strcpy (errmsg, "Error while reading vertices to file.");
	}
	
	return err;		
}
/////////////////////////////////////////////////////////////////
OSErr WriteTopology(BFPB *bfpb,TopologyHdl topH,VelocityFH velocityH, char *errmsg)
{ 
	long 		numRecs, i;
	Topology	thisTop;
	VelocityFRec	thisVelF;
	OSErr		err = noErr;
	char haveVelocityHdl;
	
	numRecs = _GetHandleSize ((Handle) topH) / sizeof (Topology);
	if (err = WriteMacValue(bfpb, numRecs)) goto done;
	
	for(i=0;i<numRecs;i++)
	{
		thisTop = (*topH)[i];
		
		if (err = WriteMacValue(bfpb, thisTop.vertex1)) goto done;
		if (err = WriteMacValue(bfpb, thisTop.vertex2)) goto done;
		if (err = WriteMacValue(bfpb, thisTop.vertex3)) goto done;
		if (err = WriteMacValue(bfpb, thisTop.adjTri1)) goto done;
		if (err = WriteMacValue(bfpb, thisTop.adjTri2)) goto done;
		if (err = WriteMacValue(bfpb, thisTop.adjTri3)) goto done;
	}
	////
	haveVelocityHdl = (velocityH != 0);
	if (err = WriteMacValue(bfpb, haveVelocityHdl)) goto done;
	
	if(haveVelocityHdl)
	{
		for(i=0;i<numRecs;i++)
		{
			thisVelF = (*velocityH)[i];
			if (err = WriteMacValue(bfpb, thisVelF.u)) goto done;
			if (err = WriteMacValue(bfpb, thisVelF.v)) goto done;		
		}
	}
	
	
done:
	
	if (err)
	{
		strcpy (errmsg, "Error while writing topology to file.");
	}
	
	return err;
}
/////////////////////////////////////////////////////////////////
OSErr ReadTopology(BFPB *bfpb,TopologyHdl *topH,VelocityFH *velocityH, char *errmsg)
{ 
	long 		numRecs, i;
	Topology	thisTop;
	VelocityFRec	thisVelF;
	OSErr		err = noErr;
	char haveVelocityHdl;
	
	*topH = 0;
	*velocityH = 0;
	
	if (err = ReadMacValue(bfpb, &numRecs)) goto done;
	if(numRecs <= 0) {err = TRUE; goto done;}
	*topH = (TopologyHdl)_NewHandle(sizeof(***topH)*(numRecs));
	
	for(i=0;i<numRecs;i++)
	{		
		memset(&thisTop,0,sizeof(thisTop));
		if (err = ReadMacValue(bfpb, &thisTop.vertex1)) goto done;
		if (err = ReadMacValue(bfpb, &thisTop.vertex2)) goto done;
		if (err = ReadMacValue(bfpb, &thisTop.vertex3)) goto done;
		if (err = ReadMacValue(bfpb, &thisTop.adjTri1)) goto done;
		if (err = ReadMacValue(bfpb, &thisTop.adjTri2)) goto done;
		if (err = ReadMacValue(bfpb, &thisTop.adjTri3)) goto done;
		(**topH)[i] = thisTop;
	}
	///
	if (err = ReadMacValue(bfpb, &haveVelocityHdl)) goto done;
	
	if(haveVelocityHdl)
	{
		*velocityH = (VelocityFH)_NewHandle(sizeof(***velocityH)*(numRecs));
		for(i=0;i<numRecs;i++)
		{		
			memset(&thisVelF,0,sizeof(thisVelF));
			if (err = ReadMacValue(bfpb, &thisVelF.u)) goto done;
			if (err = ReadMacValue(bfpb, &thisVelF.v)) goto done;		
			(**velocityH)[i] = thisVelF;
		}
	}
	
done:
	
	if (err)
	{
		strcpy (errmsg, "Error while reading topology file.");
		if(*topH) {DisposeHandle((Handle)*topH); *topH = 0;}
		if(*velocityH) {DisposeHandle((Handle)*velocityH); *velocityH = 0;}
	}
	
	return err;
}
/////
