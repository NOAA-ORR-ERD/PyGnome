
#include "Basics.h"
#include "TypeDefs.h"
#include "MemUtils.h"
#include "StringFunctions.h"

#include "RectUtils.h"
#include "DagTreeIO.h" 

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

#define DAGFILE_DELIM_STR " \t"

void StartReadWriteSequence(char *procName);

using namespace std;
/////////////////////////////////////////////////

Boolean IsTIndexedDagTreeHeaderLine(char *s, long* numRecs)
{
	// note this method requires a dummy line in ptcur file or else the next line is garbled or skipped
	/*char* token = strtok(s,DAGFILE_DELIM_STR);
	*numRecs = 0;
	if(!token || strncmp(token,"DAGTree",strlen("DAGTree")) != 0)
	{
		return FALSE;
	}

	token = strtok(NULL,DAGFILE_DELIM_STR);
	
	if(!token || sscanf(token,"%ld",numRecs) != 1)
	{
		return FALSE;
	}
	
	return TRUE;*/
	
	char* strToMatch = "DAGTree";
	long numScanned, len = strlen(strToMatch);
	if(!strncmpnocase(s,strToMatch,len)) {
		numScanned = sscanf(s+len+1,"%ld",numRecs);
		if (numScanned != 1 || *numRecs <= 0.0)
			return FALSE; 
	}
	else
		return FALSE;
	return TRUE; 
}


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
OSErr ReadTIndexedDagTreeBody(CHARH fileBufH,long *line,DAGTreeStruct *dagTree,char* errmsg,long numRecs)
{
	DAG** dagListHdl = nil;
	long i;
	long topoIndex,left,right;
	OSErr err = -1;
	char s[256];
	
	strcpy(errmsg,"");//clear it
	dagListHdl = (DAG**)_NewHandle(sizeof(DAG)*numRecs);

	if(!dagListHdl)goto done;

	for(i=0;i<numRecs;i++)
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); // 10 u values
		if(sscanf(s,"%ld%ld%ld",&topoIndex,&left,&right) < 3)
		{
			char firstPartOfLine[128];
			sprintf(errmsg,"Unable to read DAG Tree data from line %ld:%s",*line,NEWLINESTRING);
			strncpy(firstPartOfLine,s,120);
			strcpy(firstPartOfLine+120,"...");
			strcat(errmsg,firstPartOfLine);
			goto done;
		}
		if(topoIndex == -1) break; //end of the points

		(*dagListHdl)[i].topoIndex = topoIndex;
		(*dagListHdl)[i].branchLeft = left;		// left branch of dag tree - index
		(*dagListHdl)[i].branchRight = right;	// right ...
	}
	
	dagTree->numBranches = numRecs;
	dagTree->treeHdl = dagListHdl;
	
	if(i<=0)
	{
		strcpy(errmsg,"Error reading data file. No points were read."
		"Data file may contain syntax errors or may not be a recognized file type.");
		goto done;
	}
	
	err = noErr;
done:

	if(err) 
	{
		if(dagListHdl)DisposeHandle((Handle)dagListHdl);
	}

	return err;
}

/////////////////////////////////////////////////////////////////
OSErr ReadTIndexedDagTree(CHARH fileBufH,long *line,DAGTreeStruct *dagTree,char* errmsg)
{
	long numRecs;
	OSErr err = -1;
	char s[256];
	
	strcpy(errmsg,"");//clear it
		
	NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
	if(!IsTIndexedDagTreeHeaderLine(s,&numRecs))
		return -1;
		
	err = ReadTIndexedDagTreeBody(fileBufH,line,dagTree,errmsg,numRecs);
	return err;

}
/////////////////////////////////////////////////

/////////////////////////////////////////////////
	
/////////////////////////////////////////////////

Boolean IsTVerticesHeaderLine(char *s, long* numPts)
{
	char* token = strtok(s,DAGFILE_DELIM_STR);
	*numPts = 0;
	if(!token || strncmp(token,"Vertices",strlen("Vertices")) != 0)
	{
		return FALSE;
	}

	token = strtok(NULL,DAGFILE_DELIM_STR);
	
	if(!token || sscanf(token,"%ld",numPts) != 1)
	{
		return FALSE;
	}
	
	return TRUE;
}

/////////////////////////////////////////////////////////////////
OSErr ReadTVerticesBody(CHARH fileBufH,long *line,LongPointHdl *pointsH,FLOATH *depthsH,char* errmsg,long numPoints,Boolean wantDepths)
// Note: '*line' must contain the line# at which the vertex data begins
{
	LongPointHdl ptsH = nil;
	FLOATH depthValuesH = nil;
	Boolean badScan = false;
	long i;
	double z,x,y;
	OSErr err=-1;
	char s[256];

	strcpy(errmsg,"");//clear it
	*pointsH = 0;
	*depthsH = 0;

	wantDepths = true; // code goes here, decide which files have depths
	// Skip next line. It just contains number of records
	NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 


	ptsH = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numPoints));
	if(ptsH == nil)
	{
		strcpy(errmsg,"Not enough memory to read dagtree file.");
		return -1;
	}
	
	
	NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
	if (sscanf(s,lfFix("%lf%lf%lf"),&x,&y,&z) < 3) wantDepths = false;	// in case any current files don't have depth data, it's usually not used
	if (sscanf(s,lfFix("%lf%lf"),&x,&y) < 2) badScan = true;
	if(badScan)
	{
		char firstPartOfLine[128];
		sprintf(errmsg,"Unable to read vertex data from line %ld:%s",*line,NEWLINESTRING);
		strncpy(firstPartOfLine,s,120);
		strcpy(firstPartOfLine+120,"...");
		strcat(errmsg,firstPartOfLine);
		goto done;
	}
	if (wantDepths)
	{
		depthValuesH = (FLOATH)_NewHandle(sizeof(FLOATH)*(numPoints));
		if(depthValuesH == nil)
		{
			strcpy(errmsg,"Not enough memory to read dagtree file.");
			goto done;
		}
	}
	(*ptsH)[0].h = 1000000 * x;
	(*ptsH)[0].v = 1000000 * y;
	if (wantDepths) (*depthValuesH)[0] = z;
	//for(i=0;i<numPoints;i++)
	for(i=1;i<numPoints;i++)
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
		
		if(wantDepths && sscanf(s,lfFix("%lf%lf%lf"),&x,&y,&z) < 3)
		{
			if (sscanf(s,lfFix("%lf%lf%lf"),&x,&y,&z) < 3) badScan = true;
		}
		if(!wantDepths && sscanf(s,lfFix("%lf%lf"),&x,&y) < 2)
		{
			if (sscanf(s,lfFix("%lf%lf"),&x,&y) < 2) badScan = true;
		}
		if (badScan)
		{
			char firstPartOfLine[128];
			sprintf(errmsg,"Unable to read vertex data from line %ld:%s",*line,NEWLINESTRING);
			strncpy(firstPartOfLine,s,120);
			strcpy(firstPartOfLine+120,"...");
			strcat(errmsg,firstPartOfLine);
			goto done;
		}
		(*ptsH)[i].h = 1000000 * x;
		(*ptsH)[i].v = 1000000 * y;
		if (wantDepths) (*depthValuesH)[i] = z;
	}

	*pointsH = ptsH;
	*depthsH = depthValuesH;
	err = noErr;

done:
	
	if(errmsg[0]) 
	{
		if(ptsH) DisposeHandle((Handle)ptsH);
		if(depthValuesH) DisposeHandle((Handle)depthValuesH);
		return -1;
	}
	return err;		
}


/////////////////////////////////////////////////////////////////
OSErr ReadTVertices(CHARH fileBufH,long *line,LongPointHdl *pointsH,FLOATH *depthsH,char* errmsg)
// Note: '*line' must contain the line# at which the vertex data begins
{
	OSErr err=-1;
	char s[256];
	long numPoints;
	Boolean wantDepths = false;

	strcpy(errmsg,"");//clear it
	*pointsH = 0;
	*depthsH = 0;

	// Read in the vertices header "Vertices numRecs"
	NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
	if(!IsTVerticesHeaderLine(s,&numPoints))
		return -1;
	
	err = ReadTVerticesBody(fileBufH,line,pointsH,depthsH,errmsg,numPoints,wantDepths);
	return err;


}
Boolean IsTTopologyHeaderLine(char *s, long* numPts)
{
	// note this method requires a dummy line in ptcur file or else the next line is garbled or skipped
	/*char* token = strtok(s,DAGFILE_DELIM_STR);
	*numPts = 0;
	if(!token || strncmp(token,"Topology",strlen("Topology")) != 0)
	{
		return FALSE;
	}

	token = strtok(NULL,DAGFILE_DELIM_STR);
	
	if(!token || sscanf(token,"%ld",numPts) != 1)
	{
		return FALSE;
	}
	
	return TRUE;*/
	
	char* strToMatch = "Topology";
	long numScanned, len = strlen(strToMatch);
	if(!strncmpnocase(s,strToMatch,len)) {
		numScanned = sscanf(s+len+1,"%ld",numPts);
		if (numScanned != 1 || *numPts <= 0.0)
			return FALSE; 
	}
	else
		return FALSE;
	return TRUE; 
}

/////////////////////////////////////////////////////////////////
OSErr ReadTTopologyBody(CHARH fileBufH,long *line,TopologyHdl *topH,VelocityFH *velocityH,char* errmsg,long numRecs,Boolean wantVelData)
{ 
	TopologyHdl topoH=0;
	VelocityFH velH = 0;
	long i,numScanned;
	long v1,v2,v3,t1,t2,t3;
	double u,v;
	OSErr err = -1;
	char s[256];
	Boolean badScan = FALSE;
	
	strcpy(errmsg,"");
	*topH = 0;
	*velocityH = 0;
	
	if(numRecs <= 0)
	{
		strcpy(errmsg,"numRecs cannot be <= 0 in ReadTTopologyBody");
		return -1;
	}

	topoH = (TopologyHdl)_NewHandle(sizeof(Topology)*numRecs);
	if(!topoH)
	{
		strcpy(errmsg,"Not enough memory.");
		goto done;
	}

	//check whether or not velocities are included
	if(wantVelData)
	{
		velH = (VelocityFH)_NewHandle(sizeof(**velH)*numRecs);
		if(!velH)
		{
			strcpy(errmsg,"Not enough memory.");
			goto done;
		}
	}
	
	
	for(i=0;i<numRecs;i++)
	{
      	NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
		if(wantVelData)
		{
			numScanned=sscanf(s,lfFix("%ld%ld%ld%ld%ld%ld%lf%lf"),&v1,&v2,&v3,&t1,&t2,&t3,&u,&v) ;
			if (numScanned != 8)
				badScan = TRUE;
		}
		else
		{
			numScanned=sscanf(s,"%ld%ld%ld%ld%ld%ld",&v1,&v2,&v3,&t1,&t2,&t3);
			if (numScanned != 6)
				badScan = TRUE;
		}
		
		if (badScan)
		{
			char firstPartOfLine[128];
			sprintf(errmsg,"Unable to read topology data from line %ld:%s",*line,NEWLINESTRING);
			strncpy(firstPartOfLine,s,120);
			strcpy(firstPartOfLine+120,"...");
			strcat(errmsg,firstPartOfLine);
			goto done;
		}
	
		(*topoH)[i].vertex1 = v1;
		(*topoH)[i].vertex2 = v2;
		(*topoH)[i].vertex3 = v3;
		if(wantVelData) {
			(*velH)[i].u = u;
			(*velH)[i].v = v;
		}
		if(t1 == -1) ((*topoH)[i].adjTri1 = t1); 
		else ((*topoH)[i].adjTri1 = t1);	
		if(t2 == -1) ((*topoH)[i].adjTri2 = t2); 
		else ((*topoH)[i].adjTri2 = t2);	
		if(t3 == -1) ((*topoH)[i].adjTri3 = t3); 
		else ((*topoH)[i].adjTri3 = t3);
	}
		

	err = noErr;
	*topH = topoH;
	*velocityH = velH;

done:
	
	if(errmsg[0])
	{
		if (topoH) DisposeHandle((Handle)topoH);
		if(velH)DisposeHandle((Handle)velH);
		if(err == noErr) err = -1; // make sure we return an error
	}
	
	return err;
}

/////////////////////////////////////////////////////////////////
OSErr ReadTTopology(CHARH fileBufH,long *line,TopologyHdl *topH,VelocityFH *velocityH,char* errmsg)
{ 
	long numRecs;
	OSErr err = -1;
	char s[256];
	Boolean wantVelData = true;
	
	strcpy(errmsg,"");
		
	// Read in the topology header "Topology numRecs"
	NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
	
	if(!IsTTopologyHeaderLine(s,&numRecs))
		return -1;
		
	err = ReadTTopologyBody(fileBufH,line,topH,velocityH,errmsg,numRecs,wantVelData);
	return err;

}

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

/////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
Boolean IsWaterBoundaryHeaderLine(char *s, long* numWaterBoundaries, long* numBoundaryPts)
{	
	char* strToMatch = "WaterBoundaries";
	long numScanned, len = strlen(strToMatch);
	if(!strncmpnocase(s,strToMatch,len)) {
		numScanned = sscanf(s+len+1,"%ld%ld",numWaterBoundaries,numBoundaryPts);
		if (numScanned != 2 || *numWaterBoundaries < 0 || *numBoundaryPts <=0)
			return FALSE; 
	}
	else
		return FALSE;
	return TRUE; 
}

Boolean IsBoundarySegmentHeaderLine(char *s, long* numBoundarySegs)
{		
	char* strToMatch = "BoundarySegments";
	long numScanned, len = strlen(strToMatch);
	if(!strncmpnocase(s,strToMatch,len)) {
		numScanned = sscanf(s+len+1,"%ld",numBoundarySegs);
		if (numScanned != 1 || *numBoundarySegs <= 0)
			return FALSE; 
	}
	else
		return FALSE;
	return TRUE; 
}

Boolean IsBoundaryPointsHeaderLine(char *s, long* numBoundaryPts)
{		
	char* strToMatch = "BoundaryPoints";
	long numScanned, len = strlen(strToMatch);
	if(!strncmpnocase(s,strToMatch,len)) {
		numScanned = sscanf(s+len+1,"%ld",numBoundaryPts);
		if (numScanned != 1 || *numBoundaryPts < 0)	
			return FALSE; 
	}
	else
		return FALSE;
	return TRUE; 
}

OSErr ReadWaterBoundaries(CHARH fileBufH,long *line,LONGH *waterBoundaries,long numWaterBoundaries,long numBoundaryPts,char* errmsg)
// Note: '*line' must contain the line# at which the vertex data begins
{
	OSErr err=0;
	char s[64];
	long i,numScanned,waterBoundaryIndex;
	LONGH boundaryTypeH=0;
	
	strcpy(errmsg,""); // clear it

	boundaryTypeH = (LONGH)_NewHandle(sizeof(long)*numBoundaryPts);
	if(!boundaryTypeH){TechError("ReadWaterBoundaries()", "_NewHandle()", 0); err = memFullErr; goto done;}
	*waterBoundaries = 0;
	
	for (i = 0; i<numBoundaryPts; i++)
	{
		(*boundaryTypeH)[i] = 1;	// default is land
	}
	for (i = 0; i<numWaterBoundaries; i++)
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 64); 
		numScanned=sscanf(s,"%ld",&waterBoundaryIndex) ;
		if (numScanned!= 1)
			{ err = -1; TechError("ReadWaterBoundaries()", "sscanf() == 1", 0); goto done; }
		(*boundaryTypeH)[waterBoundaryIndex] = 2;
	}	
	*waterBoundaries = boundaryTypeH;

done:
	
	if(err) 
	{
		if(boundaryTypeH) {DisposeHandle((Handle)boundaryTypeH); boundaryTypeH=0;}
	}
	return err;		
}

/////////////////////////////////////////////////////////////////
OSErr ReadBoundarySegs(CHARH fileBufH,long *line,LONGH *boundarySegs,long numSegs,char* errmsg)
// Note: '*line' must contain the line# at which the vertex data begins
{ // May want to combine this with read vertices if it becomes a mandatory component of PtCur files
	OSErr err=0;
	char s[64];
	long i,numScanned,boundarySeg;
	long oldSegno=0;
	LONGH segsH = 0;
	
	strcpy(errmsg,""); // clear it

	segsH = (LONGH)_NewHandle(sizeof(long)*numSegs);
	if(!segsH){TechError("ReadBoundarySegs()", "_NewHandle()", 0); err = memFullErr; goto done;}

	*boundarySegs=0;	
	 
	for(i=0;i<numSegs;i++)
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 64); 
		numScanned=sscanf(s,"%ld",&boundarySeg) ;
		if (numScanned!= 1)
			{ err = -1; TechError("ReadBoundarySegs()", "sscanf() == 1", 0); goto done; }
		--boundarySeg;
		if(boundarySeg - oldSegno < 2)
		{
			sprintf(errmsg,
				"Less than 3 points in boundary number: %ld, from point %ld to point %ld."
				"Triangle generation will fail.",i+1, oldSegno+1,boundarySeg+1);
			printError(errmsg);
		}
		(*segsH)[i] = oldSegno = boundarySeg;
	}
	*boundarySegs = segsH;

done:
	
	if(err) 
	{
		if(segsH) {DisposeHandle((Handle)segsH); segsH=0;}
	}
	return err;		

}
/////////////////////////////////////////////////////////////////
OSErr ReadBoundaryPts(CHARH fileBufH,long *line,LONGH *boundaryPts,long numPts,char* errmsg)
// Note: '*line' must contain the line# at which the vertex data begins
{ // May want to combine this with read vertices if it becomes a mandatory component of PtCur files
	OSErr err=0;
	char s[64];
	long i,numScanned,boundaryPt;
	LONGH ptsH = 0;
	
	strcpy(errmsg,""); // clear it

	ptsH = (LONGH)_NewHandle(sizeof(long)*numPts);
	if(!ptsH){TechError("ReadBoundaryPts()", "_NewHandle()", 0); err = memFullErr; goto done;}

	*boundaryPts=0;	
	 
	for(i=0;i<numPts;i++)
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 64); 
		numScanned=sscanf(s,"%ld",&boundaryPt) ;
		if (numScanned!= 1)
			{ err = -1; TechError("ReadBoundaryPts()", "sscanf() == 1", 0); goto done; }
		(*ptsH)[i] = boundaryPt;
	}
	*boundaryPts = ptsH;

done:
	
	if(err) 
	{
		if(ptsH) {DisposeHandle((Handle)ptsH); ptsH=0;}
	}
	return err;		

}
