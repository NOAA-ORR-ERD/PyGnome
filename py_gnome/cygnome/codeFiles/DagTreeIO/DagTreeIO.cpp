#include "Earl.h"
#include "TypeDefs.h"
#include "Cross.h"
// File IO routines for reading in a dag file.
#include "RectUtils.h"

//#include "Cross.h"
#include "DagTreeIO.h" 
//#include <stdio.h>
#include <iostream>

#define DAGFILE_DELIM_STR " \t"

#ifndef pyGNOME
void StartReadWriteSequence(char *procName);
#else
#define TechError(a, b, c) printf(a)
#define printError(msg) printf(msg)
#endif

using namespace std;
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
#ifndef pyGNOME

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

#endif

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
	
	if(!IsTVerticesHeaderLine(s,&numPoints)) {	
		return -1;
	}
	err = ReadTVerticesBody(fileBufH,line,pointsH,depthsH,errmsg,numPoints,wantDepths);
	return err;


}

#ifndef pyGNOME
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
#endif

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
