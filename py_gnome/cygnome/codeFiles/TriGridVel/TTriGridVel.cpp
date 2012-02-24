/*
 *  TTriGridVel.cpp
 *  c_gnome
 *
 *  Created by Alex Hadjilambris on 2/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "TTriGridVel.h"
#include "CROSS.H"



OSErr TTriGridVel::TextRead(char *path)
{
	OSErr err=-1;
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	char s[256],errmsg[256];
	long i, line = 0;
	CHARH fileBufH = 0;
	LongPointHdl ptsH;	
	FLOATH depthsH = 0;
	tree.treeHdl = 0;
	if (!path) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &fileBufH))
	{ 
		printError("Invalid Triangle Grid file.");
		goto done; 
	}
	
	_HLock((Handle)fileBufH); // JLM 8/4/99
	
	// Read header
	line = 0;
	NthLineInTextOptimized(*fileBufH,line++, s, 256);
	
	if(strncmp(s,"DAG 1.0",strlen("DAG 1.0")) != 0)
	{
		printError("Unable to read Triangle Velocity file.");
		goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	if(err = ReadTVertices(fileBufH,&line,&pts,&depthsH,errmsg))goto done;
	MySpinCursor(); // JLM 8/4/99
	if(err = ReadTTopology(fileBufH,&line,&topo,&velH,errmsg))goto done;
	MySpinCursor(); // JLM 8/4/99
	if(err = ReadTIndexedDagTree(fileBufH,&line,&tree,errmsg))
	{
		// allow user to leave out the dagtree
		char errmsg[256];
		errmsg[0]=0;
		err = 0;
		//DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage("Making Dag Tree");
		tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); // use CATSDagTree.cpp and my_build_list.h
		DisplayMessage(0);
		if (errmsg[0])	
			err = -1; // for now we require TIndexedDagTree
		// code goes here, support Galt style ??
		if(err) goto done;
	}
	//goto done;
	MySpinCursor(); // JLM 8/4/99
	
	fDagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches);
	if(!fDagTree)
	{
		printError("Unable to read Triangle Velocity file.");
		goto done;
	}
	SetBathymetry(depthsH);
	//if(depthsH) {DisposeHandle((Handle)depthsH); depthsH=0;}	
	/////////////////////////////////////////////////
	/// figure out the bounds
	ptsH = fDagTree->GetPointsHdl();
	if(ptsH) 
	{
		long numPoints, i;
		LongPoint	thisLPoint;
		
		numPoints = _GetHandleSize((Handle)ptsH)/sizeof(LongPoint);
		if(numPoints > 0)
		{
			WorldPoint  wp;
			WorldRect bounds = voidWorldRect;
			for(i=0;i<numPoints;i++)
			{
				thisLPoint = (*ptsH)[i];
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
			fGridBounds = bounds;
		}
	}
	/////////////////////////////////////////////////
	
	
	err = noErr;
	
done:
	
	if(fileBufH) 
	{
		_HUnlock((Handle)fileBufH); // JLM 8/4/99
		DisposeHandle((Handle)fileBufH); 
		fileBufH = 0;
	}
	
	if(err)
	{
		TechError("TTriGridVel::TextRead(char* path)", errmsg, 0); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(depthsH) {DisposeHandle((Handle)depthsH); depthsH=0;}	// shouldn't exist
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		if(fDagTree)
		{
			delete fDagTree;
			fDagTree = 0;
		}
	}
	return err;
}
////////////////////////////////////////////////////////////////////////////////////
//#define TTriGridVelREADWRITEVERSION 2  // updated to 2 for Read/WriteTopology haveVelocityHdl variable
#define TTriGridVelREADWRITEVERSION 3  // updated to 3 for bathymetry
OSErr TTriGridVel::Write(BFPB *bfpb)
{
	VelocityRec velocity;
	OSErr 		err = noErr;
	long 		i, version = TTriGridVelREADWRITEVERSION, numDepths = 0;
	ClassID 	id = GetClassID ();
	float 		val;
	char 		errStr[256] = "";
	
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	if (err = WriteMacValue(bfpb, fGridBounds)) return err;
	
	if(err = WriteVertices(bfpb,fDagTree -> fPtsH,errStr))goto done;
	if(err = WriteTopology(bfpb,fDagTree -> fTopH,fDagTree -> fVelH, errStr))goto done;
	if(err = WriteIndexedDagTree(bfpb,fDagTree -> fTreeH,errStr))goto done;
	
	if (fBathymetryH) numDepths = _GetHandleSize((Handle)fBathymetryH)/sizeof(**fBathymetryH);
	if (err = WriteMacValue(bfpb, numDepths)) goto done;
	for (i=0;i<numDepths;i++)
	{
		val = INDEXH(fBathymetryH,i);
		if (err = WriteMacValue(bfpb, val)) goto done;
	}
done:
	
	if(err)
		TechError("TTriGridVel::Write(char* path)", errStr, 0); 
	
	return err;
}
////////////////////////////////////////////////////////////////////////////////////
OSErr TTriGridVel::Read(BFPB *bfpb)
{
	OSErr err = noErr;
	char errmsg[256];
	long numBranches, numDepths;
	float val;
	TopologyHdl topH=0;
	LongPointHdl ptsH=0;
	VelocityFH velH = 0;
	DAGHdl treeH = 0;
	long i, version;
	ClassID id;
	
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { printError("Bad id in TTriGridVel::Read"); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	//if (version != TTriGridVelREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	if (version < 2 || version > TTriGridVelREADWRITEVERSION) { printSaveFileVersionError(); return -1; } // broke save files on version 2
	
	if (err = ReadMacValue(bfpb, &fGridBounds)) return err;
	
	if(err = ReadVertices(bfpb,&ptsH,errmsg))goto done;
	if(err = ReadTopology(bfpb,&topH,&velH,errmsg))goto done;
	if(err = ReadIndexedDagTree(bfpb,&treeH,errmsg)) goto done;
	
	numBranches = _GetHandleSize ((Handle) treeH) / sizeof (DAG);
	fDagTree = new TDagTree(ptsH,topH,treeH,velH,numBranches);
	if(!fDagTree)
	{
		printError("Unable to read Triangle Velocity file.");
		goto done;
	}
	if (version > 2)
	{
		if (err = ReadMacValue(bfpb, &numDepths)) goto done;
		if (numDepths > 0)
		{
			fBathymetryH = (FLOATH)_NewHandleClear(sizeof(float)*numDepths);
			if (!fBathymetryH)
			{ TechError("TTriGridVel::Read()", "_NewHandleClear()", 0); goto done; }
		}
		
		for (i = 0 ; i < numDepths ; i++) {
			if (err = ReadMacValue(bfpb, &val)) goto done;
			INDEXH(fBathymetryH, i) = val;
		}
	}
	// fDagTree is now responsible for these handles
	ptsH = 0;
	topH = 0;
	velH = 0;
	treeH = 0;
	
done:
	if(err)
	{
		TechError("TTriGridVel::Read(char* path)", errmsg, 0); 
		if(ptsH) {DisposeHandle((Handle)ptsH); ptsH=0;}
		if(topH) {DisposeHandle((Handle)topH); topH=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(treeH) {DisposeHandle((Handle)treeH); treeH=0;}
		if(fDagTree)
		{
			delete fDagTree;
			fDagTree = 0;
		}
		if(fBathymetryH) {DisposeHandle((Handle)fBathymetryH); fBathymetryH=0;}
	}
	return err;
}


void TTriGridVel::Draw(Rect r, WorldRect view,WorldPoint refP,double refScale,double arrowScale,
					   Boolean bDrawArrows, Boolean bDrawGrid)
{
	short row, col, pixX, pixY;
	float inchesX, inchesY;
	Point p, p2;
	Rect c;
	WorldPoint wp;
	VelocityRec velocity;
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	LongPoint wp1,wp2,wp3;
	long i,numTri;
	Boolean offQuickDrawPlane = false;
	
	
	if(fDagTree == 0)return;
	
	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();
	numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);
	
	//p.h = SameDifferenceX(refP.pLong);
	//p.v = (r.bottom + r.top) - SameDifferenceY(refP.pLat);
	p = GetQuickDrawPt(refP.pLong, refP.pLat, &r, &offQuickDrawPlane);
	
	// draw the reference point
	if (!offQuickDrawPlane)
	{
		RGBForeColor(&colors[BLUE]);
		MySetRect(&c, p.h - 2, p.v - 2, p.h + 2, p.v + 2);
		PaintRect(&c);
	}
	//RGBForeColor(&colors[BLACK]);
	
	RGBForeColor(&colors[PURPLE]);
	
	
	for (i = 0 ; i< numTri; i++)
	{
		if (bDrawArrows)
		{
			wp1 = (*ptsH)[(*topH)[i].vertex1];
			wp2 = (*ptsH)[(*topH)[i].vertex2];
			wp3 = (*ptsH)[(*topH)[i].vertex3];
			
			wp.pLong = (wp1.h+wp2.h+wp3.h)/3;
			wp.pLat = (wp1.v+wp2.v+wp3.v)/3;
			velocity = GetPatValue(wp);
			
			//p.h = SameDifferenceX(wp.pLong);
			//p.v = (r.bottom + r.top) - SameDifferenceY(wp.pLat);
			p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
			MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
			//			PaintRect(&c);
			
			if (velocity.u != 0 || velocity.v != 0) 
			{
				inchesX = (velocity.u * refScale) / arrowScale;
				inchesY = (velocity.v * refScale) / arrowScale;
				pixX = inchesX * PixelsPerInchCurrent();
				pixY = inchesY * PixelsPerInchCurrent();
				p2.h = p.h + pixX;
				p2.v = p.v - pixY;
				MyMoveTo(p.h, p.v);
				MyLineTo(p2.h, p2.v);
				
				//DrawArrowHead (p, p2, velocity);
				MyDrawArrow(p.h,p.v,p2.h,p2.v);
			}
		}
		
		if (bDrawGrid) DrawTriangle(&r,i,FALSE);	// don't fill triangles
	}
	RGBForeColor(&colors[BLACK]);
	
	return;
}

void TTriGridVel::DrawCurvGridPts(Rect r, WorldRect view)
{
	Point p;
	Rect c;
	LongPointHdl ptsH ;
	long i,numPts;
	Boolean offQuickDrawPlane = false;
	
	if(fDagTree == 0)return;
	
	ptsH = fDagTree->GetPointsHdl();
	numPts = _GetHandleSize((Handle)ptsH)/sizeof(LongPoint);
	
	RGBForeColor(&colors[PURPLE]);
	
	for (i = 0 ; i< numPts; i++)
	{
		p = GetQuickDrawPt((*ptsH)[i].h,(*ptsH)[i].v,&r,&offQuickDrawPlane);
		MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
		PaintRect(&c);
	}
	
	RGBForeColor(&colors[BLACK]);
	
	return;
}

void TTriGridVel::DrawBitMapTriangles(Rect r)
{
	TopologyHdl topH ;
	long i,numTri;
	
	if(fDagTree == 0)return;
	
	topH = fDagTree->GetTopologyHdl();
	numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);
	
	RGBForeColor(&colors[BLACK]);
	for (i = 0 ; i< numTri; i++)
	{		
		DrawTriangle(&r,i,TRUE);	// fill triangles	
	}
	
	return;
}

void TTriGridVel::DrawTriangle(Rect *r,long triNum,Boolean fillTriangle)
{
#ifdef IBM
	POINT points[4];
#else
	PolyHandle poly;
#endif
	long v1,v2,v3;
	Point pt1,pt2,pt3;
	TopologyHdl topH = fDagTree->GetTopologyHdl();
	LongPointHdl ptsH = fDagTree->GetPointsHdl();
	Boolean offQuickDrawPlane;
	
	v1 = (*topH)[triNum].vertex1;
	v2 = (*topH)[triNum].vertex2;
	v3 = (*topH)[triNum].vertex3;
	
	
	pt1 = GetQuickDrawPt((*ptsH)[v1].h,(*ptsH)[v1].v,r,&offQuickDrawPlane);
	pt2 = GetQuickDrawPt((*ptsH)[v2].h,(*ptsH)[v2].v,r,&offQuickDrawPlane);
	pt3 = GetQuickDrawPt((*ptsH)[v3].h,(*ptsH)[v3].v,r,&offQuickDrawPlane);
	
	PenMode(patCopy);
#ifdef MAC
		poly = OpenPoly();
		MyMoveTo(pt1.h,pt1.v);
		MyLineTo(pt2.h,pt2.v);
		MyLineTo(pt3.h,pt3.v);
		MyLineTo(pt1.h,pt1.v);
		ClosePoly();
	
		if(fillTriangle)
			PaintPoly(poly);
		
		FramePoly(poly);
		
		KillPoly(poly);
#else
		points[0] = MakePOINT(pt1.h,pt1.v);
		points[1] = MakePOINT(pt2.h,pt2.v);
		points[2] = MakePOINT(pt3.h,pt3.v);
		points[3] = MakePOINT(pt1.h,pt1.v);
	
	
		if(fillTriangle)
			Polygon(currentHDC,points,4); // code goes here

		Polyline(currentHDC,points,4);
		
#endif
}
