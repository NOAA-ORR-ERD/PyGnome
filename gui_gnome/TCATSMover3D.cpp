#include "TCATSMover3D.h"
#include "DagTreeIO.h"
#include "DagTreePD.h"
#include "CROSS.H"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT TCATSMOVER3D
#endif
#endif


TCATSMover3D::TCATSMover3D (TMap *owner, char *name) : TCATSMover(owner, name)
{
	fDuration=48*3600; //48 hrs as seconds 
	fTimeUncertaintyWasSet =0;

	fGrid = 0;
	fRefinedGrid = 0;	// not using this anymore - the main grid is refined, there may be some old save files that need it though
	SetTimeDep (nil);
	bTimeFileActive = false;
	fEddyDiffusion=0; // JLM 5/20/991e6; // cm^2/sec
	fEddyV0 = 0.1; // JLM 5/20/99
	
	bShowDepthContourLabels = false;
	bShowDepthContours = false;

	memset(&fOptimize,0,sizeof(fOptimize));
	SetClassName (name);
}

void TCATSMover3D::Dispose ()
{
	/*if (fGrid)
	{
		fGrid -> Dispose();
		delete fGrid;
		fGrid = nil;
	}
	
	DeleteTimeDep ();*/
	
	if (fRefinedGrid)
	{
		fRefinedGrid -> Dispose();
		delete fRefinedGrid;
		fRefinedGrid = nil;
	}
		
	TCATSMover::Dispose ();
}

Boolean IsCATS3DFile (char *path)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	strLine [256];
	char	firstPartOfFile [256];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{	// must start with CATS3D
		char * strToMatch = "CATS3D";
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 256);
		if (!strncmp (strLine,strToMatch,strlen(strToMatch)))
			bIsValid = true;
	}
	
	return bIsValid;
}



Boolean TCATSMover3D::OkToAddToUniversalMap()
{
	// only allow this if we have grid with valid bounds
	WorldRect gridBounds;
	if (!fGrid) {
		printError("Error in TCATSMover3D::OkToAddToUniversalMap.");
		return false;
	}
	gridBounds = fGrid -> GetBounds();
	if(EqualWRects(gridBounds,emptyWorldRect)) {
		printError("You cannot create a universal mover from a current file which does not specify the grid's bounds.");
		return false;
	}
	return true;
}



/*OSErr TCATSMover3D::InitMover(TGridVel *grid, WorldPoint p)
 {
 fGrid = grid;
 refP = p;
 refZ = 0;
 scaleType = SCALE_NONE;
 scaleValue = 1.0;
 scaleOtherFile[0] = 0;
 bRefPointOpen = FALSE;
 bUncertaintyPointOpen = FALSE;
 bTimeFileOpen = FALSE;
 bShowArrows = FALSE;
 bShowGrid = FALSE;
 arrowScale = 1;// debra wanted 10, CJ wanted 5, JLM likes 5 too (was 1)
 // CJ wants it back to 1, 4/11/00
 
 this->ComputeVelocityScale();
 
 return 0;
 }*/


//#define TCATSMover3DREADWRITEVERSION 2 //JLM 	// updated with refinedGrid  3/6/02
#define TCATSMover3DREADWRITEVERSION 3 //JLM 	// updated with refinedGrid  3/6/02

OSErr TCATSMover3D::Write (BFPB *bfpb)
{
	long i, version = TCATSMover3DREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	OSErr err = 0;
	char c;
	
	if (err = TCATSMover::Write (bfpb)) return err;
	
	StartReadWriteSequence("TCATSMover3D::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	
	c = fRefinedGrid ? TRUE : FALSE;
	if (err = WriteMacValue(bfpb, c)) return err;
	if (fRefinedGrid)
	{
		id = fRefinedGrid -> GetClassID (); //JLM
		if (err = WriteMacValue(bfpb, id)) return err; //JLM
		err = fRefinedGrid -> Write (bfpb);
	}
	
	if (err = WriteMacValue(bfpb, bShowDepthContourLabels)) return err;
	if (err = WriteMacValue(bfpb, bShowDepthContours)) return err;
	
	return err;
}

OSErr TCATSMover3D::Read(BFPB *bfpb)
{
	long i, version;
	ClassID id;
	OSErr err = 0;
	char c;
	
	if (err = TCATSMover::Read(bfpb)) return err;
	SetShowGrid(bShowGrid);
	StartReadWriteSequence("TCATSMover3D::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TCATSMover3D::Read()", "id != TYPE_CATSMOVER3D", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > TCATSMover3DREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	
	
	if (version > 1)
	{
		if (err = ReadMacValue(bfpb, &c)) return err;
		if (c) 
		{
			if (err = ReadMacValue(bfpb,&id)) return err;
			switch(id)
			{
					//case TYPE_RECTGRIDVEL: fRefinedGrid = new TRectGridVel;break;
					//case TYPE_TRIGRIDVEL: fRefinedGrid = new TTriGridVel;break;
				case TYPE_TRIGRIDVEL3D: fRefinedGrid = new TTriGridVel3D;break;
				default: printError("Unrecognized Grid type in TCATSMover3D::Read()."); return -1;
			}
			fRefinedGrid -> Read (bfpb);
		}
	}
	if (version > 2)
	{
		if (err = ReadMacValue(bfpb, &bShowDepthContourLabels)) return err;
		if (err = ReadMacValue(bfpb, &bShowDepthContours)) return err;
		SetDepthContoursCheckMark(bShowDepthContours);
		SetDepthLegendCheckMark(bShowDepthContourLabels);
	}
	
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr TCATSMover3D::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	/*char ourName[kMaxNameLen];
	 
	 // see if the message is of concern to us
	 this->GetClassName(ourName);
	 
	 if(message->IsMessage(M_SETFIELD,ourName))
	 {
	 double val;
	 char str[256];
	 OSErr err = 0;
	 WorldPoint wp;
	 ////////////////
	 err = message->GetParameterAsDouble("scaleValue",&val);
	 if(!err) this->scaleValue = val; 
	 ////////////////
	 err = message->GetParameterAsDouble("EddyDiffusion",&val);
	 if(!err) this->fEddyDiffusion = val; 
	 ////////////////
	 err = message->GetParameterAsDouble("EddyV0",&val);
	 if(!err) this->fEddyV0 = val; 
	 ////////////////
	 message->GetParameterString("scaleType",str,256);
	 if(str[0]) 
	 {	
	 if(!strcmpnocase(str,"none")) this->scaleType = SCALE_NONE; 
	 else if(!strcmpnocase(str,"constant")) this->scaleType = SCALE_CONSTANT; 
	 else if(!strcmpnocase(str,"othergrid")) this->scaleType = SCALE_OTHERGRID; 
	 }
	 /////////////
	 model->NewDirtNotification();// tell model about dirt
	 }
	 /////////////////////////////////////////////////
	 //  pass on this message to our base class
	 /////////////////////////////////////////////////
	 */
	return TCATSMover::CheckAndPassOnMessage(message);
}

/////////////////////////////////////////////////
/*long TCATSMover3D::GetListLength()
 {
 long count = 1;
 
 if (bOpen) {
 count += 4;		// minimum CATS mover lines
 if (timeDep)count++;
 if (bRefPointOpen) count += 3;
 if(model->IsUncertain())count++;
 if(bUncertaintyPointOpen && model->IsUncertain())count +=6;
 // add 1 to # of time-values for active / inactive
 
 // JLM if (bTimeFileOpen) count += timeDep ? timeDep -> GetNumValues () : 0;
 if (bTimeFileOpen) count += timeDep ? (1 + timeDep -> GetListLength ()) : 0; //JLM, add 1 for the active flag
 }
 
 return count;
 }
 
 ListItem TCATSMover3D::GetNthListItem(long n, short indent, short *style, char *text)
 {
 char *p, latS[20], longS[20], timeS[32],valStr[32];
 DateTimeRec time;
 TimeValuePair pair;
 ListItem item = { this, 0, indent, 0 };
 
 if (n == 0) {
 item.index = I_CATSNAME;
 item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
 //		sprintf(text, "CATS: \"%s\"", className);
 sprintf(text, "Currents: \"%s\"", className);
 if(!bActive)*style = italic; // JLM 6/14/10
 
 return item;
 }
 
 item.indent++;
 
 if (bOpen) {
 
 
 if (--n == 0) {
 item.index = I_CATSACTIVE;
 item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
 strcpy(text, "Active");
 
 return item;
 }
 
 
 if (--n == 0) {
 item.index = I_CATSGRID;
 item.bullet = bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
 sprintf(text, "Show Grid");
 
 return item;
 }
 
 if (--n == 0) {
 item.index = I_CATSARROWS;
 item.bullet = bShowArrows ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
 StringWithoutTrailingZeros(valStr,arrowScale,6);
 sprintf(text, "Show Velocities (@ 1 in = %s m/s)", valStr);
 
 return item;
 }
 
 
 if (--n == 0) {
 item.index = I_CATSREFERENCE;
 item.bullet = bRefPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
 strcpy(text, "Reference Point");
 
 return item;
 }
 
 
 if (bRefPointOpen) {
 if (--n == 0) {
 item.index = I_CATSSCALING;
 //item.bullet = BULLET_DASH;
 item.indent++;
 switch (scaleType) {
 case SCALE_NONE:
 strcpy(text, "No reference point scaling");
 break;
 case SCALE_CONSTANT:
 StringWithoutTrailingZeros(valStr,scaleValue,6);
 sprintf(text, "Scale to: %s ", valStr);
 // units
 if (timeDep)
 strcat(text,"* file value");
 else
 strcat(text,"m/s");
 break;
 case SCALE_OTHERGRID:
 sprintf(text, "Scale to grid: %s", scaleOtherFile);
 break;
 }
 
 return item;
 }
 
 n--;
 
 if (n < 2) {
 item.indent++;
 item.index = (n == 0) ? I_CATSLAT : I_CATSLONG;
 //item.bullet = BULLET_DASH;
 WorldPointToStrings(refP, latS, longS);
 strcpy(text, (n == 0) ? latS : longS);
 
 return item;
 }
 
 n--;
 }
 
 
 if(timeDep)
 {
 if (--n == 0)
 {
 char	timeFileName [kMaxNameLen];
 
 item.index = I_CATSTIMEFILE;
 item.bullet = bTimeFileOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
 timeDep -> GetTimeFileName (timeFileName);
 if (timeDep -> GetFileType() == HYDROLOGYFILE)
 sprintf(text, "Hydrology File: %s", timeFileName);
 else
 sprintf(text, "Tide File: %s", timeFileName);
 if(!bTimeFileActive)*style = italic; // JLM 6/14/10
 return item;
 }
 }
 
 if (bTimeFileOpen && timeDep) {
 
 if (--n == 0)
 {
 item.indent++;
 item.index = I_CATSTIMEFILEACTIVE;
 item.bullet = bTimeFileActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
 strcpy(text, "Active");
 
 return item;
 }
 
 ///JLM  ///{
 // Note: n is one higher than JLM expected it to be 
 // (the CATS mover code is pre-decrementing when checking)
 if(timeDep -> GetListLength () > 0)
 {	// only check against the entries if we have some 
 n--; // pre-decrement
 if (n < timeDep -> GetListLength ()) {
 item.indent++;
 item = timeDep -> GetNthListItem(n,item.indent,style,text);
 // over-ride the objects answer ??  JLM
 // no 10/23/00 
 //item.owner = this; // so the clicks come to me
 //item.index = I_CATSTIMEENTRIES + n;
 //////////////////////////////////////
 //item.bullet = BULLET_DASH;
 return item;
 }
 n -= timeDep -> GetListLength ()-1; // the -1 is to leave the count one higher so they can pre-decrement
 }
 ////}
 
 }
 
 if(model->IsUncertain())
 {
 if (--n == 0) {
 item.index = I_CATSUNCERTAINTY;
 item.bullet = bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
 strcpy(text, "Uncertainty");
 
 return item;
 }
 
 if (bUncertaintyPointOpen) {
 
 if (--n == 0) {
 item.index = I_CATSSTARTTIME;
 //item.bullet = BULLET_DASH;
 item.indent++;
 sprintf(text, "Start Time: %.2f hours",fUncertainStartTime/3600);
 return item;
 }
 
 if (--n == 0) {
 item.index = I_CATSDURATION;
 //item.bullet = BULLET_DASH;
 item.indent++;
 sprintf(text, "Duration: %.2f hours",fDuration/3600);
 return item;
 }
 
 if (--n == 0) {
 item.index = I_CATSDOWNCUR;
 //item.bullet = BULLET_DASH;
 item.indent++;
 sprintf(text, "Down Current: %.2f to %.2f %%",fDownCurUncertainty*100,fUpCurUncertainty*100);
 return item;
 }
 
 if (--n == 0) {
 item.index = I_CATSCROSSCUR;
 //item.bullet = BULLET_DASH;
 item.indent++;
 sprintf(text, "Cross Current: %.2f to %.2f %%",fLeftCurUncertainty*100,fRightCurUncertainty*100);
 return item;
 }
 
 if (--n == 0) {
 item.index = I_CATSDIFFUSIONCOEFFICIENT;
 //item.bullet = BULLET_DASH;
 item.indent++;
 sprintf(text, "Eddy Diffusion: %.2e cm^2/sec",fEddyDiffusion);
 return item;
 }
 
 if (--n == 0) {
 item.index = I_CATSEDDYV0;
 //item.bullet = BULLET_DASH;
 item.indent++;
 sprintf(text, "Eddy V0: %.2e m/sec",fEddyV0);
 return item;
 }
 
 }
 
 }
 }
 
 item.owner = 0;
 
 return item;
 }
 
 Boolean TCATSMover3D::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
 {
 Boolean timeFileChanged = false;
 if (inBullet)
 switch (item.index) {
 case I_CATSNAME: bOpen = !bOpen; return TRUE;
 case I_CATSGRID: bShowGrid = !bShowGrid; 
 model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
 case I_CATSARROWS: bShowArrows = !bShowArrows; 
 model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
 case I_CATSREFERENCE: bRefPointOpen = !bRefPointOpen; return TRUE;
 case I_CATSUNCERTAINTY: bUncertaintyPointOpen = !bUncertaintyPointOpen; return TRUE;
 case I_CATSTIMEFILE: bTimeFileOpen = !bTimeFileOpen; return TRUE;
 case I_CATSTIMEFILEACTIVE: bTimeFileActive = !bTimeFileActive; 
 model->NewDirtNotification(); return TRUE;
 case I_CATSACTIVE:
 bActive = !bActive;
 model->NewDirtNotification(); 
 if (!bActive && bTimeFileActive)
 {
 // deactivate time file if main mover is deactivated
 //					bTimeFileActive = false;
 //					VLUpdate (&objects);
 }
 return TRUE;
 }
 
 if (doubleClick && !inBullet)
 {
 switch(item.index)
 {
 case I_CATSSTARTTIME:
 case I_CATSDURATION:
 case I_CATSDOWNCUR:
 case I_CATSCROSSCUR:
 case I_CATSDIFFUSIONCOEFFICIENT:
 case I_CATSEDDYV0:
 case I_CATSUNCERTAINTY:
 {
 Boolean userCanceledOrErr, uncertaintyValuesChanged=false ;
 CurrentUncertainyInfo info  = this -> GetCurrentUncertaintyInfo();
 userCanceledOrErr = CurrentUncertaintyDialog(&info,mapWindow,&uncertaintyValuesChanged);
 if(!userCanceledOrErr) 
 {
 if (uncertaintyValuesChanged)
 {
 this->SetCurrentUncertaintyInfo(info);
 // code goes here, if values have changed needToReInit in UpdateUncertainty
 this->UpdateUncertaintyValues(model->GetModelTime()-model->GetStartTime());
 }
 }
 return TRUE;
 break;
 }
 default:
 CATSSettingsDialog (this, this -> moverMap, &timeFileChanged);
 return TRUE;
 break;
 }
 }
 
 // do other click operations...
 
 return FALSE;
 }
 
 Boolean TCATSMover3D::FunctionEnabled(ListItem item, short buttonID)
 {
 long i;
 switch (item.index) {
 case I_CATSNAME:
 switch (buttonID) {
 case ADDBUTTON: return FALSE;
 case DELETEBUTTON: return TRUE;
 case UPBUTTON:
 case DOWNBUTTON:
 if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
 switch (buttonID) {
 case UPBUTTON: return i > 0;
 case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
 }
 }
 break;
 }
 
 if (buttonID == SETTINGSBUTTON) return TRUE;
 
 return TCATSMover::FunctionEnabled(item, buttonID);
 }
 
 OSErr TCATSMover3D::SettingsItem(ListItem item)
 {
 // JLM we want this to behave like a double click
 Boolean inBullet = false;
 Boolean doubleClick = true;
 Boolean b = this -> ListClick(item,inBullet,doubleClick);
 return 0;
 }
 
 OSErr TCATSMover3D::DeleteItem(ListItem item)
 {
 if (item.index == I_CATSNAME)
 return moverMap -> DropMover(this);
 
 return 0;
 }
 */
Boolean TCATSMover3D::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	TCATSMover::ListClick(item,inBullet,doubleClick);
	SetShowGrid(bShowGrid);
	
	return true;
}
void TCATSMover3D::Draw(Rect r, WorldRect view)
{	// draw refinedGrid if it exists - for grid, not arrows
	if(fRefinedGrid)
	{
		if (bShowArrows)
			fGrid->Draw(r,view,refP,refScale,arrowScale,bShowArrows,false,fColor);
		else
			fRefinedGrid->Draw(r,view,refP,refScale,arrowScale,false,bShowGrid,fColor);
	}
	else if(fGrid)
		fGrid->Draw(r,view,refP,refScale,arrowScale,bShowArrows,bShowGrid,fColor);
	if (bShowDepthContours) ((TTriGridVel3D*)fGrid)->DrawDepthContours(r,view,bShowDepthContourLabels);
}

/////////////////////////////////////////////////////////////////
OSErr TCATSMover3D::TextRead(char *path, TMap **newMap) 
{
	char s[1024], errmsg[256];
	long i, numPoints, line = 0;
	CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH depths=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds;
	
	TTriGridVel3D *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0;
	
	errmsg[0]=0;
	
	
	if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("TCATSMover3D::TextRead()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); 
	
	MySpinCursor(); 
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTVerticesHeaderLine(s, &numPoints))
	{
		MySpinCursor();
		err = ReadTVerticesBody(f,&line,&pts,&depths,errmsg,numPoints,true);
		if(err) goto done;
	}
	else
	{
		err = -1; 
		goto done;
	}
	MySpinCursor();
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsBoundarySegmentHeaderLine(s,&numBoundarySegs)) // Boundary data from CATS
	{
		MySpinCursor();
		err = ReadBoundarySegs(f,&line,&boundarySegs,numBoundarySegs,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		err = -1; 
		goto done;
	}
	MySpinCursor(); 
	
	if(IsWaterBoundaryHeaderLine(s,&numWaterBoundaries,&numBoundaryPts)) // Boundary types from CATS
	{
		MySpinCursor();
		err = ReadWaterBoundaries(f,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		err = -1; 
		goto done;
	}
	
	if(IsTTopologyHeaderLine(s,&numPoints)) // Topology from CATS
	{
		MySpinCursor();
		err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numPoints,TRUE);
		if(err) goto done;
	}
	/*else
	 {
	 err = -1; 
	 goto done;
	 }*/
	else
	{
		//if (!haveBoundaryData) {err=-1; strcpy(errmsg,"File must have boundary data to create topology"); goto done;}
		//DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage("Making Triangles");
		if (err = maketriangles(&topo,pts,numPoints,boundarySegs,numBoundarySegs))  // use maketriangles.cpp
			err = -1; // for now we require TTopology
		// code goes here, support Galt style ??
		DisplayMessage(0);
		velH = (VelocityFH)_NewHandleClear(sizeof(**velH)*numPoints);
		if(!velH)
		{
			strcpy(errmsg,"Not enough memory.");
			goto done;
		}
		for (i=0;i<numPoints;i++)
		{
			INDEXH(velH,i).u = 0.;
			INDEXH(velH,i).v = 0.;
		}
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATS
	{
		MySpinCursor();
		err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
		if(err) goto done;
	}
	/*else
	 {
	 err = -1; 
	 goto done;
	 }*/
	else
	{
		//DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage("Making Dag Tree");
		tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); // use CATSDagTree.cpp and my_build_list.h
		DisplayMessage(0);
		if (errmsg[0])	
			err = -1; // for now we require TIndexedDagTree
		// code goes here, support Galt style ??
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	// figure out the bounds
	bounds = voidWorldRect;
	long numPts;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		numPts = _GetHandleSize((Handle)pts)/sizeof(LongPoint);
		if(numPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
		}
	}
	
	
	/////////////////////////////////////////////////
	// create the bathymetry map 
	
	//if (waterBoundaries /*&& (this -> moverMap == model -> uMap || fVar.gridType != TWO_D)*/)
	if (waterBoundaries && (this -> moverMap == model -> uMap /*|| fVar.gridType != TWO_D*/))
	{
		PtCurMap *map = CreateAndInitPtCurMap(path,bounds); // the map bounds are the same as the grid bounds
		if (!map) goto done;
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundarySegs);	
		map->SetWaterBoundaries(waterBoundaries);
		
		*newMap = map;
	}
	else
	{
		//err = -1;
		//goto done;
		if (boundarySegs){DisposeHandle((Handle)boundarySegs); boundarySegs=0;}
		if (waterBoundaries){DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
	}
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TCATSMover3D::TextRead()","new TTriGridVel3D" ,err);
		goto done;
	}
	
	fGrid = (TGridVel*)triGrid;
	
	triGrid -> SetBounds(bounds); 
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		printError("Unable to read Triangle Velocity file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	triGrid -> SetDepths(depths);
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	depths = 0; // because fGrid is now responsible for it
	
done:
	
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TCATSMover3D::TextRead");
		printError(errmsg); 
		if(pts)DisposeHandle((Handle)pts);
		if(topo)DisposeHandle((Handle)topo);
		if(velH)DisposeHandle((Handle)velH);
		if(tree.treeHdl)DisposeHandle((Handle)tree.treeHdl);
		if(depths)DisposeHandle((Handle)depths);
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (*newMap) 
		{
			(*newMap)->Dispose();
			delete *newMap;
			*newMap=0;
		}
		if(boundarySegs)DisposeHandle((Handle)boundarySegs);
		if(waterBoundaries)DisposeHandle((Handle)waterBoundaries);
	}
	return err;
}


OSErr TCATSMover3D::ImportGrid(char *path) 
{
	char s[1024], errmsg[256];
	long i, numPoints, line = 0;
	CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH depths=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds;
	
	TTriGridVel3D *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0;
	
	errmsg[0]=0;
	
	
	if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("TCATSMover3D::ImportGrid()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); 
	
	MySpinCursor(); 
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTVerticesHeaderLine(s, &numPoints))
	{
		MySpinCursor();
		err = ReadTVerticesBody(f,&line,&pts,&depths,errmsg,numPoints,true);
		if(err) goto done;
	}
	else
	{
		err = -1; 
		goto done;
	}
	MySpinCursor();
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsBoundarySegmentHeaderLine(s,&numBoundarySegs)) // Boundary data from CATS
	{
		MySpinCursor();
		err = ReadBoundarySegs(f,&line,&boundarySegs,numBoundarySegs,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		err = -1; 
		goto done;
	}
	MySpinCursor(); 
	
	if(IsWaterBoundaryHeaderLine(s,&numWaterBoundaries,&numBoundaryPts)) // Boundary types from CATS
	{
		MySpinCursor();
		err = ReadWaterBoundaries(f,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		err = -1; 
		goto done;
	}
	
	if(IsTTopologyHeaderLine(s,&numPoints)) // Topology from CATS
	{
		MySpinCursor();
		//err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numPoints,TRUE);	// may change to false here
		err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numPoints,FALSE);	// may change to false here
		if(err) goto done;
	}
	else
	{
		err = -1; 
		goto done;
	}
	MySpinCursor(); 
	
	
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATS
	{
		MySpinCursor();
		err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
		if(err) goto done;
	}
	else
	{
		err = -1; 
		goto done;
	}
	MySpinCursor(); 
	
	// figure out the bounds
	bounds = voidWorldRect;
	long numPts;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		numPts = _GetHandleSize((Handle)pts)/sizeof(LongPoint);
		if(numPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
		}
	}
	
	
	/////////////////////////////////////////////////
	// use the original bathymetry map 
	// delete refined grid boundary information
	
	if(boundarySegs)DisposeHandle((Handle)boundarySegs);
	if(waterBoundaries)DisposeHandle((Handle)waterBoundaries);
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TCATSMover3D::ImportGrid()","new TTriGridVel3D" ,err);
		goto done;
	}
	
	if(fRefinedGrid)
	{
		fRefinedGrid ->Dispose();
		delete fRefinedGrid;
		fRefinedGrid = 0;
	}
	fRefinedGrid = triGrid;
	
	triGrid -> SetBounds(bounds); 
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		printError("Unable to read Triangle Velocity file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	triGrid -> SetDepths(depths);
	
	pts = 0;	// because fRefinedGrid is now reponsible for it
	topo = 0; // because fRefinedGrid is now reponsible for it
	tree.treeHdl = 0; // because fRefinedGrid is now reponsible for it
	velH = 0; // because fRefinedGrid is now reponsible for it
	depths = 0; // because fRefinedGrid is now reponsible for it
	
done:
	
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TCATSMover3D::ImportGrid");
		printError(errmsg); 
		if(pts)DisposeHandle((Handle)pts);
		if(topo)DisposeHandle((Handle)topo);
		if(velH)DisposeHandle((Handle)velH);
		if(tree.treeHdl)DisposeHandle((Handle)tree.treeHdl);
		if(depths)DisposeHandle((Handle)depths);
		if(fRefinedGrid)
		{
			fRefinedGrid->Dispose();
			delete fRefinedGrid;
			fRefinedGrid = 0;
		}
		if(boundarySegs)DisposeHandle((Handle)boundarySegs);
		if(waterBoundaries)DisposeHandle((Handle)waterBoundaries);
	}
	return err;
}

OSErr TCATSMover3D::CreateRefinedGrid (Boolean askForFile, char* givenPath, char* givenFileName)
{
	char path[256], s[256], fileName[256];
	Point where;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	TTriGridVel3D *grid = nil;
	OSErr err = 0;
	
	if(askForFile || !givenPath || !givenFileName)
	{
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
					 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);
#else
		where = CenteredDialogUpLeft(M38c);
		sfpgetfile(&where, "",
				   (FileFilterUPP)0,
				   -1, typeList,
				   (DlgHookUPP)0,
				   &reply, M38c,
				   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		if (!reply.good) return 0;
		
		my_p2cstr(reply.fName);
#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
#else
		strcpy(path, reply.fName);
#endif
		
#endif
		strcpy (s, path);
		SplitPathFile (s, fileName);
	}
	else
	{	// don't ask user, we were provided with the path
		strcpy(path,givenPath);
		strcpy(fileName,givenFileName);
	}
	
	if (IsCATS3DFile(path))
	{	
		err = this->ImportGrid(path); 
		if(err) {printError("Error importing refined grid"); return err;}	
		return noErr;
	}
	else
	{
		err = -1; 
		printError("Grid to import must be in a CATS3D file");
		return err;
	}
}
/**************************************************************************************************/
OSErr TCATSMover3D::ReadTopology(char* path, TMap **newMap)
{
	// import PtCur triangle info so don't have to regenerate
	char s[1024], errmsg[256];
	long i, numPoints, numTopoPoints, line = 0, numPts;
	CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH depths=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds = voidWorldRect;
	
	TTriGridVel3D *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0;
	
	errmsg[0]=0;
	
	if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("TCATSMover3D::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	
	MySpinCursor(); // JLM 8/4/99
	if(err = ReadTVertices(f,&line,&pts,&depths,errmsg)) goto done;
	
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		numPts = _GetHandleSize((Handle)pts)/sizeof(LongPoint);
		if(numPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
		}
	}
	MySpinCursor();
	
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	if(IsBoundarySegmentHeaderLine(s,&numBoundarySegs)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundarySegs>0)
			err = ReadBoundarySegs(f,&line,&boundarySegs,numBoundarySegs,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary segment header line");
		//goto done;
		// not needed for 2D files, but we require for now
	}
	MySpinCursor(); // JLM 8/4/99
	
	if(IsWaterBoundaryHeaderLine(s,&numWaterBoundaries,&numBoundaryPts)) // Boundary types from CATs
	{
		MySpinCursor();
		err = ReadWaterBoundaries(f,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Water boundaries header line");
		//goto done;
		// not needed for 2D files, but we require for now
	}
	MySpinCursor(); // JLM 8/4/99
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTTopologyHeaderLine(s,&numTopoPoints)) // Topology from CATs
	{
		MySpinCursor();
		err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numTopoPoints,FALSE);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		err = -1; // for now we require TTopology
		strcpy(errmsg,"Error in topology header line");
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATs
	{
		MySpinCursor();
		err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
		if(err) goto done;
	}
	else
	{
		err = -1; // for now we require TIndexedDagTree
		strcpy(errmsg,"Error in dag tree header line");
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	/////////////////////////////////////////////////
	// if the boundary information is in the file we'll need to create a bathymetry map (required for 3D)
	
	if (waterBoundaries && (this -> moverMap == model -> uMap))
	{
		//PtCurMap *map = CreateAndInitPtCurMap(fVar.userName,bounds); // the map bounds are the same as the grid bounds
		PtCurMap *map = CreateAndInitPtCurMap("Extended Topology",bounds); // the map bounds are the same as the grid bounds
		if (!map) {strcpy(errmsg,"Error creating ptcur map"); goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundarySegs);	
		map->SetWaterBoundaries(waterBoundaries);
		
		*newMap = map;
	}
	
	//if (!(this -> moverMap == model -> uMap))	// maybe assume rectangle grids will have map?
	else	// maybe assume rectangle grids will have map?
	{
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
	}
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TCATSMover3D::ReadTopology()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(bounds); 
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		printError("Unable to read Extended Topology file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(depths);
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	//depths = 0;
	
done:
	
	if(depths) {DisposeHandle((Handle)depths); depths=0;}
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TCATSMover3D::ReadTopology");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (*newMap) 
		{
			(*newMap)->Dispose();
			delete *newMap;
			*newMap=0;
		}
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
	}
	return err;
}

OSErr TCATSMover3D::ExportTopology(char* path)
{
	// export triangle info so don't have to regenerate each time
	OSErr err = 0;
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0;
	long i, n, v1,v2,v3,n1,n2,n3;
	double x,y;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	DAGHdl treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0;
	VelocityRec vel;
	BFPB bfpb;
	
	triGrid = (TTriGridVel*)(this->fGrid);
	if (!triGrid) {printError("There is no topology to export"); return -1;}
	dagTree = triGrid->GetDagTree();
	if (dagTree) 
	{
		ptsH = dagTree->GetPointsHdl();
		topH = dagTree->GetTopologyHdl();
		treeH = dagTree->GetDagTreeHdl();
	}
	else 
	{
		printError("There is no topology to export");
		return -1;
	}
	if(!ptsH || !topH || !treeH) 
	{
		printError("There is no topology to export");
		return -1;
	}
	
	if (moverMap->IAm(TYPE_PTCURMAP))
	{
		boundaryTypeH = /*CHECK*/(dynamic_cast<PtCurMap *>(moverMap))->GetWaterBoundaries();
		boundarySegmentsH = /*CHECK*/(dynamic_cast<PtCurMap *>(moverMap))->GetBoundarySegs();
		if (!boundaryTypeH || !boundarySegmentsH) {printError("No map info to export"); err=-1; goto done;}
	}
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	
	// Write out values
	nver = _GetHandleSize((Handle)ptsH)/sizeof(**ptsH);
	//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
	sprintf(hdrStr,"Vertices\t%ld\n",nver);	// total vertices
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	sprintf(hdrStr,"%ld\t%ld\n",nver,nver);	// junk line
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i=0;i<nver;i++)
	{	
		x = (*ptsH)[i].h/1000000.0;
		y =(*ptsH)[i].v/1000000.0;
		//sprintf(topoStr,"%ld\t%lf\t%lf\t%lf\n",i+1,x,y,(*gDepths)[i]);	
		//sprintf(topoStr,"%ld\t%lf\t%lf\n",i+1,x,y);
		sprintf(topoStr,"%lf\t%lf\n",x,y);	// add depths 
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
	if (boundarySegmentsH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundarySegmentsH)/sizeof(long);
		//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
		sprintf(hdrStr,"BoundarySegments\t%ld\n",nBoundarySegs);	// total vertices
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundarySegs;i++)
		{	
			//sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]); // when reading in subtracts 1
			sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]+1);
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}
	if (boundaryTypeH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundaryTypeH)/sizeof(long);	// should be same size as previous handle
		//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
		for(i=0;i<nBoundarySegs;i++)
		{	
			if ((*boundaryTypeH)[i]==2) nWaterBoundaries++;
		}
		sprintf(hdrStr,"WaterBoundaries\t%ld\t%ld\n",nWaterBoundaries,nBoundarySegs);	
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundarySegs;i++)
		{	
			if ((*boundaryTypeH)[i]==2)
				//sprintf(topoStr,"%ld\n",(*boundaryTypeH)[i]);
			{
				sprintf(topoStr,"%ld\n",i);
				strcpy(buffer,topoStr);
				if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			}
		}
	}
	numTriangles = _GetHandleSize((Handle)topH)/sizeof(**topH);
	sprintf(hdrStr,"Topology\t%ld\n",numTriangles);
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i = 0; i< numTriangles;i++)
	{
		v1 = (*topH)[i].vertex1;
		v2 = (*topH)[i].vertex2;
		v3 = (*topH)[i].vertex3;
		n1 = (*topH)[i].adjTri1;
		n2 = (*topH)[i].adjTri2;
		n3 = (*topH)[i].adjTri3;
		dagTree->GetVelocity(i,&vel);
		sprintf(topoStr, "%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%lf\t%lf\n",
				v1, v2, v3, n1, n2, n3, vel.u, vel.v);
		
		/////
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
	numBranches = _GetHandleSize((Handle)treeH)/sizeof(**treeH);
	sprintf(hdrStr,"DAGTree\t%ld\n",dagTree->fNumBranches);
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	
	for(i = 0; i<dagTree->fNumBranches; i++)
	{
		sprintf(topoStr,"%ld\t%ld\t%ld\n",(*treeH)[i].topoIndex,(*treeH)[i].branchLeft,(*treeH)[i].branchRight);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		printError("Error writing topology");
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}
