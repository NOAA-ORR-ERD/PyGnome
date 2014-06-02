
#include "Cross.h"
#include "MapUtils.h"
#include "GenDefs.h"
#include "GridVel.h"
#include "NetCDFMover.h"
#include "NetCDFMoverCurv.h"
#include "NetCDFWindMover.h"
#include "NetCDFMoverTri.h"
#include "NetCDFWindMoverCurv.h"
#include "TideCurCycleMover.h"
#include "Contdlg.h"
#include "ObjectUtilsPD.h"
//#include "TriCurMover.h"

/////////////////////////////////////////////////
static short probH, sizeH;
DropletInfoRecH gDropletInfoHdl = 0;
DropletInfoRecH sDropletInfoH = 0;


void DisposeDialogDataHdl(void)
{
	if (gDropletInfoHdl) {
		if(gDropletInfoHdl != sDropletInfoH)
		{	// only dispose of this if it is different
			DisposeHandle((Handle)gDropletInfoHdl); 
		}
		gDropletInfoHdl = nil;
	}
}

OSErr PrintDropletSizesToFile(void)
{
	char buffer[512],dropSizeStr[64],probStr[64];
	char path[256],hdrStr[256];
	char* suggestedFileName = "DropletSizes.dat";
	double dropletSize,probability;
	DropletInfoRec data;
	long numOutputValues,i;
	BFPB bfpb;
	OSErr err = 0;

	// use the global stuff

	//if (!gDropletInfoHdl)	// already checked this


	err = AskUserForSaveFilename(suggestedFileName,path,".DAT",TRUE);
	if(err) return err; // note: might be user cancel


	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }


	// Write out the values
	// add header line
	//sprintf(hdrStr,"Oil Type = %s",gOilName);
	//sprintf(hdrStr,"Droplet Size and Probability");
	//strcpy(buffer,hdrStr);
	//strcat(buffer,NEWLINESTRING);
	//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	//strcpy(buffer,"Day Mo Yr Hr Min\t\t\tEvap\tDisp");
	strcpy(buffer,"      Dropsize          Probability");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	numOutputValues = _GetHandleSize((Handle)gDropletInfoHdl)/sizeof(DropletInfoRec);
	for(i = 0; i< numOutputValues;i++)
	{
		data = INDEXH(gDropletInfoHdl,i);
		dropletSize = data.dropletSize;
		probability = data.probability;

		sprintf(dropSizeStr,"%3.2f",dropletSize);
		//StringWithoutTrailingZeros(evapStr,evap,3);
		sprintf(probStr,"%3.2f",probability);
		//StringWithoutTrailingZeros(dispStr,disp,3);
		/////
		strcpy(buffer,dropSizeStr);
		strcat(buffer,"		");
		//strcat(buffer,"\t");
		strcat(buffer,probStr);
		strcat(buffer,"		");
		//strcat(buffer,"\t");
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}

done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		// the user has already been told there was a problem
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	if (err) printError("Error saving droplet data to a file");
	return err;

}

OSErr ReadDropletSizeFile(char *path,DropletInfoRecH *dropletData)
{	
	char s[512],s2[512],probStr[64],dropSizeStr[64];
	long i,numValues,numLines,numScanned;
	CHARH f;
	DropletInfoRecH dropletDataH = 0;
	DropletInfoRec dropletDataLine = {0,0.};
	float dropletSize,probability;
	OSErr scanErr;
	OSErr err = noErr;
	long numDataLines;
	long numHeaderLines = 1;	
	
	if (!path) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f))
		{ TechError("ReadDropletSizeFile()", "ReadFileContents()", 0); goto done; }
	
	numLines = NumLinesInText(*f);
	
	numDataLines = numLines - numHeaderLines;
			
	*dropletData = 0;
	dropletDataH = (DropletInfoRecH)_NewHandleClear(sizeof(DropletInfoRec)*numDataLines);
	//dropletDataH = (DropletInfoRecH)_NewHandle(sizeof(DropletInfoRec)*(numDataLines+1));
	if(!dropletDataH)
		{ err = -1; TechError("ReadDropletSizeFile()", "_NewHandle()", 0); goto done; }
	numValues = 0;
	//INDEXH(dropletDataH, numValues++) = adiosDataLine;	// should start with zero evap/disp at time zero
	for (i = 0 ; i < numLines ; i++) {
		NthLineInTextOptimized(*f, i, s, 512); 
		//may want to scan header lines
		if(i < numHeaderLines)
		{
			if (i==0) 
			{
				numScanned=sscanf(s, "%s %s", dropSizeStr, probStr);
				if (numScanned!=2)	
				{ err = -1; TechError("ReadDropletSizeFile()", "sscanf() == 2", 0); goto done; }
				if (strncmpnocase (probStr, "probability", strlen("probability"))) {err = -1; printError("File is not in the correct format"); goto done;}
			}
			continue; // skip any header lines
		}
		if(i%200 == 0) MySpinCursor(); 
		RemoveLeadingAndTrailingWhiteSpace(s);
		if(s[0] == 0) continue; // it's a blank line, allow this and skip the line
		//RemoveSetFromString(s, "-,", s2);
	
		numScanned=sscanf(s, "%f %f", &dropletSize, &probability);
		if (numScanned!=2)	
			{ err = -1; TechError("ReadDropletSizeFile()", "sscanf() == 2", 0); goto done; }
		
		dropletDataLine.probability = probability;
		dropletDataLine.dropletSize = dropletSize;
		// check that all probabilities total 100? or file has bins ?
		INDEXH(dropletDataH, numValues++) = dropletDataLine;
	}
	if(numValues > 0)
	{
		long actualSize = numValues*sizeof(DropletInfoRec); 
		_SetHandleSize((Handle)dropletDataH,actualSize);
		err = _MemError();
	}
	else {
		printError("No lines were found");
		err = true;
		goto done;
	}

	*dropletData = dropletDataH;

done:
	if(f) {DisposeHandle((Handle)f); f = 0;}
	if (err && dropletDataH) {DisposeHandle((Handle)dropletDataH); dropletDataH = 0;}
	
	return err;
	
}

long GetNumDropletSizes(void)
{
	long numInHdl = 0;
	if (gDropletInfoHdl) numInHdl = _GetHandleSize((Handle)gDropletInfoHdl)/sizeof(**gDropletInfoHdl);
	
	return numInHdl;
}
/////////////////////////////////////////////////
static void DropletTableDraw(DialogPtr d, Rect *rectPtr,long i)
{
#pragma unused (rectPtr)
	short		leftOffset = 5, botOffset = 2;
	Point		p;
	short		v;
	Rect		rgbrect;
	char 		sizeStr[128], probStr[128];
	//float prob[7] = {.056,.147,.267,.414,.586,.782,1.};

	GetPen(&p);

	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
	rgbrect=GetDialogItemBox(d,DROPLET_USERITEM);
	v = p.v;
	//sprintf(probStr,"%5.3f",prob[i]);
	sprintf(probStr,"%5.3f",(*gDropletInfoHdl)[i].probability);
	//MyMoveTo(evapH,v); drawstring(numStr);
	MyMoveTo(probH-stringwidth(probStr)/2,v); drawstring(probStr);
	//sprintf(sizeStr,"%ld",(i+1)*10);
	sprintf(sizeStr,"%5.1f",(*gDropletInfoHdl)[i].dropletSize);
	//MyMoveTo(dispH,v); drawstring(numStr);
	MyMoveTo(sizeH-stringwidth(sizeStr)/2,v); drawstring(sizeStr);
	MyMoveTo(rgbrect.left,v+botOffset); MyLineTo(rgbrect.right,v+botOffset);
	
	return;
}

static void DropletTableInit(DialogPtr d, VLISTPTR L)
{
#pragma unused (L)
	Rect r;
	short IBMoffset;

	//gDropletInfoHdl = sDropletInfoH;
	r = GetDialogItemBox(d,DROPLET_USERITEM);
#ifdef IBM
	IBMoffset = r.left;
#else 
	IBMoffset = 0;
#endif
	r = GetDialogItemBox(d,DROPLET_PROBABILITYTITLE);
	probH = (r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(d,DROPLET_SIZETITLE);
	sizeH  = (r.left+r.right)/2-IBMoffset;
	//ShowHideDialogItem(d, DROPLET_CANCEL, false); 	// for now no printing to file

	return;
}

 
	
static Boolean DropletTableClick(DialogPtr d,VLISTPTR L,short itemHit,long *item,Boolean doubleClick)
{
	OSErr err = 0;
	char path[256];
	if(doubleClick)
	{
		*item = DROPLET_OK;
		return true;
	}
	switch(itemHit)
	{
		case DROPLET_OK:
			if (/*gDropletInfoHdl && */(gDropletInfoHdl != sDropletInfoH)) 
			{
				if (sDropletInfoH)
				{
					DisposeHandle((Handle)sDropletInfoH); 
				}
				sDropletInfoH = gDropletInfoHdl;
			}
			return DROPLET_OK;
		case DROPLET_CANCEL:
			if (gDropletInfoHdl && (gDropletInfoHdl != sDropletInfoH))
			{
				DisposeHandle((Handle)gDropletInfoHdl); 
				gDropletInfoHdl = 0;
			}
			return DROPLET_CANCEL;
		case DROPLET_SAVE:
			PrintDropletSizesToFile();
			break;
		case DROPLET_IMPORT:
			if (err = GetFilePath(path)) break;	// might want to check if file is the right type
			DisposeDialogDataHdl();
			err =  ReadDropletSizeFile(path,&gDropletInfoHdl);
			VLReset(L,GetNumDropletSizes()); 
			break;
		case DROPLET_DEFAULTS:
			DisposeDialogDataHdl();
			if (!gDropletInfoHdl) {gDropletInfoHdl = (DropletInfoRecH)_NewHandleClear(0);}
			if(!gDropletInfoHdl){TechError("DropletSizeClick()", "_NewHandle()", 0); err = memFullErr; break;}
			err = SetDefaultDropletSizes(gDropletInfoHdl);
			VLReset(L,GetNumDropletSizes()); 
			break;
		default:
			return false;
			break;
	}
	return 0;
}


short DropletSizeTable(DropletInfoRecH *dropletSizeInfo)		
{
	short ditem;
	long selitem;
	
	if(!dropletSizeInfo)
	{
		printError("No droplet data.");
		return 0;
	}
	sDropletInfoH = *dropletSizeInfo;
	gDropletInfoHdl = (DropletInfoRecH)_NewHandleClear(0);	// so don't lose original data if user cancels

	if(sDropletInfoH)
	{
		gDropletInfoHdl = sDropletInfoH;
		if(_HandToHand((Handle *)&gDropletInfoHdl))
		{
			printError("Not enough memory to create temporary droplet handle");
			return -1;
		}
	}

	selitem=SelectFromVListDialog(
				1355,
				DROPLET_USERITEM,
				GetNumDropletSizes(),
				DropletTableInit,
				nil,
				nil,
				DropletTableDraw,
				DropletTableClick,
				true,
				&ditem);

	if (ditem == DROPLET_OK) *dropletSizeInfo = sDropletInfoH;
	return selitem;
}
/////////////////////////////////////////////////
// Oiled shoreline table code
/////////////////////////////////////////////////

static short segNoH, startPtH, endPtH, numLEsH, segLenH, galOnSegH, galPerMileH, galPerFootH;
OiledShorelineDataHdl gOiledShorelineHdl=0; 
 
OSErr PrintOiledShorelineTableToFile(void)
{
	char path[256];
	OSErr err = 0;
//	char* suggestedFileName = "Oiled Shoreline.dat";
	PtCurMap *map = GetPtCurMap();
	if (!map) return -1;
	//TTriGridVel3D* triGrid = map->GetGrid(true);	// used refined grid if there is one	
	//if (!triGrid) return -1; 

	//err = AskUserForSaveFilename(suggestedFileName,path,".DAT",TRUE);
	//if(err) return err; // note: might be user cancel

	if (gOiledShorelineHdl)  err = map->ExportOiledShorelineData(gOiledShorelineHdl);
	//if (gOilConcHdl) err = triGrid ->ExportOilConcHdl(path);
	//else if (areaHdl) err = triGrid ->ExportTriAreaHdl(path, map->GetNumContourLevels());
	goto done;

done:
	if (err && err!=USERCANCEL) printError("Error saving table data to a file");
	return err;
}

static void OiledShorelineTableDraw(DialogPtr d, Rect *rectPtr,long i)
{
#pragma unused (rectPtr)
	short leftOffset = 0, botOffset = 2, v;
	Point p;
	Rect rgbrect;
	char numStr[128];
	long segNo, startPt, endPt, numBeachedLEs;
	float segmentLengthInMiles, segmentLengthInFeet, gallonsOnSegment;
	OiledShorelineData data;

	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
	rgbrect=GetDialogItemBox(d,OILEDSHORELINETABLE_USERITEM);

#ifdef IBM
	leftOffset = 25;
#else
	leftOffset = 0;
#endif

	GetPen(&p);
	v = p.v;

	data = INDEXH(gOiledShorelineHdl,i);

	segNo = data.segNo;
	startPt = data.startPt;
	endPt = data.endPt;
	numBeachedLEs = data.numBeachedLEs;
	segmentLengthInMiles = data.segmentLengthInKm / MILESTOKILO;
	segmentLengthInFeet = segmentLengthInMiles * MILESTOFEET;
	gallonsOnSegment = data.gallonsOnSegment;

	sprintf(numStr,"%ld",segNo);
	MyMoveTo(segNoH-leftOffset,v); drawstring(numStr);
	sprintf(numStr,"%ld",startPt);
	MyMoveTo(startPtH-leftOffset,v); drawstring(numStr);
	sprintf(numStr,"%ld",endPt);
	MyMoveTo(endPtH-leftOffset,v); drawstring(numStr);
	sprintf(numStr,"%ld",numBeachedLEs);
	MyMoveTo(numLEsH-leftOffset,v); drawstring(numStr);
	sprintf(numStr,"%.3f",segmentLengthInMiles);
	MyMoveTo(segLenH-leftOffset,v); drawstring(numStr);
	sprintf(numStr,"%.3f",gallonsOnSegment);
	MyMoveTo(galOnSegH-leftOffset,v); drawstring(numStr);
	sprintf(numStr,"%.3f",gallonsOnSegment/segmentLengthInMiles);
	MyMoveTo(galPerMileH-leftOffset,v); drawstring(numStr);
	sprintf(numStr,"%.3f",gallonsOnSegment/segmentLengthInFeet);
	MyMoveTo(galPerFootH-leftOffset,v); drawstring(numStr);

	MyMoveTo(rgbrect.left,v+botOffset); MyLineTo(rgbrect.right,v+botOffset);
	
	return;
}

static void OiledShorelineTableInit(DialogPtr d, VLISTPTR L)
{
#pragma unused (L)
	Rect r;

	mysetitext(d,OILEDSHORELINETABLE_SEGNOTITLE,"SegNo");
	mysetitext(d,OILEDSHORELINETABLE_STARTPTTITLE,"StartPt");
	mysetitext(d,OILEDSHORELINETABLE_ENDPTTITLE,"EndPt");
	mysetitext(d,OILEDSHORELINETABLE_NUMLESTITLE,"numLEs");
	mysetitext(d,OILEDSHORELINETABLE_MILESTITLE,"Miles");
	mysetitext(d,OILEDSHORELINETABLE_GALLONSTITLE,"gallons");
	mysetitext(d,OILEDSHORELINETABLE_GALPERMILETITLE,"gals/mile");
	mysetitext(d,OILEDSHORELINETABLE_GALPERFOOTTITLE,"gals/foot");

	r = GetDialogItemBox(d,OILEDSHORELINETABLE_SEGNOTITLE);
	segNoH = r.left;
	r = GetDialogItemBox(d,OILEDSHORELINETABLE_STARTPTTITLE);
	startPtH  = r.left;
	r = GetDialogItemBox(d,OILEDSHORELINETABLE_ENDPTTITLE);
	endPtH = r.left;
	r = GetDialogItemBox(d,OILEDSHORELINETABLE_NUMLESTITLE);
	numLEsH = r.left;
	r = GetDialogItemBox(d,OILEDSHORELINETABLE_MILESTITLE);
	segLenH  = r.left;
	r = GetDialogItemBox(d,OILEDSHORELINETABLE_GALLONSTITLE);
	galOnSegH = r.left;
	r = GetDialogItemBox(d,OILEDSHORELINETABLE_GALPERMILETITLE);
	galPerMileH = r.left;
	r = GetDialogItemBox(d,OILEDSHORELINETABLE_GALPERFOOTTITLE);
	galPerFootH = r.left;

	return;
} 
	
static Boolean OiledShorelineTableClick(DialogPtr d,VLISTPTR L,short itemHit,long *item,Boolean doubleClick)
{
	if(doubleClick)
	{
		*item = OILEDSHORELINETABLE_OK;
		return true;
	}

	switch(itemHit)
	{
		case OILEDSHORELINETABLE_OK:
			return true;
		case OILEDSHORELINETABLE_SAVETOFILE:
			PrintOiledShorelineTableToFile();
			return false;
			break;
		default:
			return false;
			break;
	}
	return 0;
}

short OiledShorelineTable(OiledShorelineDataHdl oiledShorelineHdl)		
{
	short			ditem;
	long			selitem;
	
	if(!oiledShorelineHdl)
	{
		printError("No shoreline oiling data.");
		return 0;
	}
	gOiledShorelineHdl = oiledShorelineHdl;

	selitem=SelectFromVListDialog(
				1390,
				OILEDSHORELINETABLE_USERITEM,
				_GetHandleSize((Handle)oiledShorelineHdl)/sizeof(OiledShorelineData),
				OiledShorelineTableInit,
				nil,
				nil,
				OiledShorelineTableDraw,
				OiledShorelineTableClick,
				true,
				&ditem);
	return selitem;
}
/////////////////////////////////////////////////

/**************************************************************************************************/
OSErr AddPtCurMap(char *path, WorldRect bounds)
{
	char 		nameStr[256];
	OSErr		err = noErr;
	PtCurMap 	*map = nil;
	char fileName[256],s[256];
	strcpy(s,path);
	SplitPathFile (s, fileName);

	strcpy (nameStr, "BathymetryMap: ");
	strcat (nameStr, fileName);

	map = new PtCurMap(nameStr, bounds);
	if (!map)
		{ TechError("AddPtCurMap()", "new PtCurMap()", 0); return -1; }

	if (err = map->InitMap()) { delete map; return err; }

	if (err = model->AddMap(map, 0))
		{ map->Dispose(); delete map; map=0; return -1; }
	else {
		model->NewDirtNotification();
	}

	return err;
}


///////////////////////////////////////////////////////////////////////////

PtCurMap::PtCurMap(char* name, WorldRect bounds) : TMap(name, bounds)
{
	fBoundarySegmentsH = 0;
	fBoundaryTypeH = 0;
	fBoundaryPointsH = 0;
	fSegSelectedH = 0;
	fSelectedBeachHdl = 0;	//not sure if both are needed
	fSelectedBeachFlagHdl = 0;
#ifdef MAC
	memset(&fWaterBitmap,0,sizeof(fWaterBitmap)); //JLM
	memset(&fLandBitmap,0,sizeof(fLandBitmap)); //JLM
#else
	fWaterBitmap = 0;
	fLandBitmap = 0;
#endif
	bDrawLandBitMap = false;	// combined option for now
	bDrawWaterBitMap = false;

	bShowSurfaceLEs = true;
	bShowLegend = true;
	
	bDrawContours = true;

	fContourDepth1 = 0;
	fContourDepth2 = 0;
	fBottomRange = 1.;
	fContourLevelsH = 0;
	
	fDropletSizesH = 0;
	
	memset(&fLegendRect,0,sizeof(fLegendRect)); 
	
	fWaterDensity = 1020;
	fMixedLayerDepth = 10.;	//meters
	fBreakingWaveHeight = 1.;	// meters
	fDiagnosticStrType = 0;
	
	fTriAreaArray = 0;
	fDepthSliceArray = 0;

	bUseSmoothing = false;
	//bShowElapsedTime = false;
	fMinDistOffshore = 0.;	//km - use bounds to set default
	bUseLineCrossAlgorithm = false;

	fBitMapBounds = bounds;
	fUseBitMapBounds = false;
	bDrawBitMapBounds = false;
	
	fWaveHtInput = 0;	// default compute from wind speed
	
	bTrackAllLayers = false;
}

void PtCurMap::Dispose()
{
	if (fBoundarySegmentsH) {
		DisposeHandle((Handle)fBoundarySegmentsH);
		fBoundarySegmentsH = 0;
	}
	
	if (fBoundaryTypeH) {
		DisposeHandle((Handle)fBoundaryTypeH);
		fBoundaryTypeH = 0;
	}
	
	if (fBoundaryPointsH) {
		DisposeHandle((Handle)fBoundaryPointsH);
		fBoundaryPointsH = 0;
	}
	
	if (fSegSelectedH) {
		DisposeHandle((Handle)fSegSelectedH);
		fSegSelectedH = 0;
	}
	
	if (fSelectedBeachHdl) {
		DisposeHandle((Handle)fSelectedBeachHdl);
		fSelectedBeachHdl = 0;
	}
	
	if (fSelectedBeachFlagHdl) {
		DisposeHandle((Handle)fSelectedBeachFlagHdl);
		fSelectedBeachFlagHdl = 0;
	}
	
	if (fContourLevelsH) {
		DisposeHandle((Handle)fContourLevelsH);
		fContourLevelsH = 0;
	}
	
	if (fDropletSizesH) {
		DisposeHandle((Handle)fDropletSizesH);
		fDropletSizesH = 0;
	}

#ifdef MAC
	DisposeBlackAndWhiteBitMap (&fWaterBitmap);
	DisposeBlackAndWhiteBitMap (&fLandBitmap);
#else
	if(fWaterBitmap) DestroyDIB(fWaterBitmap);
	fWaterBitmap = 0;
	if(fLandBitmap) DestroyDIB(fLandBitmap);
	fLandBitmap = 0;
#endif
	
	if (fTriAreaArray) {delete [] fTriAreaArray; fTriAreaArray=0;}
	if (fDepthSliceArray) {delete [] fDepthSliceArray; fDepthSliceArray=0;}
	TMap::Dispose();
}



void DrawFilledWaterTriangles(void * object,WorldRect wRect,Rect r)
{
	PtCurMap* ptCurMap = (PtCurMap*)object; // typecast
	TTriGridVel* triGrid = 0;	// use GetGrid??
	
	/*PtCurMover* mainPtCurMover = (PtCurMover*) ptCurMap -> GetMover(TYPE_PTCURMOVER);
	if(!mainPtCurMover) return;
	TTriGridVel* triGrid = (TTriGridVel*)(mainPtCurMover -> fGrid);*/
	
	triGrid = ptCurMap->GetGrid(true);	// use refined grid if there is one
	if(triGrid) {
		// draw triangles as filled polygons
		triGrid->DrawBitMapTriangles(r);
	}
	return;
}

static Boolean drawingLandBitMap;
void DrawWideLandSegments(void * object,WorldRect wRect,Rect r)
{
	PtCurMap* ptCurMap = (PtCurMap*)object; // typecast
	
		// draw land boundaries as wide lines
		drawingLandBitMap = TRUE;
		ptCurMap -> DrawBoundaries(r);
		drawingLandBitMap = FALSE;
}
Boolean PtCurMap::InMap(WorldPoint p)
{
	WorldRect ourBounds = this -> GetMapBounds(); 
	//TTriGridVel* triGrid = GetGrid(false);	// don't think need 3D here
	//TDagTree *dagTree = triGrid->GetDagTree();
	
	/*LongPoint lp;
	 lp.h = p.pLong;
	 lp.v = p.pLat;
	 if (dagTree -> WhatTriAmIIn(lp) >= 0) return true;*/
	
	if (!WPointInWRect(p.pLong, p.pLat, &ourBounds))
		return false;
	Boolean onLand = IsBlackPixel(p,ourBounds,fLandBitmap);
	Boolean inWater = IsBlackPixel(p,ourBounds,fWaterBitmap);
	if (onLand || inWater) 
		return true;
	else
		return false;
}

long PtCurMap::GetLandType(WorldPoint p)
{
	// This isn't used at the moment
	WorldRect ourBounds = this -> GetMapBounds(); 
	Boolean onLand = IsBlackPixel(p,ourBounds,fLandBitmap);
	Boolean inWater = IsBlackPixel(p,ourBounds,fWaterBitmap);
	if (onLand) 
		return LT_LAND;
	else if (inWater)
		return LT_WATER;
	else
		return LT_UNDEFINED;
	
}

Boolean PtCurMap::InWater(WorldPoint p)
{
	WorldRect ourBounds = this -> GetMapBounds(); 
	Boolean inWater = false;
	TTriGridVel* triGrid = GetGrid(false);	// don't think need 3D here
	TDagTree *dagTree = triGrid->GetDagTree();
	
	if (!WPointInWRect(p.pLong, p.pLat, &ourBounds)) return false; // off map is not in water
	
	inWater = IsBlackPixel(p,ourBounds,fLandBitmap);
	LongPoint lp;
	lp.h = p.pLong;
	lp.v = p.pLat;
	if (dagTree -> WhatTriAmIIn(lp) >= 0) inWater = true;
	
	return inWater;
}


Boolean PtCurMap::OnLand(WorldPoint p)
{
	WorldRect ourBounds = this -> GetMapBounds(); 
	Boolean onLand = false;
	TTriGridVel* triGrid = GetGrid(false);	// don't think need 3D here
	TDagTree *dagTree = triGrid->GetDagTree();
	
	if (!WPointInWRect(p.pLong, p.pLat, &ourBounds)) return false; // off map is not on land
	
	onLand = IsBlackPixel(p,ourBounds,fLandBitmap);
	
	if (bIAmPartOfACompoundMap) return onLand;	
	
	if (gDispersedOilVersion) return onLand;	
	// code goes here, for narrow channels bitmap is too large and beaches too many LEs but this allows some LEs to cross boundary...
	// maybe let user set parameter instead
	
	/*LongPoint lp;
	 lp.h = p.pLong;
	 lp.v = p.pLat;
	 if (dagTree -> WhatTriAmIIn(lp) >= 0) onLand = false;*/
	
	return onLand;
}


WorldPoint3D	PtCurMap::MovementCheck2 (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed)
{
	// check every pixel along the line it makes on the water bitmap
	// for any non-water point check the land bitmap as well and if it crosses a land boundary
	// force the point to the closest point in the bounds
#ifdef MAC
	BitMap bm = fWaterBitmap;
#else
	HDIB bm = fWaterBitmap;
#endif
	
	// this code is similar to IsBlackPixel
	Rect bounds;
	char* baseAddr= 0;
	long rowBytes;
	long rowByte,bitNum,byteNumber,offset;
	Point fromPt,toPt;
	Boolean isBlack = false;
	
#ifdef MAC
	bounds = bm.bounds;
	rowBytes = bm.rowBytes;
	baseAddr = bm.baseAddr;
#else //IBM
	if(bm)
	{
		LPBITMAPINFOHEADER lpDIBHdr  = (LPBITMAPINFOHEADER)GlobalLock(bm);
		baseAddr = (char*) FindDIBBits((LPSTR)lpDIBHdr);
#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)
		rowBytes = WIDTHBYTES(lpDIBHdr->biBitCount * lpDIBHdr->biWidth);
		MySetRect(&bounds,0,0,lpDIBHdr->biWidth,lpDIBHdr->biHeight);
	}
#endif
	
	Boolean LEsOnSurface = (fromWPt.z == 0 && toWPt.z == 0);
	if (toWPt.z == 0 && !isDispersed) LEsOnSurface = true;
	//Boolean LEsOnSurface = true;
	if (!gDispersedOilVersion) LEsOnSurface = true;	// something went wrong
	//if (bUseLineCrossAlgorithm) return SubsurfaceMovementCheck(fromWPt, toWPt, status);	// dispersed oil GNOME had some diagnostic options
	if(baseAddr)
	{
		// find the point in question in the bitmap
		// determine the pixel in the bitmap we need to look at
		// think of the bitmap as an array of pixels 
		long maxChange;
		WorldPoint3D wp = {0,0,0.};
		WorldRect mapBounds = this->GetMapBounds();
		
		fromPt = WorldToScreenPoint(fromWPt.p,mapBounds,bounds);
		toPt = WorldToScreenPoint(toWPt.p,mapBounds,bounds);
		
		// check the bitmap for each pixel when in range
		// so find the number of pixels change hori and vertically
		maxChange = _max(abs(toPt.h - fromPt.h),abs(toPt.v - fromPt.v));
		
		if(maxChange == 0) {
			// it's the same pixel, there is nothing to do
		}
		else { // maxChange >= 1
			long i;
			double fraction;
			Point pt, prevPt = fromPt;
			WorldPoint3D prevWPt = fromWPt;
			
			// note: there is no need to check the first or final pixel, so i only has to go to maxChange-1 
			//for(i = 0; i < maxChange; i++) 
			for(i = 0; i < maxChange+1; i++) 
			{
				//fraction = (i+1)/(double)(maxChange); 
				fraction = (i)/(double)(maxChange); 
				wp.p.pLat = (1-fraction)*fromWPt.p.pLat + fraction*toWPt.p.pLat;
				wp.p.pLong = (1-fraction)*fromWPt.p.pLong + fraction*toWPt.p.pLong;
				wp.z = (1-fraction)*fromWPt.z + fraction*toWPt.z;
				
				pt = WorldToScreenPoint(wp.p,mapBounds,bounds);
				
				// only check this pixel if it is in range
				// otherwise it is not on our map, hence not our problem
				// so assume it is water and OK
				
				if(bounds.left <= pt.h && pt.h < bounds.right
				   && bounds.top <= pt.v && pt.v < bounds.bottom)
				{
					
#ifdef IBM
					/// on the IBM, the rows of pixels are "upsidedown"
					offset = rowBytes*(long)(bounds.bottom - 1 - pt.v);
					/// on the IBM, for a mono map, 1 is background color,
					isBlack = !BitTst(baseAddr + offset, pt.h);
#else
					offset = (rowBytes*(long)pt.v);
					isBlack = BitTst(baseAddr + offset, pt.h);
#endif
					
					// don't beach LEs that are below the surface, reflect in some way
					if (!isBlack) // checking water bitmap, so not water
					{  // either a land point or outside a water boundary, calling code will check which is the case
						if (LEsOnSurface)
							return wp; 
						else
							// reflect and check z and return, but if not inmap return as is (or return towpt?)
						{	// here return the point and then see if it's on another map, else use toWPt
							if (!InMap(wp.p))
							{
								if(!InMap(toWPt.p))
									return toWPt;
								else
									return wp;
							}
							if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
								goto done;
							return ReflectPoint(fromWPt,toWPt,wp);
						}
					}
					else
					{	// also check if point is on both bitmaps and if so beach it
						Boolean onLand = OnLand(wp.p);	// on the boundary
						if (onLand) 
						{
							if (LEsOnSurface)	
								return wp;
							else
							{
								if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
									goto done;
								return ReflectPoint(fromWPt,toWPt,wp);
							}
						}
					}
					if (abs(pt.h - prevPt.h) == 1 && abs(pt.v - prevPt.v) == 1)
					{	// figure out which pixel was crossed
						
						float xRatio = (float)(wp.p.pLong - mapBounds.loLong) / (float)WRectWidth(mapBounds),
						yRatio = (float)(wp.p.pLat - mapBounds.loLat) / (float)WRectHeight(mapBounds);
						float ptL = bounds.left + RectWidth(bounds) * xRatio;
						float ptB = bounds.bottom - RectHeight(bounds) * yRatio;
						xRatio = (float)(prevWPt.p.pLong - mapBounds.loLong) / (float)WRectWidth(mapBounds);
						yRatio = (float)(prevWPt.p.pLat - mapBounds.loLat) / (float)WRectHeight(mapBounds);
						float prevPtL = bounds.left + RectWidth(bounds) * xRatio;
						float prevPtB = bounds.bottom - RectHeight(bounds) * yRatio;
						float dir = (ptB - prevPtB)/(ptL - prevPtL);
						float testv; 
						
						testv = dir*(_max(prevPt.h,pt.h) - prevPtL) + prevPtB;
						
						if (prevPt.v < pt.v)
						{
							if (ceil(testv) == pt.v)
								prevPt.h = pt.h;
							else if (floor(testv) == pt.v)
								prevPt.v = pt.v;
						}
						else if (prevPt.v > pt.v)
						{
							if (ceil(testv) == prevPt.v)
								prevPt.v = pt.v;
							else if (floor(testv) == prevPt.v)
								prevPt.h = pt.h;
						}
						
						if(bounds.left <= prevPt.h && prevPt.h < bounds.right
						   && bounds.top <= prevPt.v && prevPt.v < bounds.bottom)
						{
							
#ifdef IBM
							/// on the IBM, the rows of pixels are "upsidedown"
							offset = rowBytes*(long)(bounds.bottom - 1 - prevPt.v);
							/// on the IBM, for a mono map, 1 is background color,
							isBlack = !BitTst(baseAddr + offset, prevPt.h);
#else
							offset = (rowBytes*(long)prevPt.v);
							isBlack = BitTst(baseAddr + offset, prevPt.h);
#endif
							
							if (!isBlack) 
							{  // either a land point or outside a water boundary, calling code will check which is the case
								wp.p = ScreenToWorldPoint(prevPt, bounds, mapBounds);		
								if (LEsOnSurface)
									return wp; 
								else
									// reflect and check z and return, but if not inmap return as is (or return towpt?)
								{
									if (!InMap(wp.p))
									{
										if(!InMap(toWPt.p))
											return toWPt;
										else
											return wp;
									}
									if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
										goto done;
									return ReflectPoint(fromWPt,toWPt,wp);
								}
							}
							else
							{	// also check if point is on both bitmaps and if so beach it
								Boolean onLand = OnLand(ScreenToWorldPoint(prevPt, bounds, mapBounds));	// on the boundary
								if (onLand) 
								{
									wp.p = ScreenToWorldPoint(prevPt, bounds, mapBounds);	// update wp.z too
									if (LEsOnSurface)	
										return wp;
									else
									{
										if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
											goto done;
										return ReflectPoint(fromWPt,toWPt,wp);
									}
								}
							}
						}
					}
				}
				prevPt = pt;
				prevWPt = wp;
			}
		}
	}
	
done:
	
#ifdef IBM
	if(bm) GlobalUnlock(bm);
#endif
	
	if (!LEsOnSurface && InMap(toWPt.p)) // if off map let it go
	{	
		//if (toWPt.z < 0)
		//toWPt.z = -toWPt.z;
		//toWPt.z = 0.;
		if (!InVerticalMap(toWPt) || toWPt.z == 0)	// check z is ok, else use original z, or entire fromWPt
		{
			double depthAtPt = DepthAtPoint(toWPt.p);	// check depthAtPt return value
			if (depthAtPt <= 0)
			{
				OSErr err = 0;
				return fromWPt;	// something is wrong, can't force new point into vertical map
			}
			//	if (toWPt.z > depthAtPt) toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
			if (toWPt.z > depthAtPt) 
			{
				if (bUseSmoothing)	// just testing some ideas, probably don't want to do this
				{
					// get depth at previous point, add a kick of horizontal diffusion based on the difference in depth
					// this will flatten out the blips but also takes longer to pass through the area
					double dLong, dLat, horizontalDiffusionCoefficient = 0;
					float rand1,rand2,r,w;
					double horizontalScale = 1, depthAtPrevPt = DepthAtPoint(fromWPt.p);
					WorldPoint3D deltaPoint ={0,0,0.};
					TRandom3D* diffusionMover = model->Get3DDiffusionMover();
					
					if (diffusionMover) horizontalDiffusionCoefficient = diffusionMover->fHorizontalDiffusionCoefficient;
					if (depthAtPrevPt > depthAtPt) horizontalScale = 1 + sqrt(depthAtPrevPt - depthAtPt); // or toWPt.z ?
					//if (depthAtPrevPt > depthAtPt) horizontalScale = sqrt(depthAtPrevPt - depthAtPt); // or toWPt.z ?
					// then recheck if in vertical map and force up
					
					//horizontalDiffusionCoefficient = sqrt(2.*(fHorizontalDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT;
					horizontalDiffusionCoefficient = sqrt(2.*(horizontalDiffusionCoefficient/10000.)*model->GetTimeStep())/METERSPERDEGREELAT;
					if (depthAtPrevPt > depthAtPt) horizontalDiffusionCoefficient *= horizontalScale*horizontalScale;
					//if (depthAtPrevPt > depthAtPt) horizontalDiffusionCoefficient *= horizontalScale;
					GetRandomVectorInUnitCircle(&rand1,&rand2);
					r = sqrt(rand1*rand1+rand2*rand2);
					w = sqrt(-2*log(r)/r);
					//dLong = (rand1 * w * horizontalDiffusionCoefficient )/ LongToLatRatio3 (refPoint.pLat);
					dLong = (rand1 * w * horizontalDiffusionCoefficient )/ LongToLatRatio3 (fromWPt.p.pLat);
					dLat  = rand2 * w * horizontalDiffusionCoefficient;
					
					deltaPoint.p.pLong = dLong * 1000000;
					deltaPoint.p.pLat  = dLat  * 1000000;
					toWPt.p.pLong += deltaPoint.p.pLong;
					toWPt.p.pLat += deltaPoint.p.pLat;
					
					if (!InVerticalMap(toWPt))	// check z is ok, else use original z, or entire fromWPt
					{
						toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
					}	
				}
				else
					toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
			}
			if (toWPt.z <= 0) 
			{
				toWPt.z = GetRandomFloat(.01*depthAtPt,.1*depthAtPt);
			}
			//toWPt.z = fromWPt.z;
			//if (!InVerticalMap(toWPt))	
			//toWPt.p = fromWPt.p;
			//toWPt = fromWPt;
		}
	}
	
	return toWPt;
}

WorldPoint3D	PtCurMap::MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed)
{
	// check every pixel along the line it makes on the water bitmap
	// for any non-water point check the land bitmap as well and if it crosses a land boundary
	// force the point to the closest point in the bounds
#ifdef MAC
	BitMap bm = fWaterBitmap;
#else
	HDIB bm = fWaterBitmap;
#endif
	
	// this code is similar to IsBlackPixel
	Rect bounds;
	char* baseAddr= 0;
	long rowBytes;
	long rowByte,bitNum,byteNumber,offset;
	Point fromPt,toPt;
	Boolean isBlack = false;
	
#ifdef MAC
	bounds = bm.bounds;
	rowBytes = bm.rowBytes;
	baseAddr = bm.baseAddr;
#else //IBM
	if(bm)
	{
		LPBITMAPINFOHEADER lpDIBHdr  = (LPBITMAPINFOHEADER)GlobalLock(bm);
		baseAddr = (char*) FindDIBBits((LPSTR)lpDIBHdr);
#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)
		rowBytes = WIDTHBYTES(lpDIBHdr->biBitCount * lpDIBHdr->biWidth);
		MySetRect(&bounds,0,0,lpDIBHdr->biWidth,lpDIBHdr->biHeight);
	}
#endif
	
	Boolean LEsOnSurface = (fromWPt.z == 0 && toWPt.z == 0);
	if (toWPt.z == 0 && !isDispersed) LEsOnSurface = true;
	//Boolean LEsOnSurface = true;
	if (!gDispersedOilVersion) LEsOnSurface = true;	// something went wrong
	//if (bUseLineCrossAlgorithm) return SubsurfaceMovementCheck(fromWPt, toWPt, status);	// dispersed oil GNOME had some diagnostic options
	if(baseAddr)
	{
		// find the point in question in the bitmap
		// determine the pixel in the bitmap we need to look at
		// think of the bitmap as an array of pixels 
		long maxChange;
		WorldPoint3D wp = {0,0,0.};
		WorldRect mapBounds = this->GetMapBounds();
		
		fromPt = WorldToScreenPoint(fromWPt.p,mapBounds,bounds);
		toPt = WorldToScreenPoint(toWPt.p,mapBounds,bounds);
		
		// check the bitmap for each pixel when in range
		// so find the number of pixels change hori and vertically
		maxChange = _max(abs(toPt.h - fromPt.h),abs(toPt.v - fromPt.v));
		
		if(maxChange == 0) {
			// it's the same pixel, there is nothing to do
		}
		else { // maxChange >= 1
			long i;
			double fraction;
			Point pt, prevPt = fromPt;
			WorldPoint3D prevWPt = fromWPt;
			
			// note: there is no need to check the first or final pixel, so i only has to go to maxChange-1 
			for(i = 0; i < maxChange; i++) 
			{
				fraction = (i+1)/(double)(maxChange); 
				wp.p.pLat = (1-fraction)*fromWPt.p.pLat + fraction*toWPt.p.pLat;
				wp.p.pLong = (1-fraction)*fromWPt.p.pLong + fraction*toWPt.p.pLong;
				wp.z = (1-fraction)*fromWPt.z + fraction*toWPt.z;
				
				pt = WorldToScreenPoint(wp.p,mapBounds,bounds);
				
				// only check this pixel if it is in range
				// otherwise it is not on our map, hence not our problem
				// so assume it is water and OK
				
				if(bounds.left <= pt.h && pt.h < bounds.right
				   && bounds.top <= pt.v && pt.v < bounds.bottom)
				{
					
#ifdef IBM
					/// on the IBM, the rows of pixels are "upsidedown"
					offset = rowBytes*(long)(bounds.bottom - 1 - pt.v);
					/// on the IBM, for a mono map, 1 is background color,
					isBlack = !BitTst(baseAddr + offset, pt.h);
#else
					offset = (rowBytes*(long)pt.v);
					isBlack = BitTst(baseAddr + offset, pt.h);
#endif
					
					// don't beach LEs that are below the surface, reflect in some way
					if (!isBlack) // checking water bitmap, so not water
					{  // either a land point or outside a water boundary, calling code will check which is the case
						if (LEsOnSurface)
							return wp; 
						else
							// reflect and check z and return, but if not inmap return as is (or return towpt?)
						{
							if (!InMap(wp.p))
							{
								if(!InMap(toWPt.p))
									return toWPt;
								else
									return wp;
							}
							if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
								goto done;
							return ReflectPoint(fromWPt,toWPt,wp);
						}
					}
					else
					{	// also check if point is on both bitmaps and if so beach it
						Boolean onLand = OnLand(wp.p);	// on the boundary
						if (onLand) 
						{
							if (LEsOnSurface)	
								return wp;
							else
							{
								if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
									goto done;
								return ReflectPoint(fromWPt,toWPt,wp);
							}
						}
					}
					if (abs(pt.h - prevPt.h) == 1 && abs(pt.v - prevPt.v) == 1)
					{	// figure out which pixel was crossed
						
						float xRatio = (float)(wp.p.pLong - mapBounds.loLong) / (float)WRectWidth(mapBounds),
						yRatio = (float)(wp.p.pLat - mapBounds.loLat) / (float)WRectHeight(mapBounds);
						float ptL = bounds.left + RectWidth(bounds) * xRatio;
						float ptB = bounds.bottom - RectHeight(bounds) * yRatio;
						xRatio = (float)(prevWPt.p.pLong - mapBounds.loLong) / (float)WRectWidth(mapBounds);
						yRatio = (float)(prevWPt.p.pLat - mapBounds.loLat) / (float)WRectHeight(mapBounds);
						float prevPtL = bounds.left + RectWidth(bounds) * xRatio;
						float prevPtB = bounds.bottom - RectHeight(bounds) * yRatio;
						float dir = (ptB - prevPtB)/(ptL - prevPtL);
						float testv; 
						
						testv = dir*(_max(prevPt.h,pt.h) - prevPtL) + prevPtB;
						
						if (prevPt.v < pt.v)
						{
							if (ceil(testv) == pt.v)
								prevPt.h = pt.h;
							else if (floor(testv) == pt.v)
								prevPt.v = pt.v;
						}
						else if (prevPt.v > pt.v)
						{
							if (ceil(testv) == prevPt.v)
								prevPt.v = pt.v;
							else if (floor(testv) == prevPt.v)
								prevPt.h = pt.h;
						}
						
						if(bounds.left <= prevPt.h && prevPt.h < bounds.right
						   && bounds.top <= prevPt.v && prevPt.v < bounds.bottom)
						{
							
#ifdef IBM
							/// on the IBM, the rows of pixels are "upsidedown"
							offset = rowBytes*(long)(bounds.bottom - 1 - prevPt.v);
							/// on the IBM, for a mono map, 1 is background color,
							isBlack = !BitTst(baseAddr + offset, prevPt.h);
#else
							offset = (rowBytes*(long)prevPt.v);
							isBlack = BitTst(baseAddr + offset, prevPt.h);
#endif
							
							if (!isBlack) 
							{  // either a land point or outside a water boundary, calling code will check which is the case
								wp.p = ScreenToWorldPoint(prevPt, bounds, mapBounds);		
								if (LEsOnSurface)
									return wp; 
								else
									// reflect and check z and return, but if not inmap return as is (or return towpt?)
								{
									if (!InMap(wp.p))
									{
										if(!InMap(toWPt.p))
											return toWPt;
										else
											return wp;
									}
									if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
										goto done;
									return ReflectPoint(fromWPt,toWPt,wp);
								}
							}
							else
							{	// also check if point is on both bitmaps and if so beach it
								Boolean onLand = OnLand(ScreenToWorldPoint(prevPt, bounds, mapBounds));	// on the boundary
								if (onLand) 
								{
									wp.p = ScreenToWorldPoint(prevPt, bounds, mapBounds);	// update wp.z too
									if (LEsOnSurface)	
										return wp;
									else
									{
										if (InMap(toWPt.p) && !OnLand(toWPt.p)) 
											goto done;
										return ReflectPoint(fromWPt,toWPt,wp);
									}
								}
							}
						}
					}
				}
				prevPt = pt;
				prevWPt = wp;
			}
		}
	}
	
done:
	
#ifdef IBM
	if(bm) GlobalUnlock(bm);
#endif
	
	if (!LEsOnSurface && InMap(toWPt.p)) // if off map let it go
	{	
		//if (toWPt.z < 0)
		//toWPt.z = -toWPt.z;
		//toWPt.z = 0.;
		if (!InVerticalMap(toWPt) || toWPt.z == 0)	// check z is ok, else use original z, or entire fromWPt
		{
			double depthAtPt = DepthAtPoint(toWPt.p);	// check depthAtPt return value
			if (depthAtPt <= 0)
			{
				OSErr err = 0;
				return fromWPt;	// something is wrong, can't force new point into vertical map
			}
			//	if (toWPt.z > depthAtPt) toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
			if (toWPt.z > depthAtPt) 
			{
				if (bUseSmoothing)	// just testing some ideas, probably don't want to do this
				{
					// get depth at previous point, add a kick of horizontal diffusion based on the difference in depth
					// this will flatten out the blips but also takes longer to pass through the area
					double dLong, dLat, horizontalDiffusionCoefficient = 0;
					float rand1,rand2,r,w;
					double horizontalScale = 1, depthAtPrevPt = DepthAtPoint(fromWPt.p);
					WorldPoint3D deltaPoint ={0,0,0.};
					TRandom3D* diffusionMover = model->Get3DDiffusionMover();
					
					if (diffusionMover) horizontalDiffusionCoefficient = diffusionMover->fHorizontalDiffusionCoefficient;
					if (depthAtPrevPt > depthAtPt) horizontalScale = 1 + sqrt(depthAtPrevPt - depthAtPt); // or toWPt.z ?
					//if (depthAtPrevPt > depthAtPt) horizontalScale = sqrt(depthAtPrevPt - depthAtPt); // or toWPt.z ?
					// then recheck if in vertical map and force up
					
					//horizontalDiffusionCoefficient = sqrt(2.*(fHorizontalDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT;
					horizontalDiffusionCoefficient = sqrt(2.*(horizontalDiffusionCoefficient/10000.)*model->GetTimeStep())/METERSPERDEGREELAT;
					if (depthAtPrevPt > depthAtPt) horizontalDiffusionCoefficient *= horizontalScale*horizontalScale;
					//if (depthAtPrevPt > depthAtPt) horizontalDiffusionCoefficient *= horizontalScale;
					GetRandomVectorInUnitCircle(&rand1,&rand2);
					r = sqrt(rand1*rand1+rand2*rand2);
					w = sqrt(-2*log(r)/r);
					//dLong = (rand1 * w * horizontalDiffusionCoefficient )/ LongToLatRatio3 (refPoint.pLat);
					dLong = (rand1 * w * horizontalDiffusionCoefficient )/ LongToLatRatio3 (fromWPt.p.pLat);
					dLat  = rand2 * w * horizontalDiffusionCoefficient;
					
					deltaPoint.p.pLong = dLong * 1000000;
					deltaPoint.p.pLat  = dLat  * 1000000;
					toWPt.p.pLong += deltaPoint.p.pLong;
					toWPt.p.pLat += deltaPoint.p.pLat;
					
					if (!InVerticalMap(toWPt))	// check z is ok, else use original z, or entire fromWPt
					{
						double distanceBelowBottom = toWPt.z - depthAtPt;
						if (depthAtPt > distanceBelowBottom)
						{
							toWPt.z = depthAtPt - distanceBelowBottom;
						}
						else
							toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
						/*if (depthAtPt > 1.)
							toWPt.z = GetRandomFloat(depthAtPt-1.,.9999999*depthAtPt);
						else
							toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);*/
						//toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
					}	
				}
				else
				{	// instead try toWPt.z = depthAtPt - (toWPt.z - depthAtPt);
					double distanceBelowBottom = toWPt.z - depthAtPt;
					if (depthAtPt > distanceBelowBottom)
					{
						toWPt.z = depthAtPt - distanceBelowBottom;
					}
					else
						toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
					/*if (depthAtPt > 1.)
						toWPt.z = GetRandomFloat(depthAtPt-1.,.9999999*depthAtPt);
					else
						toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);*/
					//toWPt.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
				}
			}
			if (toWPt.z <= 0) 
			{
				if (isDispersed)
					toWPt.z = GetRandomFloat(.01*depthAtPt,.1*depthAtPt);
				else
					toWPt.z = 0;	// let LEs refloat - except for dispersed oil...
			}
			//toWPt.z = fromWPt.z;
			//if (!InVerticalMap(toWPt))	
			//toWPt.p = fromWPt.p;
			//toWPt = fromWPt;
		}
	}
	
	return toWPt;
}


OSErr PtCurMap::MakeBitmaps()
{
	OSErr err = 0;
	TCurrentMover *mover=0;

	{	// clear out bitmaps if they already exist
#ifdef MAC
		DisposeBlackAndWhiteBitMap (&fWaterBitmap);
		DisposeBlackAndWhiteBitMap (&fLandBitmap);
#else
		if(fWaterBitmap) DestroyDIB(fWaterBitmap);
		fWaterBitmap = 0;
		if(fLandBitmap) DestroyDIB(fLandBitmap);
		fLandBitmap = 0;
#endif
	}

	mover = Get3DCurrentMover();	
	if (!mover) return -1;	
		
	{ // make the bitmaps etc
		Rect bitMapRect;
		long bmWidth, bmHeight;
		WorldRect wRect = this -> GetMapBounds();
		err = LandBitMapWidthHeight(wRect,&bmWidth,&bmHeight);
		if (err) goto done;
		MySetRect (&bitMapRect, 0, 0, bmWidth, bmHeight);
		fWaterBitmap = GetBlackAndWhiteBitmap(DrawFilledWaterTriangles,this,wRect,bitMapRect,&err);
		if(err) goto done;
		fLandBitmap = GetBlackAndWhiteBitmap(DrawWideLandSegments,this,wRect,bitMapRect,&err); 
		if(err) goto done;
	
	}
done:	
	if(err)
	{
#ifdef MAC
		DisposeBlackAndWhiteBitMap (&fWaterBitmap);
		DisposeBlackAndWhiteBitMap (&fLandBitmap);
#else
		if(fWaterBitmap) DestroyDIB(fWaterBitmap);
		fWaterBitmap = 0;
		if(fLandBitmap) DestroyDIB(fLandBitmap);
		fLandBitmap = 0;
#endif
	}
	return err;
}

OSErr PtCurMap::AddMover(TMover *theMover, short where)
{
	PtCurMover* mainPtCurMoverB4 = dynamic_cast<PtCurMover*>(this -> GetMover(TYPE_PTCURMOVER));
	PtCurMover* mainPtCurMoverAfter = 0;
	NetCDFMoverCurv* mainNetCDFMoverB4 = dynamic_cast<NetCDFMoverCurv*>( this -> GetMover(TYPE_NETCDFMOVERCURV));
	NetCDFMoverCurv* mainNetCDFMoverAfter = 0;
	NetCDFMoverTri* mainNetCDFTriMoverB4 = dynamic_cast<NetCDFMoverTri*>( this -> GetMover(TYPE_NETCDFMOVERTRI));
	NetCDFMoverTri* mainNetCDFTriMoverAfter = 0;
	TideCurCycleMover* mainTideCurCycleMoverB4 = dynamic_cast<TideCurCycleMover*>( this -> GetMover(TYPE_TIDECURCYCLEMOVER));
	TideCurCycleMover* mainTideCurCycleMoverAfter = 0;
	TCATSMover3D* mainTCATSMover3DB4 = dynamic_cast<TCATSMover3D*>( this -> GetMover(TYPE_CATSMOVER3D));
	TCATSMover3D* mainTCATSMover3DAfter = 0;
	TriCurMover* mainTriCurMoverB4 = dynamic_cast<TriCurMover*> (this -> GetMover(TYPE_TRICURMOVER));
	TriCurMover* mainTriCurMoverAfter = 0;
	//code goes here, must be a better way to do this
	// see code in GnomeBeta
	OSErr err = 0;
	
	err = TMap::AddMover(theMover,where);
	if(err) return err;
	
	mainPtCurMoverAfter = dynamic_cast<PtCurMover*>( this -> GetMover(TYPE_PTCURMOVER));
	mainNetCDFMoverAfter = dynamic_cast<NetCDFMoverCurv*>( this -> GetMover(TYPE_NETCDFMOVERCURV));
	mainNetCDFTriMoverAfter = dynamic_cast<NetCDFMoverTri*>( this -> GetMover(TYPE_NETCDFMOVERTRI));
	mainTideCurCycleMoverAfter = dynamic_cast<TideCurCycleMover*>( this -> GetMover(TYPE_TIDECURCYCLEMOVER));
	mainTCATSMover3DAfter = dynamic_cast<TCATSMover3D*>( this -> GetMover(TYPE_CATSMOVER3D));
	mainTriCurMoverAfter = dynamic_cast<TriCurMover*>( this -> GetMover(TYPE_TRICURMOVER));
	
	if(mainPtCurMoverAfter && !mainPtCurMoverB4 || (!mainPtCurMoverAfter && (mainNetCDFMoverAfter && !mainNetCDFMoverB4))
		|| !mainPtCurMoverAfter && !mainNetCDFMoverAfter && (mainNetCDFTriMoverAfter && !mainNetCDFTriMoverB4) 
		|| !mainPtCurMoverAfter && !mainNetCDFMoverAfter && !mainNetCDFTriMoverAfter && (mainTideCurCycleMoverAfter && !mainTideCurCycleMoverB4)// huh?
		|| !mainPtCurMoverAfter && !mainNetCDFMoverAfter && !mainNetCDFTriMoverAfter && !mainTideCurCycleMoverAfter && (mainTCATSMover3DAfter && !mainTCATSMover3DB4)// huh?
	    || !mainPtCurMoverAfter && !mainNetCDFMoverAfter && !mainNetCDFTriMoverAfter && !mainTideCurCycleMoverAfter && !mainTCATSMover3DAfter && (mainTriCurMoverAfter && !mainTriCurMoverB4))// huh?
	{ // make the bitmaps etc
		Rect bitMapRect;
		long bmWidth, bmHeight;
		WorldRect wRect = this -> GetMapBounds();
		err = LandBitMapWidthHeight(wRect,&bmWidth,&bmHeight);
		if (err) goto done;
		MySetRect (&bitMapRect, 0, 0, bmWidth, bmHeight);
		fWaterBitmap = GetBlackAndWhiteBitmap(DrawFilledWaterTriangles,this,wRect,bitMapRect,&err);
		if(err) goto done;
		fLandBitmap = GetBlackAndWhiteBitmap(DrawWideLandSegments,this,wRect,bitMapRect,&err); 
		if(err) goto done;
	
	}
done:	
	if(err)
	{
#ifdef MAC
		DisposeBlackAndWhiteBitMap (&fWaterBitmap);
		DisposeBlackAndWhiteBitMap (&fLandBitmap);
#else
		if(fWaterBitmap) DestroyDIB(fWaterBitmap);
		fWaterBitmap = 0;
		if(fLandBitmap) DestroyDIB(fLandBitmap);
		fLandBitmap = 0;
#endif
	}
	return err;
}

OSErr PtCurMap::DropMover(TMover *theMover)
{
	long 	i, numMovers;
	OSErr	err = noErr;
	TCurrentMover *mover = 0;
	TMover *thisMover = 0;
	
	if (moverList->IsItemInList((Ptr)&theMover, &i))
	{
		if (err = moverList->DeleteItem(i))
			{ TechError("TMap::DropMover()", "DeleteItem()", err); return err; }
	}
	numMovers = moverList->GetItemCount();
	mover = Get3DCurrentMover();
	if (numMovers==0) err = model->DropMap(this);

	if (!mover)
	{
		for (i = 0; i < numMovers; i++)
		{
			this -> moverList -> GetListItem ((Ptr) &thisMover, 0); // will always want the first item in the list
			if (err = this->DropMover(thisMover)) return err; // gets rid of first mover, moves rest up
		}
		err = model->DropMap(this);
	}
	SetDirty (true);
	
	return err;
}

OSErr PtCurMap::ReplaceMap()	// code goes here, maybe not for NetCDF?
{
	char 		path[256], nameStr [256];
	short 		item, gridType;
	OSErr		err = noErr;
	Point 		where = CenteredDialogUpLeft(M38b);
	PtCurMap 	*map = nil;
	OSType 	typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 	reply;

#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, M38b, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);
#else
	sfpgetfile(&where, "",
			   (FileFilterUPP)0,
			   -1, typeList,
			   (DlgHookUPP)0,
			   &reply, M38b,
			   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	if (!reply.good) return USERCANCEL;

	my_p2cstr(reply.fName);
	#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
	#else
		strcpy(path, reply.fName);
	#endif
#endif
	if (IsPtCurFile (path))
	{
		TMap *newMap = 0;
		TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,"ptcurfile",&newMap);	// already have path
		
		if (newMover)
		{
			PtCurMover *ptCurMover = dynamic_cast<PtCurMover*>(newMover);
			err = ptCurMover -> SettingsDialog();
			if(err)	
			{ 
				newMover->Dispose(); delete newMover; newMover = 0;
				if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} 
			}
	
			if(newMover && !err)
			{
				Boolean timeFileChanged = false;
				if (!newMap) 
				{
					err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
				}
				else
				{
					err = model -> AddMap(newMap, 0);
					if (err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					err = AddMoverToMap(newMap, timeFileChanged, newMover);
					if(err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					newMover->SetMoverMap(newMap);
				}
			}
		}
		map = dynamic_cast<PtCurMap *>(newMap);
	}
	else if (IsNetCDFFile (path, &gridType))
	{
		TMap *newMap = 0;
		char s[256],fileName[256];
		strcpy(s,path);
		SplitPathFile (s, fileName);
		strcat (nameStr, fileName);
		TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,fileName,&newMap);	// already have path
		
		if (newMover && newMap)
		{
			NetCDFMover *netCDFMover = dynamic_cast<NetCDFMover*>(newMover);
			err = netCDFMover -> SettingsDialog();
			if(err)	
			{ 
				newMover->Dispose(); delete newMover; newMover = 0;
				if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} 
			}
	
			if(newMover && !err)
			{
				Boolean timeFileChanged = false;
				if (!newMap) 
				{
					err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
				}
				else
				{
					err = model -> AddMap(newMap, 0);
					if (err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					err = AddMoverToMap(newMap, timeFileChanged, newMover);
					if(err) 
					{
						newMap->Dispose(); delete newMap; newMap =0; 
						newMover->Dispose(); delete newMover; newMover = 0;
						return -1; 
					}
					newMover->SetMoverMap(newMap);
				}
			}
		}
		else
		{
			printError("NetCDF file must include a map.");
			if (newMover) {newMover->Dispose(); delete newMover; newMover = 0;}
			if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} // shouldn't happen
			return USERCANCEL;
		}
		map = dynamic_cast<PtCurMap *>(newMap);
	}
	else 
	{
		printError("New map must be of the same type.");
		return USERCANCEL;	// to return to the dialog
	}
	/*strcpy (nameStr, "BathymetryMap: ");
	strcat (nameStr, (char*) reply.fName);
	
	map = new PtCurMap (nameStr, voidWorldRect);
	if (!map)
		{ TechError("ReplaceMap()", "new PtCurMap()", 0); return -1; }

	//if (err = map->InitMap(path)) { delete map; return err; }
	if (err = map->InitMap()) { delete map; return err; }

	if (err = model->AddMap(map, 0))
		{ map->Dispose(); delete map; return -1; } 
	else*/ 
	{
		// put movers on the new map and activate
		TMover *thisMover = nil;
		Boolean	timeFileChanged = false;
		long k, d = this -> moverList -> GetItemCount ();
		for (k = 0; k < d; k++)
		{
			this -> moverList -> GetListItem ((Ptr) &thisMover, 0); // will always want the first item in the list
			if (!thisMover->IAm(TYPE_PTCURMOVER) && !thisMover->IAm(TYPE_NETCDFMOVERCURV) && !IAm(TYPE_NETCDFMOVERTRI) )
			{
				if (err = AddMoverToMap(map, timeFileChanged, thisMover)) return err; 
				thisMover->SetMoverMap(map);
			}
			if (err = this->DropMover(thisMover)) return err; // gets rid of first mover, moves rest up
		}
		if (err = model->DropMap(this)) return err;
		model->NewDirtNotification();
	}

	return err;
	
}


Boolean PtCurMap::ThereAreTrianglesSelected()
{
	TTriGridVel3D* triGrid = GetGrid3D(true);		// might we use this for 2D?
	if (!triGrid) return false;
	return triGrid->ThereAreTrianglesSelected();
	return false;
}


void  PtCurMap::FindNearestBoundary(Point where, long *verNum, long *segNo)
{
	long startVer = 0,i,jseg;
	WorldPoint wp = ScreenToWorldPoint(where, MapDrawingRect(), settings.currentView);
	WorldPoint wp2;
	LongPoint lp;
	long lastVer = GetNumBoundaryPts();
	//long nbounds = GetNumBoundaries();
	long nSegs = GetNumBoundarySegs();	
	float wdist = LatToDistance(ScreenToWorldDistance(4));
	LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
	if(!ptsHdl) return;
	*verNum= -1;
	*segNo =-1;
	for(i = 0; i < lastVer; i++)
	{
		//wp2 = (*gVertices)[i];
		lp = (*ptsHdl)[i];
		wp2.pLat = lp.v;
		wp2.pLong = lp.h;
		
		if(WPointNearWPoint(wp,wp2 ,wdist))
		{
			//for(jseg = 0; jseg < nbounds; jseg++)
			for(jseg = 0; jseg < nSegs; jseg++)
			{
				if(i <= (*fBoundarySegmentsH)[jseg])
				{
					*verNum  = i;
					*segNo = jseg;
					break;
				}
			}
		}
	} 
}

#define PtCurMapReadWriteVersion 5	// restricting bitmap bounds added 9/20/13
//#define PtCurMapReadWriteVersion 4	// increase to add shoreline select fields 9/9/08
//#define PtCurMapReadWriteVersion 3	// increase to add dispersed oil fields 3/27/08
OSErr PtCurMap::Write(BFPB *bfpb)
{
	long i,val;
	//long version = 1;
	long version = PtCurMapReadWriteVersion;
	ClassID id = this -> GetClassID ();
	OSErr	err = noErr;
	long numBoundarySegs = this -> GetNumBoundarySegs();
	long numBoundaryPts = this -> GetNumBoundaryPts();
	long numContourLevels = 0, numDepths = 0, numDropletSizes = 0, numSegSelected = 0;
	double val2;
	double dropsize, probability;
	
	if (err = TMap::Write(bfpb)) return err;
		
	StartReadWriteSequence("PtCurMap::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	/////
	if (fBoundarySegmentsH)
	{
		if (err = WriteMacValue(bfpb, numBoundarySegs)) return err;
		for (i = 0 ; i < numBoundarySegs ; i++) {
			val = INDEXH(fBoundarySegmentsH, i);
			if (err = WriteMacValue(bfpb, val)) return err;
		}
	}
	else
	{
		numBoundarySegs = 0;
		if (err = WriteMacValue(bfpb, numBoundarySegs)) return err;
	}
	/////
	if (fBoundaryTypeH)
	{
		if (err = WriteMacValue(bfpb, numBoundaryPts)) return err;
		for (i = 0 ; i < numBoundaryPts ; i++) {
			val = INDEXH(fBoundaryTypeH, i);
			if (err = WriteMacValue(bfpb, val)) return err;
		}
	}
	else
	{
		numBoundaryPts = 0;
		if (err = WriteMacValue(bfpb, numBoundaryPts)) return err;
	}
	
	/////
	if (fBoundaryPointsH)
	{
		if (err = WriteMacValue(bfpb, numBoundaryPts)) return err;
		for (i = 0 ; i < numBoundaryPts ; i++) {
			val = INDEXH(fBoundaryPointsH, i);
			if (err = WriteMacValue(bfpb, val)) return err;
		}
	}
	else
	{	// only curvilinear netcdf algorithm uses the full set of boundary pts
		numBoundaryPts = 0;
		if (err = WriteMacValue(bfpb, numBoundaryPts)) return err;
	}
	
	if (err = WriteMacValue(bfpb,bDrawLandBitMap)) return err;
	if (err = WriteMacValue(bfpb,bDrawWaterBitMap)) return err;
	
	if (err = WriteMacValue(bfpb,bShowSurfaceLEs)) return err;
	
	if (err = WriteMacValue(bfpb,fContourDepth1)) return err;
	if (err = WriteMacValue(bfpb,fContourDepth2)) return err;

	if (fContourLevelsH) numContourLevels = _GetHandleSize((Handle)fContourLevelsH)/sizeof(**fContourLevelsH);
	
	if (err = WriteMacValue(bfpb, numContourLevels)) return err;
	for (i = 0 ; i < numContourLevels ; i++) {
		val2 = INDEXH(fContourLevelsH, i);
		if (err = WriteMacValue(bfpb, val2)) return err;
	}
	
	/*if (fDepthSliceArray) numDepths = fDepthSliceArray[0];
	if (err = WriteMacValue(bfpb,numDepths)) return err;
	for (i=0; i<numDepths; i++)
	{
		if (err = WriteMacValue(bfpb,fDepthSliceArray[i+1])) return err;
	}*/
	if (err = WriteMacValue(bfpb,fLegendRect)) return err;
	if (err = WriteMacValue(bfpb,bShowLegend)) return err;
	if (err = WriteMacValue(bfpb,bShowSurfaceLEs)) return err;
	
	if (err = WriteMacValue(bfpb,fWaterDensity)) return err;
	if (err = WriteMacValue(bfpb,fMixedLayerDepth)) return err;
	if (err = WriteMacValue(bfpb,fBreakingWaveHeight)) return err;

	if (err = WriteMacValue(bfpb,fDiagnosticStrType)) return err;

	//if (err = WriteMacValue(bfpb,bUseLineCross)) return err;

	if (err = WriteMacValue(bfpb,fWaveHtInput)) return err;

	if (fDropletSizesH) numDropletSizes = _GetHandleSize((Handle)fDropletSizesH)/sizeof(**fDropletSizesH);
	
	if (err = WriteMacValue(bfpb, numDropletSizes)) return err;
	for (i = 0 ; i < numDropletSizes ; i++) {
		dropsize = INDEXH(fDropletSizesH, i).dropletSize;
		probability = INDEXH(fDropletSizesH, i).probability;
		if (err = WriteMacValue(bfpb, dropsize)) return err;
		if (err = WriteMacValue(bfpb, probability)) return err;
	}
	// code goes here, save the beach selection stuff
	/*if (fSegSelectedH) 	
		{numSegSelected = _GetHandleSize((Handle)fSegSelectedH)/sizeof(**fSegSelectedH); }
		else {numSegSelected = 0;}

	
	if (err = WriteMacValue(bfpb, numSegSelected)) return err;
	for (i = 0 ; i < numSegSelected ; i++) {
		val = INDEXH(fSegSelectedH, i);
		if (err = WriteMacValue(bfpb, val)) return err;
	}*/
	
	if (fSelectedBeachHdl) {numSegSelected = _GetHandleSize((Handle)fSelectedBeachHdl)/sizeof(**fSelectedBeachHdl);} else {numSegSelected = 0;}
	
	if (err = WriteMacValue(bfpb, numSegSelected)) return err;
	for (i = 0 ; i < numSegSelected ; i++) {
		val = INDEXH(fSelectedBeachHdl, i);
		if (err = WriteMacValue(bfpb, val)) return err;
	}
	
	if (fSelectedBeachFlagHdl) {numSegSelected = _GetHandleSize((Handle)fSelectedBeachFlagHdl)/sizeof(**fSelectedBeachFlagHdl);} else {numSegSelected = 0;}
	
	if (err = WriteMacValue(bfpb, numSegSelected)) return err;
	for (i = 0 ; i < numSegSelected ; i++) {
		val = INDEXH(fSelectedBeachFlagHdl, i);
		if (err = WriteMacValue(bfpb, val)) return err;
	}
	
	{	// restricting bitmap bounds added 9/20/13
		if (err = WriteMacValue(bfpb,fBitMapBounds)) return err;
		if (err = WriteMacValue(bfpb, fUseBitMapBounds)) return err;
		if (err = WriteMacValue(bfpb, bDrawBitMapBounds)) return err;
	}

	//fSegSelectedH = 0;
	//fSelectedBeachHdl = 0;	//not sure if both are needed
	//fSelectedBeachFlagHdl = 0;
	return 0;
}

OSErr PtCurMap::Read(BFPB *bfpb)
{
	long i,version,val;
	ClassID id;
	OSErr err = 0;
	long 	numBoundarySegs,numBoundaryPts,numContourLevels,numDepths,numDropletSizes,numSegSelected;
	float depthVal;
	double val2;
	double dropsize, probability;
	
	if (err = TMap::Read(bfpb)) return err;

	StartReadWriteSequence("PtCurMap::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("PtCurMap::Read()", "id == TYPE_PTCURMAP", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	//if (version != 1) { printSaveFileVersionError(); return -1; }
	//if (version > 2 || version < 1) { printSaveFileVersionError(); return -1; }
	//if (((!gMearnsVersion && version > PtCurMapReadWriteVersion) || (gMearnsVersion && version > 3))|| version < 1) { printSaveFileVersionError(); return -1; }
	if (version < 1 || version > PtCurMapReadWriteVersion) { printSaveFileVersionError(); return -1; }
	
	if (err = ReadMacValue(bfpb, &numBoundarySegs)) return err;	
	if (!err && numBoundarySegs>0)
	{
		fBoundarySegmentsH = (LONGH)_NewHandleClear(sizeof(long)*numBoundarySegs);
		if (!fBoundarySegmentsH)
			{ TechError("PtCurMap::Read()", "_NewHandleClear()", 0); return -1; }
		
		for (i = 0 ; i < numBoundarySegs ; i++) {
			if (err = ReadMacValue(bfpb, &val)) { printSaveFileVersionError(); return -1; }
			INDEXH(fBoundarySegmentsH, i) = val;
		}
	}
	if (err = ReadMacValue(bfpb, &numBoundaryPts)) return err;	
	if (!err && numBoundaryPts>0)
	{
		fBoundaryTypeH = (LONGH)_NewHandleClear(sizeof(long)*numBoundaryPts);
		if (!fBoundaryTypeH)
			{ TechError("PtCurMap::Read()", "_NewHandleClear()", 0); return -1; }
		
		for (i = 0 ; i < numBoundaryPts ; i++) {
			if (err = ReadMacValue(bfpb, &val)) { printSaveFileVersionError(); return -1; }
			INDEXH(fBoundaryTypeH, i) = val;
		}
	}
	
	if (version>1 && !gMearnsVersion)	// 1/9/04 redid algorithm, now store all boundary points
	{
		if (err = ReadMacValue(bfpb, &numBoundaryPts)) return err;	
		if (!err && numBoundaryPts>0)
		{
			fBoundaryPointsH = (LONGH)_NewHandleClear(sizeof(long)*numBoundaryPts);
			if (!fBoundaryPointsH)
				{ TechError("PtCurMap::Read()", "_NewHandleClear()", 0); return -1; }
			
			for (i = 0 ; i < numBoundaryPts ; i++) {
				if (err = ReadMacValue(bfpb, &val)) { printSaveFileVersionError(); return -1; }
				INDEXH(fBoundaryPointsH, i) = val;
			}
		}
	}
	

	if (err = ReadMacValue(bfpb, &bDrawLandBitMap)) return err;
	if (err = ReadMacValue(bfpb, &bDrawWaterBitMap)) return err;
	//if (gMearnsVersion)
	if (version>2 || gMearnsVersion)
	{
	if (err = ReadMacValue(bfpb, &bShowSurfaceLEs)) return err;
	
	if (err = ReadMacValue(bfpb,&fContourDepth1)) return err;
	if (err = ReadMacValue(bfpb,&fContourDepth2)) return err;

	fContourDepth1AtStartOfRun = fContourDepth1;	// output data is not being saved so will need to rerun to see plots anyway
	fContourDepth2AtStartOfRun = fContourDepth2;
	
	if (err = ReadMacValue(bfpb, &numContourLevels)) return err;	
	if (!err)
	{
		fContourLevelsH = (DOUBLEH)_NewHandleClear(sizeof(double)*numContourLevels);
		if (!fContourLevelsH)
			{ TechError("PtCurMap::Read()", "_NewHandleClear()", 0); return -1; }
		
		for (i = 0 ; i < numContourLevels ; i++) {
			if (err = ReadMacValue(bfpb, &val2)) { printSaveFileVersionError(); return -1; }
			INDEXH(fContourLevelsH, i) = val2;
		}
	}

	/*if (version>1)
	{
		if (err = ReadMacValue(bfpb, &numDepths)) return err;	
		if (!err && numDepths>0)
		{
			fDepthSliceArray = new float[numDepths+1];
			fDepthSliceArray[0]=numDepths;	//store size here
			if (!fDepthSliceArray)
				{ TechError("PtCurMap::Read()", "new float()", 0); return -1; }
			
			for (i = 0; i < numDepths; i++) 
			{
				if (err = ReadMacValue(bfpb, &depthVal)) { printSaveFileVersionError(); return -1; }
				fDepthSliceArray[i+1] = depthVal;
			}
		}
	}*/

	if (err = ReadMacValue(bfpb, &fLegendRect)) return err;
	if (err = ReadMacValue(bfpb, &bShowLegend)) return err;
	if (err = ReadMacValue(bfpb, &bShowSurfaceLEs)) return err;

	if (err = ReadMacValue(bfpb, &fWaterDensity)) return err;
	if (err = ReadMacValue(bfpb, &fMixedLayerDepth)) return err;
	if (err = ReadMacValue(bfpb, &fBreakingWaveHeight)) return err;
	if (err = ReadMacValue(bfpb, &fDiagnosticStrType)) return err;
	
	//if (version > 3) if (err = ReadMacValue(bfpb, &bUseLineCross)) return err;
	if ((gMearnsVersion && version>1) || version>2)
	{
		if (err = ReadMacValue(bfpb, &fWaveHtInput)) return err;
	}
	else
	{
		fWaveHtInput = 1;	// breaking wave height
	}
	if ((gMearnsVersion && version>1) || version>2)
	{
		if (err = ReadMacValue(bfpb, &numDropletSizes)) return err;
		fDropletSizesH = (DropletInfoRecH)_NewHandleClear(sizeof(DropletInfoRec)*numDropletSizes);
		if(!fDropletSizesH)
			{ err = -1; TechError("PtCurMap::Read()", "_NewHandle()", 0); return err; }
		for (i = 0 ; i < numDropletSizes ; i++) {
			if (err = ReadMacValue(bfpb, &dropsize)) return err;
			if (err = ReadMacValue(bfpb, &probability)) return err;
			INDEXH(fDropletSizesH, i).dropletSize = dropsize;
			INDEXH(fDropletSizesH, i).probability = probability;
		}
	}
	}	
	if (version>3)
	{	// clean up on err?
	// code goes here, save the beach selection stuff
	//fSegSelectedH = 0;
	//fSelectedBeachHdl = 0;	//not sure if both are needed
	//fSelectedBeachFlagHdl = 0;
		/*if (err = ReadMacValue(bfpb, &numSegSelected)) return err;	
		if (!err && numSegSelected>0)
		{
			fSegSelectedH = (LONGH)_NewHandleClear(sizeof(long)*numSegSelected);
			if (!fSegSelectedH)
				{ TechError("PtCurMap::Read()", "_NewHandleClear()", 0); return -1; }
			
			for (i = 0 ; i < numSegSelected ; i++) {
				if (err = ReadMacValue(bfpb, &val)) { printSaveFileVersionError(); return -1; }
				INDEXH(fSegSelectedH, i) = val;
			}
		}*/
		if (err = ReadMacValue(bfpb, &numSegSelected)) return err;	
		if (!err && numSegSelected>0)
		{
			fSelectedBeachHdl = (LONGH)_NewHandleClear(sizeof(long)*numSegSelected);
			if (!fSelectedBeachHdl)
				{ TechError("PtCurMap::Read()", "_NewHandleClear()", 0); return -1; }
			
			for (i = 0 ; i < numSegSelected ; i++) {
				if (err = ReadMacValue(bfpb, &val)) { printSaveFileVersionError(); return -1; }
				INDEXH(fSelectedBeachHdl, i) = val;
			}
		}
		if (err = ReadMacValue(bfpb, &numSegSelected)) return err;	
		if (!err && numSegSelected>0)
		{
			fSelectedBeachFlagHdl = (LONGH)_NewHandleClear(sizeof(long)*numSegSelected);
			if (!fSelectedBeachFlagHdl)
				{ TechError("PtCurMap::Read()", "_NewHandleClear()", 0); return -1; }
			
			for (i = 0 ; i < numSegSelected ; i++) {
				if (err = ReadMacValue(bfpb, &val)) { printSaveFileVersionError(); return -1; }
				INDEXH(fSelectedBeachFlagHdl, i) = val;
			}
		}
	}
	if (version > 4)
	{
		if (err = ReadMacValue(bfpb,&fBitMapBounds)) return err;
		if (err = ReadMacValue(bfpb, &fUseBitMapBounds)) return err;
		if (err = ReadMacValue(bfpb, &bDrawBitMapBounds)) return err;
	}
	else
	{
		fBitMapBounds = GetMapBounds();	// make sure default values are set
		fUseBitMapBounds = false;
		bDrawBitMapBounds = false;
	}
	//////////////////
	// now reconstruct the offscreen Land and Water bitmaps
	///////////////////
	//if (gMearnsVersion) SetMinDistOffshore(GetMapBounds());	// I don't think this does anything...
	
	if (!(this->IAm(TYPE_COMPOUNDMAP)))
	{
		Rect bitMapRect;
		long bmWidth, bmHeight;
		WorldRect wRect = this -> GetMapBounds(); // bounds have been read in by the base class
		err = LandBitMapWidthHeight(wRect,&bmWidth,&bmHeight);
		if (err) {printError("Unable to recreate bitmap in PtCurMap::Read"); return err;}
		MySetRect (&bitMapRect, 0, 0, bmWidth, bmHeight);

		fLandBitmap = GetBlackAndWhiteBitmap(DrawWideLandSegments,this,wRect,bitMapRect,&err);

		if(!err)
			fWaterBitmap = GetBlackAndWhiteBitmap(DrawFilledWaterTriangles,this,wRect,bitMapRect,&err);
			
		switch(err) 
		{
			case noErr: break;
			case memFullErr: printError("Out of memory in PtCurMap::Read"); break;
			default: TechError("PtCurMap::Read", "GetBlackAndWhiteBitmap", err); break;
		}
	}
	return 0;
}

/**************************************************************************************************/
long PtCurMap::GetListLength()
{
	long i, n, count = 1;
	TMover *mover;

		if (bIAmPartOfACompoundMap) {if (bOpen) count++; return count;}

	if (bOpen) {
		count++;// name

 		count++; // REFLOATHALFLIFE

		count++; // bitmap-visible-box

		if (fUseBitMapBounds) count++;
		
		if(this->ThereIsADispersedSpill()) count++; // draw contours
		if(this->ThereIsADispersedSpill()) count++; // set contours
		if(this->ThereIsADispersedSpill()) count++; // draw legend
		if(this->ThereIsADispersedSpill()) count++; // concentration table

		if(this->ThereIsADispersedSpill()) count++; // surface LEs
		if(this->ThereIsADispersedSpill()) count++; // water density
		if(this->ThereIsADispersedSpill()) count++; // mixed layer depth
		if(this->ThereIsADispersedSpill()) count++; // breaking wave height

		if(this->ThereIsADispersedSpill()) count++; // diagnostic string info
		if(this->ThereIsADispersedSpill()) count++; // droplet data

		if (bMoversOpen)
			for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
				moverList->GetListItem((Ptr)&mover, i);
				count += mover->GetListLength();
			}

	}

	return count;
}
/**************************************************************************************************/
ListItem PtCurMap::GetNthListItem(long n, short indent, short *style, char *text)
{
	long i, m, count;
	TMover *mover;
	ListItem item = { this, 0, indent, 0 };
		
	if (n == 0) {
		item.index = I_PMAPNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		strcpy(text, className);
		
		return item;
	}
	n -= 1;

	if (bIAmPartOfACompoundMap && bOpen) {
	if (n == 0) {
		item.indent++;
		item.index = I_PDRAWCONTOURSFORMAP;
		item.bullet = bDrawContours ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
		strcpy(text, "Draw Contours");
		
		return item;
	}
	n -= 1;
	}

	if (bIAmPartOfACompoundMap) { item.owner = 0;return item;}
	
	if (n == 0) {
		//item = TMap::GetNthListItem(I_REFLOATHALFLIFE, indent, style, text);
		item.index = I_PREFLOATHALFLIFE; // override the index
		item.indent = indent; // override the indent
			sprintf(text, "Refloat half life: %g hr",fRefloatHalfLifeInHrs);
		return item;
	}
	n -= 1;
	
	if (n == 0) {
		item.indent++;
		item.index = I_PDRAWLANDWATERBITMAP;
		item.bullet = (bDrawLandBitMap && bDrawWaterBitMap) ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
		strcpy(text, "Show Land / Water Map");
		
		return item;
	}
	n -= 1;
	
	if(fUseBitMapBounds)
	{
		if (n == 0) {
			item.indent++;
			item.index = I_PDRAWRESTRICTEDBITMAP;
			item.bullet = bDrawBitMapBounds ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Restricted Bitmap Domain");
			
			return item;
		}
		n -= 1;
	}
	
	if(this ->ThereIsADispersedSpill())
	{
		if (n == 0) {
			//item.indent++;
			item.index = I_PDRAWCONTOURS;
			//item.bullet = bShowContours ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			if (this -> fContourDepth2 == 0) 
			{
				this -> fContourDepth2 = 5.;	// maybe use mixed layer depth
			}
			if (this -> fContourDepth1 == BOTTOMINDEX)
				//sprintf(text, "Draw Contours for Bottom Layer (1 meter)");
				sprintf(text, "Draw Contours for Bottom Layer (%g meters)",fBottomRange);
			else
				sprintf(text, "Draw Contours for %g to %g meters",fContourDepth1,fContourDepth2);
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			long numLevels;
			item.index = I_PSETCONTOURS;
			if (!fContourLevelsH) 
			{
				if (!InitContourLevels()) return item;
			}
			numLevels = GetNumDoubleHdlItems(fContourLevelsH);
			sprintf(text, "Contour Levels (mg/L) : Min %g  Max %g",(*fContourLevelsH)[0],(*fContourLevelsH)[numLevels-1]);
			return item;
		}
		n -= 1;
		if (n == 0) {
			long numLevels;
			TTriGridVel3D* triGrid = GetGrid3D(true);	// use refined grid if there is one
			
			item.index = I_PCONCTABLE;
			// code goes here, only show if output values exist, change to button...
			// Following plume vs following point
			sprintf(text, "Show concentration plots");
			if (!triGrid) *style = normal; 
			else
			{
				outputData **oilConcHdl = triGrid -> GetOilConcHdl();	
				if (!oilConcHdl)
				{
					*style = normal;
					//printError("There is no concentration data to plot");
					//err = -1;
					//return TRUE;
				}
				else
				{
					Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
					if (triSelected)
						strcat(text," at selected triangles");
					else
						strcat(text," following the plume");
					*style = italic;
				}
			}

			return item;
		}
		n -= 1;
		if (n == 0) {
			item.indent++;
			item.index = I_PSHOWLEGEND;
			item.bullet = bShowLegend ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Contour Legend");
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			item.indent++;
			item.index = I_PSHOWSURFACELES;
			item.bullet = bShowSurfaceLEs ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Surface LEs");
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			//item.indent++;
			item.index = I_PWATERDENSITY;
			sprintf(text, "Water Density : %ld (kg/m^3)",fWaterDensity);
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			//item.indent++;
			item.index = I_PMIXEDLAYERDEPTH;
			sprintf(text, "Mixed Layer Depth : %g m",fMixedLayerDepth);
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			//item.indent++;
			item.index = I_PBREAKINGWAVEHT;
			//sprintf(text, "Breaking Wave Height : %g m",fBreakingWaveHeight);
			sprintf(text, "Breaking Wave Height : %g m",GetBreakingWaveHeight());
			//if (fWaveHtInput==0) 	// user input value by hand, also show wind speed?				
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			//item.indent++;
			item.index = I_PDIAGNOSTICSTRING;
			if (fDiagnosticStrType == NODIAGNOSTICSTRING)
				sprintf(text, "No Grid Diagnostics");
			else if (fDiagnosticStrType == TRIANGLEAREA)
				sprintf(text, "Show Triangle Areas (km^2)");
			else if (fDiagnosticStrType == NUMLESINTRIANGLE)
				sprintf(text, "Show Number LEs in Triangles");
			else if (fDiagnosticStrType == CONCENTRATIONLEVEL)
				sprintf(text, "Show Concentration Levels");
			else if (fDiagnosticStrType == DEPTHATCENTERS)
				sprintf(text, "Show Depth at Triangle Centers (m)");
			else if (fDiagnosticStrType == SUBSURFACEPARTICLES)
				sprintf(text, "Show Subsurface Particles");
			else if (fDiagnosticStrType == SHORELINEPTNUMS)
				sprintf(text, "Show Selected Shoreline Point Numbers");
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			//item.indent++;
			item.index = I_PDROPLETINFO;
			sprintf(text, "Droplet Data");
							
			return item;
		}
		n -= 1;
	}
	
	
	if (bOpen) {
		indent++;
		if (n == 0) {
			item.index = I_PMOVERS;
			item.indent = indent;
			item.bullet = bMoversOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Movers");
			
			return item;
		}
		
		n -= 1;
		
		if (bMoversOpen)
			for (i = 0, m = moverList->GetItemCount() ; i < m ; i++) {
				moverList->GetListItem((Ptr)&mover, i);
				count = mover->GetListLength();
				if (count > n)
				{
					item =  mover->GetNthListItem(n, indent + 1, style, text);
					if (mover->bActive) *style = italic;
					return item;
					//return mover->GetNthListItem(n, indent + 1, style, text);
				}
				
				n -= count;
			}
	}
	
	item.owner = 0;
	
	return item;
}

/**************************************************************************************************/
Boolean PtCurMap::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet) {
		switch (item.index) {
			case I_PMAPNAME: bOpen = !bOpen; return TRUE;
			case I_PDRAWLANDWATERBITMAP:
				bDrawLandBitMap = !bDrawLandBitMap;
				bDrawWaterBitMap = !bDrawWaterBitMap;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_PDRAWRESTRICTEDBITMAP:
				bDrawBitMapBounds = !bDrawBitMapBounds;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_PSHOWSURFACELES:
				bShowSurfaceLEs = !bShowSurfaceLEs;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_PSHOWLEGEND:
				bShowLegend = !bShowLegend;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_PMOVERS: bMoversOpen = !bMoversOpen; return TRUE;
			case I_PDRAWCONTOURSFORMAP: bDrawContours = !bDrawContours; return TRUE;
		}
	}
	
	if (doubleClick) {
		if (this -> FunctionEnabled(item, SETTINGSBUTTON)) {
			if (item.index == I_PSETCONTOURS || item.index == I_PSHOWLEGEND)
			{
				if (!fContourLevelsH) 
				{
					if (InitContourLevels()==-1) return TRUE;
				}
				ContourDialog(&fContourLevelsH,0);
				return TRUE;
			}
			if (item.index == I_PCONCTABLE)
			{
				TTriGridVel3D* triGrid = GetGrid3D(true);	// use refined grid if there is one
				
				if (!triGrid) return true; 
				outputData **oilConcHdl = triGrid -> GetOilConcHdl();	
				if (!oilConcHdl)
				{
					printError("There is no concentration data to plot");
					//err = -1;
					//return TRUE;
				}
				else
				{
					float depthRange1 = fContourDepth1, depthRange2 = fContourDepth2,  bottomRange = fBottomRange;
					Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
					// compare current contourdepths with values at start of run
					// give warning if they are different, and send original to plot
					if (fContourDepth1AtStartOfRun != fContourDepth1 || fContourDepth2AtStartOfRun != fContourDepth2)
					{
						short buttonSelected;
						// code goes here, set up a new dialog with better wording
						buttonSelected  = MULTICHOICEALERT(1690,"The depth range has been changed. To see plots at new depth range rerun the model. Do you still want to see the old plots?",FALSE);
						switch(buttonSelected){
							case 1: // continue
								break;  
							case 3: // stop
								return 0; 
								break;
						}
						//printNote("The depth range has been changed. To see plots at new depth range rerun the model");	
						depthRange1 = fContourDepth1AtStartOfRun;
						depthRange2 = fContourDepth2AtStartOfRun;
					}
					if (triSelected) 	// tracked output at a specified area
						PlotDialog(oilConcHdl,fDepthSliceArray,depthRange1,depthRange2,bottomRange,true,false);
					else 	// tracked output following the plume
						PlotDialog(oilConcHdl,fDepthSliceArray,depthRange1,depthRange2,bottomRange,false,false);
				}
				//ConcentrationTable(oilConcHdl/*,fTriAreaArray,GetNumContourLevels()*/);
				return TRUE;
			}
			if (item.index == I_PDROPLETINFO)
			{
				DropletSizeTable(&fDropletSizesH);
				return true;
			}
			if (item.index == I_PDIAGNOSTICSTRING)
			{	// some temp code for Debra
				//DropletSizeTable(&fDropletSizesH);
				/*char infoStr[256];
				long i, numOfLEs, numLESets = model->LESetsList->GetItemCount();
				for (i = 0; i < numLESets; i++)
				{
					Boolean bThereIsSubsurfaceOil = false;
					double avDepth=0, stdDev=0;
					DispersionRec dispersantData;
					TLEList *thisLEList = 0;
					model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
					if (thisLEList->fLeType == UNCERTAINTY_LE)	
						continue;	// don't draw uncertainty for now...
					numOfLEs = thisLEList->numOfLEs;
					// density set from API
					dispersantData = ((TOLEList*)thisLEList)->GetDispersionInfo();
			
					bThereIsSubsurfaceOil = dispersantData.bDisperseOil && model->GetModelTime() - model->GetStartTime() - model->GetTimeStep() >= dispersantData.timeToDisperse;
					bThereIsSubsurfaceOil = bThereIsSubsurfaceOil || (*(TOLEList*)thisLEList).fAdiosDataH;
					bThereIsSubsurfaceOil = bThereIsSubsurfaceOil || (*(TOLEList*)thisLEList).fSetSummary.z > 0;
					if (bThereIsSubsurfaceOil)
					{
						((TOLEList*)thisLEList) -> CalculateAverageIntrusionDepth(&avDepth, &stdDev);
						sprintf(infoStr,"Average intrusion depth = %.2f, standard deviation = %.2f",avDepth,stdDev);
						printNote(infoStr);
						break;
					}
				}*/
				//return true;
			}
			if (item.index == I_PDRAWCONTOURSFORMAP) return TRUE;
			
			item.index = I_PMAPNAME;
			this -> SettingsItem(item);
			return TRUE;
		}

		if (item.index == I_PMOVERS)
		{
			item.owner -> AddItem (item);
			return TRUE;
		}
		
	}

	return false;
}
/**************************************************************************************************/
Boolean PtCurMap::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	
	switch (item.index) {
		case I_PMAPNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE; 
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (!model->mapList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (model->mapList->GetItemCount() - 1);
					}
			}
			break;
		case I_PMOVERS:
			switch (buttonID) {
				case ADDBUTTON: return TRUE;
				case SETTINGSBUTTON: return FALSE;
				case DELETEBUTTON: return FALSE;
			}
			break;
		case I_PDRAWLANDWATERBITMAP:
		case I_PDRAWRESTRICTEDBITMAP:
		case I_PDRAWCONTOURS:
		case I_PSHOWLEGEND:
		case I_PSHOWSURFACELES:
		case I_PSETCONTOURS:
		case I_PCONCTABLE:
		case I_PWATERDENSITY:
		case I_PMIXEDLAYERDEPTH:
		case I_PBREAKINGWAVEHT:
		case I_PDIAGNOSTICSTRING:
		case I_PDROPLETINFO:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
		case I_PREFLOATHALFLIFE:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
			break;
	}
	
	return FALSE;
}

/**************************************************************************************************/
void PtCurMap::DrawContourScale(Rect r, WorldRect view)
{	// new version combines triangle area and contours
	Point		p;
	short		h,v,x,y,dY,widestNum=0;
	RGBColor	rgb;
	Rect		rgbrect, legendRect = fLegendRect;
	char 		numstr[30],numstr2[30],numstr3[30],text[30],titleStr[40],unitStr[40];
	long 		i,numLevels,strLen;
	double	minLevel, maxLevel;
	double 	value, value2=0, totalArea = 0, htScale = 1., wScale = 1.;
	
	TCurrentMover *mover = Get3DCurrentMover();
	if ((mover) && mover->IAm(TYPE_TRICURMOVER)) { (dynamic_cast<TriCurMover *>(mover))->DrawContourScale(r,view);/* return;*/}
	if ((mover) && mover->IAm(TYPE_NETCDFMOVER)) {(dynamic_cast<NetCDFMover*>(mover))->DrawContourScale(r,view);/* return;*/}
	if (!this->ThereIsADispersedSpill()) return;
	SetRGBColor(&rgb,0,0,0);
	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
#ifdef IBM
	float mapWidth, mapHeight, rWidth, rHeight, legendWidth, legendHeight;
	TextFont(kFontIDGeneva); TextSize(6);
#endif
	if (!fContourLevelsH) 
	{
		if (InitContourLevels()==-1) return;
	}

	if (gSavingOrPrintingPictFile)	// on Windows, saving does not use this
	{
		Rect mapRect;
#ifdef MAC
		mapRect = DrawingRect(settings.listWidth + 1, RIGHTBARWIDTH);
		if (!EqualRects(r,mapRect))
		{
			Boolean bCloserToTop = (legendRect.top - mapRect.top) <= (mapRect.bottom - legendRect.bottom);
			Boolean bCloserToLeft = (legendRect.left - mapRect.left) <= (mapRect.right - legendRect.right);
			if (bCloserToTop)
			{
				legendRect.top = legendRect.top - mapRect.top + r.top;
				legendRect.bottom = legendRect.bottom - mapRect.top + r.top;
			}
			else
			{
				legendRect.top = r.bottom - (mapRect.bottom - legendRect.top);
				legendRect.bottom = r.bottom - (mapRect.bottom - legendRect.bottom);
			}
			if (bCloserToLeft)
			{
				legendRect.left = legendRect.left - mapRect.left + r.left;
				legendRect.right = legendRect.right - mapRect.left + r.left;
			}
			else
			{
				legendRect.left = r.right - (mapRect.right - legendRect.left);
				legendRect.right = r.right - (mapRect.right - legendRect.right);
			}
			// should check if legend is too big for size of printout
			if (legendRect.bottom > r.bottom) {legendRect.top -= (legendRect.bottom - r.bottom); legendRect.top -= (legendRect.bottom - r.bottom);}
			if (legendRect.right > r.right) {legendRect.left -= (legendRect.right - r.right); legendRect.right -= (legendRect.right - r.right);}
		}
#else
		mapRect = DrawingRect(0,settings.listWidth);
		mapRect.top = mapRect.top - TOOLBARHEIGHT;
		mapRect.bottom = mapRect.bottom - TOOLBARHEIGHT;
		mapWidth = mapRect.right - mapRect.left;
		rWidth = r.right - r.left;
		legendWidth = legendRect.right - legendRect.left;
		mapHeight = mapRect.bottom - mapRect.top;
		rHeight = r.bottom - r.top;
		legendHeight = legendRect.bottom - legendRect.top;
		if (!EqualRects(r,mapRect))
		{
			htScale = rHeight / mapHeight; wScale = rWidth / mapWidth;
			legendRect.left = r.left + (legendRect.left - mapRect.left) * rWidth / mapWidth;
			legendRect.right = legendRect.left + legendWidth * rWidth / mapWidth;
			legendRect.top = r.top + (legendRect.top - mapRect.top) * rHeight / mapHeight;
			legendRect.bottom = legendRect.top + legendHeight * rHeight / mapHeight;
			// should check if legend is too big for size of printout
			if (legendRect.bottom > r.bottom) {legendRect.top -= (legendRect.bottom - r.bottom); legendRect.top -= (legendRect.bottom - r.bottom);}
			if (legendRect.right > r.right) {legendRect.left -= (legendRect.right - r.right); legendRect.right -= (legendRect.right - r.right);}
		}
#endif
	}
	else
	{
		if (EmptyRect(&legendRect)||!RectInRect2(&legendRect,&r))	// otherwise printing or saving - set a global?
		{
			legendRect.top = r.top;
			legendRect.left = r.left;
			legendRect.bottom = r.top + 120*htScale;	// reset after contour levels drawn
			legendRect.right = r.left + 80*wScale;	// reset if values go beyond default width
			if (fTriAreaArray) legendRect.right += 20*wScale;	// reset if values go beyond default width
		}
	}
	rgbrect = legendRect;
	EraseRect(&rgbrect);
	
	x = (rgbrect.left + rgbrect.right) / 2;
	//dY = RectHeight(rgbrect) / 12;
	dY = 10*htScale;
	y = rgbrect.top + dY / 2;
	MyMoveTo(rgbrect.left+20*wScale,y+dY);
	if (fTriAreaArray) 
	{
		drawstring("Conc.");
		MyMoveTo(rgbrect.left+80*wScale,y+dY);			
		drawstring("Area");
		MyMoveTo(rgbrect.left+20*wScale,y+2*dY);
		drawstring(" ppm");
		MyMoveTo(rgbrect.left+80*wScale,y+2*dY);
		drawstring("km^2");
		widestNum = 80*wScale+stringwidth("Area");
	}
	else
	{
		drawstring("Conc. (ppm)");		
		widestNum = 20*wScale+stringwidth("Conc. (ppm)");
	}
	numLevels = GetNumDoubleHdlItems(fContourLevelsH);
	v = rgbrect.top+40*htScale;
	if (!fTriAreaArray) v -= 10*htScale;
	h = rgbrect.left;
	for (i=0;i<numLevels;i++)
	{
		//float colorLevel = .8*float(i)/float(numLevels-1);
		float colorLevel = float(i)/float(numLevels-1);
		value = (*fContourLevelsH)[i];
	
		if (fTriAreaArray) 
		{
			value2 = fTriAreaArray[i];
			totalArea += value2;
		}
		MySetRect(&rgbrect,h+4*wScale,v-9*htScale,h+14*wScale,v+1*htScale);
		
#ifdef IBM		
		rgb = GetRGBColor(colorLevel);
#else
		rgb = GetRGBColor(1.-colorLevel);
#endif
		//rgb = GetRGBColor(0.8-colorLevel);
		RGBForeColor(&rgb);
		PaintRect(&rgbrect);
		MyFrameRect(&rgbrect);
	
		MyMoveTo(h+20*wScale,v+.5*htScale);
	
		RGBForeColor(&colors[BLACK]);
		if (i<numLevels-1)
		{
			MyNumToStr(value,numstr);
			strcat(numstr," - ");
			MyNumToStr((*fContourLevelsH)[i+1],numstr2);
			strcat(numstr,numstr2);
		}
		else
		{
			strcpy(numstr,"> ");
			MyNumToStr(value,numstr2);
			strcat(numstr,numstr2);
		}
		//strcat(numstr,"    mg/L");
		//drawstring(MyNumToStr(value,numstr));
		drawstring(numstr);
		strLen = stringwidth(numstr);
		if (fTriAreaArray)
		{
			MyMoveTo(h+80*wScale,v+.5*htScale);
			MyNumToStr(value2,numstr3);
			//strcat(numstr,"		");
			//strcat(numstr,numstr3);
			//drawstring(numstr);
			drawstring(numstr3);
			strLen = 80*wScale + stringwidth(numstr3);
		}
		if (strLen>widestNum) widestNum = strLen;
		v = v+9*htScale;
	}
	if (fTriAreaArray)
	{
		MyMoveTo(h+20*wScale,v+5*htScale);
		strcpy(numstr,"Total Area = ");
		MyNumToStr(totalArea,numstr2);
		strcat(numstr,numstr2);
		drawstring(numstr);
		v = v + 9*htScale;
	}	
	if (fContourDepth1==BOTTOMINDEX)
		//sprintf(text, "Depth: Bottom");
		sprintf(text, "Depth: Bottom %g m",fBottomRange);
	else
		sprintf(text, "Depth: %g to %g m",fContourDepth1,fContourDepth2);

	MyMoveTo(h+20*wScale, v+5*htScale);
	drawstring(text);
	if (stringwidth(text)+20*wScale > widestNum) widestNum = stringwidth(text)+20*wScale;
	v = v + 9*htScale;
	legendRect.bottom = v+3*htScale;
	if (legendRect.right<h+10*wScale+widestNum+4*wScale) legendRect.right = h+10*wScale+widestNum+4*wScale;
	else if (legendRect.right>legendRect.left+80*wScale && h+10*wScale+widestNum+4*wScale<=legendRect.right)
		legendRect.right = h+10*wScale+widestNum+4*wScale;	// may want to redraw to recenter the header
 	MyFrameRect(&legendRect);
	if (!gSavingOrPrintingPictFile)
		fLegendRect = legendRect;
	return;
}

void PtCurMap::DrawDepthContourScale(Rect r, WorldRect view)
{
	Point		p;
	short		h,v,x,y,dY,widestNum=0;
	RGBColor	rgb;
	Rect		rgbrect;
	char 		numstr[30],numstr2[30],text[30];
	long 		i,numLevels;
	double	minLevel, maxLevel;
	double 	value;
	TTriGridVel3D* triGrid = GetGrid3D(false);	
	
	triGrid->DrawContourScale(r,view);

	return;
}

void PtCurMap::Draw(Rect r, WorldRect view)
{
	/////////////////////////////////////////////////
	// JLM 6/10/99 maps must erase their rectangles in case a lower priority map drew in our rectangle
	// CMO 11/16/00 maps must erase their polygons in case a lower priority map drew in our polygon
	LongRect	mapLongRect;
	Rect m;
	Boolean  onQuickDrawPlane, changedLineWidth = false;
	WorldRect bounds = this -> GetMapBounds();
	RgnHandle saveClip=0, newClip=0;
	
	mapLongRect.left = SameDifferenceX(bounds.loLong);
	mapLongRect.top = (r.bottom + r.top) - SameDifferenceY(bounds.hiLat);
	mapLongRect.right = SameDifferenceX(bounds.hiLong);
	mapLongRect.bottom = (r.bottom + r.top) - SameDifferenceY(bounds.loLat);
	onQuickDrawPlane = IntersectToQuickDrawPlane(mapLongRect,&m);

	if (AreUsingThinLines())
	{
		StopThinLines();
		changedLineWidth = true;
	}
	//EraseRect(&m); 
	//EraseReg instead
	if (fBoundaryTypeH)
	{
#ifdef MAC
		saveClip = NewRgn(); //
		if(saveClip) {
			GetClip(saveClip);///
			newClip = NewRgn();
			if(newClip) {
				OpenRgn();
				DrawBoundaries(r);
				CloseRgn(newClip);
				EraseRgn(newClip);		
				DisposeRgn(newClip);
			}
			SetClip(saveClip);//
			DisposeRgn(saveClip);
		}
#else
		EraseRegion(r);
#endif
	}

	/////////////////////////////////////////////////

	if (fBoundaryTypeH) DrawBoundaries(r);

	if (this -> bDrawWaterBitMap && onQuickDrawPlane)
		DrawDIBImage(LIGHTBLUE,&fWaterBitmap,m);
		
	if (this -> bDrawLandBitMap && onQuickDrawPlane)
		DrawDIBImage(DARKGREEN,&fLandBitmap,m);
		
	if (bDrawBitMapBounds)
	{
		Rect m;
		WorldRect restrictedBounds =  this -> GetMapBounds();
		m = WorldToScreenRect(restrictedBounds,view,r);
		if (fUseBitMapBounds) MyFrameRect(&m);
	}
		
	//////
	
	if (changedLineWidth)
	{
		StartThinLines();
	}
	TMap::Draw(r, view);
}

#define POINTDRAWFLAG 0
void PtCurMap::DrawSegmentLabels(Rect r)
{
	long i, startIndex, endIndex, curIndex=0, p, afterP, firstSegIndex, lastSegIndex;
	long segNo,lastPtOnSeg,firstPtOnSeg,selectionNumber=0,midSelectionIndex;
	RGBColor sc;
	char numstr[40];
	short x,y;
	Point pt;
	Boolean offQuickDrawPlane = false;
	LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
	if(!ptsHdl) return;
	
	GetForeColor(&sc);
	RGBForeColor(&colors[RED]);
	//TextSizeTiny();	

	while(MoreSegments(fSelectedBeachHdl,&startIndex, &endIndex, &curIndex))
	{
		if(endIndex <= startIndex)continue;

		selectionNumber++;
		segNo = PointOnWhichSeg((*fSelectedBeachHdl)[startIndex]);
		firstPtOnSeg = segNo == 0 ? 0: (*fBoundarySegmentsH)[segNo-1] + 1;
		lastPtOnSeg = (*fBoundarySegmentsH)[segNo];
		firstSegIndex = (*fSelectedBeachHdl)[startIndex];
		lastSegIndex = (*fSelectedBeachHdl)[endIndex-1];
		midSelectionIndex = (firstSegIndex+lastSegIndex)/2;
		pt = GetQuickDrawPt((*ptsHdl)[midSelectionIndex].h,(*ptsHdl)[midSelectionIndex].v,&r,&offQuickDrawPlane);
		//MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		MyNumToStr(selectionNumber,numstr);
		x = pt.h;
		y = pt.v;
		MyDrawString(/*CENTERED,*/x,y,numstr,true,POINTDRAWFLAG);
	}
	RGBForeColor(&sc);
}

void PtCurMap::DrawPointLabels(Rect r)
{
	long i, startIndex, endIndex, curIndex=0, firstSegIndex, lastSegIndex;
	long segNo,lastPtOnSeg,firstPtOnSeg,selectionNumber=0;
	RGBColor sc;
	char numstr[40];
	short x,y;
	Point pt;
	Boolean offQuickDrawPlane = false;
	LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
	if(!ptsHdl) return;
	
	GetForeColor(&sc);
	RGBForeColor(&colors[BLUE]);
	TextSizeTiny();	

	while(MoreSegments(fSelectedBeachHdl,&startIndex, &endIndex, &curIndex))
	{
		if(endIndex <= startIndex)continue;

		selectionNumber++;
		segNo = PointOnWhichSeg((*fSelectedBeachHdl)[startIndex]);
		firstPtOnSeg = segNo == 0 ? 0: (*fBoundarySegmentsH)[segNo-1] + 1;
		lastPtOnSeg = (*fBoundarySegmentsH)[segNo];
		firstSegIndex = (*fSelectedBeachHdl)[startIndex];
		lastSegIndex = (*fSelectedBeachHdl)[endIndex-1];
		if (firstSegIndex < lastSegIndex)  {startIndex = firstSegIndex; endIndex = lastSegIndex;}
		else {startIndex = lastSegIndex; endIndex = firstSegIndex;}
		for (i=startIndex;i<=endIndex;i++)
		{
			pt = GetQuickDrawPt((*ptsHdl)[i].h,(*ptsHdl)[i].v,&r,&offQuickDrawPlane);
			//MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
			MyNumToStr(i,numstr);
			x = pt.h;
			y = pt.v;
			MyDrawString(/*CENTERED,*/x,y,numstr,false,POINTDRAWFLAG);
		}
	}
	RGBForeColor(&sc);
}

void PtCurMap::DrawBoundaries(Rect r)
{
	long nSegs = GetNumBoundarySegs();	
	long theSeg,startver,endver,j;
	long x,y;
	Point pt;
	Boolean offQuickDrawPlane = false;

	long penWidth = 3;
	long halfPenWidth = penWidth/2;

	PenNormal();
	RGBColor sc;
	GetForeColor(&sc);
	
	// to support new curvilinear algorithm
	if (fBoundaryPointsH)
	{
		DrawBoundaries2(r);
		return;
	}

	LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
	if(!ptsHdl) return;

#ifdef MAC
	PenSize(penWidth,penWidth);
#else
	PenStyle(BLACK,penWidth);
#endif

	// have each seg be a polygon with a fill option - land only, maybe fill with a pattern?
	for(theSeg = 0; theSeg < nSegs; theSeg++)
	{
		startver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
		endver = (*fBoundarySegmentsH)[theSeg]+1;
	
		pt = GetQuickDrawPt((*ptsHdl)[startver].h,(*ptsHdl)[startver].v,&r,&offQuickDrawPlane);
		MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		for(j = startver + 1; j < endver; j++)
		{
			if ((*fBoundaryTypeH)[j]==2)	// a water boundary
				RGBForeColor(&colors[BLUE]);
			else// add option to change color, light or dark depending on which is easier to see , see premerge GNOME_beta
			{
				RGBForeColor(&colors[BROWN]);	// land
			}
			if (fSelectedBeachFlagHdl && (*fSelectedBeachFlagHdl)[j]==1)
				RGBForeColor(&colors[DARKGREEN]);
			pt = GetQuickDrawPt((*ptsHdl)[j].h,(*ptsHdl)[j].v,&r,&offQuickDrawPlane);
			if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[j]==1))
			{
				MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
			}
			else
				MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		}
		if ((*fBoundaryTypeH)[startver]==2)	// a water boundary
			RGBForeColor(&colors[BLUE]);
		else
		{
			RGBForeColor(&colors[BROWN]);	// land
		}
		if (fSelectedBeachFlagHdl && (*fSelectedBeachFlagHdl)[startver]==1)
			RGBForeColor(&colors[DARKGREEN]);
		pt = GetQuickDrawPt((*ptsHdl)[startver].h,(*ptsHdl)[startver].v,&r,&offQuickDrawPlane);
		if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[startver]==1))
		{
			MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		}
		else
			MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
	}

#ifdef MAC
	PenSize(1,1);
#else
	PenStyle(BLACK,1);
#endif
	RGBForeColor(&sc);
	if (fSelectedBeachFlagHdl) DrawSegmentLabels(r);
	if (fSelectedBeachFlagHdl && fDiagnosticStrType==SHORELINEPTNUMS) DrawPointLabels(r);
}

void PtCurMap::DrawBoundaries2(Rect r)
{
	// should combine into original DrawBoundaries, just check for fBoundaryPointsH
	PenNormal();
	RGBColor sc;
	GetForeColor(&sc);
	
	TMover *mover=0;

	long nSegs = GetNumBoundarySegs();	
	long theSeg,startver,endver,j;
	long x,y,index1,index;
	Point pt;
	Boolean offQuickDrawPlane = false;

	long penWidth = 3;
	long halfPenWidth = penWidth/2;

	LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
	if(!ptsHdl) return;
	
	//mover = this->GetMover(TYPE_PTCURMOVER);
	//if (mover)
		//ptsHdl = ((PtCurMover *)mover)->GetPointsHdl();
	//else	return; // some error alert
	//if(!ptsHdl) return;

#ifdef MAC
	PenSize(penWidth,penWidth);
#else
	PenStyle(BLACK,penWidth);
#endif


	for(theSeg = 0; theSeg < nSegs; theSeg++)
	{
		startver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
		endver = (*fBoundarySegmentsH)[theSeg]+1;
		index1 = (*fBoundaryPointsH)[startver];
		pt = GetQuickDrawPt((*ptsHdl)[index1].h,(*ptsHdl)[index1].v,&r,&offQuickDrawPlane);
		MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		for(j = startver + 1; j < endver; j++)
		{
			index = (*fBoundaryPointsH)[j];
     		if ((*fBoundaryTypeH)[j]==2)	// a water boundary
				RGBForeColor(&colors[BLUE]);
			else
				RGBForeColor(&colors[BROWN]);	// land
			if (fSelectedBeachFlagHdl && (*fSelectedBeachFlagHdl)[j]==1)
				RGBForeColor(&colors[DARKGREEN]);
			pt = GetQuickDrawPt((*ptsHdl)[index].h,(*ptsHdl)[index].v,&r,&offQuickDrawPlane);
			if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[j]==1))
			{
				MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
			}
			else
				MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		}
		if ((*fBoundaryTypeH)[startver]==2)	// a water boundary
			RGBForeColor(&colors[BLUE]);
		else
			RGBForeColor(&colors[BROWN]);	// land
		if (fSelectedBeachFlagHdl && (*fSelectedBeachFlagHdl)[startver]==1)
			RGBForeColor(&colors[DARKGREEN]);
		pt = GetQuickDrawPt((*ptsHdl)[index1].h,(*ptsHdl)[index1].v,&r,&offQuickDrawPlane);
		if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[startver]==1))
		{
			MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		}
		else
			MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
	}

#ifdef MAC
	PenSize(1,1);
#else
	PenStyle(BLACK,1);
#endif
	RGBForeColor(&sc);
}

/**************************************************************************************************/
#ifdef IBM
void PtCurMap::EraseRegion(Rect r)
{
	long nSegs = GetNumBoundarySegs();	
	long theSeg,startver,endver,j,index;
	Point pt;
	Boolean offQuickDrawPlane = false;

	LongPointHdl ptsHdl = GetPointsHdl(false); // will use refined grid if there is one
	if(!ptsHdl) return;

	for(theSeg = 0; theSeg< nSegs; theSeg++)
	{
		startver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
		endver = (*fBoundarySegmentsH)[theSeg]+1;
		long numPts = endver - startver;
		POINT *pointsPtr = (POINT*)_NewPtr(numPts *sizeof(POINT));
		RgnHandle newClip=0;
		HBRUSH whiteBrush;
	
		for(j = startver; j < endver; j++)
		{
			if (fBoundaryPointsH)	// the reordered curvilinear grid
				index = (*fBoundaryPointsH)[j];
			else index = j;
			pt = GetQuickDrawPt((*ptsHdl)[index].h,(*ptsHdl)[index].v,&r,&offQuickDrawPlane);
			(pointsPtr)[j-startver] = MakePOINT(pt.h,pt.v);
		}

		newClip = CreatePolygonRgn((const POINT*)pointsPtr,numPts,ALTERNATE);
		whiteBrush = (HBRUSH)GetStockObject(WHITE_BRUSH);
		//err = SelectClipRgn(currentHDC,savedClip);
		FillRgn(currentHDC, newClip, whiteBrush);
		DisposeRgn(newClip);
		//DeleteObject(newClip);
		//SelectClipRgn(currentHDC,0);
		if(pointsPtr) {_DisposePtr((Ptr)pointsPtr); pointsPtr = 0;}
	}

}
#endif
/**************************************************************************************************/
void PtCurMap::DrawContours(Rect r, WorldRect view)
{	// need all LELists
	long i, j, numOfLEs, numLESets, numTri, numDepths;
	LERec LE;
	Rect leRect, beachedRect, floatingRect;
	float beachedWidthInPoints = 3, floatingWidthInPoints = 2; // a point = 1/72 of an inch
	float pixelsPerPoint = PixelsPerPoint();
	short offset, massunits;
	Point pt;
	Boolean offQuickDrawPlane = false, bShowContours, bThereIsSubsurfaceOil = false;
	RGBColor saveColor, *onLandColor, *inWaterColor;
	LONGH numLEsInTri = 0;
	DOUBLEH massInTriInGrams = 0;
	double density, LEmass, massInGrams, halfLife;
	TopologyHdl topH = 0;
	TDagTree *dagTree = 0;
	TTriGridVel3D* triGrid = GetGrid3D(true);	
	char countStr[64];
	long count=0;
	TLEList *thisLEList = 0;
	
	if (!triGrid) return; // some error alert, no depth info to check

	GetForeColor(&saveColor);

	#ifdef IBM
		short xtraOffset = 1;
	#else
		short xtraOffset = 0;
	#endif
	
	offset = _max(1,(floatingWidthInPoints*pixelsPerPoint)/2);
	MySetRect(&floatingRect,-offset,-offset,offset,offset);
	offset = _max(1,(beachedWidthInPoints*pixelsPerPoint)/2);
	MySetRect(&beachedRect,-offset,-offset,offset,offset);

	dagTree = triGrid -> GetDagTree();
	if(!dagTree)	return;
	topH = dagTree->GetTopologyHdl();
	if(!topH)	return;
	numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
	numLEsInTri = (LONGH)_NewHandleClear(sizeof(long)*numTri);
	if (!numLEsInTri)
		{ TechError("PtCurMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
	massInTriInGrams = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
	if (!massInTriInGrams)
		{ TechError("PtCurMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
	
	numLESets = model->LESetsList->GetItemCount();
	for (i = 0; i < numLESets; i++)
	{
		model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if (thisLEList->fLeType == UNCERTAINTY_LE)	
			continue;	// don't draw uncertainty for now...
		if (!thisLEList->IsActive()) continue;
		numOfLEs = thisLEList->numOfLEs;
		// density set from API
		//density =  GetPollutantDensity(thisLEList->GetOilType());	
		density = (((TOLEList*)thisLEList)->fSetSummary).density;	
		halfLife = (*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.halfLife;
		massunits = thisLEList->GetMassUnits();

		// time has already been updated at this point
		bShowContours = (*(TOLEList*)thisLEList).fDispersantData.bDisperseOil && model->GetModelTime() - model->GetStartTime() - model->GetTimeStep() >= (*(TOLEList*)thisLEList).fDispersantData.timeToDisperse;
		// code goes here, should total LEs from all spills..., numLEsInTri as a permanent field, maybe should calculate during Step()
		bShowContours = bShowContours || (*(TOLEList*)thisLEList).fAdiosDataH;
		bShowContours = bShowContours || (*(TOLEList*)thisLEList).fSetSummary.z > 0;
		if (bShowContours) 
		{
			bThereIsSubsurfaceOil = true;
			for (j = 0 ; j < numOfLEs ; j++) {
				LongPoint lp;
				long triIndex;
				thisLEList -> GetLE (j, &LE);
				//if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
				//if (LE.statusCode == OILSTAT_EVAPORATED) continue;	// shouldn't happen, temporary for dissolved chemicals 
				if (!(LE.statusCode == OILSTAT_INWATER)) continue;// Windows compiler requires extra parentheses
				lp.h = LE.p.pLong;
				lp.v = LE.p.pLat;
				// will want to calculate individual LE mass for chemicals where particles will dissolve over time
				LEmass = GetLEMass(LE,halfLife);	// will only vary for chemical with different release end time
				massInGrams = VolumeMassToGrams(LEmass, density, massunits);	// need to do this above too
				if (fContourDepth1==BOTTOMINDEX)
				{
					double depthAtLE = DepthAtPoint(LE.p);	
					//if (depthAtLE <= 0) continue;	// occasionally dagtree is messed up
					//if (LE.z > (depthAtLE-1.) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map
					if (LE.z > (depthAtLE-fBottomRange) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map
					{
						triIndex = dagTree -> WhatTriAmIIn(lp);
						//if (triIndex>=0) (*numLEsInTri)[triIndex]++;
						if (triIndex>=0)  
						{
							(*numLEsInTri)[triIndex]++;
							(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
						}
						if (fDiagnosticStrType==SUBSURFACEPARTICLES)
						{
							pt = GetQuickDrawPt(LE.p.pLong,LE.p.pLat,&r,&offQuickDrawPlane);
									
							switch (LE.statusCode) {
								case OILSTAT_INWATER:
									RGBForeColor(&colors[BLACK]);
									leRect = floatingRect;
									MyOffsetRect(&leRect,pt.h,pt.v);
									PaintRect(&leRect);
								break;
							}
						}
					}
				}
				else if (LE.z>fContourDepth1 && LE.z<=fContourDepth2) 
				{
					triIndex = dagTree -> WhatTriAmIIn(lp);
					//if (triIndex>=0) (*numLEsInTri)[triIndex]++;
					if (triIndex>=0) 
					{
						(*numLEsInTri)[triIndex]++;
						(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
					}
					if (fDiagnosticStrType==SUBSURFACEPARTICLES)
					{
						pt = GetQuickDrawPt(LE.p.pLong,LE.p.pLat,&r,&offQuickDrawPlane);
								
						switch (LE.statusCode) {
							case OILSTAT_INWATER:
								if (LE.z > 0)
								{
									//double maxDepth = GetMaxDepth2();	// allow user to set value?
									//double maxDepth = 1500;	// allow user to set value? maybe use the second contour level
									double maxDepth = fContourDepth2;	// check that fContourDepth2 > 0
									long red=0,blue=0,green=0; // roughly 50 - 245 for light black to offwhite
									//double range = 195 / maxDepth, LEcolor = 50 + range * LE.z; 
									double range, LEcolor; 
									//if (LEcolor >= 245) LEcolor = 245;
									RGBColor underWaterColor = ((TOLEList*)thisLEList)->fColor;	// do we want to relate to surface color
									if (fContourDepth2 <= 0) maxDepth = GetMaxDepth2();
									if (maxDepth > 0) range = 195. / maxDepth;
									if (LE.z > maxDepth)
									{
										LEcolor = 245;
									}
									else
									{
										LEcolor = 50. + range * LE.z;
										if (LEcolor > 245) LEcolor = 245;
									}
									red = blue = green = (long) LEcolor;
#ifdef IBM						
									//underWaterColor = RGB(red,green,blue); // minus AH 07/09/2012
									SetRGBColor(&underWaterColor,red,green,blue);
#else						
									RGBColor temp_color; 
									temp_color.red = red*256;
									temp_color.green = green*256;
									temp_color.blue = blue*256;
									underWaterColor = temp_color;		// AH 07/09/2012:
#endif
									//underWaterColor = RGB(red,green,blue);
									inWaterColor = &underWaterColor;

									/*if (LE.z > 100)
									{
										inWaterColor  = &colors[LIGHTGRAY];
									}
									else if (LE.z > 50)
									{
										inWaterColor = &colors[GRAY];
									}
									else
									{
										inWaterColor = &colors[DARKGRAY];
									}*/
								}
								else
								{
									inWaterColor = &(((TOLEList*)thisLEList)->fColor);
								}
								//RGBForeColor(&colors[BLACK]);
								RGBForeColor(inWaterColor);
								leRect = floatingRect;
								MyOffsetRect(&leRect,pt.h,pt.v);
								PaintRect(&leRect);
							break;
						}
					}
				}
			}
		}
	
	}
	
	if (bThereIsSubsurfaceOil && !(fDiagnosticStrType==SUBSURFACEPARTICLES))	// draw LEs in a given layer
	//if (bShowContours && !(fDiagnosticStrType==SUBSURFACEPARTICLES))	// draw LEs in a given layer
	{
		double triArea, triVol, oilDensityInWaterColumn, prevMax=-1;
		long numLEsInTriangle, numLevels;
		double **dosageHdl = 0;
		double concInSelectedTriangles = 0;
		RGBColor col;

		if (!fContourLevelsH)
			if (!InitContourLevels()) return;
		numLevels = GetNumDoubleHdlItems(fContourLevelsH);
		// need to track here if want to display on legend
		if (fTriAreaArray)
			{delete [] fTriAreaArray; fTriAreaArray = 0;}
		fTriAreaArray = new double[numLevels];
		for (i=0;i<numLevels;i++)
			fTriAreaArray[i] = 0.;
		// code goes here, in order to smooth out the blips will have to allow
		// max conc to peak and start to decline, then cap it
		// from then on allow the saved max to get lower, but not higher
		// and stomp out blips by comparing to 
		// this needs to be max over all triangles, not just selected
		// and there may be other issues for the bottom			
		// first time max < prevMax, set global threshold at current max, some sort of flag to change the check
		// then check each time that max <= current max and reset current max to max
		// would also need to be done in the tracking section, hmm
		//if (bUseSmoothing)
			//prevMax = triGrid->GetMaxAtPreviousTimeStep(model->GetModelTime()-model->GetTimeStep());
		if (triGrid->bShowDosage)
		{
			dosageHdl = triGrid->GetDosageHdl(false);
		}
		for (i=0;i<numTri;i++)
		{
			float colorLevel,depthAtPt = -1, depthRange;
			long roundedDepth;
			//WorldPoint centroid = {0,0};
			triArea = (triGrid -> GetTriArea(i)) * 1000 * 1000;	// convert to meters
			//if (!triGrid->GetTriangleCentroidWC(i,&centroid))
			{
				//depthAtPt = DepthAtPoint(centroid);
				depthAtPt = DepthAtCentroid(i);
				if (depthAtPt<=0) 
					roundedDepth = 0;
					//printError("Couldn't find depth at point");
				else
					roundedDepth = floor(depthAtPt) + (depthAtPt - floor(depthAtPt) >= .5 ? 1 : 0);
			}
			//else
				//roundedDepth = 0;
				//printError("Couldn't find centroid");
			if(!(fContourDepth1==BOTTOMINDEX))
			{
				if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt > 0) depthRange = depthAtPt - fContourDepth1;
				else depthRange = fContourDepth2 - fContourDepth1;
			}
			else
			{
				//if (depthAtPt<1 && depthAtPt>0) depthRange = depthAtPt;
				//else depthRange = 1.;	// for bottom layer will always use 1m
				if (depthAtPt<fBottomRange && depthAtPt>0) depthRange = depthAtPt;
				else depthRange = fBottomRange;	// for bottom layer will always use 1m
			}
			//if (depthAtPt < depthRange && depthAtPt != 0) depthRange = depthAtPt;
			//if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt != 0) depthRange = depthAtPt - fContourDepth1;
			triVol = triArea * depthRange; // code goes here, check this depth range is ok at all vertices
			//triVol = triArea * (fContourDepth2 - fContourDepth1); // code goes here, check this depth range is ok at all vertices
			numLEsInTriangle = (*numLEsInTri)[i];

			if (!(fContourDepth1==BOTTOMINDEX))		// need to decide what to do for bottom contour
				if (triGrid->CalculateDepthSliceVolume(&triVol,i,fContourDepth1,fContourDepth2)) goto done;

			oilDensityInWaterColumn = (*massInTriInGrams)[i] / triVol;	// units? milligrams/liter ?? for now gm/m^3

			//if (prevMax > 0 && prevMax < oilDensityInWaterColumn)	// change this to check global max, at each run reset to -1?
				//oilDensityInWaterColumn = prevMax;
			if (triGrid->bShowDosage && dosageHdl)	
			{
				double dosage = (*dosageHdl)[i];
				if (dosage > 2.) 	// need to get some threshold numbers from Alan
				{
					RGBForeColor (&colors[RED]);
					//triGrid->DrawTriangle(&r,i,TRUE,FALSE);	// fill triangles	
					triGrid->DrawTriangle3D(&r,i,TRUE,FALSE);	// fill triangles	
				}
			}
			else
			{
				if (numLEsInTriangle==0)
				{
					if (fDiagnosticStrType==DEPTHATCENTERS)	// uses refined grid to pick center, original to find depth
						//((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,roundedDepth);
						((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,depthAtPt);
					continue;
				}
				for (j=0;j<numLevels;j++)
				{
					colorLevel = float(j)/float(numLevels-1);
					if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
					{	// note: the lowest contour value is not included in count
						fTriAreaArray[j] = fTriAreaArray[j] + triArea/1000000;
				#ifdef IBM		
						col = GetRGBColor(colorLevel);
				#else
						col = GetRGBColor(1.-colorLevel);
				#endif
						//col = GetRGBColor(0.8-colorLevel);
						RGBForeColor(&col);
						//if (!(fDiagnosticStrType==SUBSURFACEPARTICLES))
							//triGrid->DrawTriangle(&r,i,TRUE,FALSE);	// fill triangles	
							triGrid->DrawTriangle3D(&r,i,TRUE,FALSE);	// fill triangles	
						// draw concentration or #LEs as string centered in triangle
						if (fDiagnosticStrType==TRIANGLEAREA /*&& !bShowLegend*/)
							((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,triArea/1000000);
						if (fDiagnosticStrType==NUMLESINTRIANGLE)
							((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,numLEsInTriangle);
						if (fDiagnosticStrType==CONCENTRATIONLEVEL)
							((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,oilDensityInWaterColumn);
						//if (fDiagnosticStrType==DEPTHATCENTERS)
							//((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,roundedDepth);
					}
				}
			}
			if (fDiagnosticStrType==DEPTHATCENTERS)	// uses refined grid to pick center, original to find depth
				//((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,roundedDepth);
				((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,depthAtPt);
			if (((TTriGridVel3D*)triGrid)->fMaxTri==i && ((TTriGridVel3D*)triGrid)->bShowMaxTri) 
			{
				//RGBForeColor (&colors[RED]);
				//triGrid->DrawTriangle(&r,i,TRUE,FALSE);	// fill triangles	
				((TTriGridVel3D*)triGrid)->DrawTriangleStr(&r,i,-1);
			}
		}
	}
	for ( i = 0; i < numLESets; i++)
	{
		model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if (thisLEList->fLeType == UNCERTAINTY_LE)	
			continue;	// don't draw uncertainty for now...
		if (!thisLEList->IsActive()) continue;
		
		numOfLEs = thisLEList->numOfLEs;

		for (j = 0 ; j < numOfLEs ; j++) {
			thisLEList -> GetLE (j, &LE);
			if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
			//if ((LE.z==0 && !bShowSurfaceLEs)) continue;	
			if ((LE.z==0 && bShowSurfaceLEs) || !bThereIsSubsurfaceOil /*|| (LE.z!=0 && fDiagnosticStrType==SUBSURFACEPARTICLES)*/)	// draw LEs colored based on depth, !LE.dispersionStatus==HAVE_DISPERSED
			//if ((LE.z==0 && bShowSurfaceLEs) || !bShowContours /*|| (LE.z!=0 && fDiagnosticStrType==SUBSURFACEPARTICLES)*/)	// draw LEs colored based on depth, !LE.dispersionStatus==HAVE_DISPERSED
			{
				if (!WPointInWRect(LE.p.pLong, LE.p.pLat, &view)) continue;
				
				switch(thisLEList->fLeType)
				{
					case UNCERTAINTY_LE:		// shouldn't happen...
						onLandColor  = &colors[RED];
						inWaterColor = &colors[RED];
						break;
					default:
						onLandColor = &(((TOLEList*)thisLEList)->fColor);
						inWaterColor = &(((TOLEList*)thisLEList)->fColor);
						//onLandColor  = &colors[BLACK];fCOlor
						//inWaterColor = &colors[BLACK];
						break;
				}
				
				pt = GetQuickDrawPt(LE.p.pLong,LE.p.pLat,&r,&offQuickDrawPlane);
						
				switch (LE.statusCode) {
					case OILSTAT_INWATER:
						RGBForeColor(inWaterColor);
						leRect = floatingRect;
						MyOffsetRect(&leRect,pt.h,pt.v);
						PaintRect(&leRect);
					break;
					case OILSTAT_ONLAND:	// shouldn't happen...
						RGBForeColor(onLandColor);
						leRect = beachedRect;
						MyOffsetRect(&leRect,pt.h,pt.v);
						// draw an "X"
						MyMoveTo(leRect.left,leRect.top);
						MyLineTo(leRect.right+xtraOffset,leRect.bottom+xtraOffset);
						MyMoveTo(leRect.left,leRect.bottom);
						MyLineTo(leRect.right+xtraOffset,leRect.top-xtraOffset);
					break;
				}
				/////////////////////////////////////////////////
	
			}
		}
	
	}
	
done:
	RGBForeColor(&saveColor);
	if(numLEsInTri) {DisposeHandle((Handle)numLEsInTri); numLEsInTri=0;}
	if(massInTriInGrams) {DisposeHandle((Handle)massInTriInGrams); massInTriInGrams=0;}
	return;
}

/////////////////////////////////////////////////
Rect PtCurMap::DoArrowTool(long triNum)	
{	// show depth concentration profile at selected triangle
	long n,listIndex,numDepths=0;
	//TLEList	*thisLEList;
	Rect r = MapDrawingRect();
	//TTriGridVel3D* triGrid = GetGrid(false);
	TTriGridVel3D* triGrid = GetGrid3D(true);	// since output data is stored on the refined grid need to used it here
	OSErr err = 0;
	TMover *mover=0;
	double depthAtPt=0;
	Boolean needToRefresh=false;

	if (!triGrid) return r;

	mover = this->GetMover(TYPE_TRICURMOVER);
	if (mover)
	{
		numDepths =  (dynamic_cast<TriCurMover*>(mover)) -> CreateDepthSlice(triNum,&fDepthSliceArray);
		//numDepths = ((TriCurMover*)mover) -> CreateDepthSlice(triNum,fDepthSliceArray);
		if (numDepths > 0) goto drawPlot; else return r;
	}
	//for (listIndex = 0, n = model->LESetsList -> GetItemCount (); listIndex < n; listIndex++)
	//{
		//double depthAtPt=0;
		//WorldPoint centroid = {0,0};
		//model->LESetsList -> GetListItem ((Ptr) &thisLEList, listIndex);
		// check that a list is dispersed, if not no point
		//if (!(*(TOLEList*)thisLEList).fDispersantData.bDisperseOil && !(*(TOLEList*)thisLEList).fAdiosDataH && !(*(TOLEList*)thisLEList).fSetSummary.z > 0) 
			//continue;
		// also must be LEs in that triangle
	
		if (triNum < 0)
		{
			//if (GetDepthAtMaxTri(((TOLEList*)thisLEList),&triNum,&depthAtPt)) return r;
			if (GetDepthAtMaxTri(&triNum,&depthAtPt)) return r;
		}
		else
		{
			//if (!triGrid->GetTriangleCentroidWC(triNum,&centroid))
			{
				//depthAtPt = DepthAtPoint(centroid);
				depthAtPt = DepthAtCentroid(triNum);
				if (depthAtPt < 0) depthAtPt = 100;
			}
			//else return r;
		}
		// code goes here, at the bottom have to consider that LE.z could be greater than depth at triangle center
		numDepths = floor(depthAtPt)+1;	// split into 1m increments to track vertical slice

		//if (numDepths>0) err = CreateDepthSlice(thisLEList,triNum);
		if (numDepths>0) err = CreateDepthSlice(triNum,&fDepthSliceArray);
		//if (numDepths>0) err = CreateDepthSlice(triNum,fDepthSliceArray);
		if (!err) {triGrid -> fMaxTri = triNum; needToRefresh = true;/*triGrid -> bShowMaxTri = true;*/}
		// only do for first if more than one
		//if (!err && numDepths>0) break;
	//}
drawPlot:
	if (numDepths>0 && !err)
	{
		Boolean **triSelected = triGrid -> GetTriSelection(false);	// initialize if necessary
		outputData **oilConcHdl = triGrid -> GetOilConcHdl();	
		float depthRange1 = fContourDepth1, depthRange2 = fContourDepth2, bottomRange = fBottomRange;
		// should break out once a list is found and bring up the graph
		if (triSelected) 	// tracked output at a specified area
			PlotDialog(oilConcHdl,fDepthSliceArray,depthRange1,depthRange2,bottomRange,true,true);
		else 	// tracked output following the plume
			PlotDialog(oilConcHdl,fDepthSliceArray,depthRange1,depthRange2,bottomRange,false,true);
		if (needToRefresh == true)
		{
			InvalidateMapImage();// invalidate the offscreen bitmap
			InvalMapDrawingRect();
		}
	}
	else
		SysBeep(5);

	return r;
}

void PtCurMap::MarkRect(Point p)
{
	Point where;
	Rect r;
	WorldRect check;
	long i,n,currentScale = CurrentScaleDenominator();
	long mostZoomedInScale = MostZoomedInScale();
	TLEList *thisLEList;
	LETYPE leType;
	ClassID thisClassID;
	Boolean foundLEList = 0;	
	DispersionRec dispInfo;
	
	MySetCursor(1005);
	for (i = 0, n = model->LESetsList->GetItemCount() ; i < n; i++) 
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !model->IsUncertain()) continue;
		
		thisClassID = thisLEList -> GetClassID();
		if(thisClassID == TYPE_OSSMLELIST || thisClassID == TYPE_SPRAYLELIST )
		{
			TOLEList *thisOLEList = (TOLEList*)thisLEList; // typecast
			foundLEList = true;
			break;
			// may want to require spill has not already been chemically dispersed? 
			// also check natural dispersion
			//if (thisOLEList->fDispersantData.bDisperseOil) bSomeSpillIsDispersed = true;
			//Seconds thisStartTime = thisOLEList->fSetSummary.startRelTime;
		}
	}
	if (!foundLEList) return;
	//r = DefineGrayRect(p, ZoomRectAction, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE);
	r = DefineGrayRect(p, ZoomRectAction, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE);
	
	if(currentScale <= (long)(1.01*mostZoomedInScale)) // JLM 12/19/97
	{ SysBeep(1); return;}

	if (RectWidth(r) > 2 && RectHeight(r) > 2) {
		check = ScreenToWorldRect(r, MapDrawingRect(), settings.currentView);
		if (WRectWidth(check) < MINWRECTDIST || WRectHeight(check) < MINWRECTDIST) { SysBeep(1); return ; }
		if (WRectWidth(check) > MAXWRECTDIST || WRectHeight(check) > MAXWRECTDIST) { SysBeep(1); return ; }
		//ChangeCurrentView(check, TRUE, TRUE);
	}
	else
		return;
		/*if (r.right > 1) {
			where.h = r.left;
			where.v = r.top;
			MagnifyTool(where, -ZOOMPLUSTOOL);
		}*/
	
	// bring up dialog to set api, maybe dispersion duration
	dispInfo = ((TOLEList*)thisLEList) -> GetDispersionInfo();
	// if already dispersing force to restart?
	dispInfo.bDisperseOil = 1;
	dispInfo.timeToDisperse = model->GetModelTime() - model->GetStartTime() + model->GetTimeStep();
	dispInfo.amountToDisperse = 1;
	dispInfo.duration = 0;
	dispInfo.areaToDisperse = check; 
	dispInfo.lassoSelectedLEsToDisperse = false;
	//set to entire map or could set area via the polygon - then area can't be a rect...
	//double api - undo from oil type, could set by hand, ignore for now since density already set;
	((TOLEList*)thisLEList) -> SetDispersionInfo(dispInfo); 
	((TOLEList*)thisLEList) -> bShowDispersantArea = true;
	//InvalidateMapImage();// invalidate the offscreen bitmap
	InvalMapDrawingRect();
	//model->NewDirtNotification(DIRTY_LIST);
	// make sure to redraw the screen
}

void PtCurMap::DoLassoTool(Point p)
{
	// require there is a spill, not already chemically dispersed ?
	// if more than one spill ? use up/down arrows to change top spill
	// if in mid-run, expect user wants to select LEs and disperse immediately - bring up dialog
	// will need to set duration, api, anything else? - effectiveness, but this may be a property of the spill
	long i, j, n, numsegs, numLEs, count = 0;
	Point newPoint,startPoint;
	WorldPoint w;
	WORLDPOINTH wh=0;
	TLEList *thisLEList;
	LETYPE leType;
	ClassID thisClassID;
	Boolean foundLEList = 0;
	SEGMENTH poly = 0;
	double x, effectiveness = 100.;
	OSErr err = 0;

	// code goes here, lasso should apply to all LEs it captures, which may be multiple spills
	// code goes here, bring up a dialog to ask user for effectiveness, but won't let them change the value
	for (i = 0, n = model->LESetsList->GetItemCount() ; i < n && !err; i++) 
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !model->IsUncertain()) continue;
		
		thisClassID = thisLEList -> GetClassID();
		if(thisClassID == TYPE_OSSMLELIST || thisClassID == TYPE_SPRAYLELIST )
		{
			TOLEList *thisOLEList = (TOLEList*)thisLEList; // typecast
			foundLEList = true;
			break;
			// may want to require spill has not already been chemically dispersed? 
			// also check natural dispersion
			//if (thisOLEList->fDispersantData.bDisperseOil) bSomeSpillIsDispersed = true;
			//Seconds thisStartTime = thisOLEList->fSetSummary.startRelTime;
		}
	}
	if (!foundLEList) { printNote("Set a spill before using lasso tool to disperse oil"); return;}

	startPoint = p;
	MyMoveTo(p.h,p.v);
	w = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
	AppendToWORLDPOINTH(&wh,&w);	
	while(StillDown()) 
	{
		GetMouse(&newPoint);
		if((newPoint.h != p.h || newPoint.v != p.v))
		{
			MyLineTo(newPoint.h, newPoint.v);
			p = newPoint;
			w = ScreenToWorldPoint(p, MapDrawingRect(), settings.currentView);
			if(!AppendToWORLDPOINTH(&wh,&w))goto Err;
		}
	}
	w = ScreenToWorldPoint(startPoint, MapDrawingRect(), settings.currentView);
	AppendToWORLDPOINTH(&wh,&w);
	poly = WPointsToSegments(wh,_GetHandleSize((Handle)(wh))/sizeof(WorldPoint),&numsegs);
	if (!poly) goto Err;

	// loop over the LEs and check if they are in the polygon, if so mark to disperse
	// disperse immediately (at current time) and instantaneously, though maybe over an hour to test
	for (i = 0, n = model->LESetsList->GetItemCount() ; i < n && !err; i++) 
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !model->IsUncertain()) continue;
		
		thisClassID = thisLEList -> GetClassID();
		if(thisClassID == TYPE_OSSMLELIST || thisClassID == TYPE_SPRAYLELIST )
		{
			long numToDisperse=0;
			Boolean alreadyUsingLasso = false;
			WorldPoint w;
			DispersionRec dispInfo = ((TOLEList*)thisLEList) -> GetDispersionInfo();
			LERec theLE;
			if (poly != 0)
			{
				// bring up dialog to set api, maybe dispersion duration, effectiveness
				numLEs = thisLEList->GetLECount();
				for(j=0; j < numLEs; j++) // make this numLEs
				{
					thisLEList -> GetLE (j, &theLE);
					//code goes here, should total all LEs selected at different times to get amount
					// also if selecting LEs at earlier time than others, deselect the later ones?
					// will eventually need to be able to edit the lassoed regions
					// check if already dispersed
					if (theLE.dispersionStatus == HAVE_DISPERSED || theLE.dispersionStatus == HAVE_DISPERSED_NAT || theLE.dispersionStatus == HAVE_EVAPORATED) continue;
					w.pLong = theLE.p.pLong;
					w.pLat = theLE.p.pLat;
					if (PointInPolygon(w,poly,numsegs,true))	// true -> holes ??
					{
						count++;
						if (count==1) {GetScaleFactorFromUser("Input dispersant effectiveness as a decimal (0 to 1)",&effectiveness);
						if (effectiveness > 1) effectiveness = 1;}
						x = GetRandomFloat(0, 1.0);
						if (x <= effectiveness)
						{
						theLE.dispersionStatus = DISPERSE;	// but gets reset
						theLE.beachTime = model->GetModelTime();	// use for time to disperse
						numToDisperse++;
						thisLEList -> SetLE (j, &theLE);
						}
					}
				}
				if (dispInfo.lassoSelectedLEsToDisperse) alreadyUsingLasso = true;
				if (numToDisperse>0)
				{
					dispInfo.bDisperseOil = 1;
					//if (alreadyUsingLasso && dispInfo.timeToDisperse < model->GetModelTime() - model->GetStartTime())
					if (alreadyUsingLasso && dispInfo.timeToDisperse < model->GetModelTime() - ((TOLEList*)thisLEList) ->fSetSummary.startRelTime)
					{
					}
					else
						//dispInfo.timeToDisperse = model->GetModelTime() - model->GetStartTime();
						dispInfo.timeToDisperse = model->GetModelTime() - ((TOLEList*)thisLEList) ->fSetSummary.startRelTime;
					dispInfo.amountToDisperse = (float)numToDisperse/(float)numLEs;
					dispInfo.duration = 0;
					dispInfo.areaToDisperse = GetMapBounds(); 
					dispInfo.lassoSelectedLEsToDisperse = true;
					//set to entire map or could set area via the polygon - then area can't be a rect...
					//double api - undo from oil type, could set by hand, ignore for now since density already set;
					((TOLEList*)thisLEList) -> SetDispersionInfo(dispInfo); 
				}
			}
		}
	}

Err:
	if(wh) {DisposeHandle((Handle)wh); wh=0;}	// may want to save this to draw or whatever
	model->NewDirtNotification(DIRTY_LIST);
	return;
}
/////////////////////////////////////////////////
void PtCurMap::SetSelectedBeach(LONGH *segh, LONGH selh)
{
	long n=0,i,k,next_k;
	long numSelectedBoundaryPts;
	Boolean pointsAdded = false;
	OSErr err = 0;

	if (selh) n = _GetHandleSize((Handle)selh)/sizeof(**selh);
	else return;
	if (n<1) return;
	// at this point only boundary tool can be used so all this is doing is copying one handle to another
	if((*selh)[n-1]==-1)	// boundary tool was used, all is well
	{
		for(i=0;i<n;i++)
		{
			k = (*selh)[i];
			AppendToLONGH(segh,k);
		}
	}
	else	// arrow tool was used (or both were used)
	{
		for(i=0;i<n-1;i++)
		{
			// make sure the selected points are boundary points 
			// and mark a segment or boundary switch with -1
			k = (*selh)[i];
			next_k = (*selh)[i+1];
			if (!IsBoundaryPoint(k)) 
			{
				err = -2;
				continue;
			}
			if (k==-1) continue;
			if (next_k==-1)
			{
				AppendToLONGH(segh,next_k);
				pointsAdded = false;
				continue;
			}
			if (ContiguousPoints(k,next_k))
			{
				if (!pointsAdded) 	// first point of segment
					AppendToLONGH(segh,k);
				AppendToLONGH(segh,next_k);
				pointsAdded=true;
			}
			else 
			{
				if(pointsAdded)
				{
					AppendToLONGH(segh,-1);
					pointsAdded=false;
					if (i==n-2 && err==0) err = -3;
				}
				else
					// otherwise skipping point
					if (err==0) err = -3;
			}
		}
		//numWaterBoundaryPts = GetNumLONGHItems(*segh);
		if (*segh) numSelectedBoundaryPts = _GetHandleSize((Handle)(*segh))/sizeof(***segh);
		if (numSelectedBoundaryPts>0 && (**segh)[numSelectedBoundaryPts-1]!=-1)
			AppendToLONGH(segh,-1);	// mark end of selected segment 
		if (n==1)
			printError("An isolated boundary point was selected. No water boundary will be set.");
		if (err == -2)
			printError("Non boundary points were selected and will be ignored");
		else if (err == -3)
			printError("Non contiguous boundary points were selected and will be ignored");
	}	
}  

void PtCurMap::SetBeachSegmentFlag(LONGH *beachBoundaryH, long *numBeachBoundaries)
{
	// rearranging points to parallel water boundaries to simplify drawing
	// code goes here, keep some sort of pointer to go back from new ordering to old ordering
	// that way can draw plots in order points were selected rather than in numerical order
	long i, startIndex, endIndex, curIndex=0, p, afterP;
	long segNo,lastPtOnSeg,firstPtOnSeg;
	long numBoundaryPts = GetNumBoundaryPts();

	for(i=0;i<numBoundaryPts;i++)
	{
		(**beachBoundaryH)[i] = 0;
	}

	while(MoreSegments(fSelectedBeachHdl,&startIndex, &endIndex, &curIndex))
	{
		if(endIndex <= startIndex)continue;

		segNo = PointOnWhichSeg((*fSelectedBeachHdl)[startIndex]);
		firstPtOnSeg = segNo == 0 ? 0: (*fBoundarySegmentsH)[segNo-1] + 1;
		lastPtOnSeg = (*fBoundarySegmentsH)[segNo];
		for(i=startIndex; i< endIndex -1; i++)
		{
			p = (*fSelectedBeachHdl)[i];
			afterP = (*fSelectedBeachHdl)[i+1];
			// segment endpoint indicates whether segment is selected
			if ((p<afterP && !(p==firstPtOnSeg && afterP==lastPtOnSeg)) || (afterP==firstPtOnSeg && p==lastPtOnSeg))
			{
				(**beachBoundaryH)[afterP]=1;
				*numBeachBoundaries += 1;
			}
			else if ((p>afterP && !(afterP==firstPtOnSeg && p==lastPtOnSeg)) || (p==firstPtOnSeg && afterP==lastPtOnSeg))
			{
				(**beachBoundaryH)[p]=1;
				*numBeachBoundaries += 1;
			}
		}
	}
} 

void PtCurMap::ClearSelectedBeach()
{
	MyDisposeHandle((Handle*)&fSegSelectedH);
	MyDisposeHandle((Handle*)&fSelectedBeachHdl);
	MyDisposeHandle((Handle*)&fSelectedBeachFlagHdl);
}

/////////////////////////////////////////////////


/**************************************************************************************************/
/*typedef struct ConcTriNumPair
{
	double conc;
	long triNum;
} ConcTriNumPair, *ConcTriNumPairP, **ConcTriNumPairH;
*/

int ConcentrationCompare(void const *x1, void const *x2)
{
	ConcTriNumPair *p1,*p2;	
	p1 = (ConcTriNumPair*)x1;
	p2 = (ConcTriNumPair*)x2;
	
	if (p1->conc < p2->conc) 
		return -1;  // first less than second
	else if (p1->conc > p2->conc)
		return 1;
	else return 0;// equivalent
	
}

//void PtCurMap::TrackOutputData(TOLEList *thisLEList)
// might want to track each spill separately and combined
void PtCurMap::TrackOutputData(void)
{	// need all LELists
	long i, j, numOfLEs, numLESets, numTri;
	LERec LE;
	Boolean bShowContours, bTimeZero, bTimeToOutputData = false, bThereIsSubsurfaceOil = false;
	LONGH numLEsInTri = 0;
	DOUBLEH massInTriInGrams = 0;
	TopologyHdl topH = 0;
	TDagTree *dagTree = 0;
	DOUBLEH /*concentrationH = 0,*/ dosageHdl = 0;
	ConcTriNumPairH concentrationH = 0;
	TTriGridVel3D* triGrid = GetGrid3D(true);	
	Seconds modelTime = model->GetModelTime(),timeStep = model->GetTimeStep();
	Seconds startTime = model->GetStartTime();
	short oldIndex, nowIndex, massunits;
	double depthAtPt, density, LEmass, massInGrams, halfLife;
	TLEList *thisLEList = 0;
	
	if (!triGrid) return; // some error alert, no depth info to check

	Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init

	dagTree = triGrid -> GetDagTree();
	if(!dagTree)	return;
	topH = dagTree->GetTopologyHdl();
	if(!topH)	return;
	numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
	numLEsInTri = (LONGH)_NewHandleClear(sizeof(long)*numTri);
	if (!numLEsInTri)
		{ TechError("PtCurMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
	massInTriInGrams = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
	if (!massInTriInGrams)
		{ TechError("PtCurMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
	//concentrationH = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
	concentrationH = (ConcTriNumPairH)_NewHandleClear(sizeof(ConcTriNumPair)*numTri);
	if (!concentrationH)
		{ TechError("PtCurMap::DrawContour()", "_NewHandleClear()", 0); goto done; }

	//bTimeZero = (modelTime == /*startTime*/(*(TOLEList*)thisLEList).fDispersantData.timeToDisperse+startTime);
	nowIndex = (modelTime + timeStep - startTime) / model->LEDumpInterval;
	oldIndex = (modelTime /*- timeStep*/ - startTime) / model->LEDumpInterval;
	if(nowIndex > oldIndex /*|| bTimeZero*/) bTimeToOutputData = true;

	numLESets = model->LESetsList->GetItemCount();	
	for (i = 0; i < numLESets; i++)
	{
		model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if (thisLEList->fLeType == UNCERTAINTY_LE)	
			continue;	// don't draw uncertainty for now...
		if (!thisLEList->IsActive()) continue;
		numOfLEs = thisLEList->numOfLEs;
		// density set from API
		//density =  GetPollutantDensity(thisLEList->GetOilType());	
		density = ((TOLEList*)thisLEList)->fSetSummary.density;	
		halfLife = (*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.halfLife;
		massunits = thisLEList->GetMassUnits();

		if (bTimeToOutputData)	// track budget at the same time
		{	// make budget table even if spill is not dispersed
			double amttotal,amtevap,amtbeached,amtoffmap,amtfloating,amtreleased,amtdispersed,amtremoved=0;
			Seconds timeAfterSpill;	
			// what to do for chemicals, amount dissolved?
			thisLEList->GetLEAmountStatistics(thisLEList->GetMassUnits(),&amttotal,&amtreleased,&amtevap,&amtdispersed,&amtbeached,&amtoffmap,&amtfloating,&amtremoved);
			BudgetTableData budgetTable; 
			// if chemical will need to get amount dissolved
			timeAfterSpill = nowIndex * model->LEDumpInterval;
			budgetTable.timeAfterSpill = timeAfterSpill;
			budgetTable.amountReleased = amtreleased;
			budgetTable.amountFloating = amtfloating;
			budgetTable.amountDispersed = amtdispersed;
			budgetTable.amountEvaporated = amtevap;
			budgetTable.amountBeached = amtbeached;
			budgetTable.amountOffMap = amtoffmap;
			budgetTable.amountRemoved = amtremoved;
			((TOLEList*)thisLEList)->AddToBudgetTableHdl(&budgetTable);
			// still track for each list, for output total everything
		}

		// time has not been updated at this point (in DrawContours time has been updated)
		bShowContours = (*(TOLEList*)thisLEList).fDispersantData.bDisperseOil && model->GetModelTime() - model->GetStartTime() >= (*(TOLEList*)thisLEList).fDispersantData.timeToDisperse;
		bShowContours = bShowContours || (*(TOLEList*)thisLEList).fAdiosDataH;
		bShowContours = bShowContours || (*(TOLEList*)thisLEList).fSetSummary.z > 0;	// for bottom spill
		if (bShowContours) bThereIsSubsurfaceOil = true;
	/*{// put in a field to show or not the elapsed time (bShowElapsedTime - a model field?)
		// possibly increase times by a timestep here, are we really at the next step?
		char msg[256];
		float time1,time2;
		time1 = (model->GetModelTime() - model->GetStartTime())/3600.;
		char time1Str[64],time2Str[64];
		StringWithoutTrailingZeros(time1Str,time1,2);
		if (bShowContours)
		{
			time2 = time1 - (*(TOLEList*)thisLEList).fDispersantData.timeToDisperse/3600.;
			StringWithoutTrailingZeros(time2Str,time2,2);
			sprintf(msg,"Elapsed time = %s hrs, Dispersed %s hrs ago",time1Str,time2Str);
		}
		else
			sprintf(msg,"Elapsed time = %s hrs", time1Str);
		DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage(msg); 
	}*/
		if (!bShowContours) continue;	// no need to track in this case
		// total LEs from all spills..., numLEsInTri as a permanent field, maybe should calculate during Step()
		for (j = 0 ; j < numOfLEs ; j++) {
			LongPoint lp;
			long triIndex;
			thisLEList -> GetLE (j, &LE);
			//if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
			//if (LE.statusCode == OILSTAT_EVAPORATED) continue;	// shouldn't happen, temporary for dissolved chemicals 
			if (!(LE.statusCode == OILSTAT_INWATER)) continue;	// Windows compiler requires extra parentheses
			lp.h = LE.p.pLong;
			lp.v = LE.p.pLat;
			LEmass = GetLEMass(LE,halfLife);	// will only vary for chemical with different release end time
			massInGrams = VolumeMassToGrams(LEmass, density, massunits);	// need to do this above too
			if (fContourDepth1==BOTTOMINDEX)
			{
				double depthAtLE = DepthAtPoint(LE.p);
				//if (LE.z > (depthAtLE-1.) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
				if (LE.z > (depthAtLE-fBottomRange) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
				{
					triIndex = dagTree -> WhatTriAmIIn(lp);
					if (triIndex>=0)  
					{
						(*numLEsInTri)[triIndex]++;
						(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
					}
				}
			}
			else if (LE.z>fContourDepth1 && LE.z<=fContourDepth2) 
			{
				triIndex = dagTree -> WhatTriAmIIn(lp);
				if (triIndex>=0) 
				{
					(*numLEsInTri)[triIndex]++;
					(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
				}
			}
		}
	}
	if (triGrid->bCalculateDosage)
	{
		dosageHdl = triGrid -> GetDosageHdl(true);
	}
	if (bThereIsSubsurfaceOil)
	{
		double triArea, triVol, oilDensityInWaterColumn, totalVol=0;
		long numLEsInTriangle,j,numLevels,totalLEs=0,totalMass=0,numDepths=0,maxTriNum=-1,numTrisWithOil = 0;
		double concInSelectedTriangles=0,maxConc=0,numTrisSelected=0,minLevel,maxLevel,range,avConcOverTriangles=0;
		if (!fContourLevelsH)
			if (!InitContourLevels()) return;
		numLevels = GetNumDoubleHdlItems(fContourLevelsH);
		if (fTriAreaArray)
			{delete [] fTriAreaArray; fTriAreaArray = 0;}
		fTriAreaArray = new double[numLevels];
		for (i=0;i<numLevels;i++)
			fTriAreaArray[i] = 0.;
		for (i=0;i<numTri;i++)
		{	// track ppm hours here
			double depthRange;
			depthAtPt=0;
			//WorldPoint centroid = {0,0};
			if (triSelected && !(*triSelected)[i]) continue;	// note this line keeps triareaarray from tracking for output..., dosage...
			//if (!triGrid->GetTriangleCentroidWC(i,&centroid))
			{
				//depthAtPt = DepthAtPoint(centroid);
				depthAtPt = DepthAtCentroid(i);
			}
			triArea = (triGrid -> GetTriArea(i)) * 1000 * 1000;	// convert to meters
			if (!(fContourDepth1==BOTTOMINDEX))	
			{
				if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt > 0) depthRange = depthAtPt - fContourDepth1;
				else depthRange = fContourDepth2 - fContourDepth1;
			}
			else
			{
				//if (depthAtPt<1 && depthAtPt>0) depthRange = depthAtPt;	// should do a triangle volume 
				//else depthRange = 1.; // for bottom will always contour 1m
				if (depthAtPt<fBottomRange && depthAtPt>0) depthRange = depthAtPt;	// should do a triangle volume 
				else depthRange = fBottomRange; // for bottom will always contour 1m
			}
			//if (depthAtPt < depthRange && depthAtPt != 0) depthRange = depthAtPt;
			//if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt != 0) depthRange = depthAtPt - fContourDepth1;
			triVol = triArea * depthRange; // code goes here, check this depth range is ok at all vertices
			//triVol = triArea * (fContourDepth2 - fContourDepth1); // code goes here, check this depth range is ok at all vertices
			numLEsInTriangle = (*numLEsInTri)[i];
			if (numLEsInTriangle==0)
				continue;

			if (!(fContourDepth1==BOTTOMINDEX))		// need to decide what to do for bottom contour
				if (triGrid->CalculateDepthSliceVolume(&triVol,i,fContourDepth1,fContourDepth2)) goto done;
			oilDensityInWaterColumn = (*massInTriInGrams)[i] / triVol;	// units? milligrams/liter ?? for now gm/m^3

			//(*concentrationH)[numTrisWithOil] = oilDensityInWaterColumn;
			(*concentrationH)[numTrisWithOil].conc = oilDensityInWaterColumn;
			(*concentrationH)[numTrisWithOil].triNum = i;
			numTrisWithOil++;
			//for (j=0;j<numLevels-1;j++)
			for (j=0;j<numLevels;j++)
			{
				//if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
				if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
				{
					fTriAreaArray[j] = fTriAreaArray[j] + triArea/1000000;
					totalLEs += numLEsInTriangle;
					//totalMass += massInGrams;
					totalMass += (*massInTriInGrams)[i];
					totalVol += triVol;
					concInSelectedTriangles += oilDensityInWaterColumn;	// sum or track each one?
					if (oilDensityInWaterColumn > maxConc) 
					{
						maxConc = oilDensityInWaterColumn;
						numDepths = floor(depthAtPt)+1;	// split into 1m increments to track vertical slice
						maxTriNum = i;
					}
					numTrisSelected++;
				}
			}
			if (triGrid->bCalculateDosage && dosageHdl)
			{
				double dosage;
				(*dosageHdl)[i] += oilDensityInWaterColumn * timeStep / 3600;	// 
				dosage = (*dosageHdl)[i];
			}
		}
		//_SetHandleSize((Handle) concentrationH, (numTrisWithOil)*sizeof(double));
		if (numTrisWithOil>0)
		{
			_SetHandleSize((Handle) concentrationH, (numTrisWithOil)*sizeof(ConcTriNumPair));
			if (triGrid->fPercentileForMaxConcentration < 1)	
			{
				//qsort((*concentrationH),numTrisWithOil,sizeof(double),ConcentrationCompare);
				qsort((*concentrationH),numTrisWithOil,sizeof(ConcTriNumPair),ConcentrationCompare);
				j = (long)(triGrid->fPercentileForMaxConcentration*numTrisWithOil);	// round up or down?, for selected triangles?
				if (j>0) j--;
				//maxConc = (*concentrationH)[j];	// trouble with this percentile stuff if there are only a few values
				maxConc = (*concentrationH)[j].conc;	// trouble with this percentile stuff if there are only a few values
				maxTriNum = (*concentrationH)[j].triNum;	// trouble with this percentile stuff if there are only a few values
			}
		}
		//if (thisLEList->GetOilType() == CHEMICAL) 
		if (totalVol>0) avConcOverTriangles = totalMass / totalVol;
		//else
			//avConcOverTriangles = totalLEs * massInGrams / totalVol;
		// track concentrations over time, maybe define a new data type to hold everything...
		//if (triSelected) triGrid -> AddToOutputHdl(numTrisSelected>0 ? concInSelectedTriangles/numTrisSelected : 0,maxConc,model->GetModelTime());
		if (triSelected) triGrid -> AddToOutputHdl(numTrisSelected>0 ? avConcOverTriangles : 0,maxConc,model->GetModelTime());
		else triGrid -> AddToOutputHdl(numTrisSelected>0 ? avConcOverTriangles : 0, maxConc, model->GetModelTime());
		//if (triSelected) triGrid -> AddToOutputHdl(oilDensityInWaterColumn,model->GetModelTime());
		if (/*fDiagnosticStrType==TRIANGLEAREA &&*/ bTimeToOutputData) // this will be messed up if there are triangles selected
			triGrid -> AddToTriAreaHdl(fTriAreaArray,numLevels);
		CreateDepthSlice(maxTriNum,&fDepthSliceArray);
		//CreateDepthSlice(maxTriNum,fDepthSliceArray);
		triGrid -> fMaxTri = maxTriNum; /*triGrid -> bShowMaxTri = true;*/
	}
	
done:
	if(numLEsInTri) {DisposeHandle((Handle)numLEsInTri); numLEsInTri=0;}
	if(massInTriInGrams) {DisposeHandle((Handle)massInTriInGrams); massInTriInGrams=0;}
	if(concentrationH) {DisposeHandle((Handle)concentrationH); concentrationH=0;}
	return;
}
/////////////////////////////////////////////////
void PtCurMap::TrackOutputDataInAllLayers(void)
{	// need all LELists
	long i, j, numOfLEs, numLESets, numTri;
	LERec LE;
	Boolean bShowContours, bTimeZero, bTimeToOutputData = false, bThereIsSubsurfaceOil = false;
	LONGH numLEsInTri = 0;
	DOUBLEH massInTriInGrams = 0;
	TopologyHdl topH = 0;
	TDagTree *dagTree = 0;
	DOUBLEH /*concentrationH = 0,*/ dosageHdl = 0;
	ConcTriNumPairH concentrationH = 0;
	TTriGridVel3D* triGrid = GetGrid3D(true);	
	Seconds modelTime = model->GetModelTime(),timeStep = model->GetTimeStep();
	Seconds startTime = model->GetStartTime();
	short oldIndex, nowIndex, massunits;
	double depthAtPt, density, LEmass, massInGrams, halfLife;
	TLEList *thisLEList = 0;
	
	if (!triGrid) return; // some error alert, no depth info to check

	Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init

	dagTree = triGrid -> GetDagTree();
	if(!dagTree)	return;
	topH = dagTree->GetTopologyHdl();
	if(!topH)	return;
	numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
	numLEsInTri = (LONGH)_NewHandleClear(sizeof(long)*numTri);
	if (!numLEsInTri)
		{ TechError("PtCurMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
	massInTriInGrams = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
	if (!massInTriInGrams)
		{ TechError("PtCurMap::DrawContour()", "_NewHandleClear()", 0); goto done; }
	//concentrationH = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
	concentrationH = (ConcTriNumPairH)_NewHandleClear(sizeof(ConcTriNumPair)*numTri);
	if (!concentrationH)
		{ TechError("PtCurMap::DrawContour()", "_NewHandleClear()", 0); goto done; }

	//bTimeZero = (modelTime == /*startTime*/(*(TOLEList*)thisLEList).fDispersantData.timeToDisperse+startTime);
	nowIndex = (modelTime + timeStep - startTime) / model->LEDumpInterval;
	oldIndex = (modelTime /*- timeStep*/ - startTime) / model->LEDumpInterval;
	if(nowIndex > oldIndex /*|| bTimeZero*/) bTimeToOutputData = true;

	numLESets = model->LESetsList->GetItemCount();	
	for (i = 0; i < numLESets; i++)
	{
		model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if (thisLEList->fLeType == UNCERTAINTY_LE)	
			continue;	// don't draw uncertainty for now...
		if (!thisLEList->IsActive()) continue;
		numOfLEs = thisLEList->numOfLEs;
		// density set from API
		//density =  GetPollutantDensity(thisLEList->GetOilType());	
		density = ((TOLEList*)thisLEList)->fSetSummary.density;	
		halfLife = (*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.halfLife;
		massunits = thisLEList->GetMassUnits();

		if (bTimeToOutputData)	// track budget at the same time
		{	// make budget table even if spill is not dispersed
			double amttotal,amtevap,amtbeached,amtoffmap,amtfloating,amtreleased,amtdispersed,amtremoved=0;
			Seconds timeAfterSpill;	
			// what to do for chemicals, amount dissolved?
			thisLEList->GetLEAmountStatistics(thisLEList->GetMassUnits(),&amttotal,&amtreleased,&amtevap,&amtdispersed,&amtbeached,&amtoffmap,&amtfloating,&amtremoved);
			BudgetTableData budgetTable; 
			// if chemical will need to get amount dissolved
			timeAfterSpill = nowIndex * model->LEDumpInterval;
			budgetTable.timeAfterSpill = timeAfterSpill;
			budgetTable.amountReleased = amtreleased;
			budgetTable.amountFloating = amtfloating;
			budgetTable.amountDispersed = amtdispersed;
			budgetTable.amountEvaporated = amtevap;
			budgetTable.amountBeached = amtbeached;
			budgetTable.amountOffMap = amtoffmap;
			budgetTable.amountRemoved = amtremoved;
			((TOLEList*)thisLEList)->AddToBudgetTableHdl(&budgetTable);
			// still track for each list, for output total everything
		}

		// time has not been updated at this point (in DrawContours time has been updated)
		bShowContours = (*(TOLEList*)thisLEList).fDispersantData.bDisperseOil && model->GetModelTime() - model->GetStartTime() >= (*(TOLEList*)thisLEList).fDispersantData.timeToDisperse;
		bShowContours = bShowContours || (*(TOLEList*)thisLEList).fAdiosDataH;
		bShowContours = bShowContours || (*(TOLEList*)thisLEList).fSetSummary.z > 0;	// for bottom spill
		if (bShowContours) bThereIsSubsurfaceOil = true;
	/*{// put in a field to show or not the elapsed time (bShowElapsedTime - a model field?)
		// possibly increase times by a timestep here, are we really at the next step?
		char msg[256];
		float time1,time2;
		time1 = (model->GetModelTime() - model->GetStartTime())/3600.;
		char time1Str[64],time2Str[64];
		StringWithoutTrailingZeros(time1Str,time1,2);
		if (bShowContours)
		{
			time2 = time1 - (*(TOLEList*)thisLEList).fDispersantData.timeToDisperse/3600.;
			StringWithoutTrailingZeros(time2Str,time2,2);
			sprintf(msg,"Elapsed time = %s hrs, Dispersed %s hrs ago",time1Str,time2Str);
		}
		else
			sprintf(msg,"Elapsed time = %s hrs", time1Str);
		DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage(msg); 
	}*/
		if (!bShowContours) continue;	// no need to track in this case
		// total LEs from all spills..., numLEsInTri as a permanent field, maybe should calculate during Step()
		for (j = 0 ; j < numOfLEs ; j++) {
			LongPoint lp;
			long triIndex;
			thisLEList -> GetLE (j, &LE);
			//if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
			//if (LE.statusCode == OILSTAT_EVAPORATED) continue;	// shouldn't happen, temporary for dissolved chemicals 
			if (!(LE.statusCode == OILSTAT_INWATER)) continue;	// Windows compiler requires extra parentheses
			lp.h = LE.p.pLong;
			lp.v = LE.p.pLat;
			LEmass = GetLEMass(LE,halfLife);	// will only vary for chemical with different release end time
			massInGrams = VolumeMassToGrams(LEmass, density, massunits);	// need to do this above too
			if (fContourDepth1==BOTTOMINDEX)
			{
				double depthAtLE = DepthAtPoint(LE.p);
				//if (LE.z > (depthAtLE-1.) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
				if (LE.z > (depthAtLE-fBottomRange) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
				{
					triIndex = dagTree -> WhatTriAmIIn(lp);
					if (triIndex>=0)  
					{
						(*numLEsInTri)[triIndex]++;
						(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
					}
				}
			}
			else if (LE.z>fContourDepth1 && LE.z<=fContourDepth2) 
			{
				triIndex = dagTree -> WhatTriAmIIn(lp);
				if (triIndex>=0) 
				{
					(*numLEsInTri)[triIndex]++;
					(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
				}
			}
		}
	}
	if (triGrid->bCalculateDosage)
	{
		dosageHdl = triGrid -> GetDosageHdl(true);
	}
	if (bThereIsSubsurfaceOil)
	{
		double triArea, triVol, oilDensityInWaterColumn, totalVol=0;
		long numLEsInTriangle,j,numLevels,totalLEs=0,totalMass=0,numDepths=0,maxTriNum=-1,numTrisWithOil = 0;
		double concInSelectedTriangles=0,maxConc=0,numTrisSelected=0,minLevel,maxLevel,range,avConcOverTriangles=0;
		if (!fContourLevelsH)
			if (!InitContourLevels()) return;
		numLevels = GetNumDoubleHdlItems(fContourLevelsH);
		if (fTriAreaArray)
			{delete [] fTriAreaArray; fTriAreaArray = 0;}
		fTriAreaArray = new double[numLevels];
		for (i=0;i<numLevels;i++)
			fTriAreaArray[i] = 0.;
		for (i=0;i<numTri;i++)
		{	// track ppm hours here
			double depthRange;
			depthAtPt=0;
			//WorldPoint centroid = {0,0};
			if (triSelected && !(*triSelected)[i]) continue;	// note this line keeps triareaarray from tracking for output..., dosage...
			//if (!triGrid->GetTriangleCentroidWC(i,&centroid))
			{
				//depthAtPt = DepthAtPoint(centroid);
				depthAtPt = DepthAtCentroid(i);
			}
			triArea = (triGrid -> GetTriArea(i)) * 1000 * 1000;	// convert to meters
			if (!(fContourDepth1==BOTTOMINDEX))	
			{
				if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt > 0) depthRange = depthAtPt - fContourDepth1;
				else depthRange = fContourDepth2 - fContourDepth1;
			}
			else
			{
				//if (depthAtPt<1 && depthAtPt>0) depthRange = depthAtPt;	// should do a triangle volume 
				//else depthRange = 1.; // for bottom will always contour 1m
				if (depthAtPt<fBottomRange && depthAtPt>0) depthRange = depthAtPt;	// should do a triangle volume 
				else depthRange = fBottomRange; // for bottom will always contour 1m
			}
			//if (depthAtPt < depthRange && depthAtPt != 0) depthRange = depthAtPt;
			//if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt != 0) depthRange = depthAtPt - fContourDepth1;
			triVol = triArea * depthRange; // code goes here, check this depth range is ok at all vertices
			//triVol = triArea * (fContourDepth2 - fContourDepth1); // code goes here, check this depth range is ok at all vertices
			numLEsInTriangle = (*numLEsInTri)[i];
			if (numLEsInTriangle==0)
				continue;

			if (!(fContourDepth1==BOTTOMINDEX))		// need to decide what to do for bottom contour
				if (triGrid->CalculateDepthSliceVolume(&triVol,i,fContourDepth1,fContourDepth2)) goto done;
			oilDensityInWaterColumn = (*massInTriInGrams)[i] / triVol;	// units? milligrams/liter ?? for now gm/m^3

			//(*concentrationH)[numTrisWithOil] = oilDensityInWaterColumn;
			(*concentrationH)[numTrisWithOil].conc = oilDensityInWaterColumn;
			(*concentrationH)[numTrisWithOil].triNum = i;
			numTrisWithOil++;
			//for (j=0;j<numLevels-1;j++)
			for (j=0;j<numLevels;j++)
			{
				//if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
				if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
				{
					fTriAreaArray[j] = fTriAreaArray[j] + triArea/1000000;
					totalLEs += numLEsInTriangle;
					//totalMass += massInGrams;
					totalMass += (*massInTriInGrams)[i];
					totalVol += triVol;
					concInSelectedTriangles += oilDensityInWaterColumn;	// sum or track each one?
					if (oilDensityInWaterColumn > maxConc) 
					{
						maxConc = oilDensityInWaterColumn;
						numDepths = floor(depthAtPt)+1;	// split into 1m increments to track vertical slice
						maxTriNum = i;
					}
					numTrisSelected++;
				}
			}
			if (triGrid->bCalculateDosage && dosageHdl)
			{
				double dosage;
				(*dosageHdl)[i] += oilDensityInWaterColumn * timeStep / 3600;	// 
				dosage = (*dosageHdl)[i];
			}
		}
		//_SetHandleSize((Handle) concentrationH, (numTrisWithOil)*sizeof(double));
		if (numTrisWithOil>0)
		{
			_SetHandleSize((Handle) concentrationH, (numTrisWithOil)*sizeof(ConcTriNumPair));
			if (triGrid->fPercentileForMaxConcentration < 1)	
			{
				//qsort((*concentrationH),numTrisWithOil,sizeof(double),ConcentrationCompare);
				qsort((*concentrationH),numTrisWithOil,sizeof(ConcTriNumPair),ConcentrationCompare);
				j = (long)(triGrid->fPercentileForMaxConcentration*numTrisWithOil);	// round up or down?, for selected triangles?
				if (j>0) j--;
				//maxConc = (*concentrationH)[j];	// trouble with this percentile stuff if there are only a few values
				maxConc = (*concentrationH)[j].conc;	// trouble with this percentile stuff if there are only a few values
				maxTriNum = (*concentrationH)[j].triNum;	// trouble with this percentile stuff if there are only a few values
			}
		}
		//if (thisLEList->GetOilType() == CHEMICAL) 
		if (totalVol>0) avConcOverTriangles = totalMass / totalVol;
		//else
			//avConcOverTriangles = totalLEs * massInGrams / totalVol;
		// track concentrations over time, maybe define a new data type to hold everything...
		//if (triSelected) triGrid -> AddToOutputHdl(numTrisSelected>0 ? concInSelectedTriangles/numTrisSelected : 0,maxConc,model->GetModelTime());
		if (triSelected) triGrid -> AddToOutputHdl(numTrisSelected>0 ? avConcOverTriangles : 0,maxConc,model->GetModelTime());
		else triGrid -> AddToOutputHdl(numTrisSelected>0 ? avConcOverTriangles : 0, maxConc, model->GetModelTime());
		//if (triSelected) triGrid -> AddToOutputHdl(oilDensityInWaterColumn,model->GetModelTime());
		if (/*fDiagnosticStrType==TRIANGLEAREA &&*/ bTimeToOutputData) // this will be messed up if there are triangles selected
			triGrid -> AddToTriAreaHdl(fTriAreaArray,numLevels);
		CreateDepthSlice(maxTriNum,&fDepthSliceArray);
		//CreateDepthSlice(maxTriNum,fDepthSliceArray);
		triGrid -> fMaxTri = maxTriNum; /*triGrid -> bShowMaxTri = true;*/
	}
	
done:
	if(numLEsInTri) {DisposeHandle((Handle)numLEsInTri); numLEsInTri=0;}
	if(massInTriInGrams) {DisposeHandle((Handle)massInTriInGrams); massInTriInGrams=0;}
	if(concentrationH) {DisposeHandle((Handle)concentrationH); concentrationH=0;}
	return;
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
// Concentration table code
/////////////////////////////////////////////////

static short timeH, avH, maxH;
outputDataHdl gOilConcHdl=0; 
float *gDepthSlice = 0;
short gTableType = 1;
 
OSErr ExportDepthTable(char* path)
{
	OSErr err = 0;
	double conc;
	outputData data;
	long numOutputValues,i,j,numDepths,depth;
	char buffer[512],concStr[64], depthStr[64];
	BFPB bfpb;
	
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	//strcpy(buffer,"Depth range (m)\t Concentration (ppm)");
	strcpy(buffer,"Depth (m)\t Concentration (ppm)");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;

	numDepths = gDepthSlice[0];
	for (j=0;j<numDepths;j++)
	{
		conc = gDepthSlice[j+1];
		depth = j;
		// Write out the times and values
		// add header line
		//strcpy(buffer,"Day Mo Yr Hr Min\t\tAv Conc\tMax Conc");
		//sprintf(depthStr,"%ld - %ld",j,j+1);
		StringWithoutTrailingZeros(depthStr,depth,3);
		StringWithoutTrailingZeros(concStr,conc,3);
		/////
		strcpy(buffer,depthStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,concStr);
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		// the user has already been told there was a problem
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}
OSErr PrintTableToFile(void)
{
	char path[256];
	OSErr err = 0;
	char* suggestedFileName = "Conc.dat";
	PtCurMap *map = GetPtCurMap();
	if (!map) return -1;
	TTriGridVel3D* triGrid = map->GetGrid3D(true);	// used refined grid if there is one	
	if (!triGrid) return -1; 

	// should be able to use the globals here
	//outputData **oilConcHdl = triGrid -> GetOilConcHdl();	// don't init
	//double **areaHdl = triGrid -> GetTriAreaHdl();
	//if (!oilConcHdl /*&& !areaHdl*/)
	//{
		//printError("There is no output data to export");
		//err = -1;
		//goto done;
	//}
	//else 
	//{
		err = AskUserForSaveFilename(suggestedFileName,path,".DAT",TRUE);
		if(err) return err; // note: might be user cancel
	
		if (gOilConcHdl && gTableType == 1) err = triGrid ->ExportOilConcHdl(path);
		//else if (areaHdl) err = triGrid ->ExportTriAreaHdl(path, map->GetNumContourLevels());
	
		else if (gDepthSlice && gTableType == 2) err = ExportDepthTable(path);
		goto done;
	//}
done:
	if (err) printError("Error saving table data to a file");
	return err;
}

static void ConcTableDraw(DialogPtr d, Rect *rectPtr,long i)
{
#pragma unused (rectPtr)
	short		leftOffset = 5, botOffset = 2;
	Point		p;
	short		v;
	Rect		rgbrect;
	double	av,max;	
	outputData data;
	char *q, timeS[128], numStr[128];
	DateTimeRec time;
	float depth;

	GetPen(&p);

	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
	rgbrect=GetDialogItemBox(d,CONCTABLE_USERITEM);
	v = p.v;
	if (gTableType==1)
	{
		data = INDEXH(gOilConcHdl,i);
		SecondsToDate (data.time, &time);
		Date2String(&time, timeS);
		if (q = strrchr(timeS, ':')) q[0] = 0; // remove seconds
	}
	else
	{
		sprintf(timeS,"%ld",i+1);
	}
	MyMoveTo(timeH,v);
	drawstring(timeS);

	if (gTableType==1)
	{
		av = (*gOilConcHdl)[i].avOilConcOverSelectedTri;
		max = (*gOilConcHdl)[i].maxOilConcOverSelectedTri;
		sprintf(numStr,"%5.3f",av);
		MyMoveTo(avH,v); drawstring(numStr);
		sprintf(numStr,"%5.3f",max);
		MyMoveTo(maxH,v); drawstring(numStr);
	}
	else
	{
		av = gDepthSlice[i+1];
		sprintf(numStr,"%5.3f",av);
		MyMoveTo(avH,v); drawstring(numStr);
	}
		
	MyMoveTo(rgbrect.left,v+botOffset); MyLineTo(rgbrect.right,v+botOffset);
	
	return;
}

static void ConcTableInit(DialogPtr d, VLISTPTR L)
{
#pragma unused (L)
	Rect r;
	if (gTableType==1)
	{
		mysetitext(d,CONCTABLE_TIMETITLE,"Time");
		mysetitext(d,CONCTABLE_AVTITLE,"Av Conc");
		mysetitext(d,CONCTABLE_MAXTITLE,"Max Conc");
	}
	else
	{
		mysetitext(d,CONCTABLE_TIMETITLE,"Depth");
		mysetitext(d,CONCTABLE_AVTITLE,"Conc (ppm)");
		ShowHideDialogItem(d, CONCTABLE_MAXTITLE, false); 
		//ShowHideDialogItem(d, CONCTABLE_SAVETOFILE, false); // don't save depth data
	}
	r = GetDialogItemBox(d,CONCTABLE_TIMETITLE);
	timeH = r.left;
	r = GetDialogItemBox(d,CONCTABLE_AVTITLE);
	avH  = r.left;
	r = GetDialogItemBox(d,CONCTABLE_MAXTITLE);
	maxH = r.left;
	return;
}

 
	
static Boolean ConcTableClick(DialogPtr d,VLISTPTR L,short itemHit,long *item,Boolean doubleClick)
{
	if(doubleClick)
	{
		*item = CONCTABLE_OK;
		return true;
	}

	switch(itemHit)
	{
		case CONCTABLE_OK:
			return true;
		case CONCTABLE_SAVETOFILE:
			PrintTableToFile();
			return false;
			break;
		default:
			return false;
			break;
	}
	return 0;
}


short ConcentrationTable(outputData **oilConcHdl,float *depthSlice,short tableType)		// code goes here, include depth range
{
	short			ditem;
	long			selitem;
	
	if(!oilConcHdl && tableType==1 || !depthSlice && tableType==2)
	{
		printError("No concentration data.");
		return 0;
	}
	gOilConcHdl = oilConcHdl;
	gDepthSlice = depthSlice;
	gTableType = tableType;
	selitem=SelectFromVListDialog(
				5175,
				CONCTABLE_USERITEM,
				tableType==1 ? _GetHandleSize((Handle)oilConcHdl)/sizeof(outputData) : depthSlice[0],
				ConcTableInit,
				nil,
				nil,
				ConcTableDraw,
				ConcTableClick,
				true,
				&ditem);
	return selitem;
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////
short gOutputType = 0;

OSErr M52Init(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	if (gOutputType==0)
	{
		Seconds outputStep = model -> GetOutputStep();
		Float2EditText(dialog,M52OUTPUTINTERVAL, outputStep / 3600. , 2);
	}
	else if (gOutputType==1)
		Float2EditText(dialog,M52OUTPUTINTERVAL, model -> LEDumpInterval / 3600. , 2);
	else 
		Float2EditText(dialog,M52OUTPUTOFFSET, model -> fTimeOffsetForSnapshots / 3600. , 2);
	
	if (gOutputType==0 || gOutputType==1)
	{	// movie
		ShowHideDialogItem(dialog, M52OUTPUTOFFSET, false); 
		ShowHideDialogItem(dialog, M52OUTPUTOFFSETLABEL, false); 
		ShowHideDialogItem(dialog, M52OUTPUTOFFSETLABEL2, false); 
		MySelectDialogItemText(dialog, M52OUTPUTINTERVAL, 0, 255);
	}
	else if (gOutputType==2)
	{
		ShowHideDialogItem(dialog, M52OUTPUTINTERVAL, false); 
		ShowHideDialogItem(dialog, M52OUTPUTINTERVALLABEL, false); 
		ShowHideDialogItem(dialog, M52OUTPUTINTERVALLABEL2, false); 
		MySelectDialogItemText(dialog, M52OUTPUTOFFSET, 0, 255);
	}
	else
		MySelectDialogItemText(dialog, M52OUTPUTINTERVAL, 0, 255);

	return 0;
}


short M52Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short	menuItemChosen;
	long	menuID_menuItem;

	switch (itemNum) {
		case M52CANCEL: return M52CANCEL;

		case M52OK:
			if (gOutputType==0)
				model -> SetOutputStep(EditText2Float(dialog,M52OUTPUTINTERVAL) * 3600);
			else if (gOutputType==1)
				model -> LEDumpInterval = EditText2Float(dialog,M52OUTPUTINTERVAL) * 3600;
			else if (gOutputType==2) 
				model -> fTimeOffsetForSnapshots = EditText2Float(dialog,M52OUTPUTOFFSET) * 3600;
			//model -> SetOutputStep(EditText2Float(dialog,M52OUTPUTINTERVAL) * 3600);	 
			return itemNum;
			
		case M52OUTPUTINTERVAL:
		//case M52OUTPUTOFFSET:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

		case M52OUTPUTOFFSET:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;
	}

	return 0;
}

OSErr OutputOptionsDialog(short outputType)
{
	short item;
	gOutputType = outputType;
	item = MyModalDialog(M52, mapWindow, 0, M52Init, M52Click);
	if(item == M52CANCEL) return USERCANCEL; 
	model->NewDirtNotification();	// is this necessary ?
	if(item == M52OK) return 0; 
	else return -1;
}

Boolean gUseSmoothing, gUseLineCrossAlgorithm;
float gPercentile, gMinDist;
OSErr SMOOTHINGInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	Float2EditText(dialog,SMOOTHING_PERCENTILE, round(gPercentile*100) , 2);
	Float2EditText(dialog,SMOOTHING_MINDIST, gMinDist, 2);
	SetButton (dialog, SMOOTHING_CHECKBOX, gUseSmoothing);
	SetButton (dialog, LINECROSS_CHECKBOX, gUseLineCrossAlgorithm);
	
	MySelectDialogItemText(dialog, SMOOTHING_PERCENTILE, 0, 255);

	return 0;
}


short SMOOTHINGClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short	menuItemChosen;
	long	menuID_menuItem;
	float percentile;

	switch (itemNum) {
		case SMOOTHING_CANCEL: return SMOOTHING_CANCEL;

		case SMOOTHING_OK:
			percentile = EditText2Float(dialog,SMOOTHING_PERCENTILE)/100.;
			if (percentile>1){printError("Percentile cannot be greater than 100"); break;}
			gPercentile = EditText2Float(dialog,SMOOTHING_PERCENTILE)/100.;
			gMinDist = EditText2Float(dialog,SMOOTHING_MINDIST);
			gUseSmoothing = GetButton(dialog, SMOOTHING_CHECKBOX);
			gUseLineCrossAlgorithm = GetButton(dialog, LINECROSS_CHECKBOX);			
			return itemNum;
			
		case SMOOTHING_PERCENTILE:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;

		case SMOOTHING_CHECKBOX:
		case LINECROSS_CHECKBOX:
			ToggleButton(dialog, itemNum);
			break;
	}

	return 0;
}

OSErr SmoothingParametersDialog(float *percentile,Boolean *useSmoothing, float *minDist, Boolean *useLineCrossAlgorithm)
{
	short item;
	gPercentile = *percentile;
	gUseSmoothing = *useSmoothing;
	gMinDist = *minDist;
	gUseLineCrossAlgorithm = *useLineCrossAlgorithm;
	item = MyModalDialog(SMOOTHING, mapWindow, 0, SMOOTHINGInit, SMOOTHINGClick);
	if(item == SMOOTHING_CANCEL) return USERCANCEL; 
	model->NewDirtNotification();	// is this necessary ?
	if(item == SMOOTHING_OK) 
	{
		*percentile = gPercentile;
		*useSmoothing = gUseSmoothing;
		*minDist = gMinDist;
		*useLineCrossAlgorithm = gUseLineCrossAlgorithm;
		return 0; 
	}
	else return -1;
}

void DisposeAnalysisMenu(void)
{	
	
	if (!gDispersedOilVersion) return;
#ifdef IBM
	HMENU menu, m;
#endif
		
#ifdef MAC
	DeleteMenu(ANALYSISMENU);
	DrawMenuBar();
#else
	menu = GetMenu(mapWindow);
	DeleteMenu(menu, 4, MF_BYPOSITION);
	DrawMenuBar(mapWindow);
#endif
	gDispersedOilVersion = false;

	NUMTOOLS = 8;
	tools[8] = -1;
	tools[9] = -2;
	//tools[10] = -3;
	//tools[11] = -4;
}

/*void InitAnalysisMenu(void)
{
	/// check if diagnostic version

	if (gDispersedOilVersion) return;

#ifdef MAC
	gDispersedOilVersion =true;
	GetAndInsertMenu(ANALYSISMENU, 0);
	//DrawMenuBar(); on the MAC, Draw MenuBar will be called by the calling function
#else
{
	MenuHandle hMenu = LoadMenu (hInst,"ANALYSIS");
	if(hMenu) {
		gDispersedOilVersion = true;
		//AppendMenu (GetMenu(hMainWnd), MF_POPUP,(UINT)hMenu ,"&Analysis");
		// code goes here, change all the 4's to kZeroRelativeAnalysisMenuPosition or something
		InsertMenu (GetMenu(hMainWnd), 4, MF_POPUP | MF_BYPOSITION,(UINT)hMenu ,"&Analysis");
		DrawMenuBar(hMainWnd);
	}
}
#endif

	NUMTOOLS = 10;
	tools[8] = LASSOTOOL;
	tools[9] = SHORTSELTOOL;
	tools[10] = -1;
	tools[11] = -2;
	//gMearnsVersion = true;	// temp for old .SAV files, remove this?
}*/

OSErr PtCurMap::DoAnalysisMenuItems(long menuCodedItemID)
{
	OSErr err = 0;
	switch (menuCodedItemID) {
		case DESELECTITEM:
			{
				TTriGridVel3D* triGrid = this->GetGrid3D(true);	// used refined grid if there is one	
				if (!triGrid) return 0; 
			
				//triGrid -> DeselectAll();
				triGrid -> ClearTriSelection();	// want to delete the handle and start over
				model->NewDirtNotification();	
			}
			break;
		case CONCATPTITEM:	// show grid afterwards
			{
				Boolean selectedPtToTrack = false;	
			
				if (err = AnalysisDialog2()) return err;
				// if they hit ok, fall through to show grid 
			}
			//break;		
showGrid:
		case SHOWGRIDITEM:
		{
			TMover *mover = this->GetMover(TYPE_CATSMOVER3D);
			if (!mover) return -1;

			if (menuCodedItemID == CONCATPTITEM &&  (dynamic_cast<TCATSMover*>(mover))->bShowGrid == true) 
				{model->NewDirtNotification(); break;}
#ifdef MAC
			ToggleMenuCheck(ANALYSISMENU,SHOWGRIDITEM-ANALYSISMENU); 
#else
			ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWGRIDITEM-ANALYSISMENU);
#endif
			 (dynamic_cast<TCATSMover*>(mover))->bShowGrid = !(dynamic_cast<TCATSMover*>(mover))->bShowGrid;
			model->NewDirtNotification();	
			break;
		}
		case SHOWSELECTEDTRIITEM:
		{
			TTriGridVel3D* triGrid = 0;	
			TMover *mover = this->GetMover(TYPE_CATSMOVER3D);
			if (!mover) return -1;

#ifdef MAC
			ToggleMenuCheck(ANALYSISMENU,SHOWSELECTEDTRIITEM-ANALYSISMENU); 
#else
			ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWSELECTEDTRIITEM-ANALYSISMENU);
#endif
			 triGrid = (TTriGridVel3D*)((dynamic_cast<TCATSMover3D *>(mover)) -> fGrid);
			if (!triGrid) return -1;
			
			triGrid->bShowSelectedTriangles = !triGrid->bShowSelectedTriangles;

			model->NewDirtNotification();	
			break;
		}
		case SHOWMAXTRIITEM:
		{
#ifdef MAC
			ToggleMenuCheck(ANALYSISMENU,SHOWMAXTRIITEM-ANALYSISMENU); 
#else
			ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWMAXTRIITEM-ANALYSISMENU);
#endif
			TTriGridVel3D* triGrid = GetGrid3D(false);	
			if (!triGrid) return 0; 
			triGrid->bShowMaxTri = !triGrid->bShowMaxTri;
			// need to start over
			//model->NewDirtNotification();	
			break;
		}
		case SETCONTOURSITEM:
				err = TMapSettingsDialog(this);
				break;
		
		case CONCATPLUMEITEM: 	// may want to deselect triangles
				printNote("The model automatically tracks concentration with the plume if no triangles are selected.");
				break;
		case DEPTHCONTOURSITEM: 	// may want to have a submenu, for show contours, show values check boxes
				{
					TTriGridVel3D* triGrid = GetGrid3D(false);	// depths not refined	
					if (!triGrid) return 0; 
					if (err = triGrid->DepthContourDialog()) break;
				}
				//model->NewDirtNotification();	
				//break;		// make sure contours are being shown
		case SHOWCONTOURSITEM:
		{
			TMover *mover = this->GetMover(TYPE_CATSMOVER3D);
			//if (!mover) return -1;
			if (!mover) mover = this->GetMover(TYPE_TRICURMOVER);
			if (!mover) mover = this->GetMover(TYPE_NETCDFMOVER);
			if (!mover) mover = this->GetMover(TYPE_COMPOUNDMOVER);
			if (!mover) return -1;

			if (menuCodedItemID == DEPTHCONTOURSITEM && mover->IAm(TYPE_CATSMOVER3D) &&  (dynamic_cast<TCATSMover3D*>(mover))->bShowDepthContours == true) 
				{model->NewDirtNotification(); break;}
			if (menuCodedItemID == DEPTHCONTOURSITEM && mover->IAm(TYPE_TRICURMOVER) &&  (dynamic_cast<TriCurMover*>(mover))->bShowDepthContours == true) 
				{model->NewDirtNotification(); break;}
			if (menuCodedItemID == DEPTHCONTOURSITEM && mover->IAm(TYPE_NETCDFMOVER) &&  (dynamic_cast<NetCDFMover*>(mover))->bShowDepthContours == true) 
				{model->NewDirtNotification(); break;}
			if (menuCodedItemID == DEPTHCONTOURSITEM && mover->IAm(TYPE_COMPOUNDMOVER) &&  (dynamic_cast<TCompoundMover*>(mover))->ShowDepthContourChecked()) 
				{model->NewDirtNotification(); break;}
#ifdef MAC
			ToggleMenuCheck(ANALYSISMENU,SHOWCONTOURSITEM-ANALYSISMENU); 
#else
			ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWCONTOURSITEM-ANALYSISMENU);
#endif
			if (mover->IAm(TYPE_CATSMOVER3D))  (dynamic_cast<TCATSMover3D*>(mover))->bShowDepthContours = !(dynamic_cast<TCATSMover3D*>(mover))->bShowDepthContours;
			if (mover->IAm(TYPE_TRICURMOVER))  (dynamic_cast<TriCurMover*>(mover))->bShowDepthContours = !(dynamic_cast<TriCurMover*>(mover))->bShowDepthContours;
			if (mover->IAm(TYPE_NETCDFMOVER))  (dynamic_cast<NetCDFMover*>(mover))->bShowDepthContours = !(dynamic_cast<NetCDFMover*>(mover))->bShowDepthContours;
			if (mover->IAm(TYPE_COMPOUNDMOVER))  (dynamic_cast<TCompoundMover*>(mover))->SetShowDepthContours();	// will want to set these individually fminon left hand list
			model->NewDirtNotification();	
			break;
		}
		case SHOWCONTOURLABELSITEM:
		{
#ifdef MAC
			ToggleMenuCheck(ANALYSISMENU,SHOWCONTOURLABELSITEM-ANALYSISMENU); 
#else
			ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWCONTOURLABELSITEM-ANALYSISMENU);
#endif
			TMover *mover = this->GetMover(TYPE_CATSMOVER3D);
			//if (!mover) return -1;
			//((TCATSMover3D*)mover)->bShowDepthContourLabels = !((TCATSMover3D*)mover)->bShowDepthContourLabels;
			if (mover) 
				 (dynamic_cast<TCATSMover3D*>(mover))->bShowDepthContourLabels = !(dynamic_cast<TCATSMover3D*>(mover))->bShowDepthContourLabels;
			else
			{
			 	mover = this->GetMover(TYPE_TRICURMOVER);
				if (mover) 
					 (dynamic_cast<TriCurMover*>(mover))->bShowDepthContourLabels = !(dynamic_cast<TriCurMover*>(mover))->bShowDepthContourLabels;
				else
				{
					mover = this->GetMover(TYPE_NETCDFMOVER);
					if (mover) 
						 (dynamic_cast<NetCDFMover*>(mover))->bShowDepthContourLabels = !(dynamic_cast<NetCDFMover*>(mover))->bShowDepthContourLabels;
						else
							return -1;
				}
				//else
					//return -1;
			}
			// if the contours aren't being shown the labels won't be shown either
			model->NewDirtNotification();	
			break;
		}
		case SCALEDEPTHSITEM:
		{
			char msg[256]; 
			double scaleFactor;
			TTriGridVel3D* triGrid = GetGrid3D(false);	// depths not refined	
			if (!triGrid) return 0; 
			strcpy(msg,"Input scale factor for depths");
			if (err = GetScaleFactorFromUser(msg, &scaleFactor)) return 0;
			triGrid->ScaleDepths(scaleFactor);
			break;			
		}
		case DISPERSEOILITEM: 	
				printNote("Set a spill using the spill tool or spray can, then select Apply Dispersants on the spill dialog. Or mark an area to disperse during a run using Option Lasso (rectangle) or Control Lasso (group of triangles).");
				break;
		case SHOWPLOTSITEM: 	
				{
					ListItem item;
					item.index = I_PCONCTABLE;
					ListClick(item, false, true);
				}
				break;
		case SMOOTHINGPARAMETERSITEM:
		{
			float val, minDist;
			Boolean smooth, lineCross;
			TTriGridVel3D* triGrid = GetGrid3D(false);	
			if (!triGrid) return 0; 
			val = triGrid->fPercentileForMaxConcentration;
			smooth = bUseSmoothing;
			minDist = fMinDistOffshore;
			lineCross = bUseLineCrossAlgorithm;
			if (SmoothingParametersDialog(&val,&smooth,&minDist,&lineCross))
			{
			}
			else
			{
				triGrid->fPercentileForMaxConcentration = val;
				bUseSmoothing = smooth;
				fMinDistOffshore = minDist;
				bUseLineCrossAlgorithm = lineCross;
			}
			
			break;
		}
		case CALCDOSAGEITEM:
		{
#ifdef MAC
			ToggleMenuCheck(ANALYSISMENU,CALCDOSAGEITEM-ANALYSISMENU); 
#else
			ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),CALCDOSAGEITEM-ANALYSISMENU);
#endif
			TTriGridVel3D* triGrid = GetGrid3D(false);		
			if (!triGrid) return 0; 
			triGrid->bCalculateDosage = !triGrid->bCalculateDosage;
			// need to start over
			model->NewDirtNotification();	
			break;
		}
		case SHOWDOSAGEITEM:
		{
#ifdef MAC
			ToggleMenuCheck(ANALYSISMENU,SHOWDOSAGEITEM-ANALYSISMENU); 
#else
			ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SHOWDOSAGEITEM-ANALYSISMENU);
#endif
			TTriGridVel3D* triGrid = GetGrid3D(false);	
			if (!triGrid) return 0; 
			triGrid->bShowDosage = !triGrid->bShowDosage;
			// need to start over
			model->NewDirtNotification();	
			break;
		}
		case SUPPRESSDRAWINGITEM:
		{
#ifdef MAC
			ToggleMenuCheck(ANALYSISMENU,SUPPRESSDRAWINGITEM-ANALYSISMENU); 
#else
			ToggleMenuCheck2(GetSubMenu(GetMenu(mapWindow),4),SUPPRESSDRAWINGITEM-ANALYSISMENU);
#endif
			gSuppressDrawing = !gSuppressDrawing;
			// need to start over
			model->NewDirtNotification();	
			break;
		}
		case SETSELECTEDBEACH:
		{
			long numBoundaryPts = GetNumBoundaryPts(), numBBoundaries=0;
			TTriGridVel3D* triGrid = GetGrid3D(false);	
			if (!triGrid) return 0; 

			if(!(triGrid->PointsSelected()))break;
			SetSelectedBeach(&fSelectedBeachHdl,fSegSelectedH);	// after this should get rid of all selected points?
			// Beach Boundaries
			//if(!(fSelectedBeachFlagHdl = (LONGH)MyNewHandle(sizeof(long)*(numBoundaryPts))))
			if(!(fSelectedBeachFlagHdl = (LONGH)_NewHandleClear(sizeof(long)*(numBoundaryPts))))
				printError("Not enough memory. Beach segment cannot be selected.");
			else // ensure only true segments are marked and exported
				SetBeachSegmentFlag(&fSelectedBeachFlagHdl, &numBBoundaries);
			model->NewDirtNotification();	
			triGrid->DeselectAllPoints();
			// also dispose of fSegSelected?
			MyDisposeHandle((Handle*)&fSegSelectedH);
			InvalMapDrawingRect();
			break;
		}
		case CLEARSELECTEDBEACH:
		{
			TTriGridVel3D* triGrid = GetGrid3D(false);		
			if (!triGrid) return 0; 

			if(!fSelectedBeachHdl && !fSegSelectedH){
				printError("No beach has been selected.");
				break;
			}
			ClearSelectedBeach();
			model->NewDirtNotification();	
			triGrid->DeselectAllPoints();
			InvalMapDrawingRect();
			break;
		}
		case CONCENTRATIONONBEACH:
		{
			long numLEs = 0;
			char msg[64];
			if(!fSelectedBeachHdl){
				printError("No beach has been selected.");
				break;
			}
			// track all the LEs on the selected beach
			numLEs = CountLEsOnSelectedBeach();	
			// separate call here for the table
			//sprintf(msg,"Number of LEs beached on selected segment = %ld", numLEs);
			//printNote(msg);
			break;
		}


	}
	return err;
}
/////////////////////////////////////////////////
// this isn't used
Boolean WorldPointNearSegment(long pLong, long pLat,
						  long long1, long lat1, long long2, long lat2,
						  long dLong, long dLat, float d)
{
	float a, b, x, y, h, dist, dist2, numer, dummy;
	WorldPoint p, p2, testPt, startPt, endPt;

	testPt.pLat = pLat;
	testPt.pLong = pLong;
	startPt.pLat = lat1;
	startPt.pLong = long1;
	endPt.pLat = lat2;
	endPt.pLong = long2;
	
/*	dLong = dLat = 1000000/3600.;	
	if (long1 < long2) { if (pLong < (long1 - dLong) ||
							 pLong > (long2 + dLong)) return FALSE; }
	else			   { if (pLong < (long2 - dLong) ||
							 pLong > (long1 + dLong)) return FALSE; }
	
	if (lat1 < lat2) { if (pLat < (lat1 - dLat) ||
						   pLat > (lat2 + dLat)) return FALSE; }
	else			 { if (pLat < (lat2 - dLat) ||
						   pLat > (lat1 + dLat)) return FALSE; }*/
	
	p.pLong = pLong;
	p.pLat = pLat;
	
	//float wdist = LatToDistance(ScreenToWorldDistance(4));
	//d = LatToDistance(ScreenToWorldDistance(4));
	d = LatToDistance(ScreenToWorldDistance(30));
	

	//if (WPointNearWPoint2(pLong, pLat, long1, lat1, dLong, dLat, d)) return TRUE;
	//if (WPointNearWPoint2(pLong, pLat, long2, lat2, dLong, dLat, d)) return TRUE;
	
//Boolean WPointNearWPoint(WorldPoint p1, WorldPoint p2, float d)
	if (WPointNearWPoint(testPt, startPt, d)) return TRUE;
	if (WPointNearWPoint(testPt, endPt, d)) return TRUE;
	
	// translate origin to start of segment
	
	a = LongToDistance(long2 - long1, p);
	b = LatToDistance(lat2 - lat1);
	x = LongToDistance(pLong - long1, p);
	y = LatToDistance(pLat - lat1);
	h = sqrt(a * a + b * b);
	
	// distance from point to segment
	numer = fabs(a * y - b * x);
	dist = numer / h;
	dummy = dist; // see comment below
	
	if (dist > d) return FALSE;
	
	// the rest of this code checks if the point is beyond the ends of the
	// segment (and beyond the radii of the circles at the endpoints)
	
	// length of projection of point onto segment
	numer = a * x + b * y;
	dist = numer / h;
	dummy = dist; // see comment below
	
	if (dist < 0) return FALSE;
	
	p.pLong = long1;
	p.pLat = lat1;
	p2.pLong = long2;
	p2.pLat = lat2;
	dist2 = DistanceBetweenWorldPoints(p, p2);
	
	// Due to what seems like a bug in the Microsoft C compiler, if you assign
	// a float (or double) value and do nohting but comparisons with it,
	// the value gets lost after the assignment on 386 machines.  So here we do
	// an assignment with dist2 and dist before doing the comparison to make they
	// keep their values.  (The problem may be limited to doing a comparison
	// right after an assignment.)
	
	dummy = dist2;
	dummy = dist;
	
	if (dist > dist2) return FALSE;
	
	return TRUE;
}



OSErr PtCurMap::ExportOiledShorelineData(OiledShorelineDataHdl oiledShorelineHdl)
{
	OSErr err = 0;
	float lengthInKm, lengthInMiles, lengthInFeet;
	long segNo,startPt,endPt,numBeachedLEs;
	OiledShorelineData data;
	long numOutputValues,i;
	char buffer[512],dataStr[64];
	BFPB bfpb;
	char path[256];
	char* suggestedFileName = "OiledShoreline.dat";
	TLEList *thisLEList;
	long j,n;
	LETYPE leType;
	double amtBeachedOnSegment;
	
	err = AskUserForSaveFilename(suggestedFileName,path,".DAT",TRUE);
	if(err) return err; // note: might be user cancel

	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }

	// add header line
	strcpy(buffer,"SegNo\tStartIndex\tEndIndex\tNumLEs\tGallons\tShorelineLengthInMiles\tgal/mile\tgal/foot");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	numOutputValues = _GetHandleSize((Handle)oiledShorelineHdl)/sizeof(**oiledShorelineHdl);
	for(i = 0; i< numOutputValues;i++)
	{
		data = INDEXH(oiledShorelineHdl,i);
		endPt = data.endPt;
	//	if (INDEXH(fSelectedBeachFlagHdl,endPt)==0) continue;
		numBeachedLEs = data.numBeachedLEs;
		lengthInKm = data.segmentLengthInKm;
		lengthInMiles = lengthInKm / MILESTOKILO;
		lengthInFeet = lengthInMiles * MILESTOFEET;
		amtBeachedOnSegment = data.gallonsOnSegment;
		segNo = data.segNo;
		startPt = data.startPt;

		StringWithoutTrailingZeros(dataStr,segNo,3);
		strcpy(buffer,dataStr);
		strcat(buffer,"\t");
		StringWithoutTrailingZeros(dataStr,startPt,3);
		strcat(buffer,dataStr);
		strcat(buffer,"\t");
		StringWithoutTrailingZeros(dataStr,endPt,3);
		strcat(buffer,dataStr);
		strcat(buffer,"\t");
		StringWithoutTrailingZeros(dataStr,numBeachedLEs,3);
		strcat(buffer,dataStr);
		strcat(buffer,"\t");
		StringWithoutTrailingZeros(dataStr,amtBeachedOnSegment,3);
		strcat(buffer,dataStr);
		strcat(buffer,"\t");
		StringWithoutTrailingZeros(dataStr,lengthInMiles,3);
		strcat(buffer,dataStr);
		strcat(buffer,"\t");
		StringWithoutTrailingZeros(dataStr,amtBeachedOnSegment/lengthInMiles,3);
		strcat(buffer,dataStr);
		strcat(buffer,"\t");
		StringWithoutTrailingZeros(dataStr,amtBeachedOnSegment/lengthInFeet,3);
		strcat(buffer,dataStr);
		strcat(buffer,NEWLINESTRING);
		/////
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}

done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		// the user has already been told there was a problem
		(void)hdelete(0, 0, path); // don't leave them with a partial file
		printError("Error saving oiled shoreline data to a file");
	}
	return err;
}

void PtCurMap::AddSegmentToSegHdl(long startno)
{
	AppendToLONGH(&fSegSelectedH,startno);
	return;
}
void PtCurMap::ClearSegmentHdl()
{
	MyDisposeHandle((Handle*)&fSegSelectedH);
	return;
}
/////////////////////////////////////////////////
// changed to AnalysisDialog2 - can't input point to track, just explains how to do it
Boolean gSelectedPtToTrack;
static PopInfoRec AnalysisPopTable[] = {
		{ ANALYSIS_DLGID, nil, ANALYSISTOPLATDIR, 0, pNORTHSOUTH1, 0, 1, FALSE, nil },
		{ ANALYSIS_DLGID, nil, ANALYSISLEFTLONGDIR, 0, pEASTWEST1, 0, 1, FALSE, nil },
		{ ANALYSIS_DLGID, nil, ANALYSISBOTTOMLATDIR, 0, pNORTHSOUTH2, 0, 1, FALSE, nil },
		{ ANALYSIS_DLGID, nil, ANALYSISRIGHTLONGDIR, 0, pEASTWEST2, 0, 1, FALSE, nil },
	};

void ShowHidePtToTrack(DialogPtr dialog)
{
	Boolean show  = GetButton (dialog, ANALYSIS_REGION); 
	Boolean show2  = GetButton (dialog, ANALYSIS_TRACKREG); 

	ShowHideDialogItem(dialog, ANALYSIS_TRACKREG, show); 

	SwitchLLFormatHelper(dialog, ANALYSISTOPLATDEGREES, ANALYSISDEGREES, show);
	SwitchLLFormatHelper(dialog, ANALYSISBOTTOMLATDEGREES, ANALYSISDEGREES, show && show2); 
	
	ShowHideDialogItem(dialog, ANALYSISDEGREES, show); 
	ShowHideDialogItem(dialog, ANALYSISDEGMIN, show); 
	ShowHideDialogItem(dialog, ANALYSISDMS, show); 

	ShowHideDialogItem(dialog, ANALYSISTOPLATLABEL, show); 
	ShowHideDialogItem(dialog, ANALYSISLEFTLONGLABEL, show); 
	ShowHideDialogItem(dialog, ANALYSISBOTTOMLATLABEL, show && show2); 
	ShowHideDialogItem(dialog, ANALYSISRIGHTLONGLABEL, show && show2); 
}

short AnalysisClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
#pragma unused (data)
	WorldPoint p, p2;
	Boolean changed, tempSelectTri, tempSelectTri2;
	WorldRect origBounds = emptyWorldRect;
	PtCurMap *map = GetPtCurMap();	// still could be 2D...
	long menuID_menuItem;
	OSErr err = 0;

	StandardLLClick(dialog, itemNum, ANALYSISTOPLATDEGREES, ANALYSISDEGREES, &p, &changed);
	StandardLLClick(dialog, itemNum, ANALYSISBOTTOMLATDEGREES, ANALYSISDEGREES, &p2, &changed);

	if (map)
	{
		origBounds = map -> GetMapBounds();
	}
	
	switch(itemNum)
	{
		case ANALYSIS_OK:
			
			tempSelectTri = GetButton (dialog, ANALYSIS_REGION);
			tempSelectTri2 = GetButton (dialog, ANALYSIS_TRACKREG);
			short timeType, depthInputType, seaType;
			
			if(tempSelectTri)
			{
				long oneSecond = (1000000/3600);
				// retrieve the extendedBounds
				if (err = EditTexts2LL(dialog, ANALYSISTOPLATDEGREES, &p, TRUE)) break;
				//if (err = EditTexts2LL(dialog, ANALYSISBOTTOMLATDEGREES, &p2, TRUE)) break;
				if (err = EditTexts2LL(dialog, ANALYSISBOTTOMLATDEGREES, &p2, TRUE)) break;

				// check extended bounds (oneSecond handles accuracy issue in reading from dialog)			
				if (p.pLat > origBounds.hiLat + oneSecond || p2.pLat < origBounds.loLat - oneSecond
					|| p.pLong < origBounds.loLong - oneSecond || p2.pLong > origBounds.hiLong + oneSecond)
				{
					printError("The area to track cannot be greater than the map bounds."); 
					return 0; 
				}
				
				//maybe allow a second pt to set a box ?
				// if so search over all triangles to find those inside the box (all 3 vertices)
				// then append to the list of selected triangles
				// p is top left, p2 is bottom right
				if (p.pLat < p2.pLat || p.pLong > p2.pLong)
				{
					printError("The bounds of the region to track are not consistent (top < bot or left > right)."); 
					return 0; 
				}
				
				// just in case of round off
				p.pLat = _min(p.pLat,origBounds.hiLat);
				p.pLong = _max(p.pLong,origBounds.loLong);
				p2.pLat = _max(p2.pLat,origBounds.loLat);	
				p2.pLong = _min(p2.pLong,origBounds.hiLong);
				if (!tempSelectTri2)
				{
					LongPoint lp;
					TDagTree *dagTree = 0;
					TTriGridVel3D* triGrid = map->GetGrid3D(true);	// used refined grid if there is one	
					if (!triGrid) return 0; 
					Boolean **triSelection = 0;
					lp.h = p.pLong;
					lp.v = p.pLat;
				
					dagTree = triGrid -> GetDagTree();
					if(!dagTree)	return 0;
					long trinum = dagTree->WhatTriAmIIn(lp);
					if (trinum<0)
					{
						printError("The point is not in the grid."); 
						return 0; 
					}
					triGrid->DeselectAll();
					triSelection = triGrid -> GetTriSelection(true);
					triGrid -> ToggleTriSelection(trinum);	
					gSelectedPtToTrack = true;
				}
				if (tempSelectTri2)
				{
					TTriGridVel3D* triGrid = map->GetGrid3D(true);	// used refined grid if there is one	
					if (!triGrid) return 0; 
					Boolean **triSelection = triGrid -> GetTriSelection(true), needToRefresh = false;
					if (tempSelectTri2)
					{
						WORLDPOINTH wh=0;
						WorldPoint wp,wp2;
						wp.pLat = p.pLat;
						wp.pLong = p2.pLong;
						wp2.pLat = p2.pLat;
						wp2.pLong = p.pLong;
						AppendToWORLDPOINTH(&wh,&p); 
						AppendToWORLDPOINTH(&wh,&wp); 
						AppendToWORLDPOINTH(&wh,&p2); 
						AppendToWORLDPOINTH(&wh,&wp2); 
						//triGrid->DeselectAll();	// for now have user do it
						gSelectedPtToTrack = triGrid->SelectTriInPolygon(wh, &needToRefresh);
						if (!gSelectedPtToTrack) 
						{printNote("No triangles fell inside the selected area"); return 0;}
					}
				}
				//gSelectedPtToTrack = true;
			}
			else
				gSelectedPtToTrack = false;

			return ANALYSIS_OK;
			
		case ANALYSIS_CANCEL:
			return ANALYSIS_CANCEL;
			break;
			
		case ANALYSIS_TRACKREG:
		case ANALYSIS_REGION:
			ToggleButton(dialog, itemNum);
			ShowHidePtToTrack(dialog);
			break;

		case ANALYSISDEGREES:
		case ANALYSISDEGMIN:
		case ANALYSISDMS:
				if (err = EditTexts2LL(dialog, ANALYSISTOPLATDEGREES, &p, TRUE)) break;
				if (err = EditTexts2LL(dialog, ANALYSISBOTTOMLATDEGREES, &p2, TRUE)) break;
				if (itemNum == ANALYSISDEGREES) settings.latLongFormat = DEGREES;
				if (itemNum == ANALYSISDEGMIN) settings.latLongFormat = DEGMIN;
				if (itemNum == ANALYSISDMS) settings.latLongFormat = DMS;
				//ShowHidePtToTrack(dialog);
				SwitchLLFormatHelper(dialog, ANALYSISTOPLATDEGREES, ANALYSISDEGREES, true);
				SwitchLLFormatHelper(dialog, ANALYSISBOTTOMLATDEGREES, ANALYSISDEGREES, true); 
				LL2EditTexts(dialog, ANALYSISBOTTOMLATDEGREES, &p2);
				LL2EditTexts(dialog, ANALYSISTOPLATDEGREES, &p);
			break;

	}
	return 0;
}

OSErr AnalysisInit(DialogPtr dialog, VOIDPTR data)
{
	#pragma unused (data)
	PtCurMap *map = GetPtCurMap();
	if (!map) return -1;
	TMover *mover = map->GetMover(TYPE_CATSMOVER3D);
	if (!mover) return -1;
	 WorldPoint wp = (dynamic_cast<TCATSMover*>(mover))->refP;

	SetDialogItemHandle(dialog, ANALYSIS_HILITE, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, ANALYSIS_FROST1, (Handle)FrameEmbossed);

	RegisterPopTable (AnalysisPopTable, sizeof (AnalysisPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (ANALYSIS_DLGID, dialog);
	
	SetButton (dialog, ANALYSIS_REGION, false); // what should default be ?
	SetButton (dialog, ANALYSIS_TRACKREG, false); // what should default be ?
	LL2EditTexts (dialog, ANALYSISTOPLATDEGREES, &wp);
	
	LL2EditTexts (dialog, ANALYSISBOTTOMLATDEGREES, &wp);

	//ShowHidePtToTrack(dialog);
	SwitchLLFormatHelper(dialog, ANALYSISTOPLATDEGREES, ANALYSISDEGREES, true);
	SwitchLLFormatHelper(dialog, ANALYSISBOTTOMLATDEGREES, ANALYSISDEGREES, true); 
	ShowHidePtToTrack(dialog);

	return 0;
}

OSErr AnalysisDialog(Boolean *selectedPtToTrack)	
{	// no parent in this case
	short item;
	*selectedPtToTrack = 0;
	item = MyModalDialog(ANALYSIS_DLGID, mapWindow, 0, AnalysisInit, AnalysisClick);
	if (item == ANALYSIS_OK) {
		*selectedPtToTrack = gSelectedPtToTrack;
		//model->NewDirtNotification(); 
	}
	if (item == ANALYSIS_CANCEL) {return USERCANCEL;}
	return item == ANALYSIS_OK? 0 : -1;
}

short AnalysisClick2(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
#pragma unused (data)

	switch(itemNum)
	{
		case ANALYSIS_OK:
			return ANALYSIS_OK;
			
		case ANALYSIS_CANCEL:
			return ANALYSIS_CANCEL;
			break;
			
	}
	return 0;
}

OSErr AnalysisInit2(DialogPtr dialog, VOIDPTR data)
{
	#pragma unused (data)

	SetDialogItemHandle(dialog, ANALYSIS_HILITE, (Handle)FrameDefault);

	return 0;
}
OSErr AnalysisDialog2()
{	// no parent in this case
	short item;
	item = MyModalDialog(ANALYSIS_DLG2, mapWindow, 0, AnalysisInit2, AnalysisClick2);
	if (item == ANALYSIS_CANCEL) {return USERCANCEL;}
	return item == ANALYSIS_OK? 0 : -1;
}

/////////////////////////////////////////////////
