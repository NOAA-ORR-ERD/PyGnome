#include "Earl.h"
#include "TypeDefs.h"
#include "Cross.h"
#include "Uncertainty.h"
#include "GridVel.h"
#include "TimeGridVel.h"
#include "DagTreeIO.h"
#include "netcdf.h"


TimeGridVel::TimeGridVel (/*TMap *owner, char *name*/)
{
	memset(&fVar,0,sizeof(fVar));
	//fVar.arrowScale = 1.;
	//fVar.arrowDepth = 0;

	fVar.fileScaleFactor = 1.0;
	//fVar.startTimeInHrs = 0.0;
	fVar.gridType = TWO_D; // 2D default
	fVar.maxNumDepths = 1;	// 2D default - may not need this
	
	//
	fGrid = 0;
	fTimeHdl = 0;
	
	fTimeShift = 0;	// assume file is in local time
	fOverLap = false;		// for multiple files case
	fOverLapStartTime = 0;
	
	fFillValue = -1e+34;
	//fIsNavy = false;	
	
	//fFileScaleFactor = 1.;	// let user set a scale factor in addition to what is in the file
	
	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	fInputFilesHdl = 0;	// for multiple files case
	
	fAllowExtrapolationInTime = false;
	
}

void TimeGridVel::Dispose ()
{
	if (fGrid)
	{
		fGrid -> Dispose();
		delete fGrid;
		fGrid = nil;
	}

	if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}

	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);

	if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}

}


#define TimeGridVelREADWRITEVERSION 1 // don't bother with for now

OSErr TimeGridVel::Write (BFPB *bfpb)
{
	long i, version = TimeGridVelREADWRITEVERSION; 
	ClassID id = GetClassID ();
	long numTimes = GetNumTimesInFile(), numPoints = 0, numPts = 0, numFiles = 0;
	Seconds time;
	float val, depthLevel;
	PtCurFileInfo fileInfo;
	OSErr err = 0;
	
	return err;
	
	StartReadWriteSequence("TimeGridVel::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	id = fGrid -> GetClassID (); //JLM
	if (err = WriteMacValue(bfpb, id)) return err; //JLM
	if (err = fGrid -> Write (bfpb)) goto done;
	
	if (err = WriteMacValue(bfpb, fNumRows)) goto done;
	if (err = WriteMacValue(bfpb, fNumCols)) goto done;
	if (err = WriteMacValue(bfpb, fVar.pathName, kMaxNameLen)) goto done;
	if (err = WriteMacValue(bfpb, fVar.userName, kPtCurUserNameLen)) return err;
	if (err = WriteMacValue(bfpb, fVar.fileScaleFactor)) return err;
	//
	if (err = WriteMacValue(bfpb, fVar.maxNumDepths)) return err;
	if (err = WriteMacValue(bfpb, fVar.gridType)) return err;
	//
	if (err = WriteMacValue(bfpb, fFillValue)) return err;
	//if (err = WriteMacValue(bfpb, fIsNavy)) return err;
	//
	
	if (err = WriteMacValue(bfpb, numTimes)) goto done;
	for (i=0;i<numTimes;i++)
	{
		time = INDEXH(fTimeHdl,i);
		if (err = WriteMacValue(bfpb, time)) goto done;
	}
	
	{if (err = WriteMacValue(bfpb, fTimeShift)) goto done;}
	if (err = WriteMacValue(bfpb, fAllowExtrapolationInTime)) goto done;
	
	numFiles = GetNumFiles();
	if (err = WriteMacValue(bfpb, numFiles)) goto done;
	if (numFiles > 0)
	{
		for (i = 0 ; i < numFiles ; i++) {
			fileInfo = INDEXH(fInputFilesHdl,i);
			if (err = WriteMacValue(bfpb, fileInfo.pathName, kMaxNameLen)) goto done;
			if (err = WriteMacValue(bfpb, fileInfo.startTime)) goto done;
			if (err = WriteMacValue(bfpb, fileInfo.endTime)) goto done;
		}
		if (err = WriteMacValue(bfpb, fOverLap)) return err;
		if (err = WriteMacValue(bfpb, fOverLapStartTime)) return err;
	}
	
	
done:
	if(err)
		TechError("TimeGridVel::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr TimeGridVel::Read(BFPB *bfpb)
{
	char c, msg[256], fileName[256], newFileName[64];
	long i, version, numTimes, numPoints, numFiles;
	ClassID id;
	float val, depthLevel;
	Seconds time;
	PtCurFileInfo fileInfo;
	Boolean bPathIsValid = true;
	OSErr err = 0;
	
	return err;
	StartReadWriteSequence("TimeGridVel::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TimeGridVel::Read()", "id != TYPE_TIMEGRIDVEL", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > TimeGridVelREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	// read the type of grid used 
	if (err = ReadMacValue(bfpb,&id)) return err;
	switch(id)
	{
		case TYPE_RECTGRIDVEL: fGrid = new TRectGridVel;break;
		case TYPE_TRIGRIDVEL: fGrid = new TTriGridVel;break;
		default: printError("Unrecognized Grid type in TimeGridVel::Read()."); return -1;
	}
	
	if (err = fGrid -> Read (bfpb)) goto done;
	
	if (err = ReadMacValue(bfpb, &fNumRows)) goto done;	
	if (err = ReadMacValue(bfpb, &fNumCols)) goto done;	
	if (err = ReadMacValue(bfpb, fVar.pathName, kMaxNameLen)) goto done;
	ResolvePath(fVar.pathName); // JLM 6/3/10
	if (!FileExists(0,0,fVar.pathName)) 
	{	// allow user to put file in local directory
		char newPath[kMaxNameLen],*p;
		strcpy(fileName,"");
		strcpy(newPath,fVar.pathName);
		p = strrchr(newPath,DIRDELIMITER);
		if (p) 
		{
			strcpy(fileName,p);
			ResolvePath(fileName);
		}
		if (!fileName[0] || !FileExists(0,0,fileName)) 
		{bPathIsValid = false;}
		else
			strcpy(fVar.pathName,fileName);
		
	}
	if (err = ReadMacValue(bfpb, fVar.userName, kPtCurUserNameLen)) return err;
	
	if (!bPathIsValid)
	{	// try other platform
		char delimStr[32] ={DIRDELIMITER,0};		
		strcpy(fileName,delimStr);
		strcat(fileName,fVar.userName);
		ResolvePath(fileName);
		if (!fileName[0] || !FileExists(0,0,fileName)) 
		{bPathIsValid = false;}
		else
		{
			strcpy(fVar.pathName,fileName);
			bPathIsValid = true;
		}
	}
	// otherwise ask the user
	if(!bPathIsValid)
	{
		Point where;
		OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
		MySFReply reply;
		where = CenteredDialogUpLeft(M38c);
		char newPath[kMaxNameLen], s[kMaxNameLen];
		sprintf(msg,"This save file references a file which cannot be found.  Please find the file \"%s\".",fVar.pathName);printNote(msg);
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
					 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		//if (!reply.good) return USERCANCEL;	// just keep going...
		if (reply.good)
		{
			strcpy(newPath, reply.fullPath);
			strcpy (s, newPath);
			SplitPathFile (s, newFileName);
			strcpy (fVar.pathName, newPath);
			strcpy (fVar.userName, newFileName);
		}
#else
		sfpgetfile(&where, "",
				   (FileFilterUPP)0,
				   -1, typeList,
				   (DlgHookUPP)0,
				   &reply, M38c,
				   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		//if (!reply.good) return 0;	// just keep going...
		if (reply.good)
		{
			my_p2cstr(reply.fName);
#ifdef MAC
			GetFullPath(reply.vRefNum, 0, (char *)reply.fName, newPath);
#else
			strcpy(newPath, reply.fName);
#endif
			
			strcpy (s, newPath);
			SplitPathFile (s, newFileName);
			strcpy (fVar.pathName, newPath);
			strcpy (fVar.userName, newFileName);
		}
#endif
	}
	
	if (err = ReadMacValue(bfpb, &fVar.fileScaleFactor)) return err;
	//

	if (err = ReadMacValue(bfpb, &fVar.maxNumDepths)) return err;
	if (err = ReadMacValue(bfpb, &fVar.gridType)) return err;
	//
	if (err = ReadMacValue(bfpb, &fFillValue)) return err;
	//if (err = ReadMacValue(bfpb, &fIsNavy)) return err;
	//
	if (err = ReadMacValue(bfpb, &numTimes)) goto done;	
	fTimeHdl = (Seconds**)_NewHandleClear(sizeof(Seconds)*numTimes);
	if(!fTimeHdl)
	{TechError("TimeGridVel::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numTimes ; i++) {
		if (err = ReadMacValue(bfpb, &time)) goto done;
		INDEXH(fTimeHdl, i) = time;
	}

	{if (err = ReadMacValue(bfpb, &fTimeShift)) goto done;}
	{if (err = ReadMacValue(bfpb, &fAllowExtrapolationInTime)) goto done;}
	
	{
		if (err = ReadMacValue(bfpb, &numFiles)) goto done;	
		if (numFiles > 0)
		{
			fInputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
			if(!fInputFilesHdl)
			{TechError("TimeGridVel::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
			for (i = 0 ; i < numFiles ; i++) {
				if (err = ReadMacValue(bfpb, fileInfo.pathName, kMaxNameLen)) goto done;
				ResolvePath(fileInfo.pathName); // JLM 6/3/10
				if (err = ReadMacValue(bfpb, &fileInfo.startTime)) goto done;
				if (err = ReadMacValue(bfpb, &fileInfo.endTime)) goto done;
				INDEXH(fInputFilesHdl,i) = fileInfo;
			}
			if (err = ReadMacValue(bfpb, &fOverLap)) return err;
			if (err = ReadMacValue(bfpb, &fOverLapStartTime)) return err;
		}
	}
	
		
done:
	if(err)
	{
		TechError("TimeGridVel::Read(char* path)", " ", 0); 
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}
	}
	return err;
}


/*void TimeGridVel::DrawContourScale(Rect r, WorldRect view)
{
	Point		p;
	short		h,v,x,y,dY,widestNum=0;
	RGBColor	rgb;
	Rect		rgbrect;
	Rect legendRect = fLegendRect;
	char 		numstr[30],numstr2[30],text[30],errmsg[256];
	long 		i,numLevels,istep=1;
	double	minLevel, maxLevel;
	double 	value;
	float totalDepth = 0;
	long numDepths = 0, numTris = 0, triNum = 0;
	OSErr err = 0;
	PtCurMap *map = GetPtCurMap();
	TTriGridVel *triGrid = (TTriGridVel3D*) map->GetGrid3D(false);
	Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
	
	// code goes here, need separate cases for each grid type - have depth data on points, not triangles...
	long timeDataInterval;
	Boolean loaded;
	
	return;	// will need to select grid cells or points rather than triangles here
	
	err = this -> SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
	
	if(err) return;
	
	loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
	
	if(!loaded) return;
	
	
	if (!fDepthDataInfo) return;
	numTris = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);	// depth from input file (?) at triangle center
	
	//list which triNum, use selected triangle, scale arrows, list values ??? 
	if (triSelected)
	{
		for (i=0;i<numTris; i++)
		{
			if ((*triSelected)[i]) 
			{
				triNum = i;
				break;
			}
		}
	}
	else
		triNum = GetRandom(0,numTris-1);
	
	// code goes here, probably need different code for each grid type - how to select a grid box?, allow to select triangles on curvilinear grid? different for regular grid	
	numDepths = INDEXH(fDepthDataInfo,triNum).numDepths;
	totalDepth = INDEXH(fDepthDataInfo,triNum).totalDepth;	// depth from input file (?) at triangle center
	
	//SetRGBColor(&rgb,0,0,0);
	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
#ifdef IBM
	TextFont(kFontIDGeneva); TextSize(6);
#endif
	
	if (gSavingOrPrintingPictFile)
	{
		Rect mapRect;
#ifdef MAC
		mapRect = DrawingRect(settings.listWidth + 1, RIGHTBARWIDTH);
#else
		mapRect = DrawingRect(settings.listWidth, RIGHTBARWIDTH);
#endif
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
		}
	}
	else
	{
		if (EmptyRect(&fLegendRect)||!RectInRect2(&legendRect,&r))
		{
			legendRect.top = r.top;
			legendRect.left = r.right - 80;
			//legendRect.bottom = r.top + 120;	// reset after contour levels drawn
			legendRect.bottom = r.top + 90;	// reset after contour levels drawn
			legendRect.right = r.right;	// reset if values go beyond default width
		}
	}
	rgbrect = legendRect;
	EraseRect(&rgbrect);
	
	x = (rgbrect.left + rgbrect.right) / 2;
	//dY = RectHeight(rgbrect) / 12;
	dY = 10;
	y = rgbrect.top + dY / 2;
	MyMoveTo(x - stringwidth("Depth Barbs") / 2, y + dY);
	drawstring("Depth Barbs");
	numtostring(triNum+1,numstr);
	strcpy(numstr2,"Tri Num = ");
	strcat(numstr2,numstr);
	MyMoveTo(x-stringwidth(numstr2) / 2, y + 2*dY);
	drawstring(numstr2);
	widestNum = stringwidth(numstr2);
	
	v = rgbrect.top+45;
	h = rgbrect.left;
	//if (numDepths>20) istep = (long)numDepths/20.;
	//for (i=0;i<numDepths;i++)
	for (i=0;i<numDepths;i+=istep)
	{
		WorldPoint wp;
		Point p,p2;
		VelocityRec velocity = {0.,0.};
		Boolean offQuickDrawPlane = false;
		
		long velDepthIndex1 = (*fDepthDataInfo)[triNum].indexToDepthData+i;
		
		velocity.u = INDEXH(fStartData.dataHdl,velDepthIndex1).u;
		velocity.v = INDEXH(fStartData.dataHdl,velDepthIndex1).v;
		
		MyMoveTo(h+40,v+.5);
		
		if ((velocity.u != 0 || velocity.v != 0))
		{
			float inchesX = (velocity.u * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
			float inchesY = (velocity.v * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
			short pixX = inchesX * PixelsPerInchCurrent();
			short pixY = inchesY * PixelsPerInchCurrent();
			//p.h = h+20;
			p.h = h+40;
			p.v = v+.5;
			p2.h = p.h + pixX;
			p2.v = p.v - pixY;
			//MyMoveTo(p.h, p.v);
			MyLineTo(p2.h, p2.v);
			MyDrawArrow(p.h,p.v,p2.h,p2.v);
		}
		if (p2.h-h>widestNum) widestNum = p2.h-h;	// also issue of negative velocity, or super large value, maybe scale?
		v = v+9;
	}
	sprintf(text, "Depth: %g m",totalDepth);
	//MyMoveTo(x - stringwidth(text) / 2, y + 3 * dY);
	MyMoveTo(h+20, v+5);
	drawstring(text);
	if (stringwidth(text)+20 > widestNum) widestNum = stringwidth(text)+20;
	v = v + 9;
	legendRect.bottom = v+3;
	if (legendRect.right<h+20+widestNum+4) legendRect.right = h+20+widestNum+4;
	else if (legendRect.right>legendRect.left+80 && h+20+widestNum+4<=legendRect.left+80)
		legendRect.right = legendRect.left+80;	// may want to redraw to recenter the header
	RGBForeColor(&colors[BLACK]);
 	//MyFrameRect(&legendRect);
	
	if (!gSavingOrPrintingPictFile)
		fLegendRect = legendRect;
	return;
}*/

Boolean TimeGridVelRect::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth)
{
	char uStr[32],sStr[32],errmsg[256];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha, depthAlpha;
	float topDepth, bottomDepth;
	long index;
	LongPoint indices;
	long depthIndex1,depthIndex2;	// default to -1?
	
	if (arrowDepth>0 && fVar.gridType==TWO_D)
	{		
		if (fAllowVerticalExtrapolationOfCurrents && fMaxDepthForExtrapolation >= arrowDepth)
		{
		}
		else
		{
			velocity.u = 0.;
			velocity.v = 0.;
			goto CalcStr;
		}
	}
	
	GetDepthIndices(0,arrowDepth,&depthIndex1,&depthIndex2);
	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		topDepth = INDEXH(fDepthLevelsHdl,depthIndex1);
		bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2);
		depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
	}
	
	if(GetNumTimesInFile()>1)
	{
		if (err = this->GetStartTime(&startTime)) return false;	// should this stop us from showing any velocity?
		if (err = this->GetEndTime(&endTime)) 
		{
			if ((time > startTime || time < startTime) && fAllowExtrapolationInTime)
			{
				timeAlpha = 1;
			}
			else
				return false;
		}
		else
			timeAlpha = (endTime - time)/(double)(endTime - startTime);	
	}
	
	{	
		index = this->GetVelocityIndex(wp.p);	// need alternative for curvilinear and triangular
		
		indices = this->GetVelocityIndices(wp.p);
		
		if (index >= 0)
		{
			// Check for constant current 
			if(GetNumTimesInFile()==1 || timeAlpha == 1)
			{
				if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
				{
					velocity.u = this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols);
					velocity.v = this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols);
				}
				else
				{
					velocity.u = depthAlpha*this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*this->GetStartUVelocity(index+depthIndex2*fNumRows*fNumCols);
					velocity.v = depthAlpha*this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*this->GetStartVVelocity(index+depthIndex2*fNumRows*fNumCols);
				}
			}
			else // time varying current
			{
				if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
				{
					velocity.u = timeAlpha*this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndUVelocity(index+depthIndex1*fNumRows*fNumCols);
					velocity.v = timeAlpha*this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndVVelocity(index+depthIndex1*fNumRows*fNumCols);
				}
				else	// below surface velocity
				{
					velocity.u = depthAlpha*(timeAlpha*this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndUVelocity(index+depthIndex1*fNumRows*fNumCols));
					velocity.u += (1-depthAlpha)*(timeAlpha*this->GetStartUVelocity(index+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndUVelocity(index+depthIndex2*fNumRows*fNumCols));
					velocity.v = depthAlpha*(timeAlpha*this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndVVelocity(index+depthIndex1*fNumRows*fNumCols));
					velocity.v += (1-depthAlpha)*(timeAlpha*this->GetStartVVelocity(index+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndVVelocity(index+depthIndex2*fNumRows*fNumCols));
				}
			}
		}
	}
	
CalcStr:
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = lengthU * fVar.fileScaleFactor; //factor from dialog box vs factor from file - this should be passed in 
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	
	if (indices.h >= 0 && fNumRows-indices.v-1 >=0 && indices.h < fNumCols && fNumRows-indices.v-1 < fNumRows)
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
			fVar.userName, uStr, sStr, fNumRows-indices.v-1, indices.h);
		//sprintf(diagnosticStr, " [grid: unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
				//uStr, sStr, fNumRows-indices.v-1, indices.h);
	}
	else
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
			fVar.userName, uStr, sStr);
		//sprintf(diagnosticStr, " [grid: unscaled: %s m/s, scaled: %s m/s]",
				//uStr, sStr);
	}
	
	return true;
}

void TimeGridVelRect::Draw(Rect r, WorldRect view,double refScale,double arrowScale,
					   double arrowDepth,Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor) 
{	// Use this for regular grid or regridded data
	short row, col, pixX, pixY;
	long dLong, dLat, index, timeDataInterval;
	float inchesX, inchesY;
	double timeAlpha,depthAlpha;
	float topDepth,bottomDepth;
	Seconds startTime, endTime, time = model->GetModelTime();
	Point p, p2;
	WorldPoint wp;
	WorldRect boundsRect, bounds;
	VelocityRec velocity;
	Rect c, newCATSgridRect = {0, 0, fNumRows - 1, fNumCols - 1}; // fNumRows, fNumCols members of TimeGridVel
	Boolean offQuickDrawPlane = false, loaded;
	char errmsg[256];
	Boolean showSubsurfaceVel = false;
	OSErr err = 0;
	long depthIndex1,depthIndex2;	// default to -1?
	Rect currentMapDrawingRect = MapDrawingRect();
	WorldRect cmdr;
	long startRow,endRow,startCol,endCol,dx,dy;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	
	
	if (!bDrawArrows && !bDrawGrid) return;
	
	bounds = rectGrid->GetBounds();
	
	// need to get the bounds from the grid
	dLong = (WRectWidth(bounds) / TimeGridVel_c::fNumCols) / 2;
	dLat = (WRectHeight(bounds) / TimeGridVel_c::fNumRows) / 2;
	//RGBForeColor(&colors[PURPLE]);
	RGBForeColor(&arrowColor);
	
	boundsRect = bounds;
	InsetWRect (&boundsRect, dLong, dLat);
	
	if ((fAllowVerticalExtrapolationOfCurrents && fMaxDepthForExtrapolation >= arrowDepth) || (!fAllowVerticalExtrapolationOfCurrents && arrowDepth > 0)) showSubsurfaceVel = true;
	
	if (bDrawArrows)
	{
		err = this -> SetInterval(errmsg, model->GetModelTime());	
		
		if(err && !bDrawGrid) return;	// want to show grid even if there's no current data
		
		loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	
		
		if(!loaded && !bDrawGrid) return;
		
		if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
		{
			// Calculate the time weight factor
			if (GetNumFiles()>1 && fOverLap)
				startTime = fOverLapStartTime + fTimeShift;
			else
				startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
			//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
			if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationInTime)
			{
				timeAlpha = 1;
			}
			else
			{	//return false;
				endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
		}
	}	
	
	GetDepthIndices(0,arrowDepth,&depthIndex1,&depthIndex2);
	if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
		return;	// no value for this point at chosen depth
		//continue;	// no value for this point at chosen depth

	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		topDepth = INDEXH(fDepthLevelsHdl,depthIndex1);
		bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2);
		depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
	}
	// only draw the vectors and grid points that are in the current view
	cmdr = ScreenToWorldRect(currentMapDrawingRect, MapDrawingRect(), settings.currentView);	// have a look at this to see how to recognize out of view points
	dx = (boundsRect.hiLong - boundsRect.loLong) / (fNumCols - 1);
	dy = (boundsRect.hiLat - boundsRect.loLat) / (fNumRows - 1);
	if (boundsRect.loLong < cmdr.loLong) startCol = (cmdr.loLong - boundsRect.loLong) / dx; else startCol = 0;
	if (boundsRect.hiLong > cmdr.hiLong) endCol = fNumCols - (boundsRect.hiLong - cmdr.hiLong) / dx; else endCol = fNumCols;
	if (boundsRect.loLat < cmdr.loLat) endRow =  fNumRows - (cmdr.loLat - boundsRect.loLat) / dy; else endRow = fNumRows;
	if (boundsRect.hiLat > cmdr.hiLat) startRow = (boundsRect.hiLat - cmdr.hiLat) / dy; else startRow = 0;
	
	for (row = startRow ; row < endRow ; row++)
	{
		for (col = startCol ; col < endCol ; col++) {
			
			SetPt(&p, col, row);
			wp = ScreenToWorldPoint(p, newCATSgridRect, boundsRect);
			velocity.u = velocity.v = 0.;
			if (loaded && !err)
			{
				index = this->GetVelocityIndex(wp);	// need alternative for curvilinear
				
				if (bDrawArrows && index >= 0)
				{
					// Check for constant current 
					if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha==1)
					{
						if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
						{
							velocity.u = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
							velocity.v = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
						}
						else
						{
							velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u;
							velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v;
						}
					}
					else // time varying current
					{
						if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
						{
							velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
							velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
						}
						else	// below surface velocity
						{
							velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u);
							velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u);
							velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v);
							velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v);
						}
						
					}
				}
			}
			
			p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
			MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
			
			if (bDrawGrid && bDrawArrows && (velocity.u != 0 || velocity.v != 0)) 
				PaintRect(&c);	// should check fill_value
			if (bDrawGrid && !bDrawArrows) 
				PaintRect(&c);	// should check fill_value
			
			if (bDrawArrows && (velocity.u != 0 || velocity.v != 0) && (arrowDepth==0 || showSubsurfaceVel))
			{
				inchesX = (velocity.u * refScale * fVar.fileScaleFactor) / arrowScale;
				inchesY = (velocity.v * refScale * fVar.fileScaleFactor) / arrowScale;
				pixX = inchesX * PixelsPerInchCurrent();
				pixY = inchesY * PixelsPerInchCurrent();
				p2.h = p.h + pixX;
				p2.v = p.v - pixY;
				MyMoveTo(p.h, p.v);
				MyLineTo(p2.h, p2.v);
				MyDrawArrow(p.h,p.v,p2.h,p2.v);
			}
		}
	}	
	RGBForeColor(&colors[BLACK]);
}

// for now leave this part out of the python and let the file path list be passed in
OSErr TimeGridVel::ReadInputFileNames(char *fileNamesPath)
{
	// for netcdf files, header file just has the paths, the start and end times will be read from the files
	long i,numScanned,line=0, numFiles, numLinesInText;
	DateTimeRec time;
	Seconds timeSeconds;
	OSErr err = 0;
	char s[1024], path[256], outPath[256], classicPath[kMaxNameLen];
	CHARH fileBufH = 0;
	PtCurFileInfoH inputFilesHdl = 0;
	int status, ncid, recid, timeid;
	size_t recs, t_len, t_len2;
	double timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0;	
	static size_t timeIndex;
	Seconds startTime2;
	double timeConversion = 1.;
	char errmsg[256] = "";
	
	if (err = ReadFileContents(TERMINATED,0, 0, fileNamesPath, 0, 0, &fileBufH)) goto done;
	
	numLinesInText = NumLinesInText(*fileBufH);
	numFiles = numLinesInText - 1;	// subtract off the header
	inputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
	if(!inputFilesHdl) {TechError("TimeGridVel::ReadInputFileNames()", "_NewHandle()", 0); err = memFullErr; goto done;}
	NthLineInTextNonOptimized(*fileBufH, (line)++, s, 1024); 	// header line
	for (i=0;i<numFiles;i++)	// should count files as go along
	{
		NthLineInTextNonOptimized(*fileBufH, (line)++, s, 1024); 	// check it is a [FILE] line
		//strcpy((*inputFilesHdl)[i].pathName,s+strlen("[FILE]\t"));
		RemoveLeadingAndTrailingWhiteSpace(s);
		strcpy((*inputFilesHdl)[i].pathName,s+strlen("[FILE] "));
		RemoveLeadingAndTrailingWhiteSpace((*inputFilesHdl)[i].pathName);
		ResolvePathFromInputFile(fileNamesPath,(*inputFilesHdl)[i].pathName); // JLM 6/8/10
		//strcpy(path,(*inputFilesHdl)[i].pathName);
		if((*inputFilesHdl)[i].pathName[0] && FileExists(0,0,(*inputFilesHdl)[i].pathName))
		{
#if TARGET_API_MAC_CARBON
			err = ConvertTraditionalPathToUnixPath((const char *) (*inputFilesHdl)[i].pathName, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
			strcpy((*inputFilesHdl)[i].pathName,outPath);
#endif
			strcpy(path,(*inputFilesHdl)[i].pathName);
			status = nc_open(path, NC_NOWRITE, &ncid);
			if (status != NC_NOERR) /*{err = -1; goto done;}*/
			{
//#if TARGET_API_MAC_CARBON
				//err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
				//status = nc_open(outPath, NC_NOWRITE, &ncid);
//#endif
				if (status != NC_NOERR) {err = -2; goto done;}
			}
			
			status = nc_inq_dimid(ncid, "time", &recid); 
			if (status != NC_NOERR) 
			{
				status = nc_inq_unlimdim(ncid, &recid);	// maybe time is unlimited dimension
				if (status != NC_NOERR) {err = -2; goto done;}
			}
			
			status = nc_inq_varid(ncid, "time", &timeid); 
			if (status != NC_NOERR) {err = -2; goto done;} 
			
			/////////////////////////////////////////////////
			status = nc_inq_attlen(ncid, timeid, "units", &t_len);
			if (status != NC_NOERR) 
			{
				err = -2; goto done;
			}
			else
			{
				DateTimeRec time;
				char unitStr[24], junk[10];
				
				timeUnits = new char[t_len+1];
				status = nc_get_att_text(ncid, timeid, "units", timeUnits);
				if (status != NC_NOERR) {err = -2; goto done;} 
				timeUnits[t_len] = '\0'; // moved this statement before StringSubstitute, JLM 5/2/10
				StringSubstitute(timeUnits, ':', ' ');
				StringSubstitute(timeUnits, '-', ' ');
				
				numScanned=sscanf(timeUnits, "%s %s %hd %hd %hd %hd %hd %hd",
								  unitStr, junk, &time.year, &time.month, &time.day,
								  &time.hour, &time.minute, &time.second) ;
				if (numScanned==5)	
				{time.hour = 0; time.minute = 0; time.second = 0; }
				else if (numScanned==7)	time.second = 0;
				else if (numScanned<8)	
					//if (numScanned!=8)	
				{ err = -1; TechError("TimeGridVel::ReadInputFileNames()", "sscanf() == 8", 0); goto done; }
				DateToSeconds (&time, &startTime2);	// code goes here, which start Time to use ??
				if (!strcmpnocase(unitStr,"HOURS") || !strcmpnocase(unitStr,"HOUR"))
					timeConversion = 3600.;
				else if (!strcmpnocase(unitStr,"MINUTES") || !strcmpnocase(unitStr,"MINUTE"))
					timeConversion = 60.;
				else if (!strcmpnocase(unitStr,"SECONDS") || !strcmpnocase(unitStr,"SECOND"))
					timeConversion = 1.;
				else if (!strcmpnocase(unitStr,"DAYS") || !strcmpnocase(unitStr,"DAY"))
					timeConversion = 24*3600.;
			} 
			
			status = nc_inq_dim(ncid, recid, recname, &recs);
			if (status != NC_NOERR) {err = -2; goto done;}
			{
				Seconds newTime;
				// possible units are, HOURS, MINUTES, SECONDS,...
				timeIndex = 0;	// first time
				status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
				if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); printError(errmsg); err = -1; goto done;}
				newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
				(*inputFilesHdl)[i].startTime = newTime;
				timeIndex = recs-1;	// last time
				status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
				if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); printError(errmsg); err = -1; goto done;}
				newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
				(*inputFilesHdl)[i].endTime = newTime;
			}
			status = nc_close(ncid);
			if (status != NC_NOERR) {err = -2; goto done;}
		}	
		else 
		{
			char msg[256];
			sprintf(msg,"PATH to NetCDF data File does not exist.%s%s",NEWLINESTRING,(*inputFilesHdl)[i].pathName);
			printError(msg);
			err = true;
			goto done;
		}
		
		
	}
	if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}	// so could replace list
	fInputFilesHdl = inputFilesHdl;
	
done:
	if(fileBufH) { DisposeHandle((Handle)fileBufH); fileBufH = 0;}
	if (err)
	{
		if (err==-2) {printError("Error reading netCDF file");}
		if(inputFilesHdl) {DisposeHandle((Handle)inputFilesHdl); inputFilesHdl=0;}
	}
	return err;
}

TimeGridVelRect::TimeGridVelRect () : TimeGridVel()
{
	fDepthLevelsHdl = 0;	// depth level, sigma, or sc_r
	fDepthLevelsHdl2 = 0;	// Cs_r
	hc = 1.;	// what default?
		
	//fFillValue = -1e+34;
	fIsNavy = false;	
	
	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	fDepthsH = 0;
	fDepthDataInfo = 0;
	
	fNumDepthLevels = 1;	// default surface current only
	
	fAllowVerticalExtrapolationOfCurrents = false;
	fMaxDepthForExtrapolation = 0.;	// assume 2D is just surface
	
}

void TimeGridVelRect::Dispose ()
{
	if(fDepthLevelsHdl) {DisposeHandle((Handle)fDepthLevelsHdl); fDepthLevelsHdl=0;}
	
	if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
	if(fDepthDataInfo) {DisposeHandle((Handle)fDepthDataInfo); fDepthDataInfo=0;}
	
	TimeGridVel::Dispose ();
}


#define TimeGridVelRectREADWRITEVERSION 1 // don't bother with for now

OSErr TimeGridVelRect::Write (BFPB *bfpb)
{
	long i, version = TimeGridVelRectREADWRITEVERSION; 
	ClassID id = GetClassID ();
	long numTimes = GetNumTimesInFile(), numPoints = 0, numPts = 0, numFiles = 0;
	long 	numDepths = this->GetNumDepths();
	Seconds time;
	float val, depthLevel;
	DepthDataInfo depthData;
	PtCurFileInfo fileInfo;
	OSErr err = 0;
	
	if (err = TimeGridVel::Write(bfpb)) return err;

	StartReadWriteSequence("TimeGridVelRect::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	if (err = WriteMacValue(bfpb, numDepths)) goto done;
	for (i=0;i<numDepths;i++)
	{
		val = INDEXH(fDepthsH,i);
		if (err = WriteMacValue(bfpb, val)) goto done;
	}
	
	if (fDepthDataInfo) numPoints = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i = 0 ; i < numPoints ; i++) {
		depthData = INDEXH(fDepthDataInfo,i);
		if (err = WriteMacValue(bfpb, depthData.totalDepth)) goto done;
		if (err = WriteMacValue(bfpb, depthData.indexToDepthData)) goto done;
		if (err = WriteMacValue(bfpb, depthData.numDepths)) goto done;
	}
	if (err = WriteMacValue(bfpb, fAllowVerticalExtrapolationOfCurrents)) goto done;
	if (err = WriteMacValue(bfpb, fMaxDepthForExtrapolation)) goto done;
	//fNumDepthLevels,fDepthLevelsHdl
	
	if (fDepthLevelsHdl) numPts = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	if (err = WriteMacValue(bfpb, numPts)) goto done;
	for (i = 0 ; i < numPts ; i++) {
		depthLevel = INDEXH(fDepthLevelsHdl,i);
		if (err = WriteMacValue(bfpb, depthLevel)) goto done;
	}
	
	if (fDepthLevelsHdl2) 
	{
		numPts = _GetHandleSize((Handle)fDepthLevelsHdl2)/sizeof(**fDepthLevelsHdl2);
		if (err = WriteMacValue(bfpb, numPts)) goto done;
		for (i = 0 ; i < numPts ; i++) {
			depthLevel = INDEXH(fDepthLevelsHdl2,i);
			if (err = WriteMacValue(bfpb, depthLevel)) goto done;
		}
	}
	else
	{
		numPts = 0;
		if (err = WriteMacValue(bfpb, numPts)) goto done;
	}
	if (err = WriteMacValue(bfpb, hc)) goto done;
	
done:
	if(err)
		TechError("TimeGridVelRect::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr TimeGridVelRect::Read(BFPB *bfpb)
{
	char c, msg[256], fileName[256], newFileName[64];
	long i, version, numDepths, numTimes, numPoints, numFiles;
	ClassID id;
	float val, depthLevel;
	Seconds time;
	DepthDataInfo depthData;
	Boolean bPathIsValid = true;
	OSErr err = 0;
	
	if (err = TimeGridVel::Read(bfpb)) return err;

	StartReadWriteSequence("TimeGridVelRect::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TimeGridVelRect::Read()", "id != TYPE_TIMEGRIDVELRECT", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > TimeGridVelRectREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////

	if (err = ReadMacValue(bfpb, &fIsNavy)) return err;
	//
	{
		if (err = ReadMacValue(bfpb, &numDepths)) goto done;	
		if (numDepths>0)
		{
			fDepthsH = (FLOATH)_NewHandleClear(sizeof(float)*numDepths);
			if (!fDepthsH)
			{ TechError("TimeGridVelRect::Read()", "_NewHandleClear()", 0); goto done; }
			
			for (i = 0 ; i < numDepths ; i++) {
				if (err = ReadMacValue(bfpb, &val)) goto done;
				INDEXH(fDepthsH, i) = val;
			}
		}
	}

	{
		if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
		fDepthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(DepthDataInfo)*numPoints);
		if(!fDepthDataInfo)
		{TechError("TimeGridVelRect::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
		for (i = 0 ; i < numPoints ; i++) {
			if (err = ReadMacValue(bfpb, &depthData.totalDepth)) goto done;
			if (err = ReadMacValue(bfpb, &depthData.indexToDepthData)) goto done;
			if (err = ReadMacValue(bfpb, &depthData.numDepths)) goto done;
			INDEXH(fDepthDataInfo, i) = depthData;
		}
	}

	{
		if (err = ReadMacValue(bfpb, &fAllowVerticalExtrapolationOfCurrents)) goto done;
		if (err = ReadMacValue(bfpb, &fMaxDepthForExtrapolation)) goto done;
	}
	//fNumDepthLevels, fDepthLevelsHdl
	{
		if (err = ReadMacValue(bfpb, &numPoints)) goto done;
		if (numPoints>0)
		{
			fNumDepthLevels = numPoints;
			fDepthLevelsHdl = (FLOATH)_NewHandleClear(numPoints * sizeof(float));
			if (!fDepthLevelsHdl) {err = memFullErr; goto done;}
			for (i = 0 ; i < numPoints ; i++) 
			{
				if (err = ReadMacValue(bfpb, &depthLevel)) goto done;
				INDEXH(fDepthLevelsHdl, i) = depthLevel;
			}
		}
	}
	
	{	
		if (err = ReadMacValue(bfpb, &numPoints)) goto done;
		if (numPoints>0)
		{
			//fNumDepthLevels = numPoints;	// this should be the same as above
			fDepthLevelsHdl2 = (FLOATH)_NewHandleClear(numPoints * sizeof(float));
			if (!fDepthLevelsHdl2) {err = memFullErr; goto done;}
			for (i = 0 ; i < numPoints ; i++) 
			{
				if (err = ReadMacValue(bfpb, &depthLevel)) goto done;
				INDEXH(fDepthLevelsHdl2, i) = depthLevel;
			}
		}
		if (err = ReadMacValue(bfpb, &hc)) goto done;
	}
	
done:
	if(err)
	{
		TechError("TimeGridVelRect::Read(char* path)", " ", 0); 
		if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
		if(fDepthDataInfo) {DisposeHandle((Handle)fDepthDataInfo); fDepthDataInfo=0;}
	}
	return err;
}

void TimeGridVelCurv::Dispose ()
{
	if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
	if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	
	TimeGridVelRect::Dispose ();
}



TimeGridVelCurv::TimeGridVelCurv () : TimeGridVelRect()
{
	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
	bIsCOOPSWaterMask = false;
}
	
#define TimeGridVelCurvREADWRITEVERSION 1 //JLM

OSErr TimeGridVelCurv::Write (BFPB *bfpb)
{
	long i, version = TimeGridVelCurvREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numPoints = 0, numPts = 0, index;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = TimeGridVelRect::Write (bfpb)) return err;
	
	StartReadWriteSequence("TimeGridVelCurv::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	
	if (fVerdatToNetCDFH) numPoints = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(**fVerdatToNetCDFH);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i=0;i<numPoints;i++)
	{
		index = INDEXH(fVerdatToNetCDFH,i);
		if (err = WriteMacValue(bfpb, index)) goto done;
	}
	
	if (fVertexPtsH) numPts = _GetHandleSize((Handle)fVertexPtsH)/sizeof(**fVertexPtsH);
	if (err = WriteMacValue(bfpb, numPts)) goto done;
	for (i=0;i<numPts;i++)
	{
		vertex = INDEXH(fVertexPtsH,i);
		if (err = WriteMacValue(bfpb, vertex.pLat)) goto done;
		if (err = WriteMacValue(bfpb, vertex.pLong)) goto done;
	}
	
done:
	if(err)
		TechError("TimeGridVelCurv::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr TimeGridVelCurv::Read(BFPB *bfpb)	
{
	long i, version, index, numPoints;
	ClassID id;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = TimeGridVelRect::Read(bfpb)) return err;
	
	StartReadWriteSequence("TimeGridVelCurv::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TimeGridVelCurv::Read()", "id != TYPE_TIMEGRIDVELCURV", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != TimeGridVelCurvREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVerdatToNetCDFH = (LONGH)_NewHandleClear(sizeof(long)*numPoints);	// for curvilinear
	if(!fVerdatToNetCDFH)
	{TechError("TimeGridVelCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &index)) goto done;
		INDEXH(fVerdatToNetCDFH, i) = index;
	}
	
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVertexPtsH = (WORLDPOINTFH)_NewHandleClear(sizeof(WorldPointF)*numPoints);	// for curvilinear
	if(!fVertexPtsH)
	{TechError("TimeGridVelCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &vertex.pLat)) goto done;
		if (err = ReadMacValue(bfpb, &vertex.pLong)) goto done;
		INDEXH(fVertexPtsH, i) = vertex;
	}
	
done:
	if(err)
	{
		TechError("TimeGridVelCurv::Read(char* path)", " ", 0); 
		if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
		if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	}
	return err;
}

Boolean TimeGridVelCurv::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth)
{	// code goes here, this is triangle code, not curvilinear
	char uStr[32],vStr[32],sStr[32],depthStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha, depthAlpha;
	float topDepth, bottomDepth, totalDepth = 0.;
	long index = 0;
	LongPoint indices;
	
	long ptIndex1,ptIndex2,ptIndex3; 
	InterpolationVal interpolationVal;
	long depthIndex1,depthIndex2;	// default to -1?
	
	
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		if (bIsCOOPSWaterMask)
		{
			//code goes here, put in interpolation
			interpolationVal = fGrid -> GetInterpolationValues(wp.p);
			if (interpolationVal.ptIndex1<0) return false;
			//ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			//ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			//ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
			index = (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];
			//index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp.p,fVerdatToNetCDFH,fNumCols);// curvilinear grid
		}
		else
			index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
		if (index < 0) return 0;
		indices = this->GetVelocityIndices(wp.p);
	}
	else return 0;
	totalDepth = GetTotalDepth(wp.p,index);
	GetDepthIndices(index,arrowDepth,totalDepth,&depthIndex1,&depthIndex2);
	if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
		return false;	// no value for this point at chosen depth - should show as 0,0 or nothing?
	
	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		//topDepth = INDEXH(fDepthLevelsHdl,depthIndex1)*totalDepth; // times totalDepth
		//bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2)*totalDepth;
		topDepth = GetDepthAtIndex(depthIndex1,totalDepth); // times totalDepth
		bottomDepth = GetDepthAtIndex(depthIndex2,totalDepth); // times totalDepth
		//topDepth = GetTopDepth(depthIndex1,totalDepth); // times totalDepth
		//bottomDepth = GetBottomDepth(depthIndex2,totalDepth);
		if (totalDepth == 0) depthAlpha = 1;
		else
			depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
	}
	
	// Check for constant current 
	if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime)  || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime))
		//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0 && depthIndex1 >= 0) 
		{
			//velocity.u = INDEXH(fStartData.dataHdl,index).u;
			//velocity.v = INDEXH(fStartData.dataHdl,index).v;
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				velocity.u = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				velocity.v = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}	
			else
			{
				velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u;
				velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v;
			}
		}
		else	// set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		// Calculate the time weight factor
		startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (index >= 0 && depthIndex1 >= 0) 
		{
			//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}
			else
			{
				velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u);
				velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u);
				velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v);
				velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v);
			}
		}
		else	// set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	//lengthS = this->fWindScale * lengthU;
	//lengthS = this->fVar.curScale * lengthU;
	//lengthS = this->fVar.curScale * fFileScaleFactor * lengthU;
	lengthS = lengthU * fVar.fileScaleFactor;	// pass this in
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	if (indices.h >= 0 && fNumRows-indices.v-1 >=0 && indices.h < fNumCols && fNumRows-indices.v-1 < fNumRows)
	{
		//sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
		//this->className, uStr, sStr, fNumRows-indices.v-1, indices.h);
		StringWithoutTrailingZeros(uStr,fVar.fileScaleFactor*velocity.u,4);
		StringWithoutTrailingZeros(vStr,fVar.fileScaleFactor*velocity.v,4);
		StringWithoutTrailingZeros(depthStr,depthIndex1,4);
		sprintf(diagnosticStr, " [grid: %s, u vel: %s m/s, v vel: %s m/s], file indices : [%ld, %ld]",
			fVar.userName, uStr, vStr, fNumRows-indices.v-1, indices.h);
		//sprintf(diagnosticStr, " [grid: u vel: %s m/s, v vel: %s m/s], file indices : [%ld, %ld]",
				//uStr, vStr, fNumRows-indices.v-1, indices.h);
		//if (depthIndex1>0 || !(depthIndex2==UNASSIGNEDINDEX))
		if (fVar.gridType!=TWO_D)
			sprintf(diagnosticStr, " [grid: %s, u vel: %s m/s, v vel: %s m/s], file indices : [%ld, %ld, %ld]",
				fVar.userName, uStr, vStr, fNumRows-indices.v-1, indices.h, depthIndex1);
			//sprintf(diagnosticStr, " [grid: u vel: %s m/s, v vel: %s m/s], file indices : [%ld, %ld, %ld]",
					//uStr, vStr, fNumRows-indices.v-1, indices.h, depthIndex1);
	}
	else
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
			fVar.userName, uStr, sStr);
		//sprintf(diagnosticStr, " [grid: unscaled: %s m/s, scaled: %s m/s]",
				//uStr, sStr);
	}
	
	return true;
}

void TimeGridVelCurv::Draw(Rect r, WorldRect view,double refScale,double arrowScale,
						   double arrowDepth,Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor) 
{	// use for curvilinear
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	double timeAlpha,depthAlpha;
	float topDepth,bottomDepth;
	Point p;
	Rect c;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	OSErr err = 0;
	char errmsg[256];
	long depthIndex1,depthIndex2;	// default to -1?
	long amtOfDepthData = 0;
	Rect currentMapDrawingRect = MapDrawingRect();
	WorldRect cmdr;
	
	
	//RGBForeColor(&colors[PURPLE]);
	RGBForeColor(&arrowColor);
	
	if (fDepthLevelsHdl) amtOfDepthData = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	
	if(bDrawArrows || bDrawGrid)
	{
		Boolean overrideDrawArrows = FALSE;
		//if (bDrawGrid) 	// make sure to draw grid even if don't draw arrows
		 //{
		 //((TTriGridVel*)fGrid)->DrawCurvGridPts(r,view);
		 //return;
		 //}	 // I think this is redundant with the draw triangle (maybe just a diagnostic)
		if (bDrawArrows)
		{ // we have to draw the arrows
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;
			
			err = this -> SetInterval(errmsg, model->GetModelTime());
			if(err) return;
			
			loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	 
			
			if(!loaded) return;
			
			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			// Check for time varying current 
			if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
				//if(GetNumTimesInFile()>1 && loaded && !err)
			{
				// Calculate the time weight factor
				if (GetNumFiles()>1 && fOverLap)
					startTime = fOverLapStartTime + fTimeShift;
				else
					startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationInTime)
					//if (fEndData.timeIndex == UNASSIGNEDINDEX && time != startTime && fAllowExtrapolationOfCurrentsInTime)
				{
					timeAlpha = 1;
				}
				else
				{	//return false;
					endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
					timeAlpha = (endTime - time)/(double)(endTime - startTime);
				}
				//endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
			
			cmdr = ScreenToWorldRect(currentMapDrawingRect, MapDrawingRect(), settings.currentView);	// have a look at this to see how to recognize out of view points
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				long ptIndex=-1,iIndex,jIndex;
				//long ptIndex2=-1,iIndex2,jIndex2;
				WorldPoint wp,wp2;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;				
				float totalDepth=0.;
				
				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				ptIndex = INDEXH(fVerdatToNetCDFH,i);
				
				if (bIsCOOPSWaterMask)
				{
				iIndex = ptIndex/(fNumCols);
				jIndex = ptIndex%(fNumCols);
				}
				else
				{
				iIndex = ptIndex/(fNumCols+1);
				jIndex = ptIndex%(fNumCols+1);
				}
				if (iIndex>0 && jIndex<fNumCols)
					ptIndex = (iIndex-1)*(fNumCols)+jIndex;
				else
				{ptIndex = -1; continue;}
				
				if (bIsCOOPSWaterMask) ptIndex = INDEXH(fVerdatToNetCDFH,i);
				totalDepth = GetTotalDepth(wp,ptIndex);
	 			if (amtOfDepthData>0 && ptIndex>=0)
				{
					if (totalDepth==-1)
					{
						depthIndex1 = -1; depthIndex2 = -1;
					}
					else GetDepthIndices(ptIndex,arrowDepth,totalDepth,&depthIndex1,&depthIndex2);
				}
				else
				{	// for old SAV files without fDepthDataInfo
					//depthIndex1 = ptIndex;
					depthIndex1 = 0;
					depthIndex2 = -1;
				}
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth
				
				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					//topDepth = INDEXH(fDepthLevelsHdl,depthIndex1)*totalDepth; // times totalDepth
					//bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2)*totalDepth;
					topDepth = GetDepthAtIndex(depthIndex1,totalDepth); // times totalDepth
					bottomDepth = GetDepthAtIndex(depthIndex2,totalDepth);
					//topDepth = GetTopDepth(depthIndex1,totalDepth); // times totalDepth
					//bottomDepth = GetBottomDepth(depthIndex2,totalDepth);
					if (totalDepth == 0) depthAlpha = 1;
					else
						depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
				}
				// for now draw arrow at midpoint of diagonal of gridbox
				// this will result in drawing some arrows more than once
				if (GetLatLonFromIndex(iIndex-1,jIndex+1,&wp2)!=-1)	// may want to get all four points and interpolate
				{
					wp.pLat = (wp.pLat + wp2.pLat)/2.;
					wp.pLong = (wp.pLong + wp2.pLong)/2.;
				}
				
				if (bIsCOOPSWaterMask)
				{
					wp.pLat = (long)(1e6*INDEXH(fVertexPtsH,ptIndex).pLat);
					wp.pLong = (long)(1e6*INDEXH(fVertexPtsH,ptIndex).pLong);
				}
				if (wp.pLong < cmdr.loLong || wp.pLong > cmdr.hiLong || wp.pLat < cmdr.loLat || wp.pLat > cmdr.hiLat) 
					continue;
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);	// should put velocities in center of grid box
				
				// Should check vs fFillValue
				// Check for constant current 
				if(((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha == 1) && ptIndex!=-1)
					//if(GetNumTimesInFile()==1 && ptIndex!=-1)
				{
					//velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
					//velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						velocity.u = GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						velocity.v = GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						//velocity.u = INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u;
						//velocity.v = INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v;
					}
					else 	// below surface velocity
					{
						velocity.u = depthAlpha*GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*GetStartUVelocity(ptIndex+depthIndex2*fNumRows*fNumCols);
						velocity.v = depthAlpha*GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*GetStartVVelocity(ptIndex+depthIndex2*fNumRows*fNumCols);
						//velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).u;
						//velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).v;
					}
				}
				else if (ptIndex!=-1)// time varying current
				{
					// need to rescale velocities for Navy case, store angle
					// should check for fillValue, don't want to try to interpolate in that case
					//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
					//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{	
						velocity.u = timeAlpha*GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-timeAlpha)*GetEndUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						velocity.v = timeAlpha*GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-timeAlpha)*GetEndVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u;
						//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v;
					}
					else	// below surface velocity
					{
						velocity.u = depthAlpha*(timeAlpha*GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*GetEndUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols));
						velocity.u += (1-depthAlpha)*(timeAlpha*GetStartUVelocity(ptIndex+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*GetEndUVelocity(ptIndex+depthIndex2*fNumRows*fNumCols));
						velocity.v = depthAlpha*(timeAlpha*GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*GetEndVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols));
						velocity.v += (1-depthAlpha)*(timeAlpha*GetStartVVelocity(ptIndex+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*GetEndVVelocity(ptIndex+depthIndex2*fNumRows*fNumCols));
						//velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u);
						//velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).u);
						//velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v);
						//velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).v);
					}
					//velocity.u = timeAlpha*GetStartUVelocity(ptIndex) + (1-timeAlpha)*GetEndUVelocity(ptIndex);
					//velocity.v = timeAlpha*GetStartVVelocity(ptIndex) + (1-timeAlpha)*GetEndVVelocity(ptIndex);
				}
				if ((velocity.u != 0 || velocity.v != 0) && (velocity.u != fFillValue && velocity.v != fFillValue)) // should already have handled fill value issue
				{
					float inchesX = (velocity.u * refScale * fVar.fileScaleFactor) / arrowScale;
					float inchesY = (velocity.v * refScale * fVar.fileScaleFactor) / arrowScale;
					short pixX = inchesX * PixelsPerInchCurrent();
					short pixY = inchesY * PixelsPerInchCurrent();
					p2.h = p.h + pixX;
					p2.v = p.v - pixY;
					MyMoveTo(p.h, p.v);
					MyLineTo(p2.h, p2.v);
					MyDrawArrow(p.h,p.v,p2.h,p2.v);
				}
			}
		}
	}
	if (bDrawGrid) fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,arrowDepth,false,true,arrowColor);
	//if (bShowDepthContours && fVar.gridType!=TWO_D) ((TTriGridVel*)fGrid)->DrawDepthContours(r,view,bShowDepthContourLabels);// careful with 3D grid
	
	RGBForeColor(&colors[BLACK]);
}
OSErr TimeGridVelCurv::ReadTopology(char* path)
{
	// import NetCDF curvilinear info so don't have to regenerate
	char s[1024], errmsg[256]/*, s[256], topPath[256]*/;
	long i, numPoints, numTopoPoints, line = 0, numPts;
	CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH depths=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds = voidWorldRect;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0, boundaryPts=0;
	
	errmsg[0]=0;
	
	if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("TimeGridVelCurv::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	
	// No header
	// start with transformation array and vertices
	MySpinCursor(); // JLM 8/4/99
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	if(IsTransposeArrayHeaderLine(s,&numPts)) // 
	{
		if (err = ReadTransposeArray(f,&line,&fVerdatToNetCDFH,numPts,errmsg)) 
		{strcpy(errmsg,"Error in ReadTransposeArray"); goto done;}
	}
	else {err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}
	
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
		if (numBoundaryPts>0)
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
	
	if(IsBoundaryPointsHeaderLine(s,&numBoundaryPts)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundaryPts>0)
			err = ReadBoundaryPts(f,&line,&boundaryPts,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary segment header line");
		//goto done;
		// not always needed ? probably always needed for curvilinear
	}
	MySpinCursor(); // JLM 8/4/99
	
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
	// code goes here, do we want to store the grid boundary and land/water information?
	/*if (waterBoundaries && waterBoundaries && boundaryPts)
	{
		//PtCurMap *map = CreateAndInitPtCurMap(fVar.userName,bounds); // the map bounds are the same as the grid bounds
		PtCurMap *map = CreateAndInitPtCurMap("Extended Topology",bounds); // the map bounds are the same as the grid bounds
		if (!map) {strcpy(errmsg,"Error creating ptcur map"); goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundarySegs);	
		map->SetBoundaryPoints(boundaryPts);	
		map->SetWaterBoundaries(waterBoundaries);
		
		*newMap = map;
	}	
	else*/	
	{
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs=0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts=0;}
	}
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TimeGridVelCurv::ReadTopology()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(bounds); 
	//triGrid -> SetDepths(depths);
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to read Extended Topology file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	
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
			strcpy(errmsg,"An error occurred in TimeGridVelCurv::ReadTopology");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		/*if (*newMap) 
		{
			(*newMap)->Dispose();
			delete *newMap;
			*newMap=0;
		}*/
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts = 0;}
	}
	return err;
}

OSErr TimeGridVelCurv::ExportTopology(char* path)
{
	// export NetCDF curvilinear info so don't have to regenerate each time
	// move to NetCDFMover so Tri can use it too
	OSErr err = 0;
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0, nBoundaryPts;
	long i, n, v1,v2,v3,n1,n2,n3;
	double x,y,z=0;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	FLOATH depthsH=0;
	DAGHdl		treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0, boundaryPointsH = 0;
	BFPB bfpb;
	//PtCurMap *map = GetPtCurMap();
	
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
	depthsH = ((TTriGridVel*)triGrid)->GetDepths();
	if(!ptsH || !topH || !treeH) 
	{
		printError("There is no topology to export");
		return -1;
	}
	//if (moverMap->IAm(TYPE_PTCURMAP))
	/*if (map)
	{
		//boundaryTypeH = (dynamic_cast<PtCurMap *>(moverMap))->GetWaterBoundaries();
		//boundarySegmentsH = (dynamic_cast<PtCurMap *>(moverMap))->GetBoundarySegs();
		//boundaryPointsH = (dynamic_cast<PtCurMap *>(moverMap))->GetBoundaryPoints();
		boundaryTypeH = map->GetWaterBoundaries();
		boundarySegmentsH = map->GetBoundarySegs();
		boundaryPointsH = map->GetBoundaryPoints();
		if (!boundaryTypeH || !boundarySegmentsH || !boundaryPointsH) {printError("No map info to export"); err=-1; goto done;}
	}
	else
	{
		// any issue with trying to write out non-existent fields?
	}*/
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	
	// Write out values
	if (fVerdatToNetCDFH) n = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(long);
	else {printError("There is no transpose array"); err = -1; goto done;}
	sprintf(hdrStr,"TransposeArray\t%ld\n",n);	
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i=0;i<n;i++)
	{	
		sprintf(topoStr,"%ld\n",(*fVerdatToNetCDFH)[i]);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
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
		if (depthsH) 
		{
			z = (*depthsH)[i];
			sprintf(topoStr,"%lf\t%lf\t%lf\n",x,y,z);
		}
		else
			sprintf(topoStr,"%lf\t%lf\n",x,y);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	//boundary points - an optional handle, only for curvilinear case
	
	/*if (boundarySegmentsH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundarySegmentsH)/sizeof(long);
		//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
		sprintf(hdrStr,"BoundarySegments\t%ld\n",nBoundarySegs);	// total vertices
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundarySegs;i++)
		{	
			//sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]);
			sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]+1);	// when reading in subtracts 1
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}
	nBoundarySegs = 0;
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
	
	nBoundaryPts = 0;
	if (boundaryPointsH) 
	{
		nBoundaryPts = _GetHandleSize((Handle)boundaryPointsH)/sizeof(long);	// should be same size as previous handle
		sprintf(hdrStr,"BoundaryPoints\t%ld\n",nBoundaryPts);	// total boundary points
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundaryPts;i++)
		{	
			sprintf(topoStr,"%ld\n",(*boundaryPointsH)[i]);	// when reading in subtracts 1
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}*/
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
		sprintf(topoStr, "%ld\t%ld\t%ld\t%ld\t%ld\t%ld\n",
				v1, v2, v3, n1, n2, n3);
		
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

TimeGridVelTri::TimeGridVelTri () : TimeGridVelCurv()
{
	fNumNodes = 0;
	fNumEles = 0;
	bVelocitiesOnTriangles = false;
}

void TimeGridVelTri::Dispose ()
{
	TimeGridVelCurv::Dispose ();
}

#define TimeGridVelTriREADWRITEVERSION 1 //JLM

OSErr TimeGridVelTri::Write (BFPB *bfpb)
{
	long i, version = TimeGridVelTriREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numPoints = 0, numPts = 0, index;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = TimeGridVelCurv::Write (bfpb)) return err;
	
	StartReadWriteSequence("TimeGridVelTri::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	
	if (err = WriteMacValue(bfpb, fNumNodes)) goto done;
	if (err = WriteMacValue(bfpb, fNumEles)) goto done;
	if (err = WriteMacValue(bfpb, bVelocitiesOnTriangles)) goto done;
	
	
done:
	if(err)
		TechError("TimeGridVelTri::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr TimeGridVelTri::Read(BFPB *bfpb)	
{
	long i, version, index, numPoints;
	ClassID id;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = TimeGridVelCurv::Read(bfpb)) return err;
	
	StartReadWriteSequence("TimeGridVelTri::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TimeGridVelTri::Read()", "id != TYPE_TIMEGRIDVELTRI", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != TimeGridVelTriREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	if (err = ReadMacValue(bfpb, &fNumNodes)) goto done;	
	
	if (version>1)
	{
		if (err = ReadMacValue(bfpb, &fNumEles)) goto done;
		if (err = ReadMacValue(bfpb, &bVelocitiesOnTriangles)) goto done;
	}
	
	
done:
	if(err)
	{
		TechError("TimeGridVelTri::Read(char* path)", " ", 0); 
	}
	return err;
}

Boolean TimeGridVelTri::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth)
{
	char uStr[32],sStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double topDepth, bottomDepth, depthAlpha, timeAlpha;
	VelocityRec pt1interp = {0.,0.}, pt2interp = {0.,0.}, pt3interp = {0.,0.};
	long index, amtOfDepthData = 0, triIndex;
	
	long ptIndex1=-1,ptIndex2=-1,ptIndex3=-1; 
	long pt1depthIndex1, pt1depthIndex2, pt2depthIndex1, pt2depthIndex2, pt3depthIndex1, pt3depthIndex2;
	InterpolationVal interpolationVal;
	
	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	if (!bVelocitiesOnTriangles)
	{
		interpolationVal = fGrid -> GetInterpolationValues(wp.p);
		if (interpolationVal.ptIndex1<0) return false;
	}
	else
	{
		LongPoint lp;
		TDagTree *dagTree = 0;
		dagTree = ((dynamic_cast<TTriGridVel*>(fGrid))) -> GetDagTree();
		if(!dagTree) return false;
		lp.h = wp.p.pLong;
		lp.v = wp.p.pLat;
		triIndex = dagTree -> WhatTriAmIIn(lp);
		interpolationVal.ptIndex1 = -1;
	}
	
	
	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		// this is only section that's different from ptcur
		ptIndex1 =  interpolationVal.ptIndex1;	
		ptIndex2 =  interpolationVal.ptIndex2;
		ptIndex3 =  interpolationVal.ptIndex3;
		if (fVerdatToNetCDFH)
		{
			ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
		}
		
		// probably want to extend to show the velocity at level that is being shown
		if (fDepthDataInfo) amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	 	if (amtOfDepthData>0)
	 	{
			GetDepthIndices(ptIndex1,arrowDepth,&pt1depthIndex1,&pt1depthIndex2);	
			GetDepthIndices(ptIndex2,arrowDepth,&pt2depthIndex1,&pt2depthIndex2);	
			GetDepthIndices(ptIndex3,arrowDepth,&pt3depthIndex1,&pt3depthIndex2);	
		}
		else
		{	// old version that didn't use fDepthDataInfo, must be 2D
	 		pt1depthIndex1 = ptIndex1;	pt1depthIndex2 = -1;
	 		pt2depthIndex1 = ptIndex2;	pt2depthIndex2 = -1;
	 		pt3depthIndex1 = ptIndex3;	pt3depthIndex2 = -1;
		} 
	}
	else
	{
		if (!bVelocitiesOnTriangles)
			return false;
	}
	
	// Check for constant current 
	if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime))
	{
		if (bVelocitiesOnTriangles)
		{
			pt1depthIndex1 = -1;
			pt2depthIndex1 = -1;
			pt3depthIndex1 = -1;
			if (triIndex > 0)
			{
				pt1interp.u = INDEXH(fStartData.dataHdl,triIndex).u; 
				pt1interp.v = INDEXH(fStartData.dataHdl,triIndex).v; 
			}
		}
		// Calculate the interpolated velocity at the point
		if (pt1depthIndex1!=-1)
		{
			if (pt1depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt1depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt1depthIndex2);
				depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
				pt1interp.u = depthAlpha*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex2).u));
				pt1interp.v = depthAlpha*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex2).v));
			}
			else
			{
				pt1interp.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).u); 
				pt1interp.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).v); 
			}
		}
		
		if (pt2depthIndex1!=-1)
		{
			if (pt2depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt2depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt2depthIndex2);
				depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
				pt2interp.u = depthAlpha*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex2).u));
				pt2interp.v = depthAlpha*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex2).v));
			}
			else
			{
				pt2interp.u = interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).u); 
				pt2interp.v = interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).v);
			}
		}
		
		if (pt3depthIndex1!=-1) 
		{
			if (pt3depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt3depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt3depthIndex2);
				depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
				pt3interp.u = depthAlpha*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex2).u));
				pt3interp.v = depthAlpha*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex2).v));
			}
			else
			{
				pt3interp.u = interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).u); 
				pt3interp.v = interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).v); 
			}
		}
		
		
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		if (bVelocitiesOnTriangles)
		{
			pt1depthIndex1 = -1;
			pt2depthIndex1 = -1;
			pt3depthIndex1 = -1;
			if (triIndex > 0)
			{
				pt1interp.u = INDEXH(fStartData.dataHdl,triIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,triIndex).u; 
				pt1interp.v = INDEXH(fStartData.dataHdl,triIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,triIndex).v; 
			}
		}
		
		// Calculate the interpolated velocity at the point
		if (pt1depthIndex1!=-1)
		{
			if (pt1depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt1depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt1depthIndex2);
				depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
				pt1interp.u = depthAlpha*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex2).u));
				pt1interp.v = depthAlpha*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex2).v));
			}
			else
			{
				pt1interp.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).u); 
				pt1interp.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).v); 
			}
		}
		
		if (pt2depthIndex1!=-1)
		{
			if (pt2depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt2depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt2depthIndex2);
				depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
				pt2interp.u = depthAlpha*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex2).u));
				pt2interp.v = depthAlpha*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex2).v));
			}
			else
			{
				pt2interp.u = interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).u); 
				pt2interp.v = interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).v); 
			}
		}
		
		if (pt3depthIndex1!=-1) 
		{
			if (pt3depthIndex2!=-1)
			{
				topDepth = INDEXH(fDepthsH,pt3depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt3depthIndex2);
				depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
				pt3interp.u = depthAlpha*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex2).u));
				pt3interp.v = depthAlpha*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex2).v));
			}
			else
			{
				pt3interp.u = interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).u); 
				pt3interp.v = interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).v); 
			}
		}
		
		
		
	}
	
	velocity.u = pt1interp.u + pt2interp.u + pt3interp.u; 
	velocity.v = pt1interp.v + pt2interp.v + pt3interp.v; 
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = lengthU * fVar.fileScaleFactor;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	
	if (interpolationVal.ptIndex1 >= 0 && ptIndex1>=0 && ptIndex2>=0 && ptIndex3>=0)
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices to triangle vertices : [%ld, %ld, %ld]",
			fVar.userName, uStr, sStr, ptIndex1, ptIndex2, ptIndex3);
		//sprintf(diagnosticStr, " [grid: unscaled: %s m/s, scaled: %s m/s], file indices to triangle vertices : [%ld, %ld, %ld]",
				//uStr, sStr, ptIndex1, ptIndex2, ptIndex3);
	}
	else
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
			fVar.userName, uStr, sStr);
		//sprintf(diagnosticStr, " [grid: unscaled: %s m/s, scaled: %s m/s]",
				//uStr, sStr);
	}
	return true;
}

void TimeGridVelTri::Draw(Rect r, WorldRect view,double refScale,double arrowScale,
						  double arrowDepth,Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor) 
{	// will need to update once triangle format is set
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	double timeAlpha,depthAlpha;
	float topDepth,bottomDepth;
	Point p;
	Rect c;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	OSErr err = 0;
	char errmsg[256];
	long amtOfDepthData = 0;
	Boolean overrideDrawArrows = false;
	
	//RGBForeColor(&colors[PURPLE]);
	
	if(fDepthDataInfo) amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	
	if(fGrid && (bDrawArrows || bDrawGrid))
	{
		Boolean overrideDrawArrows = FALSE;
		fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,arrowDepth,overrideDrawArrows,bDrawGrid,arrowColor);
		if(bDrawArrows && bVelocitiesOnTriangles == false)
		{ // we have to draw the arrows
			RGBForeColor(&arrowColor);
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;	// don't need 3D stuff to draw here
			
//			err = this -> SetInterval(errmsg);	// minus AH 07/17/2012
			//err = this -> SetInterval(errmsg, model->GetStartTime(), model->GetModelTime()); // AH 07/17/2012
			err = this -> SetInterval(errmsg, model->GetModelTime()); 
			
			if(err) return;
			
//			loaded = this -> CheckInterval(timeDataInterval);	// minus AH 07/17/2012
			//loaded = this -> CheckInterval(timeDataInterval, model->GetStartTime(), model->GetModelTime());	// AH 07/17/2012
			loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	
			
			if(!loaded) return;
			
			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			// Check for time varying current 
			if(GetNumTimesInFile()>1 || GetNumFiles()>1)
				//if(GetNumTimesInFile()>1)
			{
				// Calculate the time weight factor
				if (GetNumFiles()>1 && fOverLap)
					startTime = fOverLapStartTime + fTimeShift;
				else
					startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationInTime)
				{
					timeAlpha = 1;
				}
				else
				{	//return false;
					endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
					timeAlpha = (endTime - time)/(double)(endTime - startTime);
				}
				//endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
			
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				//long ptIndex = INDEXH(fVerdatToNetCDFH,i);
				long index = i;
				if (fVerdatToNetCDFH) index = INDEXH(fVerdatToNetCDFH,i);
				//long ptIndex = (*fDepthDataInfo)[index].indexToDepthData;	// not used ?
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;
				long depthIndex1,depthIndex2;	// default to -1?, eventually use in surface velocity case
				
				//GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				//amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	 			if (amtOfDepthData>0)
				{
					GetDepthIndices(index,arrowDepth,&depthIndex1,&depthIndex2);
				}
				else
				{	// for old SAV files without fDepthDataInfo
					depthIndex1 = index;
					depthIndex2 = -1;
				}
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth
				
				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					topDepth = INDEXH(fDepthsH,depthIndex1);
					bottomDepth = INDEXH(fDepthsH,depthIndex2);
					depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
				}
				
				
				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha==1)
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						//velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
						//velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
						velocity.u = INDEXH(fStartData.dataHdl,depthIndex1).u;
						velocity.v = INDEXH(fStartData.dataHdl,depthIndex1).v;
					}
					else 	// below surface velocity
					{
						velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).u;
						velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).v;
					}
				}
				else // time varying current
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
						//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v;
					}
					else	// below surface velocity
					{
						velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u);
						velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).u);
						velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v);
						velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).v);
					}
				}
				if ((velocity.u != 0 || velocity.v != 0))
				{
					float inchesX = (velocity.u * refScale * fVar.fileScaleFactor) / arrowScale;
					float inchesY = (velocity.v * refScale * fVar.fileScaleFactor) / arrowScale;
					short pixX = inchesX * PixelsPerInchCurrent();
					short pixY = inchesY * PixelsPerInchCurrent();
					p2.h = p.h + pixX;
					p2.v = p.v - pixY;
					MyMoveTo(p.h, p.v);
					MyLineTo(p2.h, p2.v);
					MyDrawArrow(p.h,p.v,p2.h,p2.v);
				}
			}
		}
		else if (bDrawArrows && bVelocitiesOnTriangles)
		{ // we have to draw the arrows
			short row, col, pixX, pixY;
			float inchesX, inchesY;
			Point p, p2;
			Rect c;
			WorldPoint wp;
			VelocityRec velocity;
			LongPoint wp1,wp2,wp3;
			Boolean offQuickDrawPlane = false;
			long numVertices,i,numTri;
			LongPointHdl ptsHdl = 0;
			TopologyHdl topH = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;	// don't need 3D stuff to draw here
			RGBForeColor(&arrowColor);
		
			err = this -> SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
			
			if(err) return;
			
			loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
			
			if(!loaded) return;
			
			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			topH = triGrid -> GetTopologyHdl();
			if (topH)
				numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);
			else 
				numTri = 0;
			
			// Check for time varying current 
			if(GetNumTimesInFile()>1 || GetNumFiles()>1)
				//if(GetNumTimesInFile()>1)
			{
				// Calculate the time weight factor
				if (GetNumFiles()>1 && fOverLap)
					startTime = fOverLapStartTime + fTimeShift;
				else
					startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationInTime)
				{
					timeAlpha = 1;
				}
				else
				{	//return false;
					endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
					timeAlpha = (endTime - time)/(double)(endTime - startTime);
				}
				//endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
			
			//for(i = 0; i < numVertices; i++)
			for(i = 0; i < numTri; i++)
			{
			 	// get the value at each vertex and draw an arrow
				//LongPoint pt = INDEXH(ptsHdl,i);
				//long index = INDEXH(fVerdatToNetCDFH,i);
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;
				long depthIndex1,depthIndex2;	// default to -1?, eventually use in surface velocity case
				
				//GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				//amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	 			if (amtOfDepthData>0 && !bVelocitiesOnTriangles)	// for now, will have to figure out how depth data is handled
				{
					//GetDepthIndices(index,fVar.arrowDepth,&depthIndex1,&depthIndex2);
					GetDepthIndices(i,arrowDepth,&depthIndex1,&depthIndex2);
				}
				else
				{	// for old SAV files without fDepthDataInfo
					//depthIndex1 = index;
					depthIndex1 = i;
					depthIndex2 = -1;
				}
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth
				
				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					topDepth = INDEXH(fDepthsH,depthIndex1);
					bottomDepth = INDEXH(fDepthsH,depthIndex2);
					depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
				}
				
				wp1 = (*ptsHdl)[(*topH)[i].vertex1];
				wp2 = (*ptsHdl)[(*topH)[i].vertex2];
				wp3 = (*ptsHdl)[(*topH)[i].vertex3];
				
				wp.pLong = (wp1.h+wp2.h+wp3.h)/3;
				wp.pLat = (wp1.v+wp2.v+wp3.v)/3;
				//velocity = GetPatValue(wp);
				
				//wp.pLat = pt.v;
				//wp.pLong = pt.h;
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha==1)
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						velocity.u = INDEXH(fStartData.dataHdl,i).u;
						velocity.v = INDEXH(fStartData.dataHdl,i).v;
					}
					else 	// below surface velocity
					{
						velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).u;
						velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).v;
					}
				}
				else // time varying current
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,i).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,i).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,i).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,i).v;
					}
					else	// below surface velocity
					{
						velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u);
						velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).u);
						velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v);
						velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).v);
					}
				}
				if ((velocity.u != 0 || velocity.v != 0))
				{
					inchesX = (velocity.u * refScale * fVar.fileScaleFactor) / arrowScale;
					inchesY = (velocity.v * refScale * fVar.fileScaleFactor) / arrowScale;
					pixX = inchesX * PixelsPerInchCurrent();
					pixY = inchesY * PixelsPerInchCurrent();
					p2.h = p.h + pixX;
					p2.v = p.v - pixY;
					MyMoveTo(p.h, p.v);
					MyLineTo(p2.h, p2.v);
					MyDrawArrow(p.h,p.v,p2.h,p2.v);
				}
			}
		}
	}
	//if (bShowDepthContours && fVar.gridType!=TWO_D) ((TTriGridVel*)fGrid)->DrawDepthContours(r,view,bShowDepthContourLabels);
	
	RGBForeColor(&colors[BLACK]);
}
OSErr TimeGridVelTri::ReadTopology(char* path)
{
	// import NetCDF triangle info so don't have to regenerate
	// this is same as curvilinear mover so may want to combine later
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
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0, boundaryPts=0;
	
	errmsg[0]=0;
	
	if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("TimeGridVelTri::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	
	// No header
	// start with transformation array and vertices
	MySpinCursor(); // JLM 8/4/99
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	if(IsTransposeArrayHeaderLine(s,&numPts)) // 
	{
		if (err = ReadTransposeArray(f,&line,&fVerdatToNetCDFH,numPts,errmsg)) 
		{strcpy(errmsg,"Error in ReadTransposeArray"); goto done;}
	}
	else 
	//{err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}
	{
		//if (!bVelocitiesOnTriangles) {err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}
		//else line--;
		line--;
	}
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
	
	if(IsBoundaryPointsHeaderLine(s,&numBoundaryPts)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundaryPts>0)
			err = ReadBoundaryPts(f,&line,&boundaryPts,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary points header line");
		//goto done;
		// not always needed ? probably always needed for curvilinear
	}
	MySpinCursor(); // JLM 8/4/99
	
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
	// code goes here, do we want to store grid boundary and land/water information?
	// check if bVelocitiesOnTriangles and boundaryPts
	/*if (waterBoundaries && boundarySegs)
	{
		//PtCurMap *map = CreateAndInitPtCurMap(fVar.userName,bounds); // the map bounds are the same as the grid bounds
		PtCurMap *map = CreateAndInitPtCurMap("Extended Topology",bounds); // the map bounds are the same as the grid bounds
		if (!map) {strcpy(errmsg,"Error creating ptcur map"); goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundarySegs);	
		map->SetWaterBoundaries(waterBoundaries);
		//if (bVelocitiesOnTriangles && boundaryPts) map->SetBoundaryPoints(boundaryPts);	
		if (boundaryPts) map->SetBoundaryPoints(boundaryPts);	
		
		*newMap = map;
	}	
	else*/	
	{
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts = 0;}
	}
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TimeGridVelTri::ReadTopology()","new TTriGridVel" ,err);
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
			strcpy(errmsg,"An error occurred in TimeGridVelTri::ReadTopology");
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
		/*if (*newMap) 
		{
			(*newMap)->Dispose();
			delete *newMap;
			*newMap=0;
		}*/
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts = 0;}
	}
	return err;
}

OSErr TimeGridVelTri::ExportTopology(char* path)
{
	// export NetCDF triangle info so don't have to regenerate each time
	// same as curvilinear so may want to combine at some point
	OSErr err = 0;
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0, nBoundaryPts;
	long i, n=0, v1,v2,v3,n1,n2,n3;
	double x,y,z=0;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	FLOATH depthsH=0;
	DAGHdl treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0, boundaryPointsH = 0;
	BFPB bfpb;
	//PtCurMap *map = GetPtCurMap();
	
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
	depthsH = ((TTriGridVel*)triGrid)->GetDepths();
	if(!ptsH || !topH || !treeH) 
	{
		printError("There is no topology to export");
		return -1;
	}
	
	//if (moverMap->IAm(TYPE_PTCURMAP))
	/*if (map)
	{
		//boundaryTypeH = ((PtCurMap*)moverMap)->GetWaterBoundaries();
		//boundarySegmentsH = ((PtCurMap*)moverMap)->GetBoundarySegs();
		boundaryTypeH = map->GetWaterBoundaries();
		boundarySegmentsH = map->GetBoundarySegs();
		if (!boundaryTypeH || !boundarySegmentsH) {printError("No map info to export"); err=-1; goto done;}
		//if (bVelocitiesOnTriangles) 
		//{
			//boundaryPointsH = map->GetBoundaryPoints();
			//if (!boundaryPointsH) {printError("No map info to export"); err=-1; goto done;}
		//}
		boundaryPointsH = map->GetBoundaryPoints();
	}*/
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	
	// Write out values
	if (fVerdatToNetCDFH) n = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(long);
	else n = 0;
	//else {printError("There is no transpose array"); err = -1; goto done;}
	//else 
		//{if (!bVelocitiesOnTriangles) {printError("There is no transpose array"); err = -1; goto done;}}
	//if (!bVelocitiesOnTriangles)
	if (n>0)
	{
		sprintf(hdrStr,"TransposeArray\t%ld\n",n);	
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<n;i++)
		{	
			sprintf(topoStr,"%ld\n",(*fVerdatToNetCDFH)[i]);
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}
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
		if (depthsH) 
		{
			z = (*depthsH)[i];
			sprintf(topoStr,"%lf\t%lf\t%lf\n",x,y,z);
		}
		else
			sprintf(topoStr,"%lf\t%lf\n",x,y);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
	/*if (boundarySegmentsH) 
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
	nBoundaryPts = 0;
	if (boundaryPointsH) 
	{
		nBoundaryPts = _GetHandleSize((Handle)boundaryPointsH)/sizeof(long);	
		sprintf(hdrStr,"BoundaryPoints\t%ld\n",nBoundaryPts);	// total boundary points
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundaryPts;i++)
		{	
			sprintf(topoStr,"%ld\n",(*boundaryPointsH)[i]);	
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}*/
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
		sprintf(topoStr, "%ld\t%ld\t%ld\t%ld\t%ld\t%ld\n",
				v1, v2, v3, n1, n2, n3);
		
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

TimeGridCurRect::TimeGridCurRect () : TimeGridVel()
{
	fTimeDataHdl = 0;
	
	fUserUnits = kUndefined;
}

void TimeGridCurRect::Dispose ()
{
	if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
	
	TimeGridVel::Dispose ();
}

#define TimeGridCurRectREADWRITEVERSION 1 

OSErr TimeGridCurRect::Write (BFPB *bfpb)
{
	char c;
	long i, version = TimeGridCurRectREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long amtTimeData = GetNumTimesInFile();
	long numPoints, numFiles;
	float val;
	PtCurTimeData timeData;
	OSErr err = 0;
	
	if (err = TimeGridVel::Write (bfpb)) return err;
	
	StartReadWriteSequence("TimeGridCurRect::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	
	if (err = WriteMacValue(bfpb, amtTimeData)) goto done;
	for (i=0;i<amtTimeData;i++)
	{
		timeData = INDEXH(fTimeDataHdl,i);
		if (err = WriteMacValue(bfpb, timeData.fileOffsetToStartOfData)) goto done;
		if (err = WriteMacValue(bfpb, timeData.lengthOfData)) goto done;
		if (err = WriteMacValue(bfpb, timeData.time)) goto done;
	}
	
done:
	if(err)
		TechError("TimeGridCurRect::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr TimeGridCurRect::Read(BFPB *bfpb)
{
	char c, msg[256];
	long i, version, amtTimeData;
	ClassID id;
	float val;
	PtCurTimeData timeData;
	OSErr err = 0;
	
	if (err = TimeGridVel::Read(bfpb)) return err;
	
	StartReadWriteSequence("TimeGridCurRect::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TimeGridCurRect::Read()", "id != TYPE_TIMEGRIDCURRECT", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > TimeGridCurRectREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////

	if (err = ReadMacValue(bfpb, &amtTimeData)) goto done;	
	fTimeDataHdl = (PtCurTimeDataHdl)_NewHandleClear(sizeof(PtCurTimeData)*amtTimeData);
	if(!fTimeDataHdl)
	{TechError("TimeGridCurRect::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < amtTimeData ; i++) {
		if (err = ReadMacValue(bfpb, &timeData.fileOffsetToStartOfData)) goto done;
		if (err = ReadMacValue(bfpb, &timeData.lengthOfData)) goto done;
		if (err = ReadMacValue(bfpb, &timeData.time)) goto done;
		INDEXH(fTimeDataHdl, i) = timeData;
	}
		
done:
	if(err)
	{
		TechError("TimeGridCurRect::Read(char* path)", " ", 0); 
		if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
	}
	return err;
}

Boolean TimeGridCurRect::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth)
{
	char uStr[32],sStr[32];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	long timeDataInterval,numTimesInFile = dynamic_cast<TimeGridCurRect *>(this)->GetNumTimesInFile();
	Boolean intervalLoaded = dynamic_cast<TimeGridCurRect *>(this) -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
	
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	long index;
	
	if (!intervalLoaded || numTimesInFile == 0) return false; // no data don't try to show velocity
	if(numTimesInFile>1)
		//&& loaded && !err)
	{
		if (err = this->GetStartTime(&startTime)) return false;	// should this stop us from showing any velocity?
		if (err = this->GetEndTime(&endTime)) return false;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);	
	}
	//if (loaded && !err)
	{
		index = this->GetVelocityIndex(wp.p);	// need alternative for curvilinear
		
		if (index >= 0)
		{
			// Check for constant current 
			if(numTimesInFile==1)
			{
				velocity.u = this->GetStartUVelocity(index);
				velocity.v = this->GetStartVVelocity(index);
			}
			else // time varying current
			{
				velocity.u = timeAlpha*this->GetStartUVelocity(index) + (1-timeAlpha)*this->GetEndUVelocity(index);
				velocity.v = timeAlpha*this->GetStartVVelocity(index) + (1-timeAlpha)*this->GetEndVVelocity(index);
			}
		}
	}
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	//lengthS = this->fVar.curScale * lengthU;
	lengthS = lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
		fVar.userName, uStr, sStr);
	//sprintf(diagnosticStr, " [unscaled: %s m/s, scaled: %s m/s]",
			//uStr, sStr);
	
	return true;
}

void TimeGridCurRect::Draw(Rect r, WorldRect view,double refScale,double arrowScale,
						  double arrowDepth,Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor) 
{
	short row, col, pixX, pixY;
	long dLong, dLat, index, timeDataInterval;
	float inchesX, inchesY;
	double timeAlpha;
	Seconds startTime, endTime, time = model->GetModelTime();
	Point p, p2;
	WorldPoint wp;
	WorldRect boundsRect, bounds;
	VelocityRec velocity;
	Rect c, newCATSgridRect = {0, 0, fNumRows - 1, fNumCols - 1}; // fNumRows, fNumCols members of GridCurMover
	Boolean offQuickDrawPlane = false, loaded;
	char errmsg[256];
	OSErr err = 0;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	
	
	if (!bDrawArrows && !bDrawGrid) return;
	
	//p = GetQuickDrawPt(refP.pLong, refP.pLat, &r, &offQuickDrawPlane);
	bounds = rectGrid->GetBounds();
	
	// draw the reference point
	//RGBForeColor(&colors[BLUE]);
	//MySetRect(&c, p.h - 2, p.v - 2, p.h + 2, p.v + 2);
	//PaintRect(&c);
	RGBForeColor(&colors[BLACK]);
	
	// need to get the bounds from the grid
	dLong = (WRectWidth(bounds) / fNumCols) / 2;
	dLat = (WRectHeight(bounds) / fNumRows) / 2;
	//RGBForeColor(&colors[PURPLE]);
	RGBForeColor(&arrowColor);
	
	boundsRect = bounds;
	InsetWRect (&boundsRect, dLong, dLat);
	
	if (bDrawArrows)
	{
		err = this -> SetInterval(errmsg, model->GetModelTime()); 
		
		if(err && !bDrawGrid) return;	// want to show grid even if there's no current data
		
		loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	
		
		if(!loaded && !bDrawGrid) return;
		
		// Check for time varying current 
		if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
		{
			// Calculate the time weight factor
			if (GetNumFiles()>1 && fOverLap)
				startTime = fOverLapStartTime;
			else
				startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
			//startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
			endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
			timeAlpha = (endTime - time)/(double)(endTime - startTime);
		}
	}	
	
	for (row = 0 ; row < fNumRows ; row++)
		for (col = 0 ; col < fNumCols ; col++) {
			SetPt(&p, col, row);
			wp = ScreenToWorldPoint(p, newCATSgridRect, boundsRect);
			velocity.u = velocity.v = 0.;
			if (loaded && !err)
			{
				index = dynamic_cast<TimeGridCurRect *>(this)->GetVelocityIndex(wp);
				if (bDrawArrows && index >= 0)
				{
					// Check for constant current 
					if(GetNumTimesInFile()==1 && !(GetNumFiles()>1))
					{
						velocity.u = INDEXH(fStartData.dataHdl,index).u;
						velocity.v = INDEXH(fStartData.dataHdl,index).v;
					}
					else // time varying current
					{
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
					}
				}
			}
			p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
			MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
			
			if (bDrawGrid) PaintRect(&c);
			
			if (bDrawArrows && (velocity.u != 0 || velocity.v != 0))
			{
				inchesX = (velocity.u * refScale) / arrowScale;
				inchesY = (velocity.v * refScale) / arrowScale;
				pixX = inchesX * PixelsPerInchCurrent();
				pixY = inchesY * PixelsPerInchCurrent();
				p2.h = p.h + pixX;
				p2.v = p.v - pixY;
				MyMoveTo(p.h, p.v);
				MyLineTo(p2.h, p2.v);
				MyDrawArrow(p.h,p.v,p2.h,p2.v);
			}
		}
	
	RGBForeColor(&colors[BLACK]);
}
TimeGridCurTri::TimeGridCurTri () : TimeGridCurRect()
{
	memset(&fVar2,0,sizeof(fVar2));
	fVar2.arrowScale = 5;
	fVar2.arrowDepth = 0;
	fVar2.alongCurUncertainty = .5;
	fVar2.crossCurUncertainty = .25;
	//fVar.uncertMinimumInMPS = .05;
	fVar2.uncertMinimumInMPS = 0.0;
	fVar2.curScale = 1.0;
	fVar2.startTimeInHrs = 0.0;
	fVar2.durationInHrs = 24.0;
	fVar2.numLandPts = 0; // default that boundary velocities are given
	fVar2.maxNumDepths = 1; // 2D default
	fVar2.gridType = TWO_D; // 2D default
	fVar2.bLayerThickness = 0.; // FREESLIP default
	//
	// Override TCurrentMover defaults
	/*fDownCurUncertainty = -fVar2.alongCurUncertainty; 
	fUpCurUncertainty = fVar2.alongCurUncertainty; 	
	fRightCurUncertainty = fVar2.crossCurUncertainty;  
	fLeftCurUncertainty = -fVar2.crossCurUncertainty; 
	fDuration=fVar2.durationInHrs*3600.; //24 hrs as seconds 
	fUncertainStartTime = (long) (fVar2.startTimeInHrs*3600.);*/
	//
	
	fDepthsH = 0;
	fDepthDataInfo = 0;
	
	//SetClassName (name); // short file name
}
void TimeGridCurTri::Dispose ()
{
	if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
	if(fDepthDataInfo) {DisposeHandle((Handle)fDepthDataInfo); fDepthDataInfo=0;}
	
	TimeGridCurRect::Dispose ();
}
#define TimeGridCurTriREADWRITEVERSION 1 //JLM

OSErr TimeGridCurTri::Write (BFPB *bfpb)
{
	char c;
	long i, version = TimeGridCurTriREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	VelocityRec velocity;
	long 	numDepths = GetNumDepths();
	long numPoints, numFiles;
	float val;
	DepthDataInfo depthData;
	OSErr err = 0;
	
	if (err = TimeGridCurRect::Write (bfpb)) return err;
	
	StartReadWriteSequence("TimeGridCurTri::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	/*if (err = WriteMacValue(bfpb, fVar.alongCurUncertainty)) return err;
	if (err = WriteMacValue(bfpb, fVar.crossCurUncertainty)) return err;
	if (err = WriteMacValue(bfpb, fVar.uncertMinimumInMPS)) return err;
	if (err = WriteMacValue(bfpb, fVar.curScale)) return err;
	if (err = WriteMacValue(bfpb, fVar.startTimeInHrs)) return err;
	if (err = WriteMacValue(bfpb, fVar.durationInHrs)) return err;
	//
	if (err = WriteMacValue(bfpb, fVar.numLandPts)) return err;
	if (err = WriteMacValue(bfpb, fVar.maxNumDepths)) return err;
	if (err = WriteMacValue(bfpb, fVar.gridType)) return err;
	if (err = WriteMacValue(bfpb, fVar.bLayerThickness)) return err;
	//
	if (err = WriteMacValue(bfpb, fVar.bShowGrid)) return err;
	if (err = WriteMacValue(bfpb, fVar.bShowArrows)) return err;
	if (err = WriteMacValue(bfpb, fVar.bUncertaintyPointOpen)) return err;
	if (err = WriteMacValue(bfpb, fVar.arrowScale)) return err;
	if (err = WriteMacValue(bfpb, fVar.arrowDepth)) return err;*/
	
	if (err = WriteMacValue(bfpb, numDepths)) goto done;
	for (i=0;i<numDepths;i++)
	{
		val = INDEXH(fDepthsH,i);
		if (err = WriteMacValue(bfpb, val)) goto done;
	}
	
	numPoints = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i = 0 ; i < numPoints ; i++) {
		depthData = INDEXH(fDepthDataInfo,i);
		if (err = WriteMacValue(bfpb, depthData.totalDepth)) goto done;
		if (err = WriteMacValue(bfpb, depthData.indexToDepthData)) goto done;
		if (err = WriteMacValue(bfpb, depthData.numDepths)) goto done;
	}
	
	
done:
	if(err)
		TechError("TimeGridCurTri::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr TimeGridCurTri::Read(BFPB *bfpb)
{
	char c, msg[256], fileName[64];
	long i, version, numDepths, numPoints;
	ClassID id;
	float val;
	DepthDataInfo depthData;
	OSErr err = 0;
	
	if (err = TimeGridCurRect::Read(bfpb)) return err;
	
	StartReadWriteSequence("TimeGridCurTri::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TimeGridCurTri::Read()", "id != TYPE_TIMEGRIDCURTRI", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != TimeGridCurTriREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	/*if (err = ReadMacValue(bfpb, &fVar.alongCurUncertainty)) return err;
	if (err = ReadMacValue(bfpb, &fVar.crossCurUncertainty)) return err;
	if (err = ReadMacValue(bfpb, &fVar.uncertMinimumInMPS)) return err;
	if (err = ReadMacValue(bfpb, &fVar.curScale)) return err;
	if (err = ReadMacValue(bfpb, &fVar.startTimeInHrs)) return err;
	if (err = ReadMacValue(bfpb, &fVar.durationInHrs)) return err;
	//
	if (err = ReadMacValue(bfpb, &fVar.numLandPts)) return err;
	if (err = ReadMacValue(bfpb, &fVar.maxNumDepths)) return err;
	if (err = ReadMacValue(bfpb, &fVar.gridType)) return err;
	if (err = ReadMacValue(bfpb, &fVar.bLayerThickness)) return err;
	//
	if (err = ReadMacValue(bfpb, &fVar.bShowGrid)) return err;
	if (err = ReadMacValue(bfpb, &fVar.bShowArrows)) return err;
	if (err = ReadMacValue(bfpb, &fVar.bUncertaintyPointOpen)) return err;
	if (err = ReadMacValue(bfpb, &fVar.arrowScale)) return err;
	if (err = ReadMacValue(bfpb, &fVar.arrowDepth)) return err;*/
	
	
	if (err = ReadMacValue(bfpb, &numDepths)) goto done;	
	if (numDepths>0)
	{
		fDepthsH = (FLOATH)_NewHandleClear(sizeof(float)*numDepths);
		if (!fDepthsH)
		{ TechError("TimeGridCurTri::Read()", "_NewHandleClear()", 0); goto done; }
		
		for (i = 0 ; i < numDepths ; i++) {
			if (err = ReadMacValue(bfpb, &val)) goto done;
			INDEXH(fDepthsH, i) = val;
		}
	}
	
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fDepthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(DepthDataInfo)*numPoints);
	if(!fDepthDataInfo)
	{TechError("TimeGridCurTri::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &depthData.totalDepth)) goto done;
		if (err = ReadMacValue(bfpb, &depthData.indexToDepthData)) goto done;
		if (err = ReadMacValue(bfpb, &depthData.numDepths)) goto done;
		INDEXH(fDepthDataInfo, i) = depthData;
	}
	
	
done:
	if(err)
	{
		TechError("TimeGridCurTri::Read(char* path)", " ", 0); 
		if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
		if(fDepthDataInfo) {DisposeHandle((Handle)fDepthDataInfo); fDepthDataInfo=0;}
	}
	return err;
}

void TimeGridCurTri::Draw(Rect r, WorldRect view,double refScale,double arrowScale,
						  double arrowDepth,Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor) 
{
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	double timeAlpha,depthAlpha;
	float topDepth,bottomDepth;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	OSErr err = 0;
	char errmsg[256];
	
	if(fGrid && (bDrawArrows || bDrawGrid))
	{
		Boolean overrideDrawArrows = FALSE;
		fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,arrowDepth,bDrawArrows,bDrawGrid,arrowColor);
		if(bDrawArrows)
		{ // we have to draw the arrows
			RGBForeColor(&arrowColor);
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;	// don't think need 3D here
			
			err = this -> SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
			
			if(err) return;
			
			loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
			
			if(!loaded) return;
			
			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			// Check for time varying current 
			if(GetNumTimesInFile()>1 || GetNumFiles()>1)
			{
				// Calculate the time weight factor
				if (GetNumFiles()>1 && fOverLap)
					startTime = fOverLapStartTime;
				else
					startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
				endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
				timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
			
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				//long ptIndex = (*fDepthDataInfo)[i].indexToDepthData;
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;
				long depthIndex1,depthIndex2;	// default to -1?
				
				GetDepthIndices(i,arrowDepth,&depthIndex1,&depthIndex2);
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth
				
				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					topDepth = INDEXH(fDepthsH,depthIndex1);
					bottomDepth = INDEXH(fDepthsH,depthIndex2);
					depthAlpha = (bottomDepth - arrowDepth)/(double)(bottomDepth - topDepth);
				}
				
				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				//p.h = SameDifferenceX(wp.pLong);
				//p.v = (r.bottom + r.top) - SameDifferenceY(wp.pLat);
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if(GetNumTimesInFile()==1 && !(GetNumFiles()>1))
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						//velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
						//velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
						velocity.u = INDEXH(fStartData.dataHdl,depthIndex1).u;
						velocity.v = INDEXH(fStartData.dataHdl,depthIndex1).v;
					}
					else 	// below surface velocity
					{
						velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).u;
						velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).v;
					}
				}
				else // time varying current
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
						//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1/*ptIndex*/).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1/*ptIndex*/).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v;
					}
					else	// below surface velocity
					{
						velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u);
						velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).u);
						velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v);
						velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).v);
					}
				}
				if ((velocity.u != 0 || velocity.v != 0))
				{
					float inchesX = (velocity.u * refScale) / arrowScale;
					float inchesY = (velocity.v * refScale) / arrowScale;
					short pixX = inchesX * PixelsPerInchCurrent();
					short pixY = inchesY * PixelsPerInchCurrent();
					p2.h = p.h + pixX;
					p2.v = p.v - pixY;
					MyMoveTo(p.h, p.v);
					MyLineTo(p2.h, p2.v);
					MyDrawArrow(p.h,p.v,p2.h,p2.v);
				}
			}
		}
		RGBForeColor(&colors[BLACK]);
	}
}

Boolean TimeGridCurTri::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth)
{
	char uStr[32],sStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	
	long ptIndex1,ptIndex2,ptIndex3; 
	InterpolationVal interpolationVal;
	
	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	// at this point this is only showing the surface velocity values
	interpolationVal = fGrid -> GetInterpolationValues(wp.p);
	
	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		ptIndex1 =  (*fDepthDataInfo)[interpolationVal.ptIndex1].indexToDepthData;
		ptIndex2 =  (*fDepthDataInfo)[interpolationVal.ptIndex2].indexToDepthData;
		ptIndex3 =  (*fDepthDataInfo)[interpolationVal.ptIndex3].indexToDepthData;
	}
	
	
	// Check for constant current 
	if(GetNumTimesInFile()==1 && !(GetNumFiles()>1))
	{
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			velocity.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).u)
			+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).u)
			+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).u );
			velocity.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).v)
			+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).v)
			+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).v);
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime;
		else
			startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			velocity.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).u)
			+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).u)
			+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).u);
			velocity.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).v)
			+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).v)
			+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).v);
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	//velocity.u *= fVar.curScale; 
	//velocity.v *= fVar.curScale; 
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	//lengthS = this->fVar.curScale * lengthU;
	lengthS = this->fVar.fileScaleFactor * lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	//sprintf(diagnosticStr, " [unscaled: %s m/s, scaled: %s m/s]",
			//uStr, sStr);
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
			fVar.userName, uStr, sStr);
	
	return true;
}

