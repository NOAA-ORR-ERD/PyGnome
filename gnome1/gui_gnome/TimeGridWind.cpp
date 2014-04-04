// This is a cross between TWindMover and NetCDFMover, possibly reorganize to build off one or the other
// The uncertainty is from the wind, the reading, storing, accessing, displaying data is from NetCDFMover


#include "Cross.h"
#include "NetCDFMover.h"
#include "netcdf.h"
#include "TWindMover.h"
#include "GridCurMover.h"
#include "GridWindMover.h"
#include "Outils.h"
#include "DagTreeIO.h"
#include "TimeGridVel.h"

TimeGridWindRect::TimeGridWindRect() : TimeGridVel()
{
	//if(!name || !name[0]) this->SetClassName("Gridded Wind");
	//else 	SetClassName (name); // short file name
	
	//fUserUnits = kMetersPerSec;	
	//fWindScale = 1.;
	//fArrowScale = 10.;
	//fFillValue = -1e+34;

}

/*void TimeGridWindRect::Dispose()
{
	TimeGridVel::Dispose ();
}*/
	 	 

#define TimeGridWindRectREADWRITEVERSION 1 //JLM	5/3/10

OSErr TimeGridWindRect::Write(BFPB *bfpb)
{
	long i, version = TimeGridWindRectREADWRITEVERSION;
	ClassID id = GetClassID ();
	Seconds time;
	OSErr err = 0;
	
	if (err = TimeGridVel::Write(bfpb)) return err;
	
	StartReadWriteSequence("TimeGridWindRect::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	// anything here?	
	
done:
	if(err)
		TechError("TimeGridWindRect::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr TimeGridWindRect::Read(BFPB *bfpb)
{
	char c, msg[256], fileName[256], newFileName[64];
	long i, version, numTimes, numPoints;
	ClassID id;
	float val;
	Seconds time;
	Boolean bPathIsValid = true;
	OSErr err = 0;
	
	if (err = TimeGridVel::Read(bfpb)) return err;
	
	StartReadWriteSequence("TimeGridWindRect::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TimeGridWindRect::Read()", "id != TYPE_TIMEGRIDWINDRECT", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > TimeGridWindRectREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	
	// anything here?	
	
done:
	if(err)
	{
		TechError("TimeGridWindRect::Read(char* path)", " ", 0); 
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
	}
	return err;
}


/*Boolean TimeGridWindRect::DrawingDependsOnTime(void)
{
	Boolean depends = bShowArrows;
	// if this is a constant wind, we can say "no"
	if(this->GetNumTimesInFile()==1 && !(GetNumFiles()>1)) depends = false;
	return depends;
}*/

Boolean TimeGridWindRect::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr,double arrowDepth)
{
	char uStr[32],sStr[32],errmsg[256];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	long index;
	LongPoint indices;
	
	if(GetNumTimesInFile()>1)
	{
		if (err = this->GetStartTime(&startTime)) return false;	// should this stop us from showing any velocity?
		if (err = this->GetEndTime(&endTime)) /*return false;*/
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
	//lengthS = this->fWindScale * lengthU;
	lengthS = lengthU * fVar.fileScaleFactor;
	if (lengthS > 1000000 || this->fVar.fileScaleFactor==0) return true;	// if bad data in file causes a crash
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
			fVar.userName, uStr, sStr, fNumRows-indices.v-1, indices.h);
	//sprintf(diagnosticStr, " [unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
			//uStr, sStr, fNumRows-indices.v-1, indices.h);
	
	return true;
}

void TimeGridWindRect::Draw(Rect r, WorldRect view, double refScale, double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor)
{	// Use this for regular grid
	short row, col, pixX, pixY;
	long dLong, dLat, index, timeDataInterval;
	float inchesX, inchesY;
	double timeAlpha;
	Seconds startTime, endTime, time = model->GetModelTime();
	Point p, p2;
	WorldPoint wp;
	WorldRect boundsRect, bounds;
	VelocityRec velocity;
	Rect c, newGridRect = {0, 0, fNumRows - 1, fNumCols - 1}; // fNumRows, fNumCols members of TimeGridVel
	Boolean offQuickDrawPlane = false, loaded;
	char errmsg[256];
	OSErr err = 0;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	
	
	if (!bDrawArrows && !bDrawGrid) return;
	
	bounds = rectGrid->GetBounds();
	
	// need to get the bounds from the grid
	dLong = (WRectWidth(bounds) / fNumCols) / 2;
	dLat = (WRectHeight(bounds) / fNumRows) / 2;
	//RGBForeColor(&colors[PURPLE]);
	RGBForeColor(&arrowColor);
	
	boundsRect = bounds;
	InsetWRect (&boundsRect, dLong, dLat);
	
	if (bDrawArrows)
	{
		err = this -> SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
		
		if(err && !bDrawGrid) return;	// want to show grid even if there's no wind data
		
		loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	
		if(!loaded && !bDrawGrid) return;
		
		if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
		{
			// Calculate the time weight factor
			if (GetNumFiles()>1 && fOverLap)
				startTime = fOverLapStartTime + fTimeShift;
			else
				startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
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
	
	for (row = 0 ; row < fNumRows ; row++)
		for (col = 0 ; col < fNumCols ; col++) {
			
			SetPt(&p, col, row);
			wp = ScreenToWorldPoint(p, newGridRect, boundsRect);
			velocity.u = velocity.v = 0.;
			if (loaded && !err)
			{
				index = dynamic_cast<TimeGridWindRect *>(this)->GetVelocityIndex(wp);	
				
				if (bDrawArrows && index >= 0)
				{
					// Check for constant wind pattern 
					if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha==1)
					{
						velocity.u = INDEXH(fStartData.dataHdl,index).u;
						velocity.v = INDEXH(fStartData.dataHdl,index).v;
					}
					else // time varying wind
					{
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
					}
				}
			}
			
			p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
			MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
			
			if (bDrawGrid && bDrawArrows && (velocity.u != 0 || velocity.v != 0)) 
				PaintRect(&c);	// should check fill_value
			if (bDrawGrid && !bDrawArrows) 
				PaintRect(&c);	// should check fill_value
			
			if (bDrawArrows && (velocity.u != 0 || velocity.v != 0))
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
	
	RGBForeColor(&colors[BLACK]);
}


TimeGridWindCurv::TimeGridWindCurv () : TimeGridWindRect()
{
	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
}

/*void TimeGridWindCurv::Dispose ()
{
	if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
	if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	
	TimeGridWindRect::Dispose ();
}*/


#define TimeGridWindCurvREADWRITEVERSION 1 //JLM

OSErr TimeGridWindCurv::Write (BFPB *bfpb)
{
	long i, version = TimeGridWindCurvREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numPoints, index;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = TimeGridWindRect::Write (bfpb)) return err;
	
	StartReadWriteSequence("TimeGridWindCurv::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	
	numPoints = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(**fVerdatToNetCDFH);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i=0;i<numPoints;i++)
	{
		index = INDEXH(fVerdatToNetCDFH,i);
		if (err = WriteMacValue(bfpb, index)) goto done;
	}
	
	numPoints = _GetHandleSize((Handle)fVertexPtsH)/sizeof(**fVertexPtsH);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i=0;i<numPoints;i++)
	{
		vertex = INDEXH(fVertexPtsH,i);
		if (err = WriteMacValue(bfpb, vertex.pLat)) goto done;
		if (err = WriteMacValue(bfpb, vertex.pLong)) goto done;
	}
	
done:
	if(err)
		TechError("TimeGridWindCurv::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr TimeGridWindCurv::Read(BFPB *bfpb)	
{
	long i, version, index, numPoints;
	ClassID id;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = TimeGridWindRect::Read(bfpb)) return err;
	
	StartReadWriteSequence("TimeGridWindCurv::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TimeGridWindCurv::Read()", "id != TYPE_TIMEGRIDWINDCURV", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != TimeGridWindCurvREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVerdatToNetCDFH = (LONGH)_NewHandleClear(sizeof(long)*numPoints);	// for curvilinear
	if(!fVerdatToNetCDFH)
	{TechError("TimeGridWindCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &index)) goto done;
		INDEXH(fVerdatToNetCDFH, i) = index;
	}
	
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVertexPtsH = (WORLDPOINTFH)_NewHandleClear(sizeof(WorldPointF)*numPoints);	// for curvilinear
	if(!fVertexPtsH)
	{TechError("TimeGridWindCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &vertex.pLat)) goto done;
		if (err = ReadMacValue(bfpb, &vertex.pLong)) goto done;
		INDEXH(fVertexPtsH, i) = vertex;
	}
	
done:
	if(err)
	{
		TechError("TimeGridWindCurv::Read(char* path)", " ", 0); 
		if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
		if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	}
	return err;
}


Boolean TimeGridWindCurv::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth)
{	
	char uStr[32],sStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	long index;
	LongPoint indices;
	
	long ptIndex1,ptIndex2,ptIndex3; 
	InterpolationVal interpolationVal;
	
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
		if (index < 0) return 0;
		indices = this->GetVelocityIndices(wp.p);
	}
	
	// Check for constant current 
	if((GetNumTimesInFile()==1 /*&& !(GetNumFiles()>1)*/) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime)  || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime))
		//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			velocity.u = INDEXH(fStartData.dataHdl,index).u;
			velocity.v = INDEXH(fStartData.dataHdl,index).v;
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
		if (index >= 0) 
		{
			velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
		}
		else	// set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	//lengthS = this->fWindScale * lengthU;	// pass this in if there is a dialog scale factor
	lengthS = lengthU * fVar.fileScaleFactor;	
	if (lengthS > 1000000 || this->fVar.fileScaleFactor==0) return true;	// if bad data in file causes a crash
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	if (indices.h >= 0 && fNumRows-indices.v-1 >=0 && indices.h < fNumCols && fNumRows-indices.v-1 < fNumRows)
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
				fVar.userName, uStr, sStr, fNumRows-indices.v-1, indices.h);
		//sprintf(diagnosticStr, " [unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
				//uStr, sStr, fNumRows-indices.v-1, indices.h);
	}
	else
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
				fVar.userName, uStr, sStr);
		//sprintf(diagnosticStr, " [unscaled: %s m/s, scaled: %s m/s]",
				//uStr, sStr);
	}
	
	return true;
}

void TimeGridWindCurv::Draw(Rect r, WorldRect view, double refScale, double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor) 
{	// use for curvilinear
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	double timeAlpha;
	Point p;
	Rect c;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	OSErr err = 0;
	char errmsg[256];
	
	RGBForeColor(&arrowColor);
	
	if(bDrawArrows || bDrawGrid)
	{
		if (bDrawGrid) 	// make sure to draw grid even if don't draw arrows
		{
			((TTriGridVel*)fGrid)->DrawCurvGridPts(r,view);
			//return;
		}
		if (bDrawArrows)
		{ // we have to draw the arrows
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;
			
			err = this -> SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
			
			if(err) return;
			
			loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
			
			if(!loaded) return;
			
			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			// Check for time varying wind 
			if( (GetNumTimesInFile()>1 || GetNumFiles()>1 )&& loaded && !err)
			{
				// Calculate the time weight factor
				startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				//endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				//timeAlpha = (endTime - time)/(double)(endTime - startTime);
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
			
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				long ptIndex=-1,iIndex,jIndex;
				WorldPoint wp,wp2;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;				
				
				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				ptIndex = INDEXH(fVerdatToNetCDFH,i);
				iIndex = ptIndex/(fNumCols+1);
				jIndex = ptIndex%(fNumCols+1);
				if (iIndex>0 && jIndex<fNumCols)
					ptIndex = (iIndex-1)*(fNumCols)+jIndex;
				else
					ptIndex = -1;
				
				// for now draw arrow at midpoint of diagonal of gridbox
				// this will result in drawing some arrows more than once
				if (GetLatLonFromIndex(iIndex-1,jIndex+1,&wp2)!=-1)	// may want to get all four points and interpolate
				{
					wp.pLat = (wp.pLat + wp2.pLat)/2.;
					wp.pLong = (wp.pLong + wp2.pLong)/2.;
				}
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);	// should put velocities in center of grid box
				
				// Should check vs fFillValue
				// Check for constant wind 
				if(   ((  GetNumTimesInFile()==1 &&!(GetNumFiles()>1)  ) || timeAlpha == 1) && ptIndex!=-1)
				{
					velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
					velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
				}
				else if (ptIndex!=-1)// time varying wind
				{
					// need to rescale velocities for Navy case, store angle
					velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
					velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
				}
				if ((velocity.u != 0 || velocity.v != 0) && (velocity.u != fFillValue && velocity.v != fFillValue))
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
	if (bDrawGrid) fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,0.,false,true,arrowColor);
	
	RGBForeColor(&colors[BLACK]);
}


/*OSErr TimeGridWindCurv::ExportTopology(char* path)
{
	// export NetCDF curvilinear info so don't have to regenerate each time
	// move to NetCDFWindMover so Tri can use it too
	OSErr err = 0;
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0, nBoundaryPts;
	long i, n, v1,v2,v3,n1,n2,n3;
	double x,y;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	DAGHdl		treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0, boundaryPointsH = 0;	// should we bother with the map stuff? 
	BFPB bfpb;
	PtCurMap *map = GetPtCurMap();
	
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
	//if (moverMap->IAm(TYPE_PTCURMAP))
	if (map)
	{
		//boundaryTypeH = (dynamic_cast<PtCurMap *>(moverMap))->GetWaterBoundaries();
		//boundarySegmentsH = (dynamic_cast<PtCurMap *>(moverMap))->GetBoundarySegs();
		//boundaryPointsH = (dynamic_cast<PtCurMap *>(moverMap))->GetBoundaryPoints();
		boundaryTypeH = map->GetWaterBoundaries();
		boundarySegmentsH = map->GetBoundarySegs();
		boundaryPointsH = map->GetBoundaryPoints();
		if (!boundaryTypeH || !boundarySegmentsH || !boundaryPointsH) {printError("No map info to export"); err=-1; goto done;}
	}
	
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
		sprintf(topoStr,"%lf\t%lf\n",x,y);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	//code goes here, boundary points - an optional handle, only for curvilinear case
	
	if (boundarySegmentsH) 
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
}*/


