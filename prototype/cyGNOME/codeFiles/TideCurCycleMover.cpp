#include "Cross.h"
#include "Uncertainty.h"
#include "GridVel.h"
#include "TideCurCycleMover.h"
#include "OUtils.h"
#include "DagTreeIO.h"
#include "netcdf.h"
#include "NetCDFMover.h"
#include "TShioTimeValue.h"


#ifdef MAC
#ifdef MPW
#pragma SEGMENT TIDECURCYCLEMOVER
#endif
#endif

enum {
		I_GRIDCURNAME = 0 ,
		I_GRIDCURACTIVE, 
		I_GRIDCURGRID, 
		I_GRIDCURARROWS,
	   //I_GRIDCURSCALE,
		I_GRIDCURUNCERTAINTY,
		I_GRIDCURSTARTTIME,
		I_GRIDCURDURATION, 
		I_GRIDCURALONGCUR,
		I_GRIDCURCROSSCUR,
		I_GRIDCURMINCURRENT
		};


Boolean IsTideCurCycleFile (char *path, short *gridType)
{
	Boolean bIsValid = false;
	OSErr	err = noErr;
	long line;
	char strLine [512], outPath[256];
	char firstPartOfFile [512], *modelTypeStr=0, *gridTypeStr=0;
	long lenToRead,fileLength;
	int status, ncid;
	size_t t_len, t_len2;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{	// must start with CDF
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		if (!strncmp (firstPartOfFile, "CDF", 3))
			bIsValid = true;
	}
	if (!bIsValid) return false;

	// need global attribute to identify grid and model type
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {bIsValid=false; goto done;}	// this should probably be an error
	}
	//if (status != NC_NOERR) 
	//{
		// check if path is resource which contains a path
		/*CHARH f;
		long lenOfHdl;
		if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f))
			{ TechError("IsTideCurCycleMover::ReadTimeValues()", "ReadFileContents()", 0); bIsValid=false; goto done; }
		
		lenOfHdl = _GetHandleSize((Handle)f);
		//err = WriteFileContents(TATvRefNum, TATdirID, "GnomeTideFile.nc", '????', 'TEXT',
		//				 		*f, lenOfHdl, 0);
		err = WriteFileContents(TATvRefNum, TATdirID, "GnomeTideFile.nc", 'ttxt', 'TEXT',
						 		0, lenOfHdl, f);
		if(f) {DisposeHandle((Handle)f); f = 0;}
		// replace file path with new file path*/
		//bIsValid=false; 
		//goto done;
	//}	// this should probably be an error

	status = nc_inq_attlen(ncid,NC_GLOBAL,"model_type",&t_len);
	if (status == NC_NOERR) /*{*gridType = CURVILINEAR; goto done;}*/
	{
		modelTypeStr = new char[t_len+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "model_type", modelTypeStr);
		if (status != NC_NOERR) {bIsValid=false; goto done;} 
		modelTypeStr[t_len] = '\0';
		
		if (!strncmpnocase (modelTypeStr, "TidalCurrentCycle", 17))
		{
			bIsValid = true;
		}
		else
		{bIsValid=false; goto done;}
	}
	else
	{
		bIsValid=false;
		goto done;
		// for now require global model identifier
	}
	status = nc_inq_attlen(ncid,NC_GLOBAL,"grid_type",&t_len2);
	if (status == NC_NOERR) /*{*gridType = CURVILINEAR; goto done;}*/
	{
		gridTypeStr = new char[t_len2+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "grid_type", gridTypeStr);
		if (status != NC_NOERR) {*gridType = CURVILINEAR; goto done;} 
		gridTypeStr[t_len2] = '\0';
		
		if (!strncmpnocase (gridTypeStr, "REGULAR", 7))
		{
			 *gridType = REGULAR;
			 goto done;
		}
		if (!strncmpnocase (gridTypeStr, "CURVILINEAR", 11))
		{
			 *gridType = CURVILINEAR;
			 goto done;
		}
		if (!strncmpnocase (gridTypeStr, "TRIANGULAR", 10))
		{
			 *gridType = TRIANGULAR;
			 goto done;
		}
	}
	else
	{
		// for now don't require global grid identifier
	}
	
done:
	if (modelTypeStr) delete [] modelTypeStr;	
	if (gridTypeStr) delete [] gridTypeStr;	
	return bIsValid;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
// for now just use the CATS current dialog
/*static TideCurCycleMover *sTideCurCycleDialogMover;

short TideCurCycleMoverSettingsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	
	switch (itemNum) {
		case M33OK:
		{
			//mygetitext(dialog, M33NAME, sTideCurCycleDialogMover->fPathName, kPtCurUserNameLen-1);
			sTideCurCycleDialogMover->bActive = GetButton(dialog, M33ACTIVE);
			sTideCurCycleDialogMover->bShowArrows = GetButton(dialog, M33SHOWARROWS);
			sTideCurCycleDialogMover->arrowScale = EditText2Float(dialog, M33ARROWSCALE);
			//sTideCurCycleDialogMover->arrowDepth = arrowDepth;
			//sTideCurCycleDialogMover->fVar.curScale = EditText2Float(dialog, M33SCALE);
			sTideCurCycleDialogMover->fUpCurUncertainty = EditText2Float(dialog, M33ALONG)/100;
			sTideCurCycleDialogMover->fDownCurUncertainty = - EditText2Float(dialog, M33ALONG)/100;
			sTideCurCycleDialogMover->fRightCurUncertainty = EditText2Float(dialog, M33CROSS)/100;
			sTideCurCycleDialogMover->fLeftCurUncertainty = - EditText2Float(dialog, M33CROSS)/100;
			sTideCurCycleDialogMover->fEddyV0 = EditText2Float(dialog, M33MINCURRENT);
			sTideCurCycleDialogMover->fUncertainStartTime = (long)(round(EditText2Float(dialog, M33STARTTIME)*3600));
			sTideCurCycleDialogMover->fDuration = EditText2Float(dialog, M33DURATION)*3600;
	
			return M33OK;
		}

		case M33CANCEL: 
			return M33CANCEL;
		
		case M33ACTIVE:
		case M33SHOWARROWS:
			ToggleButton(dialog, itemNum);
			break;
		
		case M33ARROWSCALE:
		//case M33ARROWDEPTH:
		case M33SCALE:
		case M33ALONG:
		case M33CROSS:
		case M33MINCURRENT:
		case M33STARTTIME:
		case M33DURATION:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

	}
	
	return 0;
}


OSErr TideCurCycleMoverSettingsInit(DialogPtr dialog, VOIDPTR data)
{
	char pathName[256],fileName[64];
	SetDialogItemHandle(dialog, M33HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M33UNCERTAINTYBOX, (Handle)FrameEmbossed);
	
	strcpy(pathName,sTideCurCycleDialogMover->fPathName); // SplitPathFile changes the original path name
	SplitPathFile(pathName, fileName);
	mysetitext(dialog, M33NAME, fileName); // use short file name for now
	SetButton(dialog, M33ACTIVE, sTideCurCycleDialogMover->bActive);
	
	SetButton(dialog, M33SHOWARROWS, sTideCurCycleDialogMover->bShowArrows);
	Float2EditText(dialog, M33ARROWSCALE, sTideCurCycleDialogMover->arrowScale, 6);
	//Float2EditText(dialog, M33ARROWDEPTH, sTideCurCycleDialogMover->arrowDepth, 6);
	
	//Float2EditText(dialog, M33SCALE, sTideCurCycleDialogMover->fVar.curScale, 6);
	ShowHideDialogItem(dialog, M33SCALE, false); 
	ShowHideDialogItem(dialog, M33SCALELABEL, false); 
	Float2EditText(dialog, M33ALONG, sTideCurCycleDialogMover->fUpCurUncertainty*100, 6);
	Float2EditText(dialog, M33CROSS, sTideCurCycleDialogMover->fRightCurUncertainty*100, 6);
	Float2EditText(dialog, M33MINCURRENT, sTideCurCycleDialogMover->fEddyV0, 6);	// uncertainty min in mps ?
	Float2EditText(dialog, M33STARTTIME, sTideCurCycleDialogMover->fUncertainStartTime/3600., 2);
	Float2EditText(dialog, M33DURATION, sTideCurCycleDialogMover->fDuration/3600., 2);
	
	ShowHideDialogItem(dialog, M33TIMEZONEPOPUP, false); 
	ShowHideDialogItem(dialog, M33TIMESHIFTLABEL, false); 
	ShowHideDialogItem(dialog, M33TIMESHIFT, false); 
	ShowHideDialogItem(dialog, M33GMTOFFSETS, false); 

	MySelectDialogItemText(dialog, M33ALONG, 0, 100);
	
	return 0;
}



OSErr TideCurCycleMover::SettingsDialog()
{
	short item;
	//Point where = CenteredDialogUpLeft(M33);

	//OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	//MySFReply reply;
	
	sTideCurCycleDialogMover = this; // should pass in what is needed only
	item = MyModalDialog(M33, mapWindow, 0, TideCurCycleMoverSettingsInit, TideCurCycleMoverSettingsClick);
	sTideCurCycleDialogMover = 0;

	if(M33OK == item)	model->NewDirtNotification();// tell model about dirt
	return M33OK == item ? 0 : -1;
}*/

/////////////////////////////////////////////////

TideCurCycleMover::TideCurCycleMover (TMap *owner, char *name) : TCATSMover(owner, name)
{
	fTimeHdl = 0;

	fUserUnits = kUndefined;
	
	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	// Override TCurrentMover defaults
	fDownCurUncertainty = -.5; 
	fUpCurUncertainty = .5; 	
	fRightCurUncertainty = .25;  
	fLeftCurUncertainty = -.25; 
	fDuration=24*3600.; //24 hrs as seconds 
	fUncertainStartTime = 0.; // seconds
	fEddyV0 = 0.0;	// fVar.uncertMinimumInMPS
	//SetClassName (name); // short file name

	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
	fNumNodes = 0;
	
	//fPatternStartPoint = 2;	// some default
	fPatternStartPoint = MaxFlood;	// this should be user input
	fTimeAlpha = -1;

	fFillValue = -1e+34;
	fDryValue = -1e+34;
	//fDryValue = 999;
	
	fTopFilePath[0] = 0;	// don't seem to need this
}

/*OSErr TideCurCycleMover::InitMover()
{	
	OSErr	err = noErr;

	err = TCATSMover::InitMover ();
	return err;
}*/


LongPointHdl TideCurCycleMover::GetPointsHdl()
{
	return ((TTriGridVel*)fGrid) -> GetPointsHdl();
}

OSErr TideCurCycleMover::ComputeVelocityScale()
{	// this function computes and sets this->refScale
	// returns Error when the refScale is not defined
	// or in allowable range.  
	// Note it also sets the refScale to 0 if there is an error
#define MAXREFSCALE  1.0e6  // 1 million times is too much
#define MIN_UNSCALED_REF_LENGTH 1.0E-5 // it's way too small
	//long i, j, m, n;
	double length;
	//VelocityRec myVelocity;
	//TMap *map;
	//TCATSMover *mover;
	WorldPoint3D	deltaPoint = {0,0,0.};
	//WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha;
	long ptIndex1,ptIndex2,ptIndex3; 
	long index = -1; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	InterpolationVal interpolationVal;
	VelocityRec scaledPatVelocity;
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	char errmsg[256];
	
/*	if (this->timeDep && !this->timeDep->fFileType==SHIOCURRENTSFILE)
	{
		this->refScale = 0;
		return -1;
	}*/
	

	this->refScale = 1;
	return 0;
	// probably don't need this code	
	switch (scaleType) {
		case SCALE_NONE: this->refScale = 1; return noErr;
		case SCALE_CONSTANT:
			if(!this -> fOptimize.isOptimizedForStep) 
			{
				err = this -> SetInterval(errmsg);
				if (err) 
				{
					this->refScale = 0;
					return err;	// set to zero, avoid any accidental access violation
				}
			}
							
			// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
			interpolationVal = fGrid -> GetInterpolationValues(refP);
		
			if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
			{
				// this is only section that's different from ptcur
				ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
				ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
				ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
			}
			else
			{
				this->refScale = 0;
				return -1;	// set to zero, avoid any accidental access violation
			}
		
			// Check for constant current 
			if(GetNumTimesInFile()==1)
			{
				// Calculate the interpolated velocity at the point
				if (interpolationVal.ptIndex1 >= 0) 
				{
					scaledPatVelocity.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).u)
						+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).u)
						+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).u );
					scaledPatVelocity.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).v)
						+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).v)
						+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).v);
				}
				else	// if negative corresponds to negative ntri, set vel to zero
				{
					scaledPatVelocity.u = 0.;
					scaledPatVelocity.v = 0.;
				}
			}
			else // time varying current 
			{
				// Calculate the time weight factor
				if (fTimeAlpha==-1)
				{
					Seconds relTime = time - model->GetStartTime();
				startTime = (*fTimeHdl)[fStartData.timeIndex];
				endTime = (*fTimeHdl)[fEndData.timeIndex];
				//timeAlpha = (endTime - time)/(double)(endTime - startTime);
				timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
				}
				else
				timeAlpha=fTimeAlpha;
				 
				// Calculate the interpolated velocity at the point
				if (interpolationVal.ptIndex1 >= 0) 
				{
					scaledPatVelocity.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).u)
						+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).u)
						+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).u);
					scaledPatVelocity.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).v)
						+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).v)
						+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).v);
				}
				else	// if negative corresponds to negative ntri, set vel to zero
				{
					scaledPatVelocity.u = 0.;
					scaledPatVelocity.v = 0.;
				}
			}

			//myVelocity = GetPatValue(refP);
			//length = sqrt(myVelocity.u * myVelocity.u + myVelocity.v * myVelocity.v);
			length = sqrt(scaledPatVelocity.u * scaledPatVelocity.u + scaledPatVelocity.v * scaledPatVelocity.v);
			/// check for too small lengths
			if(fabs(scaleValue) > length*MAXREFSCALE
				|| length < MIN_UNSCALED_REF_LENGTH)
				{ this->refScale = 0;return -1;} // unable to compute refScale
			this->refScale = scaleValue / length; 
			return noErr;
		default:
			break;
	}
	
	this->refScale = 0;
	return -1;
}

OSErr TideCurCycleMover::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
{
	LEUncertainRec unrec;
	double u,v,lengthS,alpha,beta,v0;
	OSErr err = 0;
	
	err = this -> UpdateUncertainty();
	if(err) return err;
	
	if(!fUncertaintyListH || !fLESetSizesH) return 0; // this is our clue to not add uncertainty

	
	if(fUncertaintyListH && fLESetSizesH)
	{
		unrec=(*fUncertaintyListH)[(*fLESetSizesH)[setIndex]+leIndex];
		lengthS = sqrt(velocity->u*velocity->u + velocity->v * velocity->v);
		
	
		u = velocity->u;
		v = velocity->v;

		//if(lengthS < fVar.uncertMinimumInMPS)
		if(lengthS < fEddyV0)	// reusing the variable for now...
		{
			// use a diffusion  ??
			printError("nonzero UNCERTMIN is unimplemented");
			//err = -1;
		}
		else
		{	// normal case, just use cross and down stuff
			alpha = unrec.downStream;
			beta = unrec.crossStream;
		
			velocity->u = u*(1+alpha)+v*beta;
			velocity->v = v*(1+alpha)-u*beta;	
		}
	}
	else 
	{
		TechError("TideCurCycleMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		err = -1;
		velocity->u=velocity->v=0;
	}
	return err;
}

OSErr TideCurCycleMover::PrepareForModelStep()
{
	long timeDataInterval;
	OSErr err=0;
	char errmsg[256];
	
	errmsg[0]=0;

	//check to see that the time interval is loaded and set if necessary
	if (!bActive) return noErr;
		err = this -> SetInterval(errmsg);
	if(err) goto done;

	fOptimize.isOptimizedForStep = true;	// don't  use CATS eddy diffusion stuff, follow ptcur
	//fOptimize.value = sqrt(6*(fEddyDiffusion/10000)/model->GetTimeStep()); // in m/s, note: DIVIDED by timestep because this is later multiplied by the timestep
	//fOptimize.isFirstStep = (model->GetModelTime() == model->GetStartTime());
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TideCurCycleMover::PrepareForModelStep");
		printError(errmsg); 
	}	
	return err;
}

void TideCurCycleMover::ModelStepIsDone()
{
	fOptimize.isOptimizedForStep = false;
}

/*Boolean TideCurCycleMover::CheckInterval(long &timeDataInterval)
{
	Seconds time =  model->GetModelTime();
	long i,numTimes;


	numTimes = this -> GetNumTimesInFile(); 
	if (numTimes==0) {timeDataInterval = 0; return false;}	// really something is wrong, no data exists

	// check for constant current
	if (numTimes==1) 
	{
		timeDataInterval = -1; // some flag here
		if(fStartData.timeIndex==0 && fStartData.dataHdl)
				return true;
		else
			return false;
	}
	
	if(fStartData.timeIndex!=UNASSIGNEDINDEX && fEndData.timeIndex!=UNASSIGNEDINDEX)
	{
		if (time>=(*fTimeHdl)[fStartData.timeIndex] && time<=(*fTimeHdl)[fEndData.timeIndex])
				{	// we already have the right interval loaded
					timeDataInterval = fEndData.timeIndex;
					return true;
				}
	}

	for (i=0;i<numTimes;i++) 
	{	// find the time interval
		if (time>=(*fTimeHdl)[i] && time<=(*fTimeHdl)[i+1])
		{
			timeDataInterval = i+1; // first interval is between 0 and 1, and so on
			return false;
		}
	}	
	// don't allow time before first or after last
	if (time<(*fTimeHdl)[0]) 
		timeDataInterval = 0;
	if (time>(*fTimeHdl)[numTimes-1]) 
		timeDataInterval = numTimes;
	return false;
		
}*/

Boolean TideCurCycleMover::CheckInterval(long &timeDataInterval)
{
	Seconds time =  model->GetModelTime();
	long i,numTimes;


	numTimes = this -> GetNumTimesInFile(); 
	if (numTimes==0) {timeDataInterval = 0; return false;}	// really something is wrong, no data exists

	// check for constant current
	if (numTimes==1) 
	{
		timeDataInterval = -1; // some flag here
		if(fStartData.timeIndex==0 && fStartData.dataHdl)
				return true;
		else
			return false;
	}
	


	//#define MinBeforeFlood  0
	//#define MaxFlood  1
	//#define MinBeforeEbb  2
	//#define MaxEbb  3
	// need to know how tidal current pattern is divided up
	// starts with max flood, then ..., what is known?
	// each interval here corresponds to 3 in the pattern, assuming we have 12 times
	// should have a fPatternStartPoint - maxFlood or whatever
	{
	TShioTimeValue *timeFile = (TShioTimeValue*)GetTimeDep();
	long numTimes = GetNumTimesInFile(), offset, index1, index2;
	float  numSegmentsPerFloodEbb = numTimes/4.;
	if (!timeFile || !bTimeFileActive)
	{
		time = time - model->GetStartTime();
		fTimeAlpha = -1;
		// what used to be here
		if(fStartData.timeIndex!=UNASSIGNEDINDEX && fEndData.timeIndex!=UNASSIGNEDINDEX)
		{
			if (time>=(*fTimeHdl)[fStartData.timeIndex] && time<=(*fTimeHdl)[fEndData.timeIndex])
					{	// we already have the right interval loaded
						timeDataInterval = fEndData.timeIndex;
						return true;
					}
		}
	
		for (i=0;i<numTimes;i++) 
		{	// find the time interval
			if (time>=(*fTimeHdl)[i] && time<=(*fTimeHdl)[i+1])
			{
				timeDataInterval = i+1; // first interval is between 0 and 1, and so on
				return false;
			}
		}	
		// don't allow time before first or after last
		if (time<(*fTimeHdl)[0]) 
			timeDataInterval = 0;
		if (time>(*fTimeHdl)[numTimes-1]) 
			timeDataInterval = numTimes;
		return false;	// need to decide what to do here
	}
	else
	{
		short ebbFloodType;
		float fraction;
		timeFile->GetLocationInTideCycle(&ebbFloodType,&fraction);
		if (ebbFloodType>=fPatternStartPoint)
		{
			offset = ebbFloodType - fPatternStartPoint;
		}
		else
		{
			offset = ebbFloodType+4 - fPatternStartPoint;
		}
		index1 = floor(numSegmentsPerFloodEbb * (fraction + offset));	// round, floor?
		index2 = ceil(numSegmentsPerFloodEbb * (fraction + offset));	// round, floor?
		timeDataInterval = index2;	// first interval is between 0 and 1, and so on
		if (fraction==0) timeDataInterval++;
		fTimeAlpha = timeDataInterval-numSegmentsPerFloodEbb * (fraction + offset);
		// check if index2==numTimes, then need to cycle back to zero
		// check if index1==index2, then only need one file loaded
		if (index2==12) index2=0;
		if(fStartData.timeIndex==index1 && fEndData.timeIndex==index2)
		{
			return true;			
		}
		return false;
	}
	// check if interval already loaded, so fStartData.timeIndex == interval-1, fEndData.timeIndex = interval
	//return 0; 
	}

	return false;
		
}

void TideCurCycleMover::DisposeLoadedData(LoadedData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}

void TideCurCycleMover::ClearLoadedData(LoadedData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}

long TideCurCycleMover::GetNumTimesInFile()
{
	long numTimes = 0;

	if (fTimeHdl) numTimes = _GetHandleSize((Handle)fTimeHdl)/sizeof(**fTimeHdl);
	return numTimes;     
}

/*OSErr TideCurCycleMover::SetInterval(char *errmsg)
{
	long timeDataInterval;
	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval);
	long indexOfStart = timeDataInterval-1;
	long indexOfEnd = timeDataInterval;
	long numTimesInFile = this -> GetNumTimesInFile();
	OSErr err = 0;
		
	strcpy(errmsg,"");
	
	if(intervalLoaded) 
		return 0;
		
	// check for constant current 
	if(numTimesInFile==1)	//or if(timeDataInterval==-1) 
	{
		indexOfStart = 0;
		indexOfEnd = UNASSIGNEDINDEX;	// should already be -1
	}
	
	if(timeDataInterval == 0 || timeDataInterval == numTimesInFile)
	{	// before the first step in the file

		if (err==0)
		{
			// this will be handled differently here
		}
		else
		{
			err = -1;
			strcpy(errmsg,"Time outside of interval being modeled");
			goto done;
		}
	}
	// load the two intervals
	{
		DisposeLoadedData(&fStartData);
		
		if(indexOfStart == fEndData.timeIndex) // passing into next interval
		{
			fStartData = fEndData;
			ClearLoadedData(&fEndData);
		}
		else
		{
			DisposeLoadedData(&fEndData);
		}
		
		//////////////////
		
		if(fStartData.dataHdl == 0 && indexOfStart >= 0) 
		{ // start data is not loaded
			err = this -> ReadTimeData(indexOfStart,&fStartData.dataHdl,errmsg);
			if(err) goto done;
			fStartData.timeIndex = indexOfStart;
		}	
		
		if(indexOfEnd < numTimesInFile && indexOfEnd != UNASSIGNEDINDEX)  // not past the last interval and not constant current
		{
			err = this -> ReadTimeData(indexOfEnd,&fEndData.dataHdl,errmsg);
			if(err) goto done;
			fEndData.timeIndex = indexOfEnd;
		}
	}
	
done:	
	if(err)
	{
		if(!errmsg[0])strcpy(errmsg,"Error in TideCurCycleMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}
	return err;

}*/

OSErr TideCurCycleMover::SetInterval(char *errmsg)
{
	long timeDataInterval;
	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval);
	long indexOfStart = timeDataInterval-1;
	long indexOfEnd = timeDataInterval;
	long numTimesInFile = this -> GetNumTimesInFile();
	OSErr err = 0;
		
	strcpy(errmsg,"");
	
	if(intervalLoaded) 
		return 0;
		
	// check for constant current 
	if(numTimesInFile==1)	//or if(timeDataInterval==-1) 
	{
		indexOfStart = 0;
		indexOfEnd = UNASSIGNEDINDEX;	// should already be -1
	}
	
	if(timeDataInterval == 0)
	{	// before the first step in the file
		// shouldn't happen
			err = -1;
			strcpy(errmsg,"Time outside of interval being modeled");
			goto done;
	}
	if(timeDataInterval == numTimesInFile)
	{	// before the first step in the file
		if(fTimeAlpha>=0/*timeDep*/)
			indexOfEnd = 0;	// start over
		else
		{
			err = -1;
			//strcpy(errmsg,"Time outside of interval being modeled");
			strcpy(errmsg,"There are no more tidal patterns.");
			goto done;
		}
	}
	// load the two intervals
	{
		DisposeLoadedData(&fStartData);
		
		if(indexOfStart == fEndData.timeIndex) // passing into next interval
		{
			fStartData = fEndData;
			ClearLoadedData(&fEndData);
		}
		else
		{
			DisposeLoadedData(&fEndData);
		}
		
		//////////////////
		
		if(fStartData.dataHdl == 0 && indexOfStart >= 0) 
		{ // start data is not loaded
			err = this -> ReadTimeData(indexOfStart,&fStartData.dataHdl,errmsg);
			if(err) goto done;
			fStartData.timeIndex = indexOfStart;
		}	
		
		if(indexOfEnd < numTimesInFile && indexOfEnd != UNASSIGNEDINDEX)  // not past the last interval and not constant current
		{
			err = this -> ReadTimeData(indexOfEnd,&fEndData.dataHdl,errmsg);
			if(err) goto done;
			fEndData.timeIndex = indexOfEnd;
		}
	}
	
done:	
	if(err)
	{
		if(!errmsg[0])strcpy(errmsg,"Error in TideCurCycleMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}
	return err;

}

Boolean TideCurCycleMover::IsDryTriangle(long index1, long index2, long index3, float timeAlpha)
{
	Boolean isDry = false;
	VelocityRec vel1, vel2, vel3;
	// check timeAlpha = -1 (no time file, using fake times), 1 (start only), 0 (end only)
	if (timeAlpha == 1)
	{
		vel1.u = INDEXH(fStartData.dataHdl,index1).u;
		vel2.u = INDEXH(fStartData.dataHdl,index2).u;
		vel3.u = INDEXH(fStartData.dataHdl,index3).u;
		if (vel1.u == fDryValue && vel2.u == fDryValue && vel3.u == fDryValue)
			return true;
	}
	else if (timeAlpha == 0)
	{
		vel1.u = INDEXH(fEndData.dataHdl,index1).u;
		vel2.u = INDEXH(fEndData.dataHdl,index2).u;
		vel3.u = INDEXH(fEndData.dataHdl,index3).u;
		if (vel1.u == fDryValue && vel2.u == fDryValue && vel3.u == fDryValue)
			return true;
	}
	else if (timeAlpha > 0 && timeAlpha < 1)
	{
		vel1.u = INDEXH(fStartData.dataHdl,index1).u;
		vel2.u = INDEXH(fStartData.dataHdl,index2).u;
		vel3.u = INDEXH(fStartData.dataHdl,index3).u;
		if (vel1.u == fDryValue && vel2.u == fDryValue && vel3.u == fDryValue)
		{
			vel1.u = INDEXH(fEndData.dataHdl,index1).u;
			vel2.u = INDEXH(fEndData.dataHdl,index2).u;
			vel3.u = INDEXH(fEndData.dataHdl,index3).u;
			if (vel1.u == fDryValue && vel2.u == fDryValue && vel3.u == fDryValue)
			return true;
		}
		else
		return false;
	}
	return isDry;
}

Boolean TideCurCycleMover::IsDryTri(long triIndex)
{
	Boolean isDry = false;
	long index1, index2, index3, ptIndex1, ptIndex2, ptIndex3;
	double timeAlpha = 1;
	Seconds time = model->GetModelTime(), startTime, endTime;
	TTriGridVel* triGrid = (TTriGridVel*)fGrid;
	TopologyHdl topH = triGrid->GetTopologyHdl();

	index1 = (*topH)[triIndex].vertex1;
	index2 = (*topH)[triIndex].vertex2;
	index3 = (*topH)[triIndex].vertex3;
	ptIndex1 =  (*fVerdatToNetCDFH)[index1];	
	ptIndex2 =  (*fVerdatToNetCDFH)[index2];
	ptIndex3 =  (*fVerdatToNetCDFH)[index3];

	if(GetNumTimesInFile()>1)
	{
		// Calculate the time weight factor
		if (fTimeAlpha==-1)
		{
			Seconds relTime = time - model->GetStartTime();
			startTime = (*fTimeHdl)[fStartData.timeIndex];
			endTime = (*fTimeHdl)[fEndData.timeIndex];
			//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
		}
		else
			timeAlpha = fTimeAlpha;
	}
	 
	isDry = IsDryTriangle(ptIndex1, ptIndex2, ptIndex3, timeAlpha);
	return isDry;
}

VelocityRec TideCurCycleMover::GetStartVelocity(long index, Boolean *isDryPt)
{
	VelocityRec vel = {0,0};
	*isDryPt = false;
	if (index>=0)
	{
		vel.u = INDEXH(fStartData.dataHdl,index).u;
		vel.v = INDEXH(fStartData.dataHdl,index).v;
	}
	if (vel.u == fDryValue || vel.v == fDryValue) 
	{
		*isDryPt = true; 
		vel.u = 0; 
		vel.v = 0;
	}
	return vel;
}

VelocityRec TideCurCycleMover::GetEndVelocity(long index, Boolean *isDryPt)
{
	VelocityRec vel = {0,0};
	*isDryPt = false;
	if (index>=0)
	{
		vel.u = INDEXH(fEndData.dataHdl,index).u;
		vel.v = INDEXH(fEndData.dataHdl,index).v;
	}
	if (vel.u == fDryValue || vel.v == fDryValue) 
	{
		*isDryPt = true; 
		vel.u = 0; 
		vel.v = 0;
	}
	return vel;
}

double TideCurCycleMover::GetStartUVelocity(long index)
{
	double u = 0;
	if (index>=0)
	{
		u = INDEXH(fStartData.dataHdl,index).u;
	}
	if (u == fDryValue) {u = 0;}
	return u;
}

double TideCurCycleMover::GetEndUVelocity(long index)
{
	double u = 0;
	if (index>=0)
	{
		u = INDEXH(fEndData.dataHdl,index).u;
	}
	if (u == fDryValue) {u = 0;}
	return u;
}

double TideCurCycleMover::GetStartVVelocity(long index)
{
	double v = 0;
	if (index>=0)
	{
		v = INDEXH(fStartData.dataHdl,index).v;
	}
	if (v == fDryValue) {v = 0;}
	return v;
}

double TideCurCycleMover::GetEndVVelocity(long index)
{
	double v = 0;
	if (index>=0)
	{
		v = INDEXH(fEndData.dataHdl,index).v;
	}
	if (v == fDryValue) {v = 0;}
	return v;
}


WorldPoint3D TideCurCycleMover::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	// see PtCurMover::GetMove - will depend on what is in netcdf files and how it's stored
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha;
	long ptIndex1,ptIndex2,ptIndex3; 
	long index = -1; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	InterpolationVal interpolationVal;
	VelocityRec scaledPatVelocity, timeValue = {1, 1};
	Boolean useEddyUncertainty = false, isDry = false;	
	OSErr err = 0;
	char errmsg[256];
	
	(*theLE).leCustomData = 0;

	if(!this -> fOptimize.isOptimizedForStep) 
	{
		err = this -> SetInterval(errmsg);
		if (err) return deltaPoint;
	}
							
	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	interpolationVal = fGrid -> GetInterpolationValues(refPoint);

	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		// this is only section that's different from ptcur
		ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
		ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
		ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
	}
	else
		return deltaPoint;	// set to zero, avoid any accidental access violation

	// Check for constant current 
	if(GetNumTimesInFile()==1)
	{
		// check if all 3 values are dry and if so set LE.customdata = -1;, else 0;
		// set u,v=0 for dry values, on return check if customdata=-1 and then don't move at all this step
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			isDry = IsDryTriangle(ptIndex1,ptIndex2,ptIndex3,1);
			if (isDry) 
			{ 
				(*theLE).leCustomData = -1;
				return deltaPoint;
			}
			scaledPatVelocity.u = interpolationVal.alpha1*GetStartUVelocity(ptIndex1)
				+interpolationVal.alpha2*GetStartUVelocity(ptIndex2)
				+interpolationVal.alpha3*GetStartUVelocity(ptIndex3);
			scaledPatVelocity.v = interpolationVal.alpha1*GetStartVVelocity(ptIndex1)
				+interpolationVal.alpha2*GetStartVVelocity(ptIndex2)
				+interpolationVal.alpha3*GetStartVVelocity(ptIndex3);
			/*scaledPatVelocity.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).u)
				+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).u)
				+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).u );
			scaledPatVelocity.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).v)
				+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).v)
				+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).v);*/
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (fTimeAlpha==-1)
		{
			Seconds relTime = time - model->GetStartTime();
			startTime = (*fTimeHdl)[fStartData.timeIndex];
			endTime = (*fTimeHdl)[fEndData.timeIndex];
			//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
		}
		else
			timeAlpha = fTimeAlpha;
		 
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			isDry = IsDryTriangle(ptIndex1,ptIndex2,ptIndex3,timeAlpha);
			if (isDry) 
			{ 
				(*theLE).leCustomData = -1; 
				return deltaPoint;
			}
			/*scaledPatVelocity.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).u)
				+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).u)
				+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).u);
			scaledPatVelocity.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).v)
				+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).v)
				+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).v);*/
			scaledPatVelocity.u = interpolationVal.alpha1*(timeAlpha*GetStartUVelocity(ptIndex1) + (1-timeAlpha)*GetEndUVelocity(ptIndex1))
				+interpolationVal.alpha2*(timeAlpha*GetStartUVelocity(ptIndex2) + (1-timeAlpha)*GetEndUVelocity(ptIndex2))
				+interpolationVal.alpha3*(timeAlpha*GetStartUVelocity(ptIndex3) + (1-timeAlpha)*GetEndUVelocity(ptIndex3));
			scaledPatVelocity.v = interpolationVal.alpha1*(timeAlpha*GetStartVVelocity(ptIndex1) + (1-timeAlpha)*GetEndVVelocity(ptIndex1))
				+interpolationVal.alpha2*(timeAlpha*GetStartVVelocity(ptIndex2) + (1-timeAlpha)*GetEndVVelocity(ptIndex2))
				+interpolationVal.alpha3*(timeAlpha*GetStartVVelocity(ptIndex3) + (1-timeAlpha)*GetEndVVelocity(ptIndex3));
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}

scale:

	// get and apply our time file scale factor
	if (timeDep && bTimeFileActive)
	{
		// VelocityRec errVelocity={1,1};
		// JLM 11/22/99, if there are no time file values, use zero not 1
		VelocityRec errVelocity={0,1};
		err = timeDep -> GetTimeValue (model -> GetModelTime(), &timeValue); 
		if(err) timeValue = errVelocity;
	}
	
	scaledPatVelocity.u *= abs(timeValue.u); // magnitude contained in u field only
	scaledPatVelocity.v *= abs(timeValue.u); 	// multiplying tide by tide, don't want to change phase
	//scaledPatVelocity.u = timeValue.u; // magnitude contained in u field only
	//scaledPatVelocity.v = timeValue.v; 	// multiplying tide by tide, don't want to change phase

	//scaledPatVelocity.u *= fVar.curScale; // may want to allow some sort of scale factor, though should be in file
	//scaledPatVelocity.v *= fVar.curScale; 
	
	
	if(leType == UNCERTAINTY_LE)
	{
		 AddUncertainty(setIndex,leIndex,&scaledPatVelocity,timeStep,useEddyUncertainty);
	}
	
	dLong = ((scaledPatVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
	dLat  =  (scaledPatVelocity.v / METERSPERDEGREELAT) * timeStep;

	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;

	return deltaPoint;
}
		
VelocityRec TideCurCycleMover::GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty)
{
	VelocityRec v = {0,0};
	printError("TideCurCycleMover::GetScaledPatValue is unimplemented");
	return v;
}

VelocityRec TideCurCycleMover::GetPatValue(WorldPoint p)
{
	VelocityRec v = {0,0};
	printError("TideCurCycleMover::GetPatValue is unimplemented");
	return v;
}

/*long TideCurCycleMover::GetVelocityIndex(WorldPoint p) 
{
	long rowNum, colNum;
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;

	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of TideCurCycleMover

	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;


	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)

		{ return -1; }
		
	//return INDEXH (fGridHdl, rowNum * fNumCols + colNum);
	return rowNum * fNumCols + colNum;
}*/

/////////////////////////////////////////////////
// routines for ShowCoordinates() to recognize gridcur currents
/*double TideCurCycleMover::GetStartUVelocity(long index)
{	// 
	double u = 0;
	if (fStartData.dataHdl && index>=0)
		u = INDEXH(fStartData.dataHdl,index).u;
	return u;
}

double TideCurCycleMover::GetEndUVelocity(long index)
{
	double u = 0;
	if (fEndData.dataHdl && index>=0)
		u = INDEXH(fEndData.dataHdl,index).u;
	return u;
}

double TideCurCycleMover::GetStartVVelocity(long index)
{
	double v = 0;
	if (fStartData.dataHdl && index >= 0)
		v = INDEXH(fStartData.dataHdl,index).v;
	return v;
}

double TideCurCycleMover::GetEndVVelocity(long index)
{
	double v = 0;
	if (fEndData.dataHdl && index >= 0)
		v = INDEXH(fEndData.dataHdl,index).v;
	return v;
}

OSErr TideCurCycleMover::GetStartTime(Seconds *startTime)
{
	OSErr err = 0;
	*startTime = 0;
	if (fStartData.timeIndex != UNASSIGNEDINDEX && fTimeHdl)
		*startTime = (*fTimeHdl)[fStartData.timeIndex];
	else return -1;
	return 0;
}

OSErr TideCurCycleMover::GetEndTime(Seconds *endTime)
{
	OSErr err = 0;
	*endTime = 0;
	if (fEndData.timeIndex != UNASSIGNEDINDEX && fTimeHdl)
		*endTime = (*fTimeHdl)[fEndData.timeIndex];
	else return -1;
	return 0;
}*/

Boolean TideCurCycleMover::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	Boolean isDry = false;

	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	long index;

	long ptIndex1,ptIndex2,ptIndex3; 
	InterpolationVal interpolationVal;

	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	err = this -> SetInterval(errmsg);
	if(err) return false;

	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	interpolationVal = fGrid -> GetInterpolationValues(wp.p);
	if (interpolationVal.ptIndex1<0) return false;

	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		// this is only section that's different from ptcur
		ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
		ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
		ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
	}

	// Check for constant current 
	if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			isDry = IsDryTriangle(ptIndex1,ptIndex2,ptIndex3,1);

			velocity.u = interpolationVal.alpha1*GetStartUVelocity(ptIndex1)
				+interpolationVal.alpha2*GetStartUVelocity(ptIndex2)
				+interpolationVal.alpha3*GetStartUVelocity(ptIndex3);
			velocity.v = interpolationVal.alpha1*GetStartUVelocity(ptIndex1)
				+interpolationVal.alpha2*GetStartUVelocity(ptIndex2)
				+interpolationVal.alpha3*GetStartUVelocity(ptIndex3);
			/*velocity.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).u)
				+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).u)
				+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).u );
			velocity.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).v)
				+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).v)
				+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).v);*/
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
		if(fTimeAlpha==-1)
		{
		Seconds relTime = time - model->GetStartTime();
		startTime = (*fTimeHdl)[fStartData.timeIndex];
		endTime = (*fTimeHdl)[fEndData.timeIndex];
		//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
		}
		else
		timeAlpha = fTimeAlpha;
		 
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			isDry = IsDryTriangle(ptIndex1,ptIndex2,ptIndex3,timeAlpha);

			velocity.u = interpolationVal.alpha1*(timeAlpha*GetStartUVelocity(ptIndex1) + (1-timeAlpha)*GetEndUVelocity(ptIndex1))
				+interpolationVal.alpha2*(timeAlpha*GetStartUVelocity(ptIndex2) + (1-timeAlpha)*GetEndUVelocity(ptIndex2))
				+interpolationVal.alpha3*(timeAlpha*GetStartUVelocity(ptIndex3) + (1-timeAlpha)*GetEndUVelocity(ptIndex3));
			velocity.v = interpolationVal.alpha1*(timeAlpha*GetStartVVelocity(ptIndex1) + (1-timeAlpha)*GetEndVVelocity(ptIndex1))
				+interpolationVal.alpha2*(timeAlpha*GetStartVVelocity(ptIndex2) + (1-timeAlpha)*GetEndVVelocity(ptIndex2))
				+interpolationVal.alpha3*(timeAlpha*GetStartVVelocity(ptIndex3) + (1-timeAlpha)*GetEndVVelocity(ptIndex3));
			/*velocity.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).u)
				+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).u)
				+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).u);
			velocity.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).v)
				+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).v)
				+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).v);*/
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
	lengthS = this->refScale * lengthU;

	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	if (!isDry)	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
							this->className, uStr, sStr);
	else sprintf(diagnosticStr, " [grid: %s, dry value]",
							this->className);

	return true;
}

void TideCurCycleMover::Dispose ()
{
	/*if (fGrid)
	{
		fGrid -> Dispose();
		delete fGrid;
		fGrid = nil;
	}*/

	if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);

	if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
	if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}


	TCATSMover::Dispose ();
}


#define TideCurCycleMoverREADWRITEVERSION 1 

OSErr TideCurCycleMover::Write (BFPB *bfpb)
{
	long i, version = TideCurCycleMoverREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numPoints, numTimes = GetNumTimesInFile(), index;
	Seconds time;
	WorldPointF vertex;
	OSErr err = 0;

	if (err = TCATSMover::Write (bfpb)) return err;

	StartReadWriteSequence("TideCurCycleMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	//if (err = WriteMacValue(bfpb, fNumRows)) goto done;
	//if (err = WriteMacValue(bfpb, fNumCols)) goto done;
	if (err = WriteMacValue(bfpb, fNumNodes)) goto done;
	if (err = WriteMacValue(bfpb, fPatternStartPoint)) goto done;
	if (err = WriteMacValue(bfpb, fTimeAlpha)) goto done;
	if (err = WriteMacValue(bfpb, fFillValue)) goto done;
	//if (err = WriteMacValue(bfpb, fDryValue)) goto done;
	if (err = WriteMacValue(bfpb, fUserUnits)) goto done;
	if (err = WriteMacValue(bfpb, fPathName, kMaxNameLen)) goto done;
	if (err = WriteMacValue(bfpb, fFileName, kMaxNameLen)) goto done;	

	//
	if (err = WriteMacValue(bfpb, numTimes)) goto done;
	for (i=0;i<numTimes;i++)
	{
		time = INDEXH(fTimeHdl,i);
		if (err = WriteMacValue(bfpb, time)) goto done;
	}
	
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
		TechError("TideCurCycleMover::Write(char* path)", " ", 0); 

	return err;
}

OSErr TideCurCycleMover::Read(BFPB *bfpb)
{
	char msg[256], fileName[64];
	long i, version, index, numPoints, numTimes;
	ClassID id;
	WorldPointF vertex;
	Seconds time;
	Boolean bPathIsValid = true;
	OSErr err = 0;
	
	if (err = TCATSMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("TideCurCycleMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TideCurCycleMover::Read()", "id != TYPE_TIDECURCYCLEMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > TideCurCycleMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	// read the type of grid used for the GridCur mover (should always be rectgrid...)
	if (err = ReadMacValue(bfpb, &fNumNodes)) goto done;	
	if (err = ReadMacValue(bfpb, &fPatternStartPoint)) goto done;
	if (err = ReadMacValue(bfpb, &fTimeAlpha)) goto done;
	if (err = ReadMacValue(bfpb, &fFillValue)) goto done;
	//if (err = ReadMacValue(bfpb, &DryValue)) goto done;
	if (err = ReadMacValue(bfpb, &fUserUnits)) goto done;
	if (err = ReadMacValue(bfpb, fPathName, kMaxNameLen)) goto done;	
	ResolvePath(fPathName); // JLM 6/3/10
	//if (!FileExists(0,0,fPathName)) {err=-1; sprintf(msg,"The file path %s is no longer valid.",fPathName); printError(msg); goto done;}
	if (!FileExists(0,0,fPathName)) 
	{	// allow user to put file in local directory
		char newPath[kMaxNameLen],/*fileName[64],*/*p;
		strcpy(fileName,"");
		strcpy(newPath,fPathName);
		p = strrchr(newPath,DIRDELIMITER);
		if (p)
		{
			strcpy(fileName,p);
			ResolvePath(fileName);
		}
		if (!fileName[0] || !FileExists(0,0,fileName)) 
		//if (!FileExists(0,0,fileName)) 
		{/*err=-1;*/ /*sprintf(msg,"The file path %s is no longer valid.",fPathName); printNote(msg);*/ bPathIsValid = false;/*goto done;*/}
		else
			strcpy(fPathName,fileName);
	}
	if (err = ReadMacValue(bfpb, fFileName, kMaxNameLen)) goto done;	
	//
	if (!bPathIsValid)
	{	// try other platform
		char delimStr[32] ={DIRDELIMITER,0};		
		strcpy(fileName,delimStr);
		strcat(fileName,fFileName);
		ResolvePath(fileName);
		if (!fileName[0] || !FileExists(0,0,fileName)) 
		{/*err=-1;*/ /*sprintf(msg,"The file path %s is no longer valid.",fVar.pathName); printNote(msg);*/ /*goto done;*/}
		else
		{
			strcpy(fPathName,fileName);
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
		sprintf(msg,"This save file references a netCDF file which cannot be found.  Please find the file \"%s\".",fPathName);printNote(msg);
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(newPath, reply.fullPath);
		strcpy (s, newPath);
		SplitPathFile (s, fileName);
		strcpy (fPathName, newPath);
		strcpy (fFileName, fileName);
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
			SplitPathFile (s, fileName);
			strcpy (fPathName, newPath);
			strcpy (fFileName, fileName);
		}
#endif
	}
	
	//
	if (err = ReadMacValue(bfpb, &numTimes)) goto done;	
	fTimeHdl = (Seconds**)_NewHandleClear(sizeof(Seconds)*numTimes);
	if(!fTimeHdl)
		{TechError("NetCDFMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numTimes ; i++) {
		if (err = ReadMacValue(bfpb, &time)) goto done;
		INDEXH(fTimeHdl, i) = time;
	}

	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVerdatToNetCDFH = (LONGH)_NewHandleClear(sizeof(long)*numPoints);	// for curvilinear
	if(!fVerdatToNetCDFH)
		{TechError("NetCDFMoverCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &index)) goto done;
		INDEXH(fVerdatToNetCDFH, i) = index;
	}
	
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVertexPtsH = (WORLDPOINTFH)_NewHandleClear(sizeof(WorldPointF)*numPoints);	// for curvilinear
	if(!fVertexPtsH)
		{TechError("NetCDFMoverCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &vertex.pLat)) goto done;
		if (err = ReadMacValue(bfpb, &vertex.pLong)) goto done;
		INDEXH(fVertexPtsH, i) = vertex;
	}
	
	
	//fUserUnits = kKnots;	// code goes here, implement using units
	
done:
	if(err)
	{
		TechError("TideCurCycleMover::Read(char* path)", " ", 0); 
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr TideCurCycleMover::CheckAndPassOnMessage(TModelMessage *message)
{
	// no good to do this here since topology needs to be read in before these messages are received
	/*char ourName[kMaxNameLen];
	OSErr err = 0;
	
	// see if the message is of concern to us
	this->GetClassName(ourName);
	
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		char str[256];
		message->GetParameterString("topFile",str,256);
		ResolvePath(str);
		if(str[0])
		{
			strcpy(fTopFilePath,str);
		}
	}*/
	return TCATSMover::CheckAndPassOnMessage(message); 
}

/////////////////////////////////////////////////
/*long TideCurCycleMover::GetListLength()
{
	long n;
	short indent = 0;
	short style;
	char text[256];
	ListItem item;
	
	for(n = 0;TRUE;n++) {
		item = this -> GetNthListItem(n,indent,&style,text);
		if(!item.owner)
			return n;
	}
	//return n;
}*/

ListItem TideCurCycleMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char valStr[64];
	ListItem item = { this, 0, indent, 0 };
	

	if (n == 0) {
		item.index = I_GRIDCURNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Tide Pattern Time Series: \"%s\"", fFileName);
		if(!bActive)*style = italic; // JLM 6/14/10 -- check this
		
		return item;
	}
	else return TCATSMover::GetNthListItem(n,indent,style,text);

	/*if (bOpen) {
	

		if (--n == 0) {
			item.index = I_GRIDCURACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			item.indent++;
			return item;
		}
		
		
		if (--n == 0) {
			item.index = I_GRIDCURGRID;
			item.bullet = bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			sprintf(text, "Show Grid");
			item.indent++;
			return item;
		}
		
		if (--n == 0) {
			item.index = I_GRIDCURARROWS;
			item.bullet = bShowArrows ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			StringWithoutTrailingZeros(valStr,arrowScale,6);
			sprintf(text, "Show Velocities (@ 1 in = %s m/s) ", valStr);
			
			item.indent++;
			return item;
		}
		
		//if (--n == 0) {
			//item.index = I_GRIDCURSCALE;
			//StringWithoutTrailingZeros(valStr,fVar.curScale,6);
			//sprintf(text, "Multiplicative Scalar: %s", valStr);
			//return item;
		//}
		
		
		if(model->IsUncertain())
		{
			if (--n == 0) {
				item.index = I_GRIDCURUNCERTAINTY;
				item.bullet = bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				strcpy(text, "Uncertainty");
				item.indent++;
				return item;
			}

			if (bUncertaintyPointOpen) {
			
				if (--n == 0) {
					item.index = I_GRIDCURALONGCUR;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fUpCurUncertainty*100,6);
					sprintf(text, "Along Current: %s %%",valStr);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_GRIDCURCROSSCUR;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fRightCurUncertainty*100,6);
					sprintf(text, "Cross Current: %s %%",valStr);
					return item;
				}
			
				if (--n == 0) {
					item.index = I_GRIDCURMINCURRENT;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fEddyV0,6);	// reusing the TCATSMover variable for now
					sprintf(text, "Current Minimum: %s m/s",valStr);
					return item;
				}

				if (--n == 0) {
					item.index = I_GRIDCURSTARTTIME;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fUncertainStartTime/3600,6);
					sprintf(text, "Start Time: %s hours",valStr);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_GRIDCURDURATION;
					//item.bullet = BULLET_DASH;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fDuration/3600,6);
					sprintf(text, "Duration: %s hours",valStr);
					return item;
				}
			
								
			}
			
		}  // uncertainty is on
		
	} // bOpen*/
	
	item.owner = 0;
	
	return item;
}

/*Boolean TideCurCycleMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_GRIDCURNAME: bOpen = !bOpen; return TRUE;
			case I_GRIDCURGRID: bShowGrid = !bShowGrid; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_GRIDCURARROWS: bShowArrows = !bShowArrows; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_GRIDCURUNCERTAINTY: bUncertaintyPointOpen = !bUncertaintyPointOpen; return TRUE;
			case I_GRIDCURACTIVE:
				bActive = !bActive;
				model->NewDirtNotification(); 
			return TRUE;
		}

	if (doubleClick && !inBullet)
	{
		Boolean userCanceledOrErr,timeFileChanged ;
		//(void) this -> SettingsDialog();
		CATSSettingsDialog (this, this -> moverMap, &timeFileChanged);
		return TRUE;
	}

	// do other click operations...
	
	return FALSE;
}*/

Boolean TideCurCycleMover::FunctionEnabled(ListItem item, short buttonID)
{
	switch (item.index) {
		case I_GRIDCURNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON: return TRUE;
			}
			break;
	}
	
	if (buttonID == SETTINGSBUTTON) return TRUE;
	
	return TCATSMover::FunctionEnabled(item, buttonID);
}

OSErr TideCurCycleMover::SettingsItem(ListItem item)
{
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = this -> ListClick(item,inBullet,doubleClick);
	return 0;
}

/*OSErr TideCurCycleMover::AddItem(ListItem item)
{
	if (item.index == I_GRIDCURNAME)
		return TMover::AddItem(item);
	
	return 0;
}*/

OSErr TideCurCycleMover::DeleteItem(ListItem item)
{
	if (item.index == I_GRIDCURNAME)
		return moverMap -> DropMover(this);
	
	return 0;
}

Boolean TideCurCycleMover::DrawingDependsOnTime(void)
{
	Boolean depends = bShowArrows;
	// if this is a constant current, we can say "no"
	//if (model->GetModelMode()==ADVANCEDMODE && bShowGrid) depends = true;
	if(this->GetNumTimesInFile()==1) depends = false;
	return depends;
}

void TideCurCycleMover::Draw(Rect r, WorldRect view) 
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
	
	if(fGrid && (bShowArrows || bShowGrid))
	{
		Boolean overrideDrawArrows = FALSE;
		TTriGridVel* triGrid = (TTriGridVel*)fGrid;
		//TopologyHdl topH = triGrid->GetTopologyHdl();
		long j,numTri;
		//Boolean offQuickDrawPlane = false;
		//numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);
		numTri = triGrid->GetNumTriangles();
		
		err = this -> SetInterval(errmsg);
		if (err) {fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,overrideDrawArrows,bShowGrid); return;}
		//if(err) return;
		
		/*if (bShowGrid)
		{	
			RGBForeColor(&colors[PURPLE]);

			for (j = 0 ; j< numTri; j++)
			{
				// get vertices, check if dry triangle, then draw gray
				Boolean isDryTri = IsDryTri(j);
				if (model->GetModelMode()==ADVANCEDMODE && isDryTri)
				{
					RGBForeColor(&colors[LIGHTGRAY]);
					//RGBForeColor(&colors[RED]);
					triGrid->DrawTriangle(&r,j,TRUE);	// fill triangles					
					RGBForeColor(&colors[PURPLE]);
				}
				else
					triGrid->DrawTriangle(&r,j,FALSE);	// don't fill triangles
			}
			RGBForeColor(&colors[BLACK]);
		}*/
		if (bShowGrid && !bShowArrows) fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,overrideDrawArrows,bShowGrid);
		//fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,overrideDrawArrows,bShowGrid);
		
		if(bShowArrows)
		{ // we have to draw the arrows
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			



		if (bShowGrid)
		{	
			RGBForeColor(&colors[PURPLE]);

			for (j = 0 ; j< numTri; j++)
			{
				// get vertices, check if dry triangle, then draw gray
				Boolean isDryTri = IsDryTri(j);
				if (model->GetModelMode()==ADVANCEDMODE && isDryTri)
				{
					RGBForeColor(&colors[LIGHTGRAY]);
					//RGBForeColor(&colors[RED]);
					triGrid->DrawTriangle(&r,j,TRUE);	// fill triangles					
					RGBForeColor(&colors[PURPLE]);
				}
				else
					triGrid->DrawTriangle(&r,j,FALSE);	// don't fill triangles
			}
			RGBForeColor(&colors[BLACK]);
		}
			//TTriGridVel* triGrid = (TTriGridVel*)fGrid;

			//err = this -> SetInterval(errmsg);
			//if(err) return;
			
			//loaded = this -> CheckInterval(timeDataInterval);
			//if(!loaded) return;

			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			// Check for time varying current 
			if(GetNumTimesInFile()>1)
			{
				// Calculate the time weight factor
				if (fTimeAlpha==-1)
				{
					Seconds relTime = time - model->GetStartTime();
					startTime = (*fTimeHdl)[fStartData.timeIndex];
					endTime = (*fTimeHdl)[fEndData.timeIndex];
					//timeAlpha = (endTime - time)/(double)(endTime - startTime);
					timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
				}
				else
					timeAlpha = fTimeAlpha;
			}
			 
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				long ptIndex = INDEXH(fVerdatToNetCDFH,i);
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.}, startVelocity = {0.,0.}, endVelocity = {0.,0.};
				Boolean offQuickDrawPlane = false, isDryPt1 = false, isDryPt2 = false;
				//long depthIndex1,depthIndex2;	// default to -1?, eventually use in surface velocity case

				/*GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth

				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					topDepth = INDEXH(fDepthsH,depthIndex1);
					bottomDepth = INDEXH(fDepthsH,depthIndex2);
					depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				}*/

				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if(GetNumTimesInFile()==1)
				{
					//if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					//{
						//velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
						//velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
						velocity = GetStartVelocity(ptIndex,&isDryPt1);
						isDryPt2 = true;
					//}
					/*else 	// below surface velocity
					{
						velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).u;
						velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).v;
					}*/
				}
				else // time varying current
				{
					//if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					//{
						//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
						//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
						startVelocity = GetStartVelocity(ptIndex,&isDryPt1);
						endVelocity = GetEndVelocity(ptIndex,&isDryPt2);
						//if (timeAlpha==1) isDryPt1 = true;
						//if (timeAlpha==0) isDryPt2 = true;
						if (timeAlpha==1) isDryPt2 = true;
						if (timeAlpha==0) isDryPt1 = true;
						velocity.u = timeAlpha*startVelocity.u + (1-timeAlpha)*endVelocity.u;
						velocity.v = timeAlpha*startVelocity.v + (1-timeAlpha)*endVelocity.v;
					//}
					/*else	// below surface velocity
					{
						velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u);
						velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).u);
						velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v);
						velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).v);
					}*/
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
				//else if (isDryPt1 && isDryPt2)
				if (bShowGrid && isDryPt1 && isDryPt2 && model->GetModelMode()==ADVANCEDMODE)
				{
					RGBForeColor(&colors[RED]);
					short offset = round(4*PixelsPerPoint()); // 2 points each way
					MyMoveTo(p.h-offset, p.v);MyLineTo(p.h+offset,p.v);
					MyMoveTo(p.h,p.v-offset);MyLineTo(p.h,p.v+offset);
					RGBForeColor(&colors[BLACK]);
				}
			}
		}
	}
}

/*void TideCurCycleMover::Draw(Rect r, WorldRect view) 
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
	
	if(fGrid && (bShowArrows || bShowGrid))
	{
		Boolean overrideDrawArrows = FALSE;
		fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,overrideDrawArrows,bShowGrid);
		if(bShowArrows)
		{ // we have to draw the arrows
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;

			err = this -> SetInterval(errmsg);
			if(err) return;
			
			//loaded = this -> CheckInterval(timeDataInterval);
			//if(!loaded) return;

			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			// Check for time varying current 
			if(GetNumTimesInFile()>1)
			{
				// Calculate the time weight factor
				if (fTimeAlpha==-1)
				{
					Seconds relTime = time - model->GetStartTime();
				startTime = (*fTimeHdl)[fStartData.timeIndex];
				endTime = (*fTimeHdl)[fEndData.timeIndex];
				//timeAlpha = (endTime - time)/(double)(endTime - startTime);
				timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
				}
				else
				timeAlpha = fTimeAlpha;
			}
			 
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				long ptIndex = INDEXH(fVerdatToNetCDFH,i);
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.}, startVelocity = {0.,0.}, endVelocity = {0.,0.};
				Boolean offQuickDrawPlane = false, isDryPt1 = false, isDryPt2 = false;
				//long depthIndex1,depthIndex2;	// default to -1?, eventually use in surface velocity case

				//GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				
				//if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					//continue;	// no value for this point at chosen depth

				//if (depthIndex2!=UNASSIGNEDINDEX)
				//{
					// Calculate the depth weight factor
					//topDepth = INDEXH(fDepthsH,depthIndex1);
					//bottomDepth = INDEXH(fDepthsH,depthIndex2);
					//depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				//}

				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if(GetNumTimesInFile()==1)
				{
					//if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					//{
						//velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
						//velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
						velocity = GetStartVelocity(ptIndex,&isDryPt1);
						isDryPt2 = true;
					//}
					//else 	// below surface velocity
					//{
						//velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).u;
						//velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).v;
					//}
				}
				else // time varying current
				{
					//if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					//{
						//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
						//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
						startVelocity = GetStartVelocity(ptIndex,&isDryPt1);
						endVelocity = GetEndVelocity(ptIndex,&isDryPt2);
						if (timeAlpha==1) isDryPt1 = true;
						if (timeAlpha==0) isDryPt2 = true;
						velocity.u = timeAlpha*startVelocity.u + (1-timeAlpha)*endVelocity.u;
						velocity.v = timeAlpha*startVelocity.v + (1-timeAlpha)*endVelocity.v;
					//}
					//else	// below surface velocity
					//{
						//velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u);
						//velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).u);
						//velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v);
						//velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).v);
					//}
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
				//else if (isDryPt1 && isDryPt2)
				if (isDryPt1 && isDryPt2 && model->GetModelMode()==ADVANCED_MODE)
				{
					RGBForeColor(&colors[RED]);
					short offset = round(2*PixelsPerPoint()); // 2 points each way
					MyMoveTo(p.h-offset, p.v);MyLineTo(p.h+offset,p.v);
					MyMoveTo(p.h,p.v-offset);MyLineTo(p.h,p.v+offset);
					RGBForeColor(&colors[BLACK]);
				}
			}
		}
	}
}*/
/////////////////////////////////////////////////////////////////

	 
// this assumes a triangle mover
//OSErr TideCurCycleMover::TextRead(char *path, TMap **newMap) 
OSErr TideCurCycleMover::TextRead(char *path, TMap **newMap, char *topFilePath) 
{
	// needs to be updated once triangle grid format is set
	
	OSErr err = 0;
	long i, numScanned;
	int status, ncid, nodeid, nbndid, bndid, neleid, latid, lonid, recid, timeid;
	size_t nodeLength, nbndLength, neleLength, recs, t_len;
	float timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0, month[10];	
	WORLDPOINTFH vertexPtsH=0;
	float *lat_vals=0,*lon_vals=0;
	short *bndry_indices=0, *bndry_nums=0, *bndry_type=0;
	static size_t latIndex=0,lonIndex=0,timeIndex,ptIndex=0,bndIndex[2]={0,0};
	static size_t pt_count, bnd_count[2];
	Seconds startTime, startTime2;
	double timeConversion = 1.;
	char errmsg[256] = "";
	char fileName[64],s[256],topPath[256],outPath[256];

	char *modelTypeStr=0;
	Point where;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	Boolean bTopFile = false;
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	LONGH waterBoundariesH=0;

	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	tree.numBranches = 0;
	TDagTree *dagTree = 0;

	if (!path || !path[0]) return 0;
	strcpy(fPathName,path);
	
	strcpy(s,path);
	SplitPathFile (s, fileName);
	strcpy(fFileName, fileName); // maybe use a name from the file

	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_unlimdim(ncid, &recid);
	if (status != NC_NOERR) {err = -1; goto done;} 

	status = nc_inq_varid(ncid, "time", &timeid); 
	if (status != NC_NOERR) {err = -1; goto done;} 

	//status = nc_inq_attlen(ncid, recid, "units", &t_len);
	// code goes here, there will be a different time scale for tidal current cycle patterns
	status = nc_inq_attlen(ncid, timeid, "units", &t_len);
	if (status != NC_NOERR) 
	{
		timeUnits = 0;	// files should always have this info
		timeConversion = 3600.;		// default is hours
		startTime2 = model->GetStartTime();	// default to model start time
		//err = -1; goto done;
	}
	else
	{
		DateTimeRec time;
		char unitStr[24], junk[10], junk2[10], junk3[10];
		
		timeUnits = new char[t_len+1];
		//status = nc_get_att_text(ncid, recid, "units", timeUnits);// recid is the dimension id not the variable id
		status = nc_get_att_text(ncid, timeid, "units", timeUnits);
		if (status != NC_NOERR) {err = -1; goto done;} 
		timeUnits[t_len] = '\0'; // moved this statement before StringSubstitute, JLM 5/2/10
		StringSubstitute(timeUnits, ':', ' ');
		StringSubstitute(timeUnits, '-', ' ');
		
		// now should be 'days since model start' - or hours, minutes,...
		// just store the seconds into the cycle and add model start time back in
		// when showing just the patterns
		numScanned=sscanf(timeUnits, "%s %s %s %s",
					  unitStr, junk, &junk2, &junk3) ;
		if (numScanned<4) // really only care about 1	
			{ err = -1; TechError("TideCurCycleMover::TextRead()", "sscanf() == 4", 0); goto done; }
		/*numScanned=sscanf(timeUnits, "%s %s %hd %hd %hd %hd %hd %hd",
					  unitStr, junk, &time.year, &time.month, &time.day,
					  &time.hour, &time.minute, &time.second) ;
		if (numScanned<8) // has two extra time entries ??	
			{ err = -1; TechError("TideCurCycleMover::TextRead()", "sscanf() == 8", 0); goto done; }
		DateToSeconds (&time, &startTime2);*/	// code goes here, which start Time to use ??
		if (!strcmpnocase(unitStr,"HOURS") || !strcmpnocase(unitStr,"HOUR"))
			timeConversion = 3600.;
		else if (!strcmpnocase(unitStr,"MINUTES") || !strcmpnocase(unitStr,"MINUTE"))
			timeConversion = 60.;
		else if (!strcmpnocase(unitStr,"SECONDS") || !strcmpnocase(unitStr,"SECOND"))
			timeConversion = 1.;
		else if (!strcmpnocase(unitStr,"DAYS") || !strcmpnocase(unitStr,"DAY"))
			timeConversion = 24*3600.;
	} 

	status = nc_inq_dimid(ncid, "node", &nodeid); 
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimlen(ncid, nodeid, &nodeLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimid(ncid, "nbnd", &nbndid);	
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "bnd", &bndid);	
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimlen(ncid, nbndid, &nbndLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	bnd_count[0] = nbndLength;
	bnd_count[1] = 1;
	bndry_indices = new short[nbndLength]; 
	bndry_nums = new short[nbndLength]; 
	bndry_type = new short[nbndLength]; 
	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}
	//bndIndex[1] = 0;
	bndIndex[1] = 1;	// take second point of boundary segments instead, so that water boundaries work out
	status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_indices);
	if (status != NC_NOERR) {err = -1; goto done;}
	bndIndex[1] = 2;
	status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_nums);
	if (status != NC_NOERR) {err = -1; goto done;}
	bndIndex[1] = 3;
	status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_type);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_inq_dimid(ncid, "nele", &neleid);	
	//if (status != NC_NOERR) {err = -1; goto done;}	// not using these right now so not required
	status = nc_inq_dimlen(ncid, neleid, &neleLength);
	//if (status != NC_NOERR) {err = -1; goto done;}	// not using these right now so not required

	// option to use index values?
	status = nc_inq_varid(ncid, "lat", &latid);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "lon", &lonid);
	if (status != NC_NOERR) {err = -1; goto done;}
	pt_count = nodeLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(nodeLength*sizeof(WorldPointF));
	if (!vertexPtsH) {err = memFullErr; goto done;}
	pts = (LongPointHdl)_NewHandleClear(nodeLength*sizeof(LongPoint));
	if (!pts) {err = memFullErr; goto done;}
	lat_vals = new float[nodeLength]; 
	lon_vals = new float[nodeLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	status = nc_get_vara_float(ncid, latid, &ptIndex, &pt_count, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, lonid, &ptIndex, &pt_count, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<nodeLength;i++)
	{
		INDEXH(vertexPtsH,i).pLat = lat_vals[i];	
		INDEXH(vertexPtsH,i).pLong = lon_vals[i];
		INDEXH(pts,i).v = (long)(lat_vals[i]*1e6);
		INDEXH(pts,i).h = (long)(lon_vals[i]*1e6);
	}
	fVertexPtsH	 = vertexPtsH;// get first and last, lat/lon values, then last-first/total-1 = dlat/dlon

	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
	
		if(nodeLength > 0)
		{
			WorldPoint  wp;
			for(i=0;i<nodeLength;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}

	status = nc_inq_dim(ncid, recid, recname, &recs);
	if (status != NC_NOERR) {err = -1; goto done;}
	fTimeHdl = (Seconds**)_NewHandleClear(recs*sizeof(Seconds));
	if (!fTimeHdl) {err = memFullErr; goto done;}
	startTime = model->GetStartTime();
	for (i=0;i<recs;i++)
	{
		Seconds newTime;
		// possible units are, HOURS, MINUTES, SECONDS,...
		timeIndex = i;
		//status = nc_get_var1_float(ncid, recid, &timeIndex, &timeVal);	// recid is the dimension id not the variable id
		status = nc_get_var1_float(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); err = -1; goto done;}
		//newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
		newTime = RoundDateSeconds(round(startTime+timeVal*timeConversion));
		//INDEXH(fTimeHdl,i) = newTime;	// which start time where?
		//if (i==0) startTime = newTime;
		INDEXH(fTimeHdl,i) = newTime - startTime;	// which start time where?
		//INDEXH(fTimeHdl,i) = startTime2+timeVal*timeConversion;	// which start time where?
		//if (i==0) startTime = startTime2+timeVal*timeConversion;
	}
	/*if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
	{
		model->SetModelTime(startTime);
		model->SetStartTime(startTime);
		model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
	}*/

	fNumNodes = nodeLength;

	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	err = this -> SetInterval(errmsg);
	if(err) goto done;

	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}
	//ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 


	// if location file don't bring up dialog ?
	//if (fTopFilePath[0]) {err = ReadTopology(fTopFilePath,newMap); goto done;}	// if location file specified path don't ask
	if (model->fWizard->HaveOpenWizardFile())
	//code goes here, come up with a graceful way to pass the message of the topology file path
	//{strcpy(fTopFilePath,"resnum 10009"); err = ReadTopology(fTopFilePath,newMap); goto done;}
	{if (topFilePath[0]) {strcpy(fTopFilePath,topFilePath); err = ReadTopology(fTopFilePath,newMap); goto done;}}
	//if (model->fWizard->HaveOpenWizardFile())
	//{	// there is a location file
		//if (fTopPath[0]) err = ReadTopology(topPath,newMap);	// if location file specified path don't ask
		//else err = ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	
		//goto done;
	//}

	// look for topology in the file
	// for now ask for an ascii file, output from Topology save option
	// need dialog to ask for file
	if (/*fIsNavy*/true)
	{
		short buttonSelected;
		buttonSelected  = MULTICHOICEALERT(1688,"Do you have an extended topology file to load?",FALSE);
		switch(buttonSelected){
			case 1: // there is an extended top file
				bTopFile = true;
				break;  
			case 3: // no extended top file
				bTopFile = false;
				break;
			case 4: // cancel
				err=-1;// stay at this dialog
				goto done;
		}
	}
	if(bTopFile)
	{
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		if (!reply.good) /*return USERCANCEL;*/
		{
			err = ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto done;
		}
		else
			strcpy(topPath, reply.fullPath);
		/*{
			err = ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto done;
		}*/
#else
		where = CenteredDialogUpLeft(M38c);
		sfpgetfile(&where, "",
					(FileFilterUPP)0,
					-1, typeList,
					(DlgHookUPP)0,
					&reply, M38c,
					(ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		if (!reply.good) 
		{
			err = ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
			goto done;	
			//return 0;
		}
		
		my_p2cstr(reply.fName);
		
	#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, topPath);
	#else
		strcpy(topPath, reply.fName);
	#endif
#endif		
		strcpy (s, topPath);
		err = ReadTopology(topPath,newMap);	// newMap here
		goto done;
		//SplitPathFile (s, fileName);
	}

	err = ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
	
	

done:
	if (err)
	{
		if (!errmsg[0]) 
			strcpy(errmsg,"Error opening NetCDF file");
		printNote(errmsg);
		//printNote("Error opening NetCDF file");
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;}
	}
	//printNote("NetCDF triangular grid model current mover is not yet implemented");

	if (timeUnits) delete [] timeUnits;
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (bndry_indices) delete [] bndry_indices;
	if (bndry_nums) delete [] bndry_nums;
	if (bndry_type) delete [] bndry_type;

	return err;
}
	 


/////////////////////////////////////////////////

OSErr TideCurCycleMover::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{	// - needs to be updated once triangle grid format is set
	OSErr err = 0;
	long i;
	char path[256], outPath[256]; 
	int status, ncid, numdims;
	int curr_ucmp_id, curr_vcmp_id;
	static size_t curr_index[] = {0,0,0};
	static size_t curr_count[3];
	float *curr_uvals,*curr_vvals, fill_value, dry_value = 0;
	long totalNumberOfVels = fNumNodes;
	VelocityFH velH = 0;
	long numNodes = fNumNodes;
	
	errmsg[0]=0;

	strcpy(path,fPathName);
	if (!path || !path[0]) return -1;

	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_ndims(ncid, &numdims);	// in general it's not the total number of dimensions but the number the variable depends on
	if (status != NC_NOERR) {err = -1; goto done;}

	curr_index[0] = index;	// time 
	curr_count[0] = 1;	// take one at a time
	//curr_count[1] = 1;	// depth
	//curr_count[2] = numNodes;

	// check for sigma or zgrid dimension
	if (numdims>=6)	// should check what the dimensions are
	{
		curr_count[1] = 1;	// depth
		//curr_count[1] = depthlength;	// depth
		curr_count[2] = numNodes;
	}
	else
	{
		curr_count[1] = numNodes;	
	}
	curr_uvals = new float[numNodes]; 
	if(!curr_uvals) {TechError("TideCurCycleMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
	curr_vvals = new float[numNodes]; 
	if(!curr_vvals) {TechError("TideCurCycleMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}

	status = nc_inq_varid(ncid, "u", &curr_ucmp_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "v", &curr_vcmp_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_get_att_float(ncid, curr_ucmp_id, "_FillValue", &fill_value);// missing_value vs _FillValue
	//if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_att_float(ncid, curr_ucmp_id, "missing_value", &fill_value);// missing_value vs _FillValue
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) fill_value=-9999.;
	status = nc_get_att_float(ncid, curr_ucmp_id, "dry_value", &dry_value);// missing_value vs _FillValue
	if (status != NC_NOERR) {/*err = -1; goto done;*/}  
	else fDryValue = dry_value;
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	velH = (VelocityFH)_NewHandleClear(totalNumberOfVels * sizeof(VelocityFRec));
	if (!velH) {err = memFullErr; goto done;}
	for (i=0;i<totalNumberOfVels;i++)
	{
		// really need to store the fill_value data and check for it when moving or drawing
		if (curr_uvals[i]==0.||curr_vvals[i]==0.)
			curr_uvals[i] = curr_vvals[i] = 1e-06;
		if (curr_uvals[i]==fill_value)
			curr_uvals[i]=0.;
		if (curr_vvals[i]==fill_value)
			curr_vvals[i]=0.;
		// for now until we decide what to do with the dry value flag
		//if (curr_uvals[i]==dry_value)
			//curr_uvals[i]=0.;
		//if (curr_vvals[i]==dry_value)
			//curr_vvals[i]=0.;
		INDEXH(velH,i).u = curr_uvals[i];	// need units
		INDEXH(velH,i).v = curr_vvals[i];
	}
	*velocityH = velH;
	fFillValue = fill_value;
	
done:
	if (err)
	{
		strcpy(errmsg,"Error reading current data from NetCDF file");
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	if (curr_uvals) delete [] curr_uvals;
	if (curr_vvals) delete [] curr_vvals;
	return err;
}

OSErr TideCurCycleMover::ReorderPoints(TMap **newMap, short *bndry_indices, short *bndry_nums, short *bndry_type, long numBoundaryPts) 
{
	OSErr err = 0;
	char errmsg[256];
	long i, n, nv = fNumNodes;
	long currentBoundary;
	long numVerdatPts = 0, numVerdatBreakPts = 0;

	LONGH vertFlagsH = (LONGH)_NewHandleClear(nv * sizeof(**vertFlagsH));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatPtsH));
	LONGH verdatBreakPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatBreakPtsH));

	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	LONGH waterBoundariesH=0;

	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	Boolean addOne = false;	// for debugging

	// write out verdat file for debugging
	/*FILE *outfile = 0;
	char name[32], path[256],m[300];
	SFReply reply;
	Point where = CenteredDialogUpLeft(M55);
	char ibmBackwardsTypeStr[32] = "";
	strcpy(name,"NewVerdat.dat");
	errmsg[0]=0;
	
 #ifdef MAC
		sfputfile(&where, "Name:", name, (DlgHookUPP)0, &reply);
 #else
		sfpputfile(&where, ibmBackwardsTypeStr, name, (MyDlgHookProcPtr)0, &reply,
	           M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
 #endif
	if (!reply.good) {err = -1; goto done;}

	my_p2cstr(reply.fName);
#ifdef MAC
	GetFullPath (reply.vRefNum, 0, (char *) "", path);
	strcat (path, ":");
	strcat (path, (char *) reply.fName);
#else
	strcpy(path, reply.fName);
#endif
	//strcpy(sExportSelectedTriPath, path); // remember the path for the user
	SetWatchCursor();
	sprintf(m, "Exporting VERDAT to %s...",path);
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(m);*/
	/////////////////////////////////////////////////


	if (!vertFlagsH || !verdatPtsH || !verdatBreakPtsH) {err = memFullErr; goto done;}
	
	// put boundary points into verdat list

	// code goes here, double check that the water boundary info is also reordered
	currentBoundary=1;
	if (bndry_nums[0]==0) addOne = true;	// for debugging
	for (i = 0; i < numBoundaryPts; i++)
	{	
		short islandNum, index;
		index = bndry_indices[i];
		islandNum = bndry_nums[i];
		if (addOne) islandNum++;	// for debugging
		INDEXH(vertFlagsH,index-1) = 1;	// note that point has been used
		INDEXH(verdatPtsH,numVerdatPts++) = index-1;	// add to verdat list
		if (islandNum>currentBoundary)
		{
			// for verdat file indices are really point numbers, subtract one for actual index
			INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = i;	// passed a break point
			currentBoundary++;
		}
	}
	INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = numBoundaryPts;

	// add the rest of the points to the verdat list (these points are the interior points)
	for(i = 0; i < nv; i++) {
		if(INDEXH(vertFlagsH,i) == 0)	
		{
			INDEXH(verdatPtsH,numVerdatPts++) = i;
			INDEXH(vertFlagsH,i) = 0; // mark as used
		}
	}
	if (numVerdatPts!=nv) 
	{
		printNote("Not all vertex points were used");
		// shrink handle
		_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(long));
	}
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	if(pts == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	/////////////////////////////////////////////////
	// write out the file
	/////////////////////////////////////////////////
	/*outfile=fopen(path,"w");
	if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	fprintf(outfile,"DOGS\tMETERS\n");*/
	
	for (i=0; i<=numVerdatPts; i++)
	{
		long index;
		float fLong, fLat, fDepth;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	
			index = i+1;
			n = INDEXH(verdatPtsH,i);
			fLat = INDEXH(fVertexPtsH,n).pLat;	// don't need to store fVertexPtsH, just pass in and use here
			fLong = INDEXH(fVertexPtsH,n).pLong;
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = 1.;	// this will be set from bathymetry, just a fudge here for outputting a verdat
			INDEXH(pts,i) = vertex;
		}
		else { // the last line should be all zeros
			index = 0;
			fLong = fLat = fDepth = 0.0;
		}
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth);	
		/////////////////////////////////////////////////
	}
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
	
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	// write out the number of chains
	/*fprintf(outfile,"%ld\n",numVerdatBreakPts);
	
	// now write out out the break points
	for(i = 0; i < numVerdatBreakPts; i++ )
	{
		fprintf(outfile,"%ld\n",INDEXH(verdatBreakPtsH,i));
	}
	/////////////////////////////////////////////////

	fclose(outfile);*/
	// shrink handle
	_SetHandleSize((Handle)verdatBreakPtsH,numVerdatBreakPts*sizeof(long));
	for(i = 0; i < numVerdatBreakPts; i++ )
	{
		INDEXH(verdatBreakPtsH,i)--;
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	// use new maketriangles to force algorithm to avoid 3 points in the same row or column
	if (err = maketriangles(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts))
	//if (err = maketriangles2(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts,verdatPtsH,fNumCols_ext))
		goto done;

	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	if (errmsg[0])	
		{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TideCurCycleMover::ReorderPoints()","new TTriGridVel" ,err);
		goto done;
	}

	fGrid = (TTriGridVel*)triGrid;

	triGrid -> SetBounds(triBounds); 

	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement

	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	//totalDepthH = 0; // because fGrid is now responsible for it
	
	/////////////////////////////////////////////////
	numBoundaryPts = INDEXH(verdatBreakPtsH,numVerdatBreakPts-1)+1;
	waterBoundariesH = (LONGH)_NewHandle(sizeof(long)*numBoundaryPts);
	if (!waterBoundariesH) {err = memFullErr; goto done;}
	
	for (i=0;i<numBoundaryPts;i++)
	{
		INDEXH(waterBoundariesH,i)=1;	// default is land
		if (bndry_type[i]==1)	
			INDEXH(waterBoundariesH,i)=2;	// water boundary, this marks start point rather than end point...
	}

	if (waterBoundariesH && this -> moverMap == model -> uMap)	// maybe assume rectangle grids will have map?
	{
		PtCurMap *map = CreateAndInitPtCurMap(fPathName,triBounds); // the map bounds are the same as the grid bounds
		if (!map) {err=-1; goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(verdatBreakPtsH);	
		map->SetWaterBoundaries(waterBoundariesH);

		*newMap = map;
	}
	else
	{
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH=0;}
	}

	/////////////////////////////////////////////////
	fVerdatToNetCDFH = verdatPtsH;	// this should be resized

done:
	if (err) printError("Error reordering gridpoints into verdat format");
	if (vertFlagsH) {DisposeHandle((Handle)vertFlagsH); vertFlagsH = 0;}

	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TideCurCycleMover::ReorderPoints");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}

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
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
	}
	return err;
}

/////////////////////////////////////////////////
OSErr TideCurCycleMover::ReadTopology(char* path, TMap **newMap)
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
	LONGH boundarySegs=0, waterBoundaries=0;

	errmsg[0]=0;
		
	if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("NetCDFMoverTri::ReadTopology()", "ReadFileContents()", err);
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
		//PtCurMap *map = CreateAndInitPtCurMap("Extended Topology",bounds); // the map bounds are the same as the grid bounds
		PtCurMap *map = CreateAndInitPtCurMap(fPathName,bounds); // the map bounds are the same as the grid bounds
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


	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverTri::ReadTopology()","new TTriGridVel" ,err);
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
			strcpy(errmsg,"An error occurred in NetCDFMoverTri::ReadTopology");
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

OSErr TideCurCycleMover::ExportTopology(char* path)
{
	// export NetCDF triangle info so don't have to regenerate each time
	// same as curvilinear so may want to combine at some point
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
		//boundaryTypeH = ((PtCurMap*)moverMap)->GetWaterBoundaries();
		//boundarySegmentsH = ((PtCurMap*)moverMap)->GetBoundarySegs();
		boundaryTypeH = map->GetWaterBoundaries();
		boundarySegmentsH = map->GetBoundarySegs();
		if (!boundaryTypeH || !boundarySegmentsH) {printError("No map info to export"); err=-1; goto done;}
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

