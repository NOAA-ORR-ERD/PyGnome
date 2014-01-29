
#include "TWindMover.h"
#include "OUtils.h"
#include "Cross.h"
#include "EditWindsDialog.h"

#ifdef MAC
#ifdef MPW
#include <QDOffscreen.h>
#pragma SEGMENT TWINDMOVER
#endif
#endif

#define TWINDMOVERMAXNUMDATALINESINLIST 201

TWindMover::TWindMover(TMap *owner,char* name) : TMover(owner, name)
{
	if(!name || !name[0]) this->SetClassName("Variable Wind"); // JLM , a default useful in the wizard
	timeDep = nil;
	
	fUncertainStartTime = 0;
	fDuration = 3*3600; // 3 hours
	
	fWindUncertaintyList = 0;
	fLESetSizes = 0;
	
	fSpeedScale = 2;
	fAngleScale = .4;
	fMaxSpeed = 30; //mps
	fMaxAngle = 60; //degrees
	fSigma2 =0;
	fSigmaTheta =  0; 
	//conversion = 1.0;// JLM , I think this field should be removed
	bTimeFileOpen = FALSE;
	bUncertaintyPointOpen=false;
	bSubsurfaceActive = false;
	fGamma = 1.;
	
	fIsConstantWind = false;
	fConstantValue.u = fConstantValue.v = 0.0;
	
	memset(&fWindBarbRect,0,sizeof(fWindBarbRect)); 
	bShowWindBarb = true;
}


//moved to windmover_c
/*void TWindMover::Dispose()
{
	DeleteTimeDep ();
	
	this->DisposeUncertainty();
	
	TMover::Dispose ();
	
}*/


long TWindMover::GetListLength()
{
	long count = 1; // wind name
	long mode = model->GetModelMode();
	
	if (bOpen) {
		float arrowDepth;
		if(mode == ADVANCEDMODE) count += 1; // active
		if (mode == ADVANCEDMODE && model->ThereIsA3DMover(&arrowDepth)) count +=1; // subsurface active
		/*if(mode == ADVANCEDMODE)*/ if (bActive) {count += 1;}// show wind barb
		if(fIsConstantWind)
		{ // constant wind
			//count += 1;// the value is on the first line
		}
		else
		{	// variable wind
			if(mode == ADVANCEDMODE && timeDep) count++;// time file name with bullet
			//NOTE: in non-advanced mode, the values are directly under the first toggle
			if (bTimeFileOpen || mode < ADVANCEDMODE)
			{
				long numDataLines = 0;
				if(timeDep) numDataLines = _min(timeDep -> GetNumValues (),TWINDMOVERMAXNUMDATALINESINLIST);
				count += numDataLines;
			}
		}
		if(mode == ADVANCEDMODE && model->IsUncertain())count++;
		if(mode == ADVANCEDMODE && model->IsUncertain() && bUncertaintyPointOpen)count+=4;
	}
	
	return count;
}

ListItem TWindMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char *p, latS[20], longS[20], timeS[30],valStr[32],valStr2[32];
	char str[128];
	DateTimeRec time;
	TimeValuePair pair;
	ListItem item = { this, n, indent, 0 };
	long mode = model->GetModelMode();
	
	if (n == 0) {
		char name[256];
		item.index = I_WINDNAME;
		
		if(fIsConstantWind)
		{ 	// add value to the first line
			// constant wind only has a toggle in advanced mode
			//if (mode == ADVANCEDMODE) item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;	// now have wind barb
			if (mode < ADVANCEDMODE) item.indent++;
			UV2RThetaStrings(fConstantValue.u,fConstantValue.v,this->timeDep -> GetUserUnits(),str);
			strcpy(text, "Constant Wind: ");
			strcat(text, str);
		}
		else
		{	// variable wind
			// variable wind always has a toggle 
			item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			// in lower modes we need to indent
			if (mode < ADVANCEDMODE) item.indent++;
			this->GetFileName(name); // show file name if there is one
			if(name[0]) sprintf(text, "Wind: \"%s\"", name);
			else strcpy(text, "Variable Wind");
		}
		if(!bActive)*style = italic; // JLM 6/14/10
		return item;
	}
	
	if (bOpen) {
		
		float arrowDepth;
		if (mode == ADVANCEDMODE && --n == 0) {
			item.indent++;
			item.index = I_WINDACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
		// code goes here, in standard mode indent?		
		if (bActive &&/*mode == ADVANCEDMODE &&*/ --n == 0) {
			item.indent++;
			item.index = I_WINDBARB;
			item.bullet = bShowWindBarb ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Wind Barb");
			
			return item;
		}
		
		if (mode == ADVANCEDMODE && model->ThereIsA3DMover(&arrowDepth) && --n == 0) {
			item.indent++;
			item.index = I_SUBSURFACEWINDACTIVE;
			item.bullet = bSubsurfaceActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "subsurface active");
			
			return item;
		}
		
		if(fIsConstantWind)
		{ // constant wind
			
			//if (--n == 0)
			//{
			//	item.index = I_WINDTIMEFILE;
			//	UV2RThetaStrings(fConstantValue.u,fConstantValue.v,this->timeDep -> GetUserUnits(),str);
			//	strcpy(text, str);
			//	
			//	return item;
			//}
		}
		else
		{ // variable wind
			
			if (mode == ADVANCEDMODE && --n == 0)
			{	// NOTE: in non-advanced mode
				// the values are directly under the first toggle
				// so we don't have this item
				char timeFileName [kMaxNameLen];
				
				item.indent++;
				item.index = I_WINDTIMEFILE;
				item.bullet = bTimeFileOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				if (timeDep)
				{
					timeDep -> GetTimeFileName (timeFileName);
					if(timeFileName[0]) sprintf(text, "Vector Time File: %s", timeFileName);
					else strcpy(text,"Values");
				}
				return item;
			}
			
			
			if ((bTimeFileOpen || mode < ADVANCEDMODE) && timeDep) 
			{
				long numDataLines = 0;
				if(timeDep) numDataLines = _min(timeDep -> GetNumValues (),TWINDMOVERMAXNUMDATALINESINLIST);
				if(numDataLines > 0)
				{
					if (--n < numDataLines) {
						item.indent++;
						item.index = I_WINDTIMEENTRIES + n;
						if(n >=(TWINDMOVERMAXNUMDATALINESINLIST-1)){
							strcpy(text,"...  (there are too many lines to show here)");
						}
						else {
							pair = INDEXH(timeDep -> GetTimeValueHandle (), n);
							SecondsToDate (pair.time, &time);
							Date2String(&time, timeS);
							if (p = strrchr(timeS, ':')) p[0] = 0; // remove seconds
							
							UV2RThetaStrings(pair.value.u,pair.value.v,this->timeDep -> GetUserUnits(),str);
							sprintf(text, "%s   %s",timeS,str);
						}
						
						return item;
					}
					n -= (numDataLines-1);
				}
			}
		}
		
		
		if(mode == ADVANCEDMODE && model->IsUncertain())
		{
			if (--n == 0) 
			{
				item.indent++;
				item.index = I_WINDUNCERTAIN;
				item.bullet = bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				strcpy(text, "Uncertainty");
				
				return item;
			}
			
			if(bUncertaintyPointOpen)
			{
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_WINDSTARTTIME;
					sprintf(text, "Start Time: %.2f hours",((double)fUncertainStartTime)/3600.);
					return item;
				}
				
				
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_WINDDURATION;
					sprintf(text, "Duration: %.2f hr", (float)(fDuration / 3600.0));
					return item;
				}
				
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_WINDSPEEDSCALE;
					sprintf(text, "Speed Scale: %.2f ", fSpeedScale);
					return item;
				}
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_WINDANGLESCALE;
					sprintf(text, "Angle Scale: %.2f ", fAngleScale);
					return item;
				}
			}
		}
	}
	
	item.owner = 0;
	
	return item;
}

Boolean TWindMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	OSErr err = 0;
	if (inBullet)
		switch (item.index) {
			case I_WINDNAME: bOpen = !bOpen; return TRUE;
			case I_WINDACTIVE: bActive = !bActive; 
				model->NewDirtNotification(); return TRUE;
			case I_SUBSURFACEWINDACTIVE: bSubsurfaceActive = !bSubsurfaceActive; 
				model->NewDirtNotification(); return TRUE;
			case I_WINDBARB: bShowWindBarb = !bShowWindBarb; 
				model->NewDirtNotification(); return TRUE;
			case I_WINDTIMEFILE: bTimeFileOpen = !bTimeFileOpen; return TRUE;
			case I_WINDUNCERTAIN:bUncertaintyPointOpen = !bUncertaintyPointOpen;return TRUE;
		}
	
	if (doubleClick)
	{	
		switch(item.index)
		{
			case I_WINDACTIVE:
			case I_SUBSURFACEWINDACTIVE:
			case I_WINDUNCERTAIN:
			case I_WINDSPEEDSCALE:
			case I_WINDANGLESCALE:
			case I_WINDSTARTTIME:
			case I_WINDDURATION:
				WindSettingsDialog(dynamic_cast<TWindMover *>(this), this -> moverMap,false,mapWindow,false);
				break;
			case I_WINDNAME:// JLM, first line brings up time dependent dialog
			case I_WINDBARB:
			default:
				if(dynamic_cast<TWindMover *>(this)->GetTimeDep())
				{
					Boolean settingsForcedAfterDialog = false;
					err=EditWindsDialog(dynamic_cast<TWindMover *>(this),model->GetStartTime(),false,settingsForcedAfterDialog);
				}
				break;
		}
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean TWindMover::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	switch (item.index) {
		case I_WINDNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON:
					if(model->GetModelMode() < ADVANCEDMODE) return FALSE; // novice users cannot delete wind
					else return TRUE;
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
	
	return TMover::FunctionEnabled(item, buttonID);
}

OSErr TWindMover::SettingsItem(ListItem item)
{
	//return WindSettingsDialog(this, this -> moverMap,false,mapWindow);
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = ListClick(item,inBullet,doubleClick);
	return 0;
}

OSErr TWindMover::DeleteItem(ListItem item)
{
	if (item.index == I_WINDNAME)
		return moverMap -> DropMover(dynamic_cast<TWindMover *>(this));
	
	return 0;
}

void TWindMover::GetFileName(char* fileName) 
{ 	// JLM 7/29/98
	// time object holds the file name
	// (we dont have to use the class name for this purpose)
	fileName[0] = 0;
	if(this->timeDep) strcpy(fileName,this->timeDep ->fileName);
}

void DrawArrowHead (Point p1, Point p2, VelocityRec velocity);

OSErr TWindMover::MakeClone(TWindMover **clonePtrPtr)
{
	// clone should be the address of a  ClassID ptr
	ClassID *clone;
	OSErr err = 0;
	Boolean weCreatedIt = false;
	if(!clonePtrPtr) return -1; // we are supposed to fill
	if(*clonePtrPtr == nil)
	{	// create and return a cloned object.
		*clonePtrPtr = new TWindMover(this->moverMap, "");
		weCreatedIt = true;
		if(!*clonePtrPtr) { TechError("MakeClone()", "new TWindMover()", 0); return memFullErr;}	
	}
	if(*clonePtrPtr)
	{	// copy the fields
		if((*clonePtrPtr)->GetClassID() == this->GetClassID()) // safety check
		{
			TWindMover * cloneP = dynamic_cast<TWindMover*>(*clonePtrPtr);// typecast 
			TMover *tObj = dynamic_cast<TMover *>(*clonePtrPtr);
			if(weCreatedIt) cloneP->InitMover();
			err =  TMover::MakeClone(&tObj);//  pass clone to base class
			if(!err) 
			{
				// we don't need to clone the uncertainty stuff
				// it can be recreated when the model is run
				if(this->timeDep)
					err = this->timeDep->MakeClone(&cloneP->timeDep);
				
				cloneP->fSpeedScale = this->fSpeedScale;
				cloneP->fAngleScale = this->fAngleScale;
				cloneP->fMaxSpeed = this->fMaxSpeed;
				cloneP->fMaxAngle = this->fMaxAngle;
				cloneP->fSigma2 = this->fSigma2;
				cloneP->fSigmaTheta = this->fSigmaTheta;
				cloneP->bTimeFileOpen = this->bTimeFileOpen;
				cloneP->bUncertaintyPointOpen = this->bUncertaintyPointOpen;
				cloneP->fIsConstantWind = this->fIsConstantWind;
				cloneP->fConstantValue = this->fConstantValue;
				
			}
		}
	}
done:
	if(err && *clonePtrPtr) 
	{
		(*clonePtrPtr)->Dispose();
		if(weCreatedIt)
		{
			delete *clonePtrPtr;
			*clonePtrPtr = nil;
		}
	}
	return err;
}


OSErr TWindMover::BecomeClone(TWindMover *clone)
{
	OSErr err = 0;
	
	if(clone)
	{
		if(clone->GetClassID() == this->GetClassID()) // safety check
		{
			//TOSSMTimeValue *timeDepClone = nil;
			TWindMover * cloneP = dynamic_cast<TWindMover *>(clone);// typecast
			
			 dynamic_cast<TWindMover *>(this)->TWindMover::Dispose(); // get rid of any memory we currently are using
			////////////////////
			// do the memory stuff first, in case it fails
			////////
			if(cloneP->timeDep)
				err = cloneP->timeDep->MakeClone(&this->timeDep);
			if(err) goto done;
			
			err =  TMover::BecomeClone(clone);//  pass clone to base class
			if(err) goto done;
			
			// we don't need to clone the uncertainty stuff
			// it can be recreated when the model is run
			// (it was de-allocated via Dispose call)
			
			//////////
			this->fSpeedScale = cloneP->fSpeedScale;
			this->fAngleScale = cloneP->fAngleScale;
			this->fMaxSpeed = cloneP->fMaxSpeed;
			this->fMaxAngle = cloneP->fMaxAngle;
			this->fSigma2 = cloneP->fSigma2;
			this->fSigmaTheta = cloneP->fSigmaTheta;
			this->bTimeFileOpen = cloneP->bTimeFileOpen;
			this->bUncertaintyPointOpen = cloneP->bUncertaintyPointOpen;
			this->fIsConstantWind = cloneP->fIsConstantWind;
			this->fConstantValue = cloneP->fConstantValue;
			
		}
	}
done:
	if(err)  dynamic_cast<TWindMover *>(this)->TWindMover::Dispose(); // don't leave ourselves in a weird state
	return err;
}

OSErr TWindMover::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	
	char ourName[kMaxNameLen];
	OSErr err = 0;
	
	// see if the message is of concern to us
	
	this->GetClassName(ourName);
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		double val;
		char str[256];
		////////////////
		err = message->GetParameterAsDouble("uncertaintySpeedScale",&val);
		if(!err)  dynamic_cast<TWindMover *>(this)->fSpeedScale = val; 
		////////////////
		err = message->GetParameterAsDouble("uncertaintyAngleScale",&val);
		if(!err)  dynamic_cast<TWindMover *>(this)->fAngleScale = val; 
		////////////////
		/*message->GetParameterString("windage",str,256); // was this ever used ?
		 if(str[0]) 
		 {
		 float minVal =0, maxVal = 0;
		 long numscanned = sscanf(str,"%f %f",&minVal,&maxVal);
		 if(numscanned == 2 && minVal >= 0.0 && maxVal >= minVal)
		 {
		 //this->windageA = minVal; 
		 //this->windageB = maxVal;
		 }
		 }*/
		/////////////
		model->NewDirtNotification();// tell model about dirt
	}
	
	/////////////////////////////////////////////////
	//  pass on this message to the TOSSMTimeValue object
	/////////////////////////////////////////////////
	if( dynamic_cast<TWindMover *>(this)->timeDep)
	{
		err =  dynamic_cast<TWindMover *>(this)->timeDep->CheckAndPassOnMessage(message);
	}
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TMover::CheckAndPassOnMessage(message);
	
}

void TWindMover::DrawWindVector(Rect r, WorldRect view)
{
	Point		p;
	short		h,v,x,y,d,dY,widestNum=0;
	RGBColor	rgb;
	Rect		rgbrect;
	Rect legendRect = fWindBarbRect;
	char 	valStr[30],text[30],errmsg[256];
	long 	i,istep=1;
	double	minLevel, maxLevel;
	double 	value, length, /*speedInKnots,*/ angle = 12., spacing = .1, scale = .5, factor = 1.;
	OSErr err = 0;
	float inchesX,inchesX2;
	float inchesY,inchesY2;
	short pixX,pixY;
	short pixX2,pixY2;
	short lowPointOfVector = 0;
	//long numPennants=0,numLongBarbs=0,numShortBarbs=0;
	Rect f;
	double conversionFactor = 1.;
	short userUnits = timeDep->GetUserUnits();
	char userUnitStr[32];
	ConvertToUnitsShort (userUnits, userUnitStr);
	
	if (gSavingOrPrintingPictFile)	return;	// don't put wind vector on the printouts
	
	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
#ifdef IBM
	//TextFont(kFontIDGeneva); TextSize(6);
	factor = .75;
#endif
	
	{
		if (EmptyRect(&fWindBarbRect)||!RectInRect2(&legendRect,&r))
		{
			legendRect.top = r.top;
			legendRect.left = r.right - 80;
			//legendRect.bottom = r.top + 90;	// reset after wind barb drawn
			legendRect.bottom = r.top + 85;	// reset after wind barb drawn
			legendRect.right = r.right;	// reset if values go beyond default width
		}
	}
	rgbrect = legendRect;
	EraseRect(&rgbrect);
	
	//widestNum = stringwidth("Wind Barb");
	
	v = rgbrect.top+35;
	h = rgbrect.left;
	for (i=0;i<istep;i++)
	{
		WorldPoint wp;
		Point p,p2,p3,midPt;
		VelocityRec velocity = {0.,0.}, velocity_orig = {0.,0.};
		
		settings.doNotPrintError = true;
		err =  dynamic_cast<TWindMover *>(this)->GetTimeValue (model->GetModelTime(), &velocity_orig);	// minus AH 07/10/2012
		
		settings.doNotPrintError = false;
		velocity.u = -1. * velocity_orig.u; velocity.v = -1. * velocity_orig.v;	// so velocity is from rather than to
		length = sqrt(velocity.u*velocity.u + velocity.v*velocity.v);
		switch(userUnits)
		{	// velocity stored in m/s so convert from this to desired units
			case kKnots: conversionFactor = 1./KNOTSTOMETERSPERSEC; break;
			case kMilesPerHour: conversionFactor = 1./MILESTOMETERSPERSEC; break;
			case kMetersPerSec: conversionFactor = 1.0; break;
			default: err = -1; return;
		}
		//speedInKnots = length * conversionFactor;
		
		if ((velocity.u != 0 || velocity.v != 0))
		{
			inchesX = factor*scale*velocity.u/length/2.;
			inchesY = factor*scale*velocity.v/length/2.;
			pixX = inchesX * PixelsPerInchCurrent();
			pixY = inchesY * PixelsPerInchCurrent();
			midPt.h = h+40;
			midPt.v = v+.5;
			p.h = midPt.h - pixX;
			p.v = midPt.v + pixY;
			p2.h = midPt.h + pixX;
			p2.v = midPt.v - pixY;
			MyMoveTo(midPt.h,midPt.v);
			MyLineTo(p2.h, p2.v);
			
			lowPointOfVector = p2.v;	// if this is high point, low point is zero, need to account for feathers though
			
			MyMoveTo(midPt.h,midPt.v);
			MyLineTo(p.h,p.v);
			DrawArrowHead (midPt, p, velocity_orig);
			if (p2.h-h>widestNum) 
				widestNum = p2.h-h;	// also issue of negative velocity, or super large value, maybe scale?
		}
		else
		{
			for (i=0;i<2;i++)
			{
				x = h + 40;
				y = v;
				d = 2*(i+1)+2;
				f.top = y - d; f.bottom = y + d;
				f.left = x - d; f.right = x + d;
				FrameOval(&f);
				if (f.right-h>widestNum) 
					widestNum = f.right-h;	// also issue of negative velocity, or super large value, maybe scale?
			}
		}
	}
	v = v + 28;
	StringWithoutTrailingZeros(valStr, length * conversionFactor, 0);
	sprintf(text, "%s %s",valStr,userUnitStr);
	x = (rgbrect.left + rgbrect.right) / 2;
	dY = 10;
	y = rgbrect.top + dY / 2;
	////MyMoveTo(x - stringwidth(text) / 2, y + 3 * dY);	//  might still want to center the text, recalc x,y though
	//MyMoveTo(h+20, v+5);
	MyMoveTo(x - stringwidth(text) / 2, v+5);
	drawstring(text);
	if (stringwidth(text)+20 > widestNum) 
		widestNum = stringwidth(text)+20;
	if (lowPointOfVector > legendRect.bottom) 
		legendRect.bottom = lowPointOfVector + 10;
	//if (legendRect.right<h+20+widestNum+4) 
	// legendRect.right = h+20+widestNum+4;
	if (legendRect.right<h+20+widestNum) 
		legendRect.right = h+20+widestNum;
	else if (legendRect.right>legendRect.left+80 && h+20+widestNum+4<=legendRect.left+80)
		legendRect.right = legendRect.left+80;	// may want to redraw to recenter the header
	RGBForeColor(&colors[BLACK]);
 	MyFrameRect(&legendRect);
	
	if (!gSavingOrPrintingPictFile)
		fWindBarbRect = legendRect;
	
	return;
}

#define TWindMoverREADWRITEVERSION 3 //CMO	7/10/02
//#define TWindMoverREADWRITEVERSION 2 //JLM	7/10/01

OSErr TWindMover::Write(BFPB *bfpb)
{
	long version = TWindMoverREADWRITEVERSION;
	ClassID id = GetClassID ();
	Boolean haveTimeDep;
	OSErr err = 0;
	
	if (err = TMover::Write(bfpb)) return err;
	
	StartReadWriteSequence("TWindMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	//if (err = WriteMacValue(bfpb, conversion)) return err;
	//if (err = WriteMacValue(bfpb, windageA)) return err;
	//if (err = WriteMacValue(bfpb, windageB)) return err;
	if (err = WriteMacValue(bfpb, fSpeedScale)) return err;
	if (err = WriteMacValue(bfpb, fAngleScale)) return err;
	if (err = WriteMacValue(bfpb, fMaxSpeed)) return err;
	if (err = WriteMacValue(bfpb, fMaxAngle)) return err;
	if (err = WriteMacValue(bfpb, fSigma2)) return err;
	if (err = WriteMacValue(bfpb, fSigmaTheta)) return err;
	
	if (err = WriteMacValue(bfpb, fIsConstantWind)) return err;
	if (err = WriteMacValue(bfpb, fConstantValue.u)) return err;
	if (err = WriteMacValue(bfpb, fConstantValue.v)) return err;
	
	if (err = WriteMacValue(bfpb, bSubsurfaceActive)) return err;
	if (err = WriteMacValue(bfpb, fGamma)) return err;
	
	if (err = WriteMacValue(bfpb,bTimeFileOpen)) return err;
	
	haveTimeDep = timeDep ? TRUE : FALSE;
	if (err = WriteMacValue(bfpb,haveTimeDep)) return err;
	if (haveTimeDep) {
		id = timeDep -> GetClassID();
		if (err = WriteMacValue(bfpb,id)) return err;
		if (err = timeDep -> Write(bfpb)) return err;
	}
	
	return 0;
}

OSErr TWindMover::Read(BFPB *bfpb)
{
	long version;
	ClassID id;
	Boolean haveTimeDep = 0;
	double windageA, windageB, conversion;
	OSErr err = 0;
	
	if (err = TMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("TWindMover::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TWindMover::Read()", "id != TYPE_WINDMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	//if (version != 1) { printSaveFileVersionError(); return -1; }
	if (version > TWindMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	if (version==1)	// windage fields have been moved to LEs
	{
		char msg[256]="";
		if (err = ReadMacValue(bfpb, &conversion)) return err;	// field removed
		if (err = ReadMacValue(bfpb, &windageA)) return err;
		if (err = ReadMacValue(bfpb, &windageB)) return err;
		if (windageA!=.01 || windageB!=.04)	
		{
			char valStr[32], valStr2[32];
			StringWithoutTrailingZeros(valStr, windageA * 100, 2);
			StringWithoutTrailingZeros(valStr2, windageB * 100, 2);
			sprintf(msg,"This save file is from an old version of Gnome. The windage values you set were %s%% and %s%%. The windage is now set in the spill dialog.",valStr,valStr2);
			printNote(msg);	// code goes here, use an Ok dialog
		}
	}
	
	if (err = ReadMacValue(bfpb, &fSpeedScale)) return err;
	if (err = ReadMacValue(bfpb, &fAngleScale)) return err;
	if (err = ReadMacValue(bfpb, &fMaxSpeed)) return err;
	if (err = ReadMacValue(bfpb, &fMaxAngle)) return err;
	if (err = ReadMacValue(bfpb, &fSigma2)) return err;
	if (err = ReadMacValue(bfpb, &fSigmaTheta)) return err;
	
	if (err = ReadMacValue(bfpb, &fIsConstantWind)) return err;
	if (err = ReadMacValue(bfpb, &fConstantValue.u)) return err;
	if (err = ReadMacValue(bfpb, &fConstantValue.v)) return err;
	
	if (version>2)
	{
		if (err = ReadMacValue(bfpb, &bSubsurfaceActive)) return err;
		if (err = ReadMacValue(bfpb, &fGamma)) return err;
	}
	
	// code goes here, either add bIsFirstStep and fModelStartTime or set them here
	bIsFirstStep = false;
	fModelStartTime = model->GetStartTime();
	
	if (err = ReadMacValue(bfpb, &bTimeFileOpen)) return err;
	
	if (err = ReadMacValue(bfpb, &haveTimeDep)) return err;
	if (haveTimeDep) {
		if (err = ReadMacValue(bfpb, &id)) return err;
		switch (id) {
				//case TYPE_TIMEVALUES: timeDep = new TTimeValue(this); break;
			case TYPE_OSSMTIMEVALUES: timeDep = new TOSSMTimeValue(dynamic_cast<TWindMover *>(this)); break;
			default: printError("Unrecognized time file type in TWindMover::Read()."); return -1;
		}
		if (!timeDep)
		{ TechError("TWindMover::Read()", "new TTimeValue()", 0); return -1; };
		if (err = timeDep -> InitTimeFunc()) return err;
		
		if (err = timeDep -> Read(bfpb)) return err;
	}
	else
		timeDep = nil;
	
	return 0;
	
}


OSErr TWindMover::ExportVariableWind(char* path)
{
	OSErr err = 0;
	TimeValuePairH timeValueH = 0;
	TimeValuePair pair;
	DateTimeRec dateTime;
	Seconds time;
	VelocityRec vel;
	double r,theta;
	long numTimeValuePairs,i;
	char buffer[512],windMagStr[64],windDirStr[64],timeStr[128],hdrStr[64];
	Boolean askForUnits; 
	double conversionFactor = 1.0;
	BFPB bfpb;
	short selectedUnits = timeDep->GetUserUnits();	
	
	timeValueH = timeDep -> GetTimeValueHandle();
	if(!timeValueH) 
	{
		printError("There are no wind values to export");
		return -1;
	}
	
	// ask user for units (or just automatically output in user's units?)
	//code goes here, we might want to put the units in the file somehow
	askForUnits = true; // default is to ask
	
	if(askForUnits)
	{	
		Boolean userCancel=false;
		//short selectedUnits = timeDep->GetUserUnits(); 
		err = AskUserForUnits(&selectedUnits,&userCancel);
		if(err || userCancel) { err = -1; goto done;}
		
		switch(selectedUnits)
		{	// velocity stored in m/s so convert from this to desired units
			case kKnots: conversionFactor = 1./KNOTSTOMETERSPERSEC; break;
			case kMilesPerHour: conversionFactor = 1./MILESTOMETERSPERSEC; break;
			case kMetersPerSec: conversionFactor = 1.0; break;
			default: err = -1; goto done;
		}
	}
	else
	{	// if units is nil, we don't do any conversion
		conversionFactor = 1.0;	
	}
	
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	
	//sprintf(hdrStr,"OSSM Long Wind from GNOME\n");	// station name
	sprintf(hdrStr,"OSSM Wind from GNOME\n");	// station name
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	sprintf(hdrStr,"Station Location unknown\n");	// station position - may want to use -1,-1,-1,-1 or something in the right format
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	SpeedUnitsToStr(selectedUnits, hdrStr);
	sprintf(hdrStr,"%s\n",hdrStr); // units
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	//sprintf(hdrStr,"LTIME\n");	// local time
	//strcpy(buffer,hdrStr);
	//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	//sprintf(hdrStr,"0,0,0,0,0,0,0,0\n");	// grid
	//strcpy(buffer,hdrStr);
	//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	// Write out the times and values
	numTimeValuePairs = timeDep->GetNumValues();
	for(i = 0; i< numTimeValuePairs;i++)
	{
		pair = INDEXH(timeValueH,i);
		time = pair.time;
		vel = pair.value;
		SecondsToDate(time,&dateTime); // convert to 2 digit year?
		//if(dateTime.year>=2000) dateTime.year-=2000;
		//if(dateTime.year>=1900) dateTime.year-=1900;
		sprintf(timeStr, "%02hd,%02hd,%02hd,%02hd,%02hd",
				dateTime.day, dateTime.month, dateTime.year,
				dateTime.hour, dateTime.minute);
		
		//UV2RTheta(vel.u,vel.v,&r,&theta); // this function is defined as static in Output.cpp so it can't be accessed from outside
		theta = fmod(atan2(vel.u,vel.v)*180/PI+360,360);
		theta = fmod(theta+180,360); // rotate the vector because wind is FROM this direction
		r=sqrt(vel.u*vel.u+vel.v*vel.v)*conversionFactor;
		
		StringWithoutTrailingZeros(windMagStr,r,1);
		StringWithoutTrailingZeros(windDirStr,theta,1);
		/////
		strcpy(buffer,timeStr);
		strcat(buffer,",");
		strcat(buffer,windMagStr);
		strcat(buffer,",");
		strcat(buffer,windDirStr);
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

///////////////////////////////////////////////////////////////////////////

static TWindMover *sharedWMover;
Boolean gSavePopTableInfo;

static PopInfoRec WindUncertaintyPopTable[] = {
	{ M18, nil, M18ANGLEUNITSPOPUP, 0, pANGLEUNITS, 0, 1, FALSE, nil }
};

short WindClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	char path[256];
	Boolean changed;
	short item;
	long menuID_menuItem; 
	Point where = CenteredDialogUpLeft(M38e);
	WorldPoint p;
	TOSSMTimeValue *timeFile;
	double gamma;
	
	switch (itemNum) {
		case M18OK:
		{
			Boolean isActive = GetButton(dialog, M18ACTIVE);
			short angleUnits = GetPopSelection(dialog, M18ANGLEUNITSPOPUP);
			float arrowDepth;
			
			//if (isActive && model->ThereIsA3DMover(&arrowDepth)) 
			if (isActive && gDispersedOilVersion) 
			{
				//PtCurMap *map = (PtCurMap*)(sharedWMover -> GetMoverMap());
				PtCurMap *map = GetPtCurMap();
				char msg[64];
				if (map)
				{
					double breakingWaveHt = map->GetBreakingWaveHeight();
					gamma = EditText2Float(dialog,M18GAMMA);
					//if (gamma >= map->fMixedLayerDepth / (map->fBreakingWaveHeight * 1.5))
					if (breakingWaveHt && gamma >= map->fMixedLayerDepth / (breakingWaveHt * 1.5))
					{	// code goes here, decide what to do if breakingWaveHt is zero
						sprintf(msg,"Gamma must be less than %.2f (mixed layer depth / breaking wave height * 1.5)",map->fMixedLayerDepth / (map->fBreakingWaveHeight * 1.5));
						printError(msg);
						break;
					}
				}
			}
			sharedWMover -> bSubsurfaceActive = GetButton(dialog, M18SUBSURFACEACTIVE);
			sharedWMover->fGamma = EditText2Float(dialog,M18GAMMA);
			sharedWMover -> bActive = GetButton(dialog, M18ACTIVE);
			sharedWMover->fAngleScale = EditText2Float(dialog,M18ANGLESCALE);
			sharedWMover->fSpeedScale = EditText2Float(dialog,M18SPEEDSCALE);
			sharedWMover->fUncertainStartTime = (long) round(EditText2Float(dialog,M18UNCERTAINSTARTTIME)*3600);
			if (angleUnits==2) sharedWMover->fAngleScale *= PI/180.;		
			sharedWMover -> fDuration = EditText2Float(dialog, M18DURATION) * 3600;			
			
			return M18OK;
		}
		case M18CANCEL: return M18CANCEL;
			
		case M18ACTIVE:
			ToggleButton(dialog, itemNum);
			break;
			
		case M18SUBSURFACEACTIVE:
			ToggleButton(dialog, itemNum);
			if (GetButton(dialog, M18SUBSURFACEACTIVE))
			{
				ShowHideDialogItem(dialog, M18GAMMA, true);
				ShowHideDialogItem(dialog, M18GAMMALABEL, true); 
			}
			else
			{
				ShowHideDialogItem(dialog, M18GAMMA, false); 
				ShowHideDialogItem(dialog, M18GAMMALABEL, false); 
			}
			break;
			
		case M18DURATION:
		case M18UNCERTAINSTARTTIME:
		case M18ANGLESCALE:
		case M18SPEEDSCALE:
		case M18GAMMA:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
			
		case M18ANGLEUNITSPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
			
	}
	
	return noErr;
}

OSErr WindInit(DialogPtr dialog, VOIDPTR data)
{
	float arrowDepth;
	SetDialogItemHandle(dialog, M18HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog,M18SETTINGSFRAME,(Handle)FrameEmbossed);
	SetDialogItemHandle(dialog,M18UNCERTAINFRAME,(Handle)FrameEmbossed);
	
	if (!gSavePopTableInfo)
		RegisterPopTable (WindUncertaintyPopTable, sizeof (WindUncertaintyPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (M18, dialog);
	
	SetPopSelection (dialog, M18ANGLEUNITSPOPUP, 1);
	
	SetButton(dialog, M18ACTIVE, sharedWMover->bActive);
	
	if (gDispersedOilVersion)
	{
		SetButton(dialog, M18SUBSURFACEACTIVE, sharedWMover->bSubsurfaceActive);
		
		if (sharedWMover->bSubsurfaceActive)
		{
			ShowHideDialogItem(dialog, M18GAMMA, true);
			ShowHideDialogItem(dialog, M18GAMMALABEL, true); 
		}
		else
		{
			ShowHideDialogItem(dialog, M18GAMMA, false); 
			ShowHideDialogItem(dialog, M18GAMMALABEL, false); 
		}
	}
	else
	{
		ShowHideDialogItem(dialog, M18SUBSURFACEACTIVE, false); // if there is a ptcurmap, should be 3D...
		ShowHideDialogItem(dialog, M18GAMMA, false); 
		ShowHideDialogItem(dialog, M18GAMMALABEL, false); 
	}
	
	Float2EditText(dialog, M18GAMMA, sharedWMover->fGamma, 4);
	Float2EditText(dialog, M18SPEEDSCALE, sharedWMover->fSpeedScale, 4);
	Float2EditText(dialog, M18ANGLESCALE, sharedWMover->fAngleScale, 4);
	
	Float2EditText(dialog, M18DURATION, sharedWMover->fDuration / 3600.0, 2);
	Float2EditText(dialog, M18UNCERTAINSTARTTIME, sharedWMover->fUncertainStartTime / 3600.0, 2);
	
	MySelectDialogItemText(dialog, M18SPEEDSCALE, 0, 255);
	
	//SetDialogItemHandle(dialog,M18SETTINGSFRAME,(Handle)FrameEmbossed);
	//SetDialogItemHandle(dialog,M18UNCERTAINFRAME,(Handle)FrameEmbossed);
	return 0;
}


OSErr GetWindFilePath(char *path)
{ // Note: returns USERCANCEL when user cancels
	Point 			where = CenteredDialogUpLeft (M38e);
	OSType 		typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 		reply;
	OSErr			err = noErr;
	
#if TARGET_API_MAC_CARBON
	mysfpgetfile(&where, "", -1, typeList,
				 (MyDlgHookUPP)0, &reply, M38e, MakeModalFilterUPP(STDFilter));
	if (!reply.good) return USERCANCEL;
	strcpy(path, reply.fullPath);
#else
	sfpgetfile(&where, "",
			   (FileFilterUPP)0,
			   -1, typeList,
			   (DlgHookUPP)0,
			   &reply, M38e,
			   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	if (!reply.good)  return USERCANCEL;
	
	my_p2cstr(reply.fName);
#ifdef MAC
	GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
#else
	strcpy(path, reply.fName);
#endif
#endif
	
	return err;
}

OSErr WindSettingsDialog(TWindMover *mover, TMap *owner,Boolean bAddMover,WindowPtr parentWindow, Boolean savePopTableInfo)
{ // Note: returns USERCANCEL when user cancels
	char 			path [256], outPath [256];
	short 			item;
	PopTableInfo saveTable = SavePopTable();
	short j, numItems = 0;
	PopInfoRec combinedDialogsPopTable[4];	// how many here? depends which type of wind was selected
	Point 			where = CenteredDialogUpLeft (M38e);
	OSType 		typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 		reply;
	TOSSMTimeValue 	*timeFile = nil;
	TWindMover 		*newMover = nil;
	OSErr			err = noErr;
	
	if(!owner && bAddMover) {printError("Programmer error"); return -1;}
	
	if (!mover) {
		// create and load 
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
					 (MyDlgHookUPP)0, &reply, M38e, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);
#else
		sfpgetfile(&where, "",
				   (FileFilterUPP)0,
				   -1, typeList,
				   (DlgHookUPP)0,
				   &reply, M38e,
				   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		if (!reply.good)  return USERCANCEL;
		
		my_p2cstr(reply.fName);
#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
#else
		strcpy(path, reply.fName);
#endif
#endif		
		newMover = new TWindMover(owner,""); //JLM, the fileName will be picked up in ReadTimeValues()
		if (!newMover)
		{ TechError("WindSettingsDialog()", "new TWindMover()", 0); return -1; }
		
		timeFile = new TOSSMTimeValue (newMover);
		if (!timeFile)
		{ TechError("WindSettingsDialog()", "new TOSSMTimeValue()", 0); delete newMover; return -1; }
		
#if TARGET_API_MAC_CARBON
		// ReadTimeValues expects unix style paths
		if (!err) err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		if (!err) strcpy(path,outPath);
#endif
		if (err = timeFile -> ReadTimeValues (path, M19MAGNITUDEDIRECTION, kUndefined)) // ask for units
		{ delete timeFile; delete newMover; return -1; }
		newMover->timeDep = timeFile;
		sharedWMover = newMover;		// new mover was created
	}
	else
		sharedWMover = mover;			// existing mover is being edited
	
	if (!err)
	{
		if(parentWindow == 0) parentWindow = mapWindow; // JLM 6/2/99
		gSavePopTableInfo = savePopTableInfo;
		// code to allow a dialog on top of another with pops
		if (gSavePopTableInfo)
		{
			for(j = 0; j < sizeof(WindUncertaintyPopTable) / sizeof(PopInfoRec);j++)
				combinedDialogsPopTable[numItems++] = WindUncertaintyPopTable[j];
			for(j= 0; j < saveTable.numPopUps ; j++)
				combinedDialogsPopTable[numItems++] = saveTable.popTable[j];
			
			RegisterPopTable(combinedDialogsPopTable,numItems);
		}
		
		item = MyModalDialog(M18, parentWindow, 0, WindInit, WindClick);
		if (gSavePopTableInfo)	RestorePopTableInfo(saveTable);
		if (item == M18OK)
		{
			if (newMover || bAddMover)
			{
				err = owner -> AddMover (sharedWMover, 0);
			}
			model->NewDirtNotification();
		}
		if(item == M18CANCEL)
		{
			err = USERCANCEL;
		}
	}
	
	if (err)
	{
		if (newMover != nil)
		{
			newMover -> Dispose ();
			delete newMover;
			newMover = nil;
		}
	}
	
	return err;
}

/*void TWindMover::DrawWindVector(Rect r, WorldRect view)
 {	// this code draws the wind barb as in meteorology, we decided to go simpler but just in case...
 short		h,v,x,y,d,dY,widestNum=0;
 RGBColor	rgb;
 Rect		rgbrect;
 Rect legendRect = fWindBarbRect;
 char 	valStr[30],text[30],errmsg[256];
 long 	i,istep=1;
 double	minLevel, maxLevel;
 double 	value, length, speedInKnots, angle = 12., spacing = .1, scale = .5, factor = 1.;
 OSErr err = 0;
 float inchesX,inchesX2;
 float inchesY,inchesY2;
 short pixX,pixY;
 short pixX2,pixY2;
 short lowPointOfVector = 0;
 long numPennants=0,numLongBarbs=0,numShortBarbs=0;
 Rect f;
 
 //SetRGBColor(&rgb,0,0,0);
 TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
 #ifdef IBM
 //TextFont(kFontIDGeneva); TextSize(6);
 factor = .75;
 #endif
 if (gSavingOrPrintingPictFile)	return;
 
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
 return;
 }
 else
 {
 if (EmptyRect(&fWindBarbRect)||!RectInRect2(&legendRect,&r))
 {
 legendRect.top = r.top;
 //legendRect.left = r.right - 80;
 legendRect.left = r.right - 120;
 legendRect.bottom = r.top + 66;	// reset after wind barb drawn
 legendRect.right = r.right;	// reset if values go beyond default width
 }
 }
 rgbrect = legendRect;
 EraseRect(&rgbrect);
 
 x = (rgbrect.left + rgbrect.right) / 2;
 //dY = RectHeight(rgbrect) / 12;
 dY = 10;
 y = rgbrect.top + dY / 2;
 widestNum = stringwidth("Wind Barb");
 
 v = rgbrect.top+45;
 h = rgbrect.left;
 for (i=0;i<istep;i++)
 {
 WorldPoint wp;
 Point p,p2,p3;
 VelocityRec velocity = {0.,0.};
 //Boolean offQuickDrawPlane = false;
 
 settings.doNotPrintError = true;
 err = GetTimeValue (model->GetModelTime(), &velocity);
 settings.doNotPrintError = false;
 velocity.u = -1. * velocity.u; velocity.v = -1. * velocity.v;	// so velocity is from rather than to
 length = sqrt(velocity.u*velocity.u + velocity.v*velocity.v);
 speedInKnots = length / KNOTSTOMETERSPERSEC;
 //MyMoveTo(h+40,v+.5);	// may want this on IBM
 MyMoveTo(h+60,v+.5);
 
 if ((velocity.u != 0 || velocity.v != 0))
 {
 inchesX = factor*scale*velocity.u/length;
 inchesY = factor*scale*velocity.v/length;
 pixX = inchesX * PixelsPerInchCurrent();
 pixY = inchesY * PixelsPerInchCurrent();
 //p.h = h+20;
 //p.h = h+40;
 p.h = h+60;
 p.v = v+.5;
 {
 x = p.h;
 y = p.v;
 d = 2;
 f.top = y - d; f.bottom = y + d;
 f.left = x - d; f.right = x + d;
 FrameOval(&f);
 }
 p2.h = p.h + pixX;
 p2.v = p.v - pixY;
 
 //MyMoveTo(p.h, p.v);
 MyLineTo(p2.h, p2.v);
 
 lowPointOfVector = p2.v;	// if this is high point, low point is zero, need to account for feathers though
 //MyDrawArrow(p.h,p.v,p2.h,p2.v);
 // code goes here, figure out wind speed and put feathers on accordingly
 // 5knots short barb, 10knots long barb, 50knots pennant
 // defaults 62 degree angle, spacing .1 * shaft length, barb length .33 * shaft length (long barb)
 //length / KNOTSTOMETERSPERSEC // speed in knots
 numPennants = floor(speedInKnots/50); 
 numLongBarbs = floor((speedInKnots - numPennants*50)/10.); 
 numShortBarbs = floor((speedInKnots - numPennants*50 - numLongBarbs*10)/5.); 
 
 if (numPennants > 0)
 {	// if (numPennants > 3) // wind speed greater than 200kts?
 for (i=0;i<numPennants;i++)
 {
 // need to connect a straight across line, maybe fill in
 pixX = spacing * (i) * inchesX * PixelsPerInchCurrent();
 pixY = spacing * (i) * inchesY * PixelsPerInchCurrent();
 MyMoveTo(p2.h-pixX,p2.v+pixY);
 inchesX2 = factor * cos(atan2(velocity.u,velocity.v)-0*PI/180.)/6.;
 inchesY2 = factor * sin(atan2(velocity.u,velocity.v)-0*PI/180.)/6.;
 pixX2 = inchesX2 * PixelsPerInchCurrent();
 pixY2 = inchesY2 * PixelsPerInchCurrent();
 p3.h = p2.h-pixX+pixX2;
 p3.v = p2.v+pixY+pixY2;
 MyLineTo(p3.h,p3.v);
 pixX = spacing * (i+1) * inchesX * PixelsPerInchCurrent();
 pixY = spacing * (i+1) * inchesY * PixelsPerInchCurrent();
 MyMoveTo(p2.h-pixX,p2.v+pixY);
 inchesX2 = factor * cos(atan2(velocity.u,velocity.v)-angle*PI/180.)/6.;
 inchesY2 = factor * sin(atan2(velocity.u,velocity.v)-angle*PI/180.)/6.;
 pixX2 = inchesX2 * PixelsPerInchCurrent();
 pixY2 = inchesY2 * PixelsPerInchCurrent();
 p3.h = p2.h-pixX+pixX2;
 p3.v = p2.v+pixY+pixY2;
 MyLineTo(p3.h,p3.v);
 }
 }
 if (numLongBarbs > 0)
 {
 for (i=0;i<numLongBarbs;i++)
 {
 pixX = spacing * (i+1) * inchesX * PixelsPerInchCurrent();
 pixY = spacing * (i+1) * inchesY * PixelsPerInchCurrent();
 MyMoveTo(p2.h-pixX,p2.v+pixY);
 inchesX2 = factor * cos(atan2(velocity.u,velocity.v)-angle*PI/180.)/6.;
 inchesY2 = factor * sin(atan2(velocity.u,velocity.v)-angle*PI/180.)/6.;
 pixX2 = inchesX2 * PixelsPerInchCurrent();
 pixY2 = inchesY2 * PixelsPerInchCurrent();
 p3.h = p2.h-pixX+pixX2;
 p3.v = p2.v+pixY+pixY2;
 MyLineTo(p3.h,p3.v);
 }
 }
 if (numShortBarbs > 0)
 {
 for (i=0;i<numShortBarbs;i++)
 {
 pixX = spacing * (i+1+numLongBarbs) * inchesX * PixelsPerInchCurrent();
 pixY = spacing * (i+1+numLongBarbs) * inchesY * PixelsPerInchCurrent();
 MyMoveTo(p2.h-pixX,p2.v+pixY);
 inchesX2 = factor * cos(atan2(velocity.u,velocity.v)-angle*PI/180.)/12.;
 inchesY2 = factor * sin(atan2(velocity.u,velocity.v)-angle*PI/180.)/12.;
 pixX2 = inchesX2 * PixelsPerInchCurrent();
 pixY2 = inchesY2 * PixelsPerInchCurrent();
 p3.h = p2.h-pixX+pixX2;
 p3.v = p2.v+pixY+pixY2;
 MyLineTo(p3.h,p3.v);
 }
 }
 if (p2.h-h>widestNum) widestNum = p2.h-h;	// also issue of negative velocity, or super large value, maybe scale?
 if (numPennants>0 || numLongBarbs>0 || numShortBarbs>0) {if (p3.h-h>widestNum) widestNum = p3.h-h;}	// also issue of negative velocity, or super large value, maybe scale?
 //v = v+9;
 }
 else
 {
 for (i=0;i<2;i++)
 {
 //x = h + 40;
 x = h + 60;
 y = v - 10;
 d = 2*(i+1)+2;
 f.top = y - d; f.bottom = y + d;
 f.left = x - d; f.right = x + d;
 FrameOval(&f);
 if (f.right-h>widestNum) widestNum = f.right-h;	// also issue of negative velocity, or super large value, maybe scale?
 }
 }
 // only 1 wind speed shown, if velocity is zero these lines should be skipped - might want to draw a calm circle...
 //if (p2.h-h>widestNum) widestNum = p2.h-h;	// also issue of negative velocity, or super large value, maybe scale?
 //if (p3.h-h>widestNum) widestNum = p3.h-h;	// also issue of negative velocity, or super large value, maybe scale?
 v = v+9;
 }
 StringWithoutTrailingZeros(valStr, length / KNOTSTOMETERSPERSEC, 0);
 sprintf(text, "Speed: %s knots",valStr);
 ////MyMoveTo(x - stringwidth(text) / 2, y + 3 * dY);
 MyMoveTo(h+20, v+5);
 drawstring(text);
 if (stringwidth(text)+20 > widestNum) widestNum = stringwidth(text)+20;
 v = v + 9;
 //legendRect.bottom = v+3;
 legendRect.bottom = v+8;
 //if (lowPointOfVector > legendRect.bottom) legendRect.bottom = lowPointOfVector + 5;
 if (lowPointOfVector > legendRect.bottom) legendRect.bottom = lowPointOfVector + 10;
 if (legendRect.right<h+20+widestNum+4) legendRect.right = h+20+widestNum+4;
 else if (legendRect.right>legendRect.left+80 && h+20+widestNum+4<=legendRect.left+80)
 legendRect.right = legendRect.left+80;	// may want to redraw to recenter the header
 RGBForeColor(&colors[BLACK]);
 MyFrameRect(&legendRect);
 
 if (!gSavingOrPrintingPictFile)
 fWindBarbRect = legendRect;
 
 return;
 }*/

