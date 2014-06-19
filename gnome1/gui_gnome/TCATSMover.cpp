#include "TCATSMover.h"
#include "GridCurMover.h"
#include "TideCurCycleMover.h"
#include "CurrentCycleMover.h"
#include "TWindMover.h"
#include "TShioTimeValue.h"
#include "DagTreePD.h"
#include "DagTreeIO.h"
#include "CROSS.H"
#include "GridCurrentMover.h"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT TCATSMOVER
#endif
#endif

//Rect CATSgridRect = { 0, 0, kOCurWidth, kOCurHeight };

static PopInfoRec csPopTable[] = {
	{ M16, nil, M16LATDIR, 0, pNORTHSOUTH1, 0, 1, FALSE, nil },
	{ M16, nil, M16LONGDIR, 0, pEASTWEST1, 0, 1, FALSE, nil },
	{ M16, nil, M16TIMEFILETYPES, 0, pTIMEFILETYPES, 0, 1, FALSE, nil }
};

static TCATSMover	*sharedCMover = 0;
static CMyList 		*sharedMoverList = 0;
static char 		sharedCMFileName[256];
static Boolean		sharedCMChangedTimeFile;
static TOSSMTimeValue *sharedCMDialogTimeDep = 0;
static CurrentUncertainyInfo sSharedCATSUncertainyInfo; // used to hold the uncertainty dialog box info in case of a user cancel

/////////////////////////////////////////////////
// JLM 11/25/98
// structure to help reset stuff when the user cancels from the uncertainty dialog box


static CATSDialogNonPtrFields sharedCatsDialogNonPtrFields;

TCATSMover::TCATSMover (TMap *owner, char *name) : TCurrentMover(owner, name)
{
	fDuration=48*3600; //48 hrs as seconds 
	fTimeUncertaintyWasSet =0;

	fGrid = 0;
	SetTimeDep (nil);
	bTimeFileActive = false;
	fEddyDiffusion=0; // JLM 5/20/991e6; // cm^2/sec
	fEddyV0 = 0.1; // JLM 5/20/99

	bApplyLogProfile = false;

	memset(&fOptimize,0,sizeof(fOptimize));
	SetClassName (name);
}

// moved to catsmover_c
/*void TCATSMover::Dispose ()
{
	if (fGrid)
	{
		fGrid -> Dispose();
		delete fGrid;
		fGrid = nil;
	}
	
	DeleteTimeDep ();
	
		
	TCurrentMover::Dispose ();
}*/


CATSDialogNonPtrFields GetCATSDialogNonPtrFields(TCATSMover	* cm)
{
	CATSDialogNonPtrFields f;
	
	f.fUncertainStartTime  = cm->fUncertainStartTime; 	
	f.fDuration  = cm->fDuration; 
	//
	f.refP  = cm->refP; 	
	f.refZ  = cm->refZ; 	
	f.scaleType = cm->scaleType; 
	f.scaleValue = cm->scaleValue;
	strcpy(f.scaleOtherFile,cm->scaleOtherFile);
	f.refScale = cm->refScale; 
	f.bTimeFileActive = cm->bTimeFileActive; 
	f.bShowGrid = cm->bShowGrid; 
	f.bShowArrows = cm->bShowArrows; 
	f.arrowScale = cm->arrowScale; 
	f.fEddyDiffusion = cm->fEddyDiffusion; 
	f.fEddyV0 = cm->fEddyV0; 
	f.fDownCurUncertainty = cm->fDownCurUncertainty; 
	f.fUpCurUncertainty = cm->fUpCurUncertainty; 
	f.fRightCurUncertainty = cm->fRightCurUncertainty; 
	f.fLeftCurUncertainty = cm->fLeftCurUncertainty; 
	return f;
}

void SetCATSDialogNonPtrFields(TCATSMover	* cm,CATSDialogNonPtrFields * f)
{
	cm->fUncertainStartTime = f->fUncertainStartTime; 	
	cm->fDuration  = f->fDuration; 
	//
	cm->refP = f->refP; 	
	cm->refZ  = f->refZ; 	
	cm->scaleType = f->scaleType; 
	cm->scaleValue = f->scaleValue;
	strcpy(cm->scaleOtherFile,f->scaleOtherFile);
	cm->refScale = f->refScale; 
	cm->bTimeFileActive = f->bTimeFileActive; 
	cm->bShowGrid = f->bShowGrid; 
	cm->bShowArrows = f->bShowArrows; 
	cm->arrowScale = f->arrowScale; 
	cm->fEddyDiffusion = f->fEddyDiffusion;
	cm->fEddyV0 = f->fEddyV0;
	cm->fDownCurUncertainty = f->fDownCurUncertainty; 
	cm->fUpCurUncertainty = f->fUpCurUncertainty; 
	cm->fRightCurUncertainty = f->fRightCurUncertainty; 
	cm->fLeftCurUncertainty = f->fLeftCurUncertainty; 
}

///////////////////////////////////////////////////////////////////////////

void ShowUnscaledValue(DialogPtr dialog)
{
	double length;
	WorldPoint p;
	WorldPoint3D p3D = {0,0,0.};
	VelocityRec velocity;
	
	(void)EditTexts2LL(dialog, M16LATDEGREES, &p, FALSE);
	p3D.p = p;
	velocity = sharedCMover->GetPatValue(p3D);
	length = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	Float2EditText(dialog, M16UNSCALEDVALUE, length, 4);
}

Boolean TCATSMover::OkToAddToUniversalMap()
{
	// only allow this if we have grid with valid bounds
	WorldRect gridBounds;
	if (!fGrid) {
		printError("Error in TCATSMover::OkToAddToUniversalMap.");
		return false;
	}
	gridBounds = fGrid -> GetBounds();
	if(EqualWRects(gridBounds,emptyWorldRect)) {
		printError("You cannot create a universal mover from a current file which does not specify the grid's bounds.");
		return false;
	}
	return true;
}



OSErr TCATSMover::InitMover(TGridVel *grid, WorldPoint p)
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

	bApplyLogProfile = false;
	arrowDepth = 0.;	// if want to show subsurface velocity for log profile
	
	dynamic_cast<TCATSMover *>(this)->ComputeVelocityScale(model->GetModelTime());	// AH 07/10/2012
	
	return 0;
}

OSErr TCATSMover::ReplaceMover()
{
	OSErr err = 0;
	TCATSMover* mover = CreateAndInitCatsCurrentsMover (this -> moverMap,true,0,0); // only allow to replace with same type of mover
	if (mover)
	{
		// save original fields
		CATSDialogNonPtrFields fields = GetCATSDialogNonPtrFields(dynamic_cast<TCATSMover *>(this));
		SetCATSDialogNonPtrFields(mover,&fields);
		if(this->timeDep)
		{
			err = this->timeDep->MakeClone(&mover->timeDep);
			if (err) { delete mover; mover=0; return err; }
			// check if shio or hydrology, save ref point 
			//if (!(this->timeDep->GetFileType() == OSSMTIMEFILE)) 
			//mover->refP = this->refP;
			//mover->bTimeFileActive=true;
			// code goes here , should replace all the fields?
			//mover->scaleType = this->scaleType;
			//mover->scaleValue = this->scaleValue;
			//mover->refScale = this->refScale;
			//strcpy(mover->scaleOtherFile,this->scaleOtherFile);
		}
		if (err = this->moverMap->AddMover(mover,0))
		{mover->Dispose(); delete mover; mover = 0; return err;}
		if (err = this->moverMap->DropMover(dynamic_cast<TCATSMover *>(this)))
		{mover->Dispose(); delete mover; mover = 0; return err;}
	}
	else 
	{err = -1; return err;}
	
	model->NewDirtNotification();
	return err;
}

#define TCATSMoverREADWRITEVERSION 1 //JLM

OSErr TCATSMover::Write (BFPB *bfpb)
{
	char c;
	long version = TCATSMoverREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	OSErr err = 0;
	
	if (err = TCurrentMover::Write (bfpb)) return err;
	
	StartReadWriteSequence("TCatsMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	if (err = WriteMacValue(bfpb, refP.pLong)) return err;
	if (err = WriteMacValue(bfpb, refP.pLat)) return err;
	if (err = WriteMacValue(bfpb, refScale)) return err;
	if (err = WriteMacValue(bfpb, refZ)) return err;
	if (err = WriteMacValue(bfpb, scaleType)) return err;
	if (err = WriteMacValue(bfpb, scaleValue)) return err;
	if (err = WriteMacValue(bfpb, scaleOtherFile, sizeof(scaleOtherFile))) return err; // don't swap !! 
	
	if (err = WriteMacValue(bfpb,bRefPointOpen)) return err;
	if (err = WriteMacValue(bfpb,bUncertaintyPointOpen)) return err;
	if (err = WriteMacValue(bfpb,bTimeFileOpen)) return err;
	if (err = WriteMacValue(bfpb,bTimeFileActive)) return err;
	if (err = WriteMacValue(bfpb,bShowGrid)) return err;
	if (err = WriteMacValue(bfpb,bShowArrows)) return err;
	
	// JLM 9/2/98 
	if (err = WriteMacValue(bfpb,fEddyDiffusion)) return err;
	if (err = WriteMacValue(bfpb,fEddyV0)) return err;
	
	// bOptimizedForStep does not need to be saved to the save file
	
	if (err = WriteMacValue(bfpb,arrowScale)) return err;
	
	c = timeDep ? TRUE : FALSE;
	if (err = WriteMacValue(bfpb, c)) return err;
	if (timeDep) {
		id = timeDep -> GetClassID();
		if (err = WriteMacValue(bfpb, id)) return err;
		if (err = timeDep -> Write(bfpb)) return err;
	}
	
	id = fGrid -> GetClassID (); //JLM
	if (err = WriteMacValue(bfpb, id)) return err; //JLM
	err = fGrid -> Write (bfpb);
	
	return err;
}

OSErr TCATSMover::Read(BFPB *bfpb)
{
	char c;
	long version;
	ClassID id;
	OSErr err = 0;
	
	if (err = TCurrentMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("TCatsMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TCATSMover::Read()", "id != TYPE_CATSMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != TCATSMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	if (err = ReadMacValue(bfpb,&refP.pLong)) return err;
	if (err = ReadMacValue(bfpb,&refP.pLat)) return err;
	if (err = ReadMacValue(bfpb,&refScale)) return err;
	if (err = ReadMacValue(bfpb,&refZ)) return err;
	if (err = ReadMacValue(bfpb,&scaleType)) return err;
	if (err = ReadMacValue(bfpb,&scaleValue)) return err;
	if (err = ReadMacValue(bfpb, scaleOtherFile, sizeof(scaleOtherFile))) return err;  // don't swap !! 
	
	if (err = ReadMacValue(bfpb, &bRefPointOpen)) return err;
	if (err = ReadMacValue(bfpb, &bUncertaintyPointOpen)) return err;
	if (err = ReadMacValue(bfpb, &bTimeFileOpen)) return err;
	if (err = ReadMacValue(bfpb, &bTimeFileActive)) return err;
	if (err = ReadMacValue(bfpb, &bShowGrid)) return err;
	if (err = ReadMacValue(bfpb, &bShowArrows)) return err;
	
	// JLM 9/2/98 
	if (err = ReadMacValue(bfpb,&fEddyDiffusion)) return err;
	if (err = ReadMacValue(bfpb,&fEddyV0)) return err;
	
	// bOptimizedForStep does not need to be saved to the save file
	
	
	if (err = ReadMacValue(bfpb,&arrowScale)) return err;
	
	if (err = ReadMacValue(bfpb, &c)) return err;
	if (c) {
		if (err = ReadMacValue(bfpb,&id)) return err;
		switch (id) {
				//case TYPE_TIMEVALUES: timeDep = new TTimeValue(this); break;
			case TYPE_OSSMTIMEVALUES: timeDep = new TOSSMTimeValue(dynamic_cast<TCATSMover *>(this)); break;
			case TYPE_SHIOTIMEVALUES: timeDep = new TShioTimeValue(dynamic_cast<TCATSMover *>(this)); break;
			default: printError("Unrecognized time file type in TCATSMover::Read()."); return -1;
		}
		if (!timeDep)
		{ TechError("TCATSMover::Read()", "new TTimeValue()", 0); return -1; };
		if (err = timeDep -> InitTimeFunc()) return err;
		
		if (err = timeDep -> Read(bfpb)) return err;
	}
	else
		timeDep = nil;
	
	// read the type of grid used for the CATS mover
	if (err = ReadMacValue(bfpb,&id)) return err;
	// JLM if (err = ReadMacValue(bfpb,&version)) return err;
	// if (version != 1) { printSaveFileVersionError(); return -1; }
	switch(id)
	{
		case TYPE_RECTGRIDVEL: fGrid = new TRectGridVel;break;
		case TYPE_TRIGRIDVEL: fGrid = new TTriGridVel;break;
		case TYPE_TRIGRIDVEL3D: fGrid = new TTriGridVel3D;break;
		default: printError("Unrecognized Grid type in TCATSMover::Read()."); return -1;
	}
	
	fGrid -> Read (bfpb);
	
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr TCATSMover::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	char ourName[kMaxNameLen];
	OSErr err = 0;
	
	// see if the message is of concern to us
	this->GetClassName(ourName);
	
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		double val;
		char str[256];
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
		err = message->GetParameterAsWorldPoint("refP",&wp,false);
		if(!err) this->refP = wp;
		//////////////
		message->GetParameterString("timeFile",str,256);
		ResolvePath(str);
		if(str[0])
		{	// str contains the PATH descriptor, e.g. "resNum 10001"
			char shortFileName[32]=""; // not really used
			short unitsIfKnownInAdvance = kUndefined;
			char str2[64];
			Boolean haveScaleFactor = false;
			TOSSMTimeValue*  timeFile = 0;
			
			message->GetParameterString("speedUnits",str2,64);
			if(str2[0]) 
			{	
				unitsIfKnownInAdvance = StrToSpeedUnits(str2);
				if(unitsIfKnownInAdvance == kUndefined) 
					printError("bad speedUnits parameter");
			}
			else
			{
				err = message->GetParameterAsDouble("scaleFactor",&val);
				if(!err) 
				{
					// code goes here, if want to apply scale factor when reading in need to pass into CreateTOSSM...
					//this->timeDep->fScaleFactor = val; 	
					// need to set refScale if hydrology
					haveScaleFactor = true;
					unitsIfKnownInAdvance = -2;
				}
			}
			
			timeFile = CreateTOSSMTimeValue(dynamic_cast<TCATSMover *>(this),str,shortFileName,unitsIfKnownInAdvance);	
			dynamic_cast<TCATSMover *>(this)->DeleteTimeDep();
			if(timeFile) 
			{
				this -> timeDep = timeFile;
				if (haveScaleFactor) 
				{
					VelocityRec dummyValue;
					this -> timeDep -> fScaleFactor = val;
					if (err = timeFile->GetTimeValue(model->GetStartTime(),&dummyValue))	// make sure data is ok		// minus AH 07/10/2012
					
					dynamic_cast<TCATSMover *>(this)->DeleteTimeDep();
					if (this -> timeDep -> GetFileType()==SHIOHEIGHTSFILE || this -> timeDep -> GetFileType()==PROGRESSIVETIDEFILE) this -> timeDep -> RescaleTimeValues(1.0,val);	// does this apply value twice??
					if (this -> timeDep -> GetFileType()==HYDROLOGYFILE) this -> refScale = this -> timeDep -> fScaleFactor;
				}
				if (this -> timeDep) this -> bTimeFileActive = true; // assume we want to make it active
			}
		}
		//////////////
		/////////////
		dynamic_cast<TCATSMover *>(this)->ComputeVelocityScale(model->GetModelTime());	// AH 07/10/2012
		model->NewDirtNotification();// tell model about dirt
	}
	
	long messageCode = message->GetMessageCode();
	switch(messageCode)
	{
		case M_UPDATEVALUES:
			VelocityRec dummyValue;
			// new data, make sure it is ok
			if (timeDep) 
			 	//err = timeDep->GetTimeValue(model->GetStartTime(),&dummyValue);	// minus AH 07/10/2012
				err = timeDep->GetTimeValue(model->GetModelTime(),&dummyValue);	// changed shio to compute only a few days, this caused a flash once run got past 3 days
			if (err) dynamic_cast<TCATSMover *>(this)->DeleteTimeDep();
			break;
	}
	
	/////////////////////////////////////////////////
	// sub-guys need us to pass this message 
	/////////////////////////////////////////////////
	if(this->timeDep) err = this->timeDep->CheckAndPassOnMessage(message);
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TCurrentMover::CheckAndPassOnMessage(message);
}

/////////////////////////////////////////////////
long TCATSMover::GetListLength()
{
	long count = 1;
	
	if (bOpen) {
		count += 4;		// minimum CATS mover lines
		if (gNoaaVersion) count++;	// apply log profile
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

ListItem TCATSMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char *p, latS[20], longS[20], valStr[32];
	ListItem item = { this, 0, indent, 0 };
	
	if (n == 0) {
		item.index = I_CATSNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		//		sprintf(text, "CATS: \"%s\"", className);
		sprintf(text, "Currents: \"%s\"", className);
		*style = bActive ? italic : normal;
		
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
		
		if (gNoaaVersion)
		{
		if (--n == 0) {
			item.index = I_CATSLOGPROFILE;
			item.bullet = bApplyLogProfile ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			//StringWithoutTrailingZeros(valStr,arrowScale,6);
			sprintf(text, "Apply log profile");
			
			return item;
		}
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
						if (timeDep && timeDep->GetFileType()==HYDROLOGYFILE)
							StringWithoutTrailingZeros(valStr,refScale,6);
						//StringWithoutTrailingZeros(valStr,timeDep->fScaleFactor,6);
						else
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
					sprintf(text, "Start Time: %.2f hours",((double)fUncertainStartTime)/3600.);
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

Boolean TCATSMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
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
			case I_CATSLOGPROFILE:
				bApplyLogProfile = !bApplyLogProfile; return TRUE;
		}
	
	if (ShiftKeyDown() && item.index == I_CATSNAME) {
		fColor = MyPickColor(fColor,mapWindow);
		model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT);
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
						dynamic_cast<TCATSMover *>(this)->UpdateUncertaintyValues(model->GetModelTime()-model->GetStartTime());
					}
				}
				return TRUE;
				break;
			}
			default:
				CATSSettingsDialog (dynamic_cast<TCATSMover *>(this), this -> moverMap, &timeFileChanged);
				return TRUE;
				break;
		}
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean TCATSMover::FunctionEnabled(ListItem item, short buttonID)
{
	long i,n,j,num;
	//TMover* mover,mover2;
	switch (item.index) {
		case I_CATSNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					// need a way to check if mover is part of a Compound Mover - thinks it's just a currentmover
					
					if (bIAmPartOfACompoundMover)
						return TCurrentMover::FunctionEnabled(item, buttonID);
					
					/*for (i = 0, n = moverMap->moverList->GetItemCount() ; i < n ; i++) {
					 moverMap->moverList->GetListItem((Ptr)&mover, i);
					 if (mover->IAm(TYPE_COMPOUNDMOVER))
					 {
					 for (j = 0, num = ((TCompoundMover*)mover)->moverList->GetItemCount() ;  j < num; j++)
					 {
					 ((TCompoundMover*)mover)->moverList->GetListItem((Ptr)&mover2, j);
					 if (!(((TCompoundMover*)mover)->moverList->IsItemInList((Ptr)&item.owner, &j))) 
					 //return FALSE;
					 continue;
					 else
					 {
					 switch (buttonID) {
					 case UPBUTTON: return j > 0;
					 case DOWNBUTTON: return j < (((TCompoundMover*)mover2)->moverList->GetItemCount()-1);
					 }
					 }
					 }
					 }
					 }*/
					
					if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
					}
					break;
			}
	}
	
	if (buttonID == SETTINGSBUTTON) return TRUE;
	
	return TCurrentMover::FunctionEnabled(item, buttonID);
}


/*OSErr TCATSMover::UpItem(ListItem item)
 {	
 long i;
 OSErr err = 0;
 
 if (item.index == I_CATSNAME)
 if (model->LESetsList->IsItemInList((Ptr)&item.owner, &i))
 //if (i > 0) {// 2 for each
 if (i > 1) {// 2 for each
 //if (err = model->LESetsList->SwapItems(i, i - 1))
 if ((err = model->LESetsList->SwapItems(i, i - 2)) || (err = model->LESetsList->SwapItems(i+1, i - 1)))
 { TechError("TCATSMover::UpItem()", "model->LESetsList->SwapItems()", err); return err; }
 SelectListItem(item);
 UpdateListLength(true);
 InvalidateMapImage();
 InvalMapDrawingRect();
 }
 
 return 0;
 }
 
 OSErr TCATSMover::DownItem(ListItem item)
 {
 long i;
 OSErr err = 0;
 
 if (item.index == I_CATSNAME)
 if (model->LESetsList->IsItemInList((Ptr)&item.owner, &i))
 //if (i < (model->LESetsList->GetItemCount() - 1)) {
 if (i < (model->LESetsList->GetItemCount() - 3)) {
 //if (err = model->LESetsList->SwapItems(i, i + 1))
 if ((err = model->LESetsList->SwapItems(i, i + 2)) || (err = model->LESetsList->SwapItems(i+1, i + 3)))
 { TechError("TCATSMover::UpItem()", "model->LESetsList->SwapItems()", err); return err; }
 SelectListItem(item);
 UpdateListLength(true);
 InvalidateMapImage();
 InvalMapDrawingRect();
 }
 
 return 0;
 }*/

OSErr TCATSMover::SettingsItem(ListItem item)
{
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = this -> ListClick(item,inBullet,doubleClick);
	return 0;
}

TOSSMTimeValue *sTimeValue;

static PopInfoRec HydrologyPopTable[] = {
	{ M32, nil, M32INFOTYPEPOPUP, 0, pHYDROLOGYINFO, 0, 1, FALSE, nil },
	{ M32, nil, M32TRANSPORT1UNITS, 0, pTRANSPORTUNITS, 0, 1, FALSE, nil },
	//{ M32, nil, M32TRANSPORT2UNITS, 0, pSPEEDUNITS, 0, 1, FALSE, nil },
	{ M32, nil, M32VELOCITYUNITS, 0, pSPEEDUNITS2, 0, 1, FALSE, nil }
};

double ConvertTransportUnitsToCMS(short transportUnits)
{
	double conversionFactor = 1.;
	switch(transportUnits)
	{
		case 1: conversionFactor = 1.0; break;	// CMS
		case 2: conversionFactor = 1000.; break;	// KCMS
		case 3: conversionFactor = .3048*.3048*.3048; break;	// CFS
		case 4: conversionFactor = .3048*.3048*.3048 * 1000.; break; // KCFS
			//default: err = -1; goto done;
	}
	return conversionFactor;
}

void ShowHideHydrologyDialogItems(DialogPtr dialog)
{
	Boolean showTransport1Items, showTransport2Items;
	short typeOfInfoSpecified = GetPopSelection(dialog, M32INFOTYPEPOPUP);
	
	switch (typeOfInfoSpecified)
	{
		default:
			//case HAVETRANSPORT:
		case 1:
			showTransport1Items=TRUE;
			showTransport2Items=FALSE;
			break;
			//case HAVETRANSPORTANDVELOCITY:
		case 2:
			showTransport2Items=TRUE;
			showTransport1Items=FALSE;
			break;
	}
	ShowHideDialogItem(dialog, M32TRANSPORT1LABELA, showTransport1Items ); 
	ShowHideDialogItem(dialog, M32TRANSPORT1, true); 
	ShowHideDialogItem(dialog, M32TRANSPORT1UNITS, true); 
	ShowHideDialogItem(dialog, M32TRANSPORT1LABELB, showTransport1Items); 
	
	ShowHideDialogItem(dialog, M32TRANSPORT2LABELA, showTransport2Items); 
	//ShowHideDialogItem(dialog, M32TRANSPORT2, showTransport2Items); 
	//ShowHideDialogItem(dialog, M32TRANSPORT2UNITS, showTransport2Items); 
	ShowHideDialogItem(dialog, M32TRANSPORT2LABELB, showTransport2Items); 
	ShowHideDialogItem(dialog, M32VELOCITY, showTransport2Items); 
	ShowHideDialogItem(dialog, M32VELOCITYUNITS, showTransport2Items); 
}

short HydrologyClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
#pragma unused (data)
	
	VelocityRec refVel;
	long menuID_menuItem;
	switch(itemNum)
	{
		case M32OK:
		{						
			double userVelocity = 0, userTransport = 0, transportConversionFactor = 1, scaleFactor = 0, conversionFactor/*, origScaleFactor*/;
			short typeOfInfoSpecified = GetPopSelection(dialog, M32INFOTYPEPOPUP);
			short transportUnits = GetPopSelection(dialog, M32TRANSPORT1UNITS);
			WorldPoint3D refPoint3D = {0,0,0.};
			// code goes here, calculations to determine scaling factor based on inputs
			
			userTransport = EditText2Float(dialog, M32TRANSPORT1);
			if (userTransport == 0)
			{
				printError("You must enter a value for the transport");
				break;
			}
			if (typeOfInfoSpecified == 2)
			{
				userVelocity = EditText2Float(dialog, M32VELOCITY);
				if (userVelocity == 0)
				{
					printError("You must enter a value for the velocity");
					break;
				}
			}
			//sTimeValue->fUserUnits is the file units
			transportConversionFactor = ConvertTransportUnitsToCMS(sTimeValue->fUserUnits) / ConvertTransportUnitsToCMS(transportUnits);
			// get value at reference point and calculate scale factor
			// need units conversion for transport and velocity
			refPoint3D.p = sTimeValue->fStationPosition;
			//refVel = ((TCATSMover*)(TTimeValue*)sTimeValue->owner)->GetPatValue(sTimeValue->fStationPosition);
			refVel = ((TCATSMover*)(TTimeValue*)sTimeValue->owner)->GetPatValue(refPoint3D);
			//origScaleFactor = sTimeValue->fScaleFactor;
			if (typeOfInfoSpecified == 1)
			{
				//scaleFactor = 1./ (userTransport * transportConversionFactor);
				scaleFactor = transportConversionFactor / userTransport;
				//sTimeValue->fScaleFactor = 1./ (userTransport * transportConversionFactor);
			}
			else if (typeOfInfoSpecified == 2)
			{
				double refSpeed = sqrt(refVel.u*refVel.u + refVel.v*refVel.v);
				short velUnits = GetPopSelection(dialog, M32VELOCITYUNITS);
				switch(velUnits)
				{
					case kKnots: conversionFactor = KNOTSTOMETERSPERSEC; break;
					case kMilesPerHour: conversionFactor = MILESTOMETERSPERSEC; break;
					case kMetersPerSec: conversionFactor = 1.0; break;
						//default: err = -1; goto done;
				}
				if (refSpeed > 1e-6) // any error if not? default = 0? 1? ...
					scaleFactor = (userVelocity * conversionFactor)/(userTransport * transportConversionFactor * refSpeed); // maybe an error if refSpeed too small
				//sTimeValue->fScaleFactor = (userVelocity * conversionFactor)/(userTransport * transportConversionFactor * refSpeed); // maybe an error if refSpeed too small
			}
			//sTimeValue->RescaleTimeValues(origScaleFactor, scaleFactor);
			sTimeValue->fScaleFactor = scaleFactor;
			//sTimeValue->fTransport = userTransport * transportConversionFactor;
			sTimeValue->fTransport = userTransport * ConvertTransportUnitsToCMS(transportUnits);
			sTimeValue->fVelAtRefPt = userVelocity * conversionFactor;
			
			return M32OK;
		}
		case M32CANCEL:
			return M32CANCEL;
			break;
			
		case M32TRANSPORT1:
			//case M32TRANSPORT2:
		case M32VELOCITY:		
			CheckNumberTextItem(dialog, itemNum, TRUE); //  allow decimals
			break;
			
		case M32INFOTYPEPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowHideHydrologyDialogItems(dialog);
			break;
			
		case M32TRANSPORT1UNITS:
			//case M32TRANSPORT2UNITS:
		case M32VELOCITYUNITS:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
	}
	return 0;
}

OSErr HydrologyInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused (data)
	char roundLat,roundLong;
	char posStr[64], latStr[64], longStr[64], unitStr[64];
	
	SetDialogItemHandle(dialog, M32HILITEDEFAULT, (Handle)FrameDefault);
	//SetDialogItemHandle(dialog, M32FROST, (Handle)FrameEmbossed);
	
	//RegisterPopTable (HydrologyPopTable, sizeof (HydrologyPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (M32, dialog);
	
	SetPopSelection (dialog, M32INFOTYPEPOPUP, 1);
	
	SetPopSelection (dialog, M32TRANSPORT1UNITS, 1);
	//SetPopSelection (dialog, M32TRANSPORT2UNITS, 1);
	SetPopSelection (dialog, M32VELOCITYUNITS, 1);
	
	ShowHideHydrologyDialogItems(dialog);
	
	mysetitext(dialog, M32FILENAME, sTimeValue->fStationName);
	settings.latLongFormat = DEGREES;
	WorldPointToStrings2(sTimeValue->fStationPosition, latStr, &roundLat, longStr, &roundLong);	
	SimplifyLLString(longStr, 3, roundLong);
	SimplifyLLString(latStr, 3, roundLat);
	sprintf(posStr, "%s, %s", latStr,longStr);
	mysetitext(dialog, M32POSITION, posStr);
	
	ConvertToTransportUnits(sTimeValue->fUserUnits,unitStr);
	mysetitext(dialog, M32UNITS, unitStr);
	MySelectDialogItemText(dialog, M32TRANSPORT1, 0, 255);
	
	return 0;
}

OSErr HydrologyDialog(TOSSMTimeValue *dialogTimeFileData, WindowPtr parentWindow)
{
	short item;
	PopTableInfo saveTable = SavePopTable();
	short j, numItems = 0;
	PopInfoRec combinedDialogsPopTable[20];
	
	if(parentWindow == nil) parentWindow = mapWindow; // we need the parent on the IBM
	sTimeValue = dialogTimeFileData;
	
	// code to allow a dialog on top of another with pops
	for(j = 0; j < sizeof(HydrologyPopTable) / sizeof(PopInfoRec);j++)
		combinedDialogsPopTable[numItems++] = HydrologyPopTable[j];
	for(j= 0; j < saveTable.numPopUps ; j++)
		combinedDialogsPopTable[numItems++] = saveTable.popTable[j];
	
	RegisterPopTable(combinedDialogsPopTable,numItems);
	
	item = MyModalDialog(M32, parentWindow, 0, HydrologyInit, HydrologyClick);
	RestorePopTableInfo(saveTable);
	if (item == M32OK) {
		dialogTimeFileData = sTimeValue;
		if(parentWindow == mapWindow) {
			model->NewDirtNotification(); // when a dialog is the parent, we rely on that dialog to notify about Dirt 
			// that way we don't get the map redrawing behind the parent dialog on the IBM
		}
	}
	if (item == M32CANCEL) {return USERCANCEL;}
	return item == M32OK? 0 : -1;
}

OSErr TCATSMover::DeleteItem(ListItem item)
{
	if (item.index == I_CATSNAME)
		return moverMap -> DropMover(dynamic_cast<TCATSMover *>(this));
	
	return 0;
}

void DisposeDialogTimeDep(void)
{
	if (sharedCMDialogTimeDep) {
		if(sharedCMDialogTimeDep != sharedCMover->timeDep)
		{	// only dispose of this if it is different
			sharedCMDialogTimeDep->Dispose();
			delete sharedCMDialogTimeDep;
		}
		sharedCMDialogTimeDep = nil;
	}
}

void ShowCatsDialogUnitLabels(DialogPtr dialog)
{
	char scaleToUnitsStr[64] = "";
	char fileUnitsStr[64] = "";
	double scaleFactor,transport,velAtRefPt;
	short fileType;
	
	if (GetPopSelection (dialog, M16TIMEFILETYPES) == NOTIMEFILE) // scaling to a velocity
		strcpy(scaleToUnitsStr,"m/s at reference point");
	else // scaling to multiple of the file value
		strcpy(scaleToUnitsStr,"* file value at reference point");
	mysetitext(dialog, M16SCALETOVALUEUNITS, scaleToUnitsStr);
	
	if(sharedCMDialogTimeDep) 
	{
		scaleFactor = sharedCMDialogTimeDep->fScaleFactor;
		transport = sharedCMDialogTimeDep->fTransport;
		velAtRefPt = sharedCMDialogTimeDep->fVelAtRefPt;
		fileType = sharedCMDialogTimeDep->GetFileType();
		if (fileType == SHIOHEIGHTSFILE || fileType == PROGRESSIVETIDEFILE)
		{
			Float2EditText(dialog, M16TIMEFILESCALEFACTOR, scaleFactor, 4);
		}
		else if (fileType == HYDROLOGYFILE)
		{
			Float2EditText(dialog, M16HYDROLOGYSCALEFACTOR, scaleFactor, 4);
			Float2EditText(dialog, M16TRANSPORT, transport, 4);
			Float2EditText(dialog, M16VELATREFPT, velAtRefPt, 4);
			
		}
		else 
		{
			ConvertToUnitsShort (sharedCMDialogTimeDep->GetUserUnits(), fileUnitsStr);
			mysetitext(dialog, M16TIMEFILEUNITS, fileUnitsStr);
		}
	}
}

void ShowHideCATSDialogItems(DialogPtr dialog)
{
	short fileType = OSSMTIMEFILE;
	
	if ((sharedCMover->fGrid->GetClassID()==TYPE_TRIGRIDVEL))	// only allow for catsmovers on triangle grid for now
		//if (!(sharedCMover->IAm(TYPE_GRIDCURMOVER)))
		ShowHideDialogItem(dialog, M16REPLACEMOVER, true); 
	else 
		ShowHideDialogItem(dialog, M16REPLACEMOVER, false); 
	
	if(!sharedCMDialogTimeDep)
	{
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTOR, false); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTOR, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITSLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITS, false); 
		ShowHideDialogItem(dialog, M16TIMEFILENAMELABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILENAME, false); 
		ShowHideDialogItem(dialog, M16TRANSPORTLABEL, false); 
		ShowHideDialogItem(dialog, M16TRANSPORT, false); 
		ShowHideDialogItem(dialog, M16TRANSPORTUNITS, false); 
		ShowHideDialogItem(dialog, M16VELLABEL, false); 
		ShowHideDialogItem(dialog, M16VELATREFPT, false); 
		ShowHideDialogItem(dialog, M16VELUNITS, false); 
		return;
	}
	else
	{
		ShowHideDialogItem(dialog, M16TIMEFILENAMELABEL, true); 
		ShowHideDialogItem(dialog, M16TIMEFILENAME, true); 	
	}
	
	if(sharedCMDialogTimeDep)
		fileType = sharedCMDialogTimeDep->GetFileType();
	
	if(fileType==SHIOHEIGHTSFILE || fileType==PROGRESSIVETIDEFILE)
	{
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTORLABEL, true); 
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTOR, true); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTOR, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITSLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITS, false); 
		ShowHideDialogItem(dialog, M16TRANSPORTLABEL, false); 
		ShowHideDialogItem(dialog, M16TRANSPORT, false); 
		ShowHideDialogItem(dialog, M16TRANSPORTUNITS, false); 
		ShowHideDialogItem(dialog, M16VELLABEL, false); 
		ShowHideDialogItem(dialog, M16VELATREFPT, false); 
		ShowHideDialogItem(dialog, M16VELUNITS, false); 
	}
	else if(fileType==HYDROLOGYFILE)
	{
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTOR, false); 
		if(sharedCMDialogTimeDep->bOSSMStyle)
		{
			ShowHideDialogItem(dialog, M16TRANSPORTLABEL, false); 
			ShowHideDialogItem(dialog, M16TRANSPORT, false); 
			ShowHideDialogItem(dialog, M16TRANSPORTUNITS, false); 
			ShowHideDialogItem(dialog, M16VELLABEL, false); 
			ShowHideDialogItem(dialog, M16VELATREFPT, false); 
			ShowHideDialogItem(dialog, M16VELUNITS, false); 
			ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTORLABEL, true); 
			ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTOR, true); 
		}
		else
		{
			ShowHideDialogItem(dialog, M16TRANSPORTLABEL, true); 
			ShowHideDialogItem(dialog, M16TRANSPORT, true); 
			ShowHideDialogItem(dialog, M16TRANSPORTUNITS, true); 
			if(sharedCMDialogTimeDep->fVelAtRefPt!=0)
			{
				ShowHideDialogItem(dialog, M16VELLABEL, true); 
				ShowHideDialogItem(dialog, M16VELATREFPT, true);
				ShowHideDialogItem(dialog, M16VELUNITS, true); 
			}
			else
			{
				ShowHideDialogItem(dialog, M16VELLABEL, false); 
				ShowHideDialogItem(dialog, M16VELATREFPT, false); 
				ShowHideDialogItem(dialog, M16VELUNITS, false); 
			}
			
			ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTORLABEL, false); 
			ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTOR, false); 
		}
		ShowHideDialogItem(dialog, M16TIMEFILEUNITSLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITS, false); 
	}
	else 
	{
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTOR, false); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTOR, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITSLABEL, true); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITS, true); 
		ShowHideDialogItem(dialog, M16TRANSPORTLABEL, false); 
		ShowHideDialogItem(dialog, M16TRANSPORT, false); 
		ShowHideDialogItem(dialog, M16TRANSPORTUNITS, false); 
		ShowHideDialogItem(dialog, M16VELLABEL, false); 
		ShowHideDialogItem(dialog, M16VELATREFPT, false); 
		ShowHideDialogItem(dialog, M16VELUNITS, false); 
	}
}

void ShowHideScaleFactorItems(DialogPtr dialog)
{
	short fileType = OSSMTIMEFILE;
	
	if(sharedCMDialogTimeDep)
		fileType = sharedCMDialogTimeDep->GetFileType();
	
	if (sharedCMDialogTimeDep && fileType == HYDROLOGYFILE)
	{
		ShowHideDialogItem(dialog, M16NOSCALING, false); 
		ShowHideDialogItem(dialog, M16SCALETOCONSTANT, false); 
		ShowHideDialogItem(dialog, M16SCALEVALUE, false); 
		ShowHideDialogItem(dialog, M16SCALETOGRID, false); 
		ShowHideDialogItem(dialog, M16SCALEGRIDNAME, false); 
		ShowHideDialogItem(dialog, M16SCALETOVALUEUNITS, false);
	}
	else
	{
		ShowHideDialogItem(dialog, M16NOSCALING, true); 
		ShowHideDialogItem(dialog, M16SCALETOCONSTANT, true); 
		ShowHideDialogItem(dialog, M16SCALEVALUE, true); 
		ShowHideDialogItem(dialog, M16SCALETOGRID, true); 
		ShowHideDialogItem(dialog, M16SCALEGRIDNAME, true); 
		ShowHideDialogItem(dialog, M16SCALETOVALUEUNITS, true); 
	}
	
}

short CATSClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	Boolean changed;
	short item;
	WorldPoint p;
	TOSSMTimeValue *timeFile;
	OSErr err = 0;
	double scaleValue;
	
	StandardLLClick(dialog, itemNum, M16LATDEGREES, M16DEGREES, &p, &changed);
	ShowUnscaledValue(dialog);
	
	switch (itemNum) {
		case M16OK:
		{
			// this is tricky , we have saved the NonPtrFields so we are free to mess with them since
			// they get restored if the user cancels.
			// We just have to be careful not to change sharedCMover -> timeDep.
			// To accomplish this we use sharedCMDialogTimeDep until the point of no return.
			///////////////////
			if (GetButton(dialog, M16NOSCALING)) sharedCMover->scaleType = SCALE_NONE;
			if (GetButton(dialog, M16SCALETOCONSTANT)) sharedCMover->scaleType = SCALE_CONSTANT;
			if (GetButton(dialog, M16SCALETOGRID)) sharedCMover->scaleType = SCALE_OTHERGRID;
			sharedCMover->scaleValue = EditText2Float(dialog, M16SCALEVALUE);
			mygetitext(dialog, M16SCALEGRIDNAME, sharedCMover->scaleOtherFile, 31);
			
			err = EditTexts2LL(dialog, M16LATDEGREES, &sharedCMover->refP,TRUE);
			if(err) break;
			
			if(!(sharedCMDialogTimeDep && (sharedCMDialogTimeDep->GetFileType() == HYDROLOGYFILE) && sharedCMDialogTimeDep->bOSSMStyle))
			{	// old OSSM style files may have refP on land, but this point is not used in calculation
				
				err = sharedCMover->ComputeVelocityScale(model->GetModelTime());	// AH 07/10/2012
				
				if(err) 
				{	// restore values and report error to user
					printError("The unscaled value is too small at the chosen reference point.");
					break;
				}
			}
			if(sharedCMDialogTimeDep && (sharedCMDialogTimeDep->GetFileType() == SHIOHEIGHTSFILE || sharedCMDialogTimeDep->GetFileType() == PROGRESSIVETIDEFILE || 
										 (sharedCMDialogTimeDep->GetFileType() == HYDROLOGYFILE && sharedCMDialogTimeDep->bOSSMStyle)) )
			{
				double newScaleFactor = EditText2Float(dialog, (sharedCMDialogTimeDep->GetFileType() == 
																SHIOHEIGHTSFILE || sharedCMDialogTimeDep->GetFileType() ==PROGRESSIVETIDEFILE) ? M16TIMEFILESCALEFACTOR : M16HYDROLOGYSCALEFACTOR);
				if (newScaleFactor == 0)
				{
					printError("The scale factor must be positive.");
					return 0;
				}
			}
			////////////////////
			// point of no return
			///////////////////
			sharedCMover->bActive = GetButton(dialog, M16ACTIVE);
			
			// deal with the timeDep guy, JLM /11/25/98
			///////////////
			if(sharedCMDialogTimeDep != sharedCMover -> timeDep)
			{
				if(sharedCMover -> timeDep)
				{	// dispose of the one we are replacing
					sharedCMover -> timeDep->Dispose();
					delete sharedCMover -> timeDep;
				}
				sharedCMover -> timeDep = sharedCMDialogTimeDep;
				sharedCMChangedTimeFile = TRUE; 
			}
			if(sharedCMover->timeDep && (sharedCMover->timeDep->GetFileType() == SHIOHEIGHTSFILE || sharedCMover->timeDep->GetFileType() == PROGRESSIVETIDEFILE))
			{
				double newScaleFactor = EditText2Float(dialog, M16TIMEFILESCALEFACTOR);
				sharedCMover->timeDep->fScaleFactor = newScaleFactor;
			}
			if(sharedCMover->timeDep && sharedCMover->timeDep->GetFileType() == HYDROLOGYFILE)
			{
				if (sharedCMover->timeDep->bOSSMStyle)
				{
					double newScaleFactor = EditText2Float(dialog, M16HYDROLOGYSCALEFACTOR);
					//sharedCMover->timeDep->RescaleTimeValues(sharedCMover->timeDep->fScaleFactor, newScaleFactor);
					sharedCMover->timeDep->fScaleFactor = newScaleFactor; // code goes here, also refScale, scaleValue
					sharedCMover->refScale = newScaleFactor; // code goes here, also refScale, scaleValue
				}
				else 
					sharedCMover->refScale = sharedCMover->timeDep->fScaleFactor;	// redundant ?
			}
			sharedCMover -> bTimeFileActive = (sharedCMover -> timeDep != 0); // active if we have one
			//err = sharedCMover->ComputeVelocityScale();	// need to set refScale if hydrology
			DisposeDialogTimeDep();
			////////////////////
			
			sharedCMover->bShowArrows = GetButton(dialog, M16SHOWARROWS);
			sharedCMover->arrowScale = EditText2Float(dialog, M16ARROWSCALE);
			
			if (!sharedCMover->CurrentUncertaintySame(sSharedCATSUncertainyInfo))
			{
				sharedCMover -> SetCurrentUncertaintyInfo(sSharedCATSUncertainyInfo);
				sharedCMover->UpdateUncertaintyValues(model->GetModelTime()-model->GetStartTime());
			}
			return M16OK;
		}
			
		case M16CANCEL: 
			DisposeDialogTimeDep();
			SetCATSDialogNonPtrFields(sharedCMover,&sharedCatsDialogNonPtrFields);
			return M16CANCEL;
			
		case M16ACTIVE:
		case M16SHOWARROWS:
			ToggleButton(dialog, itemNum);
			break;
			
		case M16SETUNCERTAINTY:
		{
			Boolean userCanceledOrErr, uncertaintyValuesChanged=false;
			//CurrentUncertainyInfo info  = sharedCMover -> GetCurrentUncertaintyInfo();
			CurrentUncertainyInfo info  = sSharedCATSUncertainyInfo;
			userCanceledOrErr = CurrentUncertaintyDialog(&info,GetDialogWindow(dialog),&uncertaintyValuesChanged);
			if(!userCanceledOrErr) 
			{
				if (uncertaintyValuesChanged)
				{
					sSharedCATSUncertainyInfo = info;
					//sharedCMover->SetCurrentUncertaintyInfo(info);	// only want to update uncertainty if something has been changed and ok button hit
				}
			}
			break;
		}
			
			
		case M16NOSCALING:
		case M16SCALETOCONSTANT:
		case M16SCALETOGRID:
			SetButton(dialog, M16NOSCALING, itemNum == M16NOSCALING);
			SetButton(dialog, M16SCALETOCONSTANT, itemNum == M16SCALETOCONSTANT);
			SetButton(dialog, M16SCALETOGRID, itemNum == M16SCALETOGRID);
			if (itemNum == M16SCALETOGRID) {
				char classNameOfSelectedGrid[32];
				ActivateParentDialog(FALSE);
				mygetitext(dialog, M16SCALEGRIDNAME, classNameOfSelectedGrid, 31);
				item = ChooseOtherGridDialog(sharedCMover,classNameOfSelectedGrid);
				ActivateParentDialog(TRUE);
				if (item == M17OK)
					mysetitext(dialog, M16SCALEGRIDNAME, classNameOfSelectedGrid);
				else {
					SetButton(dialog, M16SCALETOGRID, FALSE);
					SetButton(dialog, M16NOSCALING, TRUE);
				}
			}
			break;
			
		case M16SCALEVALUE:
			CheckNumberTextItemAllowingNegative(dialog, itemNum, TRUE);
			CATSClick(dialog, M16SCALETOCONSTANT, 0, 0);
			break;
			
		case M16ARROWSCALE:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
			
		case M16TIMEFILETYPES:
			
			short	theType;
			long	menuID_menuItem;
			PopClick(dialog, itemNum, &menuID_menuItem);
			theType = GetPopSelection (dialog, M16TIMEFILETYPES);
			if (theType==1)
			{
				// selected No Time Series
				if (sharedCMDialogTimeDep == nil) break;	// already selected
				if (!CHOICEALERT(M79, 0, TRUE)) goto donetimefile;/*break;*/		// user canceled
				DisposeDialogTimeDep();	
			}
			else 
			{
				short flag = kUndefined;
				if (theType==PROGRESSIVETIDEFILE) flag = kFudgeFlag;
				if (sharedCMDialogTimeDep && !CHOICEALERT(M79, 0, TRUE)) goto donetimefile;/*break*/;	// user canceled
				// code goes here, need to know what type of file it is to decide if user wants standing wave or progressive wave...
				timeFile = LoadTOSSMTimeValue(sharedCMover,flag); 
				// if user chose to cancel?
				if(!timeFile) goto donetimefile;/*break*/; // user canceled or an error 
				
				if (timeFile->GetFileType() != theType)	// file type doesn't match selected popup item
				{
					char fileTypeName[64], msg[128], fileTypeName2[64];
					
					switch(theType)
					{
						case OSSMTIMEFILE:
							strcpy(fileTypeName, "Tidal Current Time Series ");
							break;
						case SHIOCURRENTSFILE:
							strcpy(fileTypeName, "Shio Currents Coefficients ");
							break;
						case SHIOHEIGHTSFILE:
							strcpy(fileTypeName, "Shio Heights Coefficients ");
							break;
						case HYDROLOGYFILE:
							strcpy(fileTypeName, "Hydrology Time Series ");
							break; 
						case PROGRESSIVETIDEFILE:
							strcpy(fileTypeName, "Progressive Wave Coefficients ");
							break;
					}
					switch(timeFile->GetFileType())
					{
						case OSSMTIMEFILE:
							strcpy(fileTypeName2, "Tidal Current Time Series ");
							break;
						case SHIOCURRENTSFILE:
							strcpy(fileTypeName2, "Shio Currents Coefficients ");
							break;
						case SHIOHEIGHTSFILE:
							strcpy(fileTypeName2, "Shio Heights Coefficients ");
							break; 
						case HYDROLOGYFILE:
							strcpy(fileTypeName2, "Hydrology Time Series ");
							break;
						case PROGRESSIVETIDEFILE:
							strcpy(fileTypeName2, "Progressive Wave Coefficients ");
							break;
					}
					//if (timeFile->GetFileType()==SHIOHEIGHTSFILE && theType==PROGRESSIVETIDEFILE)
					if (timeFile->GetFileType()==PROGRESSIVETIDEFILE && theType==SHIOHEIGHTSFILE)
					{	// let it go, backwards for some reason...
						sprintf(msg,"The selected shio heights file will be treated as a progressive wave");
						//printNote(msg);
						//timeFile->SetFileType(PROGRESSIVETIDEFILE);
						timeFile->SetFileType(SHIOHEIGHTSFILE);
					}
					else
					{
						sprintf(msg,"The selected file is a %s file not a %s file. ",fileTypeName2,fileTypeName);
						printError(msg);
						goto donetimefile;
						break;
					}
				}
				if (timeFile->GetFileType() == HYDROLOGYFILE)	// bring up hydrology popup, unless file is in old OSSM format
				{
					Boolean userCanceledOrErr = false;
					{
						WorldRect gridBounds;
						char msg[256], latS[20], longS[20];
						WorldPointToStrings(timeFile->fStationPosition, latS, longS);
						if(sharedCMover -> fGrid == 0)
						{ printError("Programmer error: sharedCMover -> fGrid is nil"); break;}
						gridBounds = sharedCMover -> fGrid -> GetBounds();
						
						if(!WPointInWRect(timeFile->fStationPosition.pLong,timeFile->fStationPosition.pLat,&gridBounds))
						{
							sprintf(msg,"Check that this is the right file.%sThe reference point in this file is not within the grid bounds.%sLat: %s%sLng: %s",NEWLINESTRING,NEWLINESTRING,latS,NEWLINESTRING,longS);
							printWarning(msg);
							goto donetimefile;
						}
					}
					
					if (!timeFile->bOSSMStyle) userCanceledOrErr = HydrologyDialog(timeFile,GetDialogWindow(dialog));
					//RegisterPopTable (csPopTable, sizeof (csPopTable) / sizeof (PopInfoRec));
					if(userCanceledOrErr)
					{	// leave in previous state
						goto donetimefile;
					}
					else
					{
						sharedCMover->refP = timeFile->fStationPosition;
						if (!timeFile->bOSSMStyle) 
						{
							sharedCMover->refScale = timeFile->fScaleFactor;
							//sharedCMover->scaleValue = timeFile->fScaleFactor;
							Float2EditText(dialog, M16SCALEVALUE, timeFile->fScaleFactor, 4);
						}
						SwitchLLFormat(dialog, M16LATDEGREES, M16DEGREES);
						LL2EditTexts(dialog, M16LATDEGREES, &sharedCMover->refP);
						(void)CATSClick(dialog,M16SCALETOCONSTANT,lParam,data);
					}
				}
				// JLM 7/13/99, CJ would like the interface to set the ref pt
				if(timeFile->GetClassID () == TYPE_SHIOTIMEVALUES)
				{	// it is a SHIO mover
					TShioTimeValue * shioTimeValue = (TShioTimeValue*)timeFile; // typecast
					WorldPoint wp = shioTimeValue -> GetStationLocation();
					VelocityRec vel;
					short btnHit;
					char msg[256], latS[20], longS[20];
					WorldPointToStrings(wp, latS, longS);
					WorldRect gridBounds;
					if(sharedCMover -> fGrid == 0)
					{ printError("Programmer error: sharedCMover -> fGrid is nil"); break;}
					gridBounds = sharedCMover -> fGrid -> GetBounds();
					
					//if(WPointInWRect(wp.pLong,wp.pLat,&sharedCMover -> bounds))
					if(WPointInWRect(wp.pLong,wp.pLat,&gridBounds))
					{
						btnHit = MULTICHOICEALERT(1670, 0, FALSE);
						switch(btnHit)
						{
							case 1:  // Yes, default button
								// user want to use the ref point info from the file
								// set the lat,long 
								LL2EditTexts(dialog, M16LATDEGREES, &wp);
								// set the scale to either 1 or -1 
								scaleValue = EditText2Float(dialog, M16SCALEVALUE);
								if(scaleValue < 0.0) Float2EditText(dialog, M16SCALEVALUE,-1.0, 4);//preserve the sign, i.e. preserve the direction the user set
								else Float2EditText(dialog, M16SCALEVALUE,1.0, 4);
								// turn on the scale button
								(void)CATSClick(dialog,M16SCALETOCONSTANT,lParam,data);
								break; 
								//case 3: return USERCANCEL; //NO button
							case 3: break; //NO button, still may want to set by hand, not a user cancel
						}
					}
					else
					{
						// hmmmm, this is most likely an error
						// should we alert the user ??
						sprintf(msg,"Check that this is the right file.%sThe reference point in this file is not within the map bounds.%sLat:%s%sLng:%s",NEWLINESTRING,NEWLINESTRING,latS,NEWLINESTRING,longS);
						printWarning(msg);
					}
					// if file contains height coefficients force derivative to be calculated and scale to be input
					if (timeFile->GetFileType() == SHIOHEIGHTSFILE || timeFile->GetFileType() == SHIOCURRENTSFILE || timeFile->GetFileType() == PROGRESSIVETIDEFILE)
					{
						if (err = timeFile->GetTimeValue(model->GetStartTime(),&vel))	// minus AH 07/10/2012
							goto donetimefile;	// user cancel or error
					}
				}
				
				sharedCMDialogTimeDep = timeFile;
			}
			
		donetimefile:
			SetPopSelection (dialog, M16TIMEFILETYPES, sharedCMDialogTimeDep ? sharedCMDialogTimeDep->GetFileType() : NOTIMEFILE);
			PopDraw(dialog, M16TIMEFILETYPES);
		{	char itextstr[256] = "<none>";	// code warrior didn't like the ?: expression
			if (sharedCMDialogTimeDep) strcpy (itextstr,sharedCMDialogTimeDep->fileName);
			mysetitext(dialog, M16TIMEFILENAME, itextstr);
		}
			//mysetitext(dialog, M16TIMEFILENAME, sharedCMDialogTimeDep ? sharedCMDialogTimeDep->fileName : "<none>");
			ShowHideCATSDialogItems(dialog);
			ShowHideScaleFactorItems(dialog);
			ShowCatsDialogUnitLabels(dialog);
			break;
			
		case M16TIMEFILESCALEFACTOR:
		case M16HYDROLOGYSCALEFACTOR:
			//case M16TRANSPORT:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
			
			
		case M16REPLACEMOVER:
			err = sharedCMover -> ReplaceMover();
			if (err == USERCANCEL) break;	// stay at dialog
			return itemNum;	// what to do on error?
			break;
			
		case M16DEGREES:
		case M16DEGMIN:
		case M16DMS:
			err = EditTexts2LL(dialog, M16LATDEGREES, &p,TRUE);
			if(err) break;
			if (itemNum == M16DEGREES) settings.latLongFormat = DEGREES;
			if (itemNum == M16DEGMIN) settings.latLongFormat = DEGMIN;
			if (itemNum == M16DMS) settings.latLongFormat = DMS;
			SwitchLLFormat(dialog, M16LATDEGREES, M16DEGREES);
			LL2EditTexts(dialog, M16LATDEGREES, &p);
			break;
	}
	
	return 0;
}


OSErr CATSInit(DialogPtr dialog, VOIDPTR data)
{
	char itextstr[256] = "<none>";
	sharedCMDialogTimeDep = sharedCMover->timeDep;
	sharedCatsDialogNonPtrFields = GetCATSDialogNonPtrFields(sharedCMover);
	sSharedCATSUncertainyInfo = sharedCMover -> GetCurrentUncertaintyInfo();
	
	SetDialogItemHandle(dialog, M16HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M16FROST, (Handle)FrameEmbossed);
	
	RegisterPopTable(csPopTable, sizeof(csPopTable) / sizeof(PopInfoRec));
	RegisterPopUpDialog(M16, dialog);
	
	mysetitext(dialog, M16FILENAME, sharedCMover->className);
	SetButton(dialog, M16ACTIVE, sharedCMover->bActive);
	
	SetButton(dialog, M16SHOWARROWS, sharedCMover->bShowArrows);
	Float2EditText(dialog, M16ARROWSCALE, sharedCMover->arrowScale, 4);
	
	SetPopSelection(dialog, M16TIMEFILETYPES, sharedCMover->timeDep ? sharedCMover->timeDep->GetFileType() : NOTIMEFILE);
	if (sharedCMover->timeDep) strcpy(itextstr,sharedCMover->timeDep->fileName);
	//mysetitext(dialog, M16TIMEFILENAME, (sharedCMover->timeDep ? sharedCMover->timeDep->fileName : "<none>"));
	mysetitext(dialog, M16TIMEFILENAME, itextstr);	// code warrior doesn't like the ?: expression
	
	//if (sharedCMover->IAm(TYPE_TIDECURCYCLEMOVER)) setwtitle(dialog,"Tide Pattern Mover Settings");
	
	SetButton(dialog, M16NOSCALING, sharedCMover->scaleType == SCALE_NONE);
	SetButton(dialog, M16SCALETOCONSTANT, sharedCMover->scaleType == SCALE_CONSTANT);
	SetButton(dialog, M16SCALETOGRID, sharedCMover->scaleType == SCALE_OTHERGRID);
	Float2EditText(dialog, M16SCALEVALUE, sharedCMover->scaleValue, 4);
	mysetitext(dialog, M16SCALEGRIDNAME, sharedCMover->scaleOtherFile);
	SwitchLLFormat(dialog, M16LATDEGREES, M16DEGREES);
	LL2EditTexts(dialog, M16LATDEGREES, &sharedCMover->refP);
	
	ShowHideCATSDialogItems(dialog);
  	ShowHideScaleFactorItems(dialog);
	
	ShowUnscaledValue(dialog);
	ShowCatsDialogUnitLabels(dialog);
	
	return 0;
}



OSErr CATSSettingsDialog(TCATSMover *newMover, TMap *owner, Boolean *timeFileChanged)
{
	short item;
	
	if (!newMover)return -1;
	
	sharedCMover = newMover;
	
	sharedCMChangedTimeFile = FALSE;
	item = MyModalDialog(M16, mapWindow, newMover, CATSInit, CATSClick);
	*timeFileChanged = sharedCMChangedTimeFile;
	
	if(M16OK == item)	model->NewDirtNotification();// tell model about dirt
	return M16OK == item ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////

void DrawCGItem(DialogPtr dialog, Rect *r, long item)
{
	char s[256];
	TCATSMover *mover;
	
	TextFontSize(kFontIDGeneva,LISTTEXTSIZE);
	
	sharedMoverList->GetListItem((Ptr)&mover, item);
	strcpy(s, mover->className);
	
	drawstring(s);
	
	TextFontSize(0,12);
}

Boolean CGClick(DialogPtr dialog, VLISTPTR L, short dialogItem, long *listItem,
				Boolean doubleClick)
{
	TCATSMover *mover;
	
	switch (dialogItem) {
		case M17OK:
			sharedMoverList->GetListItem((Ptr)&mover, *listItem);
			strcpy(sharedCMFileName, mover->className);
			
			return TRUE;
		case M17CANCEL: return TRUE;
	}
	
	return FALSE;
}

void CGInit(DialogPtr dialog, VLISTPTR L)
{
	long i;
	TCATSMover *mover;
	
	for (i = 0 ; i < L->numItems ; i++) {
		sharedMoverList->GetListItem((Ptr)&mover, i);
		if (!strcmp(mover->className, sharedCMFileName)) {
			VLSetSelect(i, L);
			VLAutoScroll(L);
			break;
		}
	}
}

TCurrentMover *CreateAndInitLocationFileCurrentsMover (TMap *owner, char* givenPath,char* givenFileName,TMap **newMap, char* topFilePath)
{
	char path[256], s[256], fileName[32], fileNamesPath[256];
	short item, gridType, selectedUnits;
	TCurrentMover *newMover=nil;
	TGridVel *grid = nil;
	OSErr err = 0;
	Boolean isNetCDFPathsFile = false;
	
	{	// don't ask user, we were provided with the path
		strcpy(path,givenPath);
		strcpy(fileName,givenFileName);
	}
	
	if (IsPtCurFile(path))
	{
		PtCurMover *newPtCurMover = new PtCurMover(owner, fileName);
		if (!newPtCurMover)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new newPtCurMover()", 0);
			return 0;
		}
		newMover = newPtCurMover;
		
		err = newPtCurMover->InitMover();
		if(err) goto Error;
		
		err = newPtCurMover->TextRead(path,newMap); // outside users may supply their own map
		if(err) goto Error;	
	}
	else if (IsTriCurFile(path))
	{
		TriCurMover *newTriCurMover = new TriCurMover(owner, fileName);
		if (!newTriCurMover)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new newTriCurMover()", 0);
			return 0;
		}
		newMover = newTriCurMover;
		
		err = newTriCurMover->InitMover();
		if(err) goto Error;
		
		err = newTriCurMover->TextRead(path,newMap); // outside users may supply their own map
		if(err) goto Error;	
	}
	else if (IsCATS3DFile(path))
	{
		TCATSMover3D *newTCATSMover3D = new TCATSMover3D(owner, fileName);
		if (!newTCATSMover3D)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new newTCATSMover3D()", 0);
			return 0;
		}
		newMover = newTCATSMover3D;
		
		grid = new TTriGridVel3D;
		if (grid)
		{
			WorldRect mapBounds = voidWorldRect;
			err = newTCATSMover3D->InitMover(grid, WorldRectCenter(mapBounds));
			if(err) goto Error;
			
			err = newTCATSMover3D->TextRead(path,newMap); // outside users must supply their own map
			if(err) goto Error;	
			if (*newMap) mapBounds = (*newMap)->GetMapBounds();
			else mapBounds = (newTCATSMover3D->moverMap)->GetMapBounds();
			newTCATSMover3D->SetRefPosition(WorldRectCenter(mapBounds),0.);
			// if refP not in grid should set it inside a triangle
		}
	}
	else if (IsGridCurTimeFile(path,&selectedUnits))
	{
		GridCurMover *newGridCurMover = new GridCurMover(owner, fileName);
		if (!newGridCurMover)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new newGridCurMover()", 0);
			return 0;
		}
		newMover = newGridCurMover;
		
		grid = new TRectGridVel;
		if (grid)
		{
			WorldRect mapBounds = voidWorldRect;
			err = newGridCurMover->InitMover(grid, WorldRectCenter(mapBounds));
			if(err) goto Error;
			newGridCurMover->fUserUnits = selectedUnits;
			
			err = newGridCurMover->TextRead(path); 
			if(err) goto Error;	
		}
		if (newGridCurMover -> moverMap == model -> uMap)
			if (! newGridCurMover->OkToAddToUniversalMap())
				goto Error;
		
	}
	else if (IsTideCurCycleFile(path, &gridType))
	{	// could combine with isnetcdffile by adding an extra parameter
		WorldPoint p = {0,0};
		if (gridType!=TRIANGULAR) {err=-1; printNote("Tidal current cycle movers are only implemented for triangle grids."); goto Error;}
		TideCurCycleMover *newTideCurCycleMover = new TideCurCycleMover(owner, fileName);
		if (!newTideCurCycleMover)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new newTideCurCycleMover()", 0);
			return 0;
		}
		newMover = newTideCurCycleMover;
		
		err = newTideCurCycleMover->InitMover(grid,p);	// dummy variables for now
		if(err) goto Error;
		
		err = newTideCurCycleMover->TextRead(path,newMap,topFilePath); // outside users may supply their own map
		if(err) goto Error;	
	}
	else if (IsNetCDFFile(path, &gridType) || IsNetCDFPathsFile(path, &isNetCDFPathsFile, fileNamesPath, &gridType))
	{
		if (gridType == CURVILINEAR)
		{
			NetCDFMoverCurv *newNetCDFMover = new NetCDFMoverCurv(owner, fileName);
			if (!newNetCDFMover)
			{ 
				TechError("CreateAndInitCurrentsMover()", "new newNetCDFMoverCurv()", 0);
				return 0;
			}
			newMover = newNetCDFMover;
			
			err = newNetCDFMover->InitMover();	
			if(err) goto Error;
			
			//err = newNetCDFMover->TextRead(path,newMap); 
			err = newNetCDFMover->TextRead(path,newMap,topFilePath); 
			if(err) goto Error;	
			
		}
		else if (gridType == TRIANGULAR)
		{
			NetCDFMoverTri *newNetCDFMover = new NetCDFMoverTri(owner, fileName);
			if (!newNetCDFMover)
			{ 
				TechError("CreateAndInitCurrentsMover()", "new newNetCDFMoverTri()", 0);
				return 0;
			}
			newMover = newNetCDFMover;
			
			err = newNetCDFMover->InitMover();	
			if(err) goto Error;
			
			//err = newNetCDFMover->TextRead(path,newMap); 
			err = newNetCDFMover->TextRead(path,newMap,topFilePath); 
			if(err) goto Error;	
			
		}
		else	// regular grid or regular_swafs / ncom
		{
			NetCDFMover *newNetCDFMover = new NetCDFMover(owner, fileName);
			if (!newNetCDFMover)
			{ 
				TechError("CreateAndInitCurrentsMover()", "new newNetCDFMover()", 0);
				return 0;
			}
			newMover = newNetCDFMover;
			
			err = newNetCDFMover->InitMover();
			if(err) goto Error;
			
			//err = newNetCDFMover->TextRead(path,newMap); 
			err = newNetCDFMover->TextRead(path,newMap,topFilePath); 
			if(err) goto Error;	
			
		}
		if (isNetCDFPathsFile)
		{
			char errmsg[256];
			err = (dynamic_cast<NetCDFMover*>(newMover))->ReadInputFileNames(fileNamesPath);
			if (err) goto Error;
			(dynamic_cast<NetCDFMover*>(newMover))->DisposeAllLoadedData();
			//err = ((NetCDFMover*)newMover)->SetInterval(errmsg);	// if set interval here will get error if times are not in model range
			if(err) goto Error;
		}
	}
	else // didn't recognize the file
	{
		printError("Error reading currents file. Not a valid format.");
		goto Error;
	}
	return newMover;
	
Error: // JLM 	 10/27/98
	if(newMover) {newMover->Dispose();delete newMover;newMover = 0;}
	return 0;
	
}

TCATSMover *CreateAndInitCatsCurrentsMover (TMap *owner, Boolean askForFile, char* givenPath,char* givenFileName)
{
	TCurrentMover *mover = CreateAndInitCurrentsMover (owner,askForFile,givenPath,givenFileName,nil); // CATS movers should not have their own map
	//
	if (mover && mover->GetClassID() != TYPE_CATSMOVER)
	{ // for now, we've assumed the patterns are CATS Movers
		printError("Non-CATS Mover created in CreateAndInitCatsCurrentsMover");
		mover -> Dispose();
		delete mover;
		mover = 0;
	}
	return dynamic_cast<TCATSMover *>(mover);
}

TCurrentMover *CreateAndInitCurrentsMover (TMap *owner, Boolean askForFile, char* givenPath,char* givenFileName,TMap **newMap)
{
	char path[256], s[256], fileName[64], fileNamesPath[256], topFilePath[256], outPath[256];
	short item, gridType, selectedUnits;
	Point where = CenteredDialogUpLeft(M38c);;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	TCurrentMover *newMover=nil;
	TGridVel *grid = nil;
	TimeGridVel *timeGrid = nil;
	OSErr err = 0;
	Boolean isNetCDFPathsFile = false;
	
	if(askForFile || !givenPath || !givenFileName)
	{
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
					 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return 0;
		strcpy(path, reply.fullPath);
#else
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
	
	if (IsPtCurFile(path))
	{
#ifdef GUI_GNOME
		char errmsg[256];
		timeGrid = new TimeGridCurTri();
		//timeGrid = new TimeGridVel();
		if (timeGrid)
		{
			GridCurrentMover *newGridCurrentMover = new GridCurrentMover(owner, fileName);
			if (!newGridCurrentMover)
			{ 
				TechError("CreateAndInitCurrentsMover()", "new GridCurrentMover()", 0);
				return 0;
			}
			newMover = newGridCurrentMover;
			
			err = newGridCurrentMover->InitMover(timeGrid);
			if(err) goto Error;
#if TARGET_API_MAC_CARBON
			err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			if (!err) strcpy(path,outPath);
#endif
			err = dynamic_cast<TimeGridCurTri*>(timeGrid)->ReadHeaderLines(path,&(newGridCurrentMover->fUncertainParams));
			err = timeGrid->TextRead(path,"");
			if(err) goto Error;
			if (!err) /// JLM 5/3/10
			{
				//char errmsg[256];
				//err = timeGrid->ReadInputFileNames(fileNamesPath);
				timeGrid->DisposeAllLoadedData();
				if(!err) err = timeGrid->SetInterval(errmsg,model->GetModelTime()); // if set interval here will get error if times are not in model range
			}
		}
#else
		PtCurMover *newPtCurMover = new PtCurMover(owner, fileName);
		if (!newPtCurMover)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new newPtCurMover()", 0);
			return 0;
		}
		newMover = newPtCurMover;
		
		err = newPtCurMover->InitMover();
		if(err) goto Error;
		
		err = newPtCurMover->TextRead(path,newMap); // outside users may supply their own map
		if(err) goto Error;	
#endif
	}
	else if (IsTriCurFile(path))
	{
		TriCurMover *newTriCurMover = new TriCurMover(owner, fileName);
		if (!newTriCurMover)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new newTriCurMover()", 0);
			return 0;
		}
		newMover = newTriCurMover;
		
		err = newTriCurMover->InitMover();
		if(err) goto Error;
		
		err = newTriCurMover->TextRead(path,newMap); // outside users may supply their own map
		if(err) goto Error;	
	}
	else if (IsCATS3DFile(path))
	{
		TCATSMover3D *newTCATSMover3D = new TCATSMover3D(owner, fileName);
		if (!newTCATSMover3D)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new newTCATSMover3D()", 0);
			return 0;
		}
		newMover = newTCATSMover3D;
		
		grid = new TTriGridVel3D;
		if (grid)
		{
			WorldRect mapBounds = voidWorldRect;
			err = newTCATSMover3D->InitMover(grid, WorldRectCenter(mapBounds));
			if(err) goto Error;
			
			err = newTCATSMover3D->TextRead(path,newMap); // outside users must supply their own map
			if(err) goto Error;	
			if (*newMap) mapBounds = (*newMap)->GetMapBounds();
			else mapBounds = (newTCATSMover3D->moverMap)->GetMapBounds();
			newTCATSMover3D->SetRefPosition(WorldRectCenter(mapBounds),0.);
			// if refP not in grid should set it inside a triangle
		}
	}
	else if (IsGridCurTimeFile(path,&selectedUnits))
	{
#ifdef GUI_GNOME
		char errmsg[256];
		timeGrid = new TimeGridCurRect();
		//timeGrid = new TimeGridVel();
		if (timeGrid)
		{
			GridCurrentMover *newGridCurrentMover = new GridCurrentMover(owner, fileName);
			if (!newGridCurrentMover)
			{ 
				TechError("CreateAndInitCurrentsMover()", "new GridCurrentMover()", 0);
				return 0;
			}
			newMover = newGridCurrentMover;
		
			err = newGridCurrentMover->InitMover(timeGrid);
			if(err) goto Error;
#if TARGET_API_MAC_CARBON
			err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			if (!err) strcpy(path,outPath);
#endif
			err = timeGrid->TextRead(path,"");
			if(err) goto Error;
			if (!err /*&& isNetCDFPathsFile*/) /// JLM 5/3/10
			{
				//char errmsg[256];
				//err = timeGrid->ReadInputFileNames(fileNamesPath);
				/*if(!err)*/ timeGrid->DisposeAllLoadedData();
				if(!err) err = timeGrid->SetInterval(errmsg,model->GetModelTime()); // if set interval here will get error if times are not in model range
			}
		}
#else
	GridCurMover *newGridCurMover = new GridCurMover(owner, fileName);
		if (!newGridCurMover)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new newGridCurMover()", 0);
			return 0;
		}
		newMover = newGridCurMover;
		
		grid = new TRectGridVel;
		if (grid)
		{
			WorldRect mapBounds = voidWorldRect;
			err = newGridCurMover->InitMover(grid, WorldRectCenter(mapBounds));
			if(err) goto Error;
			newGridCurMover->fUserUnits = selectedUnits;
			
			err = newGridCurMover->TextRead(path); 
			if(err) goto Error;	
		}
		if (newGridCurMover -> moverMap == model -> uMap)
			if (! newGridCurMover->OkToAddToUniversalMap())
				goto Error;
#endif
		
	}
	else if (IsTideCurCycleFile(path, &gridType))
	{	// could combine with isnetcdffile by adding an extra parameter
		WorldPoint p = {0,0};
		if (gridType!=TRIANGULAR) {err=-1; printNote("Current cycle movers are only implemented for triangle grids."); goto Error;}
#ifdef GUI_GNOME
		timeGrid = new TimeGridVelTri();
		//timeGrid = new TimeGridVel();
		if (timeGrid)
		{
			Point where;
			OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
			MySFReply reply;
			Boolean bTopFile = false;
			topFilePath[0]=0;
			// code goes here, store path as unix
			if (gridType!=REGULAR)	// move this outside, pass the path in
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
						//err=-1;// stay at this dialog
						break;
				}
			}
			if(bTopFile)
			{
#if TARGET_API_MAC_CARBON
				mysfpgetfile(&where, "", -1, typeList,
							 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
				if (!reply.good)
				{
				}
				else
					strcpy(topFilePath, reply.fullPath);
				
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
				}
				
				my_p2cstr(reply.fName);
				
#ifdef MAC
				GetFullPath(reply.vRefNum, 0, (char *)reply.fName, topFilePath);
#else
				strcpy(topFilePath, reply.fName);
#endif
#endif		
			}
			CurrentCycleMover *newCurrentCycleMover = new CurrentCycleMover(owner, fileName);
			if (!newCurrentCycleMover)
			{ 
				TechError("CreateAndInitCurrentsMover()", "new newCurrentCycleMover()", 0);
				return 0;
			}
			newMover = newCurrentCycleMover;
		
			timeGrid->bIsCycleMover = true;
			err = newCurrentCycleMover->InitMover(timeGrid);	// dummy variables for now
			if(err) goto Error;
			
	#if TARGET_API_MAC_CARBON
			err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			if (!err) strcpy(path,outPath);
			err = ConvertTraditionalPathToUnixPath((const char *) topFilePath, outPath, kMaxNameLen) ;
			if (!err) strcpy(topFilePath,outPath);
	#endif
			//err = newCurrentCycleMover->TextRead(path,newMap,""); // outside users may supply their own map
			err = timeGrid->TextRead(path,topFilePath); // outside users may supply their own map
			if(err) goto Error;	
		}
		else{err=-1; goto Error;}
#else
		TideCurCycleMover *newTideCurCycleMover = new TideCurCycleMover(owner, fileName);
		if (!newTideCurCycleMover)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new newTideCurCycleMover()", 0);
			return 0;
		}
		newMover = newTideCurCycleMover;
		
		err = newTideCurCycleMover->InitMover(grid,p);	// dummy variables for now
		if(err) goto Error;
		
		err = newTideCurCycleMover->TextRead(path,newMap,""); // outside users may supply their own map
		if(err) goto Error;	
#endif
	}
	else if (IsADCPFile(path))
	{
		//printNote("ADCP file");
		ADCPMover *newADCPMover = new ADCPMover(owner,fileName);
		if (!newADCPMover)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new newADCPMover()", 0);
			return 0;
		}
		newMover = newADCPMover;
		
		grid = new TRectGridVel;
		if (grid)
		{
			WorldRect mapBounds = voidWorldRect;
			if (owner)
				mapBounds = owner->GetMapBounds();
			grid -> SetBounds(mapBounds); // this is for the old style ossm grids
			err = newADCPMover->InitMover(grid, WorldRectCenter(mapBounds));
			if(err) goto Error;
			//err = newADCPMover->InitMover(grid,p);	// dummy variables for now
			//if(err) goto Error;
			
			//err = newADCPMover->TextRead(path,newMap,""); // outside users may supply their own map
			
			//err = newADCPMover->TextRead(path); 
			//if(err) goto Error;	
		}
		if (newADCPMover -> moverMap == model -> uMap)
			if (! newADCPMover->OkToAddToUniversalMap())
				goto Error;
	}
	else if (IsNetCDFFile(path, &gridType) || IsNetCDFPathsFile(path, &isNetCDFPathsFile, fileNamesPath, &gridType))
	{
#ifdef GUI_GNOME
		if (gridType == CURVILINEAR)
		{
			timeGrid = new TimeGridVelCurv();
		}
		else if (gridType == TRIANGULAR)
		{
			timeGrid = new TimeGridVelTri();
		}
		
		else
		{
			timeGrid = new TimeGridVelRect();
			//timeGrid = new TimeGridVel();
		}
		if (timeGrid)
		{
			Point where;
			OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
			MySFReply reply;
			Boolean bTopFile = false;
			topFilePath[0]=0;
			// code goes here, store path as unix
			if (gridType!=REGULAR)	// move this outside, pass the path in
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
						//err=-1;// stay at this dialog
						break;
				}
			}
			if(bTopFile)
			{
#if TARGET_API_MAC_CARBON
				mysfpgetfile(&where, "", -1, typeList,
							 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
				if (!reply.good)
				{
				}
				else
					strcpy(topFilePath, reply.fullPath);
				
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
				}
				
				my_p2cstr(reply.fName);
				
#ifdef MAC
				GetFullPath(reply.vRefNum, 0, (char *)reply.fName, topFilePath);
#else
				strcpy(topFilePath, reply.fName);
#endif
#endif		
			}

			
			GridCurrentMover *newGridCurrentMover = new GridCurrentMover(owner, fileName);
			if (!newGridCurrentMover)
			{ 
				TechError("CreateAndInitCurrentsMover()", "new GridCurrentMover()", 0);
				return 0;
			}
			newMover = newGridCurrentMover;
			
			err = newGridCurrentMover->InitMover(timeGrid);
			if(err) goto Error;
#if TARGET_API_MAC_CARBON
			err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			if (!err) strcpy(path,outPath);
			err = ConvertTraditionalPathToUnixPath((const char *) topFilePath, outPath, kMaxNameLen) ;
			if (!err) strcpy(topFilePath,outPath);
#endif
			err = timeGrid->TextRead(path,topFilePath);
			if(err) goto Error;
		}
		if (isNetCDFPathsFile)
		{
			char errmsg[256];
			err = timeGrid->ReadInputFileNames(fileNamesPath);
			if (err) goto Error;
			timeGrid->DisposeAllLoadedData();
			//err = ((NetCDFMover*)newMover)->SetInterval(errmsg);	// if set interval here will get error if times are not in model range
			if(err) goto Error;
		}
#else		
		if (gridType == CURVILINEAR)
		{
			NetCDFMoverCurv *newNetCDFMover = new NetCDFMoverCurv(owner, fileName);
			if (!newNetCDFMover)
			{ 
				TechError("CreateAndInitCurrentsMover()", "new newNetCDFMoverCurv()", 0);
				return 0;
			}
			newMover = newNetCDFMover;
			
			err = newNetCDFMover->InitMover();	
			if(err) goto Error;
			
			err = newNetCDFMover->TextRead(path,newMap,""); 
			if(err) goto Error;	
			
		}
		else if (gridType == TRIANGULAR)
		{
			NetCDFMoverTri *newNetCDFMover = new NetCDFMoverTri(owner, fileName);
			if (!newNetCDFMover)
			{ 
				TechError("CreateAndInitCurrentsMover()", "new newNetCDFMoverTri()", 0);
				return 0;
			}
			newMover = newNetCDFMover;
			
			err = newNetCDFMover->InitMover();	
			if(err) goto Error;
			
			err = newNetCDFMover->TextRead(path,newMap,""); 
			if(err) goto Error;	
			
		}
		else	// regular grid
		{
			NetCDFMover *newNetCDFMover = new NetCDFMover(owner, fileName);
			if (!newNetCDFMover)
			{ 
				TechError("CreateAndInitCurrentsMover()", "new newNetCDFMover()", 0);
				return 0;
			}
			newMover = newNetCDFMover;
			
			err = newNetCDFMover->InitMover();
			if(err) goto Error;
			
			err = newNetCDFMover->TextRead(path,newMap,""); 
			if(err) goto Error;	
			
		}
		if (isNetCDFPathsFile)
		{
			char errmsg[256];
			err = (dynamic_cast<NetCDFMover*>(newMover))->ReadInputFileNames(fileNamesPath);
			if (err) goto Error;
			(dynamic_cast<NetCDFMover*>(newMover))->DisposeAllLoadedData();
			//err = ((NetCDFMover*)newMover)->SetInterval(errmsg);	// if set interval here will get error if times are not in model range
			if(err) goto Error;
		}
#endif
	}
	else
	{ // CATS mover file
		TCATSMover *newCatsMover = 0;
		if (IsTriGridFile (path))
			grid = new TTriGridVel;
		else if (IsRectGridFile (path))
			grid = new TRectGridVel;
		
		if (grid)
		{
			WorldRect mapBounds = voidWorldRect;
			
			if(owner) 
				mapBounds = owner->GetMapBounds();
			
			grid -> SetBounds(mapBounds); // this is for the old style ossm grids
			// new style grids will overwrite this when reading their files
			
			newCatsMover = new TCATSMover(owner, fileName);
			if (!newCatsMover)
			{ 
				TechError("CreateAndInitCurrentsMover()", "new TCATSMover()", 0);
				return 0;
			}
			newMover = newCatsMover;
			
			err = newCatsMover->InitMover(grid, WorldRectCenter(mapBounds));
			if(err) goto Error;
			
			err = grid->TextRead(path);
			if(err) goto Error;
			
			if (newCatsMover -> moverMap == model -> uMap)
				if (! newCatsMover->OkToAddToUniversalMap())
					goto Error;
			
		}
		else // didn't recognize the file
		{
			printError("Error reading currents file. Not a valid format.");
			goto Error;
		}
	}
	if (timeGrid)
	{
		Seconds startTime = timeGrid->GetTimeValue(0);
		if (startTime != CONSTANTCURRENT && (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime()))
		{
			if (true)	// maybe use NOAA.ver here?
			{
				short buttonSelected;
				//buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first\n time in the file?",FALSE);
				//if(!gCommandFileErrorLogPath[0])
				if(!gCommandFileRun)	// also may want to skip for location files...
					buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first time in the file?",FALSE);
				else buttonSelected = 1;	// TAP user doesn't want to see any dialogs, always reset (or maybe never reset? or send message to errorlog?)
				switch(buttonSelected){
					case 1: // reset model start time
						model->SetModelTime(startTime);
						model->SetStartTime(startTime);
						model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
						break;  
					case 3: // don't reset model start time
						break;
					case 4: // cancel
						err=-1;// user cancel
						goto Error;
						//break;
				}
			}
		}
	}
	return newMover;
	
Error: // JLM 	 10/27/98
	if(newMover) {newMover->Dispose();delete newMover;newMover = 0;};
	return 0;
	
}


void TCATSMover::Draw(Rect r, WorldRect view)
{
	if(fGrid && (bShowArrows || bShowGrid))
		fGrid->Draw(r,view,refP,refScale,arrowScale,arrowDepth,bShowArrows,bShowGrid,fColor);
	//if (bShowDepthContours) ((TTriGridVel3D*)fGrid)->DrawDepthContours(r,view,bShowDepthContourLabels);
	//if (true) ((TTriGridVel*)fGrid)->DrawDepthContours(r,view,true);
}

/**************************************************************************************************/
CurrentUncertainyInfo TCATSMover::GetCurrentUncertaintyInfo ()
{
	CurrentUncertainyInfo	info;
	
	memset(&info,0,sizeof(info));
	info.setEddyValues = TRUE;
	info.fUncertainStartTime	= this -> fUncertainStartTime;
	info.fDuration					= this -> fDuration;
	info.fEddyDiffusion			= this -> fEddyDiffusion;		
	info.fEddyV0					= this -> fEddyV0;			
	info.fDownCurUncertainty	= this -> fDownCurUncertainty;	
	info.fUpCurUncertainty		= this -> fUpCurUncertainty;	
	info.fRightCurUncertainty	= this -> fRightCurUncertainty;	
	info.fLeftCurUncertainty	= this -> fLeftCurUncertainty;	
	
	return info;
}
/**************************************************************************************************/
void TCATSMover::SetCurrentUncertaintyInfo (CurrentUncertainyInfo info)
{
	this -> fUncertainStartTime	= info.fUncertainStartTime;
	this -> fDuration 				= info.fDuration;
	this -> fEddyDiffusion 			= info.fEddyDiffusion;		
	this -> fEddyV0 					= info.fEddyV0;			
	this -> fDownCurUncertainty 	= info.fDownCurUncertainty;	
	this -> fUpCurUncertainty 		= info.fUpCurUncertainty;	
	this -> fRightCurUncertainty 	= info.fRightCurUncertainty;	
	this -> fLeftCurUncertainty 	= info.fLeftCurUncertainty;	
	
	return;
}
Boolean TCATSMover::CurrentUncertaintySame (CurrentUncertainyInfo info)
{
	if (this -> fUncertainStartTime	== info.fUncertainStartTime 
		&&	this -> fDuration 				== info.fDuration
		&&	this -> fEddyDiffusion 			== info.fEddyDiffusion		
		&&	this -> fEddyV0 				== info.fEddyV0			
		&&	this -> fDownCurUncertainty 	== info.fDownCurUncertainty	
		&&	this -> fUpCurUncertainty 		== info.fUpCurUncertainty	
		&&	this -> fRightCurUncertainty 	== info.fRightCurUncertainty	
		&&	this -> fLeftCurUncertainty 	== info.fLeftCurUncertainty	)
		return true;
	else return false;
}
OSErr TCATSMover::ExportTopology(char* path)
{
	// export triangle info so don't have to regenerate each time
	OSErr err = 0;
	long numTriangles, numBranches, nver/*, nBoundarySegs=0, nWaterBoundaries=0*/;
	long i, n, v1,v2,v3,n1,n2,n3;
	double x,y;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	DAGHdl treeH = 0;
	//LONGH	boundarySegmentsH = 0, boundaryTypeH = 0;
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
	
	/*if (moverMap->IAm(TYPE_PTCURMAP))
	 {
	 boundaryTypeH = ((PtCurMap*)moverMap)->GetWaterBoundaries();
	 boundarySegmentsH = ((PtCurMap*)moverMap)->GetBoundarySegs();
	 if (!boundaryTypeH || !boundarySegmentsH) {printError("No map info to export"); err=-1; goto done;}
	 }*/
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	
	// Write out values
	strcpy(buffer,"DAG 1.0\n");
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
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

