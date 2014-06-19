#include "ADCPMover.h"
#include "CROSS.H"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT ADCPMOVER
#endif
#endif

//Rect CATSgridRect = { 0, 0, kOCurWidth, kOCurHeight };

static PopInfoRec csPopTable[] = {
	{ M16, nil, M16LATDIR, 0, pNORTHSOUTH1, 0, 1, FALSE, nil },
	{ M16, nil, M16LONGDIR, 0, pEASTWEST1, 0, 1, FALSE, nil },
	{ M16, nil, M16TIMEFILETYPES, 0, pTIMEFILETYPES, 0, 1, FALSE, nil }
};

static ADCPMover	*sharedADCPMover = 0;
static CMyList 		*sharedMoverList = 0;
static char 		sharedCMFileName[256];
static Boolean		sharedCMChangedTimeFile;
static ADCPTimeValue *sharedCMDialogTimeDep = 0;
static CurrentUncertainyInfo sSharedADCPUncertainyInfo; // used to hold the uncertainty dialog box info in case of a user cancel
static ADCPDialogNonPtrFields sharedADCPDialogNonPtrFields;


ADCPMover::ADCPMover (TMap *owner, char *name) : TCurrentMover(owner, name)
{
	fDuration=48*3600; //48 hrs as seconds 
	fTimeUncertaintyWasSet =0;
	
	fGrid = 0;
	//SetTimeDep (nil);
	//bTimeFileActive = false;
	fEddyDiffusion=0; // JLM 5/20/991e6; // cm^2/sec
	fEddyV0 = 0.1; // JLM 5/20/99
	fBinToUse = 0;	//default use all bins
	
	timeDepList = nil;
	
	memset(&fOptimize,0,sizeof(fOptimize));
	memset(&fLegendRect,0,sizeof(fLegendRect)); 
	SetClassName (name);
}


void ADCPMover::Dispose ()
{
	long i,n;
	ADCPTimeValue *thisTimeDep;
	if (fGrid)
	{
		fGrid -> Dispose();
		delete fGrid;
		fGrid = nil;
	}
	
	if (timeDepList)
	{
		for (i = 0, n = timeDepList->GetItemCount() ; i < n ; i++) {
			timeDepList->GetListItem((Ptr)&thisTimeDep, i);
			thisTimeDep->Dispose();
			delete thisTimeDep;
			thisTimeDep = nil;
		}
		
		timeDepList->Dispose();
		delete timeDepList;
		timeDepList = nil;
	}
	
	//DeleteTimeDep ();
	
	
	TCurrentMover::Dispose ();
}

/*void ADCPMover::DeleteTimeDep () 
 {
 if (timeDep)
 {
 timeDep -> Dispose ();
 delete timeDep;
 timeDep = nil;
 }
 
 return;
 }*/



ADCPDialogNonPtrFields GetADCPDialogNonPtrFields(ADCPMover	* adcpm)
{
	ADCPDialogNonPtrFields f;
	
	f.fUncertainStartTime  = adcpm->fUncertainStartTime; 	
	f.fDuration  = adcpm->fDuration; 
	//
	f.refP  = adcpm->refP; 	
	f.refZ  = adcpm->refZ; 	
	f.scaleType = adcpm->scaleType; 
	f.scaleValue = adcpm->scaleValue;
	strcpy(f.scaleOtherFile,adcpm->scaleOtherFile);
	f.refScale = adcpm->refScale; 
	//f.bTimeFileActive = adcpm->bTimeFileActive; 
	f.bShowGrid = adcpm->bShowGrid; 
	f.bShowArrows = adcpm->bShowArrows; 
	f.arrowScale = adcpm->arrowScale; 
	f.fEddyDiffusion = adcpm->fEddyDiffusion; 
	f.fEddyV0 = adcpm->fEddyV0; 
	f.fDownCurUncertainty = adcpm->fDownCurUncertainty; 
	f.fUpCurUncertainty = adcpm->fUpCurUncertainty; 
	f.fRightCurUncertainty = adcpm->fRightCurUncertainty; 
	f.fLeftCurUncertainty = adcpm->fLeftCurUncertainty; 
	return f;
}

void SetADCPDialogNonPtrFields(ADCPMover	* adcpm,ADCPDialogNonPtrFields * f)
{
	adcpm->fUncertainStartTime = f->fUncertainStartTime; 	
	adcpm->fDuration  = f->fDuration; 
	//
	adcpm->refP = f->refP; 	
	adcpm->refZ  = f->refZ; 	
	adcpm->scaleType = f->scaleType; 
	adcpm->scaleValue = f->scaleValue;
	strcpy(adcpm->scaleOtherFile,f->scaleOtherFile);
	adcpm->refScale = f->refScale; 
	//adcpm->bTimeFileActive = f->bTimeFileActive; 
	adcpm->bShowGrid = f->bShowGrid; 
	adcpm->bShowArrows = f->bShowArrows; 
	adcpm->arrowScale = f->arrowScale; 
	adcpm->fEddyDiffusion = f->fEddyDiffusion;
	adcpm->fEddyV0 = f->fEddyV0;
	adcpm->fDownCurUncertainty = f->fDownCurUncertainty; 
	adcpm->fUpCurUncertainty = f->fUpCurUncertainty; 
	adcpm->fRightCurUncertainty = f->fRightCurUncertainty; 
	adcpm->fLeftCurUncertainty = f->fLeftCurUncertainty; 
}

///////////////////////////////////////////////////////////////////////////

Boolean IsADCPFile(char *path)
{
	
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		NthLineInTextNonOptimized (firstPartOfFile, line = 1, strLine, 512);
		if (strstr(strLine,"# Raw data"))	// metadata file
			bIsValid = true;
		else if (strstr(strLine,"# NOTICE:"))	// .hdr or one of the bin.dat files
			bIsValid = true;
	}
	// might want to check a little further
	return bIsValid;
}

/*void ShowUnscaledValue2(DialogPtr dialog)
{
	double length;
	WorldPoint p;
	VelocityRec velocity;
	
	(void)EditTexts2LL(dialog, M16LATDEGREES, &p, FALSE);
	velocity = sharedADCPMover->GetPatValue(p);
	length = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	Float2EditText(dialog, M16UNSCALEDVALUE, length, 4);
}*/

Boolean ADCPMover::OkToAddToUniversalMap()
{
	// only allow this if we have grid with valid bounds
	WorldRect gridBounds;
	if (!fGrid) {
		printError("Error in ADCPMover::OkToAddToUniversalMap.");
		return false;
	}
	gridBounds = fGrid -> GetBounds();
	if(EqualWRects(gridBounds,emptyWorldRect)) {
		printError("You cannot create a universal mover from a current file which does not specify the grid's bounds.");
		return false;
	}
	return true;
}

OSErr ADCPMover::InitMover()
{
	OSErr err = 0;
	if (!(timeDepList = new CMyList(sizeof(ADCPTimeValue *))))
	{ TechError("ADCPMover::InitModel()", "new CMyList()", 0); err = memFullErr; }
	else if (err = timeDepList->IList())
	{ TechError("ADCPMover::InitModel()", "IList()", 0); }
	
	return err;
}

OSErr ADCPMover::InitMover(TGridVel *grid, WorldPoint p)
{
	OSErr err = 0;
	fGrid = grid;
	refP = p;
	refZ = 0;
	scaleType = SCALE_NONE;
	scaleValue = 1.0;
	scaleOtherFile[0] = 0;
	bRefPointOpen = FALSE;
	bUncertaintyPointOpen = FALSE;
	//bTimeFileOpen = FALSE;
	bShowArrows = FALSE;
	bShowGrid = FALSE;
	arrowScale = 1;
	
	if (!(timeDepList = new CMyList(sizeof(ADCPTimeValue *))))
	{ TechError("ADCPMover::InitModel()", "new CMyList()", 0); err = memFullErr; }
	else if (err = timeDepList->IList())
	{ TechError("ADCPMover::InitModel()", "IList()", 0); }
	
	dynamic_cast<ADCPMover *>(this)->ComputeVelocityScale(model->GetModelTime());	// AH 07/10/2012
	
	return err;
}

OSErr ADCPMover::ReplaceMover()
{
	OSErr err = 0;
	ADCPMover* mover = CreateAndInitADCPCurrentsMover (this -> moverMap,true,0,0); // only allow to replace with same type of mover
	if (mover)
	{
		// save original fields
		ADCPDialogNonPtrFields fields = GetADCPDialogNonPtrFields(dynamic_cast<ADCPMover *>(this));
		SetADCPDialogNonPtrFields(mover,&fields);
		/*if(this->timeDep)
		 {
		 err = this->timeDep->MakeClone(&mover->timeDep);
		 if (err) { delete mover; mover=0; return err; }
		 // check if shio or hydrology, save ref point 
		 //if (!(this->timeDep->GetFileType() == ADCPTIMEFILE)) 
		 //mover->refP = this->refP;
		 //mover->bTimeFileActive=true;
		 // code goes here , should replace all the fields?
		 //mover->scaleType = this->scaleType;
		 //mover->scaleValue = this->scaleValue;
		 //mover->refScale = this->refScale;
		 //strcpy(mover->scaleOtherFile,this->scaleOtherFile);
		 }*/
		if (err = this->moverMap->AddMover(mover,0))
		{mover->Dispose(); delete mover; mover = 0; return err;}
		if (err = this->moverMap->DropMover(dynamic_cast<ADCPMover *>(this)))
		{mover->Dispose(); delete mover; mover = 0; return err;}
	}
	else 
	{err = -1; return err;}
	
	model->NewDirtNotification();
	return err;
}

#define ADCPMoverREADWRITEVERSION 1 //JLM

OSErr ADCPMover::Write (BFPB *bfpb)
{
	char c;
	long i, version = ADCPMoverREADWRITEVERSION, numADCPs = 0;
	ClassID id = GetClassID ();
	ADCPTimeValue *thisTimeDep;
	OSErr err = 0;
	
	if (err = TCurrentMover::Write (bfpb)) return err;
	
	StartReadWriteSequence("ADCPMover::Write()");
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
	//if (err = WriteMacValue(bfpb,bTimeFileOpen)) return err;
	//if (err = WriteMacValue(bfpb,bTimeFileActive)) return err;
	
	if (err = WriteMacValue(bfpb,bShowGrid)) return err;
	if (err = WriteMacValue(bfpb,bShowArrows)) return err;
	
	// JLM 9/2/98 
	if (err = WriteMacValue(bfpb,fEddyDiffusion)) return err;
	if (err = WriteMacValue(bfpb,fEddyV0)) return err;
	
	// bOptimizedForStep does not need to be saved to the save file
	
	if (err = WriteMacValue(bfpb,arrowScale)) return err;
	
	numADCPs = timeDepList -> GetItemCount ();
	if (err = WriteMacValue(bfpb,numADCPs)) return err;
	for (i = 0; i < timeDepList -> GetItemCount (); i++)
	{
		timeDepList -> GetListItem ((Ptr) &thisTimeDep, i);
		if(thisTimeDep)
		{	
			id = thisTimeDep -> GetClassID();
			if (err = WriteMacValue(bfpb, id)) return err;
			if (err = thisTimeDep -> Write(bfpb)) return err;
		}
	}
	
	id = fGrid -> GetClassID (); //JLM
	if (err = WriteMacValue(bfpb, id)) return err; //JLM
	err = fGrid -> Write (bfpb);
	
	return err;
}

OSErr ADCPMover::Read(BFPB *bfpb)
{
	char c;
	long i, version, numADCPs;
	ClassID id;
	ADCPTimeValue *thisTimeDep;
	OSErr err = 0;
	
	if (err = TCurrentMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("ADCPMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("ADCPMover::Read()", "id != TYPE_ADCPMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != ADCPMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	
	if (err = ReadMacValue(bfpb,&refP.pLong)) return err;
	if (err = ReadMacValue(bfpb,&refP.pLat)) return err;
	if (err = ReadMacValue(bfpb,&refScale)) return err;
	if (err = ReadMacValue(bfpb,&refZ)) return err;
	if (err = ReadMacValue(bfpb,&scaleType)) return err;
	if (err = ReadMacValue(bfpb,&scaleValue)) return err;
	if (err = ReadMacValue(bfpb, scaleOtherFile, sizeof(scaleOtherFile))) return err;  // don't swap !! 
	
	if (err = ReadMacValue(bfpb, &bRefPointOpen)) return err;
	if (err = ReadMacValue(bfpb, &bUncertaintyPointOpen)) return err;
	//if (err = ReadMacValue(bfpb, &bTimeFileOpen)) return err;
	//if (err = ReadMacValue(bfpb, &bTimeFileActive)) return err;
	
	if (err = ReadMacValue(bfpb, &bShowGrid)) return err;
	if (err = ReadMacValue(bfpb, &bShowArrows)) return err;
	
	// JLM 9/2/98 
	if (err = ReadMacValue(bfpb,&fEddyDiffusion)) return err;
	if (err = ReadMacValue(bfpb,&fEddyV0)) return err;
	
	// bOptimizedForStep does not need to be saved to the save file
	
	
	if (err = ReadMacValue(bfpb,&arrowScale)) return err;
	
	numADCPs = timeDepList -> GetItemCount ();
	if (err = ReadMacValue(bfpb, &numADCPs)) return err;
	for (i = 0; i < numADCPs; i++)
	{
		if (err = ReadMacValue(bfpb,&id)) return err;
		switch (id) {
				//case TYPE_TIMEVALUES: thisTimeDep = new TTimeValue(this); break;
				//case TYPE_OSSMTIMEVALUES: thisTimeDep = new TOSSMTimeValue(this); break;
				//case TYPE_SHIOTIMEVALUES: thisTimeDep = new TShioTimeValue(this); break;
			case TYPE_ADCPTIMEVALUES: thisTimeDep = new ADCPTimeValue(dynamic_cast<ADCPMover *>(this)); break;
			default: printError("Unrecognized time file type in ADCPMover::Read()."); return -1;
		}
		if (!thisTimeDep)
		{ TechError("ADCPMover::Read()", "new ADCPTimeValue()", 0); return -1; };
		if (err = thisTimeDep -> InitTimeFunc()) return err;
		
		if (err = thisTimeDep -> Read(bfpb)) return err;
		dynamic_cast<ADCPMover *>(this)->AddTimeDep(thisTimeDep,0);
	}
	
	// read the type of grid used for the ADCP mover
	if (err = ReadMacValue(bfpb,&id)) return err;
	// JLM if (err = ReadMacValue(bfpb,&version)) return err;
	// if (version != 1) { printSaveFileVersionError(); return -1; }
	switch(id)
	{	// set up a simple rect grid ?
		case TYPE_RECTGRIDVEL: fGrid = new TRectGridVel;break;
			//case TYPE_TRIGRIDVEL: fGrid = new TTriGridVel;break;
			//case TYPE_TRIGRIDVEL3D: fGrid = new TTriGridVel3D;break;
		default: printError("Unrecognized Grid type in ADCPMover::Read()."); return -1;
	}
	
	fGrid -> Read (bfpb);
	
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr ADCPMover::CheckAndPassOnMessage(TModelMessage *message)
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
			//else if(!strcmpnocase(str,"othergrid")) this->scaleType = SCALE_OTHERGRID; 
		}
		/////////////
		err = message->GetParameterAsWorldPoint("refP",&wp,false);
		if(!err) this->refP = wp;
		//////////////
		// instead of timeDep may want to set up timeDepList...
		/*message->GetParameterString("timeFile",str,256);
		 ResolvePath(str);
		 if(str[0])
		 {	// str contains the PATH descriptor, e.g. "resNum 10001"
		 char shortFileName[32]=""; // not really used
		 short unitsIfKnownInAdvance = kUndefined;
		 char str2[64];
		 Boolean haveScaleFactor = false;
		 ADCPTimeValue*  timeFile = 0;
		 
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
		 // code goes here, if want to apply scale factor when reading in need to pass into CreateADCP...
		 //this->timeDep->fScaleFactor = val; 	
		 // need to set refScale if hydrology
		 haveScaleFactor = true;
		 unitsIfKnownInAdvance = -2;
		 }
		 }
		 
		 timeFile = CreateADCPTimeValue(this,str,shortFileName,unitsIfKnownInAdvance);	
		 this->DeleteTimeDep();
		 if(timeFile) 
		 {
		 this -> timeDep = timeFile;
		 if (haveScaleFactor) 
		 {
		 VelocityRec dummyValue;
		 this -> timeDep -> fScaleFactor = val;
		 if (err = timeFile->GetTimeValue(model->GetStartTime(),&dummyValue))	// make sure data is ok
		 this->DeleteTimeDep();
		 }
		 if (this -> timeDep) this -> bTimeFileActive = true; // assume we want to make it active
		 }
		 }*/
		//////////////
		/////////////
		dynamic_cast<ADCPMover *>(this)->ComputeVelocityScale(model->GetModelTime());	// AH 07/10/2012
		model->NewDirtNotification();// tell model about dirt
	}
	
	long messageCode = message->GetMessageCode();
	switch(messageCode)
	{
		case M_UPDATEVALUES:
			VelocityRec dummyValue;
			// new data, make sure it is ok
			//if (timeDep) err = timeDep->GetTimeValue(model->GetStartTime(),&dummyValue);
			//if (err) this->DeleteTimeDep();
			break;
	}
	
	/////////////////////////////////////////////////
	// sub-guys need us to pass this message 
	/////////////////////////////////////////////////
	//if(this->timeDep) err = this->timeDep->CheckAndPassOnMessage(message);
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TCurrentMover::CheckAndPassOnMessage(message);
}

/////////////////////////////////////////////////
long ADCPMover::GetListLength()
{
	long count = 1, i, n;
	ADCPTimeValue *adcpTimeDep = 0;
	
	if (bOpen) {
		count += 4;		// minimum ADCP mover lines
		//if (timeDep)count++;
		if (bRefPointOpen) count += 3;
		if(model->IsUncertain())count++;
		if(bUncertaintyPointOpen && model->IsUncertain())count +=6;
		// add 1 to # of time-values for active / inactive
		
		// JLM if (bTimeFileOpen) count += timeDep ? timeDep -> GetNumValues () : 0;
		//if (bTimeFileOpen) count += timeDep ? (1 + timeDep -> GetListLength ()) : 0; //JLM, add 1 for the active flag
		//if (bMapsOpen)
		{	// here have a draw map and draw contours for each map, anything else? title
			n = timeDepList->GetItemCount() ;
			//listLength += n;
			for (i = 0, n = timeDepList->GetItemCount() ; i < n ; i++) {
				timeDepList->GetListItem((Ptr)&adcpTimeDep, i);
				//listLength += map->GetListLength();
				count += adcpTimeDep->GetListLength();
			}
		}
	}
	
	return count;
}

ListItem ADCPMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char *p, latS[20], longS[20], valStr[32];
	ListItem item = { dynamic_cast<ADCPMover *>(this), 0, indent, 0 };
	long i, m, count;
	ADCPTimeValue *adcpTimeDep = 0;
	
	if (n == 0) {
		item.index = I_ADCPNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "ADCP Mover: \"%s\"", className);
		//sprintf(text, "Currents: \"%s\"", className);
		*style = bActive ? italic : normal;
		
		return item;
	}
	
	item.indent++;
	
	if (bOpen) {
		
		
		if (--n == 0) {
			item.index = I_ADCPACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
		
		
		if (--n == 0) {
			item.index = I_ADCPGRID;
			item.bullet = bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			sprintf(text, "Show Grid");
			
			return item;
		}
		
		if (--n == 0) {
			item.index = I_ADCPARROWS;
			item.bullet = bShowArrows ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			StringWithoutTrailingZeros(valStr,arrowScale,6);
			sprintf(text, "Show Velocities (@ 1 in = %s m/s)", valStr);
			
			return item;
		}
		
		
		if (--n == 0) {
			item.index = I_ADCPREFERENCE;
			item.bullet = bRefPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Reference Point");
			
			return item;
		}
		
		
		if (bRefPointOpen) {
			if (--n == 0) {
				item.index = I_ADCPSCALING;
				//item.bullet = BULLET_DASH;
				item.indent++;
				switch (scaleType) {
					case SCALE_NONE:
						strcpy(text, "No reference point scaling");
						break;
					case SCALE_CONSTANT:
						//if (timeDep && timeDep->GetFileType()==HYDROLOGYFILE)
						//StringWithoutTrailingZeros(valStr,refScale,6);
						//else
						StringWithoutTrailingZeros(valStr,scaleValue,6);
						sprintf(text, "Scale to: %s ", valStr);
						// units
						//if (timeDep)
						//strcat(text,"* file value");
						//else
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
				item.index = (n == 0) ? I_ADCPLAT : I_ADCPLONG;
				//item.bullet = BULLET_DASH;
				WorldPointToStrings(refP, latS, longS);
				strcpy(text, (n == 0) ? latS : longS);
				
				return item;
			}
			
			n--;
		}
		
		
		/*if(timeDep)
		 {
		 if (--n == 0)
		 {
		 char	timeFileName [kMaxNameLen];
		 
		 item.index = I_ADCPTIMEFILE;
		 item.bullet = bTimeFileOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		 timeDep -> GetTimeFileName (timeFileName);
		 sprintf(text, "Tide File: %s", timeFileName);
		 if(!bTimeFileActive)*style = italic; // JLM 6/14/10
		 return item;
		 }
		 }*/
		
		/*if (bTimeFileOpen && timeDep) {
		 
		 if (--n == 0)
		 {
		 item.indent++;
		 item.index = I_ADCPTIMEFILEACTIVE;
		 item.bullet = bTimeFileActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
		 strcpy(text, "Active");
		 
		 return item;
		 }
		 
		 ///JLM  ///{
		 // Note: n is one higher than JLM expected it to be 
		 // (the ADCP mover code is pre-decrementing when checking)
		 if(timeDep -> GetListLength () > 0)
		 {	// only check against the entries if we have some 
		 n--; // pre-decrement
		 if (n < timeDep -> GetListLength ()) {
		 item.indent++;
		 item = timeDep -> GetNthListItem(n,item.indent,style,text);
		 // over-ride the objects answer ??  JLM
		 // no 10/23/00 
		 //item.owner = this; // so the clicks come to me
		 //item.index = I_ADCPTIMEENTRIES + n;
		 //////////////////////////////////////
		 //item.bullet = BULLET_DASH;
		 return item;
		 }
		 n -= timeDep -> GetListLength ()-1; // the -1 is to leave the count one higher so they can pre-decrement
		 }
		 ////}
		 
		 }*/
		
		
		//if (bMapsOpen)
		{
			n--; // pre-decrement	// but add back  if no adcps
			for (i = 0, m = timeDepList->GetItemCount() ; i < m ; i++) {
				timeDepList->GetListItem((Ptr)&adcpTimeDep, i);
				//strcpy(text, "Embedded Maps");
				//return item;
				count = adcpTimeDep->GetListLength();
				if (count > n)
					return adcpTimeDep->GetNthListItem(n, indent + 1, style, text);
				
				n -= count;
			}
			
			
		}
		
		if(model->IsUncertain())
		{
			if (--n == 0) {
				item.index = I_ADCPUNCERTAINTY;
				item.bullet = bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				strcpy(text, "Uncertainty");
				
				return item;
			}
			
			if (bUncertaintyPointOpen) {
				
				if (--n == 0) {
					item.index = I_ADCPSTARTTIME;
					//item.bullet = BULLET_DASH;
					item.indent++;
					sprintf(text, "Start Time: %.2f hours",((double)fUncertainStartTime)/3600.);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_ADCPDURATION;
					//item.bullet = BULLET_DASH;
					item.indent++;
					sprintf(text, "Duration: %.2f hours",fDuration/3600);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_ADCPDOWNCUR;
					//item.bullet = BULLET_DASH;
					item.indent++;
					sprintf(text, "Down Current: %.2f to %.2f %%",fDownCurUncertainty*100,fUpCurUncertainty*100);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_ADCPCROSSCUR;
					//item.bullet = BULLET_DASH;
					item.indent++;
					sprintf(text, "Cross Current: %.2f to %.2f %%",fLeftCurUncertainty*100,fRightCurUncertainty*100);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_ADCPDIFFUSIONCOEFFICIENT;
					//item.bullet = BULLET_DASH;
					item.indent++;
					sprintf(text, "Eddy Diffusion: %.2e cm^2/sec",fEddyDiffusion);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_ADCPEDDYV0;
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

Boolean ADCPMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	Boolean timeFileChanged = false;
	if (inBullet)
		switch (item.index) {
			case I_ADCPNAME: bOpen = !bOpen; return TRUE;
			case I_ADCPGRID: bShowGrid = !bShowGrid; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_ADCPARROWS: bShowArrows = !bShowArrows; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_ADCPREFERENCE: bRefPointOpen = !bRefPointOpen; return TRUE;
			case I_ADCPUNCERTAINTY: bUncertaintyPointOpen = !bUncertaintyPointOpen; return TRUE;
				//case I_ADCPTIMEFILE: bTimeFileOpen = !bTimeFileOpen; return TRUE;
				//case I_ADCPTIMEFILEACTIVE: bTimeFileActive = !bTimeFileActive; 
				//model->NewDirtNotification(); return TRUE;
			case I_ADCPACTIVE:
				bActive = !bActive;
				model->NewDirtNotification(); 
				//if (!bActive && bTimeFileActive)
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
			case I_ADCPSTARTTIME:
			case I_ADCPDURATION:
			case I_ADCPDOWNCUR:
			case I_ADCPCROSSCUR:
			case I_ADCPDIFFUSIONCOEFFICIENT:
			case I_ADCPEDDYV0:
			case I_ADCPUNCERTAINTY:
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
						dynamic_cast<ADCPMover *>(this)->UpdateUncertaintyValues(model->GetModelTime()-model->GetStartTime());
					}
				}
				return TRUE;
				break;
			}
			default:
				ADCPSettingsDialog (dynamic_cast<ADCPMover *>(this), this -> moverMap, &timeFileChanged);
				return TRUE;
				break;
		}
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean ADCPMover::FunctionEnabled(ListItem item, short buttonID)
{
	long i,n,j,num;
	//TMover* mover,mover2;
	switch (item.index) {
		case I_ADCPNAME:
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


/*OSErr ADCPMover::UpItem(ListItem item)
 {	
 long i;
 OSErr err = 0;
 
 if (item.index == I_ADCPNAME)
 if (model->LESetsList->IsItemInList((Ptr)&item.owner, &i))
 //if (i > 0) {// 2 for each
 if (i > 1) {// 2 for each
 //if (err = model->LESetsList->SwapItems(i, i - 1))
 if ((err = model->LESetsList->SwapItems(i, i - 2)) || (err = model->LESetsList->SwapItems(i+1, i - 1)))
 { TechError("ADCPMover::UpItem()", "model->LESetsList->SwapItems()", err); return err; }
 SelectListItem(item);
 UpdateListLength(true);
 InvalidateMapImage();
 InvalMapDrawingRect();
 }
 
 return 0;
 }
 
 OSErr ADCPMover::DownItem(ListItem item)
 {
 long i;
 OSErr err = 0;
 
 if (item.index == I_ADCPNAME)
 if (model->LESetsList->IsItemInList((Ptr)&item.owner, &i))
 //if (i < (model->LESetsList->GetItemCount() - 1)) {
 if (i < (model->LESetsList->GetItemCount() - 3)) {
 //if (err = model->LESetsList->SwapItems(i, i + 1))
 if ((err = model->LESetsList->SwapItems(i, i + 2)) || (err = model->LESetsList->SwapItems(i+1, i + 3)))
 { TechError("ADCPMover::UpItem()", "model->LESetsList->SwapItems()", err); return err; }
 SelectListItem(item);
 UpdateListLength(true);
 InvalidateMapImage();
 InvalMapDrawingRect();
 }
 
 return 0;
 }*/

OSErr ADCPMover::SettingsItem(ListItem item)
{
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = this -> ListClick(item,inBullet,doubleClick);
	return 0;
}

OSErr ADCPMover::DeleteItem(ListItem item)
{
	if (item.index == I_ADCPNAME)
		return moverMap -> DropMover(dynamic_cast<ADCPMover *>(this));
	
	return 0;
}


/////////////////////////////////////////////////
static VList sgObjects;
static CMyList *sTimeDepList=0;
static CMyList *sDialogList=0;
static short NAME_COL, NUMBINS_COL, TOPBINDEPTH_COL, DEPTH_COL, BINSIZE_COL;
static ADCPMover	*sSharedDialogADCPMover = 0;

#define REPLACE true

void UpdateDisplayWithTimeDepNamesSet(DialogPtr dialog,char* timeDepName)
{	// this is display outside the box
	char namestr[256];
	//StringWithoutTrailingZeros(namestr,moverName,2);
	strcpy(namestr,timeDepName);
	mysetitext(dialog,ADCPNAME,namestr);	
	
	//mysetitext(dialog,ADCPDLGPAT2NAME,namestr);	
	
	/*Float2EditText(dialog,EPU,dvals.value.u,6);
	 Float2EditText(dialog,EPV,dvals.value.v,6);
	 Float2EditText(dialog,EPTEMP,dvals.temp,2);
	 Float2EditText(dialog,EPSAL,dvals.sal,2);*/
}


static void UpdateDisplayWithCurSelection(DialogPtr dialog)
{
	ADCPTimeValue* curTimeDep;
	Point pos,mp;
	long curSelection;
	char nameStr[256];
	
	if(!VLAddRecordRowIsSelected(&sgObjects))
	{	// set the item text
		{
			VLGetSelect(&curSelection,&sgObjects);
			sTimeDepList->GetListItem((Ptr)&curTimeDep,curSelection);
		}
		if (curTimeDep) strcpy(nameStr,curTimeDep->className);
		
		UpdateDisplayWithTimeDepNamesSet(dialog,nameStr);
	}
	
	//ShowHideAutoIncrement(dialog,curSelection); // JLM 9/17/98
}

static void SelectNthRow(DialogPtr dialog,long nrow)
{
	VLSetSelect(nrow, &sgObjects); 
	if(nrow > -1)
	{
		VLAutoScroll(&sgObjects);
	}
	VLUpdate(&sgObjects);
	UpdateDisplayWithCurSelection(dialog);	
}

static OSErr AddReplaceRecord(DialogPtr dialog,/*Boolean incrementDepth,*/Boolean replace,ADCPTimeValue* theTimeDep)
{
	long n,itemnum,curSelection;
	OSErr err=0;
	
	if(!err)
	{
		// will need to define InsertSorted for the CDOG profiles data type, sort by depth
		//err=sMoverList->InsertSorted ((Ptr)&theMover,&itemnum,false);// false means don't allow duplicate times
		err=sTimeDepList->AppendItem((Ptr)&theTimeDep);	// list of names vs list of movers
		
		if(!err) // new record
		{
			itemnum = sgObjects.numItems;
			VLAddItem(1,&sgObjects);
			VLSetSelect(itemnum, &sgObjects); 
			VLAutoScroll(&sgObjects);
			VLUpdate(&sgObjects);
			//if(incrementDepth)IncrementDepth(dialog,dvals.depth);
		}
		/*else if(err == -2) // found existing record. Replace if okay to replace
		 {	// not sure if need replace option, at least not based on matching name
		 if(replace)
		 {
		 sMoverList->DeleteItem(itemnum);
		 VLDeleteItem(itemnum,&sgObjects);
		 err = AddReplaceRecord(dialog,!INCREMENT_DEPTH,REPLACE,&theMover);
		 VLUpdate(&sgObjects);
		 //if(incrementDepth)IncrementDepth(dialog,dvals.depth);
		 err=0;
		 }
		 else
		 {
		 printError("A record with the specified depth already exists."
		 "If you want to edit the existing record, select it."
		 "If you want to add a new record, change the specified depth.");
		 VLUpdate(&sgObjects);
		 }
		 }*/
		else SysBeep(5);
	}
	return err;
}

void DisposeADCPDLGStuff(void)
{
	if(sTimeDepList)
	{
		sTimeDepList->Dispose();// JLM 12/14/98
		delete sTimeDepList;
		sTimeDepList = 0;
	}
	
	//?? VLDispose(&sgObjects);// JLM 12/10/98, is this automatic on the mac ??
	memset(&sgObjects,0,sizeof(sgObjects));// JLM 12/10/98
}

/*void ShowHideADCPDialogItems(DialogPtr dialog)
 {
 ADCPMover	*mover = sSharedDialogADCPMover;
 //short moverCode = GetPopSelection(dialog, ADCPDLGMOVERTYPES);
 Boolean showPat2Items = FALSE, showScaleByItems = FALSE;
 
 return;
 
 }*/

short ADCPDlgClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short 			item;
	OSErr 			err = 0;
	ADCPTimeValue	*timeDep = 0;
	ADCPTimeValue	*currentTimeDep=0;
	//TMap			*currentMap = 0;
	long 			i,j,n,m,curSelection,binToUse,numBins;
	Point 			pos;
	Boolean			needToBreak = false;
	
	switch (itemNum) {
			
		case ADCPDLGOK:
		{
			if(sTimeDepList)
			{
				//DepthValuesSetH dvalsh = sgDepthValuesH;
				n = sTimeDepList->GetItemCount();
				if(n == 0)
				{	// no items are entered, tell the user
					char msg[512],buttonName[64];
					GetWizButtonTitle_Cancel(buttonName);
					sprintf(msg,"You have not entered any data values.  Either enter data values and use the 'Load' button, or use the '%s' button to exit the dialog.",buttonName);
					printError(msg);
					break;
				}
				
				binToUse = EditText2Long(dialog,ADCPBINNUM);
				// check that all the values are in range - if there is some range
				// or may allow the user to change units - anything to check here?
				for(i=0;i<n;i++)
				{
					char errStr[256] = "";
					err=sTimeDepList->GetListItem((Ptr)&timeDep,i);
					if(err) {SysBeep(5); break;}// this shouldn't ever happen
					if (timeDep /*&& timeDep->bActive*/) numBins = timeDep->GetNumBins();
					if (numBins<binToUse)
					{
						char msg[512];
						sprintf(msg,"You have entered a bin number larger than the total number of bins which is %ld.",numBins);
						printError(msg);
						needToBreak = true;
						break;
					}
				}
				if (needToBreak) break;
				/////////////
				// point of no return
				//////////////
				//m = sDialogList->GetItemCount() ;
				/*if (n==0)
				 {
				 ClearList();
				 //{	// not sure if want movers or mover names here or maybe another list...
				 dvalsh = (DepthValuesSetH)_NewHandle(n*sizeof(DepthValuesSet));
				 if(!dvalsh)
				 {
				 TechError("EditProfilesClick:OKAY", "_NewHandle()", 0);
				 //return EPCANCEL;
				 break; // make them cancel so that code gets executed
				 }
				 sgDepthValuesH = dvalsh;
				 }
				 else
				 {
				 _SetHandleSize((Handle)dvalsh,n*sizeof(DepthValuesSet));
				 if(_MemError())
				 {
				 TechError("EditProfilesClick:OKAY", "_NewHandle()", 0);
				 //return EPCANCEL;
				 break; // make them cancel, so that code gets executed
				 }
				 }*/
				
				
				sDialogList->ClearList();
				for(i=0;i<n;i++)
				{
					if(err=sTimeDepList->GetListItem((Ptr)&currentTimeDep,i))return ADCPDLGOK;
					err=sDialogList->AppendItem((Ptr)&currentTimeDep);
					if(err)return err;
				}
				
				
			}
			sSharedDialogADCPMover->fBinToUse = EditText2Long(dialog,ADCPBINNUM);
			////////////////////////////////////////////////////////////
			DisposeADCPDLGStuff();
			return ADCPDLGOK;
		}
			break;
			
		case ADCPDLGCANCEL:
			// delete any new patterns, restore original patterns
			//DeleteIfNotOriginalPattern(mover -> pattern1);
			//DeleteIfNotOriginalPattern(mover -> pattern2);
			//mover -> pattern1 = sSharedOriginalPattern1;
			//mover -> pattern2 = sSharedOriginalPattern2;
			
			DisposeADCPDLGStuff();
			return ADCPDLGCANCEL;
			break;
			
		case ADCPDLGDELETEALL:
			sTimeDepList->ClearList();
			VLReset(&sgObjects,1);
			UpdateDisplayWithCurSelection(dialog);
			break;
		case ADCPDLGMOVEUP:
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				if (curSelection>0) 
				{
					sTimeDepList->SwapItems(curSelection,curSelection-1);
					VLSetSelect(curSelection-1,&sgObjects);
					--curSelection;
				}
			}
			VLUpdate(&sgObjects);
			//VLReset(&sgObjects,1);
			UpdateDisplayWithCurSelection(dialog);
			break;
		case ADCPDLGDELETEROWS_BTN:
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				sTimeDepList->DeleteItem(curSelection);
				VLDeleteItem(curSelection,&sgObjects);
				if(sgObjects.numItems == 0)
				{
					VLAddItem(1,&sgObjects);
					VLSetSelect(0,&sgObjects);
				}
				--curSelection;
				if(curSelection >-1)
				{
					VLSetSelect(curSelection,&sgObjects);
				}
				VLUpdate(&sgObjects);
			}
			UpdateDisplayWithCurSelection(dialog);
			break;
			
		case ADCPDLGREPLACE:
			//err = RetrieveIncrementDepth(dialog);
		{
			ADCPTimeValue* adcpTimeValue = sSharedDialogADCPMover->AddADCP(&err);
			//sMoverList = 
			if(err) break;
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				//TMover* thisCurMover=0;
				//err=GetDepthVals(dialog,&dvals);
				if(err) break;
				
				if(curSelection==sTimeDepList->GetItemCount())
				{
					// replacing blank record
					err = AddReplaceRecord(dialog,!REPLACE,adcpTimeValue);
					//if (sMapList && newMap) err=sMapList->AppendItem((Ptr)&newMap);	
					if (!err) SelectNthRow(dialog, curSelection+1 ); 
				}
				else // replacing existing record
				{	// not allowed right now
					VLGetSelect(&curSelection,&sgObjects);
					sTimeDepList->DeleteItem(curSelection);
					//if (sMapList) sMapList->DeleteItem(curSelection);
					VLDeleteItem(curSelection,&sgObjects);		
					err = AddReplaceRecord(dialog,REPLACE,adcpTimeValue);
					//if (sMapList && newMap) err=sMapList->AppendItem((Ptr)&newMap);	
				}
			}
		}
			break;
			
		case ADCPDLGLIST:
			// retrieve every time they click on the list
			// because clicking can cause the increment to be hidden
			// and we need to verify it before it gets hidden
			//err = RetrieveIncrementDepth(dialog);
			//if(err) break;
			///////////
			pos=GetMouseLocal(GetDialogWindow(dialog));
			VLClick(pos, &sgObjects);
			VLUpdate(&sgObjects);
			VLGetSelect(&curSelection,&sgObjects);
			if(curSelection == -1 )
			{
				curSelection = sgObjects.numItems-1;
				VLSetSelect(curSelection,&sgObjects);
				VLUpdate(&sgObjects);
			}
			
			//ShowHideAutoIncrement(dialog,curSelection);
			// moved into UpdateDisplayWithCurSelection()
			
			//if (AddRecordRowIsSelected2())
			if (VLAddRecordRowIsSelected(&sgObjects))
			{
				//DepthValuesSet dvals;
				ADCPTimeValue* timeDep=0;
				sTimeDepList->GetListItem((Ptr)&timeDep,sTimeDepList->GetItemCount()-1);
				//err = RetrieveIncrementDepth(dialog);
				if(err) break;
				//IncrementDepth(dialog,dvals.depth);
			}
			UpdateDisplayWithCurSelection(dialog);
			break;
			
		case ADCPBINNUM:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;
			
			
	}
	
	return 0;
}

///////////////////////////////////////////////////////////////////////////

void DrawTimeDepNameList(DialogPtr w, RECTPTR r, long n)
{
	char s[256], text[256], fileName[64], numBinStr[64], topBinStr[64];
	ADCPTimeValue* timeDep = 0;
	long numBins, topBinIndex=0, sensorOrientation;
	double stationDepth, topBinDepth, binSize;
	
	if(n == sgObjects.numItems-1)
	{
		strcpy(s,"****");
	 	MyMoveTo(NAME_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
		MyMoveTo(NUMBINS_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
		MyMoveTo(TOPBINDEPTH_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
		MyMoveTo(BINSIZE_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
		MyMoveTo(DEPTH_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	return; 
	}
	
	sTimeDepList->GetListItem((Ptr)&timeDep,n);
	if (timeDep) 
	{
		//timeDep->GetTimeFileName(fileName);
		timeDep->GetStationName(fileName);
		//strcpy(s, timeDep->className);
		strcpy(s, fileName);
	}
	else {strcpy(s,"****");	strcpy(text,"****");} // shouldn't happen
	MyMoveTo(NAME_COL-stringwidth(s)/2,r->bottom);
	//MyMoveTo(NAME_COL-stringwidth("Station Name")/2,r->bottom);
	drawstring(s);
	if (timeDep)
	{
		numBins = timeDep->GetNumBins();
		StringWithoutTrailingZeros(numBinStr,numBins,1);
	}
	//MyMoveTo(NUMBINS_COL-stringwidth("Top Bin")/2,r->bottom);
	//drawstring(s);
	if (timeDep)
	{
		sensorOrientation = timeDep->GetSensorOrientation();
		if (sensorOrientation == 2) 
			topBinIndex = 0;	// downward facing
		else
			topBinIndex = numBins-1;	// upward facing
		topBinDepth = timeDep->GetBinDepth(topBinIndex);
		StringWithoutTrailingZeros(s,topBinDepth,1);
		StringWithoutTrailingZeros(topBinStr,topBinIndex+1,1);
		sprintf(text, "%s/%s", topBinStr, numBinStr);
	}
	//MyMoveTo(NUMBINS_COL-stringwidth("Top Bin")/2,r->bottom);
	MyMoveTo(NUMBINS_COL-stringwidth(text)/2,r->bottom);
	drawstring(text);
	//MyMoveTo(TOPBINDEPTH_COL-stringwidth("Top Depth")/2,r->bottom);
	MyMoveTo(TOPBINDEPTH_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	if (timeDep)
	{
		binSize = timeDep->GetBinSize();
		StringWithoutTrailingZeros(s,binSize,1);
	}
	//MyMoveTo(BINSIZE_COL-stringwidth("Bin Size")/2,r->bottom);
	MyMoveTo(BINSIZE_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	if (timeDep)
	{
		stationDepth = timeDep->GetStationDepth();
		StringWithoutTrailingZeros(s,stationDepth,1);
	}
	//MyMoveTo(DEPTH_COL-stringwidth("Station Depth")/2,r->bottom);
	MyMoveTo(DEPTH_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	//MyMoveTo(NAME_COL/*-stringwidth("Pattern Name")/2*/,r->bottom);
	
	
	return;
}


pascal_ifMac void TimeDepNamesListUpdate(DialogPtr dialog, short itemNum)
{
	Rect r = GetDialogItemBox(dialog,ADCPDLGLIST);
	
	VLUpdate(&sgObjects);
}

OSErr ADCPDlgInit(DialogPtr dialog, VOIDPTR data)
{
	char 			s[256];
	ADCPMover	*ADCPMover;
	Rect r = GetDialogItemBox(dialog, ADCPDLGLIST);
	char blankStr[32];
	strcpy(blankStr,"");
	ADCPTimeValue *currentTimeDep=0;
	//TMap *currentMap = 0;
	long i,n,m;
	OSErr err = 0;
	short IBMoffset;
	
	ADCPMover = sSharedDialogADCPMover;
	if (!ADCPMover) return -1;
	//sSharedOriginalPattern1 = mover -> pattern1;
	//sSharedOriginalPattern2 = mover -> pattern2;
	//sSharedCompoundUncertainyInfo = ADCPMover -> GetCurrentUncertaintyInfo();
	
	memset(&sgObjects,0,sizeof(sgObjects));// JLM 12/10/98
	
	Long2EditText(dialog,ADCPBINNUM,sSharedDialogADCPMover->fBinToUse);
	
	{
		sTimeDepList = new CMyList(sizeof(ADCPTimeValue *));
		if(!sTimeDepList)return -1;
		if(sTimeDepList->IList())return -1;
		
		n = sDialogList->GetItemCount() ;
		if (n>0)
		{
			// copy list to temp list
			for(i=0;i<n;i++)
			{	// dvals is a list too in this case...
				//dvals=(*dvalsh)[i];
				sDialogList->GetListItem((Ptr)&currentTimeDep, i);
				err=sTimeDepList->AppendItem((Ptr)&currentTimeDep);
				if(err)return err;
			}
		}
		else  n=0;
		
		n++; // Always have blank row at bottom
		
		err = VLNew(dialog, ADCPDLGLIST, &r,n, DrawTimeDepNameList, &sgObjects);
		if(err) return err;
	}
	
	SetDialogItemHandle(dialog, ADCPDLGHILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, ADCPDLGLIST, (Handle)TimeDepNamesListUpdate);
	
#ifdef IBM
	IBMoffset = r.left;
#else 
	IBMoffset = 0;
#endif
	r = GetDialogItemBox(dialog, ADCPNAME_LIST_LABEL);NAME_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, ADCPNUMBINS_LIST_LABEL);NUMBINS_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, ADCPTOPBINDEPTH_LIST_LABEL);TOPBINDEPTH_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, ADCPTOPBINSIZE_LIST_LABEL);BINSIZE_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, ADCPDEPTH_LIST_LABEL);DEPTH_COL=(r.left+r.right)/2-IBMoffset;
	
	//ShowHideADCPDialogItems(dialog);
	UpdateDisplayWithCurSelection(dialog);
	MySelectDialogItemText(dialog,ADCPBINNUM,0,100);
	
	return 0;
}

OSErr ADCPSettingsDialog(ADCPMover *theMover, TMap *ownerMap, Boolean *timeFileChanged)
{
	short item;
	OSErr err = 0;
	
	if (!theMover)return -1;
	// code goes here, need to deal with deleting the map, canceling, etc.
	// delete sub maps as we go, but check that they are the corresponding mover map
	// check if we delete all movers that any map must be deleted
	// also need to handle changing map - calling routine needs to reset I think
	sSharedDialogADCPMover = theMover;
	
	sDialogList = theMover->timeDepList;
	item = MyModalDialog(ADCPDLG, mapWindow, 0, ADCPDlgInit, ADCPDlgClick);
	sSharedDialogADCPMover = 0;
	
	if(ADCPDLGOK == item)	model->NewDirtNotification();// tell model about dirt
	return ADCPDLGOK == item ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////

ADCPMover *CreateAndInitADCPCurrentsMover (TMap *owner, Boolean askForFile, char* givenPath,char* givenFileName)
{
	TCurrentMover *mover = CreateAndInitCurrentsMover (owner,askForFile,givenPath,givenFileName,nil); // ADCP movers should not have their own map
	//
	if (mover && mover->GetClassID() != TYPE_ADCPMOVER)
	{ // for now, we've assumed the patterns are ADCP Movers
		printError("Non-ADCP Mover created in CreateAndInitADCPCurrentsMover");
		mover -> Dispose();
		delete mover;
		mover = 0;
	}
	return dynamic_cast<ADCPMover *>(mover);
}
OSErr ADCPMover::TextRead(char *path)
{
	// read the header/metadata file, then set up the timedep
	char tempStr[128], shortFileName[64];
	short unitsIfKnownInAdvance = 0;
	OSErr err = 0;
	//ADCPTimeValue *timeFile;
	//timeDep = LoadADCPTimeValue(this,flag); 
	//timeDep = CreateADCPTimeValue(this,path,shortFileName,unitsIfKnownInAdvance);	// ask user for units 
	// if user chose to cancel?
	//if(!timeDep) goto donetimefile;/*break*/; // user canceled or an error 
	/*if (IsADCPFile(path))
	 {
	 //ADCPTimeValue *timeValObj = new ADCPTimeValue(theOwner);
	 timeDep = new ADCPTimeValue(this);
	 
	 if (!timeDep)
	 { TechError("TextRead()", "new ADCPTimeValue()", 0); return -1; }
	 
	 err = timeDep->InitTimeFunc();
	 if(err) {delete timeDep; timeDep = nil; return err;}  
	 
	 err = timeDep->ReadTimeValues (path, M19REALREAL, unitsIfKnownInAdvance);
	 if(err) { delete timeDep; timeDep = nil; return err;}
	 //return timeValObj;
	 }	*/
	if (IsADCPFile(path))
	{
		ADCPTimeValue *timeValObj = new ADCPTimeValue(dynamic_cast<ADCPMover *>(this));
		//timeDep = new ADCPTimeValue(this);
		
		if (!timeValObj)
		{ TechError("TextRead()", "new ADCPTimeValue()", 0); return -1; }
		
		err = timeValObj->InitTimeFunc();
		if(err) {delete timeValObj; timeValObj = nil; return err;}  
		
		err = timeValObj->ReadTimeValues (path, M19REALREAL, unitsIfKnownInAdvance);
		if(err) { delete timeValObj; timeValObj = nil; return err;}
		//return timeValObj;
		dynamic_cast<ADCPMover *>(this)->AddTimeDep(timeValObj,0);
	}	
	// code goes here, add code for OSSMHeightFiles, need scale factor to calculate derivative
	else
	{
		sprintf(tempStr,"File %s is not a recognizable ADCP time file.",shortFileName);
		printError(tempStr);
	}
	return err;
}

void ADCPMover::DrawContourScale(Rect r, WorldRect view)
{
	Point		p;
	short		h,v,x,y,dY,widestNum=0;
	RGBColor	rgb;
	Rect		rgbrect;
	Rect legendRect = fLegendRect;
	char 		numstr[30],numstr2[30],text[30],errmsg[256],stationName[64];
	long 		i,j,numLevels,istep=1,numBins=1;
	double	minLevel, maxLevel;
	double 	value, stationDepth = 0;
	float totalDepth = 0;
	long numDepths = 0, numTris = 0, triNum = 0;
	OSErr err = 0;
	PtCurMap *map = GetPtCurMap();
	ADCPTimeValue *thisTimeDep;
	//TTriGridVel3D *triGrid = (TTriGridVel3D*) map->GetGrid3D(false);
	//Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
	long indexToDepthData = 0, index;
	//long numDepthLevels = GetNumDepthLevelsInFile();
	
	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
#ifdef IBM
	//TextFont(kFontIDGeneva); TextSize(6);
	TextFont(kFontIDGeneva); TextSize(10);	//for Brian
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
	
	for (i = 0; i < timeDepList -> GetItemCount (); i++)
	{
		timeDepList -> GetListItem ((Ptr) &thisTimeDep, i);
		if(thisTimeDep && thisTimeDep->bActive)
		{	// eventually will want to interpolate based on where p is - for now just draw the top listed adcp
			numBins = thisTimeDep->GetNumBins(); 
			break;
			//thisTimeDep->GetDepthIndices(depthAtPoint, totalDepth, &depthIndex1, &depthIndex2);
			//err = thisTimeDep -> GetTimeValue (model -> GetModelTime(), &timeValue); 
		}
	}
	if (!thisTimeDep) return;
	
	thisTimeDep->GetStationName(stationName);
	x = (rgbrect.left + rgbrect.right) / 2;
	//dY = RectHeight(rgbrect) / 12;
	dY = 10;
	y = rgbrect.top + dY / 2;
	MyMoveTo(x - stringwidth("Depth Barbs") / 2, y + dY);
	drawstring("Depth Barbs");
	numtostring(triNum+1,numstr);
	//strcpy(numstr2,"Tri Num = ");
	//strcat(numstr2,numstr);
	strcpy(numstr2,stationName);
	MyMoveTo(x-stringwidth(numstr2) / 2, y + 2*dY);
	drawstring(numstr2);
	widestNum = stringwidth(numstr2);
	
	//v = rgbrect.top+45;
	v = rgbrect.top+40;
	h = rgbrect.left;
	
	for(j=numBins-1;j>=0;j--)	
	{
		WorldPoint wp;
		Point p,p2;
		VelocityRec velocity = {0.,0.};
		Boolean offQuickDrawPlane = false;
		long depthIndex1/*, depthIndex2*/,depthIndex, sensorOrientation;
		Seconds time, startTime, endTime;
		double timeAlpha;
		
		sensorOrientation = thisTimeDep->GetSensorOrientation();
		if (sensorOrientation == 2) 
			depthIndex = numBins - j - 1;	// downward facing
		else
			depthIndex = j;	// upward facing
		
		thisTimeDep->GetTimeValueAtDepth(depthIndex, model->GetModelTime(), &velocity);
		
		MyMoveTo(h+40,v+.5);
		
		if ((velocity.u != 0 || velocity.v != 0))
		{
			//float inchesX = (velocity.u * fVar.curScale) / fVar.arrowScale;
			//float inchesY = (velocity.v * fVar.curScale) / fVar.arrowScale;
			float inchesX = velocity.u;
			float inchesY = velocity.v;
			short pixX = inchesX * PixelsPerInchCurrent();
			short pixY = inchesY * PixelsPerInchCurrent();
			p.h = h+40;
			p.v = v+.5;
			p2.h = p.h + pixX;
			p2.v = p.v - pixY;
			MyLineTo(p2.h, p2.v);
			MyDrawArrow(p.h,p.v,p2.h,p2.v);
		}
		if (p2.h-h>widestNum) widestNum = p2.h-h;	// also issue of negative velocity, or super large value, maybe scale?
		v = v+9;
	}
	stationDepth = thisTimeDep->GetStationDepth();
	sprintf(text, "Depth: %g m",stationDepth);
	//MyMoveTo(x - stringwidth(text) / 2, y + 3 * dY);
	MyMoveTo(x - stringwidth(text) / 2,v+15);
	//MyMoveTo(h+10, v+5);
	//MyMoveTo(h+5, v+5);
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
}

Boolean ADCPMover::DrawingDependsOnTime(void)
{
	//Boolean depends = fVar.bShowArrows;
	// if this is a constant current, we can say "no"
	//if(this->GetNumTimesInFile()==1) depends = false;
	//if(this->GetNumTimesInFile()==1 && !(GetNumFiles()>1)) depends = false;
	//return depends;
	return true;	// will check number of time values
}

void ADCPMover::Draw(Rect r, WorldRect view)
{
	//if(fGrid && (bShowArrows || bShowGrid))
	//fGrid->Draw(r,view,refP,refScale,arrowScale,bShowArrows,bShowGrid);
	Point p,p2;
	short pixX, pixY;
	Boolean offQuickDrawPlane = false;
	//VelocityRec velocity = {1.,1.};
	//double refScale = 10., arrowScale = 1.;
	double refScale = 1., arrowScale = 1.;	//for Brian
	WorldRect mapBounds = moverMap->GetMapBounds(); 
	WorldPoint3D center;
	WorldPoint stationPosition;
	float inchesX, inchesY;
	VelocityRec velocity = {0.,0.};
	Boolean useEddyUncertainty = false;	
	double spillStartDepth = 0.;
	ADCPTimeValue *thisTimeDep;
	Rect c;
	long i,depthIndex1, depthIndex2;
	OSErr err = 0;
	
	if (moverMap->IAm(TYPE_PTCURMAP))
		spillStartDepth = (dynamic_cast<PtCurMap *>(moverMap))->GetSpillStartDepth();
	
	//velocity = this->GetPatValue(wp.p);
	/*center.p = WorldRectCenter(mapBounds);
	 center.z = spillStartDepth;
	 velocity = GetVelocityAtPoint(center);	// draw this at specific depth, full set of depths for legend
	 p = GetQuickDrawPt(center.p.pLong, center.p.pLat, &r, &offQuickDrawPlane);
	 if (bShowGrid && (velocity.u != 0 || velocity.v != 0))
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
	 //DrawArrowHead (p, p2, velocity);
	 //DrawArrowHead(p2, velocity);
	 }*/
	
	for (i = 0; i < timeDepList -> GetItemCount (); i++)
	{
		timeDepList -> GetListItem ((Ptr) &thisTimeDep, i);
		if(thisTimeDep && thisTimeDep->bActive)
		{	// eventually will want to interpolate based on where p is - for now just draw the top listed adcp
			//numBins = thisTimeDep->GetNumBins(); 
			stationPosition = thisTimeDep->GetStationPosition();
			//end = start = WorldToScreenPoint(stationPosition, settings.currentView, MapDrawingRect());
			//MyMoveTo(start->h-offset,start->v);MyLineTo(start->h+offset+xtraOffset,start->v);
			//MyMoveTo(start->h,start->v-offset);MyLineTo(start->h,start->v+offset+xtraOffset);
			p = GetQuickDrawPt(stationPosition.pLong, stationPosition.pLat, &r, &offQuickDrawPlane);
			// draw the reference point
			if (!offQuickDrawPlane &&  bShowGrid)
			{
				RGBForeColor(&colors[BLUE]);
				//MySetRect(&c, p.h - 2, p.v - 2, p.h + 2, p.v + 2);
				MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
				PaintRect(&c);
			}
			RGBForeColor(&colors[BLACK]);
			if (fBinToUse > 0)
			{
				if (thisTimeDep->GetSensorOrientation()==2)
					depthIndex1 = fBinToUse-1;
				else depthIndex1 = thisTimeDep->GetNumBins() - fBinToUse;
				depthIndex2=UNASSIGNEDINDEX;
			}
			else
			{
				if (thisTimeDep->GetSensorOrientation()==2)
					depthIndex1 = 0;
				else depthIndex1 = thisTimeDep->GetNumBins() - 1;
				//depthIndex2=UNASSIGNEDINDEX;
				// use bathymetry or adcp station depth
				//thisTimeDep->GetDepthIndices(spillStartDepth, totalDepth, &depthIndex1, &depthIndex2);
			}
			err = thisTimeDep->GetTimeValueAtDepth(depthIndex1, model->GetModelTime(), &velocity);
			if (!err && bShowGrid && (velocity.u != 0 || velocity.v != 0))
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
				//DrawArrowHead (p, p2, velocity);
				//DrawArrowHead(p2, velocity);
			}
			
		}
	}
	if (!thisTimeDep) return;
	if (bShowArrows) this->DrawContourScale(r,view);
}

/**************************************************************************************************/
CurrentUncertainyInfo ADCPMover::GetCurrentUncertaintyInfo ()
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
void ADCPMover::SetCurrentUncertaintyInfo (CurrentUncertainyInfo info)
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
Boolean ADCPMover::CurrentUncertaintySame (CurrentUncertainyInfo info)
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
/**************************************************************************************************/


