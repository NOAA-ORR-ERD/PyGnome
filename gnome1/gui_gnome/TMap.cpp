/*
 *  TMap.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "TMap.h"
#include "TMover.h"
#include "CROSS.H"
#include "GridCurMover.h"
#include "GridWndMover.h"
#include "TideCurCycleMover.h"
#include "CurrentCycleMover.h"
#include "EditWindsDialog.h"
#include "NetCDFMoverCurv.h"
#include "NetCDFWindMover.h"
#include "NetCDFWindMoverCurv.h"
#include "NetCDFMoverTri.h"

OSErr M21Init(DialogPtr dialog, VOIDPTR data);
short M21Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data);

void TMapOutOfMemoryAlert(char* routineName)
{
	char msg[512];
	char * name = "";
	if(routineName) name = routineName; // re-assign pointer
	sprintf(msg,"There is not enough memory allocated to the program.  Out of memory in TMap %s.",name);
	printError(msg);
}


TMap::TMap(char *name, WorldRect bounds) 
{
	SetMapName(name);
	fMapBounds = bounds;
	
	moverList = 0;
	
	SetDirty(FALSE);
	
	bOpen = TRUE;
	bMoversOpen = TRUE;
	
	fRefloatHalfLifeInHrs = 1.0;
	
	bIAmPartOfACompoundMap = false;
}

void TMap::Dispose()
{
	long i, n;
	TMover *mover;
	
	if (moverList != nil)
	{
		for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
			moverList->GetListItem((Ptr)&mover, i);
			mover->Dispose();
			delete mover;
		}
		
		moverList->Dispose();
		delete moverList;
		moverList = nil;
	}
}

OSErr TMap::ReplaceMap() 
{
	printError("Button not implemented."); 
	return USERCANCEL;	// to return to dialog
}

OSErr TMap::DropMover(TMover *theMover)
{
	long 	i;
	OSErr	err = noErr;
	
	if (moverList->IsItemInList((Ptr)&theMover, &i))
	{
		if (err = moverList->DeleteItem(i))
		{ TechError("TMap::DropMover()", "DeleteItem()", err); return err; }
	}
	SetDirty (true);
	
	return err;
}

//#define kTMapVersion 1
#define kTMapVersion 2

OSErr TMap::Write(BFPB *bfpb)
{
	long i, n, version = kTMapVersion;
	ClassID id = GetClassID ();
	TMover *mover;
	OSErr err = 0;
	
	StartReadWriteSequence("TMap::Write()");
	if (err = WriteMacValue(bfpb,id)) return err;
	if (err = WriteMacValue(bfpb,version)) return err;
	if (err = WriteMacValue(bfpb, className, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb,fMapBounds)) return err;
	if (err = WriteMacValue(bfpb,fRefloatHalfLifeInHrs)) return err;
	if (err = WriteMacValue(bfpb,bOpen)) return err;
	if (err = WriteMacValue(bfpb,bMoversOpen)) return err;
	n = moverList->GetItemCount();
	if (err = WriteMacValue(bfpb,n)) return err;
	
	for (i = 0 ; i < n ; i++) {
		moverList->GetListItem((Ptr)&mover, i);
		id = mover->GetClassID();
		if (err = WriteMacValue(bfpb,id)) return err;
		if (err = mover->Write(bfpb)) return err;
	}
	
	if (err = WriteMacValue(bfpb,bIAmPartOfACompoundMap)) return err;
	
	SetDirty(false);
	return 0;
}

OSErr TMap::Read(BFPB *bfpb)
{
	long i, numMovers, version;
	ClassID id;
	TMover *mover;
	OSErr err = 0;
	
	StartReadWriteSequence("TMap::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TMap::Read()", "id == TYPE_MAP", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > kTMapVersion) { printSaveFileVersionError(); return -1; }
	if (err = ReadMacValue(bfpb, className, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb,&fMapBounds)) return err;
	if (err = ReadMacValue(bfpb,&fRefloatHalfLifeInHrs)) return err;
	if (err = ReadMacValue(bfpb, &bOpen)) return err;
	if (err = ReadMacValue(bfpb, &bMoversOpen)) return err;
	if (err = ReadMacValue(bfpb,&numMovers)) return err;
	
	// allocate and read each of the movers
	
	for (i = 0 ; i < numMovers ; i++) {
		if (err = ReadMacValue(bfpb,&id)) return err;
		mover = 0;
		switch (id) {
			case TYPE_MOVER: mover = new TMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_RANDOMMOVER: mover = new TRandom(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_CATSMOVER: mover = new TCATSMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_WINDMOVER: mover = new TWindMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_COMPONENTMOVER: mover = new TComponentMover(dynamic_cast<TMap *>(this), ""); break;
				//case TYPE_CONSTANTMOVER: mover = new TConstantMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_PTCURMOVER: mover = new PtCurMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_GRIDCURMOVER: mover = new GridCurMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_NETCDFMOVER: mover = new NetCDFMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_NETCDFMOVERCURV: mover = new NetCDFMoverCurv(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_NETCDFMOVERTRI: mover = new NetCDFMoverTri(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_NETCDFWINDMOVER: mover = new NetCDFWindMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_NETCDFWINDMOVERCURV: mover = new NetCDFWindMoverCurv(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_GRIDWNDMOVER: mover = new GridWndMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_RANDOMMOVER3D: mover = new TRandom3D(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_CATSMOVER3D: mover = new TCATSMover3D(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_TRICURMOVER: mover = new TriCurMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_TIDECURCYCLEMOVER: mover = new TideCurCycleMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_COMPOUNDMOVER: mover = new TCompoundMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_ADCPMOVER: mover = new ADCPMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_GRIDCURRENTMOVER: mover = new GridCurrentMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_CURRENTCYCLEMOVER: mover = new CurrentCycleMover(dynamic_cast<TMap *>(this), ""); break;
			case TYPE_GRIDWINDMOVER: mover = new GridWindMover(dynamic_cast<TMap *>(this), ""); break;
			default: printError("Unrecognized mover type in TMap::Read()."); return -1;
		}
		if (!mover)
		{ TechError("TMap::Read()", "new TMover()", 0); return -1; };
		if (!err) err = mover->InitMover();
		
		if (!err) err = mover->Read(bfpb);
		if (!err) {
			err = AddMover(mover, 0);
			if(err)  
				TechError("TMap::Read()", "AddMover()",err);
		}
		
		if(err)
		{ delete mover; mover = 0; return err;}
		
	}
	
	if (version > 1)
		if (err = ReadMacValue(bfpb, &bIAmPartOfACompoundMap)) return err;
	
	return err;
}

OSErr TMap::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	long i, n;
	TMover *mover;
	OSErr err = 0;
	
	char ourName[kMaxNameLen];
	
	// see if the message is of concern to us
	this->GetClassName(ourName);
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		double val;
		
		err = message->GetParameterAsDouble("RefloatHalfLifeInHrs",&val);
		if(!err)
		{	
			this->fRefloatHalfLifeInHrs = val; 
			model->NewDirtNotification();// tell model about dirt
		}
	}
	else if(message->IsMessage(M_CREATEMOVER,ourName)) 
	{
		char moverName[kMaxNameLen]="";
		char typeName[64];
		char path[256];
		char msg[512];
		TMover *mover = nil;
		TCurrentMover *newMover = nil;
		TMap *newMap = 0;
		Boolean unrecognizedType = false;
		Boolean needToInitMover = true;
		message->GetParameterString("NAME",moverName,kMaxNameLen);
		message->GetParameterString("TYPE",typeName,64);
		message->GetParameterString("PATH",path,256);
		ResolvePath(path);
		
		if(!strcmpnocase(typeName,"Random")) mover = new TRandom(dynamic_cast<TMap *>(this), moverName);
		else if(!strcmpnocase(typeName,"Wind")) 
		{
			TWindMover *newWindMover = new TWindMover(dynamic_cast<TMap *>(this), moverName);
			
			if (!newWindMover)  err = memFullErr;
			else
			{ 
				TOSSMTimeValue *timeFile = new TOSSMTimeValue (newWindMover);
				
				newWindMover->InitMover();
				needToInitMover = false;
				newWindMover->SetIsConstantWind(false);
				
				if (!timeFile) err = memFullErr;
				else
				{ 
					if(path[0])
					{
						if(FileExists(0,0,path))
						{
							short unitsIfKnownInAdvance = kUndefined;
							char str2[64], outPath[256];
							message->GetParameterString("speedUnits",str2,64);
							if(str2[0]) 
							{	
								unitsIfKnownInAdvance = StrToSpeedUnits(str2);
								if(unitsIfKnownInAdvance == kUndefined) 
									printError("bad speedUnits parameter");
							}
							
#if TARGET_API_MAC_CARBON
							// ReadTimeValues expects unix style paths
							if (!err) err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
							if (!err) strcpy(path,outPath);
#endif
							err = timeFile -> ReadTimeValues (path, M19MAGNITUDEDIRECTION,unitsIfKnownInAdvance);
						}	
						else 
						{
							sprintf(msg,"PATH to Wind File does not exist.%s%s",NEWLINESTRING,path);
							printError(msg);
							err = true;
						}
						
						if(err) 
						{ delete timeFile; timeFile = 0; }
					}
					if(timeFile)
						newWindMover->SetTimeDep(timeFile);
					
				}
				mover = newWindMover;
			}
		}
		else if(!strcmpnocase(typeName,"GridWind")) 
		{	// the netcdf mover needs a file and so is a special case
			// this will just be a file name, assumed to be with the location file
			// if not, ask for path ?
			char *p, locFilePath[256], netcdfFilePath[256], firstPartOfFile[512], strLine[512], fileNamesPath[256];
			long line, lenToRead = 512;
			short gridType, selectedUnits;
			GridWndMover *newMover = new GridWndMover(this,moverName);
			//newMover -> fUserUnits = selectedUnits;
			if (IsGridWindFile(path,&selectedUnits))			
				newMover -> fUserUnits = selectedUnits;
			else return -1;
			if (err = newMover -> TextRead(path))
			{
				newMover->Dispose(); delete newMover; newMover = 0;
				return -1; 
			}
			///////////////////
			if(err) {
				// it has already been added to the map's list,we need to get rid of it
				this -> DropMover(newMover); 
				newMover->Dispose(); delete newMover;  newMover = 0;
			}			
			//return err;
			mover = newMover;
		}
		else if(!strcmpnocase(typeName,"NetCDFWind")) 
		{	// the netcdf mover needs a file and so is a special case
			// this will just be a file name, assumed to be with the location file
			// if not, ask for path ?
			char *p, locFilePath[256], netcdfFilePath[256], firstPartOfFile[512], strLine[512], fileNamesPath[256];
			long line, lenToRead = 512;
			short gridType;
			Boolean isNetCDFPathsFile = false;
			if (model->fWizard->HaveOpenWizardFile())
			{	// there is a location file
				model->fWizard->GetLocationFileFullPathName(locFilePath);
				p = strrchr(locFilePath,DIRDELIMITER);
				if(p) *(p+1) = 0;
				err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
				
				if (!err)
				{ 
					firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
					NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
					strcpy(netcdfFilePath,locFilePath);
					// add the relative part 
					strcat(netcdfFilePath,strLine);
				}
			}
			else	// assume path is a file (for batch mode), not a file name in the resource
				strcpy(netcdfFilePath,path);
			if (!err)
			{
				if (!FileExists(0,0,netcdfFilePath)) 
				{
					sprintf(msg,"The file %s was not found in %s",strLine,locFilePath); 
					printError(msg);
				}
				else
				{
					//else
					//{
						char topFilePath[256];
						message->GetParameterString("topFile",topFilePath,256);
						if (topFilePath[0]) ResolvePath(topFilePath);
						//newMover = CreateAndInitLocationFileCurrentsMover (dynamic_cast<TMap *>(this),netcdfFilePath,moverName,&newMap,topFilePath);
					//}*/
					if (IsNetCDFFile(netcdfFilePath,&gridType) || IsNetCDFPathsFile(netcdfFilePath, &isNetCDFPathsFile, fileNamesPath, &gridType))
					{
						NetCDFWindMover *newWindMover=nil;
						
						if (gridType == CURVILINEAR)
						{
							newWindMover = new NetCDFWindMoverCurv(dynamic_cast<TMap *>(this),moverName);
							TMap *newMap = 0;
							err =  (dynamic_cast<NetCDFWindMoverCurv *>(newWindMover)) -> TextRead(netcdfFilePath,&newMap,topFilePath);
						}
						else
						{
							newWindMover = new NetCDFWindMover(dynamic_cast<TMap *>(this),moverName);
							err = newWindMover -> TextRead(netcdfFilePath);
						}
						//if(!err) err = NetCDFWindSettingsDialog(newWindMover,this,true,mapWindow);
						//if (!err && this == model -> uMap){
						//ChangeCurrentView(AddWRectBorders(newWindMover->GetGridBounds(), 10), TRUE, TRUE);	// so wind loaded on the universal map can be found
						//}
						/////////////////
						if (!err && isNetCDFPathsFile) /// JLM 5/3/10
						{
							char errmsg[256];
							err = newWindMover->ReadInputFileNames(fileNamesPath);
							if(!err) newWindMover->DisposeAllLoadedData();
							if(!err) err = newWindMover->SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
							
						}
						///////////////////
						if(err) {
							// it has already been added to the map's list,we need to get rid of it
							this -> DropMover(newWindMover); 
							newWindMover->Dispose(); delete newWindMover;  newWindMover = 0;
						}			
						//return err;
						mover = newWindMover;
					}
				}
			}
		}
		else if(!strcmpnocase(typeName,"Component")) mover = new TComponentMover(dynamic_cast<TMap *>(this), moverName); 
		//else if(!strcmpnocase(typeName,"Constant")) mover = new TConstantMover(this, moverName);
		else if(!strcmpnocase(typeName,"Cats")) 
		{	// the cats mover needs a file and so is a special case
			//mover = new TCATSMover(this, moverName);
			mover = CreateAndInitCatsCurrentsMover (dynamic_cast<TMap *>(this),false,path,moverName);
			needToInitMover = false;
		}
		else if(!strcmpnocase(typeName,"Cats3D")) 
		{	// the cats mover needs a file and so is a special case
			//mover = new TCATSMover(this, moverName);
			mover = CreateAndInitCurrentsMover (dynamic_cast<TMap *>(this),false,path,moverName,&newMap);
			needToInitMover = false;
		}
		else if(!strcmpnocase(typeName,"GridCur"))  //might work to just use Cats
		{	
			newMover = CreateAndInitCurrentsMover (dynamic_cast<TMap *>(this),false,path,moverName,&newMap);
			if (newMap) {printError("An error occurred in TMap::CheckAndPassOnMessage()");}
		}
		else if(!strcmpnocase(typeName,"PtCur")) 
		{	// the ptcur mover needs a file and so is a special case
			// this file should not contain a map, since the map is sending the message
			newMover = CreateAndInitCurrentsMover (dynamic_cast<TMap *>(this),false,path,moverName,&newMap);
			if (newMap) {printError("An error occurred in TMap::CheckAndPassOnMessage()");}
		}
		else if(!strcmpnocase(typeName,"NetCDF")) 
		{	// the netcdf mover needs a file and so is a special case
			// this will just be a file name, assumed to be with the location file
			// if not, ask for path ?
			char *p, locFilePath[256], netcdfFilePath[256], firstPartOfFile[512], strLine[512];
			long line, lenToRead = 512;
			if (model->fWizard->HaveOpenWizardFile())
			{	// there is a location file
				model->fWizard->GetLocationFileFullPathName(locFilePath);
				p = strrchr(locFilePath,DIRDELIMITER);
				if(p) *(p+1) = 0;
				err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
				
				if (!err)
				{ 
					firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
					NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
					strcpy(netcdfFilePath,locFilePath);
					// add the relative part 
					strcat(netcdfFilePath,strLine);
				}
			}
			else	// assume path is a file (for batch mode), not a file name in the resource
				strcpy(netcdfFilePath,path);
			if (!err)
			{
				if (!FileExists(0,0,netcdfFilePath)) 
				{
					sprintf(msg,"The file %s was not found in %s",strLine,locFilePath); 
					printError(msg);
				}
				else
				{
					char topFilePath[256];
					message->GetParameterString("topFile",topFilePath,256);
					if (topFilePath[0]) ResolvePath(topFilePath);
					//newMover = CreateAndInitCurrentsMover (this,false,netcdfFilePath,moverName,&newMap);
					newMover = CreateAndInitLocationFileCurrentsMover (dynamic_cast<TMap *>(this),netcdfFilePath,moverName,&newMap,topFilePath);
					//if (newMap) {printError("An error occurred in TMap::CheckAndPassOnMessage()");}
				}
			}
		}
		else if(!strcmpnocase(typeName,"TideCur")) 
		{	// the netcdf mover needs a file and so is a special case
			// this will just be a file name, assumed to be with the location file
			// if not, ask for path ?
			char *p, locFilePath[256], netcdfFilePath[256], firstPartOfFile[512], strLine[512];
			long line, lenToRead = 512;
			if (model->fWizard->HaveOpenWizardFile())
			{	// there is a location file
				model->fWizard->GetLocationFileFullPathName(locFilePath);
				p = strrchr(locFilePath,DIRDELIMITER);
				if(p) *(p+1) = 0;
				err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
				
				if (!err)
				{ 
					firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
					NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
					strcpy(netcdfFilePath,locFilePath);
					// add the relative part 
					strcat(netcdfFilePath,strLine);
				}
			}
			else	// assume path is a file (for batch mode), not a file name in the resource
				strcpy(netcdfFilePath,path);
			if (!err)
			{
				if (!FileExists(0,0,netcdfFilePath)) 
				{
					sprintf(msg,"The file %s was not found in %s",strLine,locFilePath); 
					printError(msg);
				}
				else
				{
					char topFilePath[256];
					message->GetParameterString("topFile",topFilePath,256);
					if (topFilePath[0]) ResolvePath(topFilePath);
					//newMover = CreateAndInitCurrentsMover (this,false,netcdfFilePath,moverName,&newMap);
					newMover = CreateAndInitLocationFileCurrentsMover (dynamic_cast<TMap *>(this),netcdfFilePath,moverName,&newMap,topFilePath);
					//if (newMap) {printError("An error occurred in TMap::CheckAndPassOnMessage()");}
				}
			}
		}
		else
			unrecognizedType = true;
		////////////// 
		if(mover) 
		{
			if (!newMap)
			{
				if(needToInitMover) mover->InitMover();
				this->AddMover(mover, 0);
			}
			else
			{
				Boolean	timeFileChanged = false;
				//newMap->SetClassName(mapName);	// name must match location/command file name to be able to add other movers
				err = model -> AddMap(newMap, 0);
				if (!err) err = AddMoverToMap(newMap, timeFileChanged, mover);
				//if (!err) err = ((PtCurMap*)newMap)->MakeBitmaps();
				if (!err) mover->SetMoverMap(newMap);
				if(err) 
				{
					newMap->Dispose(); delete newMap; newMap = 0; 
					mover->Dispose(); delete mover; mover = 0;
					//return -1; 
				}
			}
		}
		else if(newMover)
		{
			if (!newMap)
				this->AddMover(newMover, 0);	
			else
			{
				Boolean	timeFileChanged = false;
				//newMap->SetClassName(mapName);	// name must match location/command file name to be able to add other movers
				err = model -> AddMap(newMap, 0);
				if (!err) err = AddMoverToMap(newMap, timeFileChanged, newMover);
				//if (!err) err = ((PtCurMap*)newMap)->MakeBitmaps();
				if (!err) newMover->SetMoverMap(newMap);
				if(err) 
				{
					newMap->Dispose(); delete newMap; newMap = 0; 
					newMover->Dispose(); delete newMover; newMover = 0;
					//return -1; 
				}
			}
		}
		else
		{ // error
	 		if(unrecognizedType) printError("Unrecognized mover type in M_CREATEMOVER message");
			else if(err == memFullErr) TMapOutOfMemoryAlert("CheckAndPassOnMessage");
			else printError("An error occurred in TMap::CheckAndPassOnMessage()");
		}
		
	}
	
	// pass this message onto our movers
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++)
	{
		moverList->GetListItem((Ptr)&mover, i);
		err = mover->CheckAndPassOnMessage(message);
	}
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TClassID::CheckAndPassOnMessage(message);
}

Boolean TMap::IsDirty()
{
	long i, n;
	TMover *mover;
	Boolean	bIsDirty = false;
	
	if (TClassID::IsDirty ())
		bIsDirty = true;
	else if (moverList)
	{
		for (i = 0, n = moverList->GetItemCount() ; i < n ; i++)
		{
			moverList->GetListItem((Ptr)&mover, i);
			if (mover->IsDirty())
			{
				bIsDirty = true;
				break;
			}
		}
	}
	
	return bIsDirty;
}

void TMap::Draw(Rect r, WorldRect view)
{
	long i, n;
	Rect m;
	TMover *mover;
	WorldRect ourBounds =   dynamic_cast<TMap *>(this) -> GetMapBounds();
	
	//Boolean offQuickDrawPlane = false;
	//Point pt1 = GetQuickDrawPt(ourBounds.loLong, ourBounds.hiLat, &r, &offQuickDrawPlane);
	//Point pt2 = GetQuickDrawPt(ourBounds.hiLong, ourBounds.loLat, &r, &offQuickDrawPlane);
	//MySetRect(&m, pt1.h, pt1.v, pt2.h, pt2.v);
	
	//MySetRect(&m, SameDifferenceX(ourBounds.loLong),
	//			  (r.bottom + r.top) - SameDifferenceY(ourBounds.hiLat),
	//			  SameDifferenceX(ourBounds.hiLong),
	//			  (r.bottom + r.top) - SameDifferenceY(ourBounds.loLat));
	
	m = WorldToScreenRect(ourBounds,view,r);
	
	
	PenNormal();
	// Draw rectangular map bounds if no alternative map bounds exist
	if (!dynamic_cast<TMap *>(this)->HaveMapBoundsLayer() /*&& !this->IsIceMap()*/) MyFrameRect(&m);
	
	// for large bnas with time dependent currents showing, drawing is very slow
	// if more than one map on top of each other don't want movers all on top map
	if (model->DrawingDependsOnTime() && model->GetMapCount()==1)
		return;
	// draw each of the movers
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
		moverList->GetListItem((Ptr)&mover, i);
		mover->Draw(r, view);
	}
}

Boolean TMap::DrawingDependsOnTime(void)
{
	long i, m;
	CMyList *moverList;
	TMover *thisMover;
	ClassID id;
	Boolean depends;
	
	moverList = this -> GetMoverList ();
	for (i = 0, m = moverList->GetItemCount() ; i < m ; i++) 
	{
		moverList->GetListItem((Ptr)&thisMover, i);
		id = thisMover->GetClassID();
		
		depends = thisMover -> DrawingDependsOnTime();
		if(depends)
			return true;
		else
		{
			// continue and ask the other movers
		}
	}
	return false;
}

long TMap::GetListLength()
{
	long i, n, count = 1;
	TMover *mover;
	
	if (bOpen) {
		count += 1;		// map type & name item
		
		if (bMoversOpen)
			for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
				moverList->GetListItem((Ptr)&mover, i);
				count += mover->GetListLength();
			}
		
		//count++;// the refloat time
	}
	
	return count;
}

ListItem TMap::GetNthListItem(long n, short indent, short *style, char *text)
{
	long i, m, count;
	TMover *mover;
	ListItem item = { this, 0, indent, 0 };
	
	if (n == 0) {
		item.index = I_MAPNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		strcpy(text, className);
		
		return item;
	}
	n -= 1;
	
	if (bOpen) {
		
		/*if (n == 0) {
		 // refloat time
		 item.index = I_REFLOATHALFLIFE;
		 item.indent = indent;
		 sprintf(text, "Refloat half life: %g hr",fRefloatHalfLifeInHrs);
		 
		 return item;
		 }
		 n -= 1;*/
		/////////////
		
		indent++;
		if (n == 0) {
			item.index = I_MOVERS;
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
					return mover->GetNthListItem(n, indent + 1, style, text);
				
				n -= count;
			}
		
		
	}
	
	item.owner = 0;
	
	return item;
}

Boolean TMap::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_MAPNAME: bOpen = !bOpen; return TRUE;
			case I_MOVERS: bMoversOpen = !bMoversOpen; return TRUE;
		}
	
	if (doubleClick)
		if (this -> FunctionEnabled(item, SETTINGSBUTTON)) {
			this -> SettingsItem (item);
		}
		else if (this -> FunctionEnabled(item, ADDBUTTON)) {
			this -> AddItem (item);
		}
	
	
	// do other click operations...
	
	return FALSE;
}

Boolean TMap::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	
	switch (item.index) {
		case I_MAPNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
					//case SETTINGSBUTTON: return FALSE; // TRUE
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
			/*case I_REFLOATHALFLIFE:
			 switch (buttonID) {
			 case ADDBUTTON: return FALSE;
			 case SETTINGSBUTTON: return TRUE;
			 case DELETEBUTTON: return FALSE;
			 }
			 break;*/
		case I_MOVERS:
			switch (buttonID) {
				case ADDBUTTON: return TRUE;
				case SETTINGSBUTTON: return FALSE;
				case DELETEBUTTON: return FALSE;
			}
			break;
	}
	
	return FALSE;
}

OSErr TMap::UpItem(ListItem item)
{
	long i;
	OSErr err = 0;
	
	if (item.index == I_MAPNAME)
		if (model->mapList->IsItemInList((Ptr)&item.owner, &i))
			if (i > 0) {
				if (err = model->mapList->SwapItems(i, i - 1))
				{ TechError("TMap::UpItem()", "model->mapList->SwapItems()", err); return err; }
				SelectListItem(item);
				UpdateListLength(true);
				InvalidateMapImage();
				InvalMapDrawingRect();
			}
	
	return 0;
}

OSErr TMap::DownItem(ListItem item)
{
	long i;
	OSErr err = 0;
	
	if (item.index == I_MAPNAME)
		if (model->mapList->IsItemInList((Ptr)&item.owner, &i))
			if (i < (model->mapList->GetItemCount() - 1)) {
				if (err = model->mapList->SwapItems(i, i + 1))
				{ TechError("TMap::UpItem()", "model->mapList->SwapItems()", err); return err; }
				SelectListItem(item);
				UpdateListLength(true);
				InvalidateMapImage();
				InvalMapDrawingRect();
			}
	
	return 0;
}

OSErr AddMoverToMap(TMap *map, Boolean timeFileChanged, TMover *theMover)
{
	OSErr err;
	
	if (err = map->AddMover(theMover, 0))
	{ 
		theMover->Dispose(); 
		delete theMover; 
		return -1;
	}
	else
	{
		model->NewDirtNotification();
	}
	return 0;
}

OSErr TMap::AddItem(ListItem item)
{
	Boolean	timeFileChanged = false;
	short type, dItem;
	OSErr err = 0;
	
	if (item.index == I_MOVERS || item.index == I_UMOVERS || item.index == I_VMOVERS) {
		dItem = MyModalDialog (M21, mapWindow, (Ptr) &type, M21Init, M21Click);
		if (dItem == M21LOAD)
		{
			switch (type)
			{
				case CURRENTS_MOVERTYPE:
				{
					TMap *newMap = 0;
					TCurrentMover *newMover = CreateAndInitCurrentsMover (dynamic_cast<TMap *>(this),true,0,0,&newMap);
					if (newMover)
					{
						switch (newMover->GetClassID()) 
						{
							case TYPE_CATSMOVER:
							case TYPE_TIDECURCYCLEMOVER:
							case TYPE_CATSMOVER3D:
								err = CATSSettingsDialog (dynamic_cast<TCATSMover *>(newMover), dynamic_cast<TMap *>(this), &timeFileChanged);
								break;
							case TYPE_ADCPMOVER:
								err = ADCPSettingsDialog (dynamic_cast<ADCPMover*>(newMover), dynamic_cast<TMap *>(this), &timeFileChanged);
								break;
							case TYPE_CURRENTCYCLEMOVER:
								err = CurrentCycleSettingsDialog (dynamic_cast<CurrentCycleMover *>(newMover), dynamic_cast<TMap *>(this), &timeFileChanged);
								break;
							case TYPE_GRIDCURRENTMOVER:
							case TYPE_NETCDFMOVER:
							case TYPE_NETCDFMOVERCURV:
							case TYPE_NETCDFMOVERTRI:
							case TYPE_GRIDCURMOVER:
							case TYPE_PTCURMOVER:
							case TYPE_TRICURMOVER:
								err = newMover->SettingsDialog();
								break;
							default:
								printError("bad type in TMap::AddItem");
								break;
						}
						if(err)	{ newMover->Dispose(); delete newMover; newMover = 0;}
						
						if(newMover && !err)
						{	
							if (!newMap) 
							{
								WorldRect mapBounds = this -> GetMapBounds();
								err = AddMoverToMap (this, timeFileChanged, newMover);
								if (!err && (this == model -> uMap || (mapBounds.loLong==-179000000 && mapBounds.hiLong==179000000)))
								{
									//ChangeCurrentView(AddWRectBorders(((PtCurMover*)newMover)->fGrid->GetBounds(), 10), TRUE, TRUE);
									WorldRect newRect = newMover->GetGridBounds();
									ChangeCurrentView(AddWRectBorders(newRect, 10), TRUE, TRUE);	// so current loaded on the universal map can be found
								}
							}
							else
							{
								float arrowDepth = 0;
								MySpinCursor();
								err = model -> AddMap(newMap, 0);
								if (!err) err = AddMoverToMap(newMap, timeFileChanged, newMover);
								//if (!err) err = ((PtCurMap*)newMap)->MakeBitmaps();
								if (!err) newMover->SetMoverMap(newMap);
								else
								{
									newMap->Dispose(); delete newMap; newMap = 0; 
									newMover->Dispose(); delete newMover; newMover = 0;
									return -1; 
								}
								//if (newMover->IAm(TYPE_CATSMOVER3D) || newMover->IAm(TYPE_TRICURMOVER)) InitAnalysisMenu();
								if (model->ThereIsA3DMover(&arrowDepth)) InitAnalysisMenu();	// want to have it come and go?
								MySpinCursor();
							}
						}	
					}
					break;
				}
					
				case WIND_MOVERTYPE: 
				{	
					char path[256];
					short gridType;
					short selectedUnits;
					Boolean isNetCDFPathsFile = false;
					char fileNamesPath[256];
					TimeGridVel *timeGrid = 0;
					GridWindMover *newMover = 0;
					//TMap **newMap = 0;
					char outPath[256], topFilePath[256];
					topFilePath[0]=0;
					
					if (err = GetWindFilePath(path)) return -1;
					if (IsNetCDFFile(path,&gridType) || IsNetCDFPathsFile(path, &isNetCDFPathsFile, fileNamesPath, &gridType))
					{
#ifdef GUI_GNOME
						if (gridType == CURVILINEAR)
						{
							timeGrid = new TimeGridWindCurv();
						}
						else
						{
							timeGrid = new TimeGridWindRect();
							//timeGrid = new TimeGridVel();
						}
						if (timeGrid)
						{
							Point where;
							OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
							MySFReply reply;
							Boolean bTopFile = false;
							//char outPath[256], topFilePath[256];
							//topFilePath[0]=0;
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
							
							GridWindMover *newGridWindMover = new GridWindMover(dynamic_cast<TMap *>(this),"");
							if (!newGridWindMover)
							{ 
								TechError("TMap::AddItem()", "new GridWindMover()", 0);
								return 0;
							}
							newMover = newGridWindMover;
							
							err = newGridWindMover->InitMover(timeGrid);
							//if(err) goto Error;
#if TARGET_API_MAC_CARBON
							if (!err) err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
							if (!err) strcpy(path,outPath);
							if (!err && bTopFile) err = ConvertTraditionalPathToUnixPath((const char *) topFilePath, outPath, kMaxNameLen) ;
							if (!err && bTopFile) strcpy(topFilePath,outPath);
#endif
							//if (!err) err = timeGrid->TextRead(path,"");
							if (!err) err = timeGrid->TextRead(path,topFilePath);
							//if(err) goto Error;
						}
						if(!err) err = GridWindSettingsDialog(newMover,this,true,mapWindow);
						/////////////////
						if (!err && isNetCDFPathsFile) /// JLM 5/3/10
						{
							 //char errmsg[256];
							 err = timeGrid->ReadInputFileNames(fileNamesPath);
							 if(!err) timeGrid->DisposeAllLoadedData();
							 //if(!err) err = newMover->SetInterval(errmsg); // if set interval here will get error if times are not in model range
						}
#else
						NetCDFWindMover *newMover=nil;
						
						if (gridType == CURVILINEAR)
						{
							char topFilePath[256];
							topFilePath[0] = 0;
							newMover = new NetCDFWindMoverCurv(dynamic_cast<TMap *>(this),"");
							TMap *newMap = 0;
							 err = (dynamic_cast<NetCDFWindMoverCurv *>(newMover)) -> TextRead(path,&newMap,topFilePath);
						}
						else
						{
							newMover = new NetCDFWindMover(dynamic_cast<TMap *>(this),"");
							err = newMover -> TextRead(path);
						}
						if(!err) err = NetCDFWindSettingsDialog(newMover,dynamic_cast<TMap *>(this),true,mapWindow);
						/*if (!err && this == model -> uMap){
							ChangeCurrentView(AddWRectBorders(newMover->GetGridBounds(), 10), TRUE, TRUE);	// so wind loaded on the universal map can be found
						}*/
						/////////////////
						if (!err && isNetCDFPathsFile) /// JLM 5/3/10
						{
							char errmsg[256];
							err = newMover->ReadInputFileNames(fileNamesPath);
							if(!err) newMover->DisposeAllLoadedData();
							//if(!err) err = newMover->SetInterval(errmsg); // if set interval here will get error if times are not in model range
						}
#endif
						if (!err && this == model -> uMap){
							ChangeCurrentView(AddWRectBorders(newMover->GetGridBounds(), 10), TRUE, TRUE);	// so wind loaded on the universal map can be found
						}
						///////////////////
						if(err) {
							// it has already been added to the map's list,we need to get rid of it
							this -> DropMover(newMover); 
							newMover->Dispose(); delete newMover;  newMover = 0;
							return err;
						}			
						//return err;
					}
					else if (IsGridWindFile(path,&selectedUnits))	// code goes here, constant wind case
					{
#ifdef GUI_GNOME
						timeGrid = new TimeGridCurRect();
						
						if(timeGrid)
						{
							GridWindMover *newGridWindMover = new GridWindMover(dynamic_cast<TMap *>(this),"");
							if (!newGridWindMover)
							{ 
								TechError("TMap::AddItem()", "new GridWindMover()", 0);
								return 0;
							}
							newMover = newGridWindMover;
							(dynamic_cast<TimeGridCurRect *>(timeGrid)) -> fUserUnits = selectedUnits;
							
							err = newGridWindMover->InitMover(timeGrid);
							//if(err) goto Error;
#if TARGET_API_MAC_CARBON
							if (!err) err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
							if (!err) strcpy(path,outPath);
#endif
							//if (!err) err = timeGrid->TextRead(path,"");
							if (!err) err = timeGrid->TextRead(path,topFilePath);
							//if(err) goto Error;
							if(!err) err = GridWindSettingsDialog(newMover,this,true,mapWindow);
							/////////////////
							if (!err /*&& isNetCDFPathsFile*/) /// JLM 5/3/10
							{
								char errmsg[256];
								//err = timeGrid->ReadInputFileNames(fileNamesPath);
								//if(!err) timeGrid->DisposeAllLoadedData();
								//if(!err) err = timeGrid->SetInterval(errmsg,model->GetModelTime()); // if set interval here will get error if times are not in model range
							}
						}
#else
						GridWndMover *newMover = new GridWndMover(dynamic_cast<TMap *>(this),"");
						newMover -> fUserUnits = selectedUnits;
						if (err = newMover -> TextRead(path))
						{
							newMover->Dispose(); delete newMover; newMover = 0;
							return -1; 
						}
						//err = GridWindSettingsDialog(newMover,this,true,mapWindow);
						err = WindSettingsDialog(newMover,dynamic_cast<TMap *>(this),true,mapWindow,false);
#endif
						if(err)	{ newMover->Dispose(); delete newMover; newMover = 0; return err;}
						//return err;
					}
					else
					{
						// add a new wind mover
						TOSSMTimeValue 	*timeFile = nil;
						TWindMover 		*newMover = nil;
						newMover = new TWindMover(dynamic_cast<TMap *>(this),""); //JLM, the fileName will be picked up in ReadTimeValues()
						if (!newMover)
						{ TechError("TMap::AddItem()", "new TWindMover()", 0); return -1; }
						
						timeFile = new TOSSMTimeValue (newMover);
						if (!timeFile)
						{ TechError("TMap::AddItem()", "new TOSSMTimeValue()", 0); delete newMover; return -1; }
						
#if TARGET_API_MAC_CARBON
						// ReadTimeValues expects unix style paths
						if (!err) err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
						if (!err) strcpy(path,outPath);
#endif
						if (err = timeFile -> ReadTimeValues (path, M19MAGNITUDEDIRECTION, kUndefined)) // ask for units
						{ delete timeFile; delete newMover; return -1; }
						newMover->timeDep = timeFile;
						
						err = WindSettingsDialog(newMover,dynamic_cast<TMap *>(this),true,mapWindow,false);
						if(err)	{ newMover->Dispose(); delete newMover; newMover = 0;}
						return err;
						
						//return WindSettingsDialog   (0, this,false,mapWindow);
					}
					if (timeGrid)
					{
						Seconds startTime = timeGrid->GetTimeValue(0);
						if (startTime != CONSTANTWIND && (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime()))
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
										//goto Error;
								}
							}
						}
					}
				}
					
				default: 
					SysBeep (5); 
					return -1;
			}
		}
		else if (dItem == M21CREATE)
		{
			Boolean isConstantWind = false;
			switch (type)
			{
				case RANDOM_MOVERTYPE: 
					// just check if there is a ptcurmap and put diffusion wherever it was requested
					if (this->IAm(TYPE_PTCURMAP)) 
					{	// code goes here, careful this remains hidden from casual user
						float arrowDepth = 0;
						if (gDispersedOilVersion && model->ThereIsA3DMover(&arrowDepth))	// hopefully redundant
							//TMover *mover = ((PtCurMap*)this)->GetMover(TYPE_PTCURMOVER);
							//if (mover && ((PtCurMover*)mover)->fVar.gridType != TWO_D)
							return Random3DSettingsDialog	(0, this);	// code goes here, also check gridtype...
						/*if (!mover)
						 {
						 mover = ((PtCurMap*)this)->GetMover(TYPE_CATSMOVER3D);
						 if (mover)
						 return Random3DSettingsDialog(0,this);
						 else
						 {
						 mover = ((PtCurMap*)this)->GetMover(TYPE_TRICURMOVER);
						 if (mover)
						 return Random3DSettingsDialog(0,this);
						 }
						 }*/
					}
					else if (this->IAm(TYPE_MAP)) 
					{	// for universal mover
						float arrowDepth = 0;
						if (model->ThereIsASubsurfaceSpill()) return Random3DSettingsDialog(0,this);
						//for now ThereIsA3DMover is always set to true
						//if (model->ThereIsA3DMover(&arrowDepth)) return Random3DSettingsDialog(0,this);
						/*PtCurMap *map = GetPtCurMap();
						if (map)
						{
							TMover *mover = map->GetMover(TYPE_PTCURMOVER);
							if (mover && ((PtCurMover*)mover)->fVar.gridType != TWO_D)
								return Random3DSettingsDialog	(0, this);	// code goes here, also check gridtype...
							if (!mover)
							{
								mover = map->GetMover(TYPE_CATSMOVER3D);
								if (mover)
									return Random3DSettingsDialog(0,this);
								else
								{
									//mover = (dynamic_cast<PtCurMap *>(this))->GetMover(TYPE_TRICURMOVER);
									mover = map->GetMover(TYPE_TRICURMOVER);
									if (mover)
										return Random3DSettingsDialog(0,this);
								}
							}
						}*/
					}
					return RandomSettingsDialog   (0, this);
				case CONSTANT_MOVERTYPE: 
					isConstantWind = true;
					// fall thru
				case WIND_MOVERTYPE: 
				{
					Boolean settingsForcedAfterDialog = true;
					TWindMover *newMover = new TWindMover(dynamic_cast<TMap *>(this), "");
					if (!newMover) { TechError("WindSettingsDialog()", "new TWindMover()", 0); return -1; }
					TOSSMTimeValue *timeFile = new TOSSMTimeValue (newMover);
					if (!timeFile) { TechError("WindSettingsDialog()", "new TOSSMTimeValue()", 0); delete newMover; return -1; }
					
					newMover->SetTimeDep(timeFile);
					newMover->SetIsConstantWind(isConstantWind);
					if(isConstantWind) newMover->SetClassName("Constant Wind");
					// err= EditWindsDialog(nil,timeFile,model->GetStartTime(),false);
					// JLM 12/14/98
				doEditWindsDialog:
					err= EditWindsDialog(newMover,model->GetStartTime(),false,settingsForcedAfterDialog);
					if(!err) 
					{
						err =  WindSettingsDialog(newMover,dynamic_cast<TMap *>(this),true,mapWindow,false);
						if(err == USERCANCEL)
						{
							err = 0;
							goto doEditWindsDialog; // go back !!
						}
					}
					if(err) 
					{
						newMover->Dispose(); 
						delete newMover; 
						return err;
					}
				}
					break;
					
				case COMPONENT_MOVERTYPE:
				{
					//TComponentMover *newMover = new TComponentMover (this, "");
					TComponentMover *newMover = new TComponentMover (dynamic_cast<TMap *>(this), "Component");	// give mover a name so we can send messages (North Slope Loc File)
					if (!newMover) { TechError("TMap::AddItem()", "new TComponentMover()", 0); return -1; }
					
					newMover -> InitMover ();
					
					err = ComponentMoverSettingsDialog (newMover, dynamic_cast<TMap *>(this));
					if(err){
						// error or perhaps the user canceled
						newMover->Dispose(); 
						delete newMover; 
						return err;
					}
					
					err = AddMoverToMap (dynamic_cast<TMap *>(this), timeFileChanged, newMover);
					
					break;
				}
				case COMPOUND_MOVERTYPE:
				{	// maybe don't allow to add to a map?, or only to bna?
					TMap *newMap = 0;
					TCompoundMover *newMover = new TCompoundMover (dynamic_cast<TMap *>(this), "");
					if (!newMover) { TechError("TMap::AddItem()", "new TCompoundMover()", 0); return -1; }
					
					newMover -> InitMover ();
					
					// do a cycle of load currents or allow repeated Load on ComponentMover style dialog
					err = CompoundMoverSettingsDialog (newMover, dynamic_cast<TMap *>(this), &newMap);
					if(err){
						// error or perhaps the user canceled
						newMover->Dispose(); 
						delete newMover; 
						return err;
					}
					if(newMover && !err)
					{	
						if (!newMap) 
						{
							err = AddMoverToMap (this, timeFileChanged, newMover);
							if (!err && this == model -> uMap)
							{
								//ChangeCurrentView(AddWRectBorders(((PtCurMover*)newMover)->fGrid->GetBounds(), 10), TRUE, TRUE);
								WorldRect newRect = newMover->GetGridBounds();
								ChangeCurrentView(AddWRectBorders(newRect, 10), TRUE, TRUE);	// so current loaded on the universal map can be found
							}
						}
						else
						{
							float arrowDepth = 0;
							MySpinCursor();
							err = model -> AddMap(newMap, 0);
							if (!err) err = AddMoverToMap(newMap, timeFileChanged, newMover);
							//if (!err) err = ((PtCurMap*)newMap)->MakeBitmaps();
							if (!err) newMover->SetMoverMap(newMap);
							else
							{
								newMap->Dispose(); delete newMap; newMap = 0; 
								newMover->Dispose(); delete newMover; newMover = 0;
								return -1; 
							}
							//if (newMover->IAm(TYPE_CATSMOVER3D) || newMover->IAm(TYPE_TRICURMOVER)) InitAnalysisMenu();
							if (model->ThereIsA3DMover(&arrowDepth)) InitAnalysisMenu();	// want to have it come and go?
							MySpinCursor();
						}
					}	
					
					//err = AddMoverToMap (this, timeFileChanged, newMover);
					
					break;
				}
				case CURRENTS_MOVERTYPE:
				{
					printNote("This option has not been implemented yet");
					// code goes here - need a new basic current mover type - sort of like wind mover
					/*TCurrentMover *newMover = CreateAndInitCurrentsMover (this,true,0,0,&newMap);
					 if (newMover)
					 {
					 switch (newMover->GetClassID()) 
					 {
					 case TYPE_CATSMOVER:
					 case TYPE_TIDECURCYCLEMOVER:
					 err = CATSSettingsDialog ((TCATSMover*)newMover, this, &timeFileChanged);
					 break;
					 case TYPE_NETCDFMOVER:
					 case TYPE_NETCDFMOVERCURV:
					 case TYPE_NETCDFMOVERTRI:
					 case TYPE_GRIDCURMOVER:
					 case TYPE_PTCURMOVER:
					 case TYPE_TRICURMOVER:
					 err = newMover->SettingsDialog();
					 break;
					 default:
					 printError("bad type in TMap::AddItem");
					 break;
					 }
					 if(err)	{ newMover->Dispose(); delete newMover; newMover = 0;}
					 
					 if(newMover && !err)
					 {	
					 if (!newMap) 
					 err = AddMoverToMap (this, timeFileChanged, newMover);
					 else
					 {
					 MySpinCursor();
					 err = model -> AddMap(newMap, 0);
					 if (!err) err = AddMoverToMap(newMap, timeFileChanged, newMover);
					 if (!err) newMover->SetMoverMap(newMap);
					 else
					 {
					 newMap->Dispose(); delete newMap; newMap = 0; 
					 newMover->Dispose(); delete newMover; newMover = 0;
					 return -1; 
					 }
					 MySpinCursor();
					 }
					 }	
					 }*/
					break;
					/*Boolean settingsForcedAfterDialog = true;
					 TConstantCurrentMover *newMover = new TConstantCurrentMover(this, "");
					 if (!newMover) { TechError("TMap::AddItem()", "new TConstantCurrentMover()", 0); return -1; }
					 TOSSMTimeValue *timeFile = new TOSSMTimeValue (newMover);
					 if (!timeFile) { TechError("TMap::AddItem()", "new TOSSMTimeValue()", 0); delete newMover; return -1; }
					 
					 newMover->SetTimeDep(timeFile);
					 newMover->SetIsConstantWind(isConstantWind);
					 if(isConstantWind) newMover->SetClassName("Constant Wind");
					 // err= EditWindsDialog(nil,timeFile,model->GetStartTime(),false);
					 // JLM 12/14/98
					 doEditWindsDialog:
					 err= EditCurrentsDialog(newMover,model->GetStartTime(),false,settingsForcedAfterDialog);
					 if(!err) 
					 {
					 err =  WindSettingsDialog(newMover,this,true,mapWindow,false);
					 if(err == USERCANCEL)
					 {
					 err = 0;
					 goto doEditWindsDialog; // go back !!
					 }
					 }
					 if(err) 
					 {
					 newMover->Dispose(); 
					 delete newMover; 
					 return err;
					 }*/
				}
				default: SysBeep (5); break;
			}
		}
	}
	
	return err;
	
}

OSErr TMap::SettingsItem(ListItem item)
{
	return TMapSettingsDialog(dynamic_cast<TMap *>(this));
}

OSErr TMap::DeleteItem(ListItem item)
{
	if (item.index == I_MAPNAME)
		return model->DropMap(dynamic_cast<TMap *>(this));
	
	return 0;
}
