
#include "CROSS.h"
#include "OUtils.h"
#include "MapUtils.h"
#include "OSSM.h"

#include "TShioTimeValue.h"
#include "EditWindsDialog.h"

#ifdef MAC
#pragma segment TTimeValues
#endif

#include "TimeValuesIOPD.h"
#include "TimeValuesIO.h"

OSErr GetScaleFactorFromUser(char *msg, double *scaleFactor)
{
	long itemHit;
	char defaultAns[256];
	char ans[256];
	OSErr err = 0;
	sprintf(defaultAns,lfFix("%.1lf"),1.0);
	while(1)
	{
		// ask for a scale factor
		itemHit = REQUEST(msg,defaultAns,ans);
		if (itemHit == 2)  // user cancelled
		{
			err = -1;
			break;
		}
		short numScanned = sscanf(ans,lfFix("%lf"),scaleFactor);
		if(! numScanned == 1) 
		{
			printError("Error reading scale factor. Invalid entry.");
			err = -1;
		}
		
		if(*scaleFactor > 0 /*&& *scaleFactor <= maxScaleFactor*/) // valid entry, go on
			break;
		
		else // go back to dialog
		{
			//if (*scaleFactor > maxScaleFactor) sprintf(msg,lfFix("Maximum scale factor is %.1lf"),maxScaleFactor);
			if (*scaleFactor <= 0) sprintf(msg,"Scale factor must be positive");
			
			printNote(msg); 
		}
	}
	return err;
}


TOSSMTimeValue* CreateTOSSMTimeValue(TMover *theOwner,char* path, char* shortFileName, short unitsIfKnownInAdvance)
{
	char tempStr[256], outPath[256];
	OSErr err = 0;
	
	// will need to update the shio readtimevalues
	if(IsShioFile(path))
	{
		TShioTimeValue *timeValObj = new TShioTimeValue(theOwner);
		if (!timeValObj)
		{ TechError("LoadTOSSMTimeValue()", "new TShioTimeValue()", 0); return nil; }
		
		err = timeValObj->InitTimeFunc();
		if(err) {delete timeValObj; timeValObj = nil; return nil;}  
		err = timeValObj->ReadTimeValues (path, M19REALREAL, unitsIfKnownInAdvance);
		if(err) { delete timeValObj; timeValObj = nil; return nil;}
		return timeValObj;
	}
	else 
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		if (!err) strcpy(path,outPath);
#endif
		
		if (IsTimeFile(path) || IsHydrologyFile(path) || IsOSSMTimeFile(path, &unitsIfKnownInAdvance))
		{
			TOSSMTimeValue *timeValObj = new TOSSMTimeValue(theOwner);
			
			if (!timeValObj)
			{ TechError("LoadTOSSMTimeValue()", "new TOSSMTimeValue()", 0); return nil; }
			
			err = timeValObj->InitTimeFunc();
			if(err) {delete timeValObj; timeValObj = nil; return nil;}  
			
			err = timeValObj->ReadTimeValues (path, M19REALREAL, unitsIfKnownInAdvance);
			if(err) { delete timeValObj; timeValObj = nil; return nil;}
			return timeValObj;
		}	
		// code goes here, add code for OSSMHeightFiles, need scale factor to calculate derivative
		else
		{
			sprintf(tempStr,"File %s is not a recognizable time file.",shortFileName);
			printError(tempStr);
		}
	}
		
	return nil;
}

TOSSMTimeValue* LoadTOSSMTimeValue(TMover *theOwner, short unitsIfKnownInAdvance)
{
	char path[256],shortFileName[256];
	char tempStr[256];
	Point where = CenteredDialogUpLeft(M38d);
	WorldPoint p;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	OSErr err = 0;
	
#if TARGET_API_MAC_CARBON
	mysfpgetfile(&where, "", -1, typeList,
				 (MyDlgHookUPP)0, &reply, M38d, MakeModalFilterUPP(STDFilter));
	if (!reply.good) return 0;
	strcpy(path, reply.fullPath);
	strcpy(tempStr,path);
	SplitPathFile(tempStr,shortFileName);
#else
	sfpgetfile(&where, "",
			   (FileFilterUPP)0,
			   -1, typeList,
			   (DlgHookUPP)0,
			   &reply, M38d,
			   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	
	if (!reply.good) return nil; // user canceled
	
	my_p2cstr(reply.fName);
#ifdef MAC
	GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
	strcpy(shortFileName,(char*) reply.fName);
#else
	strcpy(path, reply.fName);
	strcpy(tempStr,path);
	SplitPathFile(tempStr,shortFileName);
#endif
#endif	
	
	//	return  CreateTOSSMTimeValue(theOwner,path,shortFileName,kUndefined);	// ask user for units 
	return  CreateTOSSMTimeValue(theOwner,path,shortFileName,unitsIfKnownInAdvance);	// ask user for units 
}


/////////////////////////////////////////////////


static short gSUUnits; // don't use data, I think it might be buggy, JLM
OSErr SUInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	short selection = gSUUnits;
	SetDialogItemHandle(dialog, SU_HILITEDEFAULT, (Handle) FrameDefault);
	RegisterPopUpDialog(SELECTUNITS, dialog);
	SetPopSelection (dialog, SU_UNITS, selection); 
	
	return 0;
}

short SUClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	long	menuID_menuItem;
	short selectedUnits;
	
	switch (itemNum) 
	{
		case SU_CANCEL: return SU_CANCEL;
			
		case SU_OK:
			selectedUnits = GetPopSelection (dialog, SU_UNITS); 
			gSUUnits = selectedUnits;
			return SU_OK;
			
		case SU_UNITS:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
	}
	
	return 0;
}

static PopInfoRec selectUnitsPopTable[] = {
	{ SELECTUNITS, nil, SU_UNITS, 0, pSPEEDUNITS, 0, 1, FALSE, nil }
};

OSErr AskUserForUnits(short* selectedUnits,Boolean *userCancel)
{
	OSErr err = 0;
	short item;
	PopTableInfo saveTable = SavePopTable();
	short j,numItems= 0;
	PopInfoRec combinedUnitsPopTable[20];
	
	// code to allow a dialog on top of another with pops
	for(j = 0; j < sizeof(selectUnitsPopTable) / sizeof(PopInfoRec);j++)
		combinedUnitsPopTable[numItems++] = selectUnitsPopTable[j];
	for(j= 0; j < saveTable.numPopUps ; j++)
		combinedUnitsPopTable[numItems++] = saveTable.popTable[j];
	
	RegisterPopTable(combinedUnitsPopTable,numItems);
	
	*userCancel = false;
	gSUUnits = *selectedUnits;
	item = MyModalDialog(SELECTUNITS, mapWindow,nil, SUInit, SUClick);
	RestorePopTableInfo(saveTable);
	if(item == SU_CANCEL) 
	{
		*userCancel = true;
		return USERCANCEL; 
	}

	if(item == SU_OK) 
	{
		*selectedUnits = gSUUnits;
		return 0; // JLM 7/8/98
	}
	
	return -1;
}

