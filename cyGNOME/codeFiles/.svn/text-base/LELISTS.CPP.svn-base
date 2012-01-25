
#include "CROSS.h"
#include "TimUtils.h"


#ifdef MAC
	#pragma segment LELISTS
#endif

static ListItem sharedItem;

OSErr M12Init(DialogPtr dialog, VOIDPTR data);
short M12Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data);

/////////////////////////////////////////////////
/////////////////////////////////////////////////
static short timeH, evapH, dispH, removedH;
AdiosInfoRec **gAdiosInfoHdl=0; 
char gOilName[256];
float gApi;
 
OSErr PrintAdiosTableToFile(void)
{
	// use this for outputting the budget table every 24 hrs
	char buffer[512],evapStr[64],dispStr[64],timeStr[128];
	char path[256],hdrStr[256];
	char* suggestedFileName = "AdiosBudgetTable.dat",*q;
	DateTimeRec dateTime;
	Seconds time;
	float evap,disp;
	AdiosInfoRec data;
	long numOutputValues,i;
	BFPB bfpb;
	OSErr err = 0;

	// use the global stuff

	//if (!gAdiosInfoHdl)	// already checked this


	err = AskUserForSaveFilename(suggestedFileName,path,".DAT",TRUE);
	if(err) return err; // note: might be user cancel


	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }


	// Write out the times and values
	// add header line
	// put info about OilType and API
	sprintf(hdrStr,"Oil Type = %s",gOilName);
	strcpy(buffer,hdrStr);
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	sprintf(hdrStr,"API = %f",gApi);
	strcpy(buffer,hdrStr);
	strcat(buffer,NEWLINESTRING);
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	//strcpy(buffer,"Day Mo Yr Hr Min\t\t\tEvap\tDisp");
	strcpy(buffer,"      Date/Time          Evap  Disp");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	numOutputValues = _GetHandleSize((Handle)gAdiosInfoHdl)/sizeof(AdiosInfoRec);
	for(i = 0; i< numOutputValues;i++)
	{
		data = INDEXH(gAdiosInfoHdl,i);
		time = data.timeAfterSpill;
		evap = data.amountEvaporated;
		disp = data.amountDispersed;
		SecondsToDate(model->GetStartTime()+time,&dateTime); // convert to 2 digit year?
		//if(dateTime.year>=2000) dateTime.year-=2000;
		//if(dateTime.year>=1900) dateTime.year-=1900;
		//sprintf(timeStr, "%02hd,%02hd,%02hd,%02hd,%02hd",
			  // dateTime.day, dateTime.month, dateTime.year,
			  // dateTime.hour, dateTime.minute);

		Date2String(&dateTime, timeStr);
		if (q = strrchr(timeStr, ':')) q[0] = 0; // remove seconds
		sprintf(evapStr,"%3.2f",evap);
		//StringWithoutTrailingZeros(evapStr,evap,3);
		sprintf(dispStr,"%3.2f",disp);
		//StringWithoutTrailingZeros(dispStr,disp,3);
		/////
		strcpy(buffer,timeStr);
		strcat(buffer,"		");
		//strcat(buffer,"\t");
		strcat(buffer,evapStr);
		strcat(buffer,"		");
		//strcat(buffer,"\t");
		strcat(buffer,dispStr);
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
	if (err) printError("Error saving table data to a file");
	return err;

}

static void AdiosTableDraw(DialogPtr d, Rect *rectPtr,long i)
{
#pragma unused (rectPtr)
	short		leftOffset = 5, botOffset = 2;
	Point		p;
	short		v;
	Rect		rgbrect;
	double	evap,disp,removed;	
	AdiosInfoRec data;
	char *q, timeS[128], numStr[128];
	DateTimeRec time;

	GetPen(&p);

	//StringWithoutTrailingZeros(s,dvals.depth,1);
	//MyMoveTo(DEPTH_COL-stringwidth(s)/2,r->bottom);
	//drawstring(s);


	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
	rgbrect=GetDialogItemBox(d,ADIOS_USERITEM);
	v = p.v;
	data = INDEXH(gAdiosInfoHdl,i);
	SecondsToDate (model->GetStartTime() + data.timeAfterSpill, &time);
	Date2String(&time, timeS);
	if (q = strrchr(timeS, ':')) q[0] = 0; // remove seconds
	//MyMoveTo(timeH,v);
	MyMoveTo(timeH-stringwidth(timeS)/2,v);
	drawstring(timeS);
	//DrawLongAt(i+1,timeH,v);
	{
	evap = (*gAdiosInfoHdl)[i].amountEvaporated;
	disp = (*gAdiosInfoHdl)[i].amountDispersed;
	removed = (*gAdiosInfoHdl)[i].amountRemoved;
	sprintf(numStr,"%5.2f",evap);
	//MyMoveTo(evapH,v); drawstring(numStr);
	MyMoveTo(evapH-stringwidth(numStr)/2,v); drawstring(numStr);
	sprintf(numStr,"%5.2f",disp);
	//MyMoveTo(dispH,v); drawstring(numStr);
	MyMoveTo(dispH-stringwidth(numStr)/2,v); drawstring(numStr);
	sprintf(numStr,"%5.2f",removed);
	//MyMoveTo(dispH,v); drawstring(numStr);
	MyMoveTo(removedH-stringwidth(numStr)/2,v); drawstring(numStr);
	}
	MyMoveTo(rgbrect.left,v+botOffset); MyLineTo(rgbrect.right,v+botOffset);
	
	return;
}

void AdiosDraw(DialogPtr d)
{
//#pragma unused (rectPtr)
	short		leftOffset = 5, botOffset = 2;
	Point		p;
	short		v;
	Rect		rgbrect;
	float	evap,disp,removed;	
	AdiosInfoRec data;
	char *q, timeS[128], numStr[128];
	DateTimeRec time;
	long i;
	
	GetPen(&p);

	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
	rgbrect=GetDialogItemBox(d,ADIOS_USERITEM);
	//v = p.v;
	v = rgbrect.top + 10;
	for (i=0;i<	_GetHandleSize((Handle)gAdiosInfoHdl)/sizeof(AdiosInfoRec);i++)
	{
		data = INDEXH(gAdiosInfoHdl,i);
		SecondsToDate (model->GetStartTime()+data.timeAfterSpill, &time);
		Date2String(&time, timeS);
		if (q = strrchr(timeS, ':')) q[0] = 0; // remove seconds
		MyMoveTo(timeH,v);
		drawstring(timeS);
		//DrawLongAt(i+1,timeH,v);
		evap = (*gAdiosInfoHdl)[i].amountEvaporated;
		disp = (*gAdiosInfoHdl)[i].amountDispersed;
		removed = (*gAdiosInfoHdl)[i].amountRemoved;
		sprintf(numStr,"%5.3f",evap);
		MyMoveTo(evapH,v); drawstring(numStr);
		sprintf(numStr,"%5.3f",disp);
		MyMoveTo(dispH,v); drawstring(numStr);
		sprintf(numStr,"%5.3f",removed);
		MyMoveTo(removedH,v); drawstring(numStr);
		MyMoveTo(rgbrect.left,v+botOffset); MyLineTo(rgbrect.right,v+botOffset);
		v=v+10;
	}
	
	return;
}


static void AdiosTableInit(DialogPtr d, VLISTPTR L)
//static OSErr ConcTableInit(DialogPtr d, VOIDPTR L)
{
#pragma unused (L)
	Rect r;
	short IBMoffset;

	//r = GetDialogItemBox(d, EPDEPTH_LIST_LABEL);DEPTH_COL=(r.left+r.right)/2-IBMoffset;

	r = GetDialogItemBox(d,ADIOS_USERITEM);
#ifdef IBM
	IBMoffset = r.left;
#else 
	IBMoffset = 0;
#endif
	r = GetDialogItemBox(d,ADIOS_TIMETITLE);
	//timeH = r.left;
	timeH = (r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(d,ADIOS_EVAPTITLE);
	//evapH  = r.left;
	evapH  = (r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(d,ADIOS_DISPTITLE);
	//dispH = r.left;
	dispH = (r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(d,ADIOS_REMOVEDTITLE);
	//removedH = r.left;
	removedH = (r.left+r.right)/2-IBMoffset;
	mysetitext(d, ADIOS_OILNAME, gOilName);
	//Float2EditText(d, ADIOS_API, gApi, 4);
	OriginalFloat2EditText(d, ADIOS_API, gApi, 4);
	ShowHideDialogItem(d, ADIOS_CANCEL, false); 	// for now no printing to file

	return;
	//ConcDraw(d);
	//return 0;
}

 
	
static Boolean AdiosTableClick(DialogPtr d,VLISTPTR L,short itemHit,long *item,Boolean doubleClick)
//static short ConcTableClick(DialogPtr d,short itemHit,long lParam, VOIDPTR data)
{
	if(doubleClick)
	{
		*item = ADIOS_OK;
		return true;
	}
	//ConcDraw(d);
	switch(itemHit)
	{
		case ADIOS_OK:
			return true;
			//return itemHit;
		case ADIOS_CANCEL:
			PrintAdiosTableToFile();
			return false;
			break;
			//return true;
		default:
			return false;
			break;
	}
	return 0;
}


short AdiosTable(AdiosInfoRec **adiosInfoHdl,char *oilTypeName,float api)		
//long VerdatDialog()
{
	short ditem;
	long selitem;
	
	
	if(!adiosInfoHdl)
	{
		printError("No adios data.");
		return 0;
	}
	gAdiosInfoHdl = adiosInfoHdl;
	strcpy(gOilName,oilTypeName);
	gApi = api;
	selitem=SelectFromVListDialog(
				1375,
				ADIOS_USERITEM,
				_GetHandleSize((Handle)adiosInfoHdl)/sizeof(AdiosInfoRec),
				AdiosTableInit,
				nil,
				nil,
				AdiosTableDraw,
				AdiosTableClick,
				true,
				&ditem);
	return selitem;
	//ditem = MyModalDialog(5175, mapWindow, 0, ConcTableInit, ConcTableClick);
	//return 0;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
// Budget table code
/////////////////////////////////////////////////

static short timeH2, evapH2, dispH2, floatH2, offmapH2, relH2, beachH2, removedH2;
BudgetTableDataH gBudgetTableHdl=0; 
short gTotalMassUnits;
double gTotalMass;
 
OSErr PrintBudgetTableToFile(void)
{
	char path[256];
	OSErr err = 0;
	char* suggestedFileName = "BudgetTable.dat";
	DateTimeRec dateTime;
	Seconds time;
	double amttotal,amtevap,amtbeached,amtoffmap,amtfloating,amtreleased,amtdispersed,amtremoved=0;
	BudgetTableData budgetTable; 
	long numOutputValues,i;
	float timeInHours;
	char buffer[512],unitsStr[64],timeStr[64],massStr[64];
	char amtEvapStr[64],amtDispStr[64],amtBeachedStr[64],amtRelStr[64],amtFloatStr[64],amtOffStr[64],amtRemStr[64];
	BFPB bfpb;

	err = AskUserForSaveFilename(suggestedFileName,path,".DAT",TRUE);
	if(err) return err; // note: might be user cancel


	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }

	GetLeUnitsStr(unitsStr,gTotalMassUnits);

	// Write out the times and values
	// add header line
	StringWithoutTrailingZeros(massStr,gTotalMass,3);
	strcpy(buffer,"Amount Spilled - ");
	strcat(buffer,massStr);
	strcat(buffer," ");
	strcat(buffer,unitsStr);
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	//strcpy(buffer,"Hr\t\tRel\t   Float\t   Evap\t   Disp\t   Beach\t   OffMap");
	//strcpy(buffer,"Hr\tRel\tFloat\tEvap\tDisp\tBeach\tOffMap");
	strcpy(buffer,"Hr\tRel\tFloat\tEvap\tDisp\tBeach\tOffMap\tRemoved");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	numOutputValues = _GetHandleSize((Handle)gBudgetTableHdl)/sizeof(BudgetTableData);
	for(i = 0; i< numOutputValues;i++)
	{
		budgetTable = INDEXH(gBudgetTableHdl,i);
		time = budgetTable.timeAfterSpill;
		amtreleased = budgetTable.amountReleased;
		amtfloating = budgetTable.amountFloating;
		amtdispersed = budgetTable.amountDispersed;
		amtevap = budgetTable.amountEvaporated;
		amtbeached = budgetTable.amountBeached;
		amtoffmap = budgetTable.amountOffMap;
		amtremoved = budgetTable.amountRemoved;

		timeInHours = (float)time/(float)model->LEDumpInterval;
		//StringWithoutTrailingZeros(timeStr,time/model->LEDumpInterval,3);
		StringWithoutTrailingZeros(timeStr,timeInHours,3);
		StringWithoutTrailingZeros(amtEvapStr,amtevap,3);
		StringWithoutTrailingZeros(amtDispStr,amtdispersed,3);
		StringWithoutTrailingZeros(amtFloatStr,amtfloating,3);
		StringWithoutTrailingZeros(amtRelStr,amtreleased,3);
		StringWithoutTrailingZeros(amtBeachedStr,amtbeached,3);
		StringWithoutTrailingZeros(amtOffStr,amtoffmap,3);
		StringWithoutTrailingZeros(amtRemStr,amtremoved,3);
		/////
		strcpy(buffer,timeStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtRelStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtFloatStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtEvapStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtDispStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtBeachedStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtOffStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtRemStr);
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}

done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		printError("Error saving table data to a file");
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}

static void BudgetTableDraw(DialogPtr d, Rect *rectPtr,long i)
{
#pragma unused (rectPtr)
	short leftOffset = 5, botOffset = 2, v;
	Point p;
	Rect rgbrect;
	BudgetTableData data;
	char *q, timeS[128], numStr[128];
	float timeVal;
	Seconds timeAfterSpill;
	float amountReleased, amountFloating, amountDispersed, amountEvaporated, amountBeached, amountOffMap, amountRemoved;

	GetPen(&p);

	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
	rgbrect=GetDialogItemBox(d,BUDGETTABLE_USERITEM);
	v = p.v;
	//if (gTableType==1)
	{
		data = INDEXH(gBudgetTableHdl,i);
		timeAfterSpill = data.timeAfterSpill;
		timeVal = ((float)timeAfterSpill)/3600.;
		sprintf(timeS,"%.1f",timeVal);
	}
	//else
	//{
		//sprintf(timeS,"%ld",i+1);
	//}
	MyMoveTo(timeH2,v);
	//MyMoveTo(timeH2-stringwidth(timeS)/2,v);
	drawstring(timeS);

	//if (gTableType==1)
	//{
		amountReleased = (*gBudgetTableHdl)[i].amountReleased;
		amountFloating = (*gBudgetTableHdl)[i].amountFloating;
		amountDispersed = (*gBudgetTableHdl)[i].amountDispersed;
		amountOffMap = (*gBudgetTableHdl)[i].amountOffMap;
		amountEvaporated = (*gBudgetTableHdl)[i].amountEvaporated;
		amountBeached = (*gBudgetTableHdl)[i].amountBeached;
		amountRemoved = (*gBudgetTableHdl)[i].amountRemoved;
		sprintf(numStr,"%.1f",amountReleased);
		MyMoveTo(relH2,v); drawstring(numStr);
		sprintf(numStr,"%.1f",amountFloating);
		MyMoveTo(floatH2,v); drawstring(numStr);
		sprintf(numStr,"%.1f",amountDispersed);
		MyMoveTo(dispH2,v); drawstring(numStr);
		sprintf(numStr,"%.1f",amountEvaporated);
		MyMoveTo(evapH2,v); drawstring(numStr);
		sprintf(numStr,"%.1f",amountBeached);
		MyMoveTo(beachH2,v); drawstring(numStr);
		sprintf(numStr,"%.1f",amountOffMap);
		MyMoveTo(offmapH2,v); drawstring(numStr);
		sprintf(numStr,"%.1f",amountRemoved);
		MyMoveTo(removedH2,v); drawstring(numStr);
	//}
	//else
	//{
		//av = gDepthSlice[i+1];
		//sprintf(numStr,"%5.3f",av);
		//MyMoveTo(avH,v); drawstring(numStr);
	//}
		
	MyMoveTo(rgbrect.left,v+botOffset); MyLineTo(rgbrect.right,v+botOffset);
	
	return;
}

static void BudgetTableInit(DialogPtr d, VLISTPTR L)
{
#pragma unused (L)
	Rect r;
	//if (gTableType==1)
	{
		mysetitext(d,BUDGETTABLE_TIMETITLE,"Hour");
		mysetitext(d,BUDGETTABLE_RELTITLE,"Released");
		mysetitext(d,BUDGETTABLE_FLOATTITLE,"Floating");
		mysetitext(d,BUDGETTABLE_EVAPTITLE,"Evaporated");
		mysetitext(d,BUDGETTABLE_DISPTITLE,"Dispersed");
		mysetitext(d,BUDGETTABLE_BEACHTITLE,"Beached");
		mysetitext(d,BUDGETTABLE_OFFMAPTITLE,"Off Map");
		mysetitext(d,BUDGETTABLE_REMOVEDTITLE,"Removed");
	}
	//else
	//{
		//mysetitext(d,CONCTABLE_TIMETITLE,"Depth");
		//mysetitext(d,CONCTABLE_AVTITLE,"Conc (ppm)");
		//ShowHideDialogItem(d, CONCTABLE_MAXTITLE, false); 
		//ShowHideDialogItem(d, CONCTABLE_SAVETOFILE, false); // don't save depth data
	//}
	r = GetDialogItemBox(d,BUDGETTABLE_TIMETITLE);
	timeH2 = r.left;
	r = GetDialogItemBox(d,BUDGETTABLE_RELTITLE);
	relH2  = r.left;
	r = GetDialogItemBox(d,BUDGETTABLE_FLOATTITLE);
	floatH2 = r.left;
	r = GetDialogItemBox(d,BUDGETTABLE_EVAPTITLE);
	evapH2 = r.left;
	r = GetDialogItemBox(d,BUDGETTABLE_DISPTITLE);
	dispH2 = r.left;
	r = GetDialogItemBox(d,BUDGETTABLE_BEACHTITLE);
	beachH2 = r.left;
	r = GetDialogItemBox(d,BUDGETTABLE_OFFMAPTITLE);
	offmapH2 = r.left;
	r = GetDialogItemBox(d,BUDGETTABLE_REMOVEDTITLE);
	removedH2 = r.left;
	//ShowHideDialogItem(d, BUDGETTABLE_SAVETOFILE, false); 	// for now no printing to file
	return;
}

 
	
static Boolean BudgetTableClick(DialogPtr d,VLISTPTR L,short itemHit,long *item,Boolean doubleClick)
{
	if(doubleClick)
	{
		*item = BUDGETTABLE_OK;
		return true;
	}

	switch(itemHit)
	{
		case BUDGETTABLE_OK:
			return true;
		case BUDGETTABLE_SAVETOFILE:
			PrintBudgetTableToFile();
			return true;
			break;
		default:
			return false;
			break;
	}
	return 0;
}


short BudgetTable(short massVolUnits, double totalMass, BudgetTableData **budgetTableHdl/*, short tableType*/)		
{
	short ditem;
	long selitem;
	
	if(!budgetTableHdl /*&& tableType==1*/)
	{
		printError("No budget data.");
		return 0;
	}
	gBudgetTableHdl = budgetTableHdl;
	gTotalMass = totalMass;
	gTotalMassUnits = massVolUnits;
	selitem = SelectFromVListDialog(
				1380,
				BUDGETTABLE_USERITEM,
				//tableType==1 ? _GetHandleSize((Handle)budgetTableHdl)/sizeof(BudgetTableData) : 1,
				_GetHandleSize((Handle)budgetTableHdl)/sizeof(BudgetTableData),
				BudgetTableInit,
				nil,
				nil,
				BudgetTableDraw,
				BudgetTableClick,
				true,
				&ditem);
	return selitem;
}

Boolean IsCJFile(char *path)
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
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 256);
		if (strstr(strLine,"CJ"))
			bIsValid = true;
	}
	
	return bIsValid;
}

/////////////////////////////////////////////////
Boolean IsOSSMFile(char *path) // how to recognize ?
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
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 256);
		if (strstr(strLine,"")) //code goes here, what is on first line?
			bIsValid = true;
	}
	
	return bIsValid;
}
/////////////////////////////////////////////////
//Oil Name = ABOOZAR
//API = 26.9                          	Pour Point = -29 deg F              	
//Wind Speed = constant at 5 mph      	Wave Height = computed from winds   	
//Water Temperature = 55 deg F
//Time of Initial Release = April 29, 1500 hours
//Total Amount of Oil Released = 5,000 bbl
//______________________________________________________________________

//Hours Into	 Released 	 	Evaporated	 	Dispersed 	 	Remaining 	
//     Spill	   bbl    	 	 percent  	 	 percent  	 	 percent  	

// or may not have the dispersed information
//Hours Into	 Released 	 	Evaporated	 	Remaining 	
//     Spill	   bbl    	 	 percent  	 	 percent  	
//         1	  5,000   	-	    1     	-	    99    	
Boolean IsAdiosBudgetTable(char* path)
{
	OSErr	err = noErr;
	long	line;
	char	strLine [512];	//maybe not enough data
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	//short selectedUnits = kUndefined;
	
	// decide what to do about various format options
	// for now just assume default format
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	// code goes here, if lines are long may run out of space in array
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		//long numScanned;
		// we check 
		// that the 1st line starts 'Oil Name =' 
		// that the 2nd line starts 'API = ' 
		// that the 9th line 'Hours Into	 Released 	 	Evaporated	 	Dispersed 	 	Remaining'
		// maybe only care about first line or 2
		// later check that there is Dispersed Oil, maybe # items in first line of data
		line = 0;
		/////////////////////////////////////////////////
		
		// first line, oil name
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		if(strncmpnocase(strLine,"Oil Name =",strlen("Oil Name =")))
			return false;
		/////////////////////////////////////////////////
		
		// second line, api - will want this eventually
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		if(strncmpnocase(strLine,"API = ",strlen("API = ")))
			return false;
		/////////////////////////////////////////////////
		// may want to scan further to see what format was used		
	}
	
	return true;
}

OSErr ReadAdiosBudgetTable(char *path,AdiosInfoRecH *adiosData,char *oilType,float *api)
{	
	char s[512],s2[512],apiStr[3];
	char eqStr[3];
	long i,numValues,numLines,numScanned;
	CHARH f;
	AdiosInfoRecH adiosDataH = 0;
	AdiosInfoRec adiosDataLine = {0,0.,0.}, previousDataLine = {0,0.,0.};
	long hrs;
	float amt_rel,amt_evap,amt_disp,amt_sur,amt_skim,amt_burn,apiNum;
	OSErr scanErr;
	OSErr err = noErr;
	long numDataLines;
	long numHeaderLines = 10;	// should always have 10 header lines, but may want to scan them
	long lastHeaderLineBeforeBlank = 8;
	Boolean bRemovalOptionSelected = false, bAllRemovalOptionsSelected = false;
	
	// code goes here, store some of the header info - oil type, api
	if (!path) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f))
		{ TechError("ReadAdiosBudgetTable()", "ReadFileContents()", 0); goto done; }
	
	numLines = NumLinesInText(*f);
	
	numDataLines = numLines - numHeaderLines;
			
	*adiosData = 0;
	//adiosDataH = (AdiosInfoRecH)_NewHandle(sizeof(AdiosInfoRec)*numDataLines);
	adiosDataH = (AdiosInfoRecH)_NewHandle(sizeof(AdiosInfoRec)*(numDataLines+1));
	if(!adiosDataH)
		{ err = -1; TechError("ReadAdiosBudgetTable()", "_NewHandle()", 0); goto done; }
	numValues = 0;
	INDEXH(adiosDataH, numValues++) = adiosDataLine;	// should start with zero evap/disp at time zero
	for (i = 0 ; i < numLines ; i++) {
		NthLineInTextOptimized(*f, i, s, 512); // time, released, evaporated, dispersed, surface
		//may want to scan header lines
		if(i < numHeaderLines)
		{
			if (i==0) strcpy(oilType,&s[10]); //Oil Name = 
			//if (i==1) {sscanf(s, "%s %s %f",apiStr,eqStr,&apiNum); //API = 42.5
			//code goes here, Pour Point = 50 deg F (or unknown)
			// line i==3 or i==4 Water Temperature = 76 deg F
			if (i==1) {sscanf(&s[6], "%f",&apiNum);} //API = 42.5
			if (i==2)
			{
				if (strstr(s,"Mousse")) 
				{	// if use Emulsification option in ADIOS get an extra header line
					numHeaderLines++;
					lastHeaderLineBeforeBlank++;
				}
			}
			if (i==lastHeaderLineBeforeBlank) {
				if (!strstr(s,"Dispersed")) 
				{	// may want to allow adios evaporation only, or burning/skimming
					printError("The selected oil does not disperse naturally"); 
					err=-1; goto done;
				}
				if (strstr(s,"Burned") || strstr(s,"Skimmed")) 
				{	// may want to allow adios evaporation only
					bRemovalOptionSelected = true;
					if (strstr(s,"Burned") && strstr(s,"Skimmed")) 
						bAllRemovalOptionsSelected = true;
				}
			}
			continue; // skip any header lines
		}
		if(i%200 == 0) MySpinCursor(); 
		RemoveLeadingAndTrailingWhiteSpace(s);
		if(s[0] == 0) continue; // it's a blank line, allow this and skip the line
		RemoveSetFromString(s, "-,", s2);
	
		if (bRemovalOptionSelected)
		{	// issue here, may not have dispersion in this case
			if (bAllRemovalOptionsSelected)
			{	
				numScanned=sscanf(s2, "%ld %f %f %f %f %f %f",
							  &hrs, &amt_rel, &amt_evap, &amt_disp, &amt_skim, &amt_burn, &amt_sur);
				if (numScanned==6)	//this probably means oil is not dispersed, but should check header line - I don't think this is an error in this case
					{ amt_sur = amt_burn; amt_burn = amt_skim; amt_skim = amt_disp; amt_disp = 0;/* err = -1; TechError("ReadAdiosBudgetTable()", "sscanf() == 4", 0); goto done;*/ }
				if (numScanned!=7)	//this might mean units are different, should check options
					{ err = -1; TechError("ReadAdiosBudgetTable()", "sscanf() == 7", 0); goto done; }
			}
			else
			{
				numScanned=sscanf(s2, "%ld %f %f %f %f %f",
							  &hrs, &amt_rel, &amt_evap, &amt_disp, &amt_skim, &amt_sur);// use skim for either, since combining anyway
				if (numScanned==5)	//this probably means oil is not dispersed, but should check header line
					{ amt_sur = amt_skim; amt_skim = amt_disp; amt_disp = 0;/*err = -1; TechError("ReadAdiosBudgetTable()", "sscanf() == 4", 0); goto done;*/ }
				if (numScanned!=6)	//this might mean units are different, should check options
					{ err = -1; TechError("ReadAdiosBudgetTable()", "sscanf() == 6", 0); goto done; }
			}
		}
		else
		{
			numScanned=sscanf(s2, "%ld %f %f %f %f",
						  &hrs, &amt_rel, &amt_evap, &amt_disp, &amt_sur);	
			if (numScanned==4)	//this probably means oil is not dispersed, but should check header line
				{ err = -1; TechError("ReadAdiosBudgetTable()", "sscanf() == 4", 0); goto done; }
			if (numScanned!=5)	//this might mean units are different, should check options
				{ err = -1; TechError("ReadAdiosBudgetTable()", "sscanf() == 5", 0); goto done; }
		}	
		// code goes here, 
		adiosDataLine.timeAfterSpill = hrs*3600;
		if (amt_disp > 100 || amt_evap > 100 || amt_sur > 100)	// percent rather than fraction, could be zero? check this
		{
			printError("Adios Budget table must be in fraction or percent.");
			err = true;
			goto done;
		}
		if (amt_disp >= 1 || amt_evap >= 1)	// percent rather than fraction, could be zero? check this
		{
			amt_disp /= 100;
			amt_evap /= 100;
		}
		// compare to previous time, if amount is the same don't include the line
		adiosDataLine.amountDispersed = amt_disp;
		adiosDataLine.amountEvaporated = amt_evap;
		//if (adiosDataLine.amountDispersed > previousDataLine.amountDispersed || adiosDataLine.amountEvaporated > previousDataLine.amountEvaporated)
			INDEXH(adiosDataH, numValues++) = adiosDataLine;
		//previousDataLine = adiosDataLine;
	}
	if(numValues > 0)
	{
		long actualSize = numValues*sizeof(AdiosInfoRec); 
		_SetHandleSize((Handle)adiosDataH,actualSize);
		err = _MemError();
	}
	else {
		printError("No lines were found");
		err = true;
		goto done;
	}

	*adiosData = adiosDataH;
	*api = apiNum;

done:
	if(f) {DisposeHandle((Handle)f); f = 0;}
	if (err && adiosDataH) {DisposeHandle((Handle)adiosDataH); adiosDataH = 0;}
	
	return err;
	
}
/////////////////////////////////////////////////
OSErr ReadAdiosBudgetTableWithRemovalOptions(char *path,AdiosInfoRecH *adiosData,char *oilType,float *api)
{	
	char s[512],s2[512],apiStr[3];
	char eqStr[3];
	long i,numValues,numLines,numScanned;
	CHARH f;
	AdiosInfoRecH adiosDataH = 0;
	AdiosInfoRec adiosDataLine = {0,0.,0.}, previousDataLine = {0,0.,0.};
	long hrs;
	float amt_rel,amt_evap,amt_disp,amt_sur,amt_skim,amt_burn=0,apiNum;
	OSErr scanErr;
	OSErr err = noErr;
	long numDataLines;
	long numHeaderLines = 10;	// should always have 10 header lines, but may want to scan them
	long numItemsInLine = 4;		// at least time, rel, evap, surf - others optional (disp, burn, skim, beach)
	long lastHeaderLineBeforeBlank = 8;
	Boolean bRemovalOptionSelected = false, bAllRemovalOptionsSelected = false;
	
	// code goes here, store some of the header info - oil type, api
	if (!path) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f))
		{ TechError("ReadAdiosBudgetTableWithRemovalOptions()", "ReadFileContents()", 0); goto done; }
	
	numLines = NumLinesInText(*f);
	
	numDataLines = numLines - numHeaderLines;
			
	*adiosData = 0;
	//adiosDataH = (AdiosInfoRecH)_NewHandle(sizeof(AdiosInfoRec)*numDataLines);
	adiosDataH = (AdiosInfoRecH)_NewHandle(sizeof(AdiosInfoRec)*(numDataLines+1));
	if(!adiosDataH)
		{ err = -1; TechError("ReadAdiosBudgetTableWithRemovalOptions()", "_NewHandle()", 0); goto done; }
	numValues = 0;
	INDEXH(adiosDataH, numValues++) = adiosDataLine;	// should start with zero evap/disp at time zero
	for (i = 0 ; i < numLines ; i++) {
		NthLineInTextOptimized(*f, i, s, 512); // time, released, evaporated, dispersed, surface
		//may want to scan header lines
		if(i < numHeaderLines)
		{
			if (i==0) strcpy(oilType,&s[10]); //Oil Name = 
			//if (i==1) {sscanf(s, "%s %s %f",apiStr,eqStr,&apiNum); //API = 42.5
			if (i==1) {sscanf(&s[6], "%f",&apiNum);} //API = 42.5
			if (i==2)
			{
				if (strstr(s,"Mousse")) 
				{	// if use Emulsification option in ADIOS get an extra header line
					numHeaderLines++;
					lastHeaderLineBeforeBlank++;
				}
			}
			if (i==lastHeaderLineBeforeBlank) {
				if (!strstr(s,"Dispersed")) 
				{	// may want to allow adios evaporation only, or burning/skimming
					printError("The selected oil does not disperse naturally"); 
					err=-1; goto done;
				}
				else numItemsInLine++;
				
				if (strstr(s,"Beached")) // an error or just ignore the column?
				{	// may want to allow adios evaporation only, or burning/skimming
					printError("GNOME does not accept beaching from ADIOS"); 
					//numItemsInLine++;
					err=-1; goto done;
				}
				
				if (strstr(s,"Burned") || strstr(s,"Skimmed")) 
				{	// may want to allow adios evaporation only
					bRemovalOptionSelected = true;
					numItemsInLine++;
					if (strstr(s,"Burned") && strstr(s,"Skimmed")) 
						{bAllRemovalOptionsSelected = true; numItemsInLine++;}
				}
			}
			continue; // skip any header lines
		}
		if(i%200 == 0) MySpinCursor(); 
		RemoveLeadingAndTrailingWhiteSpace(s);
		if(s[0] == 0) continue; // it's a blank line, allow this and skip the line
		RemoveSetFromString(s, "-,", s2);
	
		if (bRemovalOptionSelected)
		{	// issue here, may not have dispersion in this case
			if (bAllRemovalOptionsSelected)
			{	
				numScanned=sscanf(s2, "%ld %f %f %f %f %f %f",
							  &hrs, &amt_rel, &amt_evap, &amt_disp, &amt_skim, &amt_burn, &amt_sur);
				if (numScanned==6)	//this probably means oil is not dispersed, but should check header line - I don't think this is an error in this case
					{ amt_sur = amt_burn; amt_burn = amt_skim; amt_skim = amt_disp; amt_disp = 0;/* err = -1; TechError("ReadAdiosBudgetTableWithRemovalOptions()", "sscanf() == 4", 0); goto done;*/ }
				if (numScanned!=7)	//this might mean units are different, should check options
					{ err = -1; TechError("ReadAdiosBudgetTableWithRemovalOptions()", "sscanf() == 5", 0); goto done; }
			}
			else
			{
				numScanned=sscanf(s2, "%ld %f %f %f %f %f",
							  &hrs, &amt_rel, &amt_evap, &amt_disp, &amt_skim, &amt_sur);// use skim for either, since combining anyway
				if (numScanned==5)	//this probably means oil is not dispersed, but should check header line
					{ amt_sur = amt_skim; amt_skim = amt_disp; amt_disp = 0;/*err = -1; TechError("ReadAdiosBudgetTableWithRemovalOptions()", "sscanf() == 4", 0); goto done;*/ }
				if (numScanned!=6)	//this might mean units are different, should check options
					{ err = -1; TechError("ReadAdiosBudgetTableWithRemovalOptions()", "sscanf() == 5", 0); goto done; }
			}
		}
		else
		{
			printError("Either skimming or burning must be selected in ADIOS. Please check your file"); 
			/*numScanned=sscanf(s2, "%ld %f %f %f %f",
						  &hrs, &amt_rel, &amt_evap, &amt_disp, &amt_sur);	
			if (numScanned==4)	//this probably means oil is not dispersed, but should check header line
				{ err = -1; TechError("ReadAdiosBudgetTableWithRemovalOptions()", "sscanf() == 4", 0); goto done; }
			if (numScanned!=5)	//this might mean units are different, should check options
				{ err = -1; TechError("ReadAdiosBudgetTableWithRemovalOptions()", "sscanf() == 5", 0); goto done; }*/
		}	
		// code goes here, 
		adiosDataLine.timeAfterSpill = hrs*3600;
		if (amt_disp > 100 || amt_evap > 100 || amt_sur > 100)	// percent rather than fraction, could be zero? check this
		{
			printError("Adios Budget table must be in fraction or percent.");
			err = true;
			goto done;
		}
		if (amt_disp >= 1 || amt_evap >= 1 || amt_burn >= 1 || amt_skim >= 1)	// percent rather than fraction, could be zero? check this
		{
			amt_disp /= 100;
			amt_evap /= 100;
			amt_burn /= 100;
			amt_skim /= 100;
		}
		// compare to previous time, if amount is the same don't include the line
		adiosDataLine.amountDispersed = amt_disp;
		adiosDataLine.amountEvaporated = amt_evap;
		adiosDataLine.amountRemoved = amt_burn + amt_skim;
		//if (adiosDataLine.amountDispersed > previousDataLine.amountDispersed || adiosDataLine.amountEvaporated > previousDataLine.amountEvaporated)
			INDEXH(adiosDataH, numValues++) = adiosDataLine;
		//previousDataLine = adiosDataLine;
	}
	if(numValues > 0)
	{
		long actualSize = numValues*sizeof(AdiosInfoRec); 
		_SetHandleSize((Handle)adiosDataH,actualSize);
		err = _MemError();
	}
	else {
		printError("No lines were found");
		err = true;
		goto done;
	}

	*adiosData = adiosDataH;
	*api = apiNum;

done:
	if(f) {DisposeHandle((Handle)f); f = 0;}
	if (err && adiosDataH) {DisposeHandle((Handle)adiosDataH); adiosDataH = 0;}
	
	return err;
	
}
/////////////////////////////////////////////////
OSErr GetAdiosFileName(char *path)
{
	char adiosPath[256],shortFileName[64];
	char tempStr[256];
	Point where = CenteredDialogUpLeft(M38h2);
	WorldPoint p;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	OSErr err = 0;
	
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, M38h2, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(adiosPath, reply.fullPath);

		// not sure why this is here...
		strcpy(tempStr, adiosPath);
		SplitPathFile(tempStr, shortFileName);
#else
	sfpgetfile(&where, "",
				(FileFilterUPP)0,
				-1, typeList,
				(DlgHookUPP)0,
				&reply, M38h2,
				(ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));

	if (!reply.good) return 2; // user canceled
	
	my_p2cstr(reply.fName);
#ifdef MAC
	GetFullPath(reply.vRefNum, 0, (char *)reply.fName, adiosPath);
	strcpy(shortFileName,(char*) reply.fName);
#else
	strcpy(adiosPath, reply.fName);
	strcpy(tempStr,adiosPath);
	SplitPathFile(tempStr,shortFileName);
#endif
#endif	
	strcpy(path,adiosPath);
	return noErr;
}

long TOLEList::GetNumAdiosBudgetTableItems(void)
{
	long numInHdl = 0;
	if (fAdiosDataH) numInHdl = _GetHandleSize((Handle)fAdiosDataH)/sizeof(**fAdiosDataH);
	
	return numInHdl;
}
/////////////////////////////////////////////////
double GetLEMass(LERec theLE)
{
	long i;
	double tHours, fracLeft = 0.;
	TWeatherer	*thisWeatherer;
	OilComponent	component;
	CMyList	*weatherList = model->GetWeatherList();
			
	if (theLE.pollutantType == CHEMICAL)
	{
		weatherList->GetListItem((Ptr)&thisWeatherer, 0);	// assume there is only one
		((TOSSMWeatherer*)thisWeatherer)->componentsList -> GetListItem ((Ptr) &component, theLE.pollutantType - 1);
		tHours = (double) ( model -> GetModelTime () - theLE.releaseTime)/ 3600.0  + theLE.ageInHrsWhenReleased;
		// if LE has not been released yet return 0?
		//if (theLE.releaseTime > model->GetModelTime()) return theLE.mass;
		if (theLE.releaseTime > model->GetModelTime()) return 0;
		
		for(i = 0;i<3;i++)	// at this point only using 1 half life component
		{
			if(component.percent[i] > 0.0)
			{
				fracLeft +=  (component.percent[i])*pow(0.5,tHours/(component.halfLife[i]));
			}
		}
		fracLeft = _max (0.0,fracLeft);
		fracLeft = _min (1.0,fracLeft);
		return fracLeft*theLE.mass;
	}
	else
		return theLE.mass;
}
/////////////////////////////////////////////////
void GetLeUnitsStr(char* str,short massUnits)
{
	 short itemNum = massUnits; // for now these are the same
	 str[0] = 0;

	#ifdef MAC
	switch(massUnits)
	{
		case GALLONS: strcpy (str, "gallons"); break;
		case BARRELS: strcpy (str, "barrels"); break;
		case CUBICMETERS: strcpy (str, "m**3"); break;
		///
		case METRICTONS: strcpy (str, "metric tons"); break;
		case SHORTTONS:  strcpy (str, "short tons"); break;
		case KILOGRAMS: strcpy (str, "kg"); break;
	}
	/*{
		MenuHandle	h = GetMenu(pLEMASSUNITS);
		Str255 pStr;
		GetMenuItemText(h,itemNum,pStr); 
		my_p2cstr(pStr);
		strcpy(str,(char*)pStr);
	}*/
	#else
		{ // IBM code
			getindstring(str,pLEMASSUNITS*100,itemNum);
		}
	#endif
}

/////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////


TLEList::TLEList()
{
	LEHandle = 0;
	this->numOfLEs = 0;
	//this->massUnits = 0;
	this-> fLeType = FORECAST_LE;
	memset(&fOwnersUniqueID,0,sizeof(fOwnersUniqueID));
	
	SetFileDirty(FALSE);
	
	bOpen = FALSE;
}

void TLEList::Dispose()
{
	if (LEHandle)
	{
		DisposeHandle ((Handle) LEHandle);
		LEHandle = nil;
	}
	this->numOfLEs = 0;
	return;
}


void TLEList::GetLE(long i, LERecP theLE)
{
	*theLE = INDEXH(LEHandle, i);
}

void TLEList::SetLE(long i, LERecP theLE)
{
	INDEXH(LEHandle, i) =  *theLE;
}

WorldRect TLEList::GetLEBounds()
{
	long i;
	WorldPoint p;
	WorldRect bounds = voidWorldRect;
	
	for (i = 0 ; i < this->numOfLEs ; i++) {
		p = INDEXH(LEHandle, i).p;
		AddWPointToWRect(p.pLat, p.pLong, &bounds);
	}
	
	return bounds;
}

//JLE 1/6/99


short TOLEList::GetMassUnitType()
{
	short masstype;
	switch(GetMassUnits())
	{
		case GALLONS:
		case BARRELS:
		case CUBICMETERS:
			masstype= VOLUMETYPE;
		
		case KILOGRAMS: 
		case METRICTONS:
		case SHORTTONS: 
			masstype= MASSTYPE;
	}
	return masstype;
}



// Return amount statistics in desiredMassVolUnits 
void TOLEList::GetLEAmountStatistics(short desiredMassVolUnits, double *amtTotal,double *amtReleased,double *amtEvaporated,double *amtDispersed,
																	double *amtBeached,double * amtOffmap, double *amtFloating, double *amtRemoved)
{	// this is only called in Standard mode
	long numReleased,numEvap,numBeached,numOffmap,numfloating,numDisp=0,numRemoved=0;
	double massFrac = GetTotalMass()/GetNumOfLEs();
	short massunits = this->GetMassUnits();
	double density =  GetPollutantDensity(this->GetOilType());	

	this->GetLEStatistics(&numReleased,&numEvap,&numBeached,&numOffmap,&numfloating);
	if (fDispersantData.bDisperseOil || fAdiosDataH || fSetSummary.z > 0) this->RecalculateLEStatistics(&numDisp,&numfloating,&numRemoved,&numOffmap);
	
	//*amtTotal = GetTotalMass();
	*amtTotal = VolumeMassToVolumeMass(GetTotalMass(),density,massunits,desiredMassVolUnits);
	*amtReleased = VolumeMassToVolumeMass(numReleased*massFrac,density,massunits,desiredMassVolUnits);
	*amtEvaporated = VolumeMassToVolumeMass(numEvap*massFrac,density,massunits,desiredMassVolUnits);
	*amtDispersed = VolumeMassToVolumeMass(numDisp*massFrac,density,massunits,desiredMassVolUnits);
	*amtBeached = VolumeMassToVolumeMass(numBeached*massFrac,density,massunits,desiredMassVolUnits);
	*amtOffmap = VolumeMassToVolumeMass(numOffmap*massFrac,density,massunits,desiredMassVolUnits);
	*amtFloating = VolumeMassToVolumeMass(numfloating*massFrac,density,massunits,desiredMassVolUnits);
	*amtRemoved = VolumeMassToVolumeMass(numRemoved*massFrac,density,massunits,desiredMassVolUnits);
}

  
void TLEList::GetLEStatistics(long* numReleased,long* numEvaporated,long* numBeached, long* numOffMap, long* numFloating)
{
	long i,numNotReleased = 0;
	*numEvaporated = *numBeached = *numOffMap = *numFloating = 0;
	if (LEHandle) {
		for (i = 0 ; i < this->numOfLEs ; i++) {
			switch( INDEXH(LEHandle, i).statusCode)
			{
				case OILSTAT_NOTRELEASED: numNotReleased++; break;
				case OILSTAT_OFFMAPS: (*numOffMap)++; break;
				case OILSTAT_ONLAND: (*numBeached)++; break;
				case OILSTAT_EVAPORATED: (*numEvaporated)++; break;
				case OILSTAT_INWATER:(*numFloating)++; break;
			}
		}
	}
	*numReleased = this->numOfLEs - numNotReleased;
}

void TLEList::RecalculateLEStatistics(long* numDispersed,long* numFloating,long* numRemoved,long* numOffMaps)
{
	// code goes here, might want to calculate percent dissolved here
	long i;
	if (LEHandle) {
		for (i = 0 ; i < this->numOfLEs ; i++) {	// code goes here, Alan occasionally seeing budget table floating < 0, maybe not released has z>0 or dispersed Le is getting beached?
			if( ((INDEXH(LEHandle, i).dispersionStatus == HAVE_DISPERSED || INDEXH(LEHandle, i).dispersionStatus == HAVE_DISPERSED_NAT) || INDEXH(LEHandle,i).z > 0) && !(INDEXH(LEHandle, i).statusCode==OILSTAT_EVAPORATED) && !(INDEXH(LEHandle, i).statusCode==OILSTAT_OFFMAPS) && !(INDEXH(LEHandle, i).statusCode==OILSTAT_NOTRELEASED) && !(INDEXH(LEHandle, i).statusCode==OILSTAT_ONLAND))
			//if( ((INDEXH(LEHandle, i).dispersionStatus == HAVE_DISPERSED || INDEXH(LEHandle, i).dispersionStatus == HAVE_DISPERSED_NAT) || INDEXH(LEHandle,i).z > 0) && INDEXH(LEHandle, i).statusCode==OILSTAT_INWATER )
			{
				(*numDispersed)++; 
				(*numFloating)--; 
				if ((*numFloating) < 0) 
				{
					//printNote("Num floating < 0 in RecalculateLEStatistics");
				}
			}
			if( (INDEXH(LEHandle, i).dispersionStatus == HAVE_REMOVED) && (INDEXH(LEHandle, i).statusCode==OILSTAT_OFFMAPS) )
			{
				(*numRemoved)++; 
				(*numOffMaps)--; 
				if ((*numOffMaps) < 0) 
				{
					printNote("Num off maps < 0 in RecalculateLEStatistics");
				}
			}
		}
	}
}

void TLEList::BeachLE(long i, WorldPoint beachPosition)
{
	if (GetLEStatus (i) == OILSTAT_INWATER)
	{
		INDEXH(LEHandle, i).lastWaterPt = INDEXH(LEHandle, i).p;
		SetLEPosition(i, beachPosition);
		INDEXH(LEHandle, i).beachTime = model->modelTime;
		SetLEStatus(i, OILSTAT_ONLAND);
	}
}

void TLEList::ReFloatLE(long i)
{
	SetLEPosition(i, INDEXH(LEHandle, i).lastWaterPt);
	SetLEStatus(i, OILSTAT_INWATER);
}

OilStatus TLEList::GetLEStatus(long i)
{
	return INDEXH(LEHandle, i).statusCode;
}

void TLEList::SetLEStatus(long i, OilStatus status)
{
	INDEXH(LEHandle, i).statusCode = status;
}

WorldPoint TLEList::GetLEPosition(long i)
{
	return INDEXH(LEHandle, i).p;
}

void TLEList::SetLEPosition(long i, WorldPoint p)
{
	INDEXH(LEHandle, i).p = p;
}

Seconds TLEList::GetLEReleaseTime(long i)
{
	return INDEXH(LEHandle, i).releaseTime;
}

void TLEList::ReleaseLE(long i)
{
	INDEXH(LEHandle, i).statusCode = OILSTAT_INWATER;
}

//#define  TLEList_FileVersion 1
//#define  TLEList_FileVersion 2	// 6/18/01, added windage to LERec
//#define  TLEList_FileVersion 3	// 2/25/08, added dispersion fields
#define  TLEList_FileVersion 4	// 6/5/10, added rise velocity
OSErr TLEList::Write(BFPB *bfpb)
{
	char c;
	long i, version = TLEList_FileVersion;
	ClassID id = GetClassID ();
	LERec LE;
	OSErr	err = noErr;
	
	// call base class write
	if(err = TClassID::Write(bfpb)) return err;
	
	StartReadWriteSequence("TLEList::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	//if (err = WriteMacValue(bfpb, bOpen)) return err;
	if (err = WriteMacValue(bfpb, numOfLEs)) return err;
	if (err = WriteMacValue(bfpb, fLeType)) return err;
	if (err = WriteMacValue(bfpb, fOwnersUniqueID)) return err;
	
	if(numOfLEs > 0)
	{
		for (i = 0 ; i < numOfLEs ; i++) {
			LE = INDEXH(LEHandle, i);
			if(err = WriteLE (bfpb, &LE)) return err;
		}
	}

	SetFileDirty(FALSE);

	return 0;
}

OSErr TLEList::Read(BFPB *bfpb)
{
	char 	c;
	long 	i, version;
	ClassID id;
	LERec 	LE;
	OSErr	err = noErr;
	
	// call base class read
	if(err = TClassID::Read(bfpb)) return err;
	
	Dispose ();		// dispose of LE's in list if any, STH
	
	StartReadWriteSequence("TLEList::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TLEList::Read()", "id != GetClassID", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > TLEList_FileVersion) { printSaveFileVersionError(); return -1; }
	//if (err = ReadMacValue(bfpb, &bOpen)) return err;
	if (err = ReadMacValue(bfpb, &numOfLEs)) return err;
	if (err = ReadMacValue(bfpb, &fLeType)) return err;
	if (err = ReadMacValue(bfpb, &fOwnersUniqueID)) return err;


	LEHandle = 0;
	if(numOfLEs > 0)
	{
		LEHandle = (LERecH)_NewHandle(numOfLEs * sizeof(LERec));
		if (!LEHandle)
			{TechError("TLEList::Read()", "_NewHandle()", 0); return -1; }
		
		for (i = 0 ; i < numOfLEs ; i++) {
			err = ReadLE (bfpb, &LE, version);
			if (err)
				{TechError("TLEList::Read()", "ReadLE()", err); return -1; }
			else
				INDEXH(LEHandle, i) = LE;
		}
	}

	SetFileDirty (FALSE);

	return err;
}

OSErr TLEList::WriteLE (BFPB *bfpb, LERec *theLE)
{
	OSErr	err = noErr;

	if (err = WriteMacValue(bfpb, theLE -> leKey)) return err;
	if (err = WriteMacValue(bfpb, theLE -> leCustomData)) return err;
	if (err = WriteMacValue(bfpb, theLE -> p.pLong)) return err;
	if (err = WriteMacValue(bfpb, theLE -> p.pLat)) return err;
	if (err = WriteMacValue(bfpb, theLE -> z)) return err;
	if (err = WriteMacValue(bfpb, theLE -> releaseTime)) return err;
	if (err = WriteMacValue(bfpb, theLE -> ageInHrsWhenReleased)) return err;
	if (err = WriteMacValue(bfpb, theLE -> clockRef)) return err;
	if (err = WriteMacValue(bfpb, theLE -> pollutantType)) return err;
	if (err = WriteMacValue(bfpb, theLE -> mass)) return err;
	if (err = WriteMacValue(bfpb, theLE -> density)) return err;
	if (err = WriteMacValue(bfpb, theLE -> windage)) return err;
	//if (gMearnsVersion)
	{
		if (err = WriteMacValue(bfpb, theLE -> dropletSize)) return err;
		if (err = WriteMacValue(bfpb, theLE -> dispersionStatus)) return err;
	}
	if (err = WriteMacValue(bfpb, theLE -> riseVelocity)) return err;
	if (err = WriteMacValue(bfpb, theLE -> statusCode)) return err;
	if (err = WriteMacValue(bfpb, theLE -> lastWaterPt.pLong)) return err;
	if (err = WriteMacValue(bfpb, theLE -> lastWaterPt.pLat)) return err;
	if (err = WriteMacValue(bfpb, theLE -> beachTime)) return err;

	return err;
}

OSErr TLEList::ReadLE (BFPB *bfpb, LERec *theLE, long version)
{
	OSErr	err = noErr;
	double windage;
	
	if (err = ReadMacValue(bfpb, &theLE -> leKey)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> leCustomData)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> p.pLong)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> p.pLat)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> z)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> releaseTime)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> ageInHrsWhenReleased)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> clockRef)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> pollutantType)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> mass)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> density)) return err;
	if (version>1)	
		if (err = ReadMacValue(bfpb, &theLE -> windage)) return err;
	if (gMearnsVersion || version > 2)
	{
	if (err = ReadMacValue(bfpb, &theLE -> dropletSize)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> dispersionStatus)) return err;
	}
	if (version>3) 
		if (err = ReadMacValue(bfpb, &theLE -> riseVelocity)) return err;

	if (err = ReadMacValue(bfpb, &theLE -> statusCode)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> lastWaterPt.pLong)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> lastWaterPt.pLat)) return err;
	if (err = ReadMacValue(bfpb, &theLE -> beachTime)) return err;

	return err;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

TOLEList::TOLEList()
{
	memset(&fSetSummary,0,sizeof(fSetSummary));
	bMassBalanceOpen = false;
	bReleasePositionOpen = false;
	initialLEs = nil;
	binitialLEsVisible = true;
	bShowDispersantArea = false;
	memset(&fDispersantData,0,sizeof(fDispersantData));
	//memset(&fWindageData,0,sizeof(fWindageData));
	fWindageData.windageA = .01;
	fWindageData.windageB = .04;
	fWindageData.persistence = .25;	// in hours		
	fAdiosDataH = nil;	
	fBudgetTableH = nil;	
	fOilTypeName[0] = 0;
}

void TOLEList::Dispose ()
{
	if (initialLEs)	// dispose of LE distribution array, if any, STH
	{
		initialLEs->Dispose();// for some reason we need to do this , JLM 2/19/03
		delete (initialLEs);
		initialLEs = nil;
	}
	if (fAdiosDataH)
	{
		DisposeHandle((Handle)fAdiosDataH); 
		fAdiosDataH = 0;
	}
	if (fBudgetTableH)
	{
		DisposeHandle((Handle)fBudgetTableH); 
		fBudgetTableH = 0;
	}

	TLEList::Dispose ();
}

OSErr TOLEList::CalculateAverageIntrusionDepth(double *avDepth, double *stdDev)
{	// this was just a diagnostic for Debra, not using anymore...
	long j, numSubsurfaceLEs = 0;
	double totalDepth = 0, totalDev = 0;
	LERec thisLE;
	OSErr err = 0;
	for (j = 0; j < this -> numOfLEs; j++)
	{
		this -> GetLE (j, &thisLE);
		if (thisLE.z > 0)
		{
			numSubsurfaceLEs ++;
			totalDepth += thisLE.z;			
		}
	}
	if (numSubsurfaceLEs>0) totalDepth /= numSubsurfaceLEs;
	for (j = 0; j < this -> numOfLEs; j++)
	{
		this -> GetLE (j, &thisLE);
		if (thisLE.z > 0)
		{
			totalDev += (thisLE.z - totalDepth)*(thisLE.z - totalDepth);
		}
	}
	*avDepth = totalDepth;
	if (numSubsurfaceLEs > 0) *stdDev = sqrt(totalDev/numSubsurfaceLEs); 
	return err;		
}

OSErr TOLEList::Reset(Boolean newKeys)
{
	long i;
	double timeStep = 0, latStep = 0, longStep = 0;
//	LERec representativeLE = fSetSummary.representativeLE;
	LERec representativeLE;
	
	memset(&representativeLE,0,sizeof(representativeLE));
	
	representativeLE.leKey = 0;
	representativeLE.leCustomData = nil;
	representativeLE.p = fSetSummary.startRelPos;
	representativeLE.z = fSetSummary.z;
	representativeLE.releaseTime = fSetSummary.startRelTime;
	representativeLE.ageInHrsWhenReleased = fSetSummary.ageInHrsWhenReleased;
	representativeLE.clockRef = 0;
	representativeLE.pollutantType = fSetSummary.pollutantType;
	representativeLE.mass = fSetSummary.totalMass / (double) fSetSummary.numOfLEs;
	representativeLE.density = fSetSummary.density;
	representativeLE.statusCode = OILSTAT_NOTRELEASED;// set below for each LE
//	representativeLE.lastWaterPt =
//	representativeLE.beachTime =
	// code goes here, windage?
//	representativeLE.windage = GetRandomFloat(fSetSummary.windageA,fSetSummary.windageB)
	representativeLE.dropletSize = 0;
	representativeLE.dispersionStatus = 0;	// DONT_DISPERSE - what if selected via lasso ??
	representativeLE.riseVelocity = fSetSummary.riseVelocity;	// default velocity zero
	// set private field to be same as set
	numOfLEs = fSetSummary.numOfLEs;
	
	if (fSetSummary.bWantEndRelTime && numOfLEs > 1)
	{
		/*if (!(model->bHindcast))*/ timeStep = (fSetSummary.endRelTime  - fSetSummary.startRelTime) / (double) (fSetSummary.numOfLEs - 1);
		//else timeStep = -1. * (fSetSummary.startRelTime - fSetSummary.endRelTime) / (double) (fSetSummary.numOfLEs - 1);
	}
	
	if (fSetSummary.bWantEndRelPosition && numOfLEs > 1)
	{
		latStep = (fSetSummary.endRelPos.pLat - fSetSummary.startRelPos.pLat) / (double) (fSetSummary.numOfLEs - 1);
		longStep = (fSetSummary.endRelPos.pLong - fSetSummary.startRelPos.pLong) / (double) (fSetSummary.numOfLEs - 1);
	}
	
	
	for (i = 0 ; i < fSetSummary.numOfLEs ; i++) {
	
		// if we have an initial LE distribution array, use data in that array 
		if (initialLEs)
		{
			InitialLEInfoRec	thisLEInfoRec;
			
			initialLEs -> GetListItem ((Ptr) &thisLEInfoRec, i);
			
			// use the individual LE properties
			representativeLE.p = thisLEInfoRec.p;
			representativeLE.z = thisLEInfoRec.z;
			representativeLE.releaseTime = thisLEInfoRec.releaseTime;
			representativeLE.ageInHrsWhenReleased = thisLEInfoRec.ageInHrsWhenReleased;
			representativeLE.pollutantType = thisLEInfoRec.pollutantType;
			//1/23/03 we added density to the InitialLEInfoRec
			representativeLE.density = thisLEInfoRec.density;
	
		}
		else
		{	// point to line source
			representativeLE.p.pLong = fSetSummary.startRelPos.pLong + (long) (i * longStep);
			representativeLE.p.pLat = fSetSummary.startRelPos.pLat + (long) (i * latStep);
			representativeLE.releaseTime = fSetSummary.startRelTime + (long) (i * timeStep);
		}
		if (fDispersantData.lassoSelectedLEsToDisperse)	// want to reset to original status
		{
			if (INDEXH(LEHandle, i).dispersionStatus == HAVE_DISPERSED)
				representativeLE.dispersionStatus = DISPERSE;
			else if (INDEXH(LEHandle, i).dispersionStatus == HAVE_DISPERSED_NAT)
				representativeLE.dispersionStatus = DISPERSE_NAT;
			else if (INDEXH(LEHandle, i).dispersionStatus == HAVE_EVAPORATED)
				representativeLE.dispersionStatus = EVAPORATE;
			else // haven't yet done whatever
				representativeLE.dispersionStatus = INDEXH(LEHandle, i).dispersionStatus;
			representativeLE.beachTime = INDEXH(LEHandle, i).beachTime;	// the disperse time or zero
			//else // haven't yet done whatever
				//representativeLE.dispersionStatus = 0;
		}
		
		/////////////////////////////////////////////////
		// JLM 10/22/99
		// I think we should set the status for all LE's based on 
		// their release time being <= to the model time so that the LE's are in the water at step 0
		// representativeLE.statusCode = OILSTAT_NOTRELEASED;
		/////////////////////////////////////////////////
		//if (representativeLE.releaseTime > model -> GetModelTime())
		if ((!(model->bHindcast) && representativeLE.releaseTime > model -> GetModelTime()) || (model->bHindcast && representativeLE.releaseTime < model -> GetModelTime()))
			representativeLE.statusCode = OILSTAT_NOTRELEASED;
		else
			representativeLE.statusCode = OILSTAT_INWATER;
		/////////////////////////////////////////////////

		representativeLE.clockRef = 0;
		if (newKeys)
			representativeLE.leKey = model->GetNextLEKey();
		else
			representativeLE.leKey = INDEXH(LEHandle, i).leKey;


		INDEXH(LEHandle, i) = representativeLE;
	}
	
	if (fBudgetTableH)	// why is this done here?
	{
		DisposeHandle((Handle)fBudgetTableH); 
		fBudgetTableH = 0;
	}
	return 0;
}


OSErr TOLEList::Initialize(LESetSummary * summary,Boolean deleteLERecH)
{
	OSErr err = 0;
	if (LEHandle && deleteLERecH)
		{DisposeHandle ((Handle) LEHandle); LEHandle=0;}  // allow repeat initializations
	
	fSetSummary = *summary;
	
	if (!LEHandle)
	{
	LEHandle = (LERecH)_NewHandle(fSetSummary.numOfLEs * sizeof(LERec));
	if (!LEHandle)
		TechError("TOLEList::Initialize()", "_NewHandle()", 0);
	}
	
	if (err = Reset(TRUE)) return err;
	
	return 0;
}


/**************************************************************************************************/
Boolean operator == (WorldPoint p1, WorldPoint p2)
{
	if (p1.pLong == p2.pLong && p1.pLat == p2.pLat)
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
Boolean operator != (WorldPoint p1, WorldPoint p2)
{
	if (p1.pLong == p2.pLong && p1.pLat == p2.pLat)
		return (false);
	else
		return (true);
}
/**************************************************************************************************/

OSErr TOLEList::Initialize(long numElements, LERecH array, short massUnits, Seconds fileTime)
{
	short pollutantType = OIL_CONSERVATIVE;
	long i;
	double totalMass = 0;
	
	if (LEHandle)
		DisposeHandle ((Handle) LEHandle);  // allow repeat initializations

	LEHandle = array;
	
	for (i = 0 ; i < numElements ; i++) {
		totalMass += INDEXH(LEHandle, i).mass;
		if (i == 0)
			pollutantType = INDEXH(LEHandle, i).pollutantType;
		else
			if (INDEXH(LEHandle, i).pollutantType != pollutantType)
				pollutantType = OIL_COMBINATION;
	}
	
	// actually using fSetSummary does not make much sense for an LE set read from a file
	// code goes here
	
	numOfLEs = numElements;
	fSetSummary.numOfLEs = numElements;
	fSetSummary.pollutantType = pollutantType;
	fSetSummary.totalMass = totalMass;
	fSetSummary.massUnits = massUnits; // JLM, code goes here, what are the units in an LE file ? how do you get the total mass
	fSetSummary.startRelPos = INDEXH(LEHandle, 0).p;
	fSetSummary.endRelPos   = INDEXH(LEHandle, numElements - 1).p;
	fSetSummary.startRelTime = INDEXH(LEHandle, 0).releaseTime;
	fSetSummary.endRelTime   = INDEXH(LEHandle, numElements - 1).releaseTime;
	fSetSummary.z   = INDEXH(LEHandle, 0).z;	// this only works if all have same value...
//	SetWPoint(&fSetSummary.startRelPos, 0, 0);
//	SetWPoint(&fSetSummary.endRelPos, 0, 0);
//	fSetSummary.startRelTime = fileTime;
//	fSetSummary.bWantEndRelTime = false;
//	fSetSummary.endRelTime = 0;

	if (fSetSummary.startRelPos != fSetSummary.endRelPos)
		fSetSummary.bWantEndRelPosition = true;
	else
		fSetSummary.bWantEndRelPosition = false;
	
	if (fSetSummary.startRelTime != fSetSummary.endRelTime)
		fSetSummary.bWantEndRelTime = true;
	else
		fSetSummary.bWantEndRelTime = false;
	
	return 0;
}

//#define TOLEListVersion 2
//#define TOLEListVersion 3 // added windage ?
//#define TOLEListVersion 4  // 1/23/03, added density to initial LEs
//#define TOLEListVersion 5  // 11/3/04, added spill name
//#define TOLEListVersion 6  // 2/25/08, added the dispersed oil fields
#define TOLEListVersion 7  // 8/23/10, added rise velocity

OSErr TOLEList::Write(BFPB *bfpb)
{
	long i, version = TOLEListVersion, numDataValues=0;
	ClassID id = GetClassID ();
	OSErr	err = noErr;
	Boolean	bHasInitialLEs;

	if(err = TLEList::Write(bfpb)) return err;
	
	StartReadWriteSequence("TOLEList::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
//	if (err = WriteMacValue(bfpb, bOpen)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.numOfLEs)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.pollutantType)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.totalMass)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.massUnits)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.startRelTime)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.endRelTime)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.startRelPos.pLong)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.startRelPos.pLat)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.endRelPos.pLong)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.endRelPos.pLat)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.bWantEndRelTime)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.bWantEndRelPosition)) return err;
	
	if (err = WriteMacValue(bfpb, fSetSummary.z)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.density)) return err;
	if (err = WriteMacValue(bfpb, fSetSummary.ageInHrsWhenReleased)) return err;

	if (err = WriteMacValue(bfpb, fSetSummary.riseVelocity)) return err;
	//if (!gMearnsVersion)
	{if (err = WriteMacValue(bfpb, fSetSummary.spillName, kMaxNameLen)) return err;}
	
	if (err = WriteMacValue(bfpb, fWindageData.windageA)) return err;
	if (err = WriteMacValue(bfpb, fWindageData.windageB)) return err;
	if (err = WriteMacValue(bfpb, fWindageData.persistence)) return err;

	//if (gMearnsVersion)
	{
	if (err = WriteMacValue(bfpb, fDispersantData.bDisperseOil)) return err;
	if (err = WriteMacValue(bfpb, fDispersantData.timeToDisperse)) return err;
	if (err = WriteMacValue(bfpb, fDispersantData.duration)) return err;
	if (err = WriteMacValue(bfpb, fDispersantData.amountToDisperse)) return err;
	if (err = WriteMacValue(bfpb, fDispersantData.areaToDisperse)) return err;
	if (err = WriteMacValue(bfpb, fDispersantData.lassoSelectedLEsToDisperse)) return err;
	if (err = WriteMacValue(bfpb, fDispersantData.api)) return err;
	}
	// save flag indicating whether we have a spill distribution array, STH
	if (initialLEs)
		bHasInitialLEs = true;
	else
		bHasInitialLEs = false;
	if (err = WriteMacValue(bfpb, bHasInitialLEs)) return err;

	if (initialLEs)
	{
		InitialLEInfoRec	thisLEInfoRec;
		
		for (i = 0 ; i < numOfLEs ; i++) {
			initialLEs -> GetListItem ((Ptr) &thisLEInfoRec, i);

			if (err = WriteMacValue(bfpb, thisLEInfoRec.p.pLat)) return err;
			if (err = WriteMacValue(bfpb, thisLEInfoRec.p.pLong)) return err;
			if (err = WriteMacValue(bfpb, thisLEInfoRec.z)) return err;
			if (err = WriteMacValue(bfpb, thisLEInfoRec.ageInHrsWhenReleased)) return err;
			if (err = WriteMacValue(bfpb, thisLEInfoRec.releaseTime)) return err;
			if (err = WriteMacValue(bfpb, thisLEInfoRec.pollutantType)) return err;
			if (err = WriteMacValue(bfpb, thisLEInfoRec.density)) return err;//1/23/03
		}
	}

	//if (gMearnsVersion)
	{
	//if (version > 4) 
	{// 
		if (fAdiosDataH) numDataValues = _GetHandleSize((Handle)fAdiosDataH)/sizeof(**fAdiosDataH);
		if (err = WriteMacValue(bfpb, numDataValues)) return err;
		if (fAdiosDataH)
		{
			for (i=0;i<numDataValues;i++)
			{
				if (err = WriteMacValue(bfpb, (*fAdiosDataH)[i].timeAfterSpill)) return err;
				if (err = WriteMacValue(bfpb, (*fAdiosDataH)[i].amountDispersed)) return err;
				if (err = WriteMacValue(bfpb, (*fAdiosDataH)[i].amountEvaporated)) return err;
			}
		}

	}
	}
	return 0;
}

OSErr TOLEList::Read(BFPB *bfpb)
{
	char 	c;
	long 	i, version, numDataValues;
	ClassID id;
	OSErr	err = noErr;
	long	LERecSize;
	Boolean	bHadInitialLEs;
	
	
	if(err = TLEList::Read(bfpb)) return err;
	
	StartReadWriteSequence("TOLEList::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TOLEList::Read()", "id != GetClassID", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	
	//if (version != TOLEListVersion) { printSaveFileVersionError(); return -1; }
	if (version < 2 || version > TOLEListVersion) { printSaveFileVersionError(); return -1; }
	
	//if (err = ReadMacValue(bfpb, &bOpen)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.numOfLEs)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.pollutantType)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.totalMass)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.massUnits)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.startRelTime)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.endRelTime)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.startRelPos.pLong)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.startRelPos.pLat)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.endRelPos.pLong)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.endRelPos.pLat)) return err;
	
	if (err = ReadMacValue(bfpb, &fSetSummary.bWantEndRelTime)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.bWantEndRelPosition)) return err;

	if (err = ReadMacValue(bfpb, &fSetSummary.z)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.density)) return err;
	if (err = ReadMacValue(bfpb, &fSetSummary.ageInHrsWhenReleased)) return err;
	
	if (version > 6) 
		if (err = ReadMacValue(bfpb, &fSetSummary.riseVelocity)) return err;


	if ((version > 4 && !gMearnsVersion) || version > 5)
	{
		if (err = ReadMacValue(bfpb, fSetSummary.spillName, kMaxNameLen)) return err;
	}
	else
	{
		/*if (this->IAm(TYPE_SPRAYLELIST)) strcpy(fSetSummary.spillName,"Overflight");	// default value?
		else 
		{
			if (fSetSummary.bWantEndRelPosition)
				strcpy(fSetSummary.spillName,"Line Source");
			else
				strcpy(fSetSummary.spillName,"Point Source");
		}*/
	}
	
	
	if (version > 2)	// otherwise gets default values
	{
		if (err = ReadMacValue(bfpb, &fWindageData.windageA)) return err;
		if (err = ReadMacValue(bfpb, &fWindageData.windageB)) return err;
		if (err = ReadMacValue(bfpb, &fWindageData.persistence)) return err;
	}

	if (gMearnsVersion || version > 5)
	{
	if (err = ReadMacValue(bfpb, &fDispersantData.bDisperseOil)) return err;
	if (err = ReadMacValue(bfpb, &fDispersantData.timeToDisperse)) return err;
	if (err = ReadMacValue(bfpb, &fDispersantData.duration)) return err;
	if (err = ReadMacValue(bfpb, &fDispersantData.amountToDisperse)) return err;
	if (err = ReadMacValue(bfpb, &fDispersantData.areaToDisperse)) return err;
	if (version>4) if (err = ReadMacValue(bfpb, &fDispersantData.lassoSelectedLEsToDisperse)) return err;
	if (err = ReadMacValue(bfpb, &fDispersantData.api)) return err;	// set density, if bDisperseOil
	}
	// read and check flag indicating whether we have an LE distribution array
	// then create and read-fill array if necessary, STH
	if (err = ReadMacValue(bfpb, &bHadInitialLEs)) return err;
	
	if (bHadInitialLEs)
	{
		InitialLEInfoRec	thisLEInfoRec;

		initialLEs = new CMyList (sizeof (InitialLEInfoRec));
		if (!initialLEs) { err = memFullErr; return err;}
		initialLEs -> IList ();

		for (i = 0 ; i < numOfLEs ; i++) {
			memset(&thisLEInfoRec,0,sizeof(thisLEInfoRec));
			if (err = ReadMacValue(bfpb, &(thisLEInfoRec.p.pLat))) return err;
			if (err = ReadMacValue(bfpb, &(thisLEInfoRec.p.pLong))) return err;
			if (err = ReadMacValue(bfpb, &(thisLEInfoRec.z))) return err;
			if (err = ReadMacValue(bfpb, &(thisLEInfoRec.ageInHrsWhenReleased))) return err;
			if (err = ReadMacValue(bfpb, &(thisLEInfoRec.releaseTime))) return err;
			if (err = ReadMacValue(bfpb, &(thisLEInfoRec.pollutantType))) return err;
			if (version > 3) {
				if (err = ReadMacValue(bfpb, &(thisLEInfoRec.density))) return err;
			} else {
				thisLEInfoRec.density = GetPollutantDensity(thisLEInfoRec.pollutantType);
			}

			initialLEs -> AppendItem ((Ptr) &thisLEInfoRec);
		}
	}

	// do we clean up on error or just leave whatever we get?
	if (gMearnsVersion || version > 5)
	{
	if (version > 4) 
	{// 
		if (err = ReadMacValue(bfpb, &numDataValues)) return err;
		if (numDataValues>0)
		{
			fAdiosDataH = (AdiosInfoRecH)_NewHandleClear(numDataValues*sizeof(**fAdiosDataH));
			if (!fAdiosDataH) { err = memFullErr; printError("Not enough memory"); return err;}
		}
		for (i=0;i<numDataValues;i++)
		{
			if (err = ReadMacValue(bfpb, &((*fAdiosDataH)[i].timeAfterSpill))) return err;
			if (err = ReadMacValue(bfpb, &((*fAdiosDataH)[i].amountDispersed))) return err;
			if (err = ReadMacValue(bfpb, &((*fAdiosDataH)[i].amountEvaporated))) return err;
		}
	}
	}
	return err;
}


///////////////////////////////////////////////////////////////////////////
OSErr TOLEList::CheckAndPassOnMessage(TModelMessage *message)
{	
	char ourName[kMaxNameLen];
	OSErr err = 0;
	UNIQUEID ourUniqueID = this -> GetUniqueID();
	long numLEs;
	WorldPoint startRelPos,endRelPos;
	Seconds startRelTime,endRelTime,windStartTime;
	OilType pollutantType;
	short massUnits;
	double z,totalMass;
	double windageA,windageB,persistence;
	Boolean bWantEndRelPosition,bWantEndRelTime;
	double density,ageInHrsWhenReleased;
	Boolean bSetTheLEFileReleaseTime = false; // 7/13/10
	Seconds LEFileReleaseTime;
	
	// see if the message is of concern to us
	this->GetClassName(ourName);
	
	/////////////////////////////////////////////////
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		// I'm not sure if the mirrored LE set has the same name, so we'll set the LE set herre too
		TOLEList* mirroredLEList = (TOLEList*)model->GetMirroredLEList(this);
		LESetSummary fSetSummaryNewVals = this->fSetSummary;
		char str[256];
		Boolean allowLLWithoutDir = true;	
		// look for the rest of the parameters that specify the spill		
		err = message->GetParameterAsLong("numLEs",&numLEs);
		if(!err && numLEs > 0) {
			fSetSummaryNewVals.numOfLEs = numLEs;
		} ///////////////////////////////////////////////////////////	
		err = message->GetParameterAsWorldPoint("startRelPos",&startRelPos,allowLLWithoutDir);
		if(!err) {
			fSetSummaryNewVals.startRelPos = startRelPos;
		} ///////////////////////////////////////////////////////////
		err = message->GetParameterAsSeconds("startRelTime",&startRelTime);
		if(!err) {
			fSetSummaryNewVals.startRelTime = startRelTime;
		} ////////
		
		err = message->GetParameterAsSeconds("endRelTime",&endRelTime);
		if(!err) {
			fSetSummaryNewVals.endRelTime = endRelTime;
			// seems like we should auto set bWantEndRelPosition here
			// we will check to see if they are setting a differnet value below
			fSetSummaryNewVals.bWantEndRelTime = true;
		} ////////

		err = message->GetParameterAsBoolean("bWantEndRelTime",&bWantEndRelTime);
		if(!err) {
			fSetSummaryNewVals.bWantEndRelTime = bWantEndRelTime;
		}
		////////////////////////////////////////////////

		
		err = message->GetParameterAsWorldPoint("endRelPos",&endRelPos,allowLLWithoutDir);
		if(!err) {
			fSetSummaryNewVals.endRelPos = endRelPos;
			// seems like we should auto set bWantEndRelPosition here
			// we will check to see if they are setting a differnet value below
			fSetSummaryNewVals.bWantEndRelPosition = true;
		} ////////
		
		err = message->GetParameterAsBoolean("bWantEndRelPosition",&bWantEndRelPosition);
		if(!err) {
			fSetSummaryNewVals.bWantEndRelPosition = bWantEndRelPosition;
		}

		//////////////////////////////////////

		message->GetParameterString("pollutantType",str,256);
		if(str[0]) {// this parameter is optional
			if(!strcmpnocase(str,"CONSERVATIVE") || !strcmpnocase(str,"Non-Weathering")) pollutantType = OIL_CONSERVATIVE;
			else if(!strcmpnocase(str,"BUNKER") || !strcmpnocase(str,"Fuel Oil #6")) pollutantType = OIL_6;
			else if(!strcmpnocase(str,"MEDIUMCRUDE") || !strcmpnocase(str,"Medium Crude")) pollutantType = OIL_CRUDE;
			else if(!strcmpnocase(str,"IFO") || !strcmpnocase(str,"Fuel Oil #4")) pollutantType = OIL_4;
			else if(!strcmpnocase(str,"DIESEL")) pollutantType = OIL_DIESEL;
			else if(!strcmpnocase(str,"JP4") || !strcmpnocase(str,"Kerosene / Jet Fuels")) pollutantType = OIL_JETFUELS;
			else if(!strcmpnocase(str,"GAS") || !strcmpnocase(str,"Gasoline")) pollutantType = OIL_GAS;
			else err = -1;
			if(!err) {
				fSetSummaryNewVals.pollutantType = pollutantType;
			}
		} ////////
		
	
		message->GetParameterString("massUnits",str,256);
		if(str[0]) {// this parameter is optional
			if(!strcmpnocase(str,"BARRELS")) massUnits = BARRELS;
			else if(!strcmpnocase(str,"GALLONS")) massUnits = GALLONS;
			else if(!strcmpnocase(str,"CUBICMETERS")) massUnits = CUBICMETERS;
			else if(!strcmpnocase(str,"KILOGRAMS")) massUnits = KILOGRAMS;
			else if(!strcmpnocase(str,"METRICTONS")) massUnits = METRICTONS;
			else if(!strcmpnocase(str,"SHORTTONS")) massUnits = SHORTTONS;
			else err = -1;
			if(!err) {
				fSetSummaryNewVals.massUnits = massUnits;
			}
		} ////////
		
		err = message->GetParameterAsDouble("totalMass",&totalMass);
		if(!err && totalMass > 0.0) {
			fSetSummaryNewVals.totalMass = totalMass;
		} ////////
	
	
		err = message->GetParameterAsDouble("z",&z);
		if(! err && z >= 0.0) {
			fSetSummaryNewVals.z = z;
		} ////////

		err = message->GetParameterAsDouble("density",&density);
		if(! err && density >= 0.0) {
			fSetSummaryNewVals.density = density;
		} ////////

		err = message->GetParameterAsDouble("ageInHrsWhenReleased",&ageInHrsWhenReleased);
		if(! err && ageInHrsWhenReleased >= 0.0) {
			fSetSummaryNewVals.ageInHrsWhenReleased = ageInHrsWhenReleased;
		} ////////
		////////////////////////////////////////
		////////////////////////////////////////
	
		err = message->GetParameterAsDouble("windageA",&windageA);
		if(!err && windageA >= 0.0) {
			this->fWindageData.windageA = windageA;
			if(mirroredLEList) mirroredLEList->fWindageData.windageA = windageA;
		} ////////
	
		err = message->GetParameterAsDouble("windageB",&windageB);
		if(!err && windageB >= 0.0 && windageB >= this->fWindageData.windageA) {
			this->fWindageData.windageB = windageB;
			if(mirroredLEList) mirroredLEList->fWindageData.windageB = windageB;
		} ////////
	
		err = message->GetParameterAsDouble("persistence",&persistence);
		if(!err && (persistence >= 0.0 || persistence == -1)) {	// persistence of -1 is infinite
			this->fWindageData.persistence = persistence;
			if(mirroredLEList) mirroredLEList->fWindageData.persistence = persistence;
		} ////////

		// JLM 7/13/10, try setting this for Chris for TAP
		err = message->GetParameterAsSeconds("LEFileReleaseTime",&LEFileReleaseTime); 
		if(!err) {
			// set the header and also the release time in the LE list after initialize
			bSetTheLEFileReleaseTime = true;
			fSetSummaryNewVals.startRelTime = LEFileReleaseTime;
			fSetSummaryNewVals.endRelTime = LEFileReleaseTime;
			fSetSummaryNewVals.bWantEndRelTime = false;
		}
		///////////////


		this->Initialize(&fSetSummaryNewVals,true);
		if(mirroredLEList) mirroredLEList->Initialize(&fSetSummaryNewVals,true);

		/////////////
		if(bSetTheLEFileReleaseTime) {
			// set the release time for each LE 
			int i;
			InitialLEInfoRec	thisLEInfoRec;
			for (i = 0 ; i < fSetSummary.numOfLEs ; i++) {
				if(this->initialLEs) {
					this -> initialLEs -> GetListItem ((Ptr) &thisLEInfoRec, i);
					thisLEInfoRec.releaseTime = LEFileReleaseTime;
					this -> initialLEs -> SetListItem ((Ptr) &thisLEInfoRec, i);
				}
				if(mirroredLEList && mirroredLEList->initialLEs){
					mirroredLEList -> initialLEs -> GetListItem ((Ptr) &thisLEInfoRec, i);
					thisLEInfoRec.releaseTime = LEFileReleaseTime;
					mirroredLEList -> initialLEs -> SetListItem ((Ptr) &thisLEInfoRec, i);
				}
			}
		}
		/////////////////////
		model->NewDirtNotification();// tell model about dirt

	}
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TLEList::CheckAndPassOnMessage(message);
	
}



void TOLEList::AddToBudgetTableHdl(BudgetTableData *budgetTable)
{	// may want to switch this to a model field, then can total all the spills information
	// rather than have a separate budget table for each spill
	long i,sizeOfHdl;
	OSErr err = 0;
	if(!fBudgetTableH)
	{
		fBudgetTableH = (BudgetTableData**)_NewHandle(0);
		if(!fBudgetTableH) {TechError("TOLEList::AddToBudgetTableHdl()", "_NewHandle()", 0); err = memFullErr; return;}
	}
	sizeOfHdl = _GetHandleSize((Handle)fBudgetTableH)/sizeof(BudgetTableData);
	_SetHandleSize((Handle) fBudgetTableH, (sizeOfHdl+1)*sizeof(BudgetTableData));
	if (_MemError()) { TechError("TOLEList::AddToBudgetTableHdl()", "_SetHandleSize()", 0); return; }
	(*fBudgetTableH)[sizeOfHdl].timeAfterSpill = (*budgetTable).timeAfterSpill;				
	(*fBudgetTableH)[sizeOfHdl].amountReleased = (*budgetTable).amountReleased;				
	(*fBudgetTableH)[sizeOfHdl].amountFloating = (*budgetTable).amountFloating;				
	(*fBudgetTableH)[sizeOfHdl].amountDispersed = (*budgetTable).amountDispersed;				
	(*fBudgetTableH)[sizeOfHdl].amountEvaporated = (*budgetTable).amountEvaporated;				
	(*fBudgetTableH)[sizeOfHdl].amountBeached = (*budgetTable).amountBeached;				
	(*fBudgetTableH)[sizeOfHdl].amountOffMap = (*budgetTable).amountOffMap;
	(*fBudgetTableH)[sizeOfHdl].amountRemoved = (*budgetTable).amountRemoved;
	return;
}

OSErr TOLEList::ExportBudgetTableHdl(char* path, BFPB *bfpb)
{
	// want to change this to calculate total budget for multiple spills
	OSErr err = 0;
	DateTimeRec dateTime;
	Seconds time;
	double amttotal,amtevap,amtbeached,amtoffmap,amtfloating,amtreleased,amtdispersed,amtremoved;
	BudgetTableData budgetTable; 
	long numOutputValues,i;
	char buffer[512],unitsStr[64],timeStr[64],massStr[64];
	char amtEvapStr[64],amtDispStr[64],amtBeachedStr[64],amtRelStr[64],amtFloatStr[64],amtOffStr[64],amtRemStr[64];
	double totalMass = fSetSummary.totalMass;
	GetLeUnitsStr(unitsStr,fSetSummary.massUnits);

	// Write out the times and values
	// add header line
	StringWithoutTrailingZeros(massStr,totalMass,3);
	strcpy(buffer,"Amount Spilled - ");
	strcat(buffer,massStr);
	strcat(buffer," ");
	strcat(buffer,unitsStr);
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(bfpb, buffer, strlen(buffer))) goto done;
	//strcpy(buffer,"Hr\tRel\tFloat\tEvap\tDisp\tBeach\tOffMap");
	strcpy(buffer,"Hr\tRel\tFloat\tEvap\tDisp\tBeach\tOffMap\tRemoved");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(bfpb, buffer, strlen(buffer))) goto done;
	numOutputValues = _GetHandleSize((Handle)fBudgetTableH)/sizeof(BudgetTableData);
	for(i = 0; i< numOutputValues;i++)
	{
		budgetTable = INDEXH(fBudgetTableH,i);
		time = budgetTable.timeAfterSpill;
		amtreleased = budgetTable.amountReleased;
		amtfloating = budgetTable.amountFloating;
		amtdispersed = budgetTable.amountDispersed;
		amtevap = budgetTable.amountEvaporated;
		amtbeached = budgetTable.amountBeached;
		amtoffmap = budgetTable.amountOffMap;
		amtremoved = budgetTable.amountRemoved;
		/*SecondsToDate(time,&dateTime); // convert to 2 digit year?
		sprintf(timeStr, "%02hd,%02hd,%02hd,%02hd,%02hd",
			   dateTime.day, dateTime.month, dateTime.year,
			   dateTime.hour, dateTime.minute);*/

		StringWithoutTrailingZeros(timeStr,time/model->LEDumpInterval,3);
		StringWithoutTrailingZeros(amtEvapStr,amtevap,3);
		StringWithoutTrailingZeros(amtDispStr,amtdispersed,3);
		StringWithoutTrailingZeros(amtFloatStr,amtfloating,3);
		StringWithoutTrailingZeros(amtRelStr,amtreleased,3);
		StringWithoutTrailingZeros(amtBeachedStr,amtbeached,3);
		StringWithoutTrailingZeros(amtOffStr,amtoffmap,3);
		StringWithoutTrailingZeros(amtRemStr,amtremoved,3);
		/////
		strcpy(buffer,timeStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtRelStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtFloatStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtEvapStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtDispStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtBeachedStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtOffStr);
		strcat(buffer,NEWLINESTRING);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtRemStr);
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(bfpb, buffer, strlen(buffer))) goto done;
	}

done:
	return err;
}

void TOLEList::GetMassBalanceLines(Boolean includePercentage, char* line1,char*line2,char* line3,char* line4,char* line5,char* line6,char* line7)
{
	char infoStr [255];
	char unitsStr [64];
	float fractionOfReleased,fractionOfTotal;
	double totalMass,massReleased;
	long numReleased,numEvaporated,numBeached,numOffMap,numFloating,numDispersed=0,numRemoved=0;
	long numLEs = this->GetLECount();
	short numDecPlaces;
	char ofTotalStr[64] = "",ofReleasedStr[64] = "";
	char percentageStr[64];
	this->GetLEStatistics(&numReleased,&numEvaporated,&numBeached,&numOffMap,&numFloating);
	GetLeUnitsStr(unitsStr,fSetSummary.massUnits);
	totalMass = fSetSummary.totalMass;
	if(totalMass < 100) numDecPlaces = 1;
	else numDecPlaces = 0;
	massReleased = (totalMass* numReleased)/numLEs;
	if(0 < numReleased && numReleased < numLEs)
	{	// could be confusing so label percents
		strcpy(ofTotalStr," of total");
		strcpy(ofReleasedStr," of released");
	}
	////////////
	if (line1) {// released
		fractionOfTotal = numReleased/(double)numLEs;
		StringWithoutTrailingZeros(infoStr,fractionOfTotal * totalMass,numDecPlaces); 
		sprintf(line1, "Released: %s %s",infoStr,unitsStr);
		sprintf(percentageStr, " (%.1f%%%s)",100.0*fractionOfTotal,ofTotalStr);
		if(includePercentage) strcat(line1,percentageStr);
	}
	// code goes here, do we want to categorize bottom spills as floating or subsurface?
	if (line2) {// floating
		if (fDispersantData.bDisperseOil || fAdiosDataH || fSetSummary.z > 0) this->RecalculateLEStatistics(&numDispersed,&numFloating,&numRemoved,&numOffMap);
		fractionOfReleased = (numReleased <= 0) ? 0: numFloating/(double)numReleased;
		fractionOfTotal = numFloating/(double)numLEs;
		StringWithoutTrailingZeros(infoStr,fractionOfTotal * totalMass,numDecPlaces); 
		sprintf(line2, "Floating: %s %s",infoStr,unitsStr);
		sprintf(percentageStr, " (%.1f%%%s)",100.0*fractionOfReleased,ofReleasedStr);
		if(includePercentage) strcat(line2,percentageStr);
	}

	if (line3) {// beached
		fractionOfReleased = (numReleased <= 0) ? 0: numBeached/(double)numReleased;
		fractionOfTotal = numBeached/(double)numLEs;
		StringWithoutTrailingZeros(infoStr,fractionOfTotal * totalMass,numDecPlaces); 
		sprintf(line3, "Beached: %s %s",infoStr,unitsStr);
		sprintf(percentageStr, " (%.1f%%%s )",100.0*fractionOfReleased,ofReleasedStr);
		if(includePercentage) strcat(line3,percentageStr);
	}

	// code goes here, for natural dispersants may want to do something different here
	if (line4) {// evaporated/dispersed
		//if (fDispersantData.bDisperseOil || fAdiosDataH) this->RecalculateLEStatistics(&numDispersed,&numFloating,&numRemoved,&numOffMap);
		if (this->GetOilType()==CHEMICAL) 
		{ 
			double totalMassRemaining = 0;
			if (LEHandle) 
			{
				long i;
				for (i = 0 ; i < this->numOfLEs ; i++) 
				{
					LERec theLE = INDEXH(LEHandle, i);
					double LEmass = GetLEMass(theLE);	// for time dependent release might want just percent of released
					totalMassRemaining += LEmass;
				}
				
			}
			if (totalMassRemaining/(numReleased*totalMass/(double)numLEs) > 1)
			//if ((1-totalMassRemaining/totalMass)*100 > 99)
				//sprintf(line4, "Dissolved: %.2f %% of released",(1-totalMassRemaining/(numReleased*totalMass/(double)numLEs))*100);	
				sprintf(line4, "Dissolved: 0 %% of released");	// assume this is sig digit issue
			else				
				sprintf(line4, "Dissolved: %.1f %% of released",(1-totalMassRemaining/(numReleased*totalMass/(double)numLEs))*100);	
			//sprintf(line4, "Dissolved: %.1f %%",(1-totalMassRemaining/totalMass)*100);	
			return;
		}

		fractionOfReleased = (numReleased <= 0) ? 0: numEvaporated/(double)numReleased;
		fractionOfTotal = numEvaporated/(double)numLEs;
		StringWithoutTrailingZeros(infoStr,fractionOfTotal * totalMass,numDecPlaces); 
		if (fDispersantData.bDisperseOil || fAdiosDataH || fSetSummary.z > 0)
			sprintf(line4, "Evaporated: %s %s",infoStr,unitsStr);	// split into 2
		else
			sprintf(line4, "Evaporated and Dispersed: %s %s",infoStr,unitsStr);	// split into 2
		sprintf(percentageStr, " (%.1f%%%s)",100.0*fractionOfReleased,ofReleasedStr);
		if(includePercentage) strcat(line4,percentageStr);
	}
	
	if (line5) {// Dispersed
		if (fDispersantData.bDisperseOil || fAdiosDataH || fSetSummary.z > 0) this->RecalculateLEStatistics(&numDispersed,&numFloating,&numRemoved,&numOffMap);
		fractionOfReleased = (numReleased <= 0) ? 0: numDispersed/(double)numReleased;
		fractionOfTotal = numDispersed/(double)numLEs;
		StringWithoutTrailingZeros(infoStr,fractionOfTotal * totalMass,numDecPlaces); 
		if (fSetSummary.z > 0) sprintf(line5, "Subsurface: %s %s",infoStr,unitsStr);	// should check if chemical
		else sprintf(line5, "Dispersed: %s %s",infoStr,unitsStr);
		sprintf(percentageStr, " (%.1f%%%s)",100.0*fractionOfReleased,ofReleasedStr);
		if(includePercentage) strcat(line5,percentageStr);
	}

	if (line6) {// offmap
		if (fAdiosDataH) this->RecalculateLEStatistics(&numDispersed,&numFloating,&numRemoved,&numOffMap);
		fractionOfReleased = (numReleased <= 0) ? 0: numOffMap/(double)numReleased;
		fractionOfTotal = numOffMap/(double)numLEs;
		StringWithoutTrailingZeros(infoStr,fractionOfTotal * totalMass,numDecPlaces); 
		sprintf(line6, "Off map: %s %s",infoStr,unitsStr);
		sprintf(percentageStr, " (%.1f%%%s)",100.0*fractionOfReleased,ofReleasedStr);
		if(includePercentage) strcat(line6,percentageStr);
	}

	if (line7) {// removed
		if (fAdiosDataH) this->RecalculateLEStatistics(&numDispersed,&numFloating,&numRemoved,&numOffMap);
		fractionOfReleased = (numReleased <= 0) ? 0: numRemoved/(double)numReleased;
		fractionOfTotal = numRemoved/(double)numLEs;
		StringWithoutTrailingZeros(infoStr,fractionOfTotal * totalMass,numDecPlaces); 
		sprintf(line7, "Removed: %s %s",infoStr,unitsStr);
		sprintf(percentageStr, " (%.1f%%%s)",100.0*fractionOfReleased,ofReleasedStr);
		if(includePercentage) strcat(line7,percentageStr);
	}

}


long TOLEList::GetListLength()
{
	long summaryLines = 0;
	
	Boolean owned = (fOwnersUniqueID.counter != 0);
	if(owned) return 0; // we will not be in the list
	
	summaryLines++; // toggle with oil type and amount
	
	if(bOpen)
	{
		if(model->GetModelMode() == ADVANCEDMODE) summaryLines++; // active button

		if(model->GetModelMode() == ADVANCEDMODE) summaryLines += 1;// windage

		if (fDispersantData.bDisperseOil) summaryLines++;	// disperse time
		if (fDispersantData.bDisperseOil) summaryLines++;	// draw disperse area
	
		if (fAdiosDataH) summaryLines++;	// natural dispersion
		
		summaryLines++; // show initial positions
		summaryLines++; // release time/position toggle
		
		if(bReleasePositionOpen)
		{
			summaryLines++; // release time
			if (fSetSummary.bWantEndRelTime) summaryLines++; // release end time
			summaryLines++; // release position
			if (fSetSummary.bWantEndRelPosition) summaryLines++; // release end position
		}
		
		summaryLines++; // Mass Balance toggle
		
		if (bMassBalanceOpen)
		{
			summaryLines++;// released 
			summaryLines++;// floating
			summaryLines++;// beached
			summaryLines++;// evaporated/dispersed
			if (fDispersantData.bDisperseOil || fAdiosDataH || fSetSummary.z > 0) summaryLines++;// dispersed
			summaryLines++;// offmap
			if (fAdiosDataH) summaryLines++;// removed, code goes here, separate check?
		}
				
	}
	
	return summaryLines;
}

ListItem TOLEList::GetNthListItem(long n, short indent, short *style, char *text)
{
	char dateStr [64],infoStr [255];
	char polutantName [64];
	char unitsStr [64], valStr[32], valStr2[32], valStr3[32];
	char latString[20],longString[20];
	long i;
	char roundLat,roundLong;
	LERec LE;
	ListItem item = { this, 0, indent, 0 };
	
	// number  and type of LE's
	if (n == 0) {
		item.index = I_LEFIRSTLINE;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		GetPollutantName (fSetSummary.pollutantType, polutantName);
		GetLeUnitsStr(unitsStr,fSetSummary.massUnits);
		StringWithoutTrailingZeros(infoStr,fSetSummary.totalMass,6); 
		//sprintf(text, "%s : %s %s", polutantName,infoStr,unitsStr);
		sprintf(text, "%s:  %s : %s %s", fSetSummary.spillName, polutantName,infoStr,unitsStr);
		if(!bActive)*style = italic; // JLM 6/14/10
		return item;
	}
	n -= 1;
	
	if (bOpen)
	{
		if (model->GetModelMode() == ADVANCEDMODE)
		{
			if (n == 0) {
				item.indent++;
				item.index = I_LEACTIVE;
				item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
				strcpy(text, "Active");
				return item;
			}
			n -= 1;
	
			////
			if (n == 0) 
			{
				item.index = I_LEWINDAGE;
				StringWithoutTrailingZeros(valStr,fWindageData.windageA * 100,2);
				StringWithoutTrailingZeros(valStr2,fWindageData.windageB * 100,2);
				if (fWindageData.persistence==-1) 
					sprintf(text, "Windage: %s%% to %s%%, Persistence: Infinite", valStr, valStr2);
				else
				{
					StringWithoutTrailingZeros(valStr3,fWindageData.persistence,2);
					sprintf(text, "Windage: %s%% to %s%%, Persistence: %s hrs", valStr, valStr2, valStr3);
				}
				return item;
			}
			else
				n -= 1;
		}
		//n -= 1;
		
		if (fDispersantData.bDisperseOil)
		{
			if (n == 0) {
				item.index = I_LEDISPERSE;
				//item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				sprintf(text, "Disperse chemically after %g hrs, %g%%",fDispersantData.timeToDisperse/3600.,fDispersantData.amountToDisperse*100);
				return item;
			}
			n -= 1;
			//  check box to draw dispersion area
			if (n == 0) {
				item.index = I_LEDRAWRECT;
				item.indent++;
				item.bullet = this->bShowDispersantArea ?  BULLET_FILLEDBOX : BULLET_EMPTYBOX;
				strcpy(text, "Show dispersant application area");
				return item;
			}
			n -= 1;
		}
		if (fAdiosDataH)
		{
			if (n == 0) {
				item.index = I_LENATURALDISP;
				//item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				//sprintf(text, "Disperse after %g hrs",fDispersantData.timeToDisperse/3600.);
				strcpy(text, "Natural Dispersion from Adios");
				return item;
			}
			n -= 1;
		}
		item.indent++; // all items are indented
		
		if (n == 0) {
			item.index = I_LESHOWHIDE;
			item.bullet = binitialLEsVisible ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Initial Positions");
			return item;
		}
		n -= 1;

		////
		if (n == 0) {
			item.index = I_LERELEASE_TIMEPOSITION;
			item.bullet = bReleasePositionOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			sprintf(text, "Release Time/Position");
			return item;
		}
		n -= 1;


		/////////////////////////////////////////////////
		if(bReleasePositionOpen)
		{
			// release time
			if (n == 0) {
				//item.indent++;
				Secs2DateString2 (fSetSummary.startRelTime, dateStr);
				if(fSetSummary.bWantEndRelTime) sprintf (text, "Start Time: %s", dateStr);
				else sprintf (text, "Time: %s", dateStr);
				return item;
			}
			n -= 1;
			
			if (fSetSummary.bWantEndRelTime)
			{
				if (n == 0) {
					//item.indent++;
					Secs2DateString2 (fSetSummary.endRelTime, dateStr);
					sprintf (text, "End Time: %s", dateStr);
					return item;
				}
				n -= 1;
			}
			
			// release position
			if (n == 0) {
				//item.indent++;
				WorldPointToStrings2(fSetSummary.startRelPos, latString, &roundLat, longString, &roundLong);	
				if (model->GetModelMode() != ADVANCEDMODE)
				{
					SimplifyLLString(longString, 3, roundLong);
					SimplifyLLString(latString, 3, roundLat);
				}
				if(fSetSummary.bWantEndRelPosition) sprintf(text, "Start Position: %s, %s", latString,longString);
				else sprintf(text, "Position: %s, %s", latString,longString);
				return item;
			}
			n -= 1;
	
			//////
			if (fSetSummary.bWantEndRelPosition)
			{
				if (n == 0) {
					//item.indent++;
					WorldPointToStrings2(fSetSummary.endRelPos, latString, &roundLat, longString, &roundLong);	
					if (model->GetModelMode() != ADVANCEDMODE)
					{
						SimplifyLLString(longString, 3, roundLong);
						SimplifyLLString(latString, 3, roundLat);
					}
					sprintf(text, "End Position: %s, %s", latString,longString);
					return item;
				}
				n -= 1;
			}
		}

		/////////////////////////////////////////////////
		
		////
		if (n == 0) {
			item.index = I_LERELEASE_MASSBALANCE;
			//item.indent++;
			item.bullet = bMassBalanceOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			sprintf(text, "Splot Mass Balance (Best Estimate)");
			return item;
		}
		n -= 1;

		////
		if (bMassBalanceOpen) // released, evaporated/dispersed, offmap, beached, removed
		{
			////////////
			if (n == 0) {// released
				this->GetMassBalanceLines(TRUE,text,nil,nil,nil,nil,nil,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (n == 0) {// floating
				this->GetMassBalanceLines(TRUE,nil,text,nil,nil,nil,nil,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (n == 0) {// beached
				this->GetMassBalanceLines(TRUE,nil,nil,text,nil,nil,nil,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (n == 0) {// evaporated/dispersed
				this->GetMassBalanceLines(TRUE,nil,nil,nil,text,nil,nil,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
			
			if (fDispersantData.bDisperseOil || fAdiosDataH || fSetSummary.z > 0)
			{
				if (n == 0) {// dispersed
					this->GetMassBalanceLines(TRUE,nil,nil,nil,nil,text,nil,nil);
					//item.indent++;
					return item;
				}
				n -= 1;
			}
			
			if (n == 0) {// offmap
				this->GetMassBalanceLines(TRUE,nil,nil,nil,nil,nil,text,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (fAdiosDataH)
			{
				if (n == 0) {// removed
					this->GetMassBalanceLines(TRUE,nil,nil,nil,nil,nil,nil,text);
					//item.indent++;
					return item;
				}
				n -= 1;
			}
		}

	}
	
	item.owner = 0;
	
	return item;
}

Boolean TOLEList::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	long i;
	
	if (inBullet && item.index == I_LEFIRSTLINE)
		{ bOpen = !bOpen; return TRUE; }
		
	if (inBullet && item.index == I_LESHOWHIDE) {
		binitialLEsVisible = !binitialLEsVisible;
		model->NewDirtNotification(DIRTY_MAPDRAWINGRECT);
		return TRUE;
	}

	if (inBullet && item.index == I_LEACTIVE)
		{ 
			TOLEList 	*uncertaintyLEList = (TOLEList*)model->GetMirroredLEList(this);
			bActive = !bActive; 
			uncertaintyLEList->SetActive(bActive);
			model->NewDirtNotification(DIRTY_RUNBAR); model->NewDirtNotification(); return TRUE; 
		}

	if (inBullet && item.index == I_LEDRAWRECT) {
		this->bShowDispersantArea = !this->bShowDispersantArea;
		InvalMapDrawingRect();
		return TRUE;
	}
				
	if (inBullet && item.index == I_LERELEASE_TIMEPOSITION)
		{ bReleasePositionOpen = !bReleasePositionOpen; return TRUE; }
		
	if (inBullet && item.index == I_LERELEASE_MASSBALANCE)
		{ bMassBalanceOpen = !bMassBalanceOpen; return TRUE; }
	
	//JLM  9/18/98
	if (doubleClick && item.index == I_LENATURALDISP)  {AdiosTable(fAdiosDataH,fOilTypeName,fDispersantData.api); return true;}
	if (doubleClick && item.index == I_LERELEASE_MASSBALANCE) {BudgetTable(GetMassUnits(),GetTotalMass(),fBudgetTableH); return true;}
	if(doubleClick)  return this -> SettingsItem (item); // do the settings
	
	// do other click operations...
	
	return FALSE;
}

Boolean TOLEList::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	if(item.index==I_LEFIRSTLINE)
	{	// code goes here, decide if this is okay
		switch (buttonID) {
			//case UPBUTTON:
			//case DOWNBUTTON:
				//return FALSE;
			case UPBUTTON:
			case DOWNBUTTON:
				//return TRUE;
				if (!model->LESetsList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
				switch (buttonID) {
					case UPBUTTON: return i > 0;
					case DOWNBUTTON: return i < (model->LESetsList->GetItemCount()/2 - 1);
			}
		}
	}
	
	if (item.index == I_LEFIRSTLINE)
		switch (buttonID) {
			case ADDBUTTON: return FALSE;
			case SETTINGSBUTTON: return TRUE;
			case DELETEBUTTON: return TRUE;
		}
	
	switch (buttonID) {
		case ADDBUTTON: return FALSE;
		case SETTINGSBUTTON: return TRUE;
		case DELETEBUTTON: return FALSE;
	}
	
	return FALSE;
}
// code goes here, decide if moving spills up/down is useful
OSErr TLEList::UpItem(ListItem item)
{	
	long i;
	OSErr err = 0;
	
	if (item.index == I_LEFIRSTLINE)
		if (model->LESetsList->IsItemInList((Ptr)&item.owner, &i))
			//if (i > 0) {// 2 for each
			if (i > 1) {// 2 for each
				//if (err = model->LESetsList->SwapItems(i, i - 1))
				if ((err = model->LESetsList->SwapItems(i, i - 2)) || (err = model->LESetsList->SwapItems(i+1, i - 1)))
					{ TechError("TLEList::UpItem()", "model->LESetsList->SwapItems()", err); return err; }
				SelectListItem(item);
				UpdateListLength(true);
				InvalidateMapImage();
				InvalMapDrawingRect();
			}
	
	return 0;
}

OSErr TLEList::DownItem(ListItem item)
{
	long i;
	OSErr err = 0;
	
	if (item.index == I_LEFIRSTLINE)
		if (model->LESetsList->IsItemInList((Ptr)&item.owner, &i))
			//if (i < (model->LESetsList->GetItemCount() - 1)) {
			if (i < (model->LESetsList->GetItemCount() - 3)) {
				//if (err = model->LESetsList->SwapItems(i, i + 1))
				if ((err = model->LESetsList->SwapItems(i, i + 2)) || (err = model->LESetsList->SwapItems(i+1, i + 3)))
					{ TechError("TLEList::UpItem()", "model->LESetsList->SwapItems()", err); return err; }
				SelectListItem(item);
				UpdateListLength(true);
				InvalidateMapImage();
				InvalMapDrawingRect();
			}
	
	return 0;
}

OSErr TOLEList::SettingsItem(ListItem item)
{
	// JLM 9/17/99 with sohail
	// for LE sets from a file, we don't have the dialog implemented yet
	// code goes here
	
	if(this -> initialLEs) {
		// then this guy was loaded from a file
		if (item.index == I_LEWINDAGE)	// maybe have the windage dialog come up in this case
		{
			WindageRec windageData = this -> GetWindageInfo();
			//OSErr err = WindageSettingsDialog(&windageData, dialog);
			OSErr err = WindageSettingsDialog(&windageData, mapWindow);
			if (!err) 
			{
				TOLEList 	*uncertaintyLEList=0;
				this -> SetWindageInfo(windageData);
				uncertaintyLEList = (TOLEList*)model->GetMirroredLEList(this);
				if (uncertaintyLEList) uncertaintyLEList -> SetWindageInfo(windageData);
				//model->NewDirtNotification();
			}
		}
		else	// find some way to edit this
			printNote("The settings dialog for LE sets loaded from a file is unimplemented.");
		return TRUE;
	}
	
	//if (item.index == I_LEWINDAGE)	// maybe have the windage dialog come up in this case
	//if (item.index == I_LEDISPERSE)	// maybe have the dispersant dialog come up in this case
	(void)OLESettingsDialog( (TOLEList *)item.owner);
	return TRUE;
}

OSErr TOLEList::DeleteItem(ListItem item)
{
	if (item.index == I_LEFIRSTLINE)
		return model->DropLEList(this, false);
	
	return 0;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////

static WindageRec	gDialogWindage;

static PopInfoRec WindagePopTable[] = {
		{ WINDAGE_DLGID, nil, WINDAGE_PERSISTENCE, 0, pPERSISTENCE, 0, 1, FALSE, nil }
	};

OSErr WINDAGEInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	//RegisterPopTable (WindagePopTable, sizeof (WindagePopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (WINDAGE_DLGID, dialog);
	
	SetPopSelection (dialog, WINDAGE_PERSISTENCE, gDialogWindage.persistence == -1 ? 2 : 1);
	Float2EditText(dialog, WINDAGE_A, gDialogWindage.windageA * 100, 2);
	Float2EditText(dialog, WINDAGE_B, gDialogWindage.windageB * 100, 2);
	
	MySelectDialogItemText(dialog, WINDAGE_A, 0, 255);

	SetDialogItemHandle(dialog, WINDAGESETTINGSFRAME, (Handle)FrameEmbossed);

	return 0;
}


short WINDAGEClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{

	switch (itemNum) {
		case WINDAGE_CANCEL: return WINDAGE_CANCEL;

		case WINDAGE_OK:
			short	theType;
			double timeStep;
			theType = GetPopSelection (dialog, WINDAGE_PERSISTENCE);
			timeStep = model -> GetTimeStep() / 3600.;
			if (theType == 1)
				gDialogWindage.persistence = .25;	// standard persistence in hours
			else if (theType == 2)
				gDialogWindage.persistence = -1; 	// flag for infinite persistence
			gDialogWindage.windageA = EditText2Float(dialog,WINDAGE_A) / 100.;
			gDialogWindage.windageB = EditText2Float(dialog,WINDAGE_B) / 100.;
			return itemNum;
		case WINDAGE_PERSISTENCE:
			long	menuID_menuItem;
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
		case WINDAGE_A:
		case WINDAGE_B:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

	}

	return 0;
}

OSErr WindageSettingsDialog(WindageRec *windageData, WindowPtr parentWindow)
{
	short item;
	PopTableInfo saveTable = SavePopTable();
	short j, numItems = 0;
	PopInfoRec combinedDialogsPopTable[20];

	if(parentWindow == nil) parentWindow = mapWindow; // JLM 6/2/99, we need the parent on the IBM
	gDialogWindage = *windageData;

	// code to allow a dialog on top of another with pops
	for(j = 0; j < sizeof(WindagePopTable) / sizeof(PopInfoRec);j++)
		combinedDialogsPopTable[numItems++] = WindagePopTable[j];
	for(j= 0; j < saveTable.numPopUps ; j++)
		combinedDialogsPopTable[numItems++] = saveTable.popTable[j];
	
	RegisterPopTable(combinedDialogsPopTable,numItems);

	item = MyModalDialog(WINDAGE_DLGID, parentWindow, 0, WINDAGEInit, WINDAGEClick);
	RestorePopTableInfo(saveTable);
	*windageData = gDialogWindage;
	if(item == WINDAGE_CANCEL) return USERCANCEL; 
	model->NewDirtNotification();
	if(item == WINDAGE_OK) 
	{
		if(parentWindow == mapWindow) {
			model->NewDirtNotification(); // when a dialog is the parent, we rely on that  dialog's to notify about Dirt 
			// that way we don't get the map redrawing behind the parent dialog on the IBM
		}
		return 0; 
	}
	else return -1;
}

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////

static double	gDialogDensity;
static double	gDialogHalflife;

OSErr ChemParamsInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	Float2EditText(dialog, CHEMPARAMS_DENSITY, gDialogDensity, 2);
	Float2EditText(dialog, CHEMPARAMS_HALFLIFE, gDialogHalflife, 2);
	
	MySelectDialogItemText(dialog, CHEMPARAMS_DENSITY, 0, 255);

	return 0;
}


short ChemParamsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{

	switch (itemNum) {
		case CHEMPARAMS_CANCEL: return CHEMPARAMS_CANCEL;

		case CHEMPARAMS_OK:
			gDialogDensity = EditText2Float(dialog,CHEMPARAMS_DENSITY);
			gDialogHalflife = EditText2Float(dialog,CHEMPARAMS_HALFLIFE);
			return itemNum;
		case CHEMPARAMS_DENSITY:
		case CHEMPARAMS_HALFLIFE:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

	}

	return 0;
}

OSErr ChemParamsDialog(double *density, double *halflife, WindowPtr parentWindow)
{
	short item;

	if(parentWindow == nil) parentWindow = mapWindow; // JLM 6/2/99, we need the parent on the IBM
	gDialogDensity = *density;
	gDialogHalflife = *halflife;

	item = MyModalDialog(CHEM_PARAMS, parentWindow, 0, ChemParamsInit, ChemParamsClick);
	*density = gDialogDensity;
	*halflife = gDialogHalflife;
	if(item == CHEMPARAMS_CANCEL) return USERCANCEL; 
	model->NewDirtNotification();
	if(item == CHEMPARAMS_OK) 
	{
		if(parentWindow == mapWindow) {
			model->NewDirtNotification(); // when a dialog is the parent, we rely on that  dialog's to notify about Dirt 
			// that way we don't get the map redrawing behind the parent dialog on the IBM
		}
		return 0; 
	}
	else return -1;
}

///////////////////////////////////////////////////////////////////////////

// call this to init lat/long boxes
// base is item number of decimal lat text edit box
// btnOne is item number of 'degrees' radio button
void SwitchLLFormatHelper(DialogPtr dialog, short base, short btnOne, Boolean show)
{
	short f = settings.latLongFormat;
	Rect r, r2;
	
	SetButton(dialog, btnOne, f == DEGREES);
	SetButton(dialog, btnOne + 1, f == DEGMIN);
	SetButton(dialog, btnOne + 2, f == DMS);
	
	ShowHideDialogItem(dialog, base, f == DEGREES && show); // base = LATDEGREES
	ShowHideDialogItem(dialog, base + 10, f == DEGREES && show); // base + 10 = LONGDEGREES
	
	ShowHideDialogItem(dialog, base + 1, f != DEGREES && show); // base + 1 = LATDEGLABEL
	ShowHideDialogItem(dialog, base + 11, f != DEGREES && show); // base + 11 = LONGDEGLABEL
	
	ShowHideDialogItem(dialog, base + 2, f != DEGREES && show); // base + 2 = LATDEG
	ShowHideDialogItem(dialog, base + 12, f != DEGREES && show); // base + 12 = LONGDEG
	
	ShowHideDialogItem(dialog, base + 3, f != DEGREES && show); // base + 3 = LATMINLABEL
	ShowHideDialogItem(dialog, base + 13, f != DEGREES && show); // base + 13 = LONGMINLABEL
	
	ShowHideDialogItem(dialog, base + 4, f == DEGMIN && show); // base + 4 = LATMINUTES
	ShowHideDialogItem(dialog, base + 14, f == DEGMIN && show); // base + 14 = LONGMINUTES
	
	ShowHideDialogItem(dialog, base + 5, f == DMS && show); // base + 5 = LATMIN
	ShowHideDialogItem(dialog, base + 15, f == DMS && show); // base + 15 = LONGMIN
	
	ShowHideDialogItem(dialog, base + 6, f == DMS && show); // base + 6 = LATSECLABEL
	ShowHideDialogItem(dialog, base + 16, f == DMS && show); // base + 16 = LONGSECLABEL
	
	ShowHideDialogItem(dialog, base + 7, f == DMS && show); // base + 7 = LATSECONDS
	ShowHideDialogItem(dialog, base + 17, f == DMS && show); // base + 17 = LONGSECONDS
	
	ShowHideDialogItem(dialog, base + 8, show); 
	ShowHideDialogItem(dialog, base + 18, show); 

	if(show)
	{
		PopErase(dialog, base + 8); // base + 8 = NORTHSOUTH
		PopErase(dialog, base + 18); // base + 18 = EASTWEST
		switch (f) {
			case DEGREES: r2 = GetDialogItemBox(dialog, base); break; // base = LATDEGREES
			case DEGMIN: r2 = GetDialogItemBox(dialog, base + 4); break; // base + 4 = LATMINUTES
			case DMS: r2 = GetDialogItemBox(dialog, base + 7); break; // base + 7 = LATSEC
		}
		r = GetDialogItemBox(dialog, base + 8); // base + 8 = NORTHSOUTH
		MyOffsetRect(&r, (r2.right - r.left) + 20, 0);
		SetDialogItemBox(dialog, base + 8, r); // base + 8 = NORTHSOUTH
		r = GetDialogItemBox(dialog, base + 18); // base + 18 = EASTWEST
		MyOffsetRect(&r, (r2.right - r.left) + 20, 0);
		SetDialogItemBox(dialog, base + 18, r); // base + 18 = EASTWEST
	#ifdef MAC
		PopDraw(dialog, base + 8); // base + 8 = NORTHSOUTH
		PopDraw(dialog, base + 18); // base + 18 = EASTWEST
		// DrawUserItem(dialog, base + 8); // base + 8 = NORTHSOUTH
		// DrawUserItem(dialog, base + 18); // base + 18 = EASTWEST
	#endif
	}
}

void SwitchLLFormat(DialogPtr dialog, short base, short btnOne)
{
	SwitchLLFormatHelper(dialog, base, btnOne, TRUE);
}


void LL2EditTexts(DialogPtr dialog, short base, WorldPoint *p)
{
	long degLat, degLong, minLat, minLong;
	float minutesLat, minutesLong, secondsLat, secondsLong;
	char dirLat, dirLong;
	
	switch (settings.latLongFormat) {
		case DEGREES:
		{
			double dDegreesLat,dDegreesLong; // JLM 8/31/99
			WorldPointToDoubles(*p, &dDegreesLat, &dDegreesLong, &dirLat, &dirLong,6);
			Float2EditText(dialog, base, dDegreesLat, 6); // base = LATDEGREES
			Float2EditText(dialog, base + 10, dDegreesLong, 6); // base + 10 = LONGDEGREES
			break;
		}
		
		case DEGMIN:
			WorldPointToDegMin(*p, &degLat, &minutesLat, &degLong, &minutesLong,
							   &dirLat, &dirLong,2);
			// JLM 11/30/98
			//if (minutesLat >= 59.95) { minutesLat = 0; degLat++; }
			//if (minutesLong >= 59.95) { minutesLong = 0; degLong++; }
			Long2EditText(dialog, base + 2, degLat); // base + 2 = LATDEG
			Long2EditText(dialog, base + 12, degLong); // base + 12 = LONGDEG
			OriginalFloat2EditText(dialog, base + 4, minutesLat, 2); // base + 4 = LATMINUTES
			OriginalFloat2EditText(dialog, base + 14, minutesLong, 2); // base + 14 = LONGMINUTES
			break;
		
		case DMS:
			WorldPointToDMS(*p, &degLat, &minLat, &secondsLat,
								&degLong, &minLong, &secondsLong,
								&dirLat, &dirLong,1);
			// JLM 11/30/98
			//if (secondsLat >= 59.95) { secondsLat = 0; minLat++; }
			//if (secondsLong >= 59.95) { secondsLong = 0; minLong++; }
			//if (minLat >= 59.95) { minLat = 0; degLat++; }
			//if (minLong >= 59.95) { minLong = 0; degLong++; }
			Long2EditText(dialog, base + 2, degLat); // base + 2 = LATDEG
			Long2EditText(dialog, base + 12, degLong); // base + 12 = LONGDEG
			Long2EditText(dialog, base + 5, minLat); // base + 5 = LATMIN
			Long2EditText(dialog, base + 15, minLong); // base + 15 = LONGMIN
			OriginalFloat2EditText(dialog, base + 7, secondsLat, 1); // base + 7 = LATSECONDS
			OriginalFloat2EditText(dialog, base + 17, secondsLong, 1); // base + 17 = LONGSECONDS
			break;
	}
	SetPopSelection(dialog, base + 8, (dirLat == 'N') ? 1 : 2);
	PopDraw(dialog, base + 8); // base + 8 = NORTHSOUTH
	SetPopSelection(dialog, base + 18, (dirLong == 'E') ? 1 : 2);
	PopDraw(dialog, base + 18); // base + 18 = EASTWEST
	
	if (!DisabledStatTextItem(dialog, base))
		switch (settings.latLongFormat) {
			case DEGREES: MySelectDialogItemText(dialog, base, 0, 100); break; // base = LATDEGREES
			default: MySelectDialogItemText(dialog, base + 2, 0, 100); break; // base + 2 = LATDEG
		}
}

OSErr EditTexts2LL(DialogPtr dialog, short base, WorldPoint *p,Boolean enforceLimits)
{
	double degreesLat, degreesLong;
	double minutesLat=0.0, minutesLong = 0.0;
	double secondsLat = 0.0, secondsLong = 0.0;
	char dirLat, dirLong;
	OSErr err = 0;
	short degLatItem = -1,minLatItem = -1,secLatItem = -1;
	short degLongItem = -1,minLongItem = -1,secLongItem = -1;
	
	
	switch (settings.latLongFormat) {
		case DEGREES:
			degreesLat = EditText2Float(dialog, degLatItem = base); // base = LATDEGREES
			degreesLong = EditText2Float(dialog, degLongItem = base + 10); // base + 10 = LONGDEGREES
			break;
		case DEGMIN:
			degreesLat = EditText2Float(dialog, degLatItem = base + 2); // base + 2 = LATDEG
			minutesLat = EditText2Float(dialog, minLatItem = base + 4); // base + 4 = LATMINUTES
			degreesLong = EditText2Float(dialog, degLongItem = base + 12); // base + 12 = LONGDEG
			minutesLong = EditText2Float(dialog, minLongItem = base + 14); // base + 14 = LONGMINUTES
			break;
		case DMS:
			degreesLat = EditText2Float(dialog, degLatItem = base + 2); // base + 2 = LATDEG
			minutesLat = EditText2Float(dialog, minLatItem = base + 5); // base + 5 = LATMIN
			secondsLat = EditText2Float(dialog, secLatItem = base + 7); // base + 7 = LATSECONDS
			degreesLong = EditText2Float(dialog, degLongItem = base + 12); // base + 12 = LONGDEG
			minutesLong = EditText2Float(dialog, minLongItem = base + 15); // base + 15 = LONGMIN
			secondsLong = EditText2Float(dialog, secLongItem = base + 17); // base + 17 = LONGSECONDS
			break;
	}
	
	if(enforceLimits)
	{
		if(degreesLat >= 90)
		{
			MySelectDialogItemText(dialog,degLatItem,0,100);
			printError("The latitude degrees must be less than 90.");
			return -1;
		}
		if(minutesLat >= 60)
		{
			MySelectDialogItemText(dialog,minLatItem,0,100);
			printError("The latitude minutes must be less than 60.");
			return -1;
		}
		if(secondsLat >= 60)
		{
			MySelectDialogItemText(dialog,secLatItem,0,100);
			printError("The latitude seconds must be less than 60.");
			return -1;
		}
		/*if(degreesLong >= 180 && (GetPopSelection(dialog, base + 18) != 1))
		{
			MySelectDialogItemText(dialog,degLongItem,0,100);
			printError("The longitude degrees must be less than 180 W.");
			return -1;
		}*/
		if(degreesLong >= 360 /*&& (GetPopSelection(dialog, base + 18) == 1)*/)
		{
			MySelectDialogItemText(dialog,degLongItem,0,100);
			printError("The longitude degrees must be less than 360.");
			return -1;
		}
		if(minutesLong >= 60)
		{
			MySelectDialogItemText(dialog,minLongItem,0,100);
			printError("The longitude minutes must be less than 60.");
			return -1;
		}
		if(secondsLong >= 60)
		{
			MySelectDialogItemText(dialog,secLongItem,0,100);
			printError("The longitude seconds must be less than 60.");
			return -1;
		}
	}
	
	degreesLat += minutesLat/60 + secondsLat/3600;
	degreesLong += minutesLong/60 + secondsLong/3600;
	
	dirLat  = (GetPopSelection(dialog, base +  8) == 1) ? 'N' : 'S'; // base +  8 = NORTHSOUTH
	dirLong = (GetPopSelection(dialog, base + 18) == 1) ? 'E' : 'W'; // base + 18 = EASTWEST
	
	DoublesToWorldPoint(degreesLat, degreesLong, dirLat, dirLong, p);
	return err;
}

void StandardLLClick(DialogPtr dialog, short itemNum, short base, short btnOne, WorldPoint *p,
					 Boolean *changedCoords)
{
	WorldPoint tempP;
	long result;
	
	(*changedCoords) = FALSE;
	
	if (itemNum == 1)
		(void)EditTexts2LL(dialog, base, p,FALSE); // LATDEGREES
	
	if (itemNum == (base + 8) || itemNum == (base + 18)) // NORTHSOUTH or EASTWEST
		if (PopClick(dialog, itemNum, &result)) *changedCoords = TRUE;
	
	// LATDEGREES or LONGDEGREES or LATMINUTES or LONGMINUTES or LATSECONDS or LONGSECONDS
	if (itemNum == base || itemNum == (base + 10) ||
		itemNum == (base + 4) || itemNum == (base + 14) ||
		itemNum == (base + 7) || itemNum == (base + 17)) {
		CheckNumberTextItem(dialog, itemNum, TRUE);
		*changedCoords = TRUE;
	}
	
	// LATDEG or LONGDEG or LATMIN or LONGMIN
	if (itemNum == (base + 2) || itemNum == (base + 12) ||
		itemNum == (base + 5) || itemNum == (base + 15)) {
		CheckNumberTextItem(dialog, itemNum, FALSE);
		*changedCoords = TRUE;
	}
}

///////////////////////////////////////////////////////////////////////////

static PopInfoRec dlePopTable[] = {
		{ M13, nil, M13SLATDIR, 0, pNORTHSOUTH1, 0, 1, FALSE, nil },
		{ M13, nil, M13SLONGDIR, 0, pEASTWEST1, 0, 1, FALSE, nil },
		{ M13, nil, M13POLLUTANT, 0, pPOLLUTANTS, 0, 1, FALSE, nil },
		{ M13, nil, M13ELATDIR, 0, pNORTHSOUTH2, 0, 1, FALSE, nil },
		{ M13, nil, M13ELONGDIR, 0, pEASTWEST2, 0, 1, FALSE, nil },
		{ M13, nil, M13MASSUNITS, 0, pLEMASSUNITS, 0, 1, FALSE, nil },
		{ M13, nil, M13STARTMONTH, 0, pMONTHS, 0, 1, FALSE, nil },
		{ M13, nil, M13STARTYEAR, 0, pYEARS, 0, 1, FALSE, nil },
		{ M13, nil, M13ENDMONTH, 0, pMONTHS2, 0, 1, FALSE, nil },
		{ M13, nil, M13ENDYEAR, 0, pYEARS2, 0, 1, FALSE, nil },
		{ M13, nil, M13DISPERSANT, 0, pDISPERSANTMODE, 0, 1, FALSE, nil }
	};

static AdiosInfoRecH gAdiosDataH = 0;
static char gOilTypeName[256];
TOLEList* gTOLEListInDialog = 0;
LESetSummary sharedSet;
static DispersionRec sDispInfo;
static WindageRec sWindageInfo;
static AdiosInfoRecH sAdiosInfoH = 0;
static char sOilTypeName[256];
static double sDensity, sHalfLife;

void DisposeDialogDataH(void)
{
	if (gAdiosDataH) {
		if(gAdiosDataH != sAdiosInfoH)
		{	// only dispose of this if it is different
			DisposeHandle((Handle)gAdiosDataH); 
		}
		gAdiosDataH = nil;
	}
}

void EnableEndTime(DialogPtr dialog, Boolean bEnable)
{
	Boolean show  = bEnable; //JLM
	dlePopTable[8].bStatic = !bEnable;
	dlePopTable[9].bStatic = !bEnable;
	
	ShowHideDialogItem(dialog, M13ENDMONTH, show); 
	ShowHideDialogItem(dialog, M13ENDYEAR, show); 
	
	ShowHideDialogItem(dialog, M13ENDDAY, show);
	ShowHideDialogItem(dialog, M13ENDHOURS, show);
	ShowHideDialogItem(dialog, M13ENDMINUTES, show);

	ShowHideDialogItem(dialog, M13ENDTIMECOLON, show);
	ShowHideDialogItem(dialog, M13ENDTIMELABEL, show);

	if(show)
	{
		PopDraw(dialog, M13ENDMONTH);
		PopDraw(dialog, M13ENDYEAR);
	}
	
	EnableTextItem(dialog, M13ENDDAY, bEnable);
	EnableTextItem(dialog, M13ENDHOURS, bEnable);
	EnableTextItem(dialog, M13ENDMINUTES, bEnable);
}

void EnableEndPosition(DialogPtr dialog, Boolean bEnable)
{
	Boolean show  = bEnable; //JLM
	dlePopTable[3].bStatic = !bEnable;
	dlePopTable[4].bStatic = !bEnable;

	ShowHideDialogItem(dialog, M13ELATDIR, show); 
	ShowHideDialogItem(dialog, M13ELONGDIR, show); 
	
	ShowHideDialogItem(dialog, M13ELATLABEL, show); 
	ShowHideDialogItem(dialog, M13ELONGLABEL, show); 
	
	if(show)
	{
		PopDraw(dialog, M13ELATDIR);
		PopDraw(dialog, M13ELONGDIR);
	}
	
	EnableTextItem(dialog, M13ELATDEG, bEnable);
	EnableTextItem(dialog, M13ELATMINUTES, bEnable);
	EnableTextItem(dialog, M13ELATDEGREES, bEnable);
	EnableTextItem(dialog, M13ELATMIN, bEnable);
	EnableTextItem(dialog, M13ELATSECONDS, bEnable);
	
	EnableTextItem(dialog, M13ELONGDEGREES, bEnable);
	EnableTextItem(dialog, M13ELONGDEG, bEnable);
	EnableTextItem(dialog, M13ELONGMIN, bEnable);
	EnableTextItem(dialog, M13ELONGMINUTES, bEnable);
	EnableTextItem(dialog, M13ELONGSECONDS, bEnable);
}

void ShowHideOSSMLEDialogItems(DialogPtr dialog)
{
	Boolean bWantEndRelPos = GetButton (dialog, M13WANTENDPOSITION);
	Boolean show = bWantEndRelPos;

	SwitchLLFormatHelper(dialog, M13ELATDEGREES, M13DEGREES,show);
	EnableEndTime (dialog, GetButton (dialog, M13WANTENDTIME));
	EnableEndPosition (dialog, bWantEndRelPos);
	
	SwitchLLFormat(dialog, M13SLATDEGREES, M13DEGREES); // do it 2nd to set focus

}
			
void ShowHideOSSMLEDispersionItems(DialogPtr dialog)
{
	Boolean bWantBottomSpill = GetButton (dialog, M13BOTTOMSPILL);
	double z = EditText2Float (dialog, M13DEPTH);
	short theType = GetPopSelection (dialog, M13DISPERSANT);
	
	
	if (bWantBottomSpill)
	{
		ShowHideDialogItem(dialog, M13DISPERSANT, false);
		ShowHideDialogItem(dialog, M13DEPTHLABEL, false);
		ShowHideDialogItem(dialog, M13DEPTH, false);
		ShowHideDialogItem(dialog, M13DEPTHUNITS, false);
		ShowHideDialogItem(dialog, M13DENSITYUNITS, true);
		ShowHideDialogItem(dialog, M13DENSITYLABEL, true);
		ShowHideDialogItem(dialog, M13DENSITY, true);
	}
	else
	{
		ShowHideDialogItem(dialog, M13DISPERSANT, true);
		if (theType==1)
		{
			ShowHideDialogItem(dialog, M13DEPTHLABEL, true);
			ShowHideDialogItem(dialog, M13DEPTH, true);
			ShowHideDialogItem(dialog, M13DEPTHUNITS, true);
			if (z>0)
			{
				ShowHideDialogItem(dialog, M13DENSITYUNITS, true);
				ShowHideDialogItem(dialog, M13DENSITYLABEL, true);
				ShowHideDialogItem(dialog, M13DENSITY, true);
			}
			else
			{
				ShowHideDialogItem(dialog, M13DENSITYUNITS, false);
				ShowHideDialogItem(dialog, M13DENSITYLABEL, false);
				ShowHideDialogItem(dialog, M13DENSITY, false);
			}
		}
		else
		{
			ShowHideDialogItem(dialog, M13DEPTHLABEL, false);
			ShowHideDialogItem(dialog, M13DEPTH, false);
			ShowHideDialogItem(dialog, M13DEPTHUNITS, false);
			// no rise velocity if dispersing?
			ShowHideDialogItem(dialog, M13DENSITYUNITS, false);
			ShowHideDialogItem(dialog, M13DENSITYLABEL, false);
			ShowHideDialogItem(dialog, M13DENSITY, false);
		}
	}
	return;
}
			
short OSSMLEClick (DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	Boolean changed;
	long menuID_menuItem;
	WorldPoint p, p2;
	DateTimeRec time;
	OSErr err = 0;
	char errStr[256] = "";
	long amountMin = 0, amountMax;
	Boolean haveOpenWizFile = (model->fWizard && model->fWizard->HaveOpenWizardFile());
	
	LESetSummary tempSet;

	StandardLLClick(dialog, itemNum, M13SLATDEGREES, M13DEGREES, &p, &changed);
	StandardLLClick(dialog, itemNum, M13ELATDEGREES, M13DEGREES, &p, &changed);

	switch (itemNum) {
		case M13OK:
			tempSet = sharedSet;	// 6/22/00
			tempSet.numOfLEs = EditText2Float(dialog, M13NUMLES);
			tempSet.pollutantType = GetPopSelection(dialog, M13POLLUTANT);
			tempSet.totalMass = EditText2Float(dialog, M13MASS);
			tempSet.massUnits = GetPopSelection(dialog, M13MASSUNITS);
			
			tempSet.riseVelocity = EditText2Float(dialog, M13DENSITY);
			/////////////////////////////
			// enforce input limits on the amount 
			/////////////////////////////
			switch(tempSet.massUnits)
			{
				case GALLONS: amountMax = 20000000;break;
				case BARRELS: amountMax = 476000;break;
				case CUBICMETERS: amountMax = 76000;break;
				///
				case METRICTONS: amountMax = 70000;break;
				case SHORTTONS:  amountMax = 77000;break;
			 	case KILOGRAMS: amountMax = 70000000;break;
				default: amountMax =-1;break; // -1 means don't enforce a limit
			}
			if(tempSet.totalMass <= amountMin) strcpy(errStr,"The amount released must be greater than zero.");	
			else if(tempSet.totalMass > amountMax && amountMax > 0)
			{
				char unitsStr[64];
				GetLeUnitsStr(unitsStr,tempSet.massUnits);
				sprintf(errStr,"The amount released cannot exceed %ld %s.",amountMax,unitsStr);
			}
			
			if(errStr[0])
			{
				printError(errStr);
				MySelectDialogItemText(dialog, M13MASS,0,100);
				break;
			}

			if(tempSet.numOfLEs > 10000 && !gDispersedOilVersion) 
			{
				printError("The number of Splots cannot exceed 10000."); // CJ requested limit on number of LEs, 5/8/00
				break;
			}
			
			/////////////////////////////
			
			// JLM, 1/26/99
			// the user cannot set the depth field since it is not used
			// so for it to be zero here
			// code goes here, let user set depth 

			tempSet.z  = 0;
			//
			// JLM 1/26/99, the density will be set by the oil type
			if (sDispInfo.bDisperseOil || gAdiosDataH)	// 
				tempSet.density = 141.5/(sDispInfo.api+131.5);
			else	// code goes here, for chemical will want to input density?
				tempSet.density = GetPollutantDensity(tempSet.pollutantType);
			//
			if (tempSet.pollutantType == OIL_CONSERVATIVE)
				tempSet.ageInHrsWhenReleased = 0.;
			else
				tempSet.ageInHrsWhenReleased = EditText2Float(dialog, M13AGE);
			//
//			tempSet.mass = VolumeMassToKilograms(tempSet.totalMass,tempSet.representativeLE.density,tempSet.massUnits)/ (float) tempSet.numOfLEs;
			/////////////////////////////////////////////////

				
			// get start release time
			//if (GetButton (dialog, M13WANTENDTIME) && model->bHindcast)
				//tempSet.startRelTime = RetrievePopTime(dialog, M13ENDMONTH,&err);
			//else
				tempSet.startRelTime = RetrievePopTime(dialog, M13STARTMONTH,&err);
			if(err) break;
			
			// get end release time
			tempSet.bWantEndRelTime = GetButton (dialog, M13WANTENDTIME);
			//if (model->bHindcast)
				//tempSet.endRelTime = RetrievePopTime(dialog, M13STARTMONTH,&err);
			//else
				tempSet.endRelTime = RetrievePopTime(dialog, M13ENDMONTH,&err);
			if(err) break;

			// get start release position
			err = EditTexts2LL(dialog, M13SLATDEGREES, &tempSet.startRelPos,TRUE);
			if(err)break;

			// get end release position
			tempSet.bWantEndRelPosition = GetButton (dialog, M13WANTENDPOSITION);
			err = EditTexts2LL(dialog, M13ELATDEGREES, &tempSet.endRelPos,TRUE);
			if(err)break;
			
			/////////////////////////////
			
			// JLM, 1/26/99
			// the user cannot set the depth field since it is not used
			// so for it to be zero here
			// let user set depth for bottom releases 2/2/04

			// get depth	
			tempSet.z  = 0;
			{ PtCurMap *map = GetPtCurMap();
				//PtCurMap *map = Get3DMap();
				float arrowDepth;
				Boolean bottomSpill = GetButton (dialog, M13BOTTOMSPILL);
				if (map && model->ThereIsA3DMover(&arrowDepth)) 
				{
					if (!bottomSpill)
					{
						double spillZ = (double)EditText2Float(dialog,M13DEPTH), botZ = map->DepthAtPoint(tempSet.startRelPos) ;
						//float spillZ = EditText2Float(dialog,M13DEPTH), botZ = (float)map->DepthAtPoint(tempSet.startRelPos) ;
						if (spillZ > botZ)	
						{
							char errStr[64];
							sprintf(errStr,"The spill depth cannot be greater than total depth which is %g meters.",botZ);
							printError(errStr);
							break;
						}
						tempSet.z = spillZ;
					}
					else
					{	// code goes here, bottom z is losing sig digits and causing a bug...
						double botZ = map->DepthAtPoint(tempSet.startRelPos);
						//float botZ = (float)map->DepthAtPoint(tempSet.startRelPos);
						tempSet.z = botZ;
						DisposeDialogDataH();
						sDispInfo.bDisperseOil = 0;
						sDispInfo.lassoSelectedLEsToDisperse = false;
					}
				}
			}
			if (tempSet.z == 0 && tempSet.riseVelocity != 0)
			{ 
				printError("The spill must be subsurface to have rise velocity."); 
				MySelectDialogItemText(dialog, M13DENSITY, 0, 255);  
				break;
			}			//
			//if (sDispInfo.bDisperseOil) {tempSet.z = 0;	tempSet.riseVelocity = 0;}// if have switched to disperse clear out bottom spill value
			//// verify data before assigning
			//////////////////////////////////
			
			//if(tempSet.numOfLEs < 1)  { 
			//	printError("The number of LE's must be at least one."); 
			//	MySelectDialogItemText(dialog, M13NUMLES, 0, 255);  
			//	break;}
				
			if(tempSet.bWantEndRelTime && tempSet.startRelTime > tempSet.endRelTime /*&& !(model->bHindcast)*/) {
				printError("The release start time cannot be after the release end time."); 
				break;}
			
			/*if(tempSet.bWantEndRelTime && tempSet.startRelTime < tempSet.endRelTime && model->bHindcast) {
				printError("The release start time cannot be after the release end time."); 
				break;}*/
			
			if(!model->IsWaterPoint(tempSet.startRelPos) && !(gNoaaVersion && model->bHindcast)) {
				printError("The release start position must be in the water."); 
				break;}
				
			if(!model->IsAllowableSpillPoint(tempSet.startRelPos) && !(gNoaaVersion && model->bHindcast)){
				char msg[256];
				if(haveOpenWizFile)
					strcpy(msg,"This Location File has not been set up for spills in the area of your release start position."); 
					//printError("This Location File has not been set up for spills in the area of your release start position."); 
				else
					strcpy(msg,"This map has not been set up for spills in the area of your release start position.");
					//printError("This map has not been set up for spills in the area of your release start position."); 
				if (model->HaveAllowableSpillLayer(tempSet.startRelPos) && model->GetModelMode()==ADVANCEDMODE) strcat(msg," To access this area, make the Spillable Area inactive in the left hand list.");
				printError(msg);
				break;}
			
			if(tempSet.bWantEndRelPosition && !model->IsWaterPoint(tempSet.endRelPos) && !(gNoaaVersion && model->bHindcast)) {
				printError("The release end position must be in the water."); 
				break;}
			
			if(tempSet.bWantEndRelPosition && !model->IsAllowableSpillPoint(tempSet.endRelPos) && !(gNoaaVersion && model->bHindcast)){
				char msg[256];
				if(haveOpenWizFile)
					strcpy(msg,"This Location File has not been set up for spills in the area of your release end position."); 
					//printError("This Location File has not been set up for spills in the area of your release end position."); 
				else
					strcpy(msg,"This map has not been set up for spills in the area of your release end position.");
					//printError("This map has not been set up for spills in the area of your release end position."); 
				if (model->HaveAllowableSpillLayer(tempSet.endRelPos) && model->GetModelMode()==ADVANCEDMODE) strcat(msg," To access this area, make the Spillable Area inactive in the left hand list.");
				printError(msg);
				break;}
			
			////////////////////////
			///// end verification
			
			/////
			// final checks   ///{
			if(tempSet.numOfLEs != sharedSet.numOfLEs && sDispInfo.lassoSelectedLEsToDisperse) 
			{
				short buttonSelected;
				{
					// point out that this spill is not in the time interval being modeled
					buttonSelected  = MULTICHOICEALERT(1690,"If you change the number of LEs the lasso selected LEs will be unmarked.  Are you sure you want to continue?",TRUE);
					switch(buttonSelected){
						case 1:// continue
							sDispInfo.lassoSelectedLEsToDisperse = false;
							break;  
						case 3: // cancel
							return 0;// stay at this dialog
							break;
					}
				}
			}
			// code goes here need special checks if hindcasting, but what if set spill first...??
			if (model->bHindcast)
			{
				Seconds testTime;
				if (tempSet.bWantEndRelTime) testTime = tempSet.endRelTime;
				else testTime = tempSet.startRelTime;
				if(testTime != model -> GetEndTime())
				{
					short buttonSelected;
					//if(model -> ThereIsALaterSpill(tempSet.startRelTime,(TLEList*)gTOLEListInDialog))
					if(model -> ThereIsALaterSpill(testTime,(TLEList*)gTOLEListInDialog))
					{	// there is a later spill
						if(testTime > model -> GetEndTime())
						{
							// they are already in trouble, so they are on their own
						}
						else if (testTime < model -> GetStartTime())
						{
							// point out that this spill is not in the time interval being modeled
							if (tempSet.bWantEndRelTime)
								buttonSelected  = MULTICHOICEALERT(1690,"The Release End Time is outside of the time interval being modeled.  Are you sure you want to continue?",TRUE);
							else
								buttonSelected  = MULTICHOICEALERT(1690,"The Release Start Time is outside of the time interval being modeled.  Are you sure you want to continue?",TRUE);
							switch(buttonSelected){
								case 1:// continue
									break;  
								case 3: // cancel
									return 0;// stay at this dialog
									break;
							}
						}
					}
					else 
					{	// no other spill is later than this one
						//if (!(model->bHindcast) || (model->bHindcast && tempSet.startRelTime != model -> GetEndTime()))
						//if (!(model->bHindcast) || (model->bHindcast && tempSet.endRelTime != model -> GetEndTime()))
						//{
						char msg[256];
						if (tempSet.bWantEndRelTime) strcpy(msg,"Normal usage is to set the Model End Time to the Release End Time of the Spill. Change Model End Time to Release End Time?");
						else strcpy(msg,"Normal usage is to set the Model End Time to the Release Start Time of the Spill. Change Model End Time to Release Start Time?");
						
						if(msg[0]) {
							buttonSelected  = MULTICHOICEALERT(1682,msg,FALSE);
							switch(buttonSelected){
								case 1:// change
									//if (!bHindcast) model -> SetStartTime(earliestStartTime);
									//else model -> SetStartTime(latestStartTime - model->GetDuration());
										model -> SetStartTime(testTime - model->GetDuration());
									model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
									return -1;// prevent the run
									break;  
								case 3: // cancel
									return -1;// prevent the run
									break;
							}
								
							/*buttonSelected  = MULTICHOICEALERT(1681,"",TRUE);
							switch(buttonSelected){
								case 1:// change
									//if (model->bHindcast) model -> SetStartTime(tempSet.startRelTime - model->GetDuration());
									//if (model->bHindcast) model -> SetStartTime(tempSet.endRelTime - model->GetDuration());
									//else
										model -> SetStartTime(tempSet.endRelTime - model->GetDuration());
									model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
									break;  
								case 3: // don't change
									break;
								case 4: // cancel
									return 0;// stay at this dialog
									break;
							}*/
						}
					}
				}
			}
				
	
	
			else
			{
				if(tempSet.startRelTime != model -> GetStartTime())
				{
					short buttonSelected;
					if(model -> ThereIsAnEarlierSpill(tempSet.startRelTime,(TLEList*)gTOLEListInDialog))
					{	// there is an earlier spill
						if(tempSet.startRelTime < model -> GetStartTime())
						{
							// they are already in trouble, so they are on their own
						}
						else if (tempSet.startRelTime > model -> GetEndTime())
						{
							// point out that this spill is not in the time interval being modeled
							buttonSelected  = MULTICHOICEALERT(1690,"The Release Start Time is outside of the time interval being modeled.  Are you sure you want to continue?",TRUE);
							switch(buttonSelected){
								case 1:// continue
									break;  
								case 3: // cancel
									return 0;// stay at this dialog
									break;
							}
						}
					}
					else 
					{	// no other spill is earlier than this one
						//if (!(model->bHindcast) || (model->bHindcast && tempSet.startRelTime != model -> GetEndTime()))
						//if (!(model->bHindcast) || (model->bHindcast && tempSet.endRelTime != model -> GetEndTime()))
						//{
							buttonSelected  = MULTICHOICEALERT(1681,"",TRUE);
							switch(buttonSelected){
								case 1:// change
									//if (model->bHindcast) model -> SetStartTime(tempSet.startRelTime - model->GetDuration());
									//if (model->bHindcast) model -> SetStartTime(tempSet.endRelTime - model->GetDuration());
									//else
										model -> SetStartTime(tempSet.startRelTime);
									model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
									break;  
								case 3: // don't change
									break;
								case 4: // cancel
									return 0;// stay at this dialog
									break;
							}
						//}
					}
				}
			}
			
			mygetitext(dialog, M13SPILLNAME, tempSet.spillName, kMaxNameLen - 1);		// get the mover's nameStr
			///////////////////////////// 
			//check what to do about an end time that sticks out past the time being modeled

			//if(tempSet.endRelTime > model -> GetEndTime())
			//{
				//Either
				//printNote("The End Release Time is outside of the interval being modeled");
				//return 0; // stay at this dialog
				//Or
				//short buttonSelected  = MULTICHOICEALERT(1687,"",TRUE);
				//switch(buttonSelected){
					//case 1:// change
						//model -> SetDuration(tempSet.endRelTime-tempSet.startRelTime);
						//break;  
					//case 3: // don't change
						//break;
					//case 4: // cancel
						//return 0;// stay at this dialog
						//break;
				//}
			//}
			/////////////////////////////
			
			///} end of final checks
			
			/////////////////////////////////////////////////
			/////////////////////////////////////////////////

			if (sDispInfo.bDisperseOil) {tempSet.z = 0;	tempSet.riseVelocity = 0;}// if have switched to disperse clear out bottom spill value

			sharedSet = tempSet;
			if (/*gAdiosDataH && */(gAdiosDataH != sAdiosInfoH)) 
			{
				if (sAdiosInfoH)
				{
					DisposeHandle((Handle)sAdiosInfoH); 
				}
				sAdiosInfoH = gAdiosDataH;
				strcpy(sOilTypeName,gOilTypeName);
			}
			return M13OK;

		case M13CANCEL: 
		{
			if (gAdiosDataH && (gAdiosDataH != sAdiosInfoH))
			{
				DisposeHandle((Handle)gAdiosDataH); 
				gAdiosDataH = 0;
			}
			return M13CANCEL;
		}
//		case M13STARTMONTH:
		case M13STARTDAY:
//		case M13STARTYEAR:
		case M13STARTHOURS:
		case M13STARTMINUTES:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;

//		case M13ENDMONTH:
		case M13ENDDAY:
//		case M13ENDYEAR:
		case M13ENDHOURS:
		case M13ENDMINUTES:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;

		case M13DEGREES:
		case M13DEGMIN:
		case M13DMS:
			err = EditTexts2LL(dialog, M13SLATDEGREES, &p,TRUE);
			if(err) break;
			err = EditTexts2LL(dialog, M13ELATDEGREES, &p2,TRUE);
			if(err) break;
			if (itemNum == M13DEGREES) settings.latLongFormat = DEGREES;
			if (itemNum == M13DEGMIN) settings.latLongFormat = DEGMIN;
			if (itemNum == M13DMS) settings.latLongFormat = DMS;
			//SwitchLLFormat(dialog, M13ELATDEGREES, M13DEGREES);
			//SwitchLLFormat(dialog, M13SLATDEGREES, M13DEGREES); // do it 2nd to set focus
			ShowHideOSSMLEDialogItems(dialog);//JLM
			LL2EditTexts(dialog, M13ELATDEGREES, &p2);
			LL2EditTexts(dialog, M13SLATDEGREES, &p);
			break;

		case M13NUMLES:
		case M13MASS:
		//case M13DENSITY:
		case M13AGE:
			CheckNumberTextItem(dialog, itemNum, itemNum != M13NUMLES);
			break;
		case M13DEPTH:
			CheckNumberTextItem(dialog, itemNum, itemNum != M13NUMLES);
			ShowHideOSSMLEDispersionItems(dialog);//JLM
			break;

		case M13DENSITY:
			CheckNumberTextItemAllowingNegative(dialog, itemNum, true);
			break;
		case M13WANTENDTIME:
			ToggleButton(dialog, itemNum);
			//EnableEndTime (dialog, GetButton (dialog, itemNum));
			ShowHideOSSMLEDialogItems(dialog);//JLM
			break;

		case M13WANTENDPOSITION:
			ToggleButton(dialog, itemNum);
			//EnableEndPosition (dialog, GetButton (dialog, itemNum));
			ShowHideOSSMLEDialogItems(dialog);//JLM
			break;

		case M13BOTTOMSPILL:
			ToggleButton(dialog, itemNum);
			//if (GetButton (dialog, M13BOTTOMSPILL))
				//Float2EditText (dialog, M13DEPTH, 0, 2);
			//EnableEndPosition (dialog, GetButton (dialog, itemNum));
			ShowHideOSSMLEDispersionItems(dialog);//JLM
			break;

		case M13POLLUTANT:
			PopClick(dialog, itemNum, &menuID_menuItem);
			if (model->GetModelMode () < ADVANCEDMODE || GetPopSelection(dialog, itemNum)==OIL_CONSERVATIVE)
			{
				ShowHideDialogItem(dialog, M13AGEUNITS, false);
				ShowHideDialogItem(dialog, M13AGELABEL, false);
				ShowHideDialogItem(dialog, M13AGE,      false);
			}
			else
			{
				ShowHideDialogItem(dialog, M13AGEUNITS, true);
				ShowHideDialogItem(dialog, M13AGELABEL, true);
				ShowHideDialogItem(dialog, M13AGE,      true);
			}
			if (GetPopSelection(dialog, M13POLLUTANT) == CHEMICAL) ShowHideDialogItem(dialog, M13CHEMPARAMS, true);
			else ShowHideDialogItem(dialog, M13CHEMPARAMS, false);
			break;

		case M13MASSUNITS:
		//case M13POLLUTANT:
		case M13STARTMONTH:
		case M13STARTYEAR:
		case M13ENDMONTH:
		case M13ENDYEAR:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
		case M13DISPERSANT:
		{
			short theType;
			long menuID_menuItem;
			float arrowDepth;
			PopClick(dialog, itemNum, &menuID_menuItem);
			theType = GetPopSelection (dialog, M13DISPERSANT);
			if (theType==1)
			{
				// selected No Dispersants, leave current values as defaults 
				//if (gAdiosDataH == nil) break;
				//code goes here, could ask if really want to delete the natural dispersion data
				DisposeDialogDataH();
				sDispInfo.bDisperseOil = 0;
				sDispInfo.lassoSelectedLEsToDisperse = false;
				ShowHideOSSMLEDispersionItems(dialog);
				break;
			}
			else if (theType==2) //chemical dispersants
			{
				Boolean userCanceledOrErr;
				DispersionRec dispersantData = sDispInfo;
				userCanceledOrErr = DispersionDialog(&dispersantData,sharedSet.startRelTime,GetDialogWindow(dialog));
				// refresh pop-ups
				//RegisterPopTable (dlePopTable, sizeof (dlePopTable) / sizeof (PopInfoRec));
				if(userCanceledOrErr)
				{	// leave in previous state
					if (gAdiosDataH)
						SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 3);
					else
						SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 1);
					PopDraw(dialog, M13DISPERSANT);
				}
				else
				{
					sDispInfo = dispersantData;
					sDispInfo.bDisperseOil = 1;
					//sharedSet.numOfLEs = 10000;	// this causes problems with lasso saving LERecH
					//Float2EditText (dialog, M13NUMLES, 10000, 0);// may want to go back to default if turn off dispersion ?
				}
				//if (gAdiosDataH == nil) break;
				// could ask about turning off natural dispersion
				DisposeDialogDataH();
				ShowHideOSSMLEDispersionItems(dialog);
				break;
			}
			else if (theType == 3) // natural dispersants (info from Adios Budget Table)
			{
				char path[256];
				if (gAdiosDataH) // check that user really wants to delete old file
				{
					// point out there is already a file selected
					short buttonSelected  = MULTICHOICEALERT(1688,"There is already an active Adios Budget Table. Do you want to read in a different one?",TRUE);
					switch(buttonSelected){
						case 1:// continue
							break;  
						case 3: // cancel
						{
							sDispInfo.bDisperseOil = 0;
							sDispInfo.lassoSelectedLEsToDisperse = false;
							goto doneAdiosFile;
							break;
						}
					}
				}
				// should allow user to keep file 
				if (err = GetAdiosFileName(path)) break;
				if (IsAdiosBudgetTable(path))
				{
					char oilType[256];
					float apiNum=0;
					DisposeDialogDataH(); // will lose it if file is messed up
					err = ReadAdiosBudgetTable(path,&gAdiosDataH,oilType,&apiNum);
					// if successful should set dispersant info, else leave in previous state
					if(err)
					{	// leave in previous state
						if (gAdiosDataH)
							SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 3);
						else
							SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 1);
						PopDraw(dialog, M13DISPERSANT);
					}
					else
					{
						char infoStr[256];
						sDispInfo.bDisperseOil = 0;
						sDispInfo.lassoSelectedLEsToDisperse = false;
						//sharedSet.numOfLEs = 10000;	// this causes problems with lasso saving LERecH
						//Float2EditText (dialog, M13NUMLES, 10000, 0);// may want to go back to default if turn off dispersion ?
						strcpy(gOilTypeName,oilType);
						sDispInfo.api = apiNum;
						//sprintf(infoStr,"Read in adios budget table for %s with api %f",oilType,apiNum);
						//printNote(infoStr);
						AdiosTable(gAdiosDataH,gOilTypeName,apiNum);
					}
				}
				else
				{
					printError("Selected file is not an adios budget table");
					break;
				}
				ShowHideOSSMLEDispersionItems(dialog);
				break;
			}
			else if (theType==4) //chemical and natural dispersants
			{
				Boolean userCanceledOrErr;
				DispersionRec dispersantData = sDispInfo;
				if (sDispInfo.lassoSelectedLEsToDisperse)
				{
					short buttonSelected  = MULTICHOICEALERT(1688,"Do you want to unmark the lasso selected LEs?",TRUE);
					switch(buttonSelected){
						case 1:// continue
							userCanceledOrErr = DispersionDialog(&dispersantData,sharedSet.startRelTime,GetDialogWindow(dialog));
							dispersantData.lassoSelectedLEsToDisperse = false;
							break;  
						case 3: // cancel
						{
							userCanceledOrErr = 0;
							break;
						}
					}
				}
				else
					userCanceledOrErr = DispersionDialog(&dispersantData,sharedSet.startRelTime,GetDialogWindow(dialog));
				// refresh pop-ups
				//RegisterPopTable (dlePopTable, sizeof (dlePopTable) / sizeof (PopInfoRec));
				if(userCanceledOrErr)
				{	// leave in previous state
					goto doneAdiosFile;
				}
				else
				{
				
					char path[256];
					if (gAdiosDataH) 
					{
						// point out there is already a file selected, which will be lost
						short buttonSelected  = MULTICHOICEALERT(1688,"There is already an active Adios Budget Table. Do you want to read in a different one?",TRUE);
						switch(buttonSelected){
							case 1:// continue
								break;  
							case 3: // cancel
							{
								sDispInfo = dispersantData;
								sDispInfo.bDisperseOil = 1;
								goto doneAdiosFile;
								break;
							}
						}
					}
					if (err = GetAdiosFileName(path)) goto doneAdiosFile;
					if (IsAdiosBudgetTable(path))
					{
						char oilType[256];
						float apiNum=0;
						DisposeDialogDataH(); // will lose it if file is messed up
						err = ReadAdiosBudgetTable(path,&gAdiosDataH,oilType,&apiNum);
						// if successful should set dispersant info, else leave in previous state
						if(err)
						{	// leave in previous state
							if (gAdiosDataH)
								SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 3);
							else
								SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 1);
							PopDraw(dialog, M13DISPERSANT);
						}
						else
						{
							// set the chemical dispersion information
							sDispInfo = dispersantData;
							sDispInfo.bDisperseOil = 1;
							//sharedSet.numOfLEs = 10000;	// this causes problems with lasso saving LERecH
							//Float2EditText (dialog, M13NUMLES, 10000, 0);
							strcpy(gOilTypeName,oilType);// may want to go back to default if turn off dispersion ?
							sDispInfo.api = apiNum;
							AdiosTable(gAdiosDataH,gOilTypeName,apiNum);
						}
					}
					else
					{
						printError("Selected file is not an adios budget table");
						break;
					}

				}
				ShowHideOSSMLEDispersionItems(dialog);
				break;
			}
			else if (theType==5)
			{
				//printNote("Adios budget table with removal options (burning/skimming) not yet implemented");

				char path[256];
				if (gAdiosDataH) // check that user really wants to delete old file
				{
					// point out there is already a file selected
					short buttonSelected  = MULTICHOICEALERT(1688,"There is already an active Adios Budget Table. Do you want to read in a different one?",TRUE);
					switch(buttonSelected){
						case 1:// continue
							break;  
						case 3: // cancel
						{
							sDispInfo.bDisperseOil = 0;
							sDispInfo.lassoSelectedLEsToDisperse = false;
							goto doneAdiosFile;
							break;
						}
					}
				}
				// should allow user to keep file 
				if (err = GetAdiosFileName(path)) break;
				if (IsAdiosBudgetTable(path))
				{
					char oilType[256];
					float apiNum=0;
					DisposeDialogDataH(); // will lose it if file is messed up
					err = ReadAdiosBudgetTableWithRemovalOptions(path,&gAdiosDataH,oilType,&apiNum);
					//err = ReadAdiosBudgetTable(path,&gAdiosDataH,oilType,&apiNum);
					// if successful should set dispersant info, else leave in previous state
					if(err)
					{	// leave in previous state
						if (gAdiosDataH)
							SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 3);
						else
							SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 1);
						PopDraw(dialog, M13DISPERSANT);
					}
					else
					{
						char infoStr[256];
						sDispInfo.bDisperseOil = 0;
						sDispInfo.lassoSelectedLEsToDisperse = false;
						//sharedSet.numOfLEs = 10000;	// this causes problems with lasso saving LERecH
						//Float2EditText (dialog, M13NUMLES, 10000, 0);// may want to go back to default if turn off dispersion ?
						strcpy(gOilTypeName,oilType);
						sDispInfo.api = apiNum;
						//sprintf(infoStr,"Read in adios budget table for %s with api %f",oilType,apiNum);
						//printNote(infoStr);
						AdiosTable(gAdiosDataH,gOilTypeName,apiNum);
					}
				}
				else
				{
					printError("Selected file is not an adios budget table");
					break;
				}
				ShowHideOSSMLEDispersionItems(dialog);
				break;
			
			}
doneAdiosFile:
			if (gAdiosDataH)
				SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 3);
			else
				SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 1);
			PopDraw(dialog, M13DISPERSANT);
			ShowHideOSSMLEDispersionItems(dialog);
			break;
		}
		case M13WINDAGE:
		{
			WindageRec windageData = sWindageInfo;
			OSErr err = WindageSettingsDialog(&windageData, GetDialogWindow(dialog));
			//RegisterPopTable (dlePopTable, sizeof (dlePopTable) / sizeof (PopInfoRec));
			if (!err) sWindageInfo = windageData;
			break;
		}
		case M13CHEMPARAMS:
		{
			double density = sDensity, halfLife = sHalfLife;
			OSErr err = ChemParamsDialog(&density, &halfLife, GetDialogWindow(dialog));
			if (!err) {sDensity = density; sHalfLife = halfLife;}
			break;
		}
			
	}
	
	return 0;
}

OSErr OSSMLEInit (DialogPtr dialog, VOIDPTR data)
{
	WorldPoint 	p, p2;
	DateTimeRec	time;
	float arrowDepth;
	float z, z2;
	PtCurMap *map = GetPtCurMap();
	char blankStr[32];
	strcpy(blankStr,"");
	
	gAdiosDataH = sAdiosInfoH;
	
	SetDialogItemHandle(dialog, M13HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M13FROST1, (Handle) FrameEmbossed);
	SetDialogItemHandle(dialog, M13FROST2, (Handle) FrameEmbossed);
	SetDialogItemHandle(dialog, M13FROST3, (Handle) FrameEmbossed);
	
	if(UseExtendedYears()) {
		dlePopTable[7].menuID = pYEARS_EXTENDED;
		dlePopTable[9].menuID = pYEARS2_EXTENDED;
	}
	else {
		dlePopTable[7].menuID = pYEARS;
		dlePopTable[9].menuID = pYEARS2;
	}

	if (gDispersedOilVersion)
	{
		dlePopTable[2].menuID = pPOLLUTANTS2;
	}
	else
	{
		dlePopTable[2].menuID = pPOLLUTANTS;
	}
	
	RegisterPopTable (dlePopTable, sizeof (dlePopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (M13, dialog);
	
	Float2EditText (dialog, M13NUMLES, sharedSet.numOfLEs, 0);
	SetPopSelection (dialog, M13POLLUTANT, sharedSet.pollutantType);
	
	// JLM 12/29/98
	// the user should have to enter the amount of oil spilled
	// there should be no default
	// we do this by making the initial amount negative and not displaying 
	// a negative amount here
	//if(sharedSet.totalMass < 0) mysetitext(dialog, M13MASS, ""); // BLANK
	if(sharedSet.totalMass < 0) mysetitext(dialog, M13MASS, blankStr); // BLANK
	else Float2EditText (dialog, M13MASS, sharedSet.totalMass, 0);
	
	SetPopSelection (dialog, M13MASSUNITS, sharedSet.massUnits);
	//Float2EditText (dialog, M13DENSITY, sharedSet.density, 2);
	Float2EditText (dialog, M13DENSITY, sharedSet.riseVelocity, 2);
	Float2EditText (dialog, M13AGE, sharedSet.ageInHrsWhenReleased, 2);
	Float2EditText (dialog, M13DEPTH, sharedSet.z, 2);

	Float2EditText (dialog, M13DENSITY, sharedSet.riseVelocity, 2);

//	DisplayTime (dialog, M13STARTMONTH, sharedSet.startRelTime);
	//if (sharedSet.bWantEndRelTime && model->bHindcast)		// maybe a better way to do this...
		//SecondsToDate (sharedSet.endRelTime, &time);
	//else
		SecondsToDate (sharedSet.startRelTime, &time);
	SetPopSelection (dialog, M13STARTMONTH, time.month);
	SetPopSelection (dialog, M13STARTYEAR,  time.year - (FirstYearInPopup()  - 1));
	Long2EditText (dialog, M13STARTDAY, time.day);
	Long2EditText (dialog, M13STARTHOURS, time.hour);
	Long2EditText (dialog, M13STARTMINUTES, time.minute);

	if(!data)
	{
		LL2EditTexts (dialog, M13SLATDEGREES, &sharedSet.startRelPos);
	}
	else LL2EditTexts (dialog, M13SLATDEGREES, (WorldPoint*)data);
	

	LL2EditTexts (dialog, M13ELATDEGREES, &sharedSet.endRelPos);
	SetButton (dialog, M13WANTENDPOSITION, sharedSet.bWantEndRelPosition);
	if (map) {z = map->DepthAtPoint(sharedSet.startRelPos); z2 = sharedSet.z;
	SetButton (dialog, M13BOTTOMSPILL, (sharedSet.z && (z2 == z)));}

	//if (model->bHindcast)	// maybe a better way to do this...
		//SecondsToDate (sharedSet.startRelTime, &time);
	//else
		SecondsToDate (sharedSet.endRelTime, &time);
	//SecondsToDate (sharedSet.endRelTime, &time);
	SetPopSelection (dialog, M13ENDMONTH, time.month);
	SetPopSelection (dialog, M13ENDYEAR,  time.year - (FirstYearInPopup()  - 1));
	Long2EditText (dialog, M13ENDDAY, time.day);
	Long2EditText (dialog, M13ENDHOURS, time.hour);
	Long2EditText (dialog, M13ENDMINUTES, time.minute);
	SetButton (dialog, M13WANTENDTIME, sharedSet.bWantEndRelTime);

	mysetitext(dialog, M13SPILLNAME, sharedSet.spillName);

	// code goes here, use this for riseVelocity for now
	if (!model->ThereIsA3DMover(&arrowDepth) || sharedSet.z == 0)
	{
		ShowHideDialogItem(dialog, M13DENSITYUNITS, false);
		ShowHideDialogItem(dialog, M13DENSITYLABEL, false);
		ShowHideDialogItem(dialog, M13DENSITY,      false);
	}

	if (model->GetModelMode () < ADVANCEDMODE || sharedSet.pollutantType==OIL_CONSERVATIVE)
	{
		ShowHideDialogItem(dialog, M13AGEUNITS, false);
		ShowHideDialogItem(dialog, M13AGELABEL, false);
		ShowHideDialogItem(dialog, M13AGE,      false);
	}

	if (model -> GetModelMode () < ADVANCEDMODE)
	{
		ShowHideDialogItem(dialog, M13LELABEL, false);
		ShowHideDialogItem(dialog, M13NUMLES,  false);
	}

	// set the hiliting of the end-time and end-position dialogs
	// according to the corresponding checkboxes
	EnableEndTime (dialog, GetButton (dialog, M13WANTENDTIME));
	EnableEndPosition (dialog, GetButton (dialog, M13WANTENDPOSITION));

	//SwitchLLFormat(dialog, M13ELATDEGREES, M13DEGREES);
	//SwitchLLFormat(dialog, M13SLATDEGREES, M13DEGREES);
	ShowHideOSSMLEDialogItems(dialog);//JLM

	if (sAdiosInfoH)
		SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 3);
	else
		SetPopSelection(dialog, M13DISPERSANT, sDispInfo.bDisperseOil + 1);
	ShowHideOSSMLEDispersionItems(dialog);//JLM
	ShowHideDialogItem(dialog, M13DISPERSANT, model->ThereIsA3DMover(&arrowDepth) && !sharedSet.z); // if there is a ptcurmap, should be 3D...
	ShowHideDialogItem(dialog, M13BOTTOMSPILL, model->ThereIsA3DMover(&arrowDepth)); // if there is a ptcurmap, should be 3D...
	ShowHideDialogItem(dialog, M13WINDAGE, model->GetModelMode() == ADVANCEDMODE); // if there is a ptcurmap, should be 3D...
	if (GetPopSelection(dialog, M13POLLUTANT) == CHEMICAL) ShowHideDialogItem(dialog, M13CHEMPARAMS, true);
	else ShowHideDialogItem(dialog, M13CHEMPARAMS, false);


	if (!model->ThereIsA3DMover(&arrowDepth))
	{
		ShowHideDialogItem(dialog, M13DEPTHUNITS, false);
		ShowHideDialogItem(dialog, M13DEPTHLABEL, false);
		ShowHideDialogItem(dialog, M13DEPTH,      false);
	}	
	MySelectDialogItemText(dialog, M13MASS, 0, 255);
	
	return 0;
}

double GetChemicalHalfLife()
{
	TWeatherer	*thisWeatherer;
	OilComponent	component;
	CMyList	*weatherList = model->GetWeatherList();
	OilType pollutantType = CHEMICAL;
	
	weatherList->GetListItem((Ptr)&thisWeatherer, 0);	// assume there is only one
	((TOSSMWeatherer*)thisWeatherer)->componentsList -> GetListItem ((Ptr) &component, pollutantType - 1);
	
	return component.halfLife[0];
}

void SetChemicalHalfLife(double halfLife)
{
	TWeatherer	*thisWeatherer;
	OilComponent	component;
	CMyList	*weatherList = model->GetWeatherList();
	OilType pollutantType = CHEMICAL;
	
	weatherList->GetListItem((Ptr)&thisWeatherer, 0);	// assume there is only one
	((TOSSMWeatherer*)thisWeatherer)->componentsList -> GetListItem ((Ptr) &component, pollutantType - 1);
	
	component.halfLife[0] = halfLife;
	((TOSSMWeatherer*)thisWeatherer)->componentsList -> SetListItem ((Ptr) &component, pollutantType - 1);
}

OSErr OLESettingsDialog (TOLEList 	*thisLEList)
{
	short 		dialogItem;
	TOLEList 	*uncertaintyLEList=0;
	Boolean weCreatedThisLEList = false;
	Boolean weCreatedUncertaintyLEList = false;
	Boolean deleteLERecH = true;
	OSErr err = 0;
	
	if (!thisLEList) return -1; //JLM, this code is only called when the LE already exists
										// otherwise we would need code to add them to the list
	
	gTOLEListInDialog = thisLEList;
	sharedSet = thisLEList -> fSetSummary;
	sDispInfo = thisLEList -> GetDispersionInfo();
	sWindageInfo = thisLEList -> GetWindageInfo();
	sAdiosInfoH = thisLEList -> GetAdiosInfo();
	strcpy(sOilTypeName,thisLEList -> fOilTypeName);
	sDensity = sharedSet.density;
	sHalfLife = GetChemicalHalfLife();
	
	dialogItem = MyModalDialog(M13, mapWindow, nil, OSSMLEInit, OSSMLEClick);
	if (dialogItem == M13CANCEL)
		return M13CANCEL;
	else if (dialogItem == M13OK)
	{
		
		if (!thisLEList)
		{
			thisLEList = new TOLEList ();
			weCreatedThisLEList = true;
			if (!thisLEList)
				{ TechError("OLESettingsDialog()", "new TOLEList()", 0); err = -1; goto done; }
		}
		thisLEList -> SetDispersionInfo(sDispInfo); 
		thisLEList -> SetWindageInfo(sWindageInfo);
		thisLEList -> SetAdiosInfo(sAdiosInfoH);
		strcpy(thisLEList -> fOilTypeName,sOilTypeName);
		uncertaintyLEList = (TOLEList*)model->GetMirroredLEList(thisLEList);
		if (uncertaintyLEList) uncertaintyLEList -> SetDispersionInfo(sDispInfo);
		if (uncertaintyLEList) uncertaintyLEList -> SetWindageInfo(sWindageInfo);
		if (uncertaintyLEList) uncertaintyLEList -> SetAdiosInfo(sAdiosInfoH);
		if (uncertaintyLEList) strcpy(uncertaintyLEList -> fOilTypeName,sOilTypeName);
	
		if (sDispInfo.lassoSelectedLEsToDisperse) 
			deleteLERecH = false;
		sharedSet.density = sDensity;
		SetChemicalHalfLife(sHalfLife);
		err = thisLEList -> Initialize (&sharedSet,deleteLERecH);
		if(err) goto done;
		
		if(uncertaintyLEList)		
			err = uncertaintyLEList -> Initialize (&sharedSet,deleteLERecH);
		if(err) goto done; 
		
		model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
		// since the model might be in negative time, in which case the LE's would be deleted and
		// readded from the files, which do not contain these changes
		model->NewDirtNotification();
	}
done:
	if(err)
	{
		if(weCreatedThisLEList && thisLEList) { delete thisLEList; thisLEList = 0;} 
		if(weCreatedUncertaintyLEList && uncertaintyLEList) { delete uncertaintyLEList; uncertaintyLEList = 0;} 
	}
	
	return err;
}

OSErr CreateLESet(WorldPoint *startPoint,WorldPoint *endPoint)
{
	short	item;
	TOLEList 	*list=nil,*uncertaintyLEList = nil;
	OSErr err = 0;
	
	long numMaps = 0;
	
	numMaps = model -> mapList -> GetItemCount();
	
	if (numMaps <= 0) {
		printError("You cannot create a spill because there is no map.");
		return -1;
	}

	
	memset(&sharedSet,0,sizeof(sharedSet));
	sharedSet.numOfLEs = 1000;
	sharedSet.pollutantType = OIL_CONSERVATIVE;
	if (model->GetModelMode() == ADVANCEDMODE) sharedSet.totalMass = 1;// give diagnostic mode users a default value 2/25/05
	else sharedSet.totalMass = -1;// JLM 12/29/98, force user to enter a value
	sharedSet.massUnits = BARRELS;// CJ asked that this be barrels
	sharedSet.startRelPos = startPoint == 0? WorldRectCenter (settings.currentView) : *startPoint;
	sharedSet.endRelPos = endPoint == 0 ? sharedSet.startRelPos : *endPoint;
	sharedSet.startRelTime = model -> GetStartTime ();
	sharedSet.endRelTime = sharedSet.startRelTime;
	sharedSet.bWantEndRelTime = false;
	sharedSet.bWantEndRelPosition = (endPoint != nil);
	/*if (sharedSet.bWantEndRelPosition)
		strcpy(sharedSet.spillName,"Line Source");
	else
	{
		if (startPoint != nil)
			strcpy(sharedSet.spillName,"Point Source");
		else
			strcpy(sharedSet.spillName,"Point/Line Source");
	}
	strcpy(sharedSet.spillName,"Point/Line Source");*/
	
	sharedSet.z = 0.;
	sharedSet.density = GetPollutantDensity(sharedSet.pollutantType);
	sharedSet.ageInHrsWhenReleased = 0.;

	sDispInfo.bDisperseOil = 0.;
	sDispInfo.amountToDisperse = 1.;	 
	//sDispInfo.mixedLayerDepth = 10.;
	//sDispInfo.breakingWaveHeight = 5.;
	sDispInfo.timeToDisperse = 0.;	// seconds after spill
	sDispInfo.duration = 3600.;	// seconds
	sDispInfo.api = 10.;	// matches default conservative oil density
	sDispInfo.lassoSelectedLEsToDisperse = false;
	sDispInfo.areaToDisperse = emptyWorldRect;	// GetMapBounds
	{	
		PtCurMap *map = GetPtCurMap();	// still could be 2D...
		if (map)
		{
			sDispInfo.areaToDisperse = map -> GetMapBounds();
		}
	}
	
	sWindageInfo.windageA = .01;
	sWindageInfo.windageB = .04;
	sWindageInfo.persistence = .25;	// in hours
	
	sOilTypeName[0] = 0;

	item = MyModalDialog(M13, mapWindow, 0, OSSMLEInit, OSSMLEClick);
	if (item == M13CANCEL) 
		return noErr;
	else if (item == M13OK)
	{
		model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
		// since the model might be in negative time, in which case the LE's would be deleted and
		// readded from the files, which do not contain these new LE sets
		////
		// note: since the model time is going to be reset when the user adds LEs
		// we need to do this before Initialize() is called 
		// since Initialize() uses GetModelTime() -- JLM 10/18/00

		list = new TOLEList();
		if (!list) { TechError("CreateLESet()", "new TOLEList()", 0); err= -1; goto done;}
			
		uncertaintyLEList = new TOLEList ();
		if (!uncertaintyLEList) { TechError("CreateLESet()", "new TOLEList()", 0); err = -1; goto done; }
			
		list -> SetWindageInfo(sWindageInfo); 
		list -> SetAdiosInfo(sAdiosInfoH); 
		list -> SetDispersionInfo(sDispInfo);
		strcpy(list -> fOilTypeName,sOilTypeName);

		err = list->Initialize(&sharedSet,true);
		if(err) goto done;

		uncertaintyLEList->fLeType = UNCERTAINTY_LE;
		uncertaintyLEList->fOwnersUniqueID = list->GetUniqueID();
		err = uncertaintyLEList->Initialize(&sharedSet,true);
		if(err) goto done;
		uncertaintyLEList -> SetWindageInfo(sWindageInfo); 
		uncertaintyLEList -> SetAdiosInfo(sAdiosInfoH); 
		uncertaintyLEList -> SetDispersionInfo(sDispInfo); 
		strcpy(uncertaintyLEList->fOilTypeName,sOilTypeName);

		model->NewDirtNotification();

		err = model->AddLEList(list, 0);
		if(err) goto done;
		err = model->AddLEList(uncertaintyLEList, 0);
		if(err) goto done;

		{	// force user through Map Dialog if they dispersed
			PtCurMap *map = GetPtCurMap();	
			if (map)
			{
				if (sDispInfo.bDisperseOil) // also check natural dispersion
					if (err = TMapSettingsDialog(map)) goto done;
			}
		}
		
	}
			
done:
	if(err)
	{
		if(list) {delete list; list = 0;}
		if(uncertaintyLEList) {delete uncertaintyLEList; uncertaintyLEList = 0;}
	}
	
	return noErr;
}

//////////////

OSErr  LoadOSSMFile(char* path, Seconds * fileTimeP)  // JLM 2/13/01 -- I moved this code out of AddLESetDialog so that I could use it for the TAP runs 
{
	char *className = NULL;
	return LoadOSSMFile(path,fileTimeP,className);
}

///////
OSErr  LoadOSSMFile(char* path, Seconds * fileTimeP, char *className) // JLm 7/13/10
{

	LESetSummary tempSet;
	OSErr err = 0;
	LERecH array = 0;
	long n;
	short item, massUnits;
	TOLEList *list = 0;
	TOLEList *uncertaintyList = 0;
	char shortFileName[64], tempStr[256];

	if(!FileExists(0,0,path)){
		printError("Invalid path in LoadOSSMFile");
		return -1;
	}
	
	if(!IsOSSMFile(path))
	{
		printError("This is not a GNOME Splot file");
		return -1;
	}
	
	
	if (err = ReadOSSMLEFile(path, &array, &n, &massUnits, fileTimeP)) return -1;

	// ?? create uncertainty set ?
	// code goes here, ask about importing an uncertainty set
	list = new TOLEList ();
	if (!list)
		{ DisposeHandle((Handle)array); TechError("AddLESetDialog()", "new TOLEList ()", 0); return -1; }

	uncertaintyList = new TOLEList ();
	if (!uncertaintyList)
		{ DisposeHandle((Handle)array); delete list; TechError("AddLESetDialog()", "new TOLEList ()", 0); return -1; }


	if(className) {
		list -> SetClassName(className); // name is optional but necessary if the command file/wizard is going to send it messages
		uncertaintyList -> SetClassName(className); // name is optional but necessary if the command file/wizard is going to send it messages
	}

	// copy the necessary information from the newly read-in LE array into the 
	// initialLEs array so we can keep track of the initial position, age, etc
	// of each of the LE's read into this set, STH
	
	// code goes here, it would be cool to convert into a splot LE's if same oil and release time and age
	
	// create an LE-distribution array containing fields from file for use when
	// reseting the model, STH
	list -> initialLEs = new CMyList (sizeof (InitialLEInfoRec));
	uncertaintyList -> initialLEs = new CMyList (sizeof (InitialLEInfoRec));
	if (list -> initialLEs && uncertaintyList -> initialLEs)
	{
		long				i;
		InitialLEInfoRec	thisSavedRec;
		LERec				thisLERec;
		
		list -> initialLEs -> IList ();
		uncertaintyList -> initialLEs -> IList ();
		
		memset(&thisSavedRec,0,sizeof(thisSavedRec));//1/23/03
		
		//for (i = 0; i <= n; ++i)
		for (i = 0; i < n; i++)	// bug causing TAP runs to crash 12/12/06
		{
			thisLERec = INDEXH(array, i);
			
			thisSavedRec.p = thisLERec.p;
			thisSavedRec.z = thisLERec.z;
			thisSavedRec.ageInHrsWhenReleased = thisLERec.ageInHrsWhenReleased;
			thisSavedRec.releaseTime = thisLERec.releaseTime;
			//1/23/03 pollutant type wasn't being set here
			thisSavedRec.pollutantType = thisLERec.pollutantType;
			//1/23/03 adding density to InitialLEInfoRec
			thisSavedRec.density = thisLERec.density;
			
			err = list -> initialLEs -> AppendItem ((Ptr) &thisSavedRec);
			if(err) goto done;
			err = uncertaintyList -> initialLEs -> AppendItem ((Ptr) &thisSavedRec);
			if(err) goto done;
		}
	}
	else {err = memFullErr; goto done;}

	// code goes here for not putting up the LE Set dialog for instantaneous read-in LE files.
	// Maybe this should be its own type of object because end-release time doesn't make sense.
	err = list->Initialize(n, array, massUnits, *fileTimeP);

	strcpy(tempStr,path);
	SplitPathFile(tempStr,shortFileName);

	strcpy(list -> fSetSummary.spillName, shortFileName);
	array = 0;// note: We set it to zero to forget we have it.  We are no longer responsible for it !! Initialize assigns array to it's internal field so we never want to dispose of array after call this function
	if (err) goto done; 
		
	uncertaintyList->fLeType = UNCERTAINTY_LE;
	uncertaintyList->fOwnersUniqueID = list->GetUniqueID();
	tempSet = list -> fSetSummary;
	if (err = uncertaintyList->Initialize(&tempSet,true))  goto done;

	//if (err = uncertaintyList->Initialize(n, array, massUnits, *fileTimeP))
		//{ DisposeHandle((Handle)array); delete uncertaintyList; return err; }

	err = model->AddLEList(list, 0);
	if(err) goto done;
	else list = 0;// note: We set it to zero to forget we have it.  We are no longer responsible for it !! 
	err = model->AddLEList(uncertaintyList, 0);
	if(err) goto done;
	else uncertaintyList = 0;// note: We set it to zero to forget we have it.  We are no longer responsible for it !! 

done:		
	if(err) {
		if(array) {DisposeHandle((Handle)array); ; array = 0;}
		if(list) {list->Dispose(); delete list; list = 0;}
		if(uncertaintyList) {uncertaintyList->Dispose(); delete uncertaintyList; uncertaintyList = 0;}
	}	
	
	return err;

}
///////

OSErr AddLESetDialog ()
{
	OSErr		err = noErr;
	short		method, type;

	long numMaps = 0;
	
	numMaps = model -> mapList -> GetItemCount();
	
	if (numMaps <= 0) {
		printError("You cannot add a spill because there is no map.");
		return -1;
	}

	method = MyModalDialog (M12, mapWindow, (Ptr) &type, M12Init, M12Click);

	if (method != M12CANCEL)
	{
		short		item;
		TOLEList 	*list = nil, *uncertaintyList = nil;
		
		switch (type)
		{
			case M21_POINT_LINE_SOURCE: // point/line source
				if (method == M12CREATE)
				{
					err = CreateLESet((WorldPoint*)nil,(WorldPoint*)nil);
					return err;
				}
				break;
			
			case M21_SPRAYED: // Sprayed Splots
				if (method == M12CREATE)
				{
					err = CreateSprayLESet();
					if(err) return err;
					SetTool(SPRAYCANTOOL);
					SprayCanSettingsDialog();// since this will be a new spill put up the spray can size dialog too
					return err;				 
				}
				break;

			case M21_LE_GRID: // Uniform LE Grid across water
				if (method == M12CREATE)
				{
					err = DoCJFile(nil);
					if(err) return err;
					return err;				 
				}
				break;
			
			case M21_OSSMFILE: // OSSM File Splots
				if (method == M12LOAD)
				{
					OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
					MySFReply reply;
					Point where = CenteredDialogUpLeft(M38);
					char path[256];
					Seconds fileTime;
				
				#if TARGET_API_MAC_CARBON
						mysfpgetfile(&where, "", -1, typeList,
								   (MyDlgHookUPP)0, &reply, M38, MakeModalFilterUPP(STDFilter));
						if (!reply.good) return USERCANCEL;
						strcpy(path, reply.fullPath);
				#else
					sfpgetfile(&where, "",
								(FileFilterUPP)0,
								-1, typeList,
								(DlgHookUPP)0,
								&reply, M38,
								(ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
					if (!reply.good) return USERCANCEL;
		
					my_p2cstr(reply.fName);
					#ifdef MAC
						GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
					#else
						strcpy(path, reply.fName);
					#endif
				#endif					
					if(IsCJFile(path))
					{
						err = DoCJFile(path);
						return err;
					}
	
					err = LoadOSSMFile(path,&fileTime); //// JLM 2/13/01 -- I moved code from here into AddLESetDialog so that I could use it for the TAP runs
					return err;
				}
				break;
			case M21_CDOG: 
				if (CDOGAvailable())
				{
					if (method == M12CREATE)
					{
						CDOGLEList *thisCDOGLEList = GetCDOGSpill();
						if (thisCDOGLEList)
						{
							printNote("There is already a CDOG spill");
							return -1;
						}
						err = CreateCDOGLESet();
						if(err) return err;
						return err;				 
					}
				}
				printNote("CDOG is not available");
				break;
			
		}
			

	}

	return 0;
}

///////////////////////////////////////////////////////////////////////////

static PopInfoRec sLeTypesPopTable[] = {
		{ M12, nil, M12TYPESITEM, 0, 0, 0, 1, FALSE, nil }
	};

OSErr M12Init(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)

	switch(model->GetModelMode())
	{
		default: case NOVICEMODE: sLeTypesPopTable[0].menuID = pNOVICELETYPES; break;
		//case INTERMEDIATEMODE: sLeTypesPopTable[0].menuID = pINTERMEDIATELETYPES; break;
		case ADVANCEDMODE: 
		{
			if (CDOGAvailable()) sLeTypesPopTable[0].menuID = pADVANCEDLETYPES; 
			else sLeTypesPopTable[0].menuID = pADVANCEDLETYPES2;  
			break;
		}
	}

	RegisterPopTable(sLeTypesPopTable, sizeof(sLeTypesPopTable) / sizeof(PopInfoRec));
	RegisterPopUpDialog(M12, dialog);
	
	SetPopSelection (dialog, M12TYPESITEM, M21_POINT_LINE_SOURCE);  
	MyEnableControl(dialog, M12LOAD,   false); // can't load M21_POINT_LINE_SOURCE

	return 0;
}

short M12Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short	*LEType = (short*) data;
	long	menuID_menuItem;
	long	selectedType;

	switch (itemNum) {
		case M12CANCEL: return M12CANCEL;

		case M12CREATE:
		case M12LOAD:
			*LEType = GetPopSelection (dialog, M12TYPESITEM);
			return itemNum;

		case M12TYPESITEM:
			PopClick(dialog, itemNum, &menuID_menuItem);
			selectedType = GetPopSelection (dialog, M21TYPESITEM);
			switch (selectedType)
			{
				case M21_POINT_LINE_SOURCE:
				case M21_SPRAYED:
				case M21_LE_GRID:
				case M21_CDOG:
					MyEnableControl(dialog, M12CREATE, true);
					MyEnableControl(dialog, M12LOAD,   false);
				break;			
				case M21_OSSMFILE: 
					MyEnableControl(dialog, M12CREATE, false);
					MyEnableControl(dialog, M12LOAD,   true);
				break;
				default:  // a new type was added to the menu, but not coded for above.
					// This is a programmer error, so  disable all buttons to get their attention
					MyEnableControl(dialog, M12CREATE, false);
					MyEnableControl(dialog, M12LOAD,   false);
				break;
			}
		break;
	}

	return 0;
}

///void DrawLE(Rect *r, WorldPoint p, short state,LETYPE leType)
// moved into the loop in TOLEList::Draw

static void DrawLEPositionMarker(Point *start,Point *end)
{
	/// Note: in Windows,  currentHDC can have  600 dpi or more
	/// so we need to use a rectangle in all cases
	long penWidth = 2;
	if (gDispersedOilVersion)
	{
#ifdef MAC
	PenSize(penWidth,penWidth);
#else
	PenStyle(BLACK,penWidth);
#endif
	}
	#ifdef IBM
		short xtraOffset = 1;
		if (gDispersedOilVersion) {xtraOffset = -1;} // a bigger spill marker
	#else
		short xtraOffset = 0;
	#endif
	short offset = round(2*PixelsPerPoint()); // 2 points each way
	if (gDispersedOilVersion) {offset = round(3 *PixelsPerPoint());} // a bigger spill marker
	// draw plus at start
	MyMoveTo(start->h-offset,start->v);MyLineTo(start->h+offset+xtraOffset,start->v);
	MyMoveTo(start->h,start->v-offset);MyLineTo(start->h,start->v+offset+xtraOffset);
	// draw plus at end
	MyMoveTo(end->h-offset,end->v);MyLineTo(end->h+offset+xtraOffset,end->v);
	MyMoveTo(end->h,end->v-offset);MyLineTo(end->h,end->v+offset+xtraOffset);
	// draw line from start to end
	MyMoveTo(start->h,start->v);MyLineTo(end->h,end->v);
#ifdef MAC
	PenSize(1,1);
#else
	PenStyle(BLACK,1);
#endif
}

void TOLEList::Draw(Rect r, WorldRect view)
{
	long i;
	LERec LE;
	Point start,end;
	Rect leRect, beachedRect, floatingRect;
	float floatingWidthInPoints = 2; // a point = 1/72 of an inch
	float beachedWidthInPoints = 3;
	short offset/*, x, y*/;
	Point pt;
	Boolean bUseLineTo = false, offQuickDrawPlane = false;
	float pixelsPerPoint = PixelsPerPoint();
	RGBColor *onLandColor,*inWaterColor;
	#ifdef IBM
		short xtraOffset = 1;
	#else
		short xtraOffset = 0;
	#endif
	PenNormal();
	
	// move down so initial position of non-spray spills will be drawn
	//if (model -> UserIsEditingSplots() && ! this -> UserIsEditingMeInMapDrawingRect())
		//return; // so we don't confuse the user about which splots are being edited
	
	end = start = WorldToScreenPoint(fSetSummary.startRelPos, settings.currentView, MapDrawingRect());
	if(fSetSummary.bWantEndRelPosition)
		end = WorldToScreenPoint(fSetSummary.endRelPos, settings.currentView, MapDrawingRect());
	
	{	
		// check if bDisperseOil
		PtCurMap *map = GetPtCurMap();	// still could be 2D...
		if (map && map->ThereIsADispersedSpill() && !(this->IAm(TYPE_CDOGLELIST)))	// what if spill starts underwater?
		{
			//map -> DrawContours(this, r, view);	// moved to DrawCombinedImage since need to total all dispersed spills
			// move following to ptcurmap
			if (this->bShowDispersantArea && this->bActive) 
			{
				LongRect	mapLongRect;
				Rect		m;
				Boolean  onQuickDrawPlane;
				WorldRect ourBounds = this->fDispersantData.areaToDisperse; 
				
				mapLongRect.left = SameDifferenceX(ourBounds.loLong);
				mapLongRect.top = (r.bottom + r.top) - SameDifferenceY(ourBounds.hiLat);
				mapLongRect.right = SameDifferenceX(ourBounds.hiLong);
				mapLongRect.bottom = (r.bottom + r.top) - SameDifferenceY(ourBounds.loLat);
				
				onQuickDrawPlane = IntersectToQuickDrawPlane(mapLongRect,&m);
	
				MyFrameRect(&m);
			}
			if (binitialLEsVisible && !initialLEs && !(this->fLeType == UNCERTAINTY_LE) && !(this->IAm(TYPE_SPRAYLELIST))) // don't draw initial position for uncertainty LEs
				DrawLEPositionMarker(&start,&end);// draw position marker if not from a file
			return;
		}
	}
	
	/// figure out the rectangle to draw when beached,inWater,etc.
	/// Note: in Windows,  currentHDC can have  600 dpi or more
	/// so we need to use a rectangle in all cases
	
	offset = _max(1,(floatingWidthInPoints*pixelsPerPoint)/2);
	//bUseLineTo = (offset <= 1);
	MySetRect(&floatingRect,-offset,-offset,offset,offset);
	offset = _max(1,(beachedWidthInPoints*pixelsPerPoint)/2);
	MySetRect(&beachedRect,-offset,-offset,offset,offset);
	
	if (binitialLEsVisible && !initialLEs && !(this->fLeType == UNCERTAINTY_LE) && !(this->IAm(TYPE_SPRAYLELIST))) // don't draw initial position for uncertainty LEs
		DrawLEPositionMarker(&start,&end);// draw position marker if not from a file
	
	// move down so initial location for non-spray spills are marked
	if (model -> UserIsEditingSplots() && ! this -> UserIsEditingMeInMapDrawingRect())
		return; // so we don't confuse the user about which splots are being edited
	
	//if (! this->bActive)	// don't draw anything, except initial position marker if not active
		//return;

	// first draw the positions of the initial LEs (JLM 2/19/03)	

	if (binitialLEsVisible && initialLEs && !(this->fLeType == UNCERTAINTY_LE)) // don't draw initial position for uncertainty LEs
	{
		InitialLEInfoRec	thisLEInfoRec;
		Point pt2;
		RGBForeColor(&colors[BLUE]);	// may want a different color if not active
		/////////////////
		for (i = 0 ; i < numOfLEs ; i++) 
		{
			initialLEs -> GetListItem ((Ptr) &thisLEInfoRec, i);
			if (WPointInWRect(thisLEInfoRec.p.pLong, thisLEInfoRec.p.pLat, &view))
			{
				pt2 = GetQuickDrawPt(thisLEInfoRec.p.pLong,thisLEInfoRec.p.pLat,&r,&offQuickDrawPlane);
				
				MyMoveTo(pt2.h, pt2.v);
				#ifdef MAC
					Line(0, 0);
				#else
					Line(1, 1);
				#endif
			}
		}
	}
	
	if (! this->bActive)	// don't draw anything, except initial position marker if not active
		return;

	///// now draw the real LEs on the top

	for (i = 0 ; i < numOfLEs ; i++) {
		GetLE (i, &LE);
		if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
		/////////////////////////////////////////////////
		{
			if (!WPointInWRect(LE.p.pLong, LE.p.pLat, &view)) continue;
			
			switch(this->fLeType)
			{
				case UNCERTAINTY_LE:
					onLandColor  = &colors[RED];
					inWaterColor = &colors[RED];
					break;
				default:
					onLandColor  = &colors[BLACK];
					inWaterColor = &colors[BLACK];
					break;
			}
			
			pt = GetQuickDrawPt(LE.p.pLong,LE.p.pLat,&r,&offQuickDrawPlane);
					
			switch (LE.statusCode) {
				case OILSTAT_INWATER:
					if (LE.leCustomData==-1 && model->GetModelMode()==ADVANCEDMODE)
					{
						RGBForeColor(&colors[GREEN]);
						leRect = floatingRect;
						MyOffsetRect(&leRect,pt.h,pt.v);
						MyMoveTo(leRect.left,leRect.top);
						MyLineTo(leRect.right+xtraOffset,leRect.bottom+xtraOffset);
						MyMoveTo(leRect.left,leRect.bottom);
						MyLineTo(leRect.right+xtraOffset,leRect.top-xtraOffset);
						break;
					}
					RGBForeColor(inWaterColor);
					leRect = floatingRect;
					MyOffsetRect(&leRect,pt.h,pt.v);
					if(bUseLineTo)
					{
						MyMoveTo(pt.h, pt.v);
						Line(0, 0);
					}
					else 
					{
						//PaintOval(&leRect);
						PaintRect(&leRect);
					}
				break;
				case OILSTAT_ONLAND:
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
	RGBForeColor(&colors[BLACK]);
}

/**************************************************************************************************/
WindageRec TOLEList::GetWindageInfo ()
{
	WindageRec	info;

	memset(&info,0,sizeof(info));
	info.windageA = this -> fWindageData.windageA;
	info.windageB = this -> fWindageData.windageB;
	info.persistence = this -> fWindageData.persistence;

	return info;
}

void TOLEList::SetWindageInfo (WindageRec info)
{
	this -> fWindageData.windageA = info.windageA;
	this -> fWindageData.windageB = info.windageB;
	this -> fWindageData.persistence = info.persistence;

	return;
}

/**************************************************************************************************/
DispersionRec TOLEList::GetDispersionInfo ()
{
	DispersionRec	info;

	memset(&info,0,sizeof(info));
	info.bDisperseOil = this -> fDispersantData.bDisperseOil;
	info.timeToDisperse	= this -> fDispersantData.timeToDisperse;
	info.duration = this -> fDispersantData.duration;
	info.api = this -> fDispersantData.api;		
	info.areaToDisperse.hiLat	= this -> fDispersantData.areaToDisperse.hiLat;
	info.areaToDisperse.loLat	= this -> fDispersantData.areaToDisperse.loLat;
	info.areaToDisperse.hiLong	= this -> fDispersantData.areaToDisperse.hiLong;
	info.areaToDisperse.loLong	= this -> fDispersantData.areaToDisperse.loLong;
	info.amountToDisperse		= this -> fDispersantData.amountToDisperse;	
	info.lassoSelectedLEsToDisperse = this -> fDispersantData.lassoSelectedLEsToDisperse;

	return info;
}
/**************************************************************************************************/
void TOLEList::SetDispersionInfo (DispersionRec info)
{
	this -> fDispersantData.bDisperseOil = info.bDisperseOil;
	this -> fDispersantData.timeToDisperse = info.timeToDisperse;
	this -> fDispersantData.duration = info.duration;
	this -> fDispersantData.api = info.api;
	this -> fDispersantData.areaToDisperse.hiLat 	= info.areaToDisperse.hiLat;		
	this -> fDispersantData.areaToDisperse.loLat 	= info.areaToDisperse.loLat;			
	this -> fDispersantData.areaToDisperse.hiLong 	= info.areaToDisperse.hiLong;	
	this -> fDispersantData.areaToDisperse.loLong 	= info.areaToDisperse.loLong;	
	this -> fDispersantData.amountToDisperse 	= info.amountToDisperse;	
	this -> fDispersantData.lassoSelectedLEsToDisperse = info.lassoSelectedLEsToDisperse;

	return;
}
