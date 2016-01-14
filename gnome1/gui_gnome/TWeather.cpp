
#include "Cross.h"
#include "MyRandom.h"


static PopInfoRec WeatherTypesPopTable[] = {
		{ M23, nil, M23TYPESITEM, 0,pNOVICELETYPES, 0, 1, FALSE, nil }
	};

static ListItem sharedItem;

///////////////////////////////////////////////////////////////////////////

OSErr M24Init(DialogPtr dialog, VOIDPTR data)	  // weatherer name dialog
{
#pragma unused(data)

	char		weatherName [kMaxNameLen];
	TWeatherer	*thisWeatherer = (TWeatherer*) data;

	SetDialogItemHandle(dialog, M24HILITEDEFAULT, (Handle)FrameDefault);
	thisWeatherer -> GetClassName (weatherName);
	mysetitext(dialog, M24WEATHERNAME, weatherName);
	MySelectDialogItemText(dialog, M24WEATHERNAME, 0, 100);

	return 0;
}

OSErr M23Init(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)

	RegisterPopTable(WeatherTypesPopTable, sizeof(WeatherTypesPopTable) / sizeof(PopInfoRec));
	RegisterPopUpDialog(M23, dialog);
	SetPopSelection (dialog, M23TYPESITEM, 1); // OSSM type weatherer
	MyEnableControl(dialog, M23LOAD, false);		 // no load for weaterers yet

	return 0;
}

short M23Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short	*mapType = (short*) data;
	long	menuID_menuItem;

	switch (itemNum) {
		case M23CANCEL: return M23CANCEL;

		case M23CREATE:
		case M23LOAD:
			*mapType = GetPopSelection (dialog, M23TYPESITEM);
			return itemNum;

		case M23TYPESITEM:
			PopClick(dialog, itemNum, &menuID_menuItem);
		break;
	}

	return 0;
}

OSErr AddWeatherDialog()
{
	short	method, type;
	OSErr err = 0;

	method = MyModalDialog (M23, mapWindow, (Ptr) &type, M23Init, M23Click);
	if (method != M23CANCEL)
	{
		if (method == M23CREATE)
		{
			TWeatherer	*weatherer;
			
			weatherer = new TOSSMWeatherer ("Weathering");
			if (!weatherer)
				{ TechError("AddWeathererDialog()", "new TOSSMWeather()", 0); return -1; }

			if (err = weatherer->InitWeatherer())
			{
				delete weatherer;
				return err;
			}
			else
				err = WeathererNameDialog (weatherer);
		
			if (!err)
			{
				if (err = model -> AddWeatherer (weatherer, 0))
					{ weatherer -> Dispose (); delete weatherer; return -1; }
				else
					model->NewDirtNotification();
			}
		}
		else if (method == M23LOAD)
		{
			SysBeep (5);
		}
	}

	return 0;
}

short M24Click(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	TWeatherer	*thisWeatherer = (TWeatherer*) data;
	long		menuID_menuItem;
	char		weatherName [kMaxNameLen];

	switch (itemNum) {
		case M24CANCEL: return M24CANCEL;

		case M24OK:
			mygetitext(dialog, M24WEATHERNAME, weatherName, kMaxNameLen - 1);
			thisWeatherer -> SetClassName (weatherName);
			return itemNum;
	}

	return 0;
}

OSErr WeathererNameDialog (TWeatherer *theWeatherer)
{
	short	method;
	OSErr	err = noErr;

	method = MyModalDialog (M24, mapWindow, (Ptr) theWeatherer, M24Init, M24Click);
	if (method != M24CANCEL) model->NewDirtNotification();
	if (method == M24CANCEL)
		err = -1;

	return err;
}


OilComponent  GetOilComponents(short pollutantType)
{
		short	i, j, oldCodeNumber;
		OilComponent	thisComponent;

		memset(&thisComponent,0,sizeof(thisComponent));
		GetPollutantName (pollutantType, thisComponent.pollutant);
		oldCodeNumber = NewToOldPollutantCode(pollutantType);
		switch (oldCodeNumber)
		{
			case 1:
				thisComponent.halfLife [0] = .12; thisComponent.halfLife [1] = 5.3; thisComponent.halfLife [2] = 1000000000;
				thisComponent.percent  [0] = .5;  thisComponent.percent  [1] = .5;  thisComponent.percent  [2] = 0.0;
				thisComponent.EPRAC = 18.55;
			break;

			case 2:
				thisComponent.halfLife [0] = 5.3; thisComponent.halfLife [1] = 14.4; thisComponent.halfLife [2] = 69.2;
				thisComponent.percent  [0] = .35;  thisComponent.percent  [1] = .5;  thisComponent.percent  [2] = .15;
				thisComponent.EPRAC = 50.4;
			break;

			case 3:
				thisComponent.halfLife [0] = 14.4; thisComponent.halfLife [1] = 48.6; thisComponent.halfLife [2] = 243;
				thisComponent.percent  [0] = .30;  thisComponent.percent  [1] = .45;  thisComponent.percent  [2] = .25;
				thisComponent.EPRAC = 170.1;
			break;

			case 4:
				thisComponent.halfLife [0] = 14.4; thisComponent.halfLife [1] = 48.6; thisComponent.halfLife [2] = 1000000000;
				thisComponent.percent  [0] = .24;  thisComponent.percent  [1] = .37;  thisComponent.percent  [2] = .39;
				thisComponent.EPRAC = 170.1;
			break;

			case 5:
				thisComponent.halfLife [0] = 14.4; thisComponent.halfLife [1] = 48.6; thisComponent.halfLife [2] = 1000000000;
				thisComponent.percent  [0] = .22;  thisComponent.percent  [1] = .26;  thisComponent.percent  [2] = .52;
				thisComponent.EPRAC = 170.1;
			break;

			case 6:
				thisComponent.halfLife [0] = 14.4; thisComponent.halfLife [1] = 48.6; thisComponent.halfLife [2] = 1000000000;
				thisComponent.percent  [0] = .20;  thisComponent.percent  [1] = .15;  thisComponent.percent  [2] = .65;
				thisComponent.EPRAC = 170.1;
			break;

			case 7:
				thisComponent.halfLife [0] = 1000000000; thisComponent.halfLife [1] = 1000000000; thisComponent.halfLife [2] = 1000000000;
				thisComponent.percent  [0] = 1.0;  thisComponent.percent  [1] = 0.0;  thisComponent.percent  [2] = 0.0;
				thisComponent.EPRAC = 1000000000;
			break;

			case 8:
				thisComponent.halfLife [0] = 1000000000; thisComponent.halfLife [1] = 1000000000; thisComponent.halfLife [2] = 1000000000;
				thisComponent.percent  [0] = 1.0;  thisComponent.percent  [1] = 0.0;  thisComponent.percent  [2] = 0.0;
				thisComponent.EPRAC = 1000000000;
			break;

			case 9:
				thisComponent.halfLife [0] = 1000000000; thisComponent.halfLife [1] = 1000000000; thisComponent.halfLife [2] = 1000000000;
				thisComponent.percent  [0] = 1.0;  thisComponent.percent  [1] = 0.0;  thisComponent.percent  [2] = 0.0;
				thisComponent.EPRAC = 1000000000;
			break;
	
			default:
			case 10: // code goes here, what is 10, old OSSM only had 9 types  ??
				thisComponent.halfLife [0] = 1000000000; thisComponent.halfLife [1] = 1000000000; thisComponent.halfLife [2] = 1000000000;
				thisComponent.percent  [0] = 1.0;  thisComponent.percent  [1] = 0.0;  thisComponent.percent  [2] = 0.0;
				thisComponent.EPRAC = 1000000000;
			break;
		}

		for (j = 0; j <= 2; ++j)
			thisComponent.XK [j] = 0.693147 / thisComponent.halfLife [j];

		thisComponent.bModified = false;
		
		return thisComponent;
}

double FractionOilLeftAtTime(short pollutantType,double timeInHours)
{ // returns a number from 0.0-1.0
	double percentOilLeftAtTime = 0.0;
	short i;
	OilComponent component = GetOilComponents(pollutantType);
	
	for(i = 0;i<3;i++)
	{
		if(component.percent[i] > 0.0)
		{
			percentOilLeftAtTime +=  (component.percent[i])*pow(0.5,timeInHours/(component.halfLife[i]));
		}
	}
	percentOilLeftAtTime = _max (0.0,percentOilLeftAtTime);
	percentOilLeftAtTime = _min (1.0,percentOilLeftAtTime);
	return percentOilLeftAtTime;
}




///////////////////////////////////////////////////////////////////////////
TWeatherer::TWeatherer(char *name)
{
	SetClassName (name);
	
	bActive = TRUE;
	bOpen = TRUE;
}
///////////////////////////////////////////////////////////////////////////
TOSSMWeatherer::TOSSMWeatherer (char *name):TWeatherer (name)
{
	return;
}
///////////////////////////////////////////////////////////////////////////
OSErr TOSSMWeatherer::InitWeatherer ()
{
	OSErr	err = noErr;
	
	componentsList = new CMyList (sizeof (OilComponent));
	if (componentsList == nil)
		err = memFullErr;

	if (!err)	
		err = componentsList -> IList ();
	
	if (!err)
	{
		short	i;
		OilComponent	thisComponent;

		// code goes here
		// JLM 3/1/99, the new codes are not sequential
		// this will be a problem once we try to use OIL_USER1 etc
		for (i = 1; i <= 10; ++i)
		{
			thisComponent = GetOilComponents(i);
			componentsList -> AppendItem ((Ptr) &thisComponent);
		}
	}

	return err;
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
TOSSMWeatherer::~TOSSMWeatherer ()
{
	Dispose ();

	return;
}
///////////////////////////////////////////////////////////////////////////
void TOSSMWeatherer::Dispose ()
{
	if (componentsList != nil)
	{
		delete (componentsList);
		componentsList = nil;
	}

	TWeatherer::Dispose ();

	return;
}
///////////////////////////////////////////////////////////////////////////

long TWeatherer::GetListLength()
{

	long 	i, n, count = 1;	// open/close & type/name display
							
	if (bOpen) {
		count += 1;				// active status	
	}

	return count;
}
///////////////////////////////////////////////////////////////////////////
ListItem TOSSMWeatherer::GetNthListItem(long n, short indent, short *style, char *text)
{
	long i;
	ListItem item = { this, 0, indent, 0 };

	if (n == 0) {
		item.index = I_WEATHERNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "OSSM Wx: \"%s\"", className);
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}

	n -= 1;
	item.indent++;

	if (n == 0) {
		item.index = I_WEATHERACTIVE;
		item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
		strcpy(text, "Active");

		return item;
	}

	item.owner = 0;

	return item;
}
///////////////////////////////////////////////////////////////////////////
Boolean TWeatherer::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_WEATHERNAME: bOpen = !bOpen; return TRUE;
			case I_WEATHERACTIVE: bActive = !bActive; 
				model->NewDirtNotification(); return TRUE;
		}
	
//	if (item.index == I_WEATHERNAME && doubleClick)
//		ChangeCurrentView(AddWRectBorders(bounds, 10), TRUE, TRUE);
	
	// do other click operations...
	
	return FALSE;
}
///////////////////////////////////////////////////////////////////////////

OSErr TWeatherer::AddItem(ListItem item)
{
	short type = 0, dItem =0;

	if (item.index == I_WEATHERING) {
//		dItem = MyModalDialog (M21, mapWindow, (Ptr) &type, M21Init, M21Click);
		if (dItem == M21LOAD)
		{
			switch (type)
			{
			}
		}
		else if (dItem == M21CREATE)
		{
			SysBeep (5);
		}
	}

	return 0;
}
///////////////////////////////////////////////////////////////////////////
OSErr TWeatherer::Write(BFPB *bfpb)
{
	char c;
	long version = 1;
	ClassID id = GetClassID ();
	OSErr err = 0;
	
	StartReadWriteSequence("TWeatherer::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	if (err = WriteMacValue(bfpb, className, kMaxNameLen)) return err;
	c = bOpen;
	if (err = WriteMacValue(bfpb, c)) return err;

	SetDirty(FALSE);

	return 0;
}
///////////////////////////////////////////////////////////////////////////
OSErr TWeatherer::Read(BFPB *bfpb)
{
	char c;
	long version;
	ClassID id;
	OSErr err = 0;
	
	StartReadWriteSequence("TWeatherer::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TWeatherer::Read()", "id == TYPE_WEATHERER", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version != 1) { printSaveFileVersionError(); return -1; }
	if (err = ReadMacValue(bfpb, className, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb, &c)) return err;
	bOpen = c;

	SetDirty(FALSE);

	return 0;
}
///////////////////////////////////////////////////////////////////////////
OSErr TOSSMWeatherer::Write(BFPB *bfpb)
{
	long			numComponents, i;
	OilComponent	thisComponent;
	OSErr err = 0;

	if (err = TWeatherer::Write(bfpb)) return err;

	numComponents = componentsList -> GetItemCount ();
	if (err = WriteMacValue(bfpb, numComponents)) return err;

	for (i = 1; i <= numComponents; ++i)	// no point in saving this since we re-initialize each time
	{
		//componentsList -> GetListItem ((Ptr) &thisComponent, i);
		componentsList -> GetListItem ((Ptr) &thisComponent, i-1);	// but if we're going to at least save the right stuff

		if (err = WriteMacValue(bfpb, (thisComponent.halfLife [0]))) return err;
		if (err = WriteMacValue(bfpb, (thisComponent.halfLife [1]))) return err;
		if (err = WriteMacValue(bfpb, (thisComponent.halfLife [2]))) return err;
	
		if (err = WriteMacValue(bfpb, (thisComponent.percent [0]))) return err;
		if (err = WriteMacValue(bfpb, (thisComponent.percent [1]))) return err;
		if (err = WriteMacValue(bfpb, (thisComponent.percent [2]))) return err;
	
		if (err = WriteMacValue(bfpb, (thisComponent.XK [0]))) return err;
		if (err = WriteMacValue(bfpb, (thisComponent.XK [1]))) return err;
		if (err = WriteMacValue(bfpb, (thisComponent.XK [2]))) return err;
	
		if (err = WriteMacValue(bfpb, (thisComponent.EPRAC))) return err;
	}

	return 0;
}
///////////////////////////////////////////////////////////////////////////
OSErr TOSSMWeatherer::Read(BFPB *bfpb)
{
	long			numComponents, i;
	OilComponent	thisComponent;
	OSErr err = 0;

	if (err = TWeatherer::Read(bfpb)) return err;

	if (err = ReadMacValue(bfpb, &numComponents)) return err;

	for (i = 1; i <= numComponents; ++i)
	{
		if (i==CHEMICAL) componentsList -> GetListItem ((Ptr) &thisComponent, i - 1);	
		if (err = ReadMacValue(bfpb, &(thisComponent.halfLife [0]))) return err;
		if (err = ReadMacValue(bfpb, &(thisComponent.halfLife [1]))) return err;
		if (err = ReadMacValue(bfpb, &(thisComponent.halfLife [2]))) return err;
	
		if (err = ReadMacValue(bfpb, &(thisComponent.percent [0]))) return err;
		if (err = ReadMacValue(bfpb, &(thisComponent.percent [1]))) return err;
		if (err = ReadMacValue(bfpb, &(thisComponent.percent [2]))) return err;
	
		if (err = ReadMacValue(bfpb, &(thisComponent.XK [0]))) return err;
		if (err = ReadMacValue(bfpb, &(thisComponent.XK [1]))) return err;
		if (err = ReadMacValue(bfpb, &(thisComponent.XK [2]))) return err;
	
		if (err = ReadMacValue(bfpb, &(thisComponent.EPRAC))) return err;

		// don't add on saved values since they won't be used, just update chemical half life
		//if (err = componentsList -> AppendItem ((Ptr) &thisComponent)) return err;	// this is junk since we re-initialize each time
		if (i==CHEMICAL) componentsList -> SetListItem ((Ptr) &thisComponent, i - 1);	//  chemical half life could be changed
	}

	return 0;
}
///////////////////////////////////////////////////////////////////////////
Boolean TOSSMWeatherer::FunctionEnabled(ListItem item, short buttonID)
{
	switch (item.index) {
		case I_WEATHERNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON: return TRUE;
			}
			break;
	}
	
	if (buttonID == SETTINGSBUTTON) return TRUE;
	
	return TWeatherer::FunctionEnabled(item, buttonID);
}
///////////////////////////////////////////////////////////////////////////
OSErr TOSSMWeatherer::DeleteItem(ListItem item)
{
	if (item.index == I_WEATHERNAME)
		return model->DropWeatherer((TWeatherer*) this);
	
	return 0;
}
///////////////////////////////////////////////////////////////////////////
OSErr TOSSMWeatherer::SettingsItem(ListItem item)
{
	return OWeatherSettingsDialog(item);
}
///////////////////////////////////////////////////////////////////////////
Boolean TOSSMWeatherer::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	Boolean bClickResult;

	if (!(bClickResult = TWeatherer::ListClick (item, inBullet, doubleClick)))
	{
		if (doubleClick && !inBullet)
		{
			OWeatherSettingsDialog(item);
			return true;
		}
	}

   return bClickResult;
}
///////////////////////////////////////////////////////////////////////////

void SetOWeatherValues (DialogPtr dialog, long item)
{
	TOSSMWeatherer *theWeatherer;
	OilComponent	theComponent;
	OSErr err = 0;

	theWeatherer = (TOSSMWeatherer*) sharedItem.owner;
	err = theWeatherer -> componentsList -> GetListItem ((Ptr) &theComponent, item);

	Float2EditText(dialog, M26HALFLIFE1, theComponent.halfLife [0], 2);
	Float2EditText(dialog, M26HALFLIFE2, theComponent.halfLife [1], 2);
	Float2EditText(dialog, M26HALFLIFE3, theComponent.halfLife [2], 2);

	Float2EditText(dialog, M26PERCENT1, theComponent.percent [0], 2);
	Float2EditText(dialog, M26PERCENT2, theComponent.percent [1], 2);
	Float2EditText(dialog, M26PERCENT3, theComponent.percent [2], 2);

	MySelectDialogItemText(dialog, M26HALFLIFE1, 0, 255);

	return;
}

void GetOWeatherValues (DialogPtr dialog, long item)
{
	TOSSMWeatherer *theWeatherer;
	OilComponent	theComponent;
	char			wName [kMaxNameLen];
	OSErr err = 0;
	long j;

	theWeatherer = (TOSSMWeatherer*) sharedItem.owner;
	err = theWeatherer -> componentsList -> GetListItem ((Ptr) &theComponent, item);

	mygetitext(dialog, M26WNAME, wName, kMaxNameLen - 1);
	theWeatherer -> SetClassName (wName);

	theComponent.halfLife [0] = EditText2Float(dialog, M26HALFLIFE1);
	theComponent.halfLife [1] = EditText2Float(dialog, M26HALFLIFE2);
	theComponent.halfLife [2] = EditText2Float(dialog, M26HALFLIFE3);

	theComponent.percent [0] = EditText2Float(dialog, M26PERCENT1);
	theComponent.percent [1] = EditText2Float(dialog, M26PERCENT2);
	theComponent.percent [2] = EditText2Float(dialog, M26PERCENT3);

	for (j = 0; j <= 2; ++j)
		theComponent.XK [j] = 0.693147 / theComponent.halfLife [j];

	theComponent.bModified = true;

	err = theWeatherer -> componentsList -> SetListItem ((Ptr) &theComponent, item);
	model->NewDirtNotification();

	return;
}

void OWeatherInit(DialogPtr dialog, VLISTPTR L)
{
	TOSSMWeatherer *theWeatherer;
	char			wName [kMaxNameLen];

	theWeatherer = (TOSSMWeatherer*) sharedItem.owner;

	SetDialogItemHandle(dialog, M26FROST1, (Handle)FrameEmbossed);
	SetDialogItemHandle(dialog, M26FROST2, (Handle)FrameEmbossed);

	SetOWeatherValues (dialog, 1);

	theWeatherer -> GetClassName (wName);
	mysetitext(dialog, M26WNAME, wName);
	MySelectDialogItemText (dialog, M26WNAME, 0, 255);

	return;
}

void DrawOWeatherItem(DialogPtr dialog, Rect *r, long item)
{
	char 			s[255], pollutant [kMaxNameLen];
	Point			p;
	TOSSMWeatherer *theWeatherer;
	OilComponent	theComponent;
	OSErr			err = noErr;

	theWeatherer = (TOSSMWeatherer*) sharedItem.owner;
	err = theWeatherer -> componentsList -> GetListItem ((Ptr) &theComponent, item);
	if (!err)
	{
		GetPen(&p);
		#ifdef MAC
			TextFontSize(kFontIDGeneva,LISTTEXTSIZE);
		#else
			TextFontSize(kFontIDTimes,9);
		#endif
		
		MyMoveTo(RectLeft(GetDialogItemBox(dialog, M26POLNAME)), p.v);
		drawstring(theComponent.pollutant);

		theComponent.bModified ? strcpy (s, "Modified") : strcpy (s, "Standard");
		MyMoveTo(RectLeft(GetDialogItemBox(dialog, M26STANDMOD)) + 3, p.v);
		drawstring(s);

		TextFontSize(0,12);
	}

	return;
}

Boolean OWeatherClick(DialogPtr dialog, VLISTPTR L, short dialogItem, long *listItem,
					  Boolean doubleClick)
{
	long	currSelItem;

	switch (dialogItem) {
		case M26OK: return TRUE;
		case M26CANCEL: return TRUE;
		case M26LIST:
			if (VLGetSelect (&currSelItem, L))
				SetOWeatherValues (dialog, currSelItem);
		break;
		
		case M26CHANGE:
			if (VLGetSelect (&currSelItem, L))
			{
				GetOWeatherValues (dialog, currSelItem);
				SetOWeatherValues (dialog, currSelItem);
			}
		break;
	}

	return FALSE;
}

OSErr OWeatherSettingsDialog (ListItem item)
{
	short	dialogItem;
	long	pollutantCount;

	sharedItem = item;
	pollutantCount = ((TOSSMWeatherer*) item.owner) -> componentsList -> GetItemCount ();
	SelectFromVListDialog(M26, M26LIST, pollutantCount, OWeatherInit, 0, 0, DrawOWeatherItem, OWeatherClick, TRUE, &dialogItem);

	return 0;
}
