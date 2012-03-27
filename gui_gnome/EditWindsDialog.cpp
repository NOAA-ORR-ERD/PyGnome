#include "Cross.h"
#include "OSSM.h"
#include "TimUtils.h"
#include "OUtils.h"
#include "EditWindsDialog.h"

#ifdef MAC
#define USEGWORLD true //JLM

#ifdef MPW
#pragma SEGMENT EDITWINDS
#endif

#endif
	
static double sgSpeed=0,sgDirection=0;


static short kMaxPixelRadius;
static 	VList sgObjects;
static float sgMaxValue = 40.0;// knots
static Point sgOldPos={0,0};
static CTimeValuePairList *sgTimeVals=0;
static Seconds sgStartTime;
static Boolean sgForWizard = false;
static Boolean	sgSettingsForcedAfterDialog = false;
static Boolean	sgBeenThroughSettings = false;

static long sgSpeedUnits = kKnots;
static float sIncrementInHrs = 6.0;
static Boolean sIsConstantWindDialog;

static TWindMover *sgWindMover=nil;


/////////////////////////////////////////////////

static EWDialogNonPtrFields sharedEWDialogNonPtrFields;

EWDialogNonPtrFields GetEWDialogNonPtrFields(TWindMover * cm)
{
	EWDialogNonPtrFields f;
	
	f.bActive  = cm->bActive; 	
	f.fUncertainStartTime  = cm->fUncertainStartTime; 	
	f.fDuration  = cm->fDuration; 
	//
	f.fSpeedScale  = cm->fSpeedScale; 	
	f.fAngleScale  = cm->fAngleScale; 	
	//f.conversion  = cm->conversion; 	
	//f.windageA = cm->windageA; 
	//f.windageB = cm->windageB;
	//
	f.fIsConstantWind = cm->fIsConstantWind;
	f.fConstantValue = cm->fConstantValue;
	
	return f;
}

void SetEWDialogNonPtrFields(TWindMover * cm,EWDialogNonPtrFields * f)
{
	cm->bActive = f->bActive; 	
	cm->fUncertainStartTime = f->fUncertainStartTime; 	
	cm->fDuration  = f->fDuration; 
	//
	cm->fSpeedScale  = f->fSpeedScale; 	
	cm->fAngleScale  = f->fAngleScale; 	
	//cm->conversion  = f->conversion; 	
	//cm->windageA = f->windageA; 
	//cm->windageB = f->windageB;
	//
	cm->fIsConstantWind = f->fIsConstantWind;
	cm->fConstantValue = f->fConstantValue;
	
}

///////////////////////////////////////////////////////////////////////////

Boolean HideSettingsButton(void)
{
	return model -> GetModelMode () < ADVANCEDMODE || sgSettingsForcedAfterDialog;
}

/////////////////////////////////////////////////



#define EXPONENT 2.0
#define NORTH 	"N"
#define NNE 	"NNE"
#define NE 		"NE"
#define ENE 	"ENE"
#define EAST 	"E"
#define ESE 	"ESE"
#define SE 		"SE"
#define SSE 	"SSE"
#define SOUTH 	"S"
#define SSW 	"SSW"
#define SW 		"SW"
#define WSW 	"WSW"
#define WEST 	"W"
#define WNW 	"WNW"
#define NW 		"NW"
#define NNW 	"NNW"

#define INCREMENT_TIME true
#define REPLACE true

char *dirstr[]={NORTH,NNE,NE,ENE,EAST,ESE,SE,SSE,SOUTH,SSW,SW,WSW,WEST,WNW,NW,NNW};
	
static short DATE_COL,TIME_COL,SPEED_COL,DIR_COL;

static PopInfoRec prefPopTable[] =
{
	{ EDIT_WINDS_DLGID, nil, EWSPEEDPOPUP, 0, pSPEEDUNITS, 0, 1, FALSE, nil },
	{ EDIT_WINDS_DLGID, nil, EWYEARSPOPUP, 0, pYEARS, 0, 1, FALSE, nil },
	{ EDIT_WINDS_DLGID, nil, EWMONTHSPOPUP	, 0, pMONTHS, 0, 1, FALSE, nil }

};

Boolean AddRecordRowIsSelected(void)
{
	long curSelection;
	Boolean  isLastItem;
	if(sIsConstantWindDialog) return false;
	VLGetSelect(&curSelection,&sgObjects);
	isLastItem = curSelection == -1 || curSelection == sgObjects.numItems-1;
	return isLastItem;
}


Boolean ShowAutoIncrement(void)
{
	Boolean  show = AddRecordRowIsSelected();
	return show;
}


static void ShowHideAutoIncrement(DialogPtr dialog,long curSelection)
{

	Boolean show = ShowAutoIncrement();
	if(sIsConstantWindDialog) return ; 
	ShowHideDialogItem(dialog,EWINCREMENT,show);
	ShowHideDialogItem(dialog,EWAUTOINCRTEXT,show);
	ShowHideDialogItem(dialog,EWAUTOINCRHOURS,show);
}
static void  MyDisplayTime(DialogPtr dialog, short monthItem, Seconds seconds)
{
	char num[20];
	DateTimeRec time;
	
	SecondsToDate (seconds, &time);
	SetPopSelection (dialog, monthItem, time.month);
	PopDraw(dialog,monthItem);
	Float2EditText(dialog, monthItem + 1, time.day, 0);
	SetPopSelection (dialog, monthItem + 2,  time.year - (FirstYearInPopup()  - 1));
	PopDraw(dialog,monthItem+2);
	Float2EditText(dialog, monthItem + 3, time.hour, 0);
	sprintf(num, "%02hd", time.minute);
	mysetitext(dialog, monthItem + 4, num);
}

OSErr RetrieveIncrementTime(DialogPtr dialog)
{
	OSErr err = 0;
	float incrementTimeInHrs  = EditText2Float(dialog,EWINCREMENT);
	if(sIsConstantWindDialog) return 0; 
	if(incrementTimeInHrs < 0.1 || incrementTimeInHrs > 72.0)
	{
		printError("The increment time must be between 0.1 and 72.");
		MySelectDialogItemText (dialog, EWINCREMENT, 0, 255);
		return -1;
	}
	sIncrementInHrs = incrementTimeInHrs;// set the static
	return noErr;
}


static void IncrementTime(DialogPtr dialog, unsigned long time)
{
	unsigned long incr;
	if(sIsConstantWindDialog) return ; 
	incr= sIncrementInHrs*3600;
	time += incr;
	MyDisplayTime(dialog,EWMONTHSPOPUP,time);	
}

static void DrawFatArrow(short x0, short y0, short x1, short y1)
{
	double headLen,headWidth,lineWidth,dh,dv,dhB,dvB,h,v,len,dhL,dvL;
#ifdef IBM
	POINT points[8];
#else
	PolyHandle poly;
#endif
	
	
	if(x0 != x1 || y0 != y1)
	{
		dh = x1 - x0;
		dv = y1 - y0;
		len = sqrt(dh*dh+dv*dv);
		headLen= len/3;
		headLen = headLen > 10 ? 10/len: headLen/len;
		headWidth = headLen/1.5;
		lineWidth = headWidth/3;
		dhB = headWidth * dh;
		dvB = headWidth * dv;
		dhL = lineWidth * dh;
		dvL = lineWidth * dv;
		h = x1 - headLen * dh;
		v = y1 - headLen * dv;
		
#ifdef MAC
		poly = OpenPoly();
		MyMoveTo(round(x0+dvL),round(y0-dhL));
		MyLineTo(round(h+dvL),round(v-dhL));
		MyLineTo(round(h+dvB),round(v-dhB));
		MyLineTo(x1,y1);
		MyLineTo(round(h-dvB),round(v+dhB));
		MyLineTo(round(h-dvL),round(v+dhL));
		MyLineTo(round(x0-dvL),round(y0+dhL));
		MyLineTo(round(x0+dvL),round(y0-dhL));
		ClosePoly();
		PaintPoly(poly);
		KillPoly(poly);
#else
		points[0] = MakePOINT(round(x0+dvL),round(y0-dhL));
		points[1] = MakePOINT(round(h+dvL),round(v-dhL));
		points[2] = MakePOINT(round(h+dvB),round(v-dhB));
		points[3] = MakePOINT(x1,y1);
		points[4] = MakePOINT(round(h-dvB),round(v+dhB));
		points[5] = MakePOINT(round(h-dvL),round(v+dhL));
		points[6] = MakePOINT(round(x0-dvL),round(y0+dhL));
		points[7] = MakePOINT(round(x0+dvL),round(y0-dhL));
		Polygon(currentHDC,points,8);
		Polyline(currentHDC,points,8);
#endif
		
	}
	else
	{
		MyMoveTo(x0,y0);MyLineTo(x0,y0);
	}

}


static void MyFloat2EditText(DialogPtr dialog, short itemnum,float num,short numdec)
{
	char numstr[30];
	StringWithoutTrailingZeros(numstr,num,numdec);
	mysetitext(dialog,itemnum,numstr);
}


static OSErr EditText2Direction(DialogPtr dialog, short item,float *angle)
{
	char s[30];
	long n,i=0;
	float f;
	OSErr err = noErr;
	
	mygetitext(dialog,item,s,EWDIRECTION);
	StrToUpper(s);
	RemoveTrailingSpaces(s);
	//if(isdigit(s[0]))
	if (s[0]>='0' && s[0]<='9')	// for code warrior
	{
		*angle =EditText2Float(dialog,EWDIRECTION); // in degrees 
		if(*angle > 360)
		{
			printError("Your direction value cannot exceed 360.");	
			MySelectDialogItemText (dialog, EWDIRECTION, 0, 255);
			err = -2;
		}
		*angle = (*angle)* PI/180;// convert to radians
	}
	else
	{
		if(strcmp(s,NORTH) == 0){i=0;
		}
		else if(strcmp(s,NNE)==0){i=1;
		}
		else if(strcmp(s,NE)==0){i=2;
		}
		else if(strcmp(s,ENE)==0){i=3;
		}
		else if(strcmp(s,EAST)==0){i=4;
		}
		else if(strcmp(s,ESE)==0){i=5;
		}
		else if(strcmp(s,SE)==0){i=6;
		}
		else if(strcmp(s,SSE)==0){i=7;
		}
		else if(strcmp(s,SOUTH)==0){i=8;
		}
		else if(strcmp(s,SSW)==0){i=9;
		}
		else if(strcmp(s,SW)==0){i=10;
		}
		else if(strcmp(s,WSW)==0){i=11;
		}
		else if(strcmp(s,WEST)==0){i=12;
		}
		else if(strcmp(s,WNW)==0){i=13;
		}
		else if(strcmp(s,NW)==0){i=14;
		}
		else if(strcmp(s,NNW)==0){i=15;
		}
		else{
			SysBeep(5);
			err=-1;
		}
		*angle= i*PI/8;
	}
	return err;
}

OSErr CheckWindSpeedLimit(float speed, short speedUnits,char* errStr)
{
	OSErr err = 0;
	long maxSpeed;
	errStr[0] = 0;
	// enforce the limits here
	switch(speedUnits)
	{
		case kKnots: maxSpeed = 50; break;
		case kMetersPerSec: maxSpeed = 25; break;
		case kMilesPerHour: maxSpeed = 55; break;
		default: maxSpeed = -1; break; 
	}
	
	if(speed > maxSpeed && maxSpeed > 0)
	{
		char unitsStr[64];
		ConvertToUnits (speedUnits,unitsStr);
		sprintf(errStr,"The wind speed cannot exceed %ld %s.",maxSpeed,unitsStr);
		err = true;
	}
	return err;
}

static OSErr GetTimeVal(DialogPtr dialog,TimeValuePair *tval)
{
	float speed = EditText2Float(dialog,EWSPEED);
	float angle = 0;
	OSErr  err = 0;
	float maxSpeed;
	short speedUnits = GetPopSelection(dialog, EWSPEEDPOPUP);
	char errStr[256] = "";
	 
	err = CheckWindSpeedLimit(speed,speedUnits,errStr);
	if(err)
	{
		printError(errStr);
		MySelectDialogItemText(dialog, EWSPEED,0,100);
		return err;
	}
	/////////////////////////////
	
	err = EditText2Direction(dialog,EWDIRECTION,&angle);
	if(err) return err;
	
	if(sIsConstantWindDialog)
	{
		tval->time = sgStartTime;// Note: time is unused in constant wind dialog
	}
	else
	{
		tval->time=RetrievePopTime(dialog,EWMONTHSPOPUP,&err);
		if(err) return err;
	}
	
	// The wind velocity components are diametrically opposite to the wind direction specified
	// so have to add PI to the angle
	tval->value.u = speed*sin(angle+PI);
	tval->value.v = speed*cos(angle+PI);
	return err;
}
static void UV2RTheta(double u,double v, double *r, double *theta)
{
	*theta = fmod(atan2(u,v)*180/PI+360,360);
	*theta = fmod(*theta+180,360); // rotate the vector cause wind is FROM this direction
	*r=sqrt(u*u+v*v);
}

void UV2RThetaStrings(double u,double v,long speedUnits,char* text)
{	//JLM
	double r,theta;
	char speedStr[32],directionStr[32],unitStr[32] ="";
	UV2RTheta(u,v,&r,&theta);
	r /= speedconversion(speedUnits);
	
	StringWithoutTrailingZeros(speedStr,r,2);
	Direction2String(theta,directionStr);
	ConvertToUnits (speedUnits, unitStr);
	sprintf(text, "%s %s from %s",speedStr,unitStr,directionStr);
				
}


static short PointDist(Point *p1,Point *p2)
{
	short h,v;
	h = p1->h - p2->h;
	v = p1->v - p2->v;
	//return sqrt(h*h+v*v);
	return (short)sqrt((double)(h*h+v*v));
} 

// return true if direction is exactly a compass direction
// false if direction has to be rounded
static Boolean Direction2CompassDirection(double direction, long *dir)
{
	double x;
	*dir = (direction+11.5)/22.5;
	x =fabs(*dir * 22.5 - direction);
	*dir = (*dir) %16;// JLM 9/3/98
	return x < 1e-4;

}
void Direction2String(double direction,char *s)
{
	long dir;
	if(Direction2CompassDirection(direction,&dir))
	{
		sprintf(s,"%s",dirstr[dir]);
	}
	else 
		StringWithoutTrailingZeros(s,direction,0);
}
static void Direction2EditText(DialogPtr dialog, short item,float direction)
{
	char s[30];
	
	Direction2String(direction,s);
	mysetitext(dialog,item,s);
}


//Convert Velocity rec to position on bulls eye
static void Vel2BullsEyePos(VelocityRec *vel,Point *pos)
{
	double theta;
	double speed = sqrt(vel->u*vel->u+vel->v * vel->v);
	long r = round(speed/sgMaxValue * kMaxPixelRadius);
	if(r > kMaxPixelRadius)r = kMaxPixelRadius;
	theta = atan2(vel->u,vel->v);
	pos->h = round(-r * sin(theta ));
	pos->v = round(r * cos(theta ));
}



static void BullsEyePos2SpeedDir(Point  *startPoint, Point *endPoint,Point *clippedPos,double *magnitude, double *direction)
{
	double		theta, r, x, y,dtheta;
	
	x = startPoint->h - endPoint->h;
	y = endPoint->v - startPoint->v;
	
	*clippedPos = *endPoint;
	dtheta = 2*PI/16;
	theta = floor((atan2(x, y)+dtheta/2)/dtheta)*dtheta;	// snap to compass direction
	*direction = fmod(theta * 180/PI + 180,360); // have to add 180 because direction was rotated by 180 to compute x,y, since wind is FROM the direction			
	r = sqrt(x*x + y*y);
	if ( r > kMaxPixelRadius)
	{
		r = kMaxPixelRadius;
	}
	x = round(-r * sin(theta ));
	y = round(r * cos(theta ));
	clippedPos->h = startPoint->h + x;
	clippedPos->v = startPoint->v + y;

	
	//*magnitude = round(r * sgMaxValue / kMaxPixelRadius);
	*magnitude = round(r* sgMaxValue/kMaxPixelRadius );
	if(*magnitude > 30)
	{
		*magnitude = round(*magnitude/5)*5;
	}
	if (*magnitude == 0)
	{
		clippedPos->h = startPoint->h;
		clippedPos->v = startPoint->v;
	}
}

pascal_ifMac void DrawBullsEye(DialogPtr dialog, short itemNum)
{
	static short knots[]={10,20,30,40};
	Rect r = GetDialogItemBox(dialog,EWBULLSEYEFRAME);
	Rect f;
	char s[20];
	short x,y,d;
	float sqr2Radius = kMaxPixelRadius/sqrt(2.);
	x = (r.left+r.right)/2;
	y = (r.bottom+r.top)/2;
	
#ifndef USEGWORLD
	EraseRect(&r);//JLM
#endif
	DoFrameEmbossed(r);
	RGBForeColor(&colors[BLUE]);
	for(int i=0;i<4;i++)
	{	
		//d = pow(knots[i]/sgMaxValue,1.0/EXPONENT)*kMaxPixelRadius;
		d = knots[i]*kMaxPixelRadius/sgMaxValue;
		f.top = y-d; f.bottom = y+d;
		f.left = x-d; f.right = x+d;
		FrameOval(&f);
		sprintf(s,"%d",knots[i]);
		MyMoveTo(x+d-stringwidth(s)/2,y);
		drawstring(s);
	}
	f.top = y-kMaxPixelRadius; f.bottom = y+kMaxPixelRadius-1;
	f.left = x-kMaxPixelRadius; f.right = x+kMaxPixelRadius-1;
	
	MyMoveTo(f.left,y);MyLineTo(f.right,y);
	MyMoveTo(x,f.top);MyLineTo(x,f.bottom);
	
	f.top = y-sqr2Radius; f.bottom = y+sqr2Radius-1;
	f.left = x-sqr2Radius; f.right = x+sqr2Radius-1;

	MyMoveTo(f.left,f.top);MyLineTo(f.right,f.bottom);
	MyMoveTo(f.left,f.bottom);MyLineTo(f.right,f.top);
	
	RGBForeColor(&colors[BLACK]);
}

void UpdateDisplayWithTimeValuePair(DialogPtr dialog,TimeValuePair tval)
{
	Point pos,mp;
	Rect r= GetDialogItemBox(dialog,EWBULLSEYEFRAME);
	
	MyDisplayTime(dialog,EWMONTHSPOPUP,tval.time);
	UV2RTheta(tval.value.u,tval.value.v,&sgSpeed,&sgDirection);
	MyFloat2EditText(dialog,EWSPEED,sgSpeed,2);
	Direction2EditText(dialog,EWDIRECTION,sgDirection);
		
	Vel2BullsEyePos(&tval.value,&pos);
	mp.h= (r.right+r.left)/2;
	mp.v =(r.bottom+r.top)/2;
}

	
static void UpdateDisplayWithCurSelection(DialogPtr dialog)
{
	TimeValuePair tval;
	Point pos,mp;
	Rect r= GetDialogItemBox(dialog,EWBULLSEYEFRAME);
	long curSelection;
	
	if(!AddRecordRowIsSelected())
	{	// set the item text
		if(sIsConstantWindDialog)
		{ // use field, not the list
			memset(&tval,0,sizeof(tval));
			tval.value = sharedEWDialogNonPtrFields.fConstantValue;
			tval.value.u /= speedconversion(sgSpeedUnits);
			tval.value.v /= speedconversion(sgSpeedUnits);
			tval.time = sgStartTime;// Note: time is unused in constant wind dialog
		}
		else  
		{
			VLGetSelect(&curSelection,&sgObjects);
			sgTimeVals->GetListItem((Ptr)&tval,curSelection);
		}
		
		UpdateDisplayWithTimeValuePair(dialog,tval);
	}

	ShowHideAutoIncrement(dialog,curSelection); // JLM 9/17/98
}

static void SelectNthRow(DialogPtr dialog,long nrow)
{
	if(sIsConstantWindDialog) return ; 
	VLSetSelect(nrow, &sgObjects); 
	if(nrow > -1)
	{
		VLAutoScroll(&sgObjects);
	}
	VLUpdate(&sgObjects);
	UpdateDisplayWithCurSelection(dialog);	
}

static OSErr AddReplaceRecord(DialogPtr dialog,Boolean incrementTime,Boolean replace,TimeValuePair tval)
{
	float speed,angle;
	long n,itemnum,curSelection;
	OSErr err=0;
	
	if(sIsConstantWindDialog) return 0; 
	
	if(!err)
	{
		err=sgTimeVals->InsertSorted ((Ptr)&tval,&itemnum,false);// false means don't allow duplicate times
		
		if(!err) // new record
		{
			
			VLAddItem(1,&sgObjects);
			VLSetSelect(itemnum, &sgObjects); 
			VLAutoScroll(&sgObjects);
			VLUpdate(&sgObjects);
			if(incrementTime)IncrementTime(dialog,tval.time);
		}
		else if(err == -2) // found existing record. Replace if okay to replace
		{
			if(replace)
			{
				
				sgTimeVals->DeleteItem(itemnum);
				VLDeleteItem(itemnum,&sgObjects);
				err = AddReplaceRecord(dialog,!INCREMENT_TIME,REPLACE,tval);
				VLUpdate(&sgObjects);
				if(incrementTime)IncrementTime(dialog,tval.time);
				err=0;
			}
			else
			{
				printError("A record with the specified time already exists."
					"If you want to edit the existing record, select it."
					"If you want to add a new record, change the specified time.");
				VLUpdate(&sgObjects);
			}
		}
		else SysBeep(5);
	}
	return err;
}

#ifdef USEGWORLD
void DrawBullsEyeContent(DialogPtr dialog,MyGWorldRec *gworldPtr,Point *startPos,Point *endPos)
#else
void DrawBullsEyeContent(DialogPtr dialog,Handle *bogus_gworldPtr,Point *startPos,Point *endPos)
#endif
{
#ifdef USEGWORLD
	MySetGWorld(gworldPtr, false, true);
#else
	Rect r = GetDialogItemBox(dialog,EWBULLSEYEFRAME);//JLM
	SetPort(dialog); //JLM
	MyClipRect(r); //JLM
#endif
	PenSize(1,1);
	DrawBullsEye(dialog,EWBULLSEYEFRAME);
	if(startPos != nil && endPos != nil)
	{
		DrawFatArrow(startPos->h,startPos->v,endPos->h,endPos->v);
	}
#ifdef USEGWORLD
	MyRestoreGWorld(gworldPtr);
	MyBlitGWorld(gworldPtr,GetDialogWindow(dialog),patCopy);	
#endif
}

void DrawLineArrow(Point *startPos,Point *oldPos,Point *endPos)
{
		//PenSize(2,2); JLM
		PenMode(patXor);
		if(startPos != nil && oldPos != nil)
		{	// erase old line
			//MyMoveTo(startPos->h,startPos->v); JLM
			//MyLineTo(oldPos->h,oldPos->v); JLM
			if(oldPos->h != startPos->h || oldPos->v != startPos->v) // JLM don't draw the "zero" point
				DrawFatArrow(oldPos->h,oldPos->v,startPos->h,startPos->v);// JLM
		}
		if(startPos!=nil && endPos != nil)
		{
			//MyMoveTo(startPos->h,startPos->v); //JLM
			//MyLineTo(endPos->h,endPos->v); //JLM
			if(endPos->h != startPos->h || endPos->v != startPos->v) // JLM don't draw the "zero" point
				DrawFatArrow(endPos->h,endPos->v,startPos->h,startPos->v); //JLM
		}
		PenSize(1,1);
		PenMode(patCopy);
}
 
void DoClickInBullsEye(DialogPtr dialog)
{
#ifdef USEGWORLD
	MyGWorldRec gworld;
#else
	Handle gworld; // bogus trick
#endif
	Point mp,pos,clippedPos;
	Point oldPos; //JLM
	Point undefinedPt = {-32000,-32000};
	double speed,direction;
	float currentDir;
	OSErr err = noErr,dirErr;
	Boolean useDrawLineArrow = true;//JLM, default to true for the IBM
	long curSelection;
	Boolean incrementTime;
	Rect r = GetDialogItemBox(dialog,EWBULLSEYEFRAME);
	mp.h= (r.right+r.left)/2;
	mp.v =(r.bottom+r.top)/2;
	oldPos = undefinedPt; //JLM 1/12/99
	 	
	pos = GetMouseLocal(GetDialogWindow(dialog));
	BullsEyePos2SpeedDir(&mp, &pos,&clippedPos,&speed, &direction);
#ifdef USEGWORLD
	err=MyNewGWorld(GetDialogWindow(dialog), 8, &r,&gworld);
		if(!err) useDrawLineArrow = false; // JLM
#endif
	while(StillDown())
	{
		//if(sqrt((pos.h-mp.h)*(pos.h-mp.h)+(pos.v-mp.v)*(pos.v-mp.v)) > kMaxPixelRadius+5)
		if(sqrt(double((pos.h-mp.h)*(pos.h-mp.h)+(pos.v-mp.v)*(pos.v-mp.v))) > kMaxPixelRadius+5)
		{	// outside of circle
		 	// we want to have nothing showing in the bulls eye and disregard the 
			// the mouse position
			// we also want the original values to be returned to the edit text
			
			
			if(useDrawLineArrow) 
			{
				if(!EqualPoints(oldPos,undefinedPt))
					DrawLineArrow(&mp,&oldPos,nil); // JLM erase last line
				oldPos = undefinedPt; // so it will not draw the next time around
			}
			else
				DrawBullsEyeContent(dialog,&gworld,nil,nil);

			//put back saved speed and direction
			//dirErr =  EditText2Direction(dialog,EWDIRECTION,&currentDir);
			//currentDir *= 180/PI;
			if(EditText2Float(dialog,EWSPEED) != sgSpeed)
			{
				MyFloat2EditText(dialog,EWSPEED,sgSpeed,2);
				Direction2EditText(dialog,EWDIRECTION,sgDirection);

				#ifdef IBM
				{	//JLM
					// we need to send they guys paint messages right now
					// to get them to visually update
					HWND hWnd = (HWND)GetDialogItemHandle(dialog,EWSPEED);
					SendMessage(hWnd, WM_PAINT, 0, 0);
					hWnd = (HWND)GetDialogItemHandle(dialog,EWDIRECTION);
					SendMessage(hWnd, WM_PAINT, 0, 0);
				}
				#endif
			}
		}
		else if(clippedPos.h != oldPos.h || clippedPos.v != oldPos.v) //JLM
		{
		
			if(useDrawLineArrow)//JLM
			{
				if(EqualPoints(oldPos,undefinedPt))
					DrawLineArrow(&mp,nil,&clippedPos); 
				else
					DrawLineArrow(&mp,&oldPos,&clippedPos);
			}
			else DrawBullsEyeContent(dialog,&gworld,&clippedPos,&mp);
			
			oldPos = clippedPos;
			Direction2EditText(dialog,EWDIRECTION,direction);
			MyFloat2EditText(dialog,EWSPEED,speed,2);
			#ifdef MAC
				// this might mess up the IBM
				// make sure all of the test is highlighted
				MySelectDialogItemText(dialog, EWSPEED, 0, 100);
			#endif
			
			#ifdef IBM
			{	//JLM
				// we need to send they guys paint messages right now
				// to get them to visually update
				HWND hWnd = (HWND)GetDialogItemHandle(dialog,EWSPEED);
				SendMessage(hWnd, WM_PAINT, 0, 0);
				hWnd = (HWND)GetDialogItemHandle(dialog,EWDIRECTION);
				SendMessage(hWnd, WM_PAINT, 0, 0);
			}
			#endif
		}
		pos = GetMouseLocal(GetDialogWindow(dialog));
		BullsEyePos2SpeedDir(&mp, &pos,&clippedPos,&speed, &direction);
	}
	if(useDrawLineArrow)
	{
		if(!EqualPoints(oldPos,undefinedPt))
			DrawLineArrow(&mp,&oldPos,nil); // JLM erase last line
	}
	else 
	{
		DrawBullsEyeContent(dialog,&gworld,nil,nil);
	#ifdef USEGWORLD
		MyKillGWorld(&gworld);
	#endif
	}
	oldPos=undefinedPt;
	
	if(sIsConstantWindDialog) return; // don't need to add item

	//if(sqrt((pos.h-mp.h)*(pos.h-mp.h)+(pos.v-mp.v)*(pos.v-mp.v)) < kMaxPixelRadius+5)
	if(sqrt(double((pos.h-mp.h)*(pos.h-mp.h)+(pos.v-mp.v)*(pos.v-mp.v))) < kMaxPixelRadius+5)
	{
		// If this is not the last row, select the next row.
		// If it was the last row, add a new blank row and select it
		TimeValuePair tval;
		VLGetSelect(&curSelection,&sgObjects);
		Boolean incrementTime=curSelection>=sgTimeVals->GetItemCount()-1;
		err=GetTimeVal(dialog,&tval);
		if(err) return;
		err = AddReplaceRecord(dialog,incrementTime,REPLACE,tval);
		SelectNthRow(dialog, curSelection+1 ); 
	#ifdef MAC
		FlashItem(dialog,EWREPLACE);
	#endif
	}
	
}

void 	DisposeEWStuff(void)
{
	if(sgTimeVals)
	{
		sgTimeVals->Dispose();// JLM 12/14/98
		delete sgTimeVals;
		sgTimeVals = 0;
	}

	//?? VLDispose(&sgObjects);// JLM 12/10/98, is this automatic on the mac ??
	memset(&sgObjects,0,sizeof(sgObjects));// JLM 12/10/98
}

	
short EditWindsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	Point pos,mp,clippedPos;
	Rect r;
	double speed, direction;
	long curSelection;
	long dir,i,n;
	unsigned long incr;
	char s[30];
	OSErr err=0,settingsErr = 0;
	CTimeValuePairList *tlist;
	TimeValuePair tval;
	
	if(!sIsConstantWindDialog)
	{
		if (AddRecordRowIsSelected())
		{
			//Last row is selected
			//Disable delete button
			MyEnableControl(dialog,EWDELETEROWS_BTN,FALSE);
			// ANd change title in replace dialog to "Add new record"
			MySetControlTitle(dialog, EWREPLACE, "Add New Record");
		}
		else
		{
			MySetControlTitle(dialog, EWREPLACE, "Replace Selected");
			MyEnableControl(dialog,EWDELETEROWS_BTN,TRUE);
		}
	}
	
	
	switch(itemNum)
	{
		case EWOK:
		{
			// don't retrive increment here
			// Just use the value from the last time they incremented.
			// Why bother them if we are not going to use the value.
			//if(ShowAutoIncrement()) err = RetrieveIncrementTime(dialog);
			//if(err) break;
			
			TOSSMTimeValue *timeValue = sgWindMover->GetTimeDep();
				
			sgSpeedUnits = GetPopSelection(dialog, EWSPEEDPOPUP);
			
			if(sIsConstantWindDialog)
			{ // retrieve the values from the edit text and put in the field
				err=GetTimeVal(dialog,&tval);
				if(err) break;
				tval.value.u *= speedconversion(sgSpeedUnits);
				tval.value.v *= speedconversion(sgSpeedUnits);
								
				// point of no return, constant wind case
				if(timeValue) timeValue -> SetUserUnits(sgSpeedUnits);
				sgWindMover->fConstantValue = tval.value;
			}
			else
			{ // variable wind

				if(sgTimeVals && timeValue)
				{
					TimeValuePairH tvalh = timeValue -> GetTimeValueHandle();
					n= sgTimeVals->GetItemCount();
					if(n == 0)
					{	// no items are entered, tell the user
						char msg[512],buttonName[64];
						if(sgForWizard) GetWizButtonTitle_Previous(buttonName);
						else GetWizButtonTitle_Cancel(buttonName);
						sprintf(msg,"You have not entered any wind values.  Either enter wind values and use the 'Add New Record' button, or use the '%s' button to exit the dialog.",buttonName);
						printError(msg);
						break;
					}
					
					// check that all the wind values are in range
					// because the user could have changed units
					for(i=0;i<n;i++)
					{
						float speed;
						double r,theta;
						char errStr[256] = "";
						err=sgTimeVals->GetListItem((Ptr)&tval,i);
						if(err) {SysBeep(5); break;}// this shouldn't ever happen
						UV2RTheta(tval.value.u,tval.value.v,&r,&theta);
						err = CheckWindSpeedLimit(r,sgSpeedUnits,errStr);
						if(err)
						{
							strcat(errStr,"  Check your units and each of the records you entered.");
							printError(errStr);
							return 0; // stay in the dialog
						}
					}
					
					
					/////////////
					// point of no return, variable wind case
					//////////////
					if(timeValue) timeValue -> SetUserUnits(sgSpeedUnits);
					if(tvalh == 0)
					{
						tvalh = (TimeValuePairH)_NewHandle(n*sizeof(TimeValuePair));
						if(!tvalh)
						{
							TechError("EditWindsClick:OKAY", "_NewHandle()", 0);
							//return EWCANCEL;
							break; // make them cancel so that code gets executed
						}
						timeValue -> SetTimeValueHandle(tvalh);
					}
					else
					{
						 _SetHandleSize((Handle)tvalh,n*sizeof(TimeValuePair));
						 if(_MemError())
						 {
							 TechError("EditWindsClick:OKAY", "_NewHandle()", 0);
							//return EWCANCEL;
							break; // make them cancel, so that code gets executed
						 }
					}
					
	
					for(i=0;i<n;i++)
					{
						if(err=sgTimeVals->GetListItem((Ptr)&tval,i))return EWOK;
						tval.value.u *= speedconversion(sgSpeedUnits);
						tval.value.v *= speedconversion(sgSpeedUnits);
						(*tvalh)[i]=tval;					
					}
				}
			}
			
			/////////////////////////////
			DisposeEWStuff();
			return EWOK;
		}
			
		case EWCANCEL:
			SetEWDialogNonPtrFields(sgWindMover,&sharedEWDialogNonPtrFields);
			DisposeEWStuff();
			return EWCANCEL;
			break;
			
		case EWINCREMENT:
		case EWSPEED:
			CheckNumberTextItem(dialog, itemNum, TRUE); //  allow decimals
			break;
			
		case EWDAY:
		case EWHOUR:
		case EWMINUTE:
			CheckNumberTextItem(dialog, itemNum, FALSE); // don't allow decimals
			break;

		case EWDIRECTION:
			CheckDirectionTextItem(dialog, itemNum);
			break;
						
		case EWDELETEALL:
			if(sIsConstantWindDialog) break; // shouldn't happen
			sgTimeVals->ClearList();
			VLReset(&sgObjects,1);
			MyDisplayTime(dialog,EWMONTHSPOPUP,sgStartTime);// 1/6/99, CJ asked that delete all reset the time to the model start time
			UpdateDisplayWithCurSelection(dialog);
			break;
		case EWDELETEROWS_BTN:
			if(sIsConstantWindDialog) break; // shouldn't happen
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				sgTimeVals->DeleteItem(curSelection);
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
		case EWSPEEDPOPUP:
			{
				PopClick(dialog, itemNum, &sgSpeedUnits);
			}
			break;
		case EWMONTHSPOPUP:
		case EWYEARSPOPUP:
			{
				long dontcare;
				PopClick(dialog,itemNum,&dontcare);
			}
			break;
		case EWREPLACE:
			if(sIsConstantWindDialog) break; // shouldn't happen
			err = RetrieveIncrementTime(dialog);
			if(err) break;
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				err=GetTimeVal(dialog,&tval);
				if(err) break;
	
				if(curSelection==sgTimeVals->GetItemCount())
				{
					// replacing blank record
					err = AddReplaceRecord(dialog,INCREMENT_TIME,!REPLACE,tval);
					SelectNthRow(dialog, curSelection+1 ); 
				}
				else // replacing existing record
				{
					VLGetSelect(&curSelection,&sgObjects);
					sgTimeVals->DeleteItem(curSelection);
					VLDeleteItem(curSelection,&sgObjects);		
					err = AddReplaceRecord(dialog,!INCREMENT_TIME,REPLACE,tval);
				}
			}
			break;
		case EWSETTINGS:
		{
			settingsErr = WindSettingsDialog(sgWindMover, sgWindMover -> GetMoverMap(),false,GetDialogWindow(dialog),true);
			if(!settingsErr) sgBeenThroughSettings = true;
			break;
		}

		case EWLIST:
			if(sIsConstantWindDialog) break; // shouldn't happen
			// retrieve every time they click on the list
			// because clicking can cause the increment to be hidden
			// and we need to verify it before it gets hidden
			err = RetrieveIncrementTime(dialog);
			if(err) break;
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
		
			if (AddRecordRowIsSelected())
			{
				TimeValuePair tval;
				sgTimeVals->GetListItem((Ptr)&tval,sgTimeVals->GetItemCount()-1);
				err = RetrieveIncrementTime(dialog);
				if(err) break;
				IncrementTime(dialog,tval.time);
			}
			UpdateDisplayWithCurSelection(dialog);
			break;
		
		case EWBULLSEYEFRAME:
			if(!sIsConstantWindDialog) err = RetrieveIncrementTime(dialog);
			if(err) break;
			///////////
			DoClickInBullsEye(dialog);
			if(sIsConstantWindDialog) break; 
			VLGetSelect(&curSelection,&sgObjects);
			ShowHideAutoIncrement(dialog,curSelection);
			break;
	}
	 
	return 0;
}

		
void DrawWindList(DialogPtr w, RECTPTR r, long n)
{
	char s[256];
	TimeValuePair tval;
	DateTimeRec date;
	double speed,angle;
	short militarytime,twoDigitYear;
	long dir;
	
	if(sIsConstantWindDialog) return;

	if(n == sgObjects.numItems-1)
	{
		strcpy(s,"****");
	 	MyMoveTo(DATE_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	MyMoveTo(TIME_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	MyMoveTo(SPEED_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	//MyMoveTo(DIR_COL-20,r->bottom); //JLM
	 	MyMoveTo(DIR_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	return; 
	}
	
	
	sgTimeVals->GetListItem((Ptr)&tval,n);
	SecondsToDate(tval.time, &date);
	
	twoDigitYear = date.year%100;	
	sprintf(s,twoDigitYear<10?"%d/%d/0%d":"%d/%d/%d",date.month,date.day,twoDigitYear);
	MyMoveTo(DATE_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	militarytime = date.hour*100+date.minute;
	if(militarytime < 10)
	{
		sprintf(s,"000%d",militarytime);
	}
	else if(militarytime< 100)
	{
		sprintf(s,"00%d",militarytime);
	}
	else if (militarytime < 1000)
	{
		sprintf(s,"0%d",militarytime);
	}
	else sprintf(s,"%d",militarytime);
	MyMoveTo(TIME_COL-stringwidth(s)/2,r->bottom);

	drawstring(s);
	
	//speed = sqrt(tval.value.u*tval.value.u + tval.value.v*tval.value.v);
	//angle = fmod(atan2(tval.value.u,tval.value.v)*180/PI+360,360);
	UV2RTheta(tval.value.u,tval.value.v,&speed,&angle);
	StringWithoutTrailingZeros(s,speed,2);
	MyMoveTo(SPEED_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	
	Direction2String(angle,s);
	//MyMoveTo(DIR_COL-20,r->bottom);//JLM
	MyMoveTo(DIR_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
}


pascal_ifMac void WindListUpdate(DialogPtr dialog, short itemNum)
{
	Rect r = GetDialogItemBox(dialog,EWLIST);
	
	if(sIsConstantWindDialog) return;
	VLUpdate(&sgObjects);
}

void OffsetDialogItem(DialogPtr dialog,short buttonID,short dh, short dv)
{
#ifdef MAC
	short		itemType;
	Handle	itemHandle;
	Rect		itemBox;
	GetDialogItem(dialog,buttonID,&itemType,&itemHandle,&itemBox);
	MyOffsetRect(&itemBox,dh,dv);
	if(itemType & ctrlItem) MoveControl((ControlHandle)itemHandle,itemBox.left,itemBox.top);
	SetDialogItem(dialog,buttonID,itemType,itemHandle,&itemBox);
#else
	Rect itemBox = GetDialogItemBox(dialog,buttonID);
	MyOffsetRect(&itemBox,dh,dv);
	SetDialogItemBox(dialog,buttonID,itemBox);
#endif
}


OSErr EditWindsInit(DialogPtr dialog, VOIDPTR data)
{
	Rect r = GetDialogItemBox(dialog, EWLIST);
	Seconds starttime;
	CTimeValuePairList *tlist;
	TimeValuePair tval;
	TimeValuePairH tvalh = 0;
	long i,n;
	OSErr err;
	short IBMoffset;
	
	if(UseExtendedYears())
		prefPopTable[1].menuID = pYEARS_EXTENDED;
	else
		prefPopTable[1].menuID = pYEARS;
	RegisterPopTable(prefPopTable, 3);
	RegisterPopUpDialog(EDIT_WINDS_DLGID, dialog);
	
	sharedEWDialogNonPtrFields = GetEWDialogNonPtrFields(sgWindMover);// JLM 11/25/98
	memset(&sgObjects,0,sizeof(sgObjects));// JLM 12/10/98
	
	sIsConstantWindDialog = sharedEWDialogNonPtrFields.fIsConstantWind ;
	
	if(sIsConstantWindDialog) setwtitle(GetDialogWindow(dialog), "Constant Wind");
	else setwtitle(GetDialogWindow(dialog), "Variable Winds");
	
	if(sIsConstantWindDialog) 
	{
		sgTimeVals = nil;
		n = 0;
	}
	else
	{
		TOSSMTimeValue *timeValue = sgWindMover->GetTimeDep();
		TimeValuePairH tvalh = timeValue -> GetTimeValueHandle();

		sgTimeVals = new CTimeValuePairList(sizeof(TimeValuePair));
		if(!sgTimeVals)return -1;
		if(sgTimeVals->IList())return -1;

		if(tvalh)
		{
			// copy list to temp list
			n= _GetHandleSize((Handle)tvalh)/sizeof(TimeValuePair);
			for(i=0;i<n;i++)
			{
				tval=(*tvalh)[i];
				tval.value.u /= speedconversion(sgSpeedUnits);
				tval.value.v /= speedconversion(sgSpeedUnits);
				err=sgTimeVals->AppendItem((Ptr)&tval);
				if(err)return err;
			}
		}
		else  n=0;
		
		n++; // Always have blank row at bottom
			
		err = VLNew(dialog, EWLIST, &r,n, DrawWindList, &sgObjects);
		if(err) return err;
	}
	
	
	if(!sIsConstantWindDialog)
	{
		SetDialogItemHandle(dialog,EWFRAMEINPUT,(Handle)FrameEmbossed);
		SetDialogItemHandle(dialog,EWBUTTONFRAME,(Handle)FrameEmbossed);
		SetDialogItemHandle(dialog,EWLIST,(Handle)WindListUpdate);
	}
	ShowHideDialogItem(dialog,EWBUTTONFRAME,false);//JLM, hide this frame , we have a different button arrangement
	SetDialogItemHandle(dialog,EWBULLSEYEFRAME,(Handle)DrawBullsEye);
	
	r = GetDialogItemBox(dialog,EWLIST);
#ifdef IBM
	IBMoffset = r.left;
#else 
	IBMoffset = 0;
#endif
	r = GetDialogItemBox(dialog, EWDATE_LIST_LABEL);DATE_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, EWTIME_LIST_LABEL);TIME_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, EWSPEED_LIST_LABEL);SPEED_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, EWDIRECTION_LIST_LABEL);DIR_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog,EWBULLSEYEFRAME);
	sgOldPos.h=(r.right+r.left)/2;
	sgOldPos.v=(r.bottom+r.top)/2;

	kMaxPixelRadius = .95 * (r.right-r.left)/2;

	Float2EditText(dialog, EWINCREMENT,sIncrementInHrs, 0);
	Float2EditText(dialog, EWSPEED,0.0, 0);
	Float2EditText(dialog, EWDIRECTION,0.0, 0);

	SetPopSelection (dialog, EWSPEEDPOPUP, sgSpeedUnits);
	MyDisplayTime(dialog,EWMONTHSPOPUP,sgStartTime);

	///////////
	if(sIsConstantWindDialog) 
	{
		short dh,dv;
		Rect originalOKRect = GetDialogItemBox(dialog,EWOK);
		Rect originalListRect = GetDialogItemBox(dialog, EWLIST);
		Rect originalDeleteAllRect = GetDialogItemBox(dialog, EWDELETEALL);
		Rect originalSettingsRect = GetDialogItemBox(dialog, EWSETTINGS);
		
		// move the buttons up and shrink the window
		dh = originalDeleteAllRect.left -originalSettingsRect.left;
		dv = originalListRect.top -originalSettingsRect.top;
		OffsetDialogItem(dialog,EWSETTINGS,dh,dv);
		dh = 0;
		dv = originalListRect.top - originalOKRect.top;
		OffsetDialogItem(dialog,EWOK,dh,dv);
		OffsetDialogItem(dialog,EWHILITEDDEFAULT,dh,dv);
		OffsetDialogItem(dialog,EWCANCEL,dh,dv);
		OffsetDialogItem(dialog,EWHELP,dh,dv);
		
		// hide a bunch of items
		ShowHideDialogItem(dialog,EWDELETEROWS_BTN,FALSE); 
		ShowHideDialogItem(dialog,EWDELETEALL,FALSE); 
		ShowHideDialogItem(dialog,EWREPLACE,FALSE); 
		ShowHideDialogItem(dialog,EWMONTHSPOPUP,FALSE); 
		ShowHideDialogItem(dialog,EWDAY,FALSE); 
		ShowHideDialogItem(dialog,EWYEARSPOPUP,FALSE); 
		ShowHideDialogItem(dialog,EWHOUR,FALSE); 
		ShowHideDialogItem(dialog,EWMINUTE,FALSE); 
		ShowHideDialogItem(dialog,EWINCREMENT,FALSE); 
		ShowHideDialogItem(dialog,EWAUTOINCRTEXT,FALSE); 
		ShowHideDialogItem(dialog,EWAUTOINCRHOURS,FALSE); 
		ShowHideDialogItem(dialog,EWWINDATA,FALSE); 
		ShowHideDialogItem(dialog,EWDATE_LIST_LABEL,FALSE); 
		ShowHideDialogItem(dialog,EWTIME_LIST_LABEL,FALSE); 
		ShowHideDialogItem(dialog,EWSPEED_LIST_LABEL,FALSE); 
		ShowHideDialogItem(dialog,EWDIRECTION_LIST_LABEL,FALSE); 
		ShowHideDialogItem(dialog,EWLIST,FALSE);
		ShowHideDialogItem(dialog,EWBUTTONFRAME,FALSE);
		ShowHideDialogItem(dialog,EWTIMELABEL,FALSE);
		ShowHideDialogItem(dialog,EWTIMECOLONLABEL,FALSE);
		
		// resize the window
		r = GetDialogPortRect(dialog);
		SizeWindow(GetDialogWindow(dialog),RectWidth(r),RectHeight(r) + dv, true);
		#ifdef MAC
			CenterDialogUpLeft(dialog);
		#else
			CenterDialog(dialog,0);
		#endif
		
		
	}
	//////////

	if(sgForWizard)
	{
		char str[64];
		GetWizButtonTitle_Next(str);
		MySetControlTitle(dialog,EWOK,str);
		GetWizButtonTitle_Previous(str);
		MySetControlTitle(dialog,EWCANCEL,str);
	}
	SetDialogItemHandle(dialog, EWHILITEDDEFAULT, (Handle)FrameDefault);
	
	if(!sIsConstantWindDialog)
		ShowHideDialogItem(dialog,EWHILITEDDEFAULT,false);//JLM, hide this item , this dialog has no default

	
	ShowHideDialogItem(dialog,EWSETTINGS,!HideSettingsButton());
		
	
	UpdateDisplayWithCurSelection(dialog);
	
	MySelectDialogItemText(dialog, EWSPEED, 0, 100);//JLM
	return 0;
}

									    

#ifdef MAC
pascal_ifMac Boolean EditWindsFilter(DialogPtr dialog, EventRecord *theEvent, short *itemHit)
{
	char buttonTitle[30], help[20], windowTitle[50], cancel[10], close[10];
	short i;
	GrafPtr oldPort;
	ControlHandle whichControl;
	Point p;
	long curSelection;
	static DialogPtr currentDialog = nil;
	static Boolean justDragged = FALSE;
	
	// if (GetAppRefCon(globalSTPtr)->inBackground) RESUME(globalSTPtr, nil);
	
	GetPortGrafPtr(&oldPort);
	SetPortDialogPort(dialog);
	
	if (resetCursor) InitCursor();
	// if (currentDialog != dialog) InitCursor();
	currentDialog = dialog;
	
	settings.doNotPrintError = false;//JLM 6/19/98, allows dialogs to come up more than once
	
	lastEvent = *theEvent;
	
	if (!sIsConstantWindDialog && KeyEvent(theEvent, RETURN, ENTER))
	{
		// eat return and enter
		SetPortGrafPort(oldPort);
		return TRUE;
	}

	if (!sIsConstantWindDialog && KeyEvent(theEvent, DOWNARROW, DOWNARROW))
	{
		// it is not assumed they wanted to enter anything
		if (VLGetSelect(&curSelection, &sgObjects))
		{
			if(curSelection < sgObjects.numItems-1)SelectNthRow(dialog,curSelection+1);
		}
		SetPortGrafPort(oldPort);
		return TRUE;
	}

	if (!sIsConstantWindDialog && KeyEvent(theEvent, UPARROW, UPARROW))
	{
		// back up one record in list
		if (VLGetSelect(&curSelection, &sgObjects))
		{
			if(curSelection > 0)SelectNthRow(dialog,curSelection-1);
		}		
		SetPortGrafPort(oldPort);
		return TRUE;
	}
	
	SetPortGrafPort(oldPort);
	return STDFilter(dialog,theEvent,itemHit);
}
#endif	

#ifdef IBM  /////////////////////////////
	//// code for the IBM
/////////////////////////////
BOOL CALLBACK EditWindsFilter(HWND hWnd, unsigned message, WORD wParam, LONG lParam)
{
	Boolean		done;
	PAINTSTRUCT	ps;
	HDC		 	hdc;
	
	switch(message)
	{
		case WM_INITDIALOG:
			(void)STDWinDlgProc(hWnd,message,wParam,lParam);// first use standard behavior
			// then have the speed text highlighted
			MySelectDialogItemText(hWnd, EWSPEED, 0, 100);//JLM
			return FALSE;
	}
	return STDWinDlgProc(hWnd,message,wParam,lParam);
}
	
LRESULT CALLBACK BullsEyeWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	HWND hWndParent = GetParent(hWnd);
	HDC hDC;
	PAINTSTRUCT ps;
	
	switch (message) {
		case WM_PAINT:      
		{ // JLM code
			hDC = BeginPaint(hWnd, &ps);
			EndPaint(hWnd, &ps); // so it thinks we painted
			DrawBullsEye(hWndParent,EWBULLSEYEFRAME);
			return TRUE;    
		}

		
		
		case WM_LBUTTONDBLCLK:
		case WM_LBUTTONDOWN:
		{	// JLM code
			EditWindsClick(hWndParent,EWBULLSEYEFRAME,0,0);
			return TRUE;
		}

		
		
		case WM_SETCURSOR:
			if (hWndParent == hMainWnd) {
				InitCursor(); // keep Windows from resetting the arrow cursor within the client rect
				break;
			}
			// fall through
		
		default: return DefWindowProc(hWnd, message, wParam, lParam);
	}
	
	return 0;
}

void RegisterBullsEye(void)
{
	WNDCLASS wc;
	
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
	wc.lpfnWndProc = BullsEyeWndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInst;
	wc.hIcon = NULL;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH); // (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = "BullsEye";
	
	if (!RegisterClass(&wc)) SysBeep(1);
	
}
#endif ////////////////////////////////////

OSErr EditWindsDialog(TWindMover *windMover,Seconds startTime,Boolean forWizard,Boolean settingsForcedAfterDialog)
{
	short item;
	//TimeValuePairH timevals = 0;
	//timevals = timeValue -> GetTimeValueHandle();
	TOSSMTimeValue *timeValue = 0;
	if(!windMover) return -1;
	timeValue = windMover->GetTimeDep();
	if(!timeValue) return -1;
	sgStartTime = startTime;
	sgForWizard = forWizard;
	sgSettingsForcedAfterDialog = settingsForcedAfterDialog;
	sgBeenThroughSettings = false;
	sgSpeedUnits = timeValue -> GetUserUnits();
	if(sgSpeedUnits == kUndefined) sgSpeedUnits = kKnots; //std default
	sgWindMover = windMover;
	#ifdef MAC
	item = MyModalDialogF(EDIT_WINDS_DLGID, mapWindow, 0, EditWindsInit,EditWindsFilter, EditWindsClick);
	#else
	item = MyModalDialogF(EDIT_WINDS_DLGID, mapWindow, 0, EditWindsInit,(DLGPROC)EditWindsFilter, EditWindsClick);
	#endif
	SetWatchCursor();
	if(item == EWOK)
	{
		model->NewDirtNotification();// JLM
		return 0;
	}
	else if(item == EWCANCEL) return USERCANCEL;
	else return -1;
}
