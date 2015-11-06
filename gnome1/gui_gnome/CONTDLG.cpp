#include "ContDlg.h"
#include "Cross.h"

long gMaxNumContours;
short gContourType;
static DOUBLEH *gContourLevels = 0;
static DOUBLEH gEditContourLevels = 0;

#define POINTDRAWFLAG 0
#define POINTSELECTFLAG 1
#define POINTDESELECTFLAG 2
#define BADPOINTDRAWFLAG 3
/////////////////////////////////////////////////

// move to Genutil or Cross2...
char *MyNumToStr(double x, char *s)
{
	double ax = fabs(x);
	//if(ax <= .01)
	if(ax < .01)
	{
		if(ax < 1e-6)
		{
			sprintf(s,"0");
		}
		//else sprintf(s,"%.3e",x);
		else sprintf(s,"%.2e",x);
	}
	//else sprintf(s,lfFix("%.3lf"),x);
	else sprintf(s,lfFix("%.2lf"),x);
	ChopEndZeros(s);
	if(strlen(s) > 1 && s[0] == '0')
	{
		return s+1;
	}
	else return s;
}

// move to genutil...
Boolean EmptyEditText(DialogPtr theDialog, short item)
{
	char str[30];
	mygetitext(theDialog,item,str,30);
	return strlen(str) == 0;
}

// move to cross2 or genutil...
void  SetRGBColor(RGBColor *rgb,long red, long green, long blue)
{
#ifdef MAC	
	rgb->red = red;	
	rgb->green = green;	
	rgb->blue = blue;
#else	
	*rgb = RGB((int)red,(int)green,(int)blue);
#endif
}

#ifdef IBM

Boolean FloatEqZero(float x)
{
	return fabs(x) < 1e-6;
}

RGBColor HSV2RGB(float h, float s ,float v)
{
	int i;
	float f,p,q,t;
	if(	FloatEqZero(s))
	{
		i = floor(v*256);
		return RGB(v,v,v);
	}
	else
	{
		if(FloatEqZero(h-360))h = 0;
		h /= 60;
		i = floor(h);
		f = h-i;
		p = 255*v*(1-s);
		q = 255*v*(1-s*f);
		t = 255*v*(1-(s*(1-f)));
		v *= 255;
		switch(i)
		{
			/*case 0: return RGB(v,t,p); 
			case 1: return RGB(q,v,p); 
			case 2: return RGB(p,v,t); 
			case 3: return RGB(p,q,v); 
			case 4: return RGB(t,p,v); 
			case 5: return RGB(v,p,q); */
			case 5: return RGB(v,t,p); 
			case 0: return RGB(q,v,p); 
			case 1: return RGB(p,v,t); 
			case 2: return RGB(p,q,v); 
			case 3: return RGB(t,p,v); 
			case 4: return RGB(v,p,q); 
		
		}
	}
	return RGB(v,v,v); // shouldn't get here
}

RGBColor GetRGBColor(double val)
{

	return HSV2RGB(val*270,1.0,1.0);
}

#else
RGBColor GetRGBColor(double val)
{
	RGBColor rgb;
	HSVColor hsv;
	hsv.hue = (SmallFract)Fix2SmallFract(FixRatio(32768*val*.8,32768));
	//hsv.hue = (SmallFract)Fix2SmallFract(FixRatio(32767*val*.8,32767));	// tweak to change color scheme
	hsv.saturation = (SmallFract)0xFFFF;
	hsv.value = (SmallFract)0xFFFF;
	HSV2RGB(&hsv,&rgb);
	return rgb;
}
#endif

		
long DeleteDoubleHdlItem(DOUBLEH h, long itemno)
{
	long nelements = GetNumDoubleHdlItems(h);
	long i;
	
	if(h == 0 || nelements == 0 || itemno  > nelements) return 0;
	--nelements;
	for(i = itemno; i < nelements; i++)
	{
		(*h)[i] = (*h)[i+1];
	}
	
	_SetHandleSize((Handle)h,nelements*sizeof(double));
	
	return nelements;
}

void MyDisposeHandle(Handle *g)
{
	if(*g)DisposeHandle(*g);
	*g = 0;
}

long AddUniquelySorted(DOUBLEH h, double value)
{
	long curSize = _GetHandleSize((Handle)h);
	long i,index,nElements = GetNumDoubleHdlItems(h);
	double diff,epsilon=1e-6;
	
	for(i=0; i<nElements; i++)
	{
		diff = value - (*h)[i];
		if(diff < 0) break;
		if(diff >= -epsilon && diff <= epsilon)
		{
			return 0; // already in list
		}
	}
	
	index = i;
	_SetHandleSize((Handle)h,curSize + sizeof(double));
	if (_MemError()) 
	{
		TechError("Out of memory", "_SetHandleSize()", 0);
		SysBeep(5);
		return -1;
	}

	if(nElements > 0)
	{
		for(i= nElements - 1; i >= index; i--)
		{
			(*h)[i+1] = (*h)[i];
		}
	}
	(*h)[index] = value;
	return index;
}

void DeleteSelectedItem(DialogPtr d, VLISTPTR L)
{
#pragma unused (d)
	long lineno,nitems;
	VLGetSelect(&lineno,L);
	nitems = DeleteDoubleHdlItem(gEditContourLevels,lineno);
	VLReset(L,nitems); 
	//VLUpdate(L); 
	VLAutoScroll(L);
	
	if(nitems > 0) VLSetSelect(lineno,L);
}

void DeleteAll(DialogPtr d, VLISTPTR L)
{
#pragma unused (d)
	_SetHandleSize((Handle)gEditContourLevels,0);
	VLReset(L,0);
	//VLUpdate(L);
}

void VectDraw(DialogPtr d, Rect *rectPtr, long itemNumber)
{
#pragma unused (rectPtr)
	Point		p;
	short		h,v;
	RGBColor	rgb;
	Rect		rgbrect;
	char 		numstr[30],numstr2[30];
	double 		x  = (*gEditContourLevels)[itemNumber];
	float		colorLevel;
	long		numLevels = GetNumDoubleHdlItems(gEditContourLevels);
	
	SetRGBColor(&rgb,0,0,0);

	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
	
	rgbrect=GetDialogItemBox(d,CONT_LISTID);
	h=(rgbrect.left);
	GetPen(&p);
	v=p.v;

	MySetRect(&rgbrect,h+4,v-9,h+14,v+1);
	
	// set unique color for each value, not on linear scale
	colorLevel = float(itemNumber)/float(numLevels-1);
	//rgb = GetRGBColor(colorLevel);
#ifdef IBM
	rgb = GetRGBColor(colorLevel);
#else
	rgb = GetRGBColor(1.-colorLevel);
#endif
	//rgb = GetRGBColor(0.8-colorLevel);
	RGBForeColor(&rgb);
	PaintRect(&rgbrect);
	MyFrameRect(&rgbrect);

	MyMoveTo(h+30,v+1);

	RGBForeColor(&colors[BLACK]);
	if (itemNumber<numLevels-1)
	{
		MyNumToStr(x,numstr);
		MyNumToStr((*gEditContourLevels)[itemNumber+1],numstr2);
		strcat(numstr," - ");
		strcat(numstr,numstr2);
	}
	else
	{
		strcpy(numstr,"> ");
		MyNumToStr(x,numstr2);
		strcat(numstr,numstr2);
	}
	if (gContourType==0)
	{
		strcat(numstr,"    mg/L");
	}
	else
	{
		MyNumToStr(x,numstr);
		strcat(numstr,"    m");
	}
	//drawstring(MyNumToStr(x,numstr));
	drawstring(numstr);
 
	return;
}

void VectInit(DialogPtr d, VLISTPTR L)
{
#pragma unused (L)

	SetDialogItemHandle(d, CONT_FRAME_MAKELEVELS, (Handle)FrameBlack);
	SetDialogItemHandle(d, CONT_FRAMEONE, (Handle)FrameBlack);
	
	// code goes here, allow units options mg/L or mL/L (factor of 1000 smaller)
	if (gContourType==0)
	{
		//ShowHideDialogItem(d,CONT_DEFAULT,true);	// reset defaults
		mysetitext(d,CONT_UNITS,"Units : mg/L (ppm)");
	}
	else
	{
		// may want to reset the defaults, but need the depths
		//ShowHideDialogItem(d,CONT_DEFAULT,false);	// don't reset defaults
		mysetitext(d,CONT_UNITS,"Units : meters");
	}

	ShowHideDialogItem(d,CONT_SHOWVALBOX,false);	// don't have help item

	return;
}

void AddOneLevelValue(DialogPtr d, VLISTPTR L)
{
	long index;
	double level = EditText2Float(d,CONT_LEVELVALUE);
	long nElements = GetNumDoubleHdlItems(gEditContourLevels);
	
	if (nElements+1>gMaxNumContours)
	{
		char msg[64]="";
		sprintf(msg,"There is a limit of %ld contour levels",gMaxNumContours);
		printError(msg);
		return;
	}
	
	if((index = AddUniquelySorted(gEditContourLevels,level)) >= 0)
	{
		VLReset(L,GetNumDoubleHdlItems(gEditContourLevels));
		VLSetSelect(index,L); 
		VLAutoScroll(L);
	}
	else
	{
		printError("Value already exists.");
	}
}

void MakeLevels(DOUBLEH contourLevels, double startval, double endval, double interval, Boolean enforceLimit)
{
	// pass in 
	double x;
	double eps = 1e-6;
	//long nElements = GetNumDoubleHdlItems(gEditContourLevels);
	long nElements = GetNumDoubleHdlItems(contourLevels);
	long numNewElements;
	
	if(startval > endval)
	{
		printError("Starting value is greater than final value");
		return;
	}
	
	numNewElements = (endval+interval-eps-startval)/interval+1;
	if (enforceLimit && nElements+numNewElements>gMaxNumContours)
	{
		char msg[64]="";
		sprintf(msg,"There is a limit of %ld contour levels",gMaxNumContours);
		printError(msg);
		return;
	}
	
	if(interval > eps)
	{		
		//for(x = startval; x <= endval + interval; x += interval)
		for(x = startval; x < endval + interval - eps; x += interval)
		{
			AddUniquelySorted(contourLevels,x);
		}
	}
} 

/*OSErr TTriGridVel3D::SetDefaultContours(DOUBLEH contourLevels)
{
	// default values selected by Alan
	_SetHandleSize((Handle)contourLevels,6*sizeof(double));
	//_SetHandleSize((Handle)contourLevels,7*sizeof(double));
	if (_MemError()) { TechError("SetDefaultContours()", "_SetHandleSize()", 0); return -1; }
	//(*contourLevels)[0] = 0;
	(*contourLevels)[0] = 5.;
	(*contourLevels)[1] = 10.;
	(*contourLevels)[2] = 30.;
	//(*contourLevels)[2] = 2;
	(*contourLevels)[3] = 60;
	(*contourLevels)[4] = 100;
	(*contourLevels)[5] = 200;	
	return noErr;
}*/

void MakeContourLevels(DialogPtr d,VLISTPTR L)
{
	long nlevels;
	double startval,endval,interval;

	if(EmptyEditText(d,CONT_INITVALUE) || EmptyEditText(d,CONT_FINALVALUE) || EmptyEditText(d,CONT_INCR))
	{
		printError("All values must be filled in.");
		return;
	}
	
	startval = EditText2Float(d,CONT_INITVALUE);
	endval = EditText2Float(d,CONT_FINALVALUE);
	interval = EditText2Float(d,CONT_INCR);
	MakeLevels(gEditContourLevels,startval,endval,interval,true);
	
	nlevels = GetNumDoubleHdlItems(gEditContourLevels);
	VLReset(L,nlevels);
	VLSetSelect(nlevels-1,L); 
	//VLUpdate(L);
	VLAutoScroll(L);
}

void CopyContourLevels()
{
	if(*gContourLevels)
	{
		MyDisposeHandle((Handle*)gContourLevels);
		*gContourLevels = gEditContourLevels;
		if(_HandToHand((Handle *)gContourLevels))
		{
			printError("Not Enough Memory to copy contour levels");
		}
	}
	else *gContourLevels = gEditContourLevels;
}

Boolean VectClick(DialogPtr d,VLISTPTR L,short itemHit,long *item,Boolean doubleClick)
{
#pragma unused (item)
#pragma unused (doubleClick)
	
	switch(itemHit)
	{
		case CONT_CANCEL:
			MyDisposeHandle((Handle *)&gEditContourLevels);
			return(true);
		case CONT_OKAY:
			if (GetNumDoubleHdlItems(gEditContourLevels)==0)
			{
				printError("You must have at least one contour level");
				return false;
			}
			CopyContourLevels();
			// code goes here, should the dialog edit contours be deleted?
			//sharedCMover->bShowArrows = GetButton(dialog, M16SHOWARROWS);
			model->NewDirtNotification();	// need to refresh concentration scale on the map
			return(true);	
		case CONT_ADD:
			MakeContourLevels(d,L);
			return false;
		case CONT_ADDONE:
			AddOneLevelValue(d,L);
			return false;
		case CONT_DEL:
			DeleteSelectedItem(d,L);
			return(false);
		case CONT_DELALL:
			DeleteAll(d,L);
			return(false);
		case CONT_LEVELVALUE:
			CheckNumberTextItem(d,CONT_LEVELVALUE,true);
			return(false);
		case CONT_INITVALUE:
			CheckNumberTextItem(d,CONT_INITVALUE,true);
			return(false);
		case CONT_FINALVALUE:
			CheckNumberTextItem(d,CONT_FINALVALUE,true);
			return(false);
		case CONT_INCR:
			CheckNumberTextItem(d,CONT_INCR,true);
			return(false);
		case CONT_DEFAULT:
			DeleteAll(d,L);
			SetDefaultContours(gEditContourLevels,gContourType);
			VLReset(L,GetNumDoubleHdlItems(gEditContourLevels));
			VLSetSelect(GetNumDoubleHdlItems(gEditContourLevels)-1,L); 
			VLAutoScroll(L);
			return(false);
		default:
			return(false);
	}
}

OSErr ContourDialog(DOUBLEH *clevels,short contourType)
{
	short	ditem = -1;
	long	selitem;
	
	gContourLevels = clevels;
	gContourType = contourType;
	if (contourType==0)	// subsurface oil concentration
		gMaxNumContours = 12;
	else	// bathymetry - maybe don't want any limit?
		gMaxNumContours = 20;
	gEditContourLevels = (DOUBLEH)_NewHandleClear(0);
	if(*gContourLevels)
	{
		gEditContourLevels = *gContourLevels;
		if(_HandToHand((Handle *)&gEditContourLevels))
		{
			printError("Not enough memory to create temporary contour levels");
			return -1;
		}
	}
		
	selitem=SelectFromVListDialog(
				CONTOUR_DLGID,
				CONT_LISTID,
				GetNumDoubleHdlItems(gEditContourLevels),
				VectInit,
				nil,
				nil,
				VectDraw,
				VectClick,
				true,
				&ditem);
				
	if(ditem == M25OK)
	{
		InvalMapDrawingRect();
	}
	return 0;
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////
// code for depth contours
/////////////////////////////////////////////////
void FindRange(DOUBLEH val, double *minval,double *maxval)
{
	long n = GetNumDoubleHdlItems(val);
	long i;
	double valmin,valmax;
	if(n==0)
	{
		*minval = 1e10;
		*maxval = -1e10;
	}
	valmin = valmax = (*val)[0]; 
	for(i=1;i<n;i++)
	{
		if((*val)[i]<valmin)
		{
			valmin=(*val)[i];
		}
		else if((*val)[i] > valmax)
		{
			valmax = (*val)[i];
		}
	}
	
	*minval = valmin;
	*maxval = valmax;
}

void MyDrawString(/*short dir,*/ short h, short v, char *s,Boolean framed,short selectMode)
{
	Rect r;
	if(sharedPrinting)selectMode = POINTDRAWFLAG;
	if(strlen(s) >0)
	{
		GetTextOffsets(/*dir,*/s,&h,&v);
		GetStringRect(s,h,v,&r);
		if(selectMode == BADPOINTDRAWFLAG)selectMode = POINTDRAWFLAG;
		if(selectMode == POINTDRAWFLAG)
		{
			EraseRect(&r);
			MyMoveTo(h,v);
			drawstring(s);
		}
		if(framed)
		{		
			
			if(selectMode != POINTDRAWFLAG)
			{
				PenMode(patXor);PaintRect(&r);
			}
			PenMode(patCopy);
			MyFrameRect(&r);
		}
	}
}
void DrawContourLevelValue(short x, short y, double level)
{
	char numstr[40];
	RGBColor savecolor;
	MyNumToStr(level,numstr);
	GetForeColor(&savecolor);
	RGBForeColor(&colors[RED]);
	MyDrawString(/*CENTERED,*/x,y,numstr,true,POINTDRAWFLAG);
	RGBForeColor(&savecolor);
}

void TTriGridVel::DrawContourLine(short *ix, short *iy, double** contourValue,Boolean showvals,double level)
{
	long i,j,k,p,p1,p2,p3,count=0;
	long x0,y0,xold,yold;
	long miny0, maxy0, avex0, avey0, ave;
	float trival[3];
	float max,min,diff;
	long ntri = GetNumTriangles();
	TopologyHdl topH = 0;
	if (!contourValue) return;
	if (!fDagTree) return;

	topH = fDagTree->GetTopologyHdl();
	if (!topH) return;

	miny0 = 500;
	maxy0 = -500;
	avey0 = 500;
	for(i=0;i< ntri;i++)
	{
		p1 = (*topH)[i].vertex1;
		p2 = (*topH)[i].vertex2;
		p3 = (*topH)[i].vertex3;
		//GetTriVertices(i,&p1,&p2,&p3);
		trival[0]=(*contourValue)[p1];trival[1]=(*contourValue)[p2];trival[2]=(*contourValue)[p3];
		for(;;)
		{
			if((trival[0]>=trival[1])&&(trival[0]>=trival[2]))
			{
				max=trival[0];
				if(trival[1]<=trival[2]){min=trival[1];}else{min=trival[2];}
				break;
			}
			if((trival[1]>=trival[2])&&(trival[1]>=trival[0]))
			{
				max=trival[1];
				if(trival[2]<=trival[0]){min=trival[2];}else{min=trival[0];}
				break;
			}
			if((trival[2]>=trival[0])&&(trival[2]>=trival[0]))
			{
				max=trival[2];
				if(trival[0]<=trival[1]){min=trival[0];}else{min=trival[1];}
				break;
			}
		}
		if(level<min)continue;
		if(level>max)continue;
		if(max==min){// Don't draw constant contour
			//MyMoveTo(ix[p1],iy[p1]);
			//MyLineTo(ix[p2],iy[p2]);
			//MyLineTo(ix[p3],iy[p3]);
			//MyLineTo(ix[p1],iy[p1]);
			//DrawContourLevelValue((ix[p1]+ix[p2])>>1,(iy[p1]+iy[p2])>>1,level);
		}
		else
		{
			j=0;
			for(k=0;k<3;k++)
			{ 
				trival[0]=(*contourValue)[p1];trival[1]=(*contourValue)[p2];trival[2]=(*contourValue)[p3];
				if(trival[0]>trival[1])
				{
					max=trival[0];min=trival[1];
				}
				else
				{
					max=trival[1];	min=trival[0];
				}
				
				diff=trival[0]-trival[1];
				if((diff==0)&&(max==level))
				{
					MyMoveTo(ix[p1],iy[p1]); MyLineTo(ix[p2],iy[p2]);
					xold = ix[p1]; yold = iy[1]; x0 = ix[2]; y0 = iy[p2];
					avex0=(ix[p1]+ix[p2])>>1;  avey0=(iy[p1]+iy[p2])>>1;
					break;
				}
				if((level<=max)&&(level>min))
				{
					x0=(short)(((float)ix[p1])*(level-trival[1])/diff+((float)ix[p2])*(trival[0]-level)/diff);
					y0=(short)(((float)iy[p1])*(level-trival[1])/diff+((float)iy[p2])*(trival[0]-level)/diff);
					if(y0<miny0) miny0 = y0;
					if(y0>maxy0) maxy0 = y0;
					ave = (maxy0 - miny0)/2 + miny0;
					if(fabs((float)(y0 - ave)) < fabs((float)(avey0 - ave)))
					//if(fabs(y0 - ave) < fabs(avey0 - ave))	// new Windows compiler stricter
					{
						avex0 = x0;avey0 = y0;
					}
					j++;  
					if(j==1)
					{
						MyMoveTo(x0,y0);
						xold = x0;yold = y0;
						if(showvals)
						{
						//	DrawContourLevelValue((xold+x0)>>1, (yold+y0)>>1,level);
						}
					}
					else
					{						
						// may want to be able to set number of labels
						MyLineTo(x0,y0);
						// we are using a legend instead of the labels
						/*if(showvals && (++count % 12 == 0))
						{
							count = 0;
							DrawContourLevelValue((xold+x0)>>1, (yold+y0)>>1,level);

						}*/
						break;
					}
				}
				p=p1;p1=p2;p2=p3;p3=p;
			}
		}
	}
	//if(showvals)DrawContourLevelValue((xold+x0)>>1, (yold+y0)>>1,level);
	//RGBForeColor(savecolor);
	return;
}

void TTriGridVel::DrawContourLines(Boolean printing,DOUBLEH dataVals,Boolean showvals,DOUBLEH contourLevels,short *sxi,short *syi)
{
	RGBColor col;
	double minlevel,maxlevel,level,range;
	long i,nlevels;
	Boolean bDrawBlackAndWhite = settings.printMode == BANDWMODE;

	if (!contourLevels) {printError("no contour levels handle"); return;}
	if (!dataVals) {printError("no data values handle"); return;}
	TextSizeTiny();	
	nlevels = GetNumDoubleHdlItems(contourLevels);
	FindRange(contourLevels,&minlevel,&maxlevel);
	range = maxlevel - minlevel;
	for(i = 0; i< nlevels; i++)
	{
		level = (*contourLevels)[i];
		//if(printing)	// might just check if printing in black and white
		if(printing && bDrawBlackAndWhite)	// might just check if printing in black and white
		{
			RGBForeColor(&colors[BLACK]);
		}
		else
		{
			float colorLevel = float(i)/float(nlevels-1);
#ifdef IBM
			col = GetRGBColor(colorLevel);
#else
			col = GetRGBColor(1.-colorLevel);
#endif
			//col = GetRGBColor((level-minlevel)/range);
			RGBForeColor(&col);
		}
		DrawContourLine(sxi,syi,dataVals,showvals,level);
	}
}

// Create a set of contour levels for contourValues
// Create the handle of contour level values if it doesn't already exist
Boolean AutoContour(DOUBLEH contourValues, DOUBLEH *contourLevels)
{
	double maxval,minval,incr;
	short n;
	FindRange(contourValues,&minval,&maxval);
	minval = floor(minval);
	//maxval = floor(maxval);
	n = floor(log10(fabs(maxval-minval))+.5);
	incr = pow(10.,n-1);
	if(!(*contourLevels = (DOUBLEH)_NewHandleClear(0)))
	{ printError("Not enough memory to copy contours for editing in AutoContour"); return false;}
	MakeLevels(*contourLevels,minval,maxval,incr,false);
	return true;
}

OSErr TTriGridVel::DepthContourDialog()
{
	long i,numDepths;
	OSErr err = 0;
	//DOUBLEH depthsAsDoubleHdl=0;
	//FLOATH theDepths = 0;
	//theDepths = GetDepths();
	//numDepths = GetNumDepths();
	//depthsAsDoubleHdl = (DOUBLEH)_NewHandle(sizeof(double)*numDepths);
	//if (!depthsAsDoubleHdl) {printError("Not enough memory to copy depths"); err = -1; return err;}
	//for (i=0;i<numDepths;i++)
	//{
		//INDEXH(depthsAsDoubleHdl,i) = INDEXH(theDepths,i);
	//}
	if (!fDepthContoursH) 
	{
		//if (!AutoContour(depthsAsDoubleHdl,&fDepthContoursH)) 
		fDepthContoursH = (DOUBLEH)_NewHandleClear(0);
		if(!fDepthContoursH){TechError("TTriGridVel3D::DepthContourDialog()", "_NewHandle()", 0); err = memFullErr; return -1;}
		if (err = SetDefaultContours(fDepthContoursH,1)) 
		{
			//err = -1; 
			return err;
		}
	}
	err = ContourDialog(&fDepthContoursH,1);
	//if (depthsAsDoubleHdl){DisposeHandle((Handle)depthsAsDoubleHdl);depthsAsDoubleHdl = 0;}
	return err;
}

void TTriGridVel::DrawDepthContours(Rect r, WorldRect view, Boolean showLabels)
{
	short *cxi=0, *cyi= 0;
	long i,nv,numDepths;	
	Point p;
	RGBColor saveColor;
	WorldPoint wp;
	LongPointHdl ptsH = 0;
	DOUBLEH depthsAsDoubleHdl = 0;
	Boolean offQuickDrawPlane = false;				
	
	if(fDagTree == 0) return;

	ptsH = fDagTree->GetPointsHdl();
	if (ptsH) nv = _GetHandleSize((Handle)ptsH)/sizeof(**ptsH);
	else return;
	
	//cxi = (short *)_NewPtr(sizeof(short)*nv);
	//cyi = (short *)_NewPtr(sizeof(short)*nv);
	//if (!cxi || !cyi) {printError("Not enough memory in DrawDepthContours"); return;}
	cxi = (short *)calloc(nv,sizeof(short));
	if (cxi==NULL) {printError("Not enough memory in DrawDepthContours"); return;}
	cyi = (short *)calloc(nv,sizeof(short));
	if (cyi==NULL) {printError("Not enough memory in DrawDepthContours"); return;}

	numDepths = GetNumDepths();
	if (nv != numDepths) {printError("The number of depths does not equal the number of vertices. Check your data"); return;}

	depthsAsDoubleHdl = (DOUBLEH)_NewHandle(sizeof(double)*numDepths);
	if (!depthsAsDoubleHdl) {printError("Not enough memory to copy depth values in DrawDepthContours"); return;}

	for(i=nv-1; i>=0; i--)
	{
		INDEXH(depthsAsDoubleHdl,i) = INDEXH(fBathymetryH,i);
		wp.pLong = (*ptsH)[i].h; wp.pLat = (*ptsH)[i].v;
		//p = WorldToScreenPoint(wp,view,r); // or use mapbounds?
		p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);	
		cxi[i] = p.h; cyi[i] = p.v;
	}
	GetForeColor(&saveColor);
	//if (!fDepthContoursH) if (!AutoContour(depthsAsDoubleHdl,&fDepthContoursH)) return;
	if (!fDepthContoursH) 
	{
		fDepthContoursH = (DOUBLEH)_NewHandleClear(0);
		if(!fDepthContoursH){TechError("TTriGridVel::DrawDepthContours()", "_NewHandle()", 0); /*err = memFullErr;*/ return;}
		if (SetDefaultContours(fDepthContoursH,1)) return;
	}
	DrawContourLines(sharedPrinting,depthsAsDoubleHdl,showLabels,fDepthContoursH,cxi,cyi);

	RGBForeColor(&saveColor);
	//if (cxi) _DisposePtr((Ptr)cxi);
	//if (cyi) _DisposePtr((Ptr)cyi);
	if(cxi) {free(cxi); cxi = NULL;}
	if(cyi) {free(cyi); cyi = NULL;}
	if (depthsAsDoubleHdl){DisposeHandle((Handle)depthsAsDoubleHdl);depthsAsDoubleHdl = 0;}
	return;
}

void TTriGridVel3D::DrawDepthContours(Rect r, WorldRect view, Boolean showLabels)
{
	short *cxi=0, *cyi= 0;
	long i,nv,numDepths;	
	Point p;
	RGBColor saveColor;
	WorldPoint wp;
	LongPointHdl ptsH = 0;
	DOUBLEH depthsAsDoubleHdl = 0;
	Boolean offQuickDrawPlane = false;				
	
	if(fDagTree == 0) return;

	ptsH = fDagTree->GetPointsHdl();
	if (ptsH) nv = _GetHandleSize((Handle)ptsH)/sizeof(**ptsH);
	else return;
	
	//cxi = (short *)_NewPtr(sizeof(short)*nv);
	//cyi = (short *)_NewPtr(sizeof(short)*nv);
	//if (!cxi || !cyi) {printError("Not enough memory in DrawDepthContours"); return;}
	cxi = (short *)calloc(nv,sizeof(short));
	if (cxi==NULL) {printError("Not enough memory in DrawDepthContours"); return;}
	cyi = (short *)calloc(nv,sizeof(short));
	if (cyi==NULL) {printError("Not enough memory in DrawDepthContours"); return;}

	numDepths = GetNumDepths();
	if (nv != numDepths) {printError("The number of depths does not equal the number of vertices. Check your data"); return;}

	depthsAsDoubleHdl = (DOUBLEH)_NewHandle(sizeof(double)*numDepths);
	if (!depthsAsDoubleHdl) {printError("Not enough memory to copy depth values in DrawDepthContours"); return;}

	for(i=nv-1; i>=0; i--)
	{
		INDEXH(depthsAsDoubleHdl,i) = INDEXH(fDepthsH,i);
		wp.pLong = (*ptsH)[i].h; wp.pLat = (*ptsH)[i].v;
		//p = WorldToScreenPoint(wp,view,r); // or use mapbounds?
		p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);	
		cxi[i] = p.h; cyi[i] = p.v;
	}
	GetForeColor(&saveColor);
	//if (!fDepthContoursH) if (!AutoContour(depthsAsDoubleHdl,&fDepthContoursH)) return;
	if (!fDepthContoursH) 
	{
		fDepthContoursH = (DOUBLEH)_NewHandleClear(0);
		if(!fDepthContoursH){TechError("TTriGridVel3D::DrawDepthContours()", "_NewHandle()", 0); /*err = memFullErr;*/ return;}
		if (SetDefaultContours(fDepthContoursH,1)) return;
	}
	DrawContourLines(sharedPrinting,depthsAsDoubleHdl,showLabels,fDepthContoursH,cxi,cyi);

	RGBForeColor(&saveColor);
	//if (cxi) _DisposePtr((Ptr)cxi);
	//if (cyi) _DisposePtr((Ptr)cyi);
	if(cxi) {free(cxi); cxi = NULL;}
	if(cyi) {free(cyi); cyi = NULL;}
	if (depthsAsDoubleHdl){DisposeHandle((Handle)depthsAsDoubleHdl);depthsAsDoubleHdl = 0;}
	return;
}
