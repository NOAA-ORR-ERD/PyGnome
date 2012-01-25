#include "CROSS.h"
#include "Graphing.h"
#include "Contdlg.h"

static short gnewplot = 0, prev_h = 0, prev_v = 0, h_axis, v_axis;
static GrafVal	**ggrafhdl = nil;

short GetNumDecPlaces(double* theValue)
{
	short	ndecs=0;

	if (*theValue > 10.0){ ndecs=0; }
	else if (*theValue >= 1.0){ ndecs=1; }
	else if (*theValue >= 0.10){ ndecs=2; }
	else if (*theValue >= 0.010){ ndecs=3; }
	else if (*theValue >= 0.0010){ ndecs=4; }
	else if (*theValue >= 0.00010){ ndecs=5; }
	else{ ndecs=6; }

	return ndecs;
}

void CalcGraphMinMax(GrafValHdl theGraph)
{
	double maxX=0.0, maxY=0.0, minX=0.0, minY=0.0;
	
	// If fields 'xData' and 'yData' do not have information in them, get out of here.
	if ((**theGraph).xData == nil || (**theGraph).yData == nil)
		return;
	// If the data type of the arrays, etc. in 'xData' and 'yData' are not recognized, 
	//    we should also beat feet.
	if ((**theGraph).dataType < 1 || (**theGraph).dataType > 4)
		return;
	// If the min/max values for both X and Y are manually set we should not override
	if ((**theGraph).mmXSetBy == kSetManually && (**theGraph).mmYSetBy == kSetManually)
		return;
	
	if ((**theGraph).numDataElems == -1)
		// Here we will be working with linked lists of data in the near future.
		return;
	// If we are given no number of elements, we also leave
	if ((**theGraph).numDataElems < 1)
		return;
	
	// Proceed to get the min/max values of the X and Y components by calling a specialized
	//   function depending on the data length and type in the arrays.
	switch ((**theGraph).dataType)
	{
		case kShort:
			ArrayMinMaxShort((short*)(**theGraph).xData, (**theGraph).numDataElems, 
					&minX, &maxX);
			ArrayMinMaxShort((short*)(**theGraph).yData, (**theGraph).numDataElems, 
					&minY, &maxY);
			break;
		case kLong:
			ArrayMinMaxLong((long*)(**theGraph).xData, (**theGraph).numDataElems, 
					&minX, &maxX);
			ArrayMinMaxLong((long*)(**theGraph).yData, (**theGraph).numDataElems, 
					&minY, &maxY);
			break;
		case kDouble:
			ArrayMinMaxDouble((double*)(**theGraph).xData, (**theGraph).numDataElems, 
					&minX, &maxX);
			ArrayMinMaxDouble((double*)(**theGraph).yData, (**theGraph).numDataElems, 
					&minY, &maxY);
			break;
		case kFloat:
			ArrayMinMaxFloat((float*)(**theGraph).xData, (**theGraph).numDataElems, 
					&minX, &maxX);
			ArrayMinMaxFloat((float*)(**theGraph).yData, (**theGraph).numDataElems, 
					&minY, &maxY);
			break;
		default:
			return; break;
	}
	
	// If the min/max values have not been set manually, override the current value with
	//   the values just received
	if ((**theGraph).mmXSetBy != kSetManually)
	{
		SetXMinMax(theGraph, &minX, &maxX);
		(**theGraph).mmXSetBy = kSetByDefault;
	}
	if ((**theGraph).mmYSetBy != kSetManually)
	{
		SetYMinMax(theGraph, &minY, &maxY);
		(**theGraph).mmYSetBy = kSetByDefault;
	}

	return;
}

OSErr Coord2GraphPt(GrafValHdl theGraph, double* xValPtr, double* yValPtr, short* xDistPtr, short* yDistPtr)
{
	OSErr	theResult1 = noErr, theResult2 = noErr;
	
	// Convert each value separately and bail out if either conversions returns an error.
	theResult1 = Ext2PixelNum(xValPtr, theGraph, kHorizontal, xDistPtr);
	theResult2 = Ext2PixelNum(yValPtr, theGraph, kVertical, yDistPtr);
	
	if (theResult1)
		return theResult1;
	else
		return theResult2;
}

void DrawAllBorders(GrafValHdl theGraph)
{
	short	i=0;
	
	for (i = 0; i < 4; i++)
	{ 
		DrawBorder(theGraph, i); 
	}

	return;
}

void DrawBorder(GrafValHdl theGraph, short borderElem)
{
	double endH=0.0, endV=0.0, startH=0.0, startV=0.0;
	MyDrawFuncProc	drawingFunc=nil;
	
	// If there is no specified border, skip the rest of the drawing routine.
	if ((**theGraph).theLabels[borderElem].type == 0)
		return;
	
	// Depending on the type of border, preset some of the starting and ending location data.
	// Top and Right borders are not supported yet.
	switch (borderElem)
	{
		case kTopAxis:
			return; break; 

		case kLeftAxis:
			startH = endH = (**theGraph).minXLabVal;
			startV = (**theGraph).minYLabVal;
			endV = (**theGraph).maxYLabVal;
			break;
		case kBottomAxis:
			startH = (**theGraph).minXLabVal;
			endH = (**theGraph).maxXLabVal;
			startV = endV = (**theGraph).minYLabVal;
			break;

		case kRightAxis:
			return; break;
		default:
			return; break;
	}
	
	// Set a pointer to the proper drawing func.
	// Maybe this should be done at set up?
	switch((**theGraph).theLabels[borderElem].type)
	{
		case kProgrammerDefinedBorder:{
			drawingFunc = (**theGraph).theLabels[borderElem].drawingFunc;
			if (drawingFunc == nil){ return; }
			break; }
		case kPlainBorder:{
			drawingFunc = DrawDepth1Line;
			break; }
	}
	// Draw the border line
	DrawLine(theGraph, &startH, &startV, &endH, &endV, drawingFunc);
	
	// Draw the primary and secondary tick marks, if any.
	DrawTicks(theGraph, borderElem, kPrimaryTicks);
	DrawTicks(theGraph, borderElem, kSecondaryTicks);
	
	// Title this border as requested.
	DrawBorderTitle(theGraph, borderElem);

	return;
}

void DrawBorderTitle(GrafValHdl theGraph, short borderElem)
{
	short	rotation, startH, startV;
	TextInfo	theTextInfo;
	char theTitleStr[256];
	FontInfo	theFont;

	// Check to see if there is a title to draw first
	if ((**theGraph).theLabels[borderElem].titleStr[0] == 0)
		return;
	
	// Get all the necessary information to draw the border title.
	//   ie. the size, style, title string, etc.
	theTextInfo = (**theGraph).theLabels[borderElem].titleTextInfo;
	strcpy(theTitleStr, (**theGraph).theLabels[borderElem].titleStr);
	rotation = (**theGraph).theLabels[borderElem].titleRotation;	// would like to use this to rotate left axis label
	
	TextFont(theTextInfo.theFont);
	TextFace(bold);
	TextSize(theTextInfo.theSize);
	GetFontInfo(&theFont);
	
	// Find the starting position based on the type of border we are drawing the label for.
	switch (borderElem)
	{
		case kTopAxis:
			break; 
		case kLeftAxis:
			startV = (**theGraph).theArea.top + (**theGraph).borderDistTop 
					- theFont.ascent - theFont.descent - theFont.leading;
					
			startH = (stringwidth(theTitleStr) / 2);	// eventually try to rotate
			if (startH > (**theGraph).borderDistLeft - 6)
			{ 
				startH = 6; 
			}
			else
			{ 
				startH = (**theGraph).borderDistLeft - startH; 
			}
			startH += (**theGraph).theArea.left;
			MyMoveTo(startH, startV);
			TextDraw(theTitleStr, kLeftJust);
			
			break;
		case kBottomAxis:
			// add in code later to handle title strings that are too long 
			// and actual justification
			
			// left + (right - left) / 2 {round down}
			// don't forget the border distances of the perpendicular axes 
			startH = (**theGraph).theArea.left + (**theGraph).borderDistLeft 
					+ (((**theGraph).theArea.right - (**theGraph).borderDistRight)
					- ((**theGraph).theArea.left + (**theGraph).borderDistLeft)) / 2;
			startV = (**theGraph).theArea.bottom - (theFont.descent + theFont.leading) - 2;
			
			// here only temp, later to be moved to a central location with all the others
			MyMoveTo(startH, startV);
			TextDraw(theTitleStr, kCenterJust);
			
			break; 
		case kRightAxis:
			break;
	}

	TextFace(normal);

	return;
}

void DrawGraph(GrafValHdl theGraphHdl)
{
	short err = 0, height = 0, width = 0, myfont = 0, fontsize = 0;
	Rect theArea, gridrect;
	
	_HLock((Handle)theGraphHdl);

	myfont = ((*theGraphHdl)->graphTitleInfo).theFont;
	fontsize = ((*theGraphHdl)->graphTitleInfo).theSize;

	theArea = (**theGraphHdl).theArea;

	_HUnlock((Handle)theGraphHdl);
	
	StartThinLines();
	DrawGrid(theGraphHdl);
	StopThinLines();

	GetGridRect(theGraphHdl,&gridrect);
	(**theGraphHdl).gridrect = gridrect;

	PlotData(theGraphHdl);	
	
	if ((**theGraphHdl).ShowToxicityThresholds == true)
		PlotOverLay(theGraphHdl);

	DrawAllBorders(theGraphHdl);
	DrawTitle(theGraphHdl);
}

void DrawGrid(GrafValHdl theGraph)
{
	Boolean isVertical=false;
	short	borderElem=0;
	short	i=0;
	double curTickVal=0.0, minLabVal=0.0, maxLabVal=0.0, otherPartEnd=0.0, otherPartStart=0.0;
	double primTickSpread=0.0, tickErrorFactor=0.0, tickRemainder=0.0;
 	PenState	oldPen;
	AxisInfo	theAxis;
	TickInfo	theTicks;
	MyGridFuncProc	drawingFunc=nil;
	
	switch((**theGraph).gridType)
	{
		case kProgrammerDefinedGrid: 
			drawingFunc = (**theGraph).gridDrawFunc;
			break; 
		case kStandardGrid:
			drawingFunc = DrawDepth1Line;
			break; 
		default:
			return; break; 
	}
	if(drawingFunc == nil){ return; }

	_HLock((Handle)theGraph);
	GetPenState(&oldPen);

#ifdef MAC
	//PenPat(&GRAY_BRUSH);
	PenPatQDGlobalsGray();
#else
	PenStyle(GRAY, 1);
#endif

	for (i = 0; i < 8; i++)
	{
		borderElem = (i +.5) / 2;
		if ((**theGraph).theLabels[borderElem].type == 0){ continue; }
		
		theAxis = (**theGraph).theLabels[borderElem];
		if (i % 2 != 0)
		{
			theTicks = theAxis.primaryTicks;
			primTickSpread = 0;
		}
		else
		{
			theTicks = theAxis.secondTicks;
			primTickSpread = theAxis.primaryTicks.spreadVal;
		}
		tickErrorFactor = (**theGraph).theLabels[borderElem].secondTicks.spreadVal / 10;
		
		if (!theTicks.drawGrid){ continue; }
		
		switch (borderElem)
		{
			case kTopAxis:
				isVertical = false;
				minLabVal = (**theGraph).minXLabVal;
				maxLabVal = (**theGraph).maxXLabVal;
				otherPartStart = (**theGraph).minYLabVal;
				otherPartEnd = (**theGraph).maxYLabVal;
				break;
			case kLeftAxis:
				isVertical = true;
				otherPartStart = (**theGraph).minXLabVal;
				otherPartEnd = (**theGraph).maxXLabVal;
				minLabVal = (**theGraph).minYLabVal;
				maxLabVal = (**theGraph).maxYLabVal;
				break;
			case kBottomAxis:
				isVertical = false;
				minLabVal = (**theGraph).minXLabVal;
				maxLabVal = (**theGraph).maxXLabVal;
				otherPartStart = (**theGraph).minYLabVal;
				otherPartEnd = (**theGraph).maxYLabVal;
				break;
			case kRightAxis: 
				otherPartStart = (**theGraph).minXLabVal;
				otherPartEnd = (**theGraph).maxXLabVal;
				minLabVal = (**theGraph).minYLabVal;
				maxLabVal = (**theGraph).maxYLabVal;
				break;
		}
		curTickVal = (short)(minLabVal / theTicks.spreadVal);
		if (curTickVal != (minLabVal / theTicks.spreadVal) && minLabVal > 0){ curTickVal++; }
		
		curTickVal *= theTicks.spreadVal;
		
		RGBForeColor(&colors[BLACK]);
		for ( ; curTickVal <= maxLabVal + tickErrorFactor; curTickVal += theTicks.spreadVal )
		{
			if (primTickSpread > 0)
			{
				tickRemainder = GetRemainder(curTickVal, primTickSpread);
				if((tickRemainder < tickErrorFactor) || (tickRemainder-primTickSpread > -tickErrorFactor)) { continue; }
			}
	
			if (isVertical)
				DrawLine(theGraph, &otherPartStart, &curTickVal, &otherPartEnd, &curTickVal, drawingFunc);
			else
				DrawLine(theGraph, &curTickVal, &otherPartStart, &curTickVal, &otherPartEnd, drawingFunc);
		}
	}

	_HUnlock((Handle)theGraph);
	PenNormal();
	return;
}

void DrawLine(GrafValHdl theGraph, double* startXPtr, double* startYPtr, double* endXPtr, double* endYPtr, 
				MyDrawFuncProc	drawLineFunc)
{
	short	endPixX=0,endPixY=0,startPixX=0,startPixY=0;

	if (drawLineFunc == nil){ return; }

	if (Coord2GraphPt(theGraph, startXPtr, startYPtr, &startPixX, &startPixY)){ return; }
	if (Coord2GraphPt(theGraph, endXPtr, endYPtr, &endPixX, &endPixY)){ return; }

	// now that we have the 'grafPort' coordinates, lets draw the line
	(*drawLineFunc)(startPixX, startPixY, endPixX, endPixY);

	return;
}

void DrawTicks(GrafValHdl theGraph, short borderElem, Boolean isPrimary)
{
	Boolean isVertical=false;
	double curTickVal=0.0, maxLabVal=0.0, minLabVal=0.0, missingCoordPart=0.0;
	double primTickSpread=0.0, tickErrorFactor=0.0, tickRemainder=0.0;
	short	numDecimalPlaces=0, xDist=0, yDist=0;
	MyTickFuncProc	drawingFunc=nil;
	AxisInfo	theAxis;
	TickInfo	theTicks;
	TextInfo	oldText;
	
	if (theGraph==nil || borderElem < 0 || borderElem > 3){ return; }
	
	_HLock((Handle)theGraph);
	theAxis = (**theGraph).theLabels[borderElem];
	if(isPrimary)
	{
		theTicks = theAxis.primaryTicks;
	}
	else
	{
		theTicks = theAxis.secondTicks;
		primTickSpread = theAxis.primaryTicks.spreadVal;
	}

	tickErrorFactor = (**theGraph).theLabels[borderElem].secondTicks.spreadVal / 10.0;
	numDecimalPlaces = GetNumDecPlaces(&theTicks.spreadVal);
		
	switch(theTicks.type)
	{
		case kProgrammerDefinedTicks:
			drawingFunc = theTicks.drawingFunc;
			if (drawingFunc == nil)
			{
				_HUnlock((Handle)theGraph);
				return;
			}
			break;
		case kNoTicks:
			_HUnlock((Handle)theGraph);
			return; break;
		case kPlainTicks:
			drawingFunc = DrawPlainTick;
			break;
		case kSmallPlainTicks:
			drawingFunc = DrawSmallTick;
			break;
	}

	switch (borderElem)
	{
		case kTopAxis:
			isVertical = false;
			minLabVal = (**theGraph).minXLabVal;
			maxLabVal = (**theGraph).maxXLabVal;
			missingCoordPart = (**theGraph).maxYLabVal;
			break;
		case kLeftAxis:
			isVertical = true;
			minLabVal = (**theGraph).minYLabVal;
			maxLabVal = (**theGraph).maxYLabVal;
			missingCoordPart = (**theGraph).minXLabVal;
			break; 
		case kBottomAxis:
			isVertical = false;
			minLabVal = (**theGraph).minXLabVal;
			maxLabVal = (**theGraph).maxXLabVal;
			missingCoordPart = (**theGraph).minYLabVal;
			break; 
		case kRightAxis:
			isVertical = true;
			minLabVal = (**theGraph).minYLabVal;
			maxLabVal = (**theGraph).maxYLabVal;
			missingCoordPart = (**theGraph).maxXLabVal;
			break;
	}

	curTickVal = (double)floor(minLabVal/(theTicks.spreadVal));
	if (curTickVal != (minLabVal / theTicks.spreadVal) && minLabVal > 0.0){ curTickVal++; }
	curTickVal *= (theTicks.spreadVal);
	
	oldText = GetTextType();
	SetTextType(&theTicks.labelTextInfo);
		
	for ( ; curTickVal <= maxLabVal + tickErrorFactor; curTickVal += theTicks.spreadVal )
	{
		if (primTickSpread > 0.0)
		{
			tickRemainder = GetRemainder(curTickVal, primTickSpread);
			if (tickRemainder < 0.0){ tickRemainder *= -1.0; }
			if ((tickRemainder < tickErrorFactor) || ((tickRemainder - primTickSpread) > -tickErrorFactor)){ continue; }
		}

		if (isVertical)
		{
			if (Coord2GraphPt(theGraph,&missingCoordPart,&curTickVal,&xDist,&yDist)){ continue; }
		}
		else
		{
			if (Coord2GraphPt(theGraph,&curTickVal,&missingCoordPart,&xDist,&yDist)){ continue; }
		}

		(*drawingFunc)(borderElem,theTicks.useLabels,xDist,yDist,&curTickVal,numDecimalPlaces);
	}

	SetTextType(&oldText);
	_HUnlock((Handle)theGraph);

	return;
}

void DrawTitle(GrafValHdl theGraphHdl)
{
	Point	centerTextPt;
	FontInfo	theFont;
	TextInfo	theTextInfo;
	char titleStr[128]="",currentinfo[128]="";
	short numhits=0,i=0,j=0,k=0;

	if((**theGraphHdl).graphTitle[0]=='\0'){ return; }
	strcpy(titleStr, (**theGraphHdl).graphTitle);
		
	theTextInfo = (**theGraphHdl).graphTitleInfo;
	TextFont(theTextInfo.theFont);
	TextFace(bold);
	TextSize(theTextInfo.theSize);
	GetFontInfo(&theFont);

	centerTextPt.v = (**theGraphHdl).theArea.top + theTextInfo.theSize + 3;
	centerTextPt.h = (**theGraphHdl).theArea.left + ((**theGraphHdl).theArea.right - (**theGraphHdl).theArea.left) / 2;
	MyMoveTo(centerTextPt.h, centerTextPt.v);
	TextDraw(titleStr, kCenterJust);

	if((**theGraphHdl).graphSubTitle[0]=='\0'){ goto done; }
	strcpy(titleStr, (**theGraphHdl).graphSubTitle);
	MyMoveTo(centerTextPt.h, centerTextPt.v+10);
	TextDraw(titleStr, kCenterJust);


done:

	TextFace(normal);

	return;
}

OSErr Ext2PixelNum(double*	value, GrafValHdl theGraph, Boolean isVertical, short* pixelDist)
{
	double minVal = 0.0, maxVal = 0.0, pixelRangeVal = 0.0, ratio = 0.0, tolerance = 0.0;
	short endPt = 0, startPt = 0;
	OSErr err = 0;
	
	if (isVertical==kVertical)
	{
		startPt = (**theGraph).theArea.bottom - (**theGraph).borderDistBot;
		endPt = (**theGraph).theArea.top + (**theGraph).borderDistTop;

		minVal = (**theGraph).minYLabVal;
		maxVal = (**theGraph).maxYLabVal;
		pixelRangeVal = (double)(startPt - endPt);
	}
	else
	{
		startPt = (**theGraph).theArea.left + (**theGraph).borderDistLeft;
		endPt = (**theGraph).theArea.right - (**theGraph).borderDistRight;

		minVal = (**theGraph).minXLabVal;
		maxVal = (**theGraph).maxXLabVal;
		pixelRangeVal = (double)(endPt - startPt);
	}
	ratio =  pixelRangeVal / (maxVal - minVal);
	tolerance = (maxVal - minVal) / (pixelRangeVal * 10.0);
	
	// check to see if the value to be converted is out of the graphs range
	if (*value < minVal - tolerance || *value > maxVal + tolerance){ err = -1; }

	if (isVertical == kVertical)
		*pixelDist = startPt - (*value - minVal) * ratio;
	else
		*pixelDist = startPt + (*value - minVal) * ratio;

	return err;
}

// this code is not used 
short GetBorderDistance(GrafValHdl theGraph, short fromWhichSide)
{
	short	borderDist = 0;

	if (theGraph == nil){ return kGetBorderError; }

	switch (fromWhichSide)
	{
		case kTopAxis:
			borderDist = (**theGraph).borderDistTop;
			break; 
		case kLeftAxis:
			borderDist = (**theGraph).borderDistLeft;
			break; 
		case kBottomAxis:
			borderDist = (**theGraph).borderDistBot;
			break;
		case kRightAxis:
			borderDist = (**theGraph).borderDistRight;
			break; 
	}
	if (borderDist > 0)
		return borderDist;
	else
		return 0;
}

OSErr GetGraphDataElem(GrafValHdl theGraph, double* xValPtr, double* yValPtr, long elemNum)
{
	OSErr err = 0;
	short	dataType = 0;
	long numElems = 0;

	dataType = (**theGraph).dataType;
	numElems = (**theGraph).numDataElems;

	switch (dataType)
	{
		case kProgrammerDefinedDataType:{ break; }
		case kShort:{
			*xValPtr = ((short*)((**theGraph).xData))[elemNum];
			*yValPtr = ((short*)((**theGraph).yData))[elemNum];
			break; }
		case kLong:{
			*xValPtr = ((long*)((**theGraph).xData))[elemNum];
			*yValPtr = ((long*)((**theGraph).yData))[elemNum];
			break; }
		case kDouble:{
			*xValPtr = ((double*)((**theGraph).xData))[elemNum];
			*yValPtr = ((double*)((**theGraph).yData))[elemNum];
			break; }
		case kFloat:{
			*xValPtr = ((float*)((**theGraph).xData))[elemNum];
			*yValPtr = ((float*)((**theGraph).yData))[elemNum];
			break; }
		default:{ break; }
	}

	return err;
}

TextInfo GetTextType(void)
{
	TextInfo	theTextType;
	GrafPtr thePort = nil;
	memset(&theTextType,0,sizeof(theTextType));
#ifdef MAC
	GetPortGrafPtr(&thePort);
#if TARGET_API_MAC_CARBON
	theTextType.theFont = GetPortTextFont(thePort);
	theTextType.theSize = GetPortTextSize(thePort);
#else
	theTextType.theFont = thePort->txFont;
	theTextType.theSize = thePort->txSize;
#endif
	theTextType.theJust = -2;
#endif	
	return theTextType;
}

GrafValHdl InitGraph(Boolean hasVScroll, Boolean hasHScroll)
{
	short i = 0, err = 0;
	GrafVal **ghdl = nil;
	
	ggrafhdl = nil;

	ghdl=(GrafVal **)_NewHandleClear(sizeof(GrafVal));
	if (err = (short)_MemError()){TechError("InitGraph()","_NewHandle()",0); goto Error;}

	_HLock((Handle)ghdl); ggrafhdl=ghdl;
	
	// the title
	(**ghdl).graphTitle[0] = '\0';
	InitTextInfo(&(**ghdl).graphTitleInfo);
	
	// the area
	MySetRect(&(**ghdl).theArea, 0, 0, 0, 0);
	
	// width and height
	(**ghdl).width = (**ghdl).height = 0;
	
	// the scroll bars
	(**ghdl).hasVertScroll	= hasVScroll;
	(**ghdl).hasHorizScroll	= hasHScroll;
	
	// zero out the grid info
	(**ghdl).islogplot = 0;
	(**ghdl).gridType = 0;
	(**ghdl).gridPat = 0;
	(**ghdl).gridDrawFunc = nil;
	
	// the labels
	for (i=0; i<4; i++){ InitLabel(&(**ghdl).theLabels[i]); }
	
	// min & max values
	(**ghdl).minXValue = (**ghdl).maxXValue = 0.0;
	(**ghdl).mmXSetBy	= 0;
	(**ghdl).minXLabVal = (**ghdl).maxXLabVal	= 0.0;
	(**ghdl).minYValue = (**ghdl).maxYValue = 0.0;
	(**ghdl).mmYSetBy	= 0;
	(**ghdl).minYLabVal = (**ghdl).maxYLabVal = 0.0;

	(**ghdl).ndcmlplcs = 0;
	(**ghdl).islogplot = 0;

	// set data info to zeros
	(**ghdl).numDataElems = 0;
	(**ghdl).dataType = 0;
	(**ghdl).xData = nil;
	(**ghdl).yData = nil;
	(**ghdl).dataPlotFunc = nil;
	(**ghdl).dataOffPlotFunc = nil;

	(**ghdl).multipleCurves = false;

	_HUnlock((Handle)ghdl);

Error:
	return(ghdl);
}

void InitLabel( AxisInfo* theAxis )
{
	theAxis->type = 0;
	theAxis->drawingFunc = nil;
	theAxis->titleStr[0] = 0;
	InitTextInfo(&theAxis->titleTextInfo);
	theAxis->titleRotation = 0;
	InitTickInfo(&theAxis->primaryTicks);
	InitTickInfo(&theAxis->secondTicks);

	return;
}

void InitTextInfo(TextInfo* textInfo)
{
	textInfo->theFont = kFontIDCourier;
	textInfo->theSize = 10;
	textInfo->theJust = kLeftJust;

	return;
}

void InitTickInfo(TickInfo* tickInfo)
{
	tickInfo->type = 0;
	tickInfo->useLabels = false;
	tickInfo->drawGrid = false;
	tickInfo->spreadVal = 0;
	tickInfo->iconToUse = 0;
	InitTextInfo(&tickInfo->labelTextInfo);

	return;
}

void SetToxicityThresholds(GrafValHdl theGraph, OverLayInfo toxicityInfo, short overLayType)
{
	long i;
	for (i=0;i<3;i++)
	{
		(**theGraph).ToxicityLevels[overLayType].xOverLay[i] = toxicityInfo.xOverLay[i];
		(**theGraph).ToxicityLevels[overLayType].yOverLay[i] = toxicityInfo.yOverLay[i];
	}
	(**theGraph).ToxicityLevels[overLayType].showOverLay = toxicityInfo.showOverLay;
	strcpy((**theGraph).ToxicityLevels[overLayType].labelStr,toxicityInfo.labelStr);
	return;
}

OverLayInfo GetToxicityThresholds(GrafValHdl theGraph, short overLayType)
{
	long i;
	OverLayInfo toxicityInfo;
	for (i=0;i<3;i++)
	{
		toxicityInfo.xOverLay[i] = (**theGraph).ToxicityLevels[overLayType].xOverLay[i];
		toxicityInfo.yOverLay[i] = (**theGraph).ToxicityLevels[overLayType].yOverLay[i];
	}
	toxicityInfo.showOverLay = (**theGraph).ToxicityLevels[overLayType].showOverLay;
	strcpy(toxicityInfo.labelStr,(**theGraph).ToxicityLevels[overLayType].labelStr);
	return toxicityInfo;
}

void IntDraw(Point textPt, double *val, short just)
{
	char	str[64]="";

	sprintf(str,"%.0lf",(*val));

	MyMoveTo(textPt.h,textPt.v);
	TextDraw(str,just);

	return;
}

void DblDraw(short ndcmlplcs, Point textPt, double *val, short just)
{
	char	str[64]="";

	switch(ndcmlplcs){
		case -1:{
			sprintf(str,"%.0lf",(*val)); break; }
		case 0:{
			sprintf(str,"%.0lf",(*val)); break; }
		case 1:{
			sprintf(str,"%.1lf",(*val)); break; }
		case 2:{
			sprintf(str,"%.2lf",(*val)); break; }
		case 3:{
			sprintf(str,"%.3lf",(*val)); break; }
		case 4:{
			sprintf(str,"%.4lf",(*val)); break; }
		case 5:{
			sprintf(str,"%.5lf",(*val)); break; }
		case 6:{
			sprintf(str,"%.6lf",(*val)); break; }
	}

	MyMoveTo(textPt.h,textPt.v);
	TextDraw(str,just);

	return;
}

void DblExpDraw(short ndcmlplcs,Point textPt,double *val,short just)
{
	char	str[64]="";

	switch(ndcmlplcs){
		case -1:
		case 0:{
			sprintf(str,"%.0le",(*val)); break; }
		case 1:{
			sprintf(str,"%.1le",(*val)); break; }
		case 2:{
			sprintf(str,"%.2le",(*val)); break; }
		case 3:{
			sprintf(str,"%.3le",(*val)); break; }
		case 4:{
			sprintf(str,"%.4le",(*val)); break; }
		case 5:{
			sprintf(str,"%.5le",(*val)); break; }
		case 6:{
			sprintf(str,"%.6le",(*val)); break; }
	}

	MyMoveTo(textPt.h,textPt.v);
	TextDraw(str,just);

	return;
}

void PlotData(GrafValHdl theGraphHdl)
{
	double xVal=0.0, yVal=0.0, prevXVal=0.0, prevYVal=0.0;
	GrafVal *theGraphPtr=nil;
	MyPlotProc dataPlotFunc;
	MyOffPlotProc dataOffPlotFunc;
	Rect theArea;
	long i=0, numElems=0;
	short	dataType=0;
	OSErr err=0;
	Boolean moveOnly=true, prevDataPlotted=false;
	// maybe want some option to plot individual points, not connect them
	// use mydrawstring and put an x there
				//MyDrawString(/*CENTERED,*/x,y,"x",false,POINTDRAWFLAG);
	
	// set the global (for no duplicate point plotting)
	//gnewplot=1; prev_h=0; prev_v=0;
	gnewplot=0; prev_h=0; prev_v=0;
	h_axis = (**theGraphHdl).theArea.bottom - (**theGraphHdl).borderDistBot;
	v_axis = (**theGraphHdl).theArea.top + (**theGraphHdl).borderDistTop;

	_HLock((Handle)theGraphHdl); theGraphPtr = (*theGraphHdl);
	dataType = theGraphPtr->dataType;
	numElems = theGraphPtr->numDataElems;
	theArea = theGraphPtr->theArea;
	
   if ((dataType != kProgrammerDefinedDataType && (dataType < 1 || dataType > 4)) 
         || (numElems < 0 && (dataType > 0 || dataType < 5)))
      goto PlotDataOUT;
   
   if (theGraphPtr->dataPlotFunc)
   {
      dataPlotFunc = theGraphPtr->dataPlotFunc;
      dataOffPlotFunc = theGraphPtr->dataOffPlotFunc;
   }
   else
   {
      dataPlotFunc = StandardPlot;
      dataOffPlotFunc = StandardOffPlot;
   }

	if ((**theGraphHdl).multipleCurves) numElems = numElems/2;	// at this point can only have 2 

	for (i=0, moveOnly=true; i<numElems; i++)
	{
		if (GetGraphDataElem(theGraphHdl, &xVal, &yVal, i)){ goto PlotDataOUT; }
      
		if (prevDataPlotted && moveOnly==false){ err = (*dataPlotFunc)(&xVal, &yVal, i); }
      
		if (err || moveOnly)
		{
			if (i > 0)
				if (GetGraphDataElem(theGraphHdl, &prevXVal, &prevYVal, i-1)){ goto PlotDataOUT; }
			else
				moveOnly = true; 

			err = (*dataOffPlotFunc)(&prevXVal, &prevYVal, &xVal, &yVal, moveOnly, i);
			if (err)
				prevDataPlotted = false; 
			else 
				prevDataPlotted = true; 

			moveOnly = false;
      }
      else
			prevDataPlotted = true; 
   }
	if ((**theGraphHdl).multipleCurves)
	{
		SetDataPlotType(theGraphHdl,(MyPlotProc)StandardPlot3,(MyOffPlotProc)StandardOffPlot3);
      dataPlotFunc = theGraphPtr->dataPlotFunc;
      dataOffPlotFunc = theGraphPtr->dataOffPlotFunc;
		for (i=numElems, moveOnly=true; i<2*numElems; i++)
		{
			if (GetGraphDataElem(theGraphHdl, &xVal, &yVal, i)){ goto PlotDataOUT; }
			if (prevDataPlotted && moveOnly==false){ err = (*dataPlotFunc)(&xVal, &yVal, i); }
			
			if (err || moveOnly)
			{
				if (i > numElems)
				{
					if (GetGraphDataElem(theGraphHdl, &prevXVal, &prevYVal, i-1)){ goto PlotDataOUT; }
				}
				else
					moveOnly = true; 
	
				err = (*dataOffPlotFunc)(&prevXVal, &prevYVal, &xVal, &yVal, moveOnly, i);
				if (err)
					prevDataPlotted = false; 
				else 
					prevDataPlotted = true; 
	
				moveOnly = false;
			}
			else
				prevDataPlotted = true; 
		}
	}
PlotDataOUT:

	_HUnlock( (Handle)theGraphHdl );

	return;
}

long GetMaxLen()
{
	long maxlen = stringwidth("Toxicity Thresholds");
	return  maxlen;
}

void PlotOverLay(GrafValHdl theGraphHdl)
{
	long i,j,numOverLays=0,len,maxLen=0;
	RGBColor	saveColor;
	PenState pensave;
	OverLayInfo toxicityInfo;
	OSErr err = 0;
	short xDistStart, yDistStart, xDistEnd, yDistEnd, mycol[3] = {12,15,9};
	Point	labelTextPt;
	char labelStr[64], levelOfConcernStr[10];
	Rect labelRect;
	
	GetForeColor (&saveColor);		// save original forecolor
	// Pen Info 
	GetPenState(&pensave);
	PenNormal();
	PenSize(1,1);
#ifdef MAC
	//PenSize(2,2);
	//PenPat((ConstPatternParam)&qd.gray);	// dashed lines
#else
	//PenStyle(BLACK,2);
	//FillPat(DARKGRAY);
#endif
	// may want to use different line types instead of colors - DASHDOTDOT	
	maxLen = GetMaxLen(); // this should check the data
	//maxLen = stringwidth("Toxicity Thresholds");
	//maxLen = stringwidth("for Adult Crustaceans");
	for (i=0;i<3;i++)	// numOverlays
	{
		toxicityInfo.showOverLay = (**theGraphHdl).ToxicityLevels[i].showOverLay;
		if (!toxicityInfo.showOverLay) continue;
		strcpy(toxicityInfo.labelStr,(**theGraphHdl).ToxicityLevels[i].labelStr);
		//len = stringwidth(toxicityInfo.labelStr) + stringwidth("--- ");
		len = stringwidth(toxicityInfo.labelStr) + stringwidth("___ ");
		if (len>maxLen) maxLen = len;
		numOverLays++;
	}
	if (numOverLays==0) return;
	//RGBForeColor(&colors[RED]);
	labelTextPt.v = (**theGraphHdl).theArea.top + (**theGraphHdl).borderDistTop /*+ theTextInfo.theSize*/ + 10;
	strcpy(labelStr,"Toxicity Thresholds");
	labelTextPt.h = (**theGraphHdl).theArea.right - (**theGraphHdl).borderDistRight - maxLen - 2;
	MyMoveTo(labelTextPt.h, labelTextPt.v);
	TextDraw(labelStr, kLeftJust);
	labelTextPt.v += 10;

	strcpy(labelStr,"for ");
	strcat(labelStr,(**theGraphHdl).LegendStr);
	labelTextPt.h = (**theGraphHdl).theArea.right - (**theGraphHdl).borderDistRight - maxLen - 2;
	MyMoveTo(labelTextPt.h, labelTextPt.v);
	TextDraw(labelStr, kLeftJust);
	labelTextPt.v += 15;

	for (i=0;i<3;i++)	
	{
		RGBForeColor(&colors[mycol[i]]);
		toxicityInfo.xOverLay[0] = (**theGraphHdl).ToxicityLevels[i].xOverLay[0];
		toxicityInfo.yOverLay[0] = (**theGraphHdl).ToxicityLevels[i].yOverLay[0];
		toxicityInfo.showOverLay = (**theGraphHdl).ToxicityLevels[i].showOverLay;
		strcpy(toxicityInfo.labelStr,(**theGraphHdl).ToxicityLevels[i].labelStr);
		if (!toxicityInfo.showOverLay) continue;
		err = Ext2PixelNum(&toxicityInfo.xOverLay[0], theGraphHdl, kHorizontal, &xDistStart);
		err = Ext2PixelNum(&toxicityInfo.yOverLay[0], theGraphHdl, kVertical, &yDistStart);
		MyMoveTo(xDistStart,yDistStart);
		for (j=0;j<2;j++)
		{
			toxicityInfo.xOverLay[j+1] = (**theGraphHdl).ToxicityLevels[i].xOverLay[j+1];
			toxicityInfo.yOverLay[j+1] = (**theGraphHdl).ToxicityLevels[i].yOverLay[j+1];
			err = Ext2PixelNum(&toxicityInfo.xOverLay[j+1], theGraphHdl, kHorizontal, &xDistEnd);
			err = Ext2PixelNum(&toxicityInfo.yOverLay[j+1], theGraphHdl, kVertical, &yDistEnd);
			MyLineTo(xDistEnd,yDistEnd);
		}
		//strcpy(labelStr,"--- ");
		strcpy(labelStr,"___ ");
		strcat(labelStr,toxicityInfo.labelStr);
		labelTextPt.h = (**theGraphHdl).theArea.right - (**theGraphHdl).borderDistRight - maxLen - 2;
		MyMoveTo(labelTextPt.h, labelTextPt.v);
		TextDraw(labelStr, kLeftJust);
		labelTextPt.v += 10;
	}
	RGBForeColor(&colors[BLACK]);
	PenNormal();

	labelRect.left = labelTextPt.h - 4;
	labelRect.right = labelTextPt.h + maxLen + 4;
	labelRect.top = labelTextPt.v - (numOverLays+1)*10 - 25;
	labelRect.bottom = labelTextPt.v - 5;
	MyFrameRect(&labelRect);

	SetPenState(&pensave);
	RGBForeColor(&saveColor);
}

void SetAxisFontData(GrafValHdl theGraph, short labelElem, short theFont, short theSize, short theJustification)
{	
	if (!theGraph || labelElem > 4 || labelElem < 1)
		return;
	
	(**theGraph).theLabels[labelElem].titleTextInfo.theFont = theFont;
	(**theGraph).theLabels[labelElem].titleTextInfo.theSize = theSize;
	(**theGraph).theLabels[labelElem].titleTextInfo.theJust = theJustification;

	return;
}

void SetAxisTypeData(GrafValHdl theGraph,short borderNum,short axisType,MyDrawFuncProc drawingFunc)
{
	if (theGraph == nil){ return; }
	
	(**theGraph).theLabels[borderNum].type = axisType;
	(**theGraph).theLabels[borderNum].drawingFunc = drawingFunc;

	return;
}

void SetBorderDistance(GrafValHdl theGraph, short	fromTop, short	fromLeft, short fromBottom, short fromRight)
{
	if (theGraph == nil){ return; }
	
	if (fromTop >= 0)
		(**theGraph).borderDistTop = fromTop;
	if (fromLeft >= 0)
		(**theGraph).borderDistLeft = fromLeft;
	if (fromBottom >= 0)
		(**theGraph).borderDistBot = fromBottom;
	if (fromRight >= 0)
		(**theGraph).borderDistRight = fromRight;

	return;
}

void SetBorderTitle(GrafValHdl theGraph, short borderElem, char *theTitleStr, short	theFont, short theSize, 
						short theJustification, short rotation)
{
	TextInfoPtr theTextPtr;
	
	theTextPtr = &(**theGraph).theLabels[borderElem].titleTextInfo;
	theTextPtr->theFont = theFont;
	theTextPtr->theSize = theSize;
	theTextPtr->theJust = theJustification;
	
	if (strlen(theTitleStr) < 256)
	{
		strcpy((**theGraph).theLabels[borderElem].titleStr, theTitleStr);
	}
	else
	{
		printError("String exceeds limit in SetBorderTitle()");
	}
	
	(**theGraph).theLabels[borderElem].titleRotation = rotation;

	return;
}

void SetData(GrafValHdl theGraph, Ptr xDataPtr, Ptr yDataPtr, long numDataElems, 
		short	dataType, Boolean calcDefaultMM)
{
	if (xDataPtr == nil)
	{
		printError("X axis data is not available in SetData()");
		return;
	}
	if (yDataPtr == nil)
	{
		printError("Y axis data is not available in SetData()");
		return;
	}
	if (numDataElems == 0 && dataType != kProgrammerDefinedDataType)
	{
		printError("Number of elements is not available in SetData()");
		return;
	}

	if ((dataType > 0 && dataType < 5) || dataType == kProgrammerDefinedDataType)
	{
		if ((**theGraph).xData)
		{
			_DisposePtr( (**theGraph).xData );
			(**theGraph).xData = nil;
		}
		if ((**theGraph).yData)
		{
			_DisposePtr((**theGraph).yData);
			(**theGraph).yData = nil;
		}
			
		(**theGraph).xData = xDataPtr;
		(**theGraph).yData = yDataPtr;
		(**theGraph).dataType = dataType;
		// Set the data conversion function pointers if any (hopefully?) valid ones were provided.
		// If the programmer passed a special data type that is not normally part of the package, 
		//   we may be expected to call the function passed to get the number of data elements.
		if (dataType > 0)
		{ 
			(**theGraph).numDataElems = numDataElems; 
		}
	}
	else
	{
		printError("Data type not defined in SetData()");
		return;
	}

	if (calcDefaultMM){ CalcGraphMinMax(theGraph); }

	return;
}

void SetDataPlotType(GrafValHdl theGraphHdl, MyPlotProc dataPlotFunc, MyOffPlotProc dataOffPlotFunc)
{
	(**theGraphHdl).dataPlotFunc = dataPlotFunc;
	(**theGraphHdl).dataOffPlotFunc = dataOffPlotFunc;

	return;
}

void SetGraphArea(GrafValHdl theGraphHdl, Rect* theDimensions)
{
	(**theGraphHdl).theArea = *theDimensions;
	(**theGraphHdl).width = theDimensions->right - theDimensions->left;
	(**theGraphHdl).height = theDimensions->bottom - theDimensions->top;

	return;
}

void SetGraphTitle(GrafValHdl	theGraphHdl, char *titleStr, char *subTitleStr, short theFont, short theSize)
{
	if (titleStr[0] == '\0' || titleStr == nil){ return; }

	strcpy((**theGraphHdl).graphTitle, titleStr);
	(**theGraphHdl).graphTitleInfo.theFont = theFont;
	(**theGraphHdl).graphTitleInfo.theSize = theSize;

	if (subTitleStr[0] == '\0' || subTitleStr == nil){ return; }
	strcpy((**theGraphHdl).graphSubTitle, subTitleStr);
	return;
}

void SetGrid(GrafValHdl theGraph, short gridType, MyGridFuncProc drawingFunc, short gridPat)
{
	if (gridType < 0){ return; }
	(**theGraph).gridType = gridType;
	
	switch(gridType)
	{
		case kProgrammerDefinedGrid:{
			(**theGraph).gridDrawFunc = drawingFunc;	// may not need this if always using default
			break; }

		default:{ break; }
	}

	if (gridPat < 0){ gridPat = 0; }

	(**theGraph).gridPat = gridPat;

	return;
}

void SetTextType(TextInfo* theTextType)
{
#ifdef MAC
	TextFont(theTextType->theFont);
	TextSize(theTextType->theSize);
	TextFace(normal);
#endif
	return;
}

void SetTickInfo(GrafValHdl theGraph, short labelNum, Boolean isPrimary, Boolean useLabels,
					Boolean drawGrid, short type, double spreadVal, short iconToUse, TextInfo *labelText)
{
	if (labelNum < 0 || labelNum > 5)
	{
		printError("Label out of range in SetTickInfo()");
		return;
	}
	
	_HLock((Handle)theGraph);
	if (isPrimary)
	{
		((**theGraph).theLabels[labelNum]).primaryTicks.type = type;
		((**theGraph).theLabels[labelNum]).primaryTicks.useLabels = useLabels;
		((**theGraph).theLabels[labelNum]).primaryTicks.drawGrid	= drawGrid;
		((**theGraph).theLabels[labelNum]).primaryTicks.spreadVal = spreadVal;
		((**theGraph).theLabels[labelNum]).primaryTicks.iconToUse = iconToUse;

		// if 'nil' was sent as the textInfo, leave the default
		if (labelText != kDefaultSetting)
		{
			(**theGraph).theLabels[labelNum].primaryTicks.labelTextInfo = *labelText;
		}
	}
	else
	{
		((**theGraph).theLabels[labelNum]).secondTicks.type = type;
		((**theGraph).theLabels[labelNum]).secondTicks.useLabels	= useLabels;
		((**theGraph).theLabels[labelNum]).secondTicks.drawGrid = drawGrid;
		((**theGraph).theLabels[labelNum]).secondTicks.spreadVal	= spreadVal;
		((**theGraph).theLabels[labelNum]).secondTicks.iconToUse	= iconToUse;
	
		// if 'nil' was sent as the textInfo, leave the default
		if (labelText != kDefaultSetting)
		{
			((**theGraph).theLabels[labelNum]).secondTicks.labelTextInfo = *labelText;
		}
	}

	_HUnlock((Handle)theGraph);

	return;
}

void SetXMinMax(GrafValHdl theGraph, double* min, double* max)
{
	double spreadVal=0.0;
	long maxTimesSpread=0, minTimesSpread=0;
	
	spreadVal=((**theGraph).theLabels[kBottomAxis]).primaryTicks.spreadVal;

	(**theGraph).minXValue = 0.0;
	(**theGraph).maxXValue = 0.0;

	if (*min > *max)
	{
		printError("Min greater than Max in SetXMinMax()");
		return;
	}
	
	(**theGraph).minXValue = *min;
	(**theGraph).maxXValue = *max;
	(**theGraph).mmXSetBy = kSetManually;
	if (spreadVal > 0.0)
	{
		minTimesSpread = (long)floor((*min)/spreadVal);
		maxTimesSpread = (long)floor((*max)/spreadVal);
		
		if ((((double)minTimesSpread) != ((*min)/spreadVal)) && ((*min)<0.0)){ minTimesSpread--; }
		if ((((double)maxTimesSpread) != ((*max)/spreadVal)) && ((*max)>0.0)){ maxTimesSpread++; }

		(**theGraph).minXLabVal = ((double)minTimesSpread) * spreadVal;
		(**theGraph).maxXLabVal = ((double)maxTimesSpread) * spreadVal;
	}
	
	// Check to see if the x value is a constant and is equal to the minimum or maximum label
	//   value.  If so change that min or max value to show the constant value.
	if((**theGraph).minXValue == (**theGraph).maxXValue)
	{
		if ((**theGraph).minXValue == (**theGraph).minXLabVal)
			(**theGraph).minXLabVal -= spreadVal;
		if ((**theGraph).maxXValue == (**theGraph).maxXLabVal)
			(**theGraph).maxXLabVal += spreadVal;
	}

	return;
}

void SetXMinMaxLong(GrafValHdl theGraph, long min, long max)
{
	double newMin, newMax;
	
	newMin = min;
	newMax = max;
	SetXMinMax(theGraph, &newMin, &newMax);

	return;
}

void SetYMinMax(GrafValHdl theGraph, double *min, double *max)
{
	double spreadVal=0.0;
	long maxTimesSpread=0, minTimesSpread=0;
	
	spreadVal=((**theGraph).theLabels[kLeftAxis]).primaryTicks.spreadVal;

	(**theGraph).minYValue = 0.0;
	(**theGraph).maxYValue = 0.0;

	if (*min > *max)
	{
		printError("Min greater than Max in SetYMinMax()");
		return;
	}
	
	(**theGraph).minYValue	= *min;
	(**theGraph).maxYValue	= *max;
	(**theGraph).mmYSetBy	= kSetManually;
	if (spreadVal > 0.0)
	{
		minTimesSpread = (long)floor((*min)/spreadVal);
		maxTimesSpread = (long)floor((*max)/spreadVal);
		
		if ( ( ((double)minTimesSpread) != ((*min)/spreadVal) ) && ((*min)<0.0) ){ minTimesSpread--; }
		if ( ( ((double)maxTimesSpread) != ((*max)/spreadVal) ) && ((*max)>0.0) ){ maxTimesSpread++; }

		(**theGraph).minYLabVal = ((double)minTimesSpread) * spreadVal;
		(**theGraph).maxYLabVal = ((double)maxTimesSpread) * spreadVal;
	}
	
	// Check to see if the y value is a constant and is equal to the minimum or maximum label
	//   value.  If so change that min or max value to show the constant value.
	if((**theGraph).minYValue == (**theGraph).maxYValue)
	{
		if ((**theGraph).minYValue == (**theGraph).minYLabVal)
			(**theGraph).minYLabVal -= spreadVal;
		if ((**theGraph).maxYValue == (**theGraph).maxYLabVal)
			(**theGraph).maxYLabVal += spreadVal;
	}

	return;
}

void SetYMinMaxLong(GrafValHdl theGraph, long min, long max)
{
	double newMin, newMax;
	
	newMin = min;
	newMax = max;
	SetYMinMax(theGraph, &newMin, &newMax);

	return;
}

double GetRemainder(double X, double Y)
{
	double realQuotient=0.0;
	short	intQuotient=0;

	if(Y==0.0){ return(realQuotient); }

	realQuotient = (X / Y);
	intQuotient = realQuotient;
	// If the quotient is negative, the number of integer quotient is not rounded right, so
	//   '1' is added.
	
	// if realQuotient is an integer return '0'
	if (realQuotient - intQuotient == 0)
		return ((double)0.0);
	
	return((realQuotient - intQuotient) * Y);
}

void GetGridRect(GrafValHdl grafHdl,Rect *theRect)
{
	theRect->top = (**grafHdl).theArea.top + (**grafHdl).borderDistTop;
	theRect->left = (**grafHdl).theArea.left + (**grafHdl).borderDistLeft;
	theRect->bottom = (**grafHdl).theArea.bottom - (**grafHdl).borderDistBot;
	theRect->right	= (**grafHdl).theArea.right - (**grafHdl).borderDistRight;
}


/////////////////////////////////////////////////
// code not used
/*OSErr DataPlot1DepthBlue(GrafValHdl	theGraphHdl, double *xValPtr, double *yValPtr, long pntnumber)
{
	OSErr err = 0;
	RGBForeColor(&colors[BLUE]);
	PenSize(1,1);
	err = StandardPlot(xValPtr, yValPtr, pntnumber);
	RGBForeColor(&colors[BLACK]);
	return err;
}

OSErr DataOffPlot1DepthBlue(GrafValHdl theGraphHdl, double *prevXValPtr, double *prevYValPtr, double *xValPtr, double *yValPtr, Boolean moveOnly, long pntnumber)
{
	OSErr err = 0;
	
	RGBForeColor(&colors[BLUE]);
	PenSize(1,1);
	err = StandardOffPlot(prevXValPtr, prevYValPtr, xValPtr, yValPtr, moveOnly, pntnumber);
	RGBForeColor(&colors[BLACK]);
	return err;
}*/
/////////////////////////////////////////////////

void DrawDepth1Line(short xStart, short yStart, short xEnd, short yEnd)
{
	PenSize(1, 1);
	MyMoveTo(xStart, yStart);
	MyLineTo(xEnd, yEnd);

	return;
}

void DrawPlainTick(short borderElem, Boolean useLabels, short startH, short startV, double *tickVal, short numDecPos)
{
	short just=0;
	double val=0.0;
	FontInfo	theFont;
	Point end,start,textPt;
	short myfont=0,fontsize=0,ndcmlplcs=0,power=0;
	
	myfont=((*ggrafhdl)->graphTitleInfo).theFont;
	fontsize=((*ggrafhdl)->graphTitleInfo).theSize;
	ndcmlplcs=(**ggrafhdl).ndcmlplcs;
	
	GetFontInfo(&theFont);
	switch (borderElem)
	{
		case kTopAxis:{ return; break; }
		case kLeftAxis:{
			start.v = end.v = startV;
			start.h = startH;
			end.h = startH - 7;
			textPt.h = end.h - 5;
			textPt.v = startV + (short)(theFont.ascent / 2 + .5 - 1);
			just = kRightJust;
			break; }
		case kBottomAxis:{
			start.h = end.h = textPt.h = startH;
			start.v = startV;
			end.v = startV + 7;
			textPt.v = end.v + fontsize;
			just = kCenterJust;
			//numDecPos = 0;
			break; }
		case kRightAxis:{ return; break; }
		default:{
			printError("Axis type not defined in DrawPlainTick()");
			return; break; }
	}
	
	DrawDepth1Line(start.h,start.v,end.h,end.v);
	if (useLabels)
	{
		if (borderElem==kBottomAxis) 
		{
			//IntDraw(textPt,tickVal,just);
			DblDraw(numDecPos,textPt,tickVal,just);
		}
		else
		{
			if((**ggrafhdl).islogplot == 0)
			{
				DblDraw(ndcmlplcs,textPt,tickVal,just);
			}
			else
			{
				val=(*tickVal);
				power = (short)val; val=10.0;
				Raise2Pow(&val,&val,power);
				DblExpDraw(1,textPt,&val,just);
			}
		}
	}

	return;
}

void DrawSmallTick(short borderElem, Boolean useLabels, short startH, short startV, double *tickVal, short numDecPos)
{
	short just=0;
	Point end, start, textPt;
	FontInfo	theFont;
	
	GetFontInfo(&theFont);
	switch (borderElem)
	{
		case kTopAxis:{ return; break; }
		case kLeftAxis:{
			start.v = end.v = startV;
			start.h = startH;
			end.h = startH - 3;
			textPt.h = end.h - 5;
			textPt.v = startV + (short)(theFont.ascent / 2 + .5 - 1);
			just = kRightJust;
			break; }
		case kBottomAxis:{
			start.h = end.h = textPt.h = startH;
			start.v = startV;
			end.v = startV + 3;
			textPt.v = end.v + theFont.ascent + theFont.descent + theFont.leading;
			just = kCenterJust;
			break; }
		case kRightAxis:{ return; break; }
		default:{
			printError("Axis type not defined in DrawSmallTick()");
			return; break; }
	}
	
	DrawDepth1Line(start.h, start.v, end.h, end.v);

	return;
}

OSErr StandardPlot2(double* xPtr, double* yPtr, long pntnum)
{
	OSErr	err=0;
	short h=0, v=0;
	double newY = *yPtr+1;
	err = Coord2GraphPt(ggrafhdl,xPtr,yPtr,&h,&v); if(err){ goto Error; }

	if (gnewplot == 0)
	{
		// check if this point equal last point
		if (h==prev_h && v==prev_v){ goto Error; }
	}
	else
	{ 	
		gnewplot=0; 
	}

	PenSize(1,1); 
/*#ifdef MAC
	PenSize(2,2);
#else
	PenStyle(BLACK,2);
#endif*/
	//RGBForeColor(&colors[BLACK]);
	RGBForeColor(&colors[RED]);

	//option to only draw points, not lines
	MyMoveTo(h,v);
	 prev_h=h; prev_v=v;
	MyDrawString(/*CENTERED,*/h,v,"x",false,0);
		//MyMoveTo(h,v);
		//err = Coord2GraphPt(ggrafhdl,xPtr,&newY,&h,&v);
		//MyLineTo(h,v);
		//MyDrawString(/*CENTERED,*/h,v,"x",false,0);
	//MyLineTo(h,v); prev_h=h; prev_v=v;
	RGBForeColor(&colors[BLACK]);

Error:

#ifdef MAC
	PenSize(1,1);
#else
	PenStyle(BLACK,1);
#endif
	return err;
}

OSErr StandardPlot(double* xPtr, double* yPtr, long pntnum)
{
	OSErr	err=0;
	short h=0, v=0;
	
	err = Coord2GraphPt(ggrafhdl,xPtr,yPtr,&h,&v); if(err){ goto Error; }

	if (gnewplot == 0)
	{
		// check if this point equal last point
		if (h==prev_h && v==prev_v){ goto Error; }
	}
	else
	{ 	
		gnewplot=0; 
	}

#ifdef MAC
	PenSize(1,1);
	//PenSize(2,2);
#else
	PenStyle(BLACK,1);
	//PenStyle(BLACK,2);
#endif
	RGBForeColor(&colors[BLACK]);
	//RGBForeColor(&colors[RED]);

	//option to only draw points, not lines
	//MyMoveTo(h,v);
	//MyDrawString(/*CENTERED,*/h,v,"x",false,0);
	MyLineTo(h,v); prev_h=h; prev_v=v;
	//RGBForeColor(&colors[BLACK]);

Error:

	return err;
}

OSErr StandardPlot4(double* xPtr, double* yPtr, long pntnum)
{
	OSErr	err=0;
	short h=0, v=0;
	Point p;
	
#ifdef IBM
	POINT points[4];
#else
	PolyHandle poly;
#endif

	GetPen(&p);
	err = Coord2GraphPt(ggrafhdl,xPtr,yPtr,&h,&v); if(err){ goto Error; }

#ifdef MAC
	PenSize(2,2);
	SetPenPat(UPSTRIPES);
	//PenPat((ConstPatternParam)&qd.gray);	// dashed lines
#else
	PenStyle(BLACK,2);
	//FillPat(DARKGRAY);
#endif

	if (gnewplot == 0)
	{
		// check if this point equal last point
		if (h==prev_h && v==prev_v){ goto Error; }
		if (prev_h == 0 && prev_v == 0) {prev_h = v_axis; prev_v = h_axis;}
#ifdef MAC
		poly = OpenPoly();
		MyMoveTo(prev_h,h_axis);
		MyLineTo(prev_h,prev_v);
		MyLineTo(h,prev_v);
		MyLineTo(h,h_axis);
		ClosePoly();
		PaintPoly(poly);
		PenNormal();
		FramePoly(poly);
		KillPoly(poly);
		prev_h=h; prev_v=v;
#else
		MyLineTo(h,prev_v);
		MyLineTo(h,h_axis);
		MyLineTo(h,v); prev_h=h; prev_v=v;
#endif
	}
	else
	{ 	
		gnewplot=0; 
		MyLineTo(h,v); prev_h=h; prev_v=v;
	}

	RGBForeColor(&colors[BLACK]);
	PenNormal();

Error:

	return err;
}
OSErr StandardPlot3(double* xPtr, double* yPtr, long pntnum)
{
	OSErr	err=0;
	short h=0, v=0;
	
	err = Coord2GraphPt(ggrafhdl,xPtr,yPtr,&h,&v); if(err){ goto Error; }

	if (gnewplot == 0)
	{
		// check if this point equal last point
		if (h==prev_h && v==prev_v){ goto Error; }
	}
	else
	{ 	
		gnewplot=0; 
	}

#ifdef MAC
	PenSize(2,2);
	//PenPat(&GRAY_BRUSH);	// dashed lines
	PenPatQDGlobalsGray();
#else
	PenStyle(BLACK,2);
	FillPat(DARKGRAY);
#endif
	//PenSize(1,1); 
	//PenStyle(DOWNSTRIPES,1);
	//SetPenPat(DOWNSTRIPES);
	//RGBForeColor(&colors[BLACK]);
	//RGBForeColor(&colors[RED]);

	//option to only draw points, not lines
	//MyMoveTo(h,v);
	//MyDrawString(/*CENTERED,*/h,v,"x",false,0);
	MyLineTo(h,v); prev_h=h; prev_v=v;
	RGBForeColor(&colors[BLACK]);
	PenNormal();

Error:

	return err;
}
OSErr StandardOffPlot2(double* pxPtr, double* pyPtr, double* xPtr, double* yPtr, Boolean moveOnly, long pntnum)
{
	OSErr	err=0;
	short x=0, y=0;
	double newY = *yPtr+1;

	RGBForeColor(&colors[RED]);
	//if (moveOnly)
	{
		err = Coord2GraphPt(ggrafhdl,xPtr,yPtr,&x,&y);
		MyMoveTo(x,y);
		MyDrawString(/*CENTERED,*/x,y,"x",false,0);
		//MyMoveTo(x,y);
		//err = Coord2GraphPt(ggrafhdl,xPtr,&newY,&x,&y);
		//MyLineTo(x,y);
		//MyDrawString(/*CENTERED,*/x,y,"x",false,0);
		//goto Error;
	}
	RGBForeColor(&colors[BLACK]);
	
//Error:
	if(err){
	}
	return err;
}

OSErr StandardOffPlot(double* pxPtr, double* pyPtr, double* xPtr, double* yPtr, Boolean moveOnly, long pntnum)
{
	OSErr	err=0;
	short x=0, y=0;

	if (moveOnly)
	{
		err = Coord2GraphPt(ggrafhdl,xPtr,yPtr,&x,&y);
		MyMoveTo(x,y);
		goto Error;
	}
	
Error:
	if(err){
	}
	return err;
}

OSErr StandardOffPlot3(double* pxPtr, double* pyPtr, double* xPtr, double* yPtr, Boolean moveOnly, long pntnum)
{
	OSErr	err=0;
	short x=0, y=0;

#ifdef MAC
	PenSize(1,1);
	//PenPat(&GRAY_BRUSH);	// dashed lines
	PenPatQDGlobalsGray();
#else
	PenStyle(BLACK,1);
	FillPat(DARKGRAY);
#endif
	//PenStyle(DOWNSTRIPES,1);
	//SetPenPat(DOWNSTRIPES);
	//RGBForeColor(&colors[RED]);
	if (moveOnly)
	{
		err = Coord2GraphPt(ggrafhdl,xPtr,yPtr,&x,&y);
		MyMoveTo(x,y);
		goto Error;
	}
	RGBForeColor(&colors[BLACK]);
	PenNormal();
	
Error:
	if(err){
	}
	return err;
}

OSErr StandardOffPlot4(double* pxPtr, double* pyPtr, double* xPtr, double* yPtr, Boolean moveOnly, long pntnum)
{
	OSErr	err=0;
	short x=0, y=0;

/*#ifdef MAC
	PenSize(1,1);
	PenPat((ConstPatternParam)&qd.gray);	// dashed lines
#else
	PenStyle(BLACK,1);
	FillPat(DARKGRAY);
#endif*/
	//PenStyle(DOWNSTRIPES,1);
	//SetPenPat(DOWNSTRIPES);
	//RGBForeColor(&colors[RED]);
	if (moveOnly)
	{
		err = Coord2GraphPt(ggrafhdl,xPtr,yPtr,&x,&y);
		MyMoveTo(x,y); prev_h=x; prev_v=y;	
		goto Error;
	}
	RGBForeColor(&colors[BLACK]);
	PenNormal();
	
Error:
	if(err){
	}
	return err;
}

void SplitDouble(double *value, double *frac_value, short *power)
{
	short	pow_val=0;
	double dbl_val=0.0;

	dbl_val = fabs(*value);

	if(dbl_val==0.0)
	{
		// Drop through w/o doing anything
	}
	else if(dbl_val<1.0)
	{
		while (dbl_val<1.0)
		{
			dbl_val = 10.0*dbl_val;
			pow_val--;
		}
	}
	else if(dbl_val<10.0)
	{
		// Drop through w/o doing anything
	}
	else if(dbl_val>=10.0)
	{
		while (dbl_val>=10.0){ dbl_val=(dbl_val/10.0); pow_val++; }
	}

	*power = pow_val;
	*frac_value = dbl_val;

	return;
}

void Raise2Pow(double *inval, double *outval, short power)
{
	double val = *inval;
	short	i = 0, pow = 0;

	pow=abs(power);

	if (pow==0){ val=1.0; goto done; }

	for (i=2; i<(pow+1); i++){ val=((*inval) * val); }

	if (power<0){ val=1.0/val; }
	
done:
	*outval = val;
}

void ArrayMinMaxDouble(double *array, long n, double *minv, double *maxv)
{
	double min=0.0, max=0.0, val=0.0;
	long i=0;

	min = max = array[0];
	
	for (i=1; i<n; i++)
	{
		val=array[i];
		if (val<min)
		{
			min=val;
		}
		else if (val>max)
		{
			max=val;
		}
	}
	*minv = min;
	*maxv = max;

	return;
}

void ArrayMinMaxFloat(float *array, long n, double *minv, double *maxv)
{
	double min=0.0, max=0.0, val=0.0;
	long i=0;

	min = max = (double)array[0];
	
	for (i=1; i<n; i++)
	{
		val=(double)array[i];
		if (val<min)
		{
			min=val;
		}
		else if (val>max)
		{
			max=val;
		}
	}
	*minv = min;
	*maxv = max;

	return;
}

void ArrayMinMaxLong(long *array, long n, double *minv, double *maxv)
{
	double min=0.0, max=0.0, val=0.0;
	long i=0;

	min = max = (double)array[0];
	
	for (i=1; i<n; i++)
	{
		val=(double)array[i];
		if (val<min)
		{
			min=val;
		}
		else if (val>max)
		{
			max=val;
		}
	}
	*minv = min;
	*maxv = max;

	return;
}

void ArrayMinMaxShort(short *array, long n, double *minv, double *maxv)
{
	double min=0.0, max=0.0, val=0.0;
	long i=0;

	min = max = (double)array[0];
	
	for (i=1; i<n; i++)
	{
		val=(double)array[i];
		if (val<min)
		{
			min=val;
		}
		else if (val>max)
		{
			max=val;
		}
	}
	*minv = min;
	*maxv = max;

	return;
}

