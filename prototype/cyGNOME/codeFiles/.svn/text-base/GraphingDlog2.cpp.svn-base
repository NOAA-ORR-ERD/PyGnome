/////////////////////////////////////////////////////////////////
////////  PLOTDLG DIALOG CODE  Plot Dialog ////////////
/////////////////////////////////////////////////////////////////

// rework for an oiled shoreline plot, not sure if threshold overlays and axis setting will be necessary, or multiple plot types?
#include "Cross.h"
#include "Graphing.h"
#include "Ossm.h"
#include "GridVel.h"

// may want to reset everything rather than remember (or vice versa ?) // Alan requested that we remember selections 5/1/03
static short gplottype = OSPLOTDLG_SHORELINEGALSPLOTBTN, gplotgrid = 1;
static long gstrtpnt, gendpnt;
static ExRect gplotbnds;	
static Rect ggridrect;
static Boolean gprintingplot = false/*, gTriSelected = false*/;
static short gFont=kFontIDGeneva, gSize=LISTTEXTSIZE; 
static OiledShorelineDataHdl gOiledShorelineHdl = 0;
//static float gContourDepth1, gContourDepth2;

//double gXminVal = 0;
static double gShorelineGalsXMax, gShorelineGalsYMax;	// to allow to set endpoints of axes
static double gShorelineMilesXMax, gShorelineMilesYMax;	// to allow to set endpoints of axes
static double gShorelineMilesXMin, gShorelineGalsXMin;	// to allow to set endpoints of axes
static Boolean gSetShorelineGalsAxes = false, gSetShorelineMilesAxes = false;

OSErr OiledShorelinePlot(Rect *drawrect, long strtpnt, long endpnt, short plottype, short usegrid, Boolean printing, Rect *rGrid, ExRect *plotBnds)
{
	double ymin, ymax, xmin, xmax, val, non_log_y_max;
	double xprim, xscnd, yprim, yscnd, val_range, val_frac, mult_scale=1.0;
	GrafVal **grafhdl=nil;
	PenState	pensave;
	double **xdata=nil, **ydata=nil;
	char xtitle[64]="", ytitle[64]="", graftitle[64]="", grafsubtitle[64]=""; 
	char str[32]="", ustr[16]="";
	short	val_pow=0, errtype=0, ndcmlplcs=0;
	long i, npnts;
	TextInfo	obj_text;
	DateTimeRec time;
	Seconds timeZero;
	long segIndex;
	OiledShorelineData data;
	OSErr err = 0;

	SetWatchCursor();

	// Port Info
	GetPenState(&pensave); PenNormal();

	// init
	npnts	= (endpnt-strtpnt+1);
	//npnts = _GetHandleSize((Handle)gOiledShorelineHdl)/sizeof(OiledShorelineData);	// check on it
	

	obj_text.theFont = gFont;
	obj_text.theSize = gSize;
	obj_text.theJust = kCenterJust;

	// can't plot fewer than two points
	//if(npnts < 2){ SysBeep(1); goto done; }	// just show an 'x' at the spot

	// check for dialog button states (function called from dialog)
	if(plottype == OSPLOTDLG_SHORELINEGALSPLOTBTN)
	{
		strcpy(ustr,"gallons");
		errtype = kShorelineGals; // may not need this
	}
	else if(plottype == OSPLOTDLG_SHORELINEMILESPLOTBTN)
	{
		strcpy(ustr,"miles");
		errtype = kShorelineMiles;	
	}

	// Init new graph
	grafhdl = InitGraph(false,false);	// no horiz or vert scroll bars
	if(grafhdl == nil){ goto done; }

	// Init xdata array
	xdata = (double **)_NewHandleClear(npnts*sizeof(double)); err=(short)_MemError();
	if(err){ TechError("OiledShorelinePlot()","_NewHandle()",0); goto done; }

	// Init ydata array
	ydata = (double **)_NewHandleClear(npnts*sizeof(double)); err=(short)_MemError();
	if(err){ TechError("OiledShorelinePlot()","_NewHandle()",0); goto done; }

	// Get axis and graph titles
	strcpy(xtitle,"Segment (Point indices of selected segments)");
	switch( errtype ){
		case kShorelineGals:{
			sprintf(ytitle,"%s%s%s","Concentration (",ustr,")");
			/*if (gTriSelected)
				sprintf(graftitle,"%s","Average Concentration Over Selected Triangles vs. Time");
			else*/
				sprintf(graftitle,"%s","Gallons of oil on selected shoreline.");
			/*if (gContourDepth1 == BOTTOMINDEX)
				sprintf(grafsubtitle, "Depth range : Bottom Layer (1 meter)");
			else
				sprintf(grafsubtitle, "Depth range : %g to %g meters",gContourDepth1,gContourDepth2);*/
			break;}
		case kShorelineMiles:{
			sprintf(ytitle,"%s%s%s","Miles (",ustr,")");
			//mult_scale = 1.0;
			/*if (gTriSelected)
				sprintf(graftitle,"%s","Maximum Concentration Over Selected Triangles vs. Time");
			else*/
				sprintf(graftitle,"%s","Miles of shoreline at selected segments.");
			/*if (gContourDepth1 == BOTTOMINDEX)
				sprintf(grafsubtitle, "Depth range :  Bottom Layer (1 meter)");
			else
				sprintf(grafsubtitle, "Depth range :  %g to %g meters",gContourDepth1,gContourDepth2);*/
			break;}
	}

	// assign data to x array ( vertex number w/ +1 offset )
	//for(i = 0 ; i < npnts ; i++){ (*xdata)[i] = (double)(i+strtpnt); }
	//timeZero = INDEXH(gOiledShorelineHdl,0).time;
	for (i = 0; i < npnts; i++)
	{ 
		/*if (errtype==kDepthSlice)
			(*xdata)[i]=gDepthSlice[i+1];
			//(*xdata)[i]=i+1;
		else*/
		{
			//segIndex = INDEXH(gOiledShorelineHdl,0).startPt;
			//timeZero = INDEXH(gOiledShorelineHdl,0).time;
			{
				data = INDEXH(gOiledShorelineHdl,i);
				(*xdata)[i] = ((double)(data.startPt)); 	// may want to reorder
			}
		}
	}

	// assign data to y array
	if (plottype == OSPLOTDLG_SHORELINEGALSPLOTBTN)
	{
		for (i = 0; i < npnts; i++)
		{
			data = INDEXH(gOiledShorelineHdl,i);
			//val = i;
			//if (val > 0.0001){ (*ydata)[i] = val; }
			//else{ (*ydata)[i] = 0.0001; }
			// if (gTriSelected) shift data back to first non-zero value, reduce npnts and x values
			(*ydata)[i] = data.gallonsOnSegment;	// put in some sort of minimum?
		}
	}
	else if (plottype == OSPLOTDLG_SHORELINEMILESPLOTBTN)
	{
		for (i = 0; i < npnts; i++)
		{
			data = INDEXH(gOiledShorelineHdl,i);
			//val = 2*i;
			//if (val > 0.0001){ (*ydata)[i] = val; }
			//else{ (*ydata)[i] = 0.0001; }
			(*ydata)[i] = data.segmentLengthInKm;	// change to miles
		}
	}

	/*if (gTriSelected)
	{
		for (i = 0; i < npnts; i++)
		{
			if ((*ydata)[i] != 0.)
			{
				gXminVal = (*xdata)[i];	// re-think for combination natural and chemical
				break;
			}
		}
	}*/

	
	(**grafhdl).ShowToxicityThresholds = false;
		
	_HLock((Handle)xdata); _HLock((Handle)ydata);
	ArrayMinMaxDouble((double *)(*xdata),npnts,&xmin,&xmax);
	ArrayMinMaxDouble((double *)(*ydata),npnts,&ymin,&ymax);
	//ymin = 0.;	// check vs overlay, but don't turn into log based on overlay
	{
		//ymin = 0.;
		if (gSetShorelineGalsAxes && plottype == OSPLOTDLG_SHORELINEGALSPLOTBTN)
		{
			//ymax = gShorelineGalsYMax;
			xmin = gShorelineGalsXMin;
			xmax = gShorelineGalsXMax;
		}
		else if (gSetShorelineMilesAxes && plottype == OSPLOTDLG_SHORELINEMILESPLOTBTN)
		{
			//ymax = gShorelineMilesYMax;
			xmin = gShorelineMilesXMin;
			xmax = gShorelineMilesXMax;
		}
	}

	// Set graph area (subtract window controls)
	SetGraphArea(grafhdl,drawrect);

	// Set the type of borders to be drawn
	SetAxisTypeData(grafhdl, kBottomAxis, kPlainBorder, nil);
	SetAxisTypeData(grafhdl, kLeftAxis, kPlainBorder, nil);

	// Set the type of grid to be drawn
	SetGrid(grafhdl, kStandardGrid, nil, 0);	

	// set the graph titles
	SetGraphTitle(grafhdl,graftitle,grafsubtitle,gFont,gSize);
	SetBorderTitle(grafhdl,kBottomAxis,xtitle,gFont,gSize,kCenterJust,0);
	SetBorderTitle(grafhdl,kLeftAxis,ytitle,gFont,gSize,kLeftJust,0);

	// calculate spread for x axis ticks
	val_range=(xmax-xmin); if( val_range == 0.0 ){ val_range=xmax; }

/*	if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
	{
		SplitDouble(&val_range,&val_frac,&val_pow); xprim = 10.0; // set xprim to 10.0 and pass it in to raise it
	}
	else*/
	{
		SplitDouble(&val_range,&val_frac,&val_pow); xprim = 12.0; // change to powers of 12 (day segments), maybe not for depth slice?
	}
	Raise2Pow(&xprim,&xprim,val_pow);
 	if (val_frac <= 1.5)
	{
		//xprim /= 5.0;
		//xscnd = xprim/5.0;
		/*if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
		{
			xprim /= 5.0;
			xscnd = xprim/5.0;
		}
		else*/	
		{
			xprim /= 6.0;
			xscnd = xprim/6.0;
		}
	}
	else if (val_frac <= 3.0)
	{
		xprim /= 2.0;
		//xscnd = xprim/5.0;
		/*if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
			xscnd = xprim/5.0;
		else*/		
			xscnd = xprim/6.0;
	}
	else if (val_frac <= 6.0)
	{
		//xscnd = xprim/5.0;
		/*if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
			xscnd = xprim/5.0;
		else*/		
			xscnd = xprim/6.0;
	}
	else
	{
		xprim *= 2.0;
		xscnd = xprim/4.0;
	}

	// Set up the primary and secondary tick marks for bottom border
	SetTickInfo( grafhdl, kBottomAxis, kPrimaryTicks, kUseLabels, (Boolean)usegrid, kPlainTicks, 
			xprim, kNoIcon,&obj_text );
	SetTickInfo( grafhdl, kBottomAxis, kSecondaryTicks, kNoLabels, kNoGrid, kSmallPlainTicks, 
			xscnd, kNoIcon,&obj_text );

	// set AFTER calling SetTickInfo()
	SetXMinMax(grafhdl, &xmin, &xmax);

	// calculate spread for Y axis ticks
	val_range=(ymax-ymin); if (val_range == 0.0){ val_range=ymax; }	// option to set min/max manually ?
	SplitDouble(&val_range,&val_frac,&val_pow); yprim=10.0;
	if (val_pow > 4)
	{
		// set up log10 plot
		(**grafhdl).islogplot = 1;

		// convert to log10
		for(i=0;i<npnts;i++)
		{
			(*ydata)[i] = (double)log10( ((*ydata)[i]) );
		}

		// get the array min max again
		ArrayMinMaxDouble((double *)(*ydata),npnts,&ymin,&ymax);
		/*if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
			ymax = 0.;
		else*/
			ymin = 0.;
		// the tick spread is 1.0 for a log10 plot (always)
		yprim = yscnd = 1.0;

		// set the tick information
		SetTickInfo(grafhdl,kLeftAxis,kPrimaryTicks,kUseLabels,(Boolean)usegrid,kPlainTicks,yprim,kNoIcon,&obj_text);
		SetTickInfo(grafhdl,kLeftAxis,kSecondaryTicks,kNoLabels,kNoGrid,kNoTicks,yscnd,kNoIcon,&obj_text);
		
		// set AFTER calling SetTickInfo()
		SetYMinMax(grafhdl, &ymin, &ymax);

		// set aside max label space
		TextFont(gFont); TextSize(gSize); TextFace(normal);
		val_pow=(short)((**grafhdl).maxYLabVal); non_log_y_max = 10.0;
		Raise2Pow(&non_log_y_max,&non_log_y_max,(val_pow+0));
	
		(**grafhdl).ndcmlplcs=1; sprintf(str,"%.1le",non_log_y_max);
		SetBorderDistance(grafhdl,50,(25+stringwidth(str)),(gSize*5),30);

		SetBorderTitle( grafhdl,kLeftAxis,ytitle,gFont,gSize,kLeftJust,0);
	}
	else
	{
		Raise2Pow(&yprim,&yprim,val_pow);
		if (val_frac <= 1.5)
		{
			yprim /= 5.0;
			yscnd = yprim/5.0;
		}
		else if (val_frac <= 3.0)
		{
			yprim /= 2.0;
			yscnd = yprim/5.0;
		}
		else if (val_frac <= 6.0)
		{
			yscnd = yprim/5.0;
		}
		else
		{
			yprim *= 2.0;
			yscnd = yprim/4.0; 
		}

		// set the tick information
		SetTickInfo(grafhdl,kLeftAxis,kPrimaryTicks,kUseLabels, (Boolean)usegrid,kPlainTicks,yprim,kNoIcon,&obj_text);
		SetTickInfo(grafhdl,kLeftAxis,kSecondaryTicks,kNoLabels,kNoGrid,kSmallPlainTicks,yscnd,kNoIcon,&obj_text);
		
		// set AFTER calling SetTickInfo()
		SetYMinMax( grafhdl, &ymin, &ymax );
		if (plottype == OSPLOTDLG_SHORELINEGALSPLOTBTN)
		{
			if (gSetShorelineGalsAxes)
			{
				//(**grafhdl).maxYLabVal = gShorelineGalsYMax;	//ymax
				(**grafhdl).minXLabVal = gShorelineGalsXMin;	//ymax
				(**grafhdl).maxXLabVal = gShorelineGalsXMax;	//xmax
			}
			else
			{
				//gShorelineGalsYMax = (**grafhdl).maxYLabVal;	//ymax
				gShorelineGalsXMin = (**grafhdl).minXLabVal;	//ymax
				gShorelineGalsXMax = (**grafhdl).maxXLabVal;	//xmax
			}
		}
		else if (plottype == OSPLOTDLG_SHORELINEMILESPLOTBTN)
		{
			if (gSetShorelineMilesAxes)
			{
				//(**grafhdl).maxYLabVal = gShorelineMilesYMax;	//ymax
				(**grafhdl).minXLabVal = gShorelineMilesXMin;	//ymax
				(**grafhdl).maxXLabVal = gShorelineMilesXMax;	//xmax
			}
			else
			{
				//gShorelineMilesYMax = (**grafhdl).maxYLabVal;	//ymax
				gShorelineMilesXMin = (**grafhdl).minXLabVal;	//ymax
				gShorelineMilesXMax = (**grafhdl).maxXLabVal;	//xmax
			}
		}
		
		TextFont(gFont); TextSize(gSize); TextFace(normal);

		// returns 0, 1, 2, 3, 4, 5, or 6
		val=(**grafhdl).theLabels[kLeftAxis].primaryTicks.spreadVal;
		ndcmlplcs=GetNumDecPlaces(&val); (**grafhdl).ndcmlplcs=ndcmlplcs;
		switch(ndcmlplcs){
			case 0:{
				sprintf(str,"%.0lf",val); break; }
			case 1:{
				sprintf(str,"%.1lf",val); break; }
			case 2:{
				sprintf(str,"%.2lf",val); break; }
			case 3:{
				sprintf(str,"%.3lf",val); break; }
			case 4:{
				sprintf(str,"%.4lf",val); break; }
			case 5:{
				sprintf(str,"%.5lf",val); break; }
			case 6:{
				sprintf(str,"%.6lf",val); break; }
		}
	
		SetBorderDistance(grafhdl,50,(25+stringwidth(str)),(gSize*5),30);
	}

	// Place the data to be graphed
	SetData(grafhdl,(Ptr)(*xdata),(Ptr)(*ydata),npnts,kDouble,true);
	//SetDataPlotType(grafhdl,(MyPlotProc)StandardPlot,(MyOffPlotProc)StandardOffPlot);
	// might want to use a different function for depth profile plots - points not lines?
	if (/*plottype == PLOTDLG_DEPTHSLICEPLOTBTN && */npnts == 1)
		SetDataPlotType(grafhdl,(MyPlotProc)StandardPlot2,(MyOffPlotProc)StandardOffPlot2);
	else
		//SetDataPlotType(grafhdl,(MyPlotProc)StandardPlot,(MyOffPlotProc)StandardOffPlot);
		SetDataPlotType(grafhdl,(MyPlotProc)StandardPlot4,(MyOffPlotProc)StandardOffPlot4);

	EraseRect(drawrect);

	// draw graph
	DrawGraph(grafhdl);

	// store max grid values
	if( printing == FALSE ){
		(*rGrid) = (**grafhdl).gridrect;
		plotBnds->right = (**grafhdl).maxXLabVal;
		plotBnds->top = (**grafhdl).maxYLabVal;
		plotBnds->left = (**grafhdl).minXLabVal;
		plotBnds->bottom = (**grafhdl).minYLabVal;
	}

done:

	// reset port 
	SetPenState(&pensave);
	
	if(xdata){
		_HUnlock((Handle)xdata);
		DisposeHandle((Handle)xdata);
	}
	if(ydata){
		_HUnlock((Handle)ydata);
		DisposeHandle((Handle)ydata);
	}
	if(grafhdl){
		_HUnlock((Handle)grafhdl);
		DisposeHandle((Handle)grafhdl);
	}

	return err;
}

/////////////////////////////////////////////////
Boolean OSPlotClick(DialogPtr d, Point clk)
{
	Rect r;
	Point	mse, oldmse;
	PenState	pensave;
	long newstart, newend, npts;

	mse = oldmse = clk;
	
	if (!MyPtInRect(clk, &ggridrect)) return false;
	
	// set the gridrect
	r = ggridrect;
	r.left = clk.h; r.right = clk.h;
	
	// Pen Info 
	GetPenState(&pensave);
	PenNormal();
	RGBForeColor(&colors[BLACK]);
	PenMode(patXor);
#ifdef MAC
	//PenPat(&GRAY_BRUSH);
	PenPatQDGlobalsGray();
#else
	FillPat(DARKGRAY);
#endif
	
	PaintRect(&r); // draw
	while (Button()) {
		GetMouse(&mse);
		if (!EqualPoints(mse, oldmse)) 
		{
			PaintRect(&r); // erase
			if (mse.h <= clk.h){ r.left = mse.h; r.right = clk.h; }
			if (mse.h > clk.h){ r.right = mse.h; r.left = clk.h; }
			
			if (r.left < ggridrect.left) r.left = ggridrect.left;
			if (r.right > ggridrect.right) r.right = ggridrect.right;
			
			PaintRect(&r); // draw
			oldmse = mse;
		}
	}
	PaintRect(&r); // erase
	if (RectWidth(r) > 3)
	{
		newstart = floor(gplotbnds.left + ((gplotbnds.right-gplotbnds.left) * (double)(r.left-ggridrect.left) / (double)(ggridrect.right-ggridrect.left)));
		newend = ceil(gplotbnds.left + ((gplotbnds.right-gplotbnds.left) * (double)(r.right-ggridrect.left) / (double)(ggridrect.right-ggridrect.left)));
		
		// safety check to be sure
		if (newstart < 1) newstart = 1;
		//if( newend > 100 ) newend = 100;
		npts = _GetHandleSize((Handle)gOiledShorelineHdl)/sizeof(OiledShorelineData);
		if (newend > npts) newend = npts;
		
		if (newend - newstart >= 8)
		{
			gstrtpnt = newstart;
			gendpnt = newend;
			InvalDItemBox(d, OSPLOTDLG_PLOTBOX);
		}
	}
	SetPenState(&pensave);
	
	return TRUE;
}

// deal with this later when building on IBM
#ifdef IBM

LRESULT CALLBACK PlotProc2(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	short itemID;
	Point p;
	POINT P;
	PAINTSTRUCT ps;
	WindowPtr savePort;
	pascal_ifMac void DrawOSPlot(DialogPtr d, short frameItem);
	
	itemID = GetDlgCtrlID(hWnd);
	if (!itemID) { SysBeep(1); return 0; }
	
	switch (message) {
		case WM_PAINT:
			GetPort(&savePort);
			SetPortBP(hWnd, &ps);
			SetPortEP(savePort, &ps);
			SetPort(GetParent(hWnd));
			DrawOSPlot(GetParent(hWnd), OSPLOTDLG_PLOTBOX);
			SetPort(savePort);
			break;
		
		case WM_LBUTTONDOWN:
			P.x = LOWORD(lParam);
			P.y = HIWORD(lParam);
			MapWindowPoints(hWnd, GetParent(hWnd), &P, 1);
			MakeMacPoint(&P, &p);
			GetPort(&savePort);
			SetPort(GetParent(hWnd));
			OSPlotClick(GetParent(hWnd), p);
			SetPort(savePort);
			break;
		
		default: return DefWindowProc(hWnd, message, wParam, lParam);
	}
	
	return 0;
}

#endif

pascal_ifMac void DrawOSPlot(DialogPtr d, short frameItem)
{
	Rect r = GetDialogItemBox(d,frameItem);
	short	err = 0;
	GrafPtr sp;

#ifdef MAC
	GetPortGrafPtr(&sp);
	SetPortDialogPort(d);
#endif
	err = OiledShorelinePlot(&r, gstrtpnt, gendpnt, gplottype, gplotgrid, gprintingplot, &ggridrect, &gplotbnds);

#ifdef MAC
	SetPortGrafPort(sp);
#endif
	
	if (err != 0){	// do something ?
		
	}

	return;
}

void OSPLOTDLG_BTNHANDLER(DialogPtr d, short id)
{
	switch(id){
		case OSPLOTDLG_SHORELINEMILESPLOTBTN:
		case OSPLOTDLG_SHORELINEGALSPLOTBTN:
			SetButton(d,OSPLOTDLG_SHORELINEMILESPLOTBTN,0);
			SetButton(d,OSPLOTDLG_SHORELINEGALSPLOTBTN,0);
			SetButton(d,gplottype,1); 
			break;

		case OSPLOTDLG_SHOWPLOTGRIDBTN:
			SetButton(d,id,0);
			SetButton(d,id,gplotgrid); 
			break;

		default: 
			break;
	}
}

OSErr OSPLOTDLG_INIT(DialogPtr d, VOIDPTR data)
{
	short i;
	// assign draw proc handles
	SetDialogItemHandle(d, OSPLOTDLG_HILITE, (Handle)FrameDefault);
	SetDialogItemHandle(d, OSPLOTDLG_PLOTBOX, (Handle)DrawOSPlot);

	// set check boxes
	OSPLOTDLG_BTNHANDLER(d,gplottype);
	OSPLOTDLG_BTNHANDLER(d,OSPLOTDLG_SHOWPLOTGRIDBTN);
	
	gstrtpnt = 1;
	//gendpnt = 100;
	{
		gendpnt = _GetHandleSize((Handle)gOiledShorelineHdl)/sizeof(OiledShorelineData);
	}
	
	return 0;
}

short OSPLOTDLG_CLICK(DialogPtr d, short itemHit, long lParam, VOIDPTR data)
{
	char path[256];
	Point p, where;
	Rect r;
	MySFReply reply;
	OSErr err = 0;
	
	switch (itemHit) {
		/////////////////////////////////////////////////
		case OSPLOTDLG_CANCEL: return OSPLOTDLG_CANCEL;
		
		/////////////////////////////////////////////////
		case OSPLOTDLG_HELP: break;	// put something into startscreens
		
		/////////////////////////////////////////////////
		case OSPLOTDLG_OK: return OSPLOTDLG_OK;
		
		/////////////////////////////////////////////////
		case OSPLOTDLG_SHOWPLOTGRIDBTN:{
			if (gplotgrid == 0)
				gplotgrid = 1; 
			else
				gplotgrid = 0; 

			OSPLOTDLG_BTNHANDLER(d,itemHit);
			gprintingplot = false;
			InvalDItemBox(d, OSPLOTDLG_PLOTBOX);
			break;
		}
		
		/////////////////////////////////////////////////
		case OSPLOTDLG_SHORELINEMILESPLOTBTN:{
			if (gplottype != OSPLOTDLG_SHORELINEMILESPLOTBTN)
			{
				gplottype = OSPLOTDLG_SHORELINEMILESPLOTBTN;
				OSPLOTDLG_BTNHANDLER(d,itemHit);
				gprintingplot = false;
				InvalDItemBox(d, OSPLOTDLG_PLOTBOX);
			} break;
		}
		
		/////////////////////////////////////////////////
		case OSPLOTDLG_SHORELINEGALSPLOTBTN:{
			if (gplottype != OSPLOTDLG_SHORELINEGALSPLOTBTN) 
			{
				gplottype = OSPLOTDLG_SHORELINEGALSPLOTBTN;
				OSPLOTDLG_BTNHANDLER(d,itemHit);
				gprintingplot = false;
				InvalDItemBox(d, OSPLOTDLG_PLOTBOX);
			} break;
		}
		/////////////////////////////////////////////////
#ifdef MAC
		case OSPLOTDLG_PLOTBOX:{
			p = lastEvent.where;
			GlobalToLocal(&p);
			OSPlotClick(d, p);
			break;
		}
#endif

		/////////////////////////////////////////////////
		case OSPLOTDLG_SETAXES:{	// allow to change axes
		 	if (!SetAxesDialog2())
			{			
				InvalDItemBox(d, OSPLOTDLG_PLOTBOX);
			}
			break;
		}
		
		/////////////////////////////////////////////////
		case OSPLOTDLG_SHOWTABLE:{	// show concentration table
			gprintingplot = false;
			InvalDItemBox(d, OSPLOTDLG_PLOTBOX);
			OiledShorelineTable(gOiledShorelineHdl);	
			break;
		}
		
		/////////////////////////////////////////////////
		case OSPLOTDLG_SAVEPLOT:{
			char ibmBackwardsTypeStr[32] = "";
			Boolean changeExtension = false;	// for now
			char previousPath[256]="",defaultExtension[3]="";
			where = CenteredDialogUpLeft(M55);
#if TARGET_API_MAC_CARBON
		err = AskUserForSaveFilename("Beach_graph.pic",path,".pic",true);
		if (err) /*return USERCANCEL;*/ break;
#else
#ifdef MAC
			sfputfile(&where, "Name:", "Beach_graph.pic", (DlgHookUPP)0, &reply);
#else
			sfpputfile(&where, ibmBackwardsTypeStr, "Beach_graph.bmp", (MyDlgHookProcPtr)0, &reply,
						  M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
#endif
			if (!reply.good) break;
			
			my_p2cstr(reply.fName);
			hdelete(reply.vRefNum, 0, (char *)reply.fName);
			
			if (err = hcreate(reply.vRefNum, 0, (char *)reply.fName, 'ttxt', 'PICT'))
				{ TechError("OSPLOTDLG_CLICK()", "hcreate()", err); return 0; }
			
#ifdef MAC
			GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
#else
			strcpy(path, reply.fName);
#endif
#endif
			// fall through
		}
		case OSPLOTDLG_PRINTPLOT:
			r = GetDialogItemBox(d, OSPLOTDLG_PLOTBOX);
			err = PrintPlot(itemHit == OSPLOTDLG_PRINTPLOT ? 0 : path,
								 itemHit == OSPLOTDLG_PRINTPLOT,
								 r, gstrtpnt, gendpnt, gplottype, gplotgrid, ggridrect, gplotbnds,false);
			break;
	}
	
	return 0;
}

Boolean OSPlotDialog(OiledShorelineData** oiledShorelineHdl)	
{
	short item;
	if (oiledShorelineHdl)
	{
		gOiledShorelineHdl = oiledShorelineHdl;
	}
	else
		return false;

	item = MyModalDialog(OSPLOTDLG, mapWindow, 0, OSPLOTDLG_INIT, OSPLOTDLG_CLICK);
	
	if (item == OSPLOTDLG_OK) 
		return true;
	
	return false;
}

void ShowHideSetAxesDialogItems2(DialogPtr dialog)
{
	Boolean setAxesCheckBox = GetButton(dialog, SETAXES_CHECKBOX);

	if (setAxesCheckBox)
	{
		ShowHideDialogItem(dialog, SETAXES_MINYLABEL, true ); 
		ShowHideDialogItem(dialog, SETAXES_MINY, true); 
		ShowHideDialogItem(dialog, SETAXES_MINYUNITS, true); 
		ShowHideDialogItem(dialog, SETAXES_MAXXLABEL, true); 
		ShowHideDialogItem(dialog, SETAXES_MAXX, true); 
		ShowHideDialogItem(dialog, SETAXES_MAXXUNITS, true); 
	}
	else 
	{
		ShowHideDialogItem(dialog, SETAXES_MINYLABEL, false ); 
		ShowHideDialogItem(dialog, SETAXES_MINY, false); 
		ShowHideDialogItem(dialog, SETAXES_MINYUNITS, false); 
		ShowHideDialogItem(dialog, SETAXES_MAXXLABEL, false); 
		ShowHideDialogItem(dialog, SETAXES_MAXX, false); 
		ShowHideDialogItem(dialog, SETAXES_MAXXUNITS, false); 
	}
}

OSErr SETAXESInit2(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)

	double xval = 0, yval = 0;
	Boolean setAxes = 0;

	// assign draw proc handles
	SetDialogItemHandle(dialog, SETAXES_HILITE, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, SETAXES_FROST, (Handle)FrameEmbossed);

	//Float2EditText (dialog, SETAXES_MAXX, gDepthXMax, 2);
	//Float2EditText (dialog, SETAXES_MINY, fabs(gDepthYMin), 2);

	switch (gplottype) {
		//case OSPLOTDLG_SHORELINEGALSPLOTBTN: xval = gShorelineGalsXMax; yval = gShorelineGalsYMax; setAxes = gSetShorelineGalsAxes; break;
		//case OSPLOTDLG_SHORELINEMILESPLOTBTN: xval = gShorelineMilesXMax; yval = gShorelineMilesYMax; setAxes = gSetShorelineMilesAxes; break;
		case OSPLOTDLG_SHORELINEGALSPLOTBTN: xval = gShorelineGalsXMax; yval = gShorelineGalsXMin; setAxes = gSetShorelineGalsAxes; break;
		case OSPLOTDLG_SHORELINEMILESPLOTBTN: xval = gShorelineMilesXMax; yval = gShorelineMilesXMin; setAxes = gSetShorelineMilesAxes; break;
	}
	Float2EditText (dialog, SETAXES_MAXX, xval, 2);
	Float2EditText (dialog, SETAXES_MINY, yval, 2);
	SetButton (dialog, SETAXES_CHECKBOX, setAxes);

	ShowHideSetAxesDialogItems2(dialog);
	
	return 0;
	
}

short SETAXESClick2(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	switch (itemNum) {
		case SETAXES_CANCEL: return SETAXES_CANCEL;

		case SETAXES_OK:
			double xMaxVal, yMinVal;
			xMaxVal = EditText2Float(dialog, SETAXES_MAXX);
			yMinVal = EditText2Float(dialog, SETAXES_MINY);
			//if (xMaxVal<3) {printNote("The toxicity data will not appear for end time less than 3 hours after dispersal"); break;}
			//if (xMaxVal>96) {printNote("The toxicity data does not extend beyond 96 hours after dispersal"); break;}
			//if (xMinVal>xMaxVal) {printNote("The start time must be smaller than the end time"); break;}
			//gDepthXMax = xMaxVal;
			//gDepthYMin = -yMinVal;

			switch (gplottype) {
				//case OSPLOTDLG_SHORELINEGALSPLOTBTN: gShorelineGalsXMax = xMaxVal; gShorelineGalsYMax = yMinVal; break;
				//case OSPLOTDLG_SHORELINEMILESPLOTBTN: gShorelineMilesXMax = xMaxVal; gShorelineMilesYMax = yMinVal; break;
				case OSPLOTDLG_SHORELINEGALSPLOTBTN: gShorelineGalsXMax = xMaxVal; gShorelineGalsXMin = yMinVal; break;
				case OSPLOTDLG_SHORELINEMILESPLOTBTN: gShorelineMilesXMax = xMaxVal; gShorelineMilesXMin = yMinVal; break;
			}

			return SETAXES_OK;
		case SETAXES_MAXX:
		case SETAXES_MINY:
			CheckNumberTextItem(dialog, itemNum, TRUE);	// allowing decimals
			break;
		case SETAXES_CHECKBOX:
			ToggleButton(dialog, itemNum);
			ShowHideSetAxesDialogItems2(dialog);
			//gSetDepthAxes = !gSetDepthAxes;
			switch (gplottype) {
				case OSPLOTDLG_SHORELINEGALSPLOTBTN: gSetShorelineGalsAxes = !gSetShorelineGalsAxes; break;
				case OSPLOTDLG_SHORELINEMILESPLOTBTN: gSetShorelineMilesAxes = !gSetShorelineMilesAxes; break;
			}
			break;
	}

	return 0;
}


OSErr SetAxesDialog2()	
{	
	short item, i;
	//if (gplottype == PLOTDLG_DEPTHSLICEPLOTBTN)
		//item = MyModalDialog(SETAXES, mapWindow, 0, SETAXESInit, SETAXESClick);
	//else
		item = MyModalDialog(5290, mapWindow, 0, SETAXESInit2, SETAXESClick2);
	if(item == SETAXES_CANCEL) return USERCANCEL; 
	model->NewDirtNotification();	// is this necessary ?
	if(item == SETAXES_OK) 
	{
		return 0; 
	}
	else return -1;
}

