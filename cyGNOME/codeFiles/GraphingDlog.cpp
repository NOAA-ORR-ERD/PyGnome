/////////////////////////////////////////////////////////////////
////////  PLOTDLG DIALOG CODE  Plot Dialog ////////////
/////////////////////////////////////////////////////////////////
#include "Cross.h"
#include "Graphing.h"
#include "Ossm.h"
#include "GridVel.h"

// may want to reset everything rather than remember (or vice versa ?) // Alan requested that we remember selections 5/1/03
static short gplottype = PLOTDLG_AVCONCPLOTBTN, gplotgrid = 1, gLevelOfConcern = MEDIUMCONCERN, gSpecies = ADULTFISH;
static long gstrtpnt, gendpnt/*, *gDepthSlice*/;
static float *gDepthSlice;
static ExRect gplotbnds;	
static Rect ggridrect;
static Boolean gprintingplot = false, gTriSelected = false, gShowDepthPlot=false;
static short gFont=kFontIDGeneva, gSize=LISTTEXTSIZE; 
static outputDataHdl gOilConcHdl = 0;
static float gContourDepth1, gContourDepth2, gBottomRange;

static Boolean gOverLay;	
static Boolean gShowSpeciesCheckBoxes = false;	
static OverLayDataBaseH gOverLayDataBase;
//Boolean gMultSpecies[kNumSpecies], gMultConcernLevels[3];
static Boolean gMultSpecies[kNumSpecies], gMultConcernLevels[3];
double gXminVal = 0, gOverLayEndTime = 24, gOverLayStartTime = 3;
static double gDepthXMax, gDepthYMin;	// to allow to set endpoints of axes
static double gAvConcXMax, gAvConcYMax;	// to allow to set endpoints of axes
static double gMaxConcXMax, gMaxConcYMax;	// to allow to set endpoints of axes
static double gComboConcXMax, gComboConcYMax;	// to allow to set endpoints of axes
static Boolean gSetDepthAxes = false, gSetAvConcAxes = false, gSetMaxConcAxes = false, gSetComboConcAxes = false;

void TextDraw(char *s, short just)
{
	short w;
	
	TextFont(gFont); TextSize(gSize); TextFace(normal);
	w = stringwidth(s);
	
	// adjust the starting position if centered or right just
	switch (just) {
		case kLeftJust: break;
		case kCenterJust: Move(-w / 2, 0); break;
		case kRightJust: Move(-w, 0); break;
	}
	
	drawstring(s);
}

void GetSpeciesName(short species, char *name)
{
	switch (species) {
		case ADULTFISH: strcpy(name, "Adult Fish"); break;
		case CRUSTACEANS: strcpy(name, "Crustaceans"); break;
		case SENSLIFESTAGES: strcpy(name, "Sens. Life Stages"); break;
		case ADULTCORAL: strcpy(name, "Adult Coral"); break;
		case STRESSEDCORAL: strcpy(name, "Stressed Coral"); break;
		case CORALEGGS: strcpy (name, "Coral Eggs"); break;
		case SEAGRASS: strcpy (name, "Sea Grass"); break;
		//case OIL_CONSERVATIVE: strcpy (name, kConservStr); break;
		//default: strcpy(name, kUnknownStr); break;
		default: strcpy(name, "Unknown"); break;
	}
}

void GetConcernLevelStr(short concernLevel, char *name)
{
	switch (concernLevel) {
		case HIGHCONCERN: strcpy(name, "High Concern"); break;
		case MEDIUMCONCERN: strcpy(name, "Medium Concern"); break;
		case LOWCONCERN: strcpy(name, "Low Concern"); break;
		default: strcpy(name, "Unknown"); break;
	}
}

void SetThresholdLevel(short species, short concernLevel, OverLayInfo *thresholdLevels)
{
	// all species have levels set at the same times
	(*thresholdLevels).xOverLay[0] = 3;
	(*thresholdLevels).xOverLay[1] = 24;
	(*thresholdLevels).xOverLay[2] = 96;
	
	// default zero out the values
	(*thresholdLevels).yOverLay[0] = 0;
	(*thresholdLevels).yOverLay[1] = 0;
	(*thresholdLevels).yOverLay[2] = 0;

	(*thresholdLevels).showOverLay = false;	// this will be set when item is selected

	switch (species) {
		case ADULTFISH: 
		{
			//thresholdLevels.xOverLay[0] = 16.125; // corresponds to Fish with max of 20
			if(concernLevel==LOWCONCERN)
			{
				(*thresholdLevels).yOverLay[0] = 10.;
				//thresholdLevels.yOverLay[0] = 20.;
				(*thresholdLevels).yOverLay[1] = 1.;
				//thresholdLevels.yOverLay[1] = 5.;
				(*thresholdLevels).yOverLay[2] = .5;
				strcpy((*thresholdLevels).labelStr,"Low Concern");
			}
			else if(concernLevel==MEDIUMCONCERN)
			{
				(*thresholdLevels).yOverLay[0] = 50.;
				(*thresholdLevels).yOverLay[1] = 2.;
				(*thresholdLevels).yOverLay[2] = 1.;
				strcpy((*thresholdLevels).labelStr,"Medium Concern");
			}
			else if(concernLevel==HIGHCONCERN)
			{
				(*thresholdLevels).yOverLay[0] = 100.;
				(*thresholdLevels).yOverLay[1] = 10.;
				(*thresholdLevels).yOverLay[2] = 1.;
				strcpy((*thresholdLevels).labelStr,"High Concern");
			}
			//(*thresholdLevels).showOverLay = false;
			break;
		}
		case CRUSTACEANS: 
		{
			if(concernLevel==1)
			{
				(*thresholdLevels).yOverLay[0] = 5.;
				(*thresholdLevels).yOverLay[1] = .5;
				(*thresholdLevels).yOverLay[2] = .5;
			}
			else if(concernLevel==2)
			{
				(*thresholdLevels).yOverLay[0] = 10.;
				(*thresholdLevels).yOverLay[1] = 2.;
				(*thresholdLevels).yOverLay[2] = 1.;
				strcpy((*thresholdLevels).labelStr,"Medium Concern");
			}
			else if(concernLevel==3)
			{
				(*thresholdLevels).yOverLay[0] = 50.;
				(*thresholdLevels).yOverLay[1] = 5.;
				(*thresholdLevels).yOverLay[2] = 1.;
			}
			//(*thresholdLevels).showOverLay = false;
			break;
		}
		case SENSLIFESTAGES: 
		{
			if(concernLevel==1)	
			{
				(*thresholdLevels).yOverLay[0] = 1.;
				(*thresholdLevels).yOverLay[1] = .5;
				(*thresholdLevels).yOverLay[2] = .01;
				strcpy((*thresholdLevels).labelStr,"Low Concern");
			}
			else if(concernLevel==2)	
			{
				(*thresholdLevels).yOverLay[0] = 5.;
				(*thresholdLevels).yOverLay[1] = 1.;
				(*thresholdLevels).yOverLay[2] = .5;
				strcpy((*thresholdLevels).labelStr,"Medium Concern");
			}
			else if(concernLevel==3)	
			{
				(*thresholdLevels).yOverLay[0] = 10.;
				(*thresholdLevels).yOverLay[1] = 1.;
				(*thresholdLevels).yOverLay[2] = 1.;
				strcpy((*thresholdLevels).labelStr,"High Concern");
			}
			//(*thresholdLevels).showOverLay = false;
			break;
		}
		case ADULTCORAL: 
		{
			if(concernLevel==LOWCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = 1.;
				(*thresholdLevels).yOverLay[1] = .5;
				(*thresholdLevels).yOverLay[2] = .5;
				strcpy((*thresholdLevels).labelStr,"Low Concern");
			}
			else if(concernLevel==MEDIUMCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = 10.;
				(*thresholdLevels).yOverLay[1] = 5.;
				(*thresholdLevels).yOverLay[2] = 1.;
				strcpy((*thresholdLevels).labelStr,"Medium Concern");
			}
			else if(concernLevel==HIGHCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = 50.;
				(*thresholdLevels).yOverLay[1] = 5.;
				(*thresholdLevels).yOverLay[2] = 5.;
				strcpy((*thresholdLevels).labelStr,"High Concern");
			}
			//(*thresholdLevels).showOverLay = false;
			break;
		}
		case STRESSEDCORAL: 
		{
			if(concernLevel==LOWCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = 1.;
				(*thresholdLevels).yOverLay[1] = .5;
				(*thresholdLevels).yOverLay[2] = .5;
				strcpy((*thresholdLevels).labelStr,"Low Concern");
			}
			else if(concernLevel==MEDIUMCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = 5.;
				(*thresholdLevels).yOverLay[1] = 1.;
				(*thresholdLevels).yOverLay[2] = .5;
				strcpy((*thresholdLevels).labelStr,"Medium Concern");
			}
			else if(concernLevel==HIGHCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = 50.;
				(*thresholdLevels).yOverLay[1] = 5.;
				(*thresholdLevels).yOverLay[2] = 1.;
				strcpy((*thresholdLevels).labelStr,"High Concern");
			}
			//(*thresholdLevels).showOverLay = false;
			break;
		}
		case CORALEGGS: 
		{
			if(concernLevel==LOWCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = .01;
				(*thresholdLevels).yOverLay[1] = .01;
				(*thresholdLevels).yOverLay[2] = .01;
				strcpy((*thresholdLevels).labelStr,"Low Concern");
			}
			else if(concernLevel==MEDIUMCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = .01;
				(*thresholdLevels).yOverLay[1] = .01;
				(*thresholdLevels).yOverLay[2] = .01;
				strcpy((*thresholdLevels).labelStr,"Medium Concern");
			}
			else if(concernLevel==HIGHCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = .5;
				(*thresholdLevels).yOverLay[1] = .01;
				(*thresholdLevels).yOverLay[2] = .01;
				strcpy((*thresholdLevels).labelStr,"High Concern");
			}
			//(*thresholdLevels).showOverLay = false;
			break;
		}
		case SEAGRASS: 
		{
			if(concernLevel==LOWCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = 35.;
				(*thresholdLevels).yOverLay[1] = 15.;
				(*thresholdLevels).yOverLay[2] = 5.;
				strcpy((*thresholdLevels).labelStr,"Low Concern");
			}
			else if(concernLevel==MEDIUMCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = 80.;
				(*thresholdLevels).yOverLay[1] = 25.;
				(*thresholdLevels).yOverLay[2] = 5.;
				strcpy((*thresholdLevels).labelStr,"Medium Concern");
			}
			else if(concernLevel==HIGHCONCERN)	
			{
				(*thresholdLevels).yOverLay[0] = 200.;
				(*thresholdLevels).yOverLay[1] = 50.;
				(*thresholdLevels).yOverLay[2] = 10.;
				strcpy((*thresholdLevels).labelStr,"High Concern");
			}
			//(*thresholdLevels).showOverLay = false;
			break;
		}
		default: 
		{
			break;
		}
	}
}

OSErr SetOverLayDataBase()
{
	long i;
	OSErr err = 0;
	if (gOverLayDataBase) return 0;
	gOverLayDataBase = (OverLayDataBaseH)_NewHandleClear(kNumSpecies*sizeof(OverLayDataBase)); err=(short)_MemError();
	if(err){ TechError("SetOverLayDataBase()","_NewHandle()",0); return -1; }
	//enum { ADULTFISH = 1, CRUSTACEANS, SENSLIFESTAGES, ADULTCORAL, STRESSEDCORAL, CORALEGGS, SEAGRASS };
	// store all the data and retrieve as needed
	for (i=0;i<kNumSpecies;i++)
	{
		GetSpeciesName(i+1,(*gOverLayDataBase)[i].speciesTypeStr);
		SetThresholdLevel(i+1,HIGHCONCERN,&((*gOverLayDataBase)[i].highConcern));
		SetThresholdLevel(i+1,MEDIUMCONCERN,&((*gOverLayDataBase)[i].mediumConcern));
		SetThresholdLevel(i+1,LOWCONCERN,&((*gOverLayDataBase)[i].lowConcern));
	}
	return err;
}

double GetMaxYVal(long species, long levelOfConcern)
{
	double yval = 0;
	if (levelOfConcern == LOWCONCERN)
	{
		yval = (*gOverLayDataBase)[species-1].lowConcern.yOverLay[0];
	}
	else if (levelOfConcern == MEDIUMCONCERN)
	{
		yval = (*gOverLayDataBase)[species-1].mediumConcern.yOverLay[0];
	}
	else if (levelOfConcern == HIGHCONCERN)
	{
		yval = (*gOverLayDataBase)[species-1].highConcern.yOverLay[0];
	}
		
	return yval;
}

double GetOverLayMaxY()
{
	long i;
	double yval = 0, ymax = 0;
	// look at which type of check boxes are showing
	// if species, loop through the gmultspecies array to get species indices and use gLevelofConcern
	if (gShowSpeciesCheckBoxes)
	{
		for (i=0;i<kNumSpecies;i++)
		{
			if (gMultSpecies[i]) yval = GetMaxYVal(i+1,gLevelOfConcern);
			if (yval > ymax) ymax = yval;
		}
	
	}
	else	// if level of concern do the inverse
	{
		for (i=0;i<kNumConcernLevels;i++)
		{
			if (gMultConcernLevels[i]) yval = GetMaxYVal(gSpecies,i+1);
			if (yval > ymax) ymax = yval;
		}
	}

	return ymax;
}

OSErr GetThresholdLevel(OverLayInfo *thresholdLevels,short speciesType,short levelOfConcern)
{
	long i;
	OSErr err = 0;
	if (speciesType>kNumSpecies || speciesType<1)
	{printError("Species not valid"); return -1;}
	if (levelOfConcern>kNumConcernLevels || levelOfConcern<1)
	{printError("Level of concern not valid"); return -1;}
	if (levelOfConcern==HIGHCONCERN)
	{
		for (i=0;i<3;i++)
		{
			(*thresholdLevels).xOverLay[i] = (*gOverLayDataBase)[speciesType-1].highConcern.xOverLay[i];
			(*thresholdLevels).yOverLay[i] = (*gOverLayDataBase)[speciesType-1].highConcern.yOverLay[i];
		}
	}
	else if (levelOfConcern==MEDIUMCONCERN)
	{
		for (i=0;i<3;i++)
		{
			(*thresholdLevels).xOverLay[i] = (*gOverLayDataBase)[speciesType-1].mediumConcern.xOverLay[i];
			(*thresholdLevels).yOverLay[i] = (*gOverLayDataBase)[speciesType-1].mediumConcern.yOverLay[i];
		}
	}
	else if (levelOfConcern==LOWCONCERN)
	{
		for (i=0;i<3;i++)
		{
			(*thresholdLevels).xOverLay[i] = (*gOverLayDataBase)[speciesType-1].lowConcern.xOverLay[i];
			(*thresholdLevels).yOverLay[i] = (*gOverLayDataBase)[speciesType-1].lowConcern.yOverLay[i];
		}
	}
	else
	{
		(*thresholdLevels).xOverLay[0] = 3;
		(*thresholdLevels).xOverLay[1] = 24;
		(*thresholdLevels).xOverLay[2] = 96;
	
		(*thresholdLevels).yOverLay[0] = 0;
		(*thresholdLevels).yOverLay[1] = 0;
		(*thresholdLevels).yOverLay[2] = 0;
	}
	return err;
}

void ResetOverLays(GrafValHdl theGrafHdl)
{
	long i,j;
	double xval[3],yval[3];
	OverLayInfo toxicityInfo;

	for (i=0;i<3;i++)
	{
		toxicityInfo = GetToxicityThresholds(theGrafHdl,i);
		if (!toxicityInfo.showOverLay) return;	// there will be no more overlays
		//if (!toxicityInfo.showOverLay) continue;	// there will be no more overlays
		for (j=0;j<3;j++)
		{
			if (gTriSelected)
				// switch to gOverLayStartTime
				toxicityInfo.xOverLay[j] = toxicityInfo.xOverLay[j] + gXminVal;	// shift the overlay to start where plume first passes over selected area
			
			toxicityInfo.xOverLay[j] = toxicityInfo.xOverLay[j] + gOverLayStartTime - 3;	// shift the overlay to start where plume first passes over selected area

			xval[j] = toxicityInfo.xOverLay[j];
			yval[j] = toxicityInfo.yOverLay[j];
		}
		if (gOverLayEndTime >= xval[2]) return;	// overlays are within the timescale of the data
		else if (gOverLayEndTime >= xval[1])
		{
			// reset third overlay to be xmax, y(xmax)
			toxicityInfo.yOverLay[2] = yval[1] + ((yval[2]-yval[1])*(gOverLayEndTime-xval[1]))/(xval[2]-xval[1]);
			toxicityInfo.xOverLay[2] = gOverLayEndTime;
		}
		else if (gOverLayEndTime >= xval[0])
		{
			//reset third overlay to be xmax, y(xmax)
			toxicityInfo.yOverLay[2] = yval[0] + ((yval[1]-yval[0])*(gOverLayEndTime-xval[0]))/(xval[1]-xval[0]);
			toxicityInfo.xOverLay[2] = gOverLayEndTime;
			// reset second overlay to be midpoint between the first and second point
			toxicityInfo.yOverLay[1] = (toxicityInfo.yOverLay[2]+toxicityInfo.yOverLay[0])/2.;
			toxicityInfo.xOverLay[1] = (toxicityInfo.xOverLay[2]+toxicityInfo.xOverLay[0])/2.;
			//reset third overlay point to be second overlay point
			//toxicityInfo.yOverLay[2] = yval[1];
			//toxicityInfo.xOverLay[2] = xval[1];
			// reset second overlay to be midpoint between the first and second point
			//toxicityInfo.yOverLay[1] = (toxicityInfo.yOverLay[2]+toxicityInfo.yOverLay[0])/2.;
			//toxicityInfo.xOverLay[1] = (toxicityInfo.xOverLay[2]+toxicityInfo.xOverLay[0])/2.;
		}
		else // xmax < val1
		{
			// reset overlays to max out at 3 ??
			// do nothing ??
			// give a note that overlays won't be shown??
			// shouldn't happen
			toxicityInfo.yOverLay[2] = yval[1];
			toxicityInfo.xOverLay[2] = xval[1];
			// reset second overlay to be midpoint between the first and second point
			toxicityInfo.yOverLay[1] = (toxicityInfo.yOverLay[2]+toxicityInfo.yOverLay[0])/2.;
			toxicityInfo.xOverLay[1] = (toxicityInfo.xOverLay[2]+toxicityInfo.xOverLay[0])/2.;
		}
		SetToxicityThresholds(theGrafHdl,toxicityInfo,i);
	}
	return;
}

OSErr ConcPlot(Rect *drawrect, long strtpnt, long endpnt, short plottype, short usegrid, Boolean printing, Rect *rGrid, ExRect *plotBnds)
{
	double ymin, ymax, xmin, xmax, val, non_log_y_max;
	double xprim, xscnd, yprim, yscnd, val_range, val_frac, mult_scale=1.0;
	GrafVal **grafhdl=nil;
	PenState	pensave;
	double **xdata=nil, **ydata=nil;
	OverLayInfo thresholdLevels;
	double overLayMaxX, overLayMaxY;
	char xtitle[64]="", ytitle[64]="", graftitle[64]="", grafsubtitle[64]=""; 
	char str[32]="", ustr[16]="";
	short	val_pow=0, errtype=0, ndcmlplcs=0;
	long i, npnts;
	TextInfo	obj_text;
	DateTimeRec time;
	Seconds timeZero;
	outputData data;
	OSErr err = 0;

	SetWatchCursor();

	// Port Info
	GetPenState(&pensave); PenNormal();

	// init
	npnts	= (endpnt-strtpnt+1);
	//npnts = _GetHandleSize((Handle)gOilConcHdl)/sizeof(outputData);	// check on it
	

	obj_text.theFont = gFont;
	obj_text.theSize = gSize;
	obj_text.theJust = kCenterJust;

	// can't plot fewer than two points
	//if(npnts < 2){ SysBeep(1); goto done; }	// just show an 'x' at the spot

	// check for dialog button states (function called from dialog)
	if(plottype == PLOTDLG_AVCONCPLOTBTN)
	{
		strcpy(ustr,"mg/L");
		errtype = kAvConc; // may not need this
	}
	else if(plottype == PLOTDLG_MAXCONCPLOTBTN)
	{
		strcpy(ustr,"mg/L");
		errtype = kMaxConc;	
	}
	else if(plottype == PLOTDLG_COMBOCONCPLOTBTN)
	{
		strcpy(ustr,"mg/L");
		errtype = kComboConc;	
		npnts = 2*npnts;
	}
	else if(plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
	{
		errtype = kDepthSlice;
		npnts = gDepthSlice[0];
	}

	// Init new graph
	grafhdl = InitGraph(false,false);	// no horiz or vert scroll bars
	if(grafhdl == nil){ goto done; }

	// Init xdata array
	xdata = (double **)_NewHandleClear(npnts*sizeof(double)); err=(short)_MemError();
	if(err){ TechError("ConcPlot()","_NewHandle()",0); goto done; }

	// Init ydata array
	ydata = (double **)_NewHandleClear(npnts*sizeof(double)); err=(short)_MemError();
	if(err){ TechError("ConcPlot()","_NewHandle()",0); goto done; }

	// Get axis and graph titles
	strcpy(xtitle,"Time (Hours since Dispersant Applied)");
	switch( errtype ){
		case kDepthSlice:{	// change to area or depth slice
			//strcpy(xtitle,"Depth level (1m per level)");
			//strcpy(ytitle,"Number of LEs");
			strcpy(ytitle,"Depth level (1m per level)");
			//strcpy(xtitle,"Number of LEs");
			strcpy(xtitle,"ppm");
			if (gShowDepthPlot)
				sprintf(graftitle,"%s","Depth Profile at Selected Triangle");
			else
				sprintf(graftitle,"%s","Depth Profile at Triangle with Max Concentration");
			break;}
		case kTriArea:{	// change to area or depth slice
			sprintf(ytitle,"%s%s%s","Error(",ustr,")");
			mult_scale = 1.0;
			sprintf(graftitle,"%s","Relative Volume Error vs. Vertex");
			break;}
		case kAvConc:{
			sprintf(ytitle,"%s%s%s","Concentration (",ustr,")");
			if (gTriSelected)
				sprintf(graftitle,"%s","Average Concentration Over Selected Triangles vs. Time");
			else
				sprintf(graftitle,"%s","Average Concentration Following Plume vs. Time");
			if (gContourDepth1 == BOTTOMINDEX)
				//sprintf(grafsubtitle, "Depth range : Bottom Layer (1 meter)");
				sprintf(grafsubtitle, "Depth range :  Bottom Layer (%g meters)",gBottomRange);
			else
				sprintf(grafsubtitle, "Depth range : %g to %g meters",gContourDepth1,gContourDepth2);
			break;}
		case kMaxConc:{
			sprintf(ytitle,"%s%s%s","Concentration (",ustr,")");
			//mult_scale = 1.0;
			if (gTriSelected)
				sprintf(graftitle,"%s","Maximum Concentration Over Selected Triangles vs. Time");
			else
				sprintf(graftitle,"%s","Maximum Concentration Following Plume vs. Time");
			if (gContourDepth1 == BOTTOMINDEX)
				//sprintf(grafsubtitle, "Depth range :  Bottom Layer (1 meter)");
				sprintf(grafsubtitle, "Depth range :  Bottom Layer (%g meters)",gBottomRange);
			else
				sprintf(grafsubtitle, "Depth range :  %g to %g meters",gContourDepth1,gContourDepth2);
			break;}
		case kComboConc:{
			(**grafhdl).multipleCurves = true;
			sprintf(ytitle,"%s%s%s","Concentration (",ustr,")");
			//mult_scale = 1.0;
			if (gTriSelected)
				sprintf(graftitle,"%s","Av and Max Concentration Over Selected Triangles vs. Time");
			else
				sprintf(graftitle,"%s","Av and Max Concentration Following Plume vs. Time");
			if (gContourDepth1 == BOTTOMINDEX)
				//sprintf(grafsubtitle, "Depth range :  Bottom Layer (1 meter)");
				sprintf(grafsubtitle, "Depth range :  Bottom Layer (%g meters)",gBottomRange);
			else
				sprintf(grafsubtitle, "Depth range :  %g to %g meters",gContourDepth1,gContourDepth2);
			break;}
	}

	// assign data to x array ( vertex number w/ +1 offset )
	//for(i = 0 ; i < npnts ; i++){ (*xdata)[i] = (double)(i+strtpnt); }
	//timeZero = INDEXH(gOilConcHdl,0).time;
	for (i = 0; i < npnts; i++)
	{ 
		if (errtype==kDepthSlice)
			(*xdata)[i]=gDepthSlice[i+1];
			//(*xdata)[i]=i+1;
		else
		{
			timeZero = INDEXH(gOilConcHdl,0).time;
			if (errtype==kComboConc)
			{
				long index,n_2;
				n_2 = npnts/2;
				if (i<n_2) index = i; else index = i-n_2;
				data = INDEXH(gOilConcHdl,index);
				(*xdata)[i] = ((double)(data.time - timeZero)) / 3600.; 	// hours since dispersed
				//(*xdata)[i+npnts] = ((double)(data.time - timeZero)) / 3600.; 	// hours since dispersed
			}
			else
			{
				data = INDEXH(gOilConcHdl,i);
				(*xdata)[i] = ((double)(data.time - timeZero)) / 3600.; 	// hours since dispersed
			}
		}
	}

	// assign data to y array
	if (plottype == PLOTDLG_AVCONCPLOTBTN)
	{
		for (i = 0; i < npnts; i++)
		{
			data = INDEXH(gOilConcHdl,i);
			//val = i;
			//if (val > 0.0001){ (*ydata)[i] = val; }
			//else{ (*ydata)[i] = 0.0001; }
			// if (gTriSelected) shift data back to first non-zero value, reduce npnts and x values
			(*ydata)[i] = data.avOilConcOverSelectedTri;	// put in some sort of minimum?
		}
	}
	else if (plottype == PLOTDLG_MAXCONCPLOTBTN)
	{
		for (i = 0; i < npnts; i++)
		{
			data = INDEXH(gOilConcHdl,i);
			//val = 2*i;
			//if (val > 0.0001){ (*ydata)[i] = val; }
			//else{ (*ydata)[i] = 0.0001; }
			(*ydata)[i] = data.maxOilConcOverSelectedTri;	// put in some sort of minimum?
		}
	}

	else if (plottype == PLOTDLG_COMBOCONCPLOTBTN)
	{
		for (i = 0; i < npnts; i++)
		{
			long index,n_2;
			n_2 = npnts/2;
			if (i<n_2) index = i; else index = i-n_2;
			data = INDEXH(gOilConcHdl,index);
			//val = 2*i;
			//if (val > 0.0001){ (*ydata)[i] = val; }
			//else{ (*ydata)[i] = 0.0001; }
			if (i<n_2) (*ydata)[i] = data.avOilConcOverSelectedTri;	// put in some sort of minimum?
			else (*ydata)[i] = data.maxOilConcOverSelectedTri;	// put in some sort of minimum?
		}
	}

	else if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
	{
		for (i = 0; i < npnts; i++)
		{
			//(*ydata)[i]=gDepthSlice[i+1];
			//(*ydata)[i]=i+1;
			//(*ydata)[i]=-i-1;
			(*ydata)[i]=-i-.5;	// each ppm value really corresponds to a 1m layer, not a point
			// could reset npts to gDepthMin based on the number of depth points with data
			//if ((*xdata)[i]>0 && (*ydata)[i] < gDepthMin) gDepthMin = (*ydata)[i];
		}
	}

	/*else if (plottype == PLOTDLG_COMBOPLOTBTN)
	{
		for (i = 0; i < npnts; i++)
		{
			(*ydata)[i]=gDepthSlice[i+1];
		}
	}*/
	if (gTriSelected)
	{
		for (i = 0; i < npnts; i++)
		{
			if ((*ydata)[i] != 0.)
			{
				gXminVal = (*xdata)[i];	// re-think for combination natural and chemical
				break;
			}
		}
	}

	if (gOverLay)
	{
		long numOverLays = 0;
		if (gShowSpeciesCheckBoxes)
		{
			char speciesName[64],levelOfConcernStr[64];
			for (i=0;i<kNumSpecies;i++)
			{
				if (gMultSpecies[i])
				{
					GetSpeciesName(i+1,speciesName);
					numOverLays++;
					if (numOverLays>3) {printNote("User error"); break;}
					if (err = GetThresholdLevel(&thresholdLevels,i+1,gLevelOfConcern)) break;
					thresholdLevels.showOverLay = true;
					strcpy(thresholdLevels.labelStr,speciesName);
					SetToxicityThresholds(grafhdl,thresholdLevels, numOverLays-1);
					GetConcernLevelStr(gLevelOfConcern,levelOfConcernStr);
					strcpy((**grafhdl).LegendStr,levelOfConcernStr);
				}
			}
		}
		else
		{
			char levelOfConcernStr[64],speciesName[64];
			for (i=0;i<kNumConcernLevels;i++)
			{
				if (gMultConcernLevels[i])
				{
					GetConcernLevelStr(i+1,levelOfConcernStr);
					numOverLays++;
					if (numOverLays>3) {printNote("User error"); break;}
					if (err = GetThresholdLevel(&thresholdLevels,gSpecies,i+1)) break;
					thresholdLevels.showOverLay = true;
					strcpy(thresholdLevels.labelStr,levelOfConcernStr);
					SetToxicityThresholds(grafhdl,thresholdLevels, numOverLays-1);
					GetSpeciesName(gSpecies,speciesName);
					strcpy((**grafhdl).LegendStr,speciesName);
				}
			}
		}
	}
	overLayMaxX = 96;
	
	overLayMaxY = GetOverLayMaxY();	// may want to set by hand, initial x,y value so doesn't dwarf graph
	// should also allow for a shift of entire overlay
	
	(**grafhdl).ShowToxicityThresholds = false;
	if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
	{
		(**grafhdl).ShowToxicityThresholds = false;
	}
	else if (gOverLay) 
		(**grafhdl).ShowToxicityThresholds = true;
		
	// turn this into a flag for which check boxes are showing
	//(**grafhdl).LevelOfConcern = gLevelOfConcern;	// will need to relate this to levels

	_HLock((Handle)xdata); _HLock((Handle)ydata);
	ArrayMinMaxDouble((double *)(*xdata),npnts,&xmin,&xmax);
	ArrayMinMaxDouble((double *)(*ydata),npnts,&ymin,&ymax);
	//ymin = 0.;	// check vs overlay, but don't turn into log based on overlay
		if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
		{
			ymax = 0.;
			xmin = 0.;
			// use xmax, ymin as set on dialog
			//xmax = 100.;
			if (gSetDepthAxes)
			{
				if (gContourDepth1 == BOTTOMINDEX)
					ymax = gDepthYMin;
				else
					ymin = gDepthYMin;
				xmax = gDepthXMax;
			}
		}
		else
		{
			ymin = 0.;
			if (gSetAvConcAxes && plottype == PLOTDLG_AVCONCPLOTBTN)
			{
				ymax = gAvConcYMax;
				xmax = gAvConcXMax;
			}
			else if (gSetMaxConcAxes && plottype == PLOTDLG_MAXCONCPLOTBTN)
			{
				ymax = gMaxConcYMax;
				xmax = gMaxConcXMax;
			}
			else if (gSetComboConcAxes && plottype == PLOTDLG_COMBOCONCPLOTBTN)
			{
				ymax = gComboConcYMax;
				xmax = gComboConcXMax;
			}
		}

	// allow to increase max/decrease min only
	if ((**grafhdl).ShowToxicityThresholds)
	//if (overLayMaxX > xmax && (**grafhdl).ShowToxicityThresholds)
	{
		 //xmax = overLayMaxX;
		ResetOverLays(grafhdl); //then reset toxicity thresholds
		if (gOverLayEndTime>xmax) xmax = gOverLayEndTime;
	}
	if (overLayMaxY > ymax && 	(**grafhdl).ShowToxicityThresholds)
		ymax = overLayMaxY;

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

	if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
	{
		SplitDouble(&val_range,&val_frac,&val_pow); xprim = 10.0; // set xprim to 10.0 and pass it in to raise it
	}
	else
	{
		SplitDouble(&val_range,&val_frac,&val_pow); xprim = 12.0; // change to powers of 12 (day segments), maybe not for depth slice?
	}
	Raise2Pow(&xprim,&xprim,val_pow);
 	if (val_frac <= 1.5)
	{
		//xprim /= 5.0;
		//xscnd = xprim/5.0;
		if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
		{
			xprim /= 5.0;
			xscnd = xprim/5.0;
		}
		else	
		{
			xprim /= 6.0;
			xscnd = xprim/6.0;
		}
	}
	else if (val_frac <= 3.0)
	{
		xprim /= 2.0;
		//xscnd = xprim/5.0;
		if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
			xscnd = xprim/5.0;
		else		
			xscnd = xprim/6.0;
	}
	else if (val_frac <= 6.0)
	{
		//xscnd = xprim/5.0;
		if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
			xscnd = xprim/5.0;
		else		
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
		if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
			ymax = 0.;
		else
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
		if (plottype == PLOTDLG_DEPTHSLICEPLOTBTN)
		{
			if (gSetDepthAxes)
			{
				if (gContourDepth1 == BOTTOMINDEX)
					(**grafhdl).maxYLabVal = gDepthYMin;	//ymin
				else
					(**grafhdl).minYLabVal = gDepthYMin;	//ymin
				(**grafhdl).maxXLabVal = gDepthXMax;	//xmax
			}
			else
			{
				gDepthYMin = (**grafhdl).minYLabVal;	//ymin
				gDepthXMax = (**grafhdl).maxXLabVal;	//xmax
			}
			if (npnts==1 && (**grafhdl).minYLabVal > -1) (**grafhdl).minYLabVal = -1;
		}
		else if (plottype == PLOTDLG_AVCONCPLOTBTN)
		{
			if (gSetAvConcAxes)
			{
				(**grafhdl).maxYLabVal = gAvConcYMax;	//ymax
				(**grafhdl).maxXLabVal = gAvConcXMax;	//xmax
			}
			else
			{
				gAvConcYMax = (**grafhdl).maxYLabVal;	//ymax
				gAvConcXMax = (**grafhdl).maxXLabVal;	//xmax
			}
		}
		else if (plottype == PLOTDLG_MAXCONCPLOTBTN)
		{
			if (gSetMaxConcAxes)
			{
				(**grafhdl).maxYLabVal = gMaxConcYMax;	//ymax
				(**grafhdl).maxXLabVal = gMaxConcXMax;	//xmax
			}
			else
			{
				gMaxConcYMax = (**grafhdl).maxYLabVal;	//ymax
				gMaxConcXMax = (**grafhdl).maxXLabVal;	//xmax
			}
		}
		else if (plottype == PLOTDLG_COMBOCONCPLOTBTN)
		{
			if (gSetComboConcAxes)
			{
				(**grafhdl).maxYLabVal = gComboConcYMax;	//ymax
				(**grafhdl).maxXLabVal = gComboConcXMax;	//xmax
			}
			else
			{
				gComboConcYMax = (**grafhdl).maxYLabVal;	//ymax
				gComboConcXMax = (**grafhdl).maxXLabVal;	//xmax
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
		SetDataPlotType(grafhdl,(MyPlotProc)StandardPlot,(MyOffPlotProc)StandardOffPlot);

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
void InvalDItemBox(DialogPtr d, short item)
{
#ifdef MAC
	InvalRectInWindow(GetDialogItemBox(d, item), GetDialogWindow(d));
#else
	InvalidateRect(GetDlgItem(d, item), 0, false);
#endif
}

Boolean PlotClick(DialogPtr d, Point clk)
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
	//PenPat((ConstPatternParam)&GRAY_BRUSH);
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
		npts = _GetHandleSize((Handle)gOilConcHdl)/sizeof(outputData);
		if (newend > npts) newend = npts;
		
		if (newend - newstart >= 8)
		{
			gstrtpnt = newstart;
			gendpnt = newend;
			InvalDItemBox(d, PLOTDLG_PLOTBOX);
		}
	}
	SetPenState(&pensave);
	
	return TRUE;
}

#ifdef IBM

LRESULT CALLBACK PlotProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	short itemID;
	Point p;
	POINT P;
	PAINTSTRUCT ps;
	WindowPtr savePort;
	pascal_ifMac void DrawPlot(DialogPtr d, short frameItem);
	
	itemID = GetDlgCtrlID(hWnd);
	if (!itemID) { SysBeep(1); return 0; }
	
	switch (message) {
		case WM_PAINT:
			GetPort(&savePort);
			SetPortBP(hWnd, &ps);
			SetPortEP(savePort, &ps);
			SetPort(GetParent(hWnd));
			DrawPlot(GetParent(hWnd), PLOTDLG_PLOTBOX);
			SetPort(savePort);
			break;
		
		case WM_LBUTTONDOWN:
			P.x = LOWORD(lParam);
			P.y = HIWORD(lParam);
			MapWindowPoints(hWnd, GetParent(hWnd), &P, 1);
			MakeMacPoint(&P, &p);
			GetPort(&savePort);
			SetPort(GetParent(hWnd));
			PlotClick(GetParent(hWnd), p);
			SetPort(savePort);
			break;
		
		default: return DefWindowProc(hWnd, message, wParam, lParam);
	}
	
	return 0;
}

#endif

pascal_ifMac void DrawPlot(DialogPtr d, short frameItem)
{
	Rect r = GetDialogItemBox(d,frameItem);
	short	err = 0;
	GrafPtr sp;

#ifdef MAC
	GetPortGrafPtr(&sp);
	SetPortDialogPort(d);
#endif
	err = ConcPlot(&r, gstrtpnt, gendpnt, gplottype, gplotgrid, gprintingplot, &ggridrect, &gplotbnds);

#ifdef MAC
	SetPortGrafPort(sp);
#endif
	
	if (err != 0){	// do something ?
		
	}

	return;
}

void PLOTDLG_BTNHANDLER(DialogPtr d, short id)
{
	switch(id){
		case PLOTDLG_MAXCONCPLOTBTN:
		case PLOTDLG_AVCONCPLOTBTN:
		case PLOTDLG_COMBOCONCPLOTBTN:
		case PLOTDLG_DEPTHSLICEPLOTBTN:
			SetButton(d,PLOTDLG_MAXCONCPLOTBTN,0);
			SetButton(d,PLOTDLG_AVCONCPLOTBTN,0);
			SetButton(d,PLOTDLG_COMBOCONCPLOTBTN,0);
			SetButton(d,PLOTDLG_DEPTHSLICEPLOTBTN,0);
			SetButton(d,gplottype,1); 
			break;

		case PLOTDLG_SHOWPLOTGRIDBTN:
			SetButton(d,id,0);
			SetButton(d,id,gplotgrid); 
			break;

		default: 
			break;
	}
}

OSErr PLOTDLG_INIT(DialogPtr d, VOIDPTR data)
{
	short i;
	// assign draw proc handles
	SetDialogItemHandle(d, PLOTDLG_HILITE, (Handle)FrameDefault);
	SetDialogItemHandle(d, PLOTDLG_PLOTBOX, (Handle)DrawPlot);

	// set check boxes
	PLOTDLG_BTNHANDLER(d,gplottype);
	PLOTDLG_BTNHANDLER(d,PLOTDLG_SHOWPLOTGRIDBTN);
	
	if (!gDepthSlice) 
	{
		ShowHideDialogItem(d,PLOTDLG_DEPTHSLICEPLOTBTN,false);
		//ShowHideDialogItem(d,PLOTDLG_TOXICITY,false);
	}
	else if (!gOilConcHdl)	
	{
		ShowHideDialogItem(d,PLOTDLG_MAXCONCPLOTBTN,false);
		ShowHideDialogItem(d,PLOTDLG_AVCONCPLOTBTN,false);
		ShowHideDialogItem(d,PLOTDLG_COMBOCONCPLOTBTN,false);
		ShowHideDialogItem(d,PLOTDLG_TOXICITY,false);
		/*long npts = gDepthSlice[0];
		gDepthXMax = 0;
		gDepthYMin = -npts;
		for (i=npts;i>0;i--)
		{
			if (gDepthSlice[i] != 0) gDepthXMax = gDepthSlice[i];
		}*/
	}
	/*gOverLay = false;	// reset every time
	gOverLayEndTime = 24;	// reset every time
	for (i=0;i<kNumConcernLevels;i++)
	{
		gMultConcernLevels[i] = false;
	}
	for (i=0;i<kNumSpecies;i++)
	{
		gMultSpecies[i] = false;
	}*/
	gstrtpnt = 1;
	//gendpnt = 100;
	if (!gDepthSlice && gplottype == PLOTDLG_DEPTHSLICEPLOTBTN) gplottype = PLOTDLG_AVCONCPLOTBTN;
	if (gplottype != PLOTDLG_DEPTHSLICEPLOTBTN) 
	{
		gendpnt = _GetHandleSize((Handle)gOilConcHdl)/sizeof(outputData);
		ShowHideDialogItem(d,PLOTDLG_TOXICITY,true);
		//ShowHideDialogItem(d,PLOTDLG_SETAXES,false);
		ShowHideDialogItem(d,PLOTDLG_SETAXES,true);
	}
	else
	{
		gendpnt = gDepthSlice[0];
		ShowHideDialogItem(d,PLOTDLG_TOXICITY,false);
		ShowHideDialogItem(d,PLOTDLG_SETAXES,true);
	}
	
	return 0;
}

short PLOTDLG_CLICK(DialogPtr d, short itemHit, long lParam, VOIDPTR data)
{
	char path[256];
	Point p, where;
	Rect r;
	MySFReply reply;
	OSErr err = 0;
	
	switch (itemHit) {
		/////////////////////////////////////////////////
		case PLOTDLG_CANCEL: return PLOTDLG_CANCEL;
		
		/////////////////////////////////////////////////
		case PLOTDLG_HELP: break;	// put something into startscreens
		
		/////////////////////////////////////////////////
		case PLOTDLG_OK: return PLOTDLG_OK;
		
		/////////////////////////////////////////////////
		case PLOTDLG_SHOWPLOTGRIDBTN:{
			if (gplotgrid == 0)
				gplotgrid = 1; 
			else
				gplotgrid = 0; 

			PLOTDLG_BTNHANDLER(d,itemHit);
			gprintingplot = false;
			InvalDItemBox(d, PLOTDLG_PLOTBOX);
			break;
		}
		
		/////////////////////////////////////////////////
		case PLOTDLG_MAXCONCPLOTBTN:{
			if (gplottype != PLOTDLG_MAXCONCPLOTBTN)
			{
				ShowHideDialogItem(d,PLOTDLG_TOXICITY,true);
				//ShowHideDialogItem(d,PLOTDLG_SETAXES,false);
				gplottype = PLOTDLG_MAXCONCPLOTBTN;
				PLOTDLG_BTNHANDLER(d,itemHit);
				gprintingplot = false;
				InvalDItemBox(d, PLOTDLG_PLOTBOX);
			} break;
		}
		
		/////////////////////////////////////////////////
		case PLOTDLG_AVCONCPLOTBTN:{
			if (gplottype != PLOTDLG_AVCONCPLOTBTN) 
			{
				ShowHideDialogItem(d,PLOTDLG_TOXICITY,true);
				//ShowHideDialogItem(d,PLOTDLG_SETAXES,false);
				gplottype = PLOTDLG_AVCONCPLOTBTN;
				PLOTDLG_BTNHANDLER(d,itemHit);
				gprintingplot = false;
				InvalDItemBox(d, PLOTDLG_PLOTBOX);
			} break;
		}
		
		/////////////////////////////////////////////////
		case PLOTDLG_COMBOCONCPLOTBTN:{
			if (gplottype != PLOTDLG_COMBOCONCPLOTBTN) 
			{
				ShowHideDialogItem(d,PLOTDLG_TOXICITY,true);
				//ShowHideDialogItem(d,PLOTDLG_SETAXES,false);
				gplottype = PLOTDLG_COMBOCONCPLOTBTN;
				PLOTDLG_BTNHANDLER(d,itemHit);
				gprintingplot = false;
				InvalDItemBox(d, PLOTDLG_PLOTBOX);
			} break;
		}
		
		/////////////////////////////////////////////////
		case PLOTDLG_DEPTHSLICEPLOTBTN:{
			if (gplottype != PLOTDLG_DEPTHSLICEPLOTBTN) 
			{
				ShowHideDialogItem(d,PLOTDLG_TOXICITY,false);
				ShowHideDialogItem(d,PLOTDLG_SETAXES,true);
				gplottype = PLOTDLG_DEPTHSLICEPLOTBTN;
				PLOTDLG_BTNHANDLER(d,itemHit);
				gprintingplot = false;
				InvalDItemBox(d, PLOTDLG_PLOTBOX);
				//gOverLay = false;
			} break;
		}
		
		/////////////////////////////////////////////////
#ifdef MAC
		case PLOTDLG_PLOTBOX:{
			p = lastEvent.where;
			GlobalToLocal(&p);
			PlotClick(d, p);
			break;
		}
#endif

		/////////////////////////////////////////////////
		case PLOTDLG_TOXICITY:{	// bring up dialog with overlay options
		 	if (!ToxicityOverLayDialog())
			{			
				InvalDItemBox(d, PLOTDLG_PLOTBOX);
			}
			break;
		}
		
		/////////////////////////////////////////////////
		case PLOTDLG_SETAXES:{	// allow to change axes
		 	if (!SetAxesDialog())
			{			
				InvalDItemBox(d, PLOTDLG_PLOTBOX);
			}
			break;
		}
		
		/////////////////////////////////////////////////
		case PLOTDLG_SHOWTABLE:{	// show concentration table
			/*gstrtpnt = 1;
			//gendpnt = 100; // total number of vertex points
			if (gOilConcHdl) gendpnt = _GetHandleSize((Handle)gOilConcHdl)/sizeof(outputData);
			else gendpnt = gDepthSlice[0];*/
			gprintingplot = false;
			InvalDItemBox(d, PLOTDLG_PLOTBOX);
			ConcentrationTable(gOilConcHdl,gDepthSlice,gplottype != PLOTDLG_DEPTHSLICEPLOTBTN ? 1 : 2);	
			break;
		}
		
		/////////////////////////////////////////////////
		case PLOTDLG_SAVEPLOT:{
			char ibmBackwardsTypeStr[32] = "",fileName[64];
			Boolean changeExtension = false;	// for now
			char previousPath[256]="",defaultExtension[3]="";
			where = CenteredDialogUpLeft(M55);
			switch (gplottype) {
				case PLOTDLG_DEPTHSLICEPLOTBTN: strcpy(fileName,"DepthSlice"); break;
				case PLOTDLG_AVCONCPLOTBTN: strcpy(fileName,"AverageConcentration"); break;
				case PLOTDLG_MAXCONCPLOTBTN: strcpy(fileName,"MaxConcentration"); break;
				case PLOTDLG_COMBOCONCPLOTBTN: strcpy(fileName,"AverageMaxConcentration"); break;
			}
			
#ifdef MAC
			strcat(fileName,".pic");
#else
			strcat(fileName,".bmp");
#endif

#if TARGET_API_MAC_CARBON
		//err = AskUserForSaveFilename("graph.pic",path,".pic",true);
		err = AskUserForSaveFilename(fileName,path,".pic",true);
		if (err) /*return USERCANCEL;*/ break;
#else
#ifdef MAC
			//sfputfile(&where, "Name:", "graph.pic", (DlgHookUPP)0, &reply);
			sfputfile(&where, "Name:", fileName, (DlgHookUPP)0, &reply);
#else
			//sfpputfile(&where, ibmBackwardsTypeStr, "graph.bmp", (MyDlgHookProcPtr)0, &reply,
			sfpputfile(&where, ibmBackwardsTypeStr, fileName, (MyDlgHookProcPtr)0, &reply,
						  M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
#endif
			if (!reply.good) break;
			
			my_p2cstr(reply.fName);
			hdelete(reply.vRefNum, 0, (char *)reply.fName);
			
			if (err = hcreate(reply.vRefNum, 0, (char *)reply.fName, 'ttxt', 'PICT'))
				{ TechError("PLOTDLG_CLICK()", "hcreate()", err); return 0; }
			
#ifdef MAC
			GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
#else
			strcpy(path, reply.fName);
#endif
#endif
			// fall through
		}
		case PLOTDLG_PRINTPLOT:
			r = GetDialogItemBox(d, PLOTDLG_PLOTBOX);
			err = PrintPlot(itemHit == PLOTDLG_PRINTPLOT ? 0 : path,
								 itemHit == PLOTDLG_PRINTPLOT,
								 r, gstrtpnt, gendpnt, gplottype, gplotgrid, ggridrect, gplotbnds, true);
			break;
	}
	
	return 0;
}

Boolean PlotDialog(outputData** concHdl, float* depthSlice, float contourDepth1, float contourDepth2, float bottomRange, Boolean triSelected, Boolean showDepthPlot)	
{// code goes here, maybe include triangle area or vertical cross section
	short item;
	if (concHdl || depthSlice)
	{
		gOilConcHdl = concHdl;
		gContourDepth1 = contourDepth1;
		gContourDepth2 = contourDepth2;
		gBottomRange = bottomRange;
		gTriSelected = triSelected;
		gDepthSlice = depthSlice;
		if (!concHdl) showDepthPlot = true;
		gShowDepthPlot = showDepthPlot;
		if (showDepthPlot) gplottype = PLOTDLG_DEPTHSLICEPLOTBTN;
	}
	else
		return false;

	if (SetOverLayDataBase()) return false;	// but when to delete? - part of ptcurmap??
	item = MyModalDialog(PLOTDLG, mapWindow, 0, PLOTDLG_INIT, PLOTDLG_CLICK);
	
	if(gOverLayDataBase){
		DisposeHandle((Handle)gOverLayDataBase);
		gOverLayDataBase = 0;
	}

	if (item == PLOTDLG_OK) 
		return true;
	
	return false;
}

static Boolean gOverLayFish, gOverLayCrustaceans, gOverLaySensLifeStage;	
static Boolean gOverLayAdultCoral, gOverLayStressedCoral, gOverLayCoralEggs, gOverLaySeaGrass;	
static Boolean gOverLayHigh, gOverLayMedium, gOverLayLow;	
static PopInfoRec ToxicityPopTable[] = {
		{ TOXICITY, nil, TOXICITY_CHECKBOXTYPE, 0, pCHECKBOXTYPE, 0, 1, FALSE, nil },
		{ TOXICITY, nil, TOXICITY_LEVELOFCONCERN, 0, pLEVELOFCONCERN, 0, 1, FALSE, nil },
		{ TOXICITY, nil, TOXICITY_SPECIES, 0, pSPECIES, 0, 1, FALSE, nil }
		};

void ShowHideToxicityDialogItems(DialogPtr dialog)
{
	Boolean showSpeciesCheckBoxes = true;
	short typeOfInfoSpecified = GetPopSelection(dialog, TOXICITY_CHECKBOXTYPE);
	if (typeOfInfoSpecified == 1) showSpeciesCheckBoxes = false;	// Level of Concern selected

	if (showSpeciesCheckBoxes)
	{
		ToxicityPopTable[1].bStatic = false;
		ToxicityPopTable[2].bStatic = true;
	}
	else
	{
		ToxicityPopTable[1].bStatic = true;
		ToxicityPopTable[2].bStatic = false;
	}

	if (showSpeciesCheckBoxes)
	{
		ShowHideDialogItem(dialog, TOXICITY_FISH, true ); 
		ShowHideDialogItem(dialog, TOXICITY_CRUSTACEANS, true); 
		ShowHideDialogItem(dialog, TOXICITY_SENSLIFESTAGE, true); 
		ShowHideDialogItem(dialog, TOXICITY_ADULTCORAL, true); 
		ShowHideDialogItem(dialog, TOXICITY_STRESSEDCORAL, true); 
		ShowHideDialogItem(dialog, TOXICITY_CORALEGGS, true); 
		ShowHideDialogItem(dialog, TOXICITY_SEAGRASS, true); 
	
		ShowHideDialogItem(dialog, TOXICITY_LOW, false); 
		ShowHideDialogItem(dialog, TOXICITY_MEDIUM, false); 
		ShowHideDialogItem(dialog, TOXICITY_HIGH, false); 

		ShowHideDialogItem(dialog, TOXICITY_LEVELOFCONCERN, true); 
		ShowHideDialogItem(dialog, TOXICITY_SPECIES, false); 
	}
	else // showLevelOfConcernCheckBoxes
	{
		ShowHideDialogItem(dialog, TOXICITY_FISH, false ); 
		ShowHideDialogItem(dialog, TOXICITY_CRUSTACEANS, false); 
		ShowHideDialogItem(dialog, TOXICITY_SENSLIFESTAGE, false); 
		ShowHideDialogItem(dialog, TOXICITY_ADULTCORAL, false); 
		ShowHideDialogItem(dialog, TOXICITY_STRESSEDCORAL, false); 
		ShowHideDialogItem(dialog, TOXICITY_CORALEGGS, false); 
		ShowHideDialogItem(dialog, TOXICITY_SEAGRASS, false); 
	
		ShowHideDialogItem(dialog, TOXICITY_LOW, true); 
		ShowHideDialogItem(dialog, TOXICITY_MEDIUM, true); 
		ShowHideDialogItem(dialog, TOXICITY_HIGH, true); 

		ShowHideDialogItem(dialog, TOXICITY_LEVELOFCONCERN, false); 
		ShowHideDialogItem(dialog, TOXICITY_SPECIES, true); 
	}

	if(showSpeciesCheckBoxes)
	{
		PopDraw(dialog, TOXICITY_LEVELOFCONCERN);
	}
	else
	{
		PopDraw(dialog, TOXICITY_SPECIES);
	}
}

void SetButtonFlags()
{
	gOverLayFish = gShowSpeciesCheckBoxes && gMultSpecies[ADULTFISH-1];
	gOverLayCrustaceans = gShowSpeciesCheckBoxes && gMultSpecies[CRUSTACEANS-1];
	gOverLaySensLifeStage = gShowSpeciesCheckBoxes && gMultSpecies[SENSLIFESTAGES-1];
	gOverLayAdultCoral = gShowSpeciesCheckBoxes && gMultSpecies[ADULTCORAL-1];
	gOverLayStressedCoral = gShowSpeciesCheckBoxes && gMultSpecies[STRESSEDCORAL-1];
	gOverLayCoralEggs = gShowSpeciesCheckBoxes && gMultSpecies[CORALEGGS-1];
	gOverLaySeaGrass = gShowSpeciesCheckBoxes && gMultSpecies[SEAGRASS-1];

	gOverLayHigh = !gShowSpeciesCheckBoxes && gMultConcernLevels[HIGHCONCERN-1];
	gOverLayMedium = !gShowSpeciesCheckBoxes && gMultConcernLevels[MEDIUMCONCERN-1];
	gOverLayLow = !gShowSpeciesCheckBoxes && gMultConcernLevels[LOWCONCERN-1];
}

OSErr TOXICITYInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	RegisterPopTable (ToxicityPopTable, sizeof (ToxicityPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (TOXICITY, dialog);
	
	// assign draw proc handles
	SetDialogItemHandle(dialog, TOXICITY_HILITE, (Handle)FrameDefault);

	SetButtonFlags();
	// set check boxes
	SetButton (dialog, TOXICITY_FISH, gOverLayFish);
	SetButton (dialog, TOXICITY_CRUSTACEANS, gOverLayCrustaceans);
	SetButton (dialog, TOXICITY_SENSLIFESTAGE, gOverLaySensLifeStage);
	SetButton (dialog, TOXICITY_ADULTCORAL, gOverLayAdultCoral);
	SetButton (dialog, TOXICITY_STRESSEDCORAL, gOverLayStressedCoral);
	SetButton (dialog, TOXICITY_CORALEGGS, gOverLayCoralEggs);
	SetButton (dialog, TOXICITY_SEAGRASS, gOverLaySeaGrass);
	
	SetButton (dialog, TOXICITY_LOW, gOverLayLow);
	SetButton (dialog, TOXICITY_MEDIUM, gOverLayMedium);
	SetButton (dialog, TOXICITY_HIGH, gOverLayHigh);
	
	Float2EditText (dialog, TOXICITY_ENDTIME, gOverLayEndTime, 2);
	Float2EditText (dialog, TOXICITY_STARTTIME, gOverLayStartTime, 2);

	ShowHideToxicityDialogItems(dialog);
	
	if (gShowSpeciesCheckBoxes)
	{
		SetPopSelection (dialog, TOXICITY_CHECKBOXTYPE, 2);
	}
	else
	{
		SetPopSelection (dialog, TOXICITY_CHECKBOXTYPE, 1);
	}
	SetPopSelection (dialog, TOXICITY_LEVELOFCONCERN, gLevelOfConcern);
	SetPopSelection (dialog, TOXICITY_SPECIES, gSpecies);

	return 0;
}


short TOXICITYClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	short	menuItemChosen;
	short checkBoxesType;
	long	menuID_menuItem;
	long i, numChecked = 0;

	switch (itemNum) {
		case TOXICITY_CANCEL: return TOXICITY_CANCEL;

		case TOXICITY_OK:
			double xMaxVal,xMinVal;
			xMaxVal = EditText2Float(dialog, TOXICITY_ENDTIME);
			xMinVal = EditText2Float(dialog, TOXICITY_STARTTIME);
			if (xMaxVal<3) {printNote("The toxicity data will not appear for end time less than 3 hours after dispersal"); break;}
			if (xMaxVal>96) {printNote("The toxicity data does not extend beyond 96 hours after dispersal"); break;}
			if (xMinVal>xMaxVal) {printNote("The start time must be smaller than the end time"); break;}
			checkBoxesType = GetPopSelection(dialog, TOXICITY_CHECKBOXTYPE);
			if(checkBoxesType==2) // species check boxes
			{
				//sharedWMover -> bActive = GetButton(dialog, M18ACTIVE);
				// check that no more than 3 boxes are checked
				for (i=0;i<kNumSpecies;i++)
				{
					if (gMultSpecies[i]) numChecked++;
				}
				if (numChecked>3) {printNote("No more than 3 species can be overlayed on one plot"); break;}
				gMultSpecies[ADULTFISH-1] = gOverLayFish;
				gMultSpecies[CRUSTACEANS-1] = gOverLayCrustaceans;
				gMultSpecies[SENSLIFESTAGES-1] = gOverLaySensLifeStage;
				gMultSpecies[ADULTCORAL-1] = gOverLayAdultCoral;
				gMultSpecies[STRESSEDCORAL-1] = gOverLayStressedCoral;
				gMultSpecies[CORALEGGS-1] = gOverLayCoralEggs;
				gMultSpecies[SEAGRASS-1] = gOverLaySeaGrass;
				gShowSpeciesCheckBoxes = true;
				gLevelOfConcern = GetPopSelection(dialog, TOXICITY_LEVELOFCONCERN);
			}
			else // level of concern check boxes
			{
				gMultConcernLevels[LOWCONCERN-1] = gOverLayLow;
				gMultConcernLevels[MEDIUMCONCERN-1] = gOverLayMedium;
				gMultConcernLevels[HIGHCONCERN-1] = gOverLayHigh;
				gShowSpeciesCheckBoxes = false;
				// if selected from pop-up, there will only be one
				gSpecies = GetPopSelection(dialog, TOXICITY_SPECIES);	
			}
			gOverLayEndTime = xMaxVal;
			gOverLayStartTime = xMinVal;
			return TOXICITY_OK;
			
		/////////////////////////////////////////////////
		case TOXICITY_FISH:{
			ToggleButton(dialog, itemNum);
			gOverLayFish = !gOverLayFish;
			break;
		}
		
		/////////////////////////////////////////////////
		case TOXICITY_CRUSTACEANS:{
			ToggleButton(dialog, itemNum);
			gOverLayCrustaceans = !gOverLayCrustaceans;
			break;
		}
		
		/////////////////////////////////////////////////
		case TOXICITY_SENSLIFESTAGE:{
			ToggleButton(dialog, itemNum);
			gOverLaySensLifeStage = !gOverLaySensLifeStage;
			break;
		}
		
		/////////////////////////////////////////////////
		case TOXICITY_ADULTCORAL:{
			ToggleButton(dialog, itemNum);
			gOverLayAdultCoral = !gOverLayAdultCoral;
			break;
		}
		
		/////////////////////////////////////////////////
		case TOXICITY_STRESSEDCORAL:{
			ToggleButton(dialog, itemNum);
			gOverLayStressedCoral = !gOverLayStressedCoral;
			break;
		}
		
		/////////////////////////////////////////////////
		case TOXICITY_CORALEGGS:{
			ToggleButton(dialog, itemNum);
			gOverLayCoralEggs = !gOverLayCoralEggs;
			break;
		}
		
		/////////////////////////////////////////////////
		case TOXICITY_SEAGRASS:{
			ToggleButton(dialog, itemNum);
			gOverLaySeaGrass = !gOverLaySeaGrass;
			break;
		}
		
		/////////////////////////////////////////////////
		case TOXICITY_LOW:{
			ToggleButton(dialog, itemNum);
			gOverLayLow = !gOverLayLow;
			break;
		}
		
		/////////////////////////////////////////////////
		case TOXICITY_MEDIUM:{
			ToggleButton(dialog, itemNum);
			gOverLayMedium = !gOverLayMedium;
			break;
		}
		
		/////////////////////////////////////////////////
		case TOXICITY_HIGH:{
			ToggleButton(dialog, itemNum);
			gOverLayHigh = !gOverLayHigh;
			break;
		}
		/////////////////////////////////////////////////
		case TOXICITY_LEVELOFCONCERN:{
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
		}
		case TOXICITY_CHECKBOXTYPE:{
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowHideToxicityDialogItems(dialog);
			break;
		}
		case TOXICITY_SPECIES:{
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
		}
		case TOXICITY_ENDTIME:
			CheckNumberTextItem(dialog, itemNum, TRUE);	// allowing decimals
			break;
	}

	return 0;
}

OSErr ToxicityOverLayDialog()	
{
	short item, i;
	item = MyModalDialog(TOXICITY, mapWindow, 0, TOXICITYInit, TOXICITYClick);
	if(item == TOXICITY_CANCEL) return USERCANCEL; 
	model->NewDirtNotification();	// is this necessary ?
	if(item == TOXICITY_OK) 
	{
		if (gShowSpeciesCheckBoxes)
		{
			for (i=0;i<kNumSpecies;i++)
			{
				if (gMultSpecies[i]) 
				{
					gOverLay = true;
					return 0;
				}
			}
			gOverLay = false;
			return 0;
		}
		else	// show Concern Level as check boxes
		{
			for (i=0;i<kNumConcernLevels;i++)
			{
				if (gMultConcernLevels[i]) 
				{
					gOverLay = true;
					return 0;
				}
			}
			gOverLay = false;
			return 0;
		}
		return 0; 
	}
	else return -1;
}

void ShowHideSetAxesDialogItems(DialogPtr dialog)
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

OSErr SETAXESInit(DialogPtr dialog, VOIDPTR data)
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
		case PLOTDLG_DEPTHSLICEPLOTBTN: xval = gDepthXMax; yval = fabs(gDepthYMin); setAxes = gSetDepthAxes; break;
		case PLOTDLG_AVCONCPLOTBTN: xval = gAvConcXMax; yval = gAvConcYMax; setAxes = gSetAvConcAxes; break;
		case PLOTDLG_MAXCONCPLOTBTN: xval = gMaxConcXMax; yval = gMaxConcYMax; setAxes = gSetMaxConcAxes; break;
		case PLOTDLG_COMBOCONCPLOTBTN: xval = gComboConcXMax; yval = gComboConcYMax; setAxes = gSetComboConcAxes; break;
	}
	Float2EditText (dialog, SETAXES_MAXX, xval, 2);
	Float2EditText (dialog, SETAXES_MINY, yval, 2);
	SetButton (dialog, SETAXES_CHECKBOX, setAxes);

	ShowHideSetAxesDialogItems(dialog);
	
	return 0;
	
}

short SETAXESClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
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
				case PLOTDLG_DEPTHSLICEPLOTBTN: gDepthXMax = xMaxVal; gDepthYMin = -yMinVal; break;
				case PLOTDLG_AVCONCPLOTBTN: gAvConcXMax = xMaxVal; gAvConcYMax = yMinVal; break;
				case PLOTDLG_MAXCONCPLOTBTN: gMaxConcXMax = xMaxVal; gMaxConcYMax = yMinVal; break;
				case PLOTDLG_COMBOCONCPLOTBTN: gComboConcXMax = xMaxVal; gComboConcYMax = yMinVal; break;
			}

			return SETAXES_OK;
		case SETAXES_MAXX:
		case SETAXES_MINY:
			CheckNumberTextItem(dialog, itemNum, TRUE);	// allowing decimals
			break;
		case SETAXES_CHECKBOX:
			ToggleButton(dialog, itemNum);
			ShowHideSetAxesDialogItems(dialog);
			//gSetDepthAxes = !gSetDepthAxes;
			switch (gplottype) {
				case PLOTDLG_DEPTHSLICEPLOTBTN: gSetDepthAxes = !gSetDepthAxes; break;
				case PLOTDLG_AVCONCPLOTBTN: gSetAvConcAxes = !gSetAvConcAxes; break;
				case PLOTDLG_MAXCONCPLOTBTN: gSetMaxConcAxes = !gSetMaxConcAxes; break;
				case PLOTDLG_COMBOCONCPLOTBTN: gSetComboConcAxes = !gSetComboConcAxes; break;
			}
			break;
	}

	return 0;
}


OSErr SetAxesDialog(/*double xlimit, double ylimit*/)	
{	
	short item, i;
	if (gplottype == PLOTDLG_DEPTHSLICEPLOTBTN)
		item = MyModalDialog(SETAXES, mapWindow, 0, SETAXESInit, SETAXESClick);
	else
		item = MyModalDialog(5290, mapWindow, 0, SETAXESInit, SETAXESClick);
	if(item == SETAXES_CANCEL) return USERCANCEL; 
	model->NewDirtNotification();	// is this necessary ?
	if(item == SETAXES_OK) 
	{
		return 0; 
	}
	else return -1;
}

