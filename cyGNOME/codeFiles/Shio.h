
#include "cross.h"

	// values used in the timeHdl.flag
	#define outOfSequenceFlag  2

	// values used in the EbbFloodHdl
	#define MinBeforeFlood  0
	#define MaxFlood  1
	#define MinBeforeEbb  2
	#define MaxEbb  3

	// values used in the HighLowHdl
	#define LowTide	0
	#define HighTide	1

/*---------------------------------------------*/
typedef struct dblflagstruct
{
	double	val;	/* double value */
	short	flag;	/* data flag */
	short	xtra;	/* xtra space for alignment */
}EXTFLAG,*EXTFLAGPTR,**EXTFLAGHDL;


/*---------------------------------------------*/
/*---------------------------------------------*/
#ifdef MAC
#pragma options align=mac68k
#endif

		typedef struct
		{
			float		datum;
			short		FDir;
			short		EDir;
			short		L2Flag;
			short		HFlag;
			short		RotFlag;
		}CONTROLVAR,*CONTROLVARPTR, **CONTROLVARHDL;
		
		typedef struct constiuentstruct
		{
			CONTROLVAR		DatumControls;
			float			**H;
			float			**kPrime;
		}CONSTITUENT,*CONSTITUENTPTR, **CONSTITUENTHDL;

#ifdef MAC
#pragma options align=reset
#endif
/*---------------------------------------------*/
/*---------------------------------------------*/


/*---------------------------------------------*/
typedef struct heightcompstruct
{
	long			nPts;
	EXTFLAG			**timeHdl;            	// flag = 0 means plot
	double			**heightHdl;

	short			numHighLows;			// Number of high and low tide occurrences.
	short			xtra;					// padding for power pc
	double			**HighLowHeightsHdl;	// double feet
	EXTFLAG			**HighLowTimesHdl;		// double hours, flag=0 means plot 
	short			**HighLowHdl;			// 0 -> low tide.
												// 1 -> high tide.
}COMPHEIGHTS,*COMPHEIGHTSPTR, **COMPHEIGHTSHDL;


/*---------------------------------------------*/
typedef struct currentcompstruct
{
	long			nPts;
	EXTFLAG			**timeHdl;				// flag = 0 means plot
	double			**speedHdl;				// vel. vector magnitude (- for ebb)
	double			**uHdl;					// u velocities ... east positive
	double			**vHdl;					// v velocities ... north positive
	double			**uMinorHdl;			// velocity projection on minor axis
	double			**vMajorHdl;			// velocity projection on major axis 
	short           speedKey;				// speedKey = 0 no smoothing for rotary currents
											// speedKey = 1 hermite smoothing for rotary currents

	short			numEbbFloods;			// Number of ebb and flood occurrences.								
	double			**EbbFloodSpeedsHdl;	// double knots
	EXTFLAG			**EbbFloodTimesHdl;		// double hours, flag=0 means plot
	short			**EbbFloodHdl;			// 0 -> Min Before Flood.
											// 1 -> Max Flood.
											// 2 -> Min Before Ebb.
											// 3 -> Max Ebb.
}COMPCURRENTS,*COMPCURRENTSPTR, **COMPCURRENTSHDL;


/*---------------------------------------------*/
typedef struct yeardatstruct
{
	float	XODE;
	float	VPU;

} YEARDATA,*YEARDATAPTR, **YEARDATAHDL;


/*---------------------------------------------*/
typedef struct datapntstruct
{
	double	val;			// double value for valStr.
	long	dataAvailFlag;	// Availablity flag
							// dataAvailFlag = 0 -> no valid data available.
							// dataAvailFlag = 1 -> valid data available.
} DATA,*DATAPTR, **DATAHDL;


/*---------------------------------------------*/
typedef struct heightoffsetstruct
{
	DATA		HighTime;			/* High water time offset		*/
	DATA		LowTime;			/* Low water time offset		*/

	DATA		HighHeight_Mult;	/* High water height multiplier	*/
	DATA		HighHeight_Add;		/* High water height additive	*/

	DATA		LowHeight_Mult;		/* Low water height multiplier	*/
	DATA		LowHeight_Add;		/* Low water height additive	*/

}HEIGHTOFFSET,*HEIGHTOFFSETPTR, **HEIGHTOFFSETHDL;


/*---------------------------------------------*/
typedef struct currentoffsetstruct
{
	DATA		MinBefFloodTime;	/* Min Before Flood time offset	*/
	DATA		FloodTime;			/* Flood time offset			*/

	DATA		MinBefEbbTime;		/* Min Before Ebb time offset	*/
	DATA		EbbTime;			/* Ebb time offset				*/

	DATA		FloodSpdRatio;		/* Flood Speed Ratio			*/
	DATA		EbbSpdRatio;		/* Ebb Speed Ratio				*/

	DATA		MinBFloodSpd;
	DATA		MinBFloodDir;
	DATA		MaxFloodSpd;
	DATA		MaxFloodDir;

	DATA		MinBEbbSpd;
	DATA		MinBEbbDir;
	DATA		MaxEbbSpd;
	DATA		MaxEbbDir;

}CURRENTOFFSET,*CURRENTOFFSETPTR, **CURRENTOFFSETHDL;


/*---------------------------------------------*/
typedef struct heightstationrec
{
	char	name[64];
	char	subname[64];
	char	supername[64];
	char	id[8];
	char	warnings[32];
	char	region[32];
	char	refsta[32];
	char	lat[12];
	char	lng[12];

	char	hitime[8];
	char	lotime[8];
	char	hihghtMult[8];
	char	hihghtAdd[8];
	char	lohghtMult[8];
	char	lohghtAdd[8];

	char	daylght[4];
	char	booktag[8];
	char	bookyear[8];
} TIDESTATION,*TIDESTATIONPTR, **TIDESTATIONHDL;


/*---------------------------------------------*/
typedef struct currentstationrec
{
	char	name[64];
	char	subname[64];
	char	supername[64];
	char	id[8];
	char	warnings[32];
	char	region[32];
	char	refsta[32];
	char	lat[12];
	char	lng[12];
	char	depth[8];

	char	minfldtime[8];
	char	maxfldtime[8];
	char	minebbtime[8];
	char	maxebbtime[8];
	char	fldspdratio[8];
	char	ebbspdratio[8];
	char	avgminfldspd[8];
	char	avgminflddir[8];
	char	avgmaxfldspd[8];
	char	avgmaxflddir[8];
	char	avgminebbspd[8];
	char	avgminebbdir[8];
	char	avgmaxebbspd[8];
	char	avgmaxebbdir[8];

	char	daylght[4];
	char	booktag[8];
	char	bookyear[8];
} CURRENTSTATION,*CURRENTSTATIONPTR, **CURRENTSTATIONHDL;

/*---------------------------------------------*/
typedef struct textliststruct
{
	char	str[128];
}TextList,*TextListPtr, **TextListHdl;


/*---------------------------------------------*/
typedef struct stationinfostruct
{

	/* graph window information */
	//WindowPeek			wpk;					// plot window handle
	PicHandle			pichdl;					// plot picture handle
	Rect				contrect;				// modified content rectangle of window


	/* Region File Information */
	char				filename[128];			// name of region file.					
	short				refnum;					// refNum for region file.

	/* units for height and speed */
	short				hghtunits;				// units for height
	short				spdunits;				// units for speed
	short				DayLightSavingsFlag;	// 0 = automatic.
												// 1 = force on.
												// 2 = force off.

	/* calendar info */
	DateTimeRec			BeginDate;				// Start date
	DateTimeRec			EndDate;				// End date

	long				windmode;				// show text or plot

	/* using custom reference curve data */
	short				usingcustomrefdata;		// using custom reference data instead of constituent values
	short				nminmax;				// number of min max values entered
	double				**minmaxvalhdl;			// array of min max values
	double				**minmaxtimehdl;		// array of min max values
	CONTROLVAR			cntrlvars;				// control variables for reference stations
	short				xtra;					// pad for PPC (take this out in new release (redo CONTROLVAR struct)
	/* start/end usingcustomrefdata dates */
	short				s_month,s_day,s_year;	// starting date of reference file data
	short				e_month,e_day,e_year;	// ending date of reference file data

	/* text list information */
	VList				vlistinfo;				// VLIST information
	TextList			**txthdl;				// handle to the text

	/* Graphing Information */
	short				showLabels;				// Toggle labels on the plot.
	short				showGrids;				// Toggle grids on the plot.
	short				showsunrisesunset;		// Toggle sunrise and sunset shading on plot.
	short				sunrisesunsetavail;		// true if valid lat longs for station
	double				lng;					// selected station's longitude (decimal degrees)
	double				lat;					// selected station's latitude (decimal degrees)

	short				twentyfourmode;			// Toggle 24/12 hour modes.
	short				font;					// Select font.
	short				size;					// Select font size.
	short				grafType;				// Toggle the plot type.
												// 1 = SMOOTH.
												// 2 = ROUGH.
												// 3 = STICK.
												// 4 = ROSE.

	Rect				plot_grid;				// plot grid rectangle

	double				ymin;					// Minimum y axis value.
	double				ymax;					// Maximum y axis value.
	double				xmin;					// Minimum x axis value.
	double				xmax;					// Maximum x axis value.

	/* Current station specific info */
	CURRENTSTATION		coffsetdata;			// current offset data.
	CONSTITUENT			**CKInfoHdl;			// Hdl to current constituent data.
	CURRENTOFFSET		**CurrentDataHdl;		// Hdl to converted offset data (used in the computations).
	COMPCURRENTS		**CurrentGraphHdl;		// Hdl to current graph data.

	// Specific to Tide station.
	TIDESTATION			toffsetdata;			// height offset data.
	CONSTITUENT			**TKInfoHdl;			// Hdl to tide constituent data.
	HEIGHTOFFSET		**TideDataHdl;			// Hdl to converted offset data (used in the computations).
	COMPHEIGHTS			**TideGraphHdl;			// Hdl to tide graph data.

}STATION, *STATIONPTR, **STATIONHDL;





///// Stuff from old Compheights.h


short CheckHeightOffsets(HEIGHTOFFSET htOffset);



void CleanUpCompHeights (COMPHEIGHTSPTR cPtr);        

void CompErrors(short errorNum,                       // error number
				char *errStr);                        // error string returned

short DoOffSetCorrection (COMPHEIGHTSPTR answers,  // Ptr to reference station heights
				       	  HEIGHTOFFSET htOffset); // offset data

void DumpAnswers(COMPHEIGHTSPTR answers, double *julianoffset);

short FindExtraHL(double *AMPA,					// amplitude corrected for year
					double *epoch,				// epoch corrected for year
					short numOfConstituents,	// number of frequencies
					COMPHEIGHTSPTR answers,		// Height-time struc with answers
					double *zeroTime,			// first high-low
					double *extraTime,			// last high-low
					double datum);

short FindHighLow(double startTime,				// start time in hrs from begining of year
					double endTime,				// end time in hrs from begining of year
					short HighLowFlag,			// flag = 0 for low, flag = 1 for high
					double *AMPA,				// corrected amplitude array
					double *epoch,				// corrected epoch array
					short numOfConstituents,	// number of frequencies
					double *theHeight,			// the high or low tide value
					double *theTime,			// the high or low tide time
					double datum);				// height datum

void FixHEnds(COMPHEIGHTSPTR answers,    // answers
			 double beginTime,            // beginTime in hours
			 double endTime);             // endTime in hours

short GetReferenceCurve(CONSTITUENTPTR constituent,	// Amplitude-phase array structs
					    short numOfConstituents,		// number of frequencies
					    YEARDATAHDL	YHdl,           // Year correction
						double beginHour,				// beginning hr from start of year
						double endHour,					// ending hr from start of year
						double TimeStep,				// time step in minutes
					    COMPHEIGHTSPTR Answers);		// Height-time struc with answers


short GetJulianDayHr(short day,		// day of month (1 - 31)
					short month,	// month (1- 12)
					short year,		// year (1993 - 2020)
					double *hour);	// returns hours from beginning of year

short GetTideHeight(	DateTimeRec *BeginDate,DateTimeRec *EndDate,
						CONSTITUENTPTR constituent,	// Amplitude-phase array structs
					    YEARDATAHDL	YHdl,           // Year correction
						HEIGHTOFFSET htOffset,		// Height offset data
						COMPHEIGHTSPTR answers,		// Height-time struc with answers
						//double **minmaxvalhdl,			// Custom entered reference curve vals
						//double **minmaxtimehdl,			// Custom entered reference curve vals
						//long nminmax,					// number of entries
						//CONTROLVAR *cntrlvars,			// custom control vars
						Boolean DaylightSavings);		// Daylight Savings Flag.

short GetWeights(double t,                 // the time to interpolate at hrs
				EXTFLAGPTR TArray,           // the high low array of time in hrs
				double *w1,                // weight factor 1
				double *w2,                // weight factor 2
				short *index1,               // index of last high low hour
				short NoOfHighsLows);        // number of members in high low array
					
//short GetYearData(double *XODE,double *VPU, short year);

void ResetTime(COMPHEIGHTSPTR answers,double beginHour);

double RStatHeight(double	theTime,	// time in hrs from begin of year
                     double	*AMPA,		// amplitude corrected for year
					 double	*epoch,		// epoch corrected for year
					 short	ncoeff,		// number of coefficients to use
					 double	datum);		// datum in feet




///// Stuff from old Compcurrent.h

void AkutanFix (COMPCURRENTSPTR answers);

short CheckCurrentOffsets(CURRENTOFFSETPTR offset);

short CheckCurrDir(CONSTITUENTPTR constituent,  // flips answers if direction wrong
                  COMPCURRENTSPTR answers);

void CleanUpCompCurrents (COMPCURRENTSPTR CHdl);        // Hdl to answers

void CompCErrors(short errorNum,                       // error number
				 char *errStr);                        // error string returned

short CurveFitOffsetCurve(COMPCURRENTSPTR AnswerHdl,
							double startTime,
							double endTime,
							double timeStep);

void DoHermite(double v0,     // value at t0
			   double t0,     // time t0
			   double v1,     
			   double t1,     
			   double v2,     
			   double t2,     
			   double v3,    
			   double t3,    
			   double time,   // time for interpolation
			   double *vTime, // returns value at time
			   short flag);  // if flag == 0 then compute slope
			                    // if flag == 1 then v0 contains slope at t1
								// if flag == 2 then v3 contains slope at t2
								// if flag == 3 then v0 and v3 contain slopes

short DoOffSetCurrents (COMPCURRENTSPTR AnswerHdl,  // Hdl to reference station heights
					CURRENTOFFSETPTR offset,     
					double OldBeginHour,
					double OldEndHour,
					double timeStep,
					short rotFlag);


short FindExtraFE(double			*AMPA,				// amplitude corrected for year
					double			*epoch,				// epoch corrected for year
					short			numOfConstituents,	// number of frequencies
					COMPCURRENTSPTR	Answers,			// Height-time struc with answers
					double			*zeroTime,			// first max min time
					double			*zeroValue,			// Value at *zeroTime
					short			*zeroFlag,			// key to what *zeroValue is
					double			*extraTime,			// last max min time
					double			*extraValue,		// Value at *extraTime
					short			*extraFlag,			// key to what *extraValue is
					double			refCur,				// reference current in knots
					short			CFlag);				// hydraulic station flag

short FindExtraFERot(double				*AMPA,				// amplitude corrected for year
						double			*epoch,				// epoch corrected for year
						short			numOfConstituents,	// number of frequencies
						COMPCURRENTSPTR	Answers,			// Height-time struc with answers
						double			*zeroTime,			// first max-min time
						double			*zeroValue,			// first max-min value
						short			*zeroFlag,			// first max-min flag
						double			*lastTime,			// last max-min value
						double			*lastValue,			// last max-min value
						short			*lastFlag,			// last max-min flag
						CONSTITUENTPTR	constituent);	// Handle to constituent data               

short FindFloodEbb(	double	startTime,			// start time in hrs from begining of year
					double	endTime,			// end time in hrs from begining of year
					short	MaxMinFlag,			// flag = 0 for low, flag = 1 for high
					double	*AMPA,				// corrected amplitude array
					double	*epoch,				// corrected epoch array
					short	numOfConstituents,	// number of frequencies
					double	*theCurrent,		// the high or low tide value
					double	*theTime,			// the high or low tide time
					double	refCur,				// reference current in knots
					short	CFlag);				// hydraulic station flag

short FindFloodEbbRot(double		startTime,			// start time in hrs from begining of year
					double			endTime,			// end time in hrs from begining of year
					short			MaxMinFlag,			// flag = 0 for low, flag = 1 for high
					double			*AMPA,				// corrected amplitude array
					double			*epoch,				// corrected epoch array
					short			numOfConstituents,	// number of frequencies
					double			*theCurrent,		// the high or low tide value
					double			*theTime,			// the high or low tide time
					CONSTITUENTPTR	constituent,
					double			oldTwoCurrentsAgo,
					double			oldLastCurrent);

short FixAngle(short angle);

void FixCEnds(COMPCURRENTSPTR answers,  // Answers
			  double beginTime,          // Begin time in hours
			  double endTime,            // End time in hours
			  short    rotFlag);           // rotate flag

void FixCurve(COMPCURRENTSPTR answers);

short FixMajMinFlags(EXTFLAGHDL THdl,
					 double **CHdl,
					 short NumOfSteps,
					 double **MaxMinHdl,
					 EXTFLAGHDL MaxMinTHdl,
					 short MaxMinCount);

short GetCDirec(CONSTITUENTPTR constituent, // Constituent data handle
                short *FDir,                   // Flood direction 0 - 360 degrees
				short *EDir);                  // Ebb direction 0 - 360 degrees

short GetControlFlags(CONSTITUENTPTR constituent,	// Constituent data handle
                      short *L2Flag,					// L2 frequency flag
					  short *HFlag,						// Hydraulic station flag
                      short *RotFlag);					// Rotary station flag.........if any flag = 1 we got anomaly

short GetControlFlagsAlternate(	CONTROLVAR *cntrlvars,	// Control variables structure
						short *L2Flag,					// L2 frequency flag
						short *HFlag,					// Hydraulic station flag
						short *RotFlag);				// Rotary station flag.........if any flag = 1 we got anomaly
			  
double GetDatum(CONSTITUENTPTR constituent);  // returns datum field

double GetDatum2(CONSTITUENTPTR constituent,short index); // return data for rotary currents

short GetFloodEbbKey(double t,
					  EXTFLAGPTR TArrayPtr,
					  short *MaxMinFlagPtr,
					  double *CArrayPtr,
					  short numOfMaxMins,
					  short *flag);

short GetFloodEbbSpans(double t,
					  EXTFLAGPTR TArrayPtr,
					  short *MaxMinFlagPtr,
					  double *CArrayPtr,
					  short numOfMaxMins,
					  double *previousTime,
					  double *nextTime,
					  double *previousValue,
					  double *nextValue,
					  double *previousMinusOneTime,
					  double *nextPlusOneTime,
					  double *previousMinusOneValue,
					  double *nextPlusOneValue,
					  short whatFlag);

void GetMajorMinorAxis(CONSTITUENTPTR constituent,
                                    short *majorAxis,
									short *minorAxis);

short GetRefCurrent(CONSTITUENTPTR constituent,// Amplitude-phase array structs
					    YEARDATAHDL	YHdl,           // Year correction
					    short numOfConstituents,      // number of frequencies
						double beginHour,           // beginning hr from start of year
						double endHour,             // ending hr from start of year
						double TimeStep,            // time step in minutes
					    COMPCURRENTSPTR Answers);       // Height-time struc with answers

short GetSlackRatio(double t,                        // Time in hours
                    short flag,                        // flag for where in curve we are
				    EXTFLAGPTR EbbFloodTimesPtr,       // Time array of max mins
					short *EbbFloodArrayPtr,        // Ebb Flood key array
				    short NoOfMaxMins,                 // Number or max and mins
				    double FloodRatio,               // Flood ratio
				    double EbbRatio,                 // Ebb ratio
					double *newRatio);               // new interpolated ratio

short GetTideCurrent(DateTimeRec *BeginDate,DateTimeRec *EndDate,
						CONSTITUENTPTR constituent,	// Amplitude-phase array structs
						CURRENTOFFSET *offset,		// Current offset data
						COMPCURRENTSPTR answers,		// Current-time struc with answers
						YEARDATAHDL		YHdl,
						Boolean DaylightSavings,		// Daylight Savings Flag
						char *staname);					// Station name

void getVector(short degrees, double *u, double *v);

short GetVelDir(CONSTITUENTPTR constituent,  // return 1 if flood, -1 if ebb
                double u,
				double v);

void Hermite(double v1,     // value at t1
             double s1,     // slope at t1
			 double t1,     // time t1
			 double v2,     // value at t2
			 double s2,     // slope at t2
			 double t2,     // time t2
			 double time,   // time for interpolation
			 double *vTime);// returns value at time

short OffsetReferenceCurve(COMPCURRENTSPTR AnswerHdl,    // Hdl to reference station heights
						   CURRENTOFFSETPTR offset);  

short OffsetUV ( COMPCURRENTSPTR answers,		// Current-time struc with answers
			     CURRENTOFFSETPTR offset);   	// Current offset data

void ResetCTime(COMPCURRENTSPTR answers,double beginHour);

short RotToMajorMinor(CONSTITUENTPTR constituent,
					  CURRENTOFFSETPTR offset,
                      COMPCURRENTSPTR answers);

double RStatCurrent(double	theTime,	// time in hrs from begin of year
                     double	*AMPA,		// amplitude corrected for year
					 double	*epoch,		// epoch corrected for year
					 double	refCur,		// reference current in knots
					 short	ncoeff,		// number of coefficients to use
					 short	CFlag);		// hydraulic station flag

double RStatCurrentRot(double		theTime,		// time in hrs from begin of year
                     double			*AMPA,			// amplitude corrected for year
					 double			*epoch,			// epoch corrected for year
					 short			ncoeff,			// number of coefficients to use
					 CONSTITUENTPTR constituent,	// constituent handle
					 double			twoValuesAgo,	// previous, previous value
					 double			lastValue,		// previous value
					 double			*uVelocity,		// east-west component of velocity
					 double			*vVelocity,		// north-south component of velocity
					 double			*vmajor,		// major axis velocity if computed
					 double			*vminor,		// minor axis velocity if computed
					 short			*direcKey);		// direction key -1 ebb, 1 flood, 0 indeterminate

void short2Str(short sValue, char *theStr);

void hr2TStr(double exHours, char *theStr);


short slopeChange(double lastValue,double nowValue,double nextValue);

short zeroCross(double lastValue,double nowValue,double nextValue);








