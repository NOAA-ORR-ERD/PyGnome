
#ifndef __CLASSES__
#define __CLASSES__
#include <map>
#include <string>


using std::map;
using std::string;

#include "TypeDefs.h"
#include "GuiTypeDefs.h"
#include "CMyList.h"
#include "RectUtils.h"
#include "ObjectUtils.h"
#include "Uncertainty.h"
#include "GridVel.h"
#include "MemUtils.h"

Boolean EqualUniqueIDs(UNIQUEID uid,UNIQUEID uid2);
UNIQUEID ZeroID(void);

#define ITEM_OFFSET(index) ((index) * elementSize)
#define ITEM_PTR(index) (&(*L)[ITEM_OFFSET(index)])

enum { // moved TModel defines to TModel.cpp // JLM
	I_CLASSNAME = 0, I_CLASSACTIVE,
	
	I_MAPNAME = 0, I_REFLOATHALFLIFE, I_MOVERS,
	I_MAPS = 10, I_UMOVERS = 15, I_WIND = 16, I_LESETS = 20,I_MASSBALANCETOTALS = 21 ,I_MASSBALANCELINE,  I_WEATHERING = 30, I_OVERLAYS = 35,
	
	I_OVERLAYNAME = 0,
	
	// vector map overrides standard map items
	I_VMAPNAME = 0, I_VDRAWLANDBITMAP, I_VMOVERS, I_VSPILLBITMAP, I_VSPILLBITMAPACTIVE, I_VDRAWALLOWABLESPILLBITMAP, I_VREFLOATHALFLIFE, I_VDRAWESIBITMAP, I_VSHOWESILEGEND,
	
	// ptcur map overrides standard map items
	// I_PMAPNAME = 0, I_PDRAWLANDWATERBITMAP, I_PMOVERS, I_PREFLOATHALFLIFE,
	I_PMAPNAME = 0, I_PDRAWLANDWATERBITMAP, I_PMOVERS, I_PDRAWCONTOURS, I_PSETCONTOURS, I_PCONCTABLE, I_PSHOWLEGEND, I_PSHOWSURFACELES,
	I_PWATERDENSITY, I_PMIXEDLAYERDEPTH, I_PBREAKINGWAVEHT, I_PDIAGNOSTICSTRING, I_PDROPLETINFO, I_PREFLOATHALFLIFE, I_PDRAWCONTOURSFORMAP, I_PDRAWRESTRICTEDBITMAP, 
	
	// ptcur map overrides standard map items
	// I_PMAPNAME = 0, I_PDRAWLANDWATERBITMAP, I_PMOVERS, I_PREFLOATHALFLIFE,
	I_3DMAPNAME = 0, I_3DDRAWLANDWATERBITMAP, I_3DMOVERS, I_3DREFLOATHALFLIFE, I_3DSHOWGRID, I_3DSHOWDEPTHCONTOURS, I_3DDRAWCONTOURS, I_3DSETCONTOURS, I_3DSHOWLEGEND,
	I_3DWATERDENSITY, I_3DMIXEDLAYERDEPTH, I_3DBREAKINGWAVEHT, I_3DDRAWCONTOURSFORMAP,
	
	I_LEFIRSTLINE = 1, I_LEACTIVE, I_LEWINDAGE, I_LESHOWHIDE, I_LEDISPERSE, I_LEDRAWRECT, I_LENATURALDISP, I_LERELEASE_TIMEPOSITION,I_LERELEASE_MASSBALANCE, I_LELIST = 100,
	
	I_RANDOMNAME = 0, I_RANDOMACTIVE, I_RANDOMUFACTOR,
	I_RANDOMAREA, I_RANDOMMAGNITUDE, I_RANDOMANGLE,
	I_RANDOMDURATION, I_RANDOMVERTAREA,
	
	I_COMPONENTNAME = 0, I_COMPONENTACTIVE, I_COMPONENTSCALEBY, I_COMPONENTREFERENCE, I_COMPONENTLAT, I_COMPONENTLONG,
	I_COMPONENT1NAME, I_COMPONENT1GRID, I_COMPONENT1ARROWS, I_COMPONENT1DIRECTION, I_COMPONENT1SCALE,
	I_COMPONENT2NAME, I_COMPONENT2GRID, I_COMPONENT2ARROWS, I_COMPONENT2DIRECTION, I_COMPONENT2SCALE,
	
	I_COMPOUNDNAME = 0, I_COMPOUNDACTIVE, I_COMPOUNDCURRENT,
	
	I_COMPOUNDMAPNAME = 0, I_COMPOUNDMAP, I_COMPOUNDMOVERS,
	
	I_CATSNAME = 0 ,I_CATSACTIVE, I_CATSGRID, I_CATSARROWS, I_CATSLOGPROFILE,
	I_CATSREFERENCE, I_CATSSCALING, I_CATSLAT, I_CATSLONG,
	I_CATSTIMEFILE, I_CATSTIMEFILEACTIVE, 
	I_CATSUNCERTAINTY,I_CATSSTARTTIME,I_CATSDURATION, I_CATSDOWNCUR, I_CATSCROSSCUR, I_CATSDIFFUSIONCOEFFICIENT,I_CATSEDDYV0,
	I_CATSTIMEENTRIES = 100,
	
	I_ADCPNAME = 0 ,I_ADCPACTIVE, I_ADCPGRID, I_ADCPARROWS,
	I_ADCPREFERENCE, I_ADCPSCALING, I_ADCPLAT, I_ADCPLONG,
	I_ADCPTIMEFILE, I_ADCPTIMEFILEACTIVE, 
	I_ADCPUNCERTAINTY,I_ADCPSTARTTIME,I_ADCPDURATION, I_ADCPDOWNCUR, I_ADCPCROSSCUR, I_ADCPDIFFUSIONCOEFFICIENT,I_ADCPEDDYV0,
	I_ADCPTIMEENTRIES = 100,
	
	I_ADCPSTATIONNAME = 0 ,I_ADCPSTATIONACTIVE, /*I_ADCPSTATIONGRID, I_ADCPSTATIONARROWS,*/
	I_ADCPSTATIONREFERENCE, /*I_ADCPSTATIONSCALING,*/ I_ADCPSTATIONLAT, I_ADCPSTATIONLONG, I_ADCPSTATIONDATA,
	
	I_CONSTNAME = 0, I_CONSTACTIVE, I_CONSTWINDAGE, I_CONSTMAGDIRECTION,
	
	I_WINDNAME = 0, I_WINDUNCERTAIN,I_WINDACTIVE,I_SUBSURFACEWINDACTIVE,
	I_WINDCONVERSION, I_WINDWINDAGE, 
	I_WINDTIMEFILE,I_WINDSPEEDSCALE,I_WINDANGLESCALE, I_WINDSTARTTIME,I_WINDDURATION,
	//I_WINDTIMEENTRIES = 100, //LE & JLM 6/18/98
	I_WINDTIMEENTRIES, I_WINDBARB,
	
	I_WEATHERNAME = 0, I_WEATHERACTIVE };

enum {
	I_PTCURNAME = 0 ,
	I_PTCURACTIVE, 
	I_PTCURGRID, 
	I_PTCURARROWS,
	I_PTCURSCALE,
	I_PTCURUNCERTAINTY,
	I_PTCURSTARTTIME,
	I_PTCURDURATION, 
	I_PTCURALONGCUR,
	I_PTCURCROSSCUR,
	I_PTCURMINCURRENT
};

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
//////////////////////////////     Classes     ////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

extern class TMover dummyMover;
extern class TOSSMTimeValue dummyTimeValue;
extern class ADCPTimeValue dummyADCPTimeValue;

///////////////////////////////////////////////////////////////////////////

class TModelMessage
{
	public:
		// constructors
		TModelMessage(long messageCode,char* targetName,char *dataStr);
		TModelMessage(long messageCode,char* targetName,CHARH dataHdl);
		TModelMessage(long messageCode,UNIQUEID targetUniqueID,char *dataStr);
		TModelMessage(long messageCode,UNIQUEID targetUniqueID,CHARH dataHdl);
		// destructor
		~TModelMessage(); // does this need to be virtual?
		
		// Message utilities
		Boolean IsMessage(long messageCode);
		Boolean IsMessage(char* targetName);
		Boolean IsMessage(long messageCode,char* targetName);
		Boolean IsMessage(long messageCode,UNIQUEID targetUniqueID);
		long GetMessageCode(void) {return fMessageCode;}
		void GetParameterString(char * key,char * answerStr,long maxNumChars);
		OSErr GetParameterAsDouble(char * key,double * val);
		OSErr GetParameterAsLong(char * key,long * val);
		OSErr GetParameterAsBoolean(char * key,Boolean * val);
		OSErr GetParameterAsWorldPoint(char * key,WorldPoint * val,Boolean checkForLLInputWithoutDirection);
		OSErr GetParameterAsSeconds(char * key,Seconds * val);
		

	private:
		Boolean StringsMatch(char* str1,char* str2);
		// instance variables
		long fMessageCode;
		UNIQUEID fTargetUniqueID;
		char* fTargetName;
		char *fDataStr;
		CHARH fDataHdl;
};


//++ TClassID

#include "ClassID_c.h"
#include "TClassID.h"

// --

#include "Weatherer_c.h"

class TWeatherer : virtual public Weatherer_c, public TClassID
{
	public:
							TWeatherer (char *name);
		virtual			   ~TWeatherer () { Dispose (); }
		virtual OSErr		InitWeatherer () { return noErr; }
		virtual void		Dispose () { return; }

		virtual ClassID 	GetClassID () { return TYPE_WEATHERER; }
	
		// I/O methods
		virtual OSErr 		Read  (BFPB *bfpb);  			// read from current position
		virtual OSErr 		Write (BFPB *bfpb); 			// write to  current position

		// list display methods
		virtual long 		GetListLength ();
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual OSErr 		AddItem (ListItem item);
};

#include "OSSMWeatherer_c.h"

class TOSSMWeatherer : virtual public OSSMWeatherer_c, public TWeatherer
{

 	public:
							TOSSMWeatherer (char *name);
		virtual				~TOSSMWeatherer ();
		virtual OSErr		InitWeatherer ();
		virtual void		Dispose ();
	
		virtual ClassID		GetClassID () { return TYPE_OSSMWEATHERER; }
	
		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		// list display methods
		virtual Boolean		FunctionEnabled (ListItem item, short buttonID);
		virtual OSErr		DeleteItem (ListItem item);
		virtual OSErr		SettingsItem (ListItem item);
		virtual Boolean		ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual ListItem	GetNthListItem (long n, short indent, short *style, char *text);
};

///////////////////////////////////////////////////////////////////////////

// JLM 9/4/98 
// Note: the types determined by bit operations
// so are powers of 2

#include "TLEList.h"

#include "TOLEList.h"

OSErr DispersionDialog(DispersionRec *info, Seconds startRelTime, WindowPtr parentWindow);
OSErr WindageSettingsDialog(WindageRec *info, WindowPtr parentWindow);
Boolean IsAdiosBudgetTable(char* path);
OSErr ReadAdiosBudgetTable(char *path,AdiosInfoRecH *adiosData,char *oilType,float *apiNum);
OSErr GetAdiosFileName(char *path);
short BudgetTable(short massVolUnits, double amtSpilled, BudgetTableData **budgetTableHdl);		

/////////////////////////////////////////////////

typedef struct
{
	long 		numOfLEs;			// number of L.E.'s in this set
	OilType		pollutantType;		// type of pollutant
	double		totalMass; 			// total mass of all le's in list (in massUnits)
	short		massUnits; 			// units for mass quantity
	double		density;		// added for dispersed oil
	Seconds		overflightTime;		// start release time
	long		ageHours;
	short		whenAmountIsSpecified;
	Seconds		spillTime;
	
	char		spillName[kMaxNameLen];
	
} SprayDialogInfo;

OSErr CreateSprayLESet(void);
Boolean IsEditTool(long tool);
void SprayCanSettingsDialog(void);
OSErr DoCJFile(char *path);

OSErr  LoadOSSMFile(char* path, Seconds * fileTimeP);  // JLM 2/13/01  
OSErr  LoadOSSMFile(char* path, Seconds * fileTimeP, char *className); // JLm 7/13/10


class TSprayLEList : public TOLEList
{
	public:
		WORLDPOINTH			fSprayedH;
		long				fNumSprayedPts;
		/////
	private:
		Handle				fPixelBitsH;
		WorldRect			fPixelWorldRect;
		Rect				fPixelScreenRect;
		Boolean				fInPixelEditMode;
		short				fWhenAmountIsSpecified;
		Seconds				fSpillTime;
		long				fAgeAtOverflightTimeInHoursEnteredByUser;
		double				fAmountEnteredByUser;
		////////
		
	public:
							TSprayLEList ();
		virtual			   ~TSprayLEList () { Dispose (); }
		virtual OSErr		Initialize (LESetSummary *summary,Boolean deleteLERecH);
		virtual OSErr		Reset (Boolean newKeys);

		virtual double 		AgeInHrsAtOverflightTime();
		virtual double 		AmountAtOverflightTime();

		virtual ClassID 	GetClassID () { return TYPE_SPRAYLELIST; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_SPRAYLELIST) return TRUE; return TOLEList::IAm(id); }
		virtual	void		Dispose ();
		
		void 				DisposeSprayedPoints(void);
		void 				DisposePixelBits(void);
		long			 	PixelToIndex(Point pixel);
		Boolean 			CanSprayPixel(Point pixel);
		Boolean				OilAtPixel(Point pixel);
		Boolean				OilAtPixel(WorldPoint wp);
		void 				SetPixel(Point pixel);
		void 				ClearPixel(Point pixel);
		Point 				PixelToSpray(void);
		void	 			SprayCanMouseDown(void);
		void	 			EraserMouseDown(void);
		OSErr 				AddSprayPoint(Point pixel);
		OSErr 				AddSprayPoint(WorldPoint wp);
		OSErr 				VerifyPixelBitmap(void);
		void 				EraseWorldRect(Rect r);
		long 				LEIndexToSprayedIndex(long leIndex);

		SprayDialogInfo 	GetSprayDialogInfo ();
		OSErr 				SetSprayDialogInfo (SprayDialogInfo info);
		virtual Boolean 	IAmEditableInMapDrawingRect(void);
		virtual Boolean 	IAmCurrentlyEditableInMapDrawingRect(void);
		virtual Boolean 	UserIsEditingMeInMapDrawingRect(void);
		virtual void	 	StartEditingInMapDrawingRect(void);
		virtual OSErr 		StopEditingInMapDrawingRect(Boolean *deleteMe);

		// I/O methods
		virtual OSErr 		Read  (BFPB *bfpb); // read from the current position
		virtual OSErr 		Write (BFPB *bfpb); // write to    the current position

		// list display methods: base class functionality
		virtual void		Draw (Rect r, WorldRect view);
		virtual long		GetListLength ();
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
//		virtual OSErr 		AddItem (ListItem item);
		virtual OSErr 		SettingsItem (ListItem item);
		virtual OSErr 		DeleteItem (ListItem item);

		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *model);

};

/////////////////////////////////////////////////
typedef struct
{
	long		equilibriumCurves;		// 1=CH4 in seawater, 2=NG in seawater
	double	bubbleRadius;	// m
	double	molecularWt;	// CH4=.016, NG=.019 kg/mol
	double	hydrateDensity;	// CH4 = 900, NG=900-940 kg/m^3
	long		separationFlag;	// 0 no separation, 1 separation possible
	long		hydrateProcess; // 0 suppress, 1 do not suppress
	long		dropSize;	// 0 use CDOG default, 1 set by hand
//	
	Seconds	duration;
	Boolean	isContinuousRelease;
} CDOGParameters;

typedef struct
{
	double 		oilDischargeRate;			// m^3/s
	double 		gasDischargeRate;			// m^3/s

} CDOGSpillParameters;

typedef struct
{
	char	 	hydrodynamicFilesFolderPath[kMaxNameLen];	 // location of currents
	double	timeInterval;		// hrs
	double	period; 			// hrs
	short		methodOfDeterminingCurrents;	// 1 select a folder, 2 export Gnome with profile, 3 already in CDOG folder
	
} CDOGHydrodynamicInfo;

typedef struct
{
	double		timeStep; 			// hrs
	double		horizDiff; 			// m^2/s
	double		vertDiff;		// m^2/s
	
} CDOGDiffusivityInfo;

typedef struct
{
	char	 	temperatureFieldFilePath[kMaxNameLen];	 // location of temp file
	char		salinityFieldFilePath[kMaxNameLen];		// location of salinity file
	short		methodOfDeterminingTempSal;		// 1 files already in CDOG folder, 2 select temp/sal files
	
} CDOGTempSalInfo;

typedef struct
{
	short	 	temperatureUnits;	 // deg C, deg F
	short		densityUnits;		// kg/m3, API
	short		diameterUnits;		// m, cm, in
	short		dischargeType;		// oil, gas
	short		gorUnits;			// gas to oil ratio, SI units, industry units
	short		dischargeUnits;			// m3/s, BOPD (oil) or MCFS (gas)
	short 	molWtUnits;			// g/mol, kg/mol
	
} CDOGUserUnits;

typedef struct
{	// for CDOG
	float depth;
	VelocityRec value;	//u,v
	double w;
	float temp;
	float sal;
} DepthValuesSet, *DepthValuesSetP, **DepthValuesSetH;

typedef struct
{	// for CDOG
	double time;	// time after model start in hours
	double q_oil;	// oil discharge rate
	double q_gas;	// gas discharge rate
	double temp;	// release temp (deg C)
	double diam;	// orifice diameter (m)
	double rho_oil;	// density of oil (kg/m^3)
	long n_den;	// release oil (n_den>=0) or water (n_den<0)
	long output_int; // output files after output_int steps
} DischargeData, *DischargeDataP, **DischargeDataH;

enum {	kDegreesC = 1, kDegreesF };
enum {	kKgM3 = 1, kAPI };
enum {	kMeters = 1, kCentimeters, kInches };

class CDOGLEList : public TOLEList
{
	public:
		//WORLDPOINTH			fSprayedH;
		//long				fNumSprayedPts;
		/////
	private:
		CDOGParameters		fCDOGParameters;
		CDOGHydrodynamicInfo	fCDOGHydrodynamicInfo;
		CDOGDiffusivityInfo	fCDOGDiffusivityInfo;
		CDOGTempSalInfo		fCDOGTempSalInfo;
		Boolean				fOutputSubsurfaceFiles;
		Boolean				fOutputGasFiles;
		DepthValuesSetH		fDepthValuesH;
		DischargeDataH		fDischargeDataH;	// new variable discharge option 5/12/05
		float				fLengthOfGridCellInMeters;	// dx = dy
		long				fNumGridCells;	// will use 1 giant grid cell 
		Boolean			bIsHydrogenSulfide;
		CDOGUserUnits	fCDOGUserUnits;
		short				fMethodOfDeterminingHydrodynamics;	//
		short				fDepthUnits;
		////////
		
	public:
							CDOGLEList ();
		virtual			   ~CDOGLEList () { Dispose (); }
		//virtual OSErr		Initialize (LESetSummary *summary);
		virtual OSErr		Reset (Boolean newKeys);

		virtual ClassID 	GetClassID () { return TYPE_CDOGLELIST; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_CDOGLELIST) return TRUE; return TOLEList::IAm(id); }
		virtual	void		Dispose ();
		
		// I/O methods
		virtual OSErr 		Read  (BFPB *bfpb); // read from the current position
		virtual OSErr 		Write (BFPB *bfpb); // write to    the current position

		// list display methods: base class functionality
		virtual void		Draw (Rect r, WorldRect view);
		virtual long		GetListLength ();
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
//		virtual OSErr 		AddItem (ListItem item);
		virtual OSErr 		SettingsItem (ListItem item);
		virtual OSErr 		DeleteItem (ListItem item);

		//virtual OSErr 		CheckAndPassOnMessage(TModelMessage *model);
		CDOGHydrodynamicInfo 	GetCDOGHydrodynamicInfo ();
		void 			SetCDOGHydrodynamicInfo (CDOGHydrodynamicInfo info);
		CDOGDiffusivityInfo 	GetCDOGDiffusivityInfo ();
		void 			SetCDOGDiffusivityInfo (CDOGDiffusivityInfo info);
		CDOGParameters 	GetCDOGParameters ();
		void 			SetCDOGParameters (CDOGParameters params);
		CDOGTempSalInfo GetCDOGTempSalInfo();
		void 			SetCDOGTempSalInfo(CDOGTempSalInfo info);
		Seconds		GetSpillDuration();
		Boolean		GetOutputSubsurfaceFiles(){return fOutputSubsurfaceFiles;}
		Boolean		GetOutputGasFiles(){return fOutputGasFiles;}
		void			SetOutputSubsurfaceFiles(Boolean outputSubsurfaceFiles){fOutputSubsurfaceFiles=outputSubsurfaceFiles;}
		void			SetOutputGasFiles(Boolean outputGasFiles){fOutputGasFiles=outputGasFiles;}
		short 		GetMethodOfDeterminingCurrents (){return fCDOGHydrodynamicInfo.methodOfDeterminingCurrents;}
		short 		GetMethodOfDeterminingHydrodynamics (){return fMethodOfDeterminingHydrodynamics;}
		void 			SetMethodOfDeterminingHydrodynamics (short methodOfDeterminingHydrodynamics){fMethodOfDeterminingHydrodynamics = methodOfDeterminingHydrodynamics;}
		short 		GetMethodOfDeterminingTempSal (){return fCDOGTempSalInfo.methodOfDeterminingTempSal;}
		Boolean		TemperatureProfilePathSet();
		Boolean		SalinityProfilePathSet();
		long 			GetNumProfiles();
		DepthValuesSetH	GetDepthValuesHandle () { return fDepthValuesH; }
		void			SetDepthValuesHandle (DepthValuesSetH dvals) ;
		long 			GetNumDischarges();
		DischargeDataH	GetDischargeDataHandle () { return fDischargeDataH; }
		void 			SetDischargeDataHandle(DischargeDataH dischargevals);
		Boolean 		GetHydrogenSulfideInfo(){return bIsHydrogenSulfide;}
		void 			SetHydrogenSulfideInfo(Boolean hydrogenSulfide) {bIsHydrogenSulfide = hydrogenSulfide;}
		CDOGUserUnits			GetCDOGUserUnits();
		void			SetCDOGUserUnits(CDOGUserUnits userUnits);
		short			GetDepthUnits() {return fDepthUnits;}
		void			SetDepthUnits(short depthUnits) {fDepthUnits = depthUnits;}
			
		OSErr			ReadCDOGOutputFiles();
		OSErr 		ExportCDOGGeninf();	// consolidated for CDOGG2.1
		OSErr 		ExportCDOGZeroWind();	// consolidated for CDOGG2.1
		OSErr 		ExportProfilesAsNetCDF(TCurrentMover *hydrodynamicData);
		OSErr 		ExportProfilesToCDOGInputFolder(TCurrentMover *hydrodynamicData);
		OSErr 		ExportProfilesToCDOGInputFolder();
		OSErr 		ExportGNOMECurrentsForCDOG();
		OSErr 		CopyHydrodynamicsFilesToCDOGInputFolder();	
		OSErr 		CopyTemperatureFilesToCDOGInputFolder();	
		OSErr 		CopySalinityFilesToCDOGInputFolder();	

};
CDOGLEList* GetCDOGSpill();
OSErr CreateCDOGLESet();
OSErr CDOGSpillSettingsDialog (CDOGLEList 	*thisLEList);
Boolean CDOGAvailable();

//++ TMap

#include "Map_c.h"
#include "TMap.h"

//--

OSErr TMapSettingsDialog(TMap* theMap);
///////////////////////////////////////////////////////////////////////////

class TOSSMMap : public TMap
{
	private:
		CHARH			fGridHdl;
		FLOATH			fRefloatTimesHdl;
	protected:
		long 			fNumRows;
		long 			fNumCols;
 	public:
 						TOSSMMap (char* name, WorldRect bounds);
 		virtual		   ~TOSSMMap () { Dispose (); }

		virtual OSErr	InitMap();
		virtual OSErr	InitMap (char *path);
		virtual OSErr 	ReplaceMap();
		virtual ClassID GetClassID () { return TYPE_OSSMMAP; }
		virtual Boolean	IAm(ClassID id) { if(id==TYPE_OSSMMAP) return TRUE; return TMap::IAm(id); }
		virtual void	Dispose ();

		virtual void	Draw (Rect r, WorldRect view);
		virtual Boolean DrawingDependsOnTime(void){return TMap::DrawingDependsOnTime();}
		virtual Boolean OnLand (WorldPoint p);
		virtual float	RefloatHalfLifeInHrs(WorldPoint p);
		virtual LongPoint	GetGridRowCol(WorldPoint p);
		virtual long 	GetLandType (WorldPoint p);
		virtual	short 	ConvertToLandType(char *code);
		virtual void 	SetDefaultRefloatTimes();
		virtual	OSErr 	OverrideRefloatTimes(char *s);
//		virtual Boolean	CanReFloat (Seconds time, LERec *theLE) { return true; }

		// I/O methods
		virtual OSErr 	Read  (BFPB *bfpb); // read from the current position
		virtual OSErr 	Write (BFPB *bfpb); // write to the current position

		virtual long 	NumPtsInGridHdl(void);

	private:
		virtual void 	DrawCell (Rect *cellRect, long mode, Boolean bColor, long cellCol, long cellRow);
};

Boolean IsGridMap (char *path);
///////////////////////////////////////////////////////////////////////////

enum {	NODIAGNOSTICSTRING=0, TRIANGLEAREA, NUMLESINTRIANGLE, CONCENTRATIONLEVEL, DEPTHATCENTERS, SUBSURFACEPARTICLES, SHORELINEPTNUMS };


typedef struct ConcTriNumPair
{
	double conc;
	long triNum;
} ConcTriNumPair, *ConcTriNumPairP, **ConcTriNumPairH;

int ConcentrationCompare(void const *x1, void const *x2);

//++ PtCurMap

#include "PtCurMap_c.h"
#include "PtCurMap.h"

//-- 
#include "Map3D_c.h"
#include "Map3D.h"

//-- 


OSErr AddPtCurMap(char *path, WorldRect bounds);
PtCurMap* CreateAndInitPtCurMap(char *path, WorldRect bounds);
Map3D* CreateAndInitMap3D(char *path, WorldRect bounds);
void DrawFilledWaterTriangles(void * object,WorldRect wRect,Rect r);
void DrawWideLandSegments(void * object,WorldRect wRect,Rect r);
//PtCurMap* GetPtCurMap(void);	// defined in Netcdfmover.h maybe should be part of model class?
OSErr OutputOptionsDialog(short outputType);
OSErr SmoothingParametersDialog(float *percentile,Boolean *useSmoothing,float *minDist,Boolean *useLineCrossAlgorithm);
Boolean AppendToWORLDPOINTH(WORLDPOINTH *wh, WorldPoint *wp);
Boolean AppendToLONGH(LONGH *h, long val);
Boolean PlotDialog(outputData** concHdl, float* depthSlice, float contourDepth1, float contourDepth2, float bottomRange, Boolean triSelected, Boolean showDepthPlot);
OSErr AnalysisDialog(Boolean *selectedPtToTrack);
OSErr AnalysisDialog2();
short OiledShorelineTable(OiledShorelineData **oiledShorelineHdl);	
Boolean OSPlotDialog(OiledShorelineData** oiledShorelineHdl);	

//++ TCompoundMap

#include "CompoundMap_c.h"
#include "TCompoundMap.h"

//--

#ifndef IBM
//#pragma options align=reset
#endif

OSErr ScanMatrixPt (char *startChar, LongPoint *MatrixLPtPtr);
OSErr GetHeaderLineInfo (char *StrLine, char *Str1, char *Str2, long *NumOfPtsPtr);
Boolean GetStrExtension (char *labelStr, char *extStr);

extern Boolean gDrawBitmapInBlackAndWhite;

//++

#include "TVectMap.h"

//--

/////////////////////////////////////////////////
// Files for CDOG
OSErr CreateMapBox();
OSErr MapBoxDialog(WorldRect *mapBoxBounds);
OSErr CDOGHydrodynamicsDialog(CDOGHydrodynamicInfo *hydrodynamicInfo,DialogPtr parentWindow);
OSErr CDOGDiffusivityDialog(CDOGDiffusivityInfo *diffusivityInfo,DialogPtr parentWindow);
OSErr TempSalDialog(CDOGTempSalInfo *tempSalInfo);
OSErr CDOGTimeStepDialog(Seconds *timeStep,DialogPtr parentWindow);
//OSErr CDOGSettingsDialog(CDOGParameters *spillParameters,DialogPtr dialog);
OSErr CDOGOutputOptionsDialog(Boolean *outputSubsurfaceFiles,Boolean *outputGasFiles,DialogPtr parentWindow);
OSErr CircInfoDialog(short *methodOfDeterminingHydrodynamics);
/////////////////////////////////////////////////
extern RGBColor esicolors[];
Boolean IsVectorMap (char *path, Boolean *isESI);
#ifdef MAC
Boolean IsBlackPixel(WorldPoint p,WorldRect mapBounds,BitMap bm);
#else
Boolean IsBlackPixel(WorldPoint p,WorldRect mapBounds,HDIB bm);
#endif
OSErr 	LandBitMapWidthHeight(WorldRect wRect, long *width, long* height);
OSErr	AddMapsDialog ();
OSErr	AddMapsDialog2 ();

///////////////////////////////////////////////////////////////////////////

//++ TMover

#include "Mover_c.h"
#include "TMover.h"

//--

//++ TCurrentMover

#include "CurrentMover_c.h"
#include "TCurrentMover.h"

//--


//++ TRandom


#include "Random_c.h"
#include "TRandom.h"


//--


//++ TRandom3D

#include "Random3D_c.h"
#include "TRandom3D.h"

//--


///////////////////////////////////////////////////////////////////////////
class TGridVel;
///////////////////////////////////////////////////////////////////////////

#define		kUCode			0					// supplied to UorV routine
#define		kVCode			1

//++ TCATSMover

#include "CATSMover_c.h"
#include "TCATSMover.h"

//--

//++ TCATSMover3D

#include "CATSMover3D_c.h"
#include "TCATSMover3D.h"

//--

Boolean IsCATS3DFile (char *path);
void SetShowGrid(Boolean set);
void SetShowSelectedTriangles(Boolean set);
void SetDosageCheckMarks(Boolean set);
void SetDepthContoursCheckMark(Boolean set);
void SetDepthLegendCheckMark(Boolean set);

//++ ADCPMover 

#include "ADCPMover_c.h"
#include "ADCPMover.h"

//--

Boolean IsADCPFile (char *path);

//++ TWindMover

#include "WindMover_c.h"
#include "TWindMover.h"

//--




//++ TComponentMover

#include "ComponentMover_c.h"
#include "TComponentMover.h"

//--

//++ TCompoundMover

#include "CompoundMover_c.h"
#include "TCompoundMover.h"

//--

#include "Wizard.h" //JLM
#include "Overlays.h"

//++ TModel

#include "Model_c.h"
#include "TModel.h"

//--



TModelDialogVariables DefaultTModelDialogVariables(Seconds start);
Boolean EqualTModelDialogVariables(TModelDialogVariables* var1,TModelDialogVariables *var2);
void TellPlayersAboutNewDirt(void);

// powers of 2 so we can do bit checks

//++ TTimeValue

#include "TimeValue_c.h"
#include "TTimeValue.h"

//--

//++ TOSSMTimeValue

#include "OSSMTimeValue_c.h"
#include "TOSSMTimeValue.h"


//--

#include "TimeValuesIOPD.h"

//++ ADCPTimeValue

#include "ADCPTimeValue_c.h"
#include "ADCPTimeValue.h"

//--

//++ TriCurMover

#include "TriCurMover_c.h"
#include "TriCurMover.h"

//--

//++ PtCurMover

#include "PtCurMover_c.h"
#include "PtCurMover.h"

//++ GridCurrentMover

#include "GridCurrentMover_c.h"
#include "GridCurrentMover.h"

//++ CurrentCycleMover

#include "CurrentCycleMover_c.h"
#include "CurrentCycleMover.h"

//++ GridWindMover

#include "GridWindMover_c.h"
#include "GridWindMover.h"
//--

//ADCPTimeValue* LoadADCPTimeValue(TMover *theOwner,short unitsIfKnownInAdvance);
//ADCPTimeValue* CreateADCPTimeValue(TMover *theOwner,char* path, char* shortFileName, short unitsIfKnownInAdvance); //JLM

#define DIRTY_EVERYTHING -1
#define DIRTY_RUNBAR 1
#define DIRTY_LIST 2
#define DIRTY_MAPDRAWINGRECT  4
#define DIRTY_TOOLBAR  8
#define DIRTY_ENTIREWINDOW  16

void StartReadWriteSequence(char *procName);
OSErr WriteValue(BFPB *bfpb, char *data, short length, Boolean swap);
OSErr ReadValue(BFPB *bfpb, char *data, short length, Boolean swap);
BFPB* SetReadValueBFPBFTrick(CHARH h); 
void ClearReadValueBFPBFTrick(void);

OSErr 	ReadMacValue(BFPB *bfpb, Seconds* val);
OSErr 	ReadMacValue(BFPB *bfpb, char* val);
OSErr 	ReadMacValue(BFPB *bfpb, long* val);
OSErr 	ReadMacValue(BFPB *bfpb, short* val);
OSErr 	ReadMacValue(BFPB *bfpb, float* val);
OSErr 	ReadMacValue(BFPB *bfpb, double* val);
OSErr 	ReadMacValue(BFPB *bfpb, Boolean* val);
OSErr 	ReadMacValue(BFPB *bfpb, UNIQUEID* val);
OSErr	ReadMacValue(BFPB *bfpb, LongPoint *lp);
OSErr 	ReadMacValue(BFPB *bfpb, WorldRect *wRect);
OSErr 	ReadMacValue(BFPB *bfpb, Rect *theRect);
OSErr 	ReadMacValue(BFPB *bfpb, WorldPoint *wp);
OSErr 	ReadMacValue(BFPB *bfpb, char* str, long len);

OSErr 	WriteMacValue(BFPB *bfpb, Seconds val);
OSErr 	WriteMacValue(BFPB *bfpb, char val);
OSErr 	WriteMacValue(BFPB *bfpb, long val);
OSErr 	WriteMacValue(BFPB *bfpb, short val);
OSErr 	WriteMacValue(BFPB *bfpb, float val);
OSErr 	WriteMacValue(BFPB *bfpb, double val);
OSErr 	WriteMacValue(BFPB *bfpb, Boolean val);
OSErr 	WriteMacValue(BFPB *bfpb, UNIQUEID val);
OSErr 	WriteMacValue(BFPB *bfpb, LongPoint lp);
OSErr 	WriteMacValue(BFPB *bfpb, WorldRect wRect);
OSErr 	WriteMacValue(BFPB *bfpb, Rect theRect);
OSErr 	WriteMacValue(BFPB *bfpb, WorldPoint wp);
OSErr 	WriteMacValue(BFPB *bfpb, char* str, long len);


void OutOffMemoryAlert(void);

#include "NetCDFMover.h"
#include "NetCDFMoverCurv.h"
#include "NetCDFWindMover.h"
#include "NetCDFMoverTri.h"
#include "NetCDFWindMoverCurv.h"




#ifdef MAC
#ifdef MPW
void MyDebugSysBeep(short x);
#define SysBeep(x)  MyDebugSysBeep(x) // magic macro
#endif
#endif

#endif
