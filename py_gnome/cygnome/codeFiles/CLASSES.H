
#ifndef __CLASSES__
#define __CLASSES__
#include <map>
#include <string>


using std::map;
using std::string;

#include "TypeDefs.h"
#include "CMyList.h"
#include "RectUtils.h"
#include "ObjectUtils.h"
#include "Uncertainty.h"
#include "GridVel.h"
#include "MemUtils.h"

Boolean EqualUniqueIDs(UNIQUEID uid,UNIQUEID uid2);
UNIQUEID ZeroID(void);


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

#include "ClassID/ClassID_b.h"
#include "ClassID/ClassID_c.h"
#include "ClassID/ClassID_g.h"
#include "ClassID/TClassID.h"

// --

///////////////////////////////////////////////////////////////////////////

class TWeatherer : public TClassID
{
	public:
							TWeatherer (char *name);
		virtual			   ~TWeatherer () { Dispose (); }
		virtual OSErr		InitWeatherer () { return noErr; }
		virtual ClassID 	GetClassID () { return TYPE_WEATHERER; }
		virtual void		Dispose () { return; }

//		void				SetWeathererName (char *newName)    { SetClassName (newName); }
//		void				GetWeathererName (char *returnName) { GetClassName (returnName); }

		virtual void		WeatherLE (LERec *theLE) {}

		// I/O methods
		virtual OSErr 		Read  (BFPB *bfpb);  			// read from current position
		virtual OSErr 		Write (BFPB *bfpb); 			// write to  current position

		// list display methods
		virtual long 		GetListLength ();
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual OSErr 		AddItem (ListItem item);
};

///////////////////////////////////////////////////////////////////////////

class TOSSMWeatherer : public TWeatherer
{
 	public:
		CMyList				*componentsList;

 	public:
							TOSSMWeatherer (char *name);
		virtual				~TOSSMWeatherer ();
		virtual OSErr		InitWeatherer ();
		virtual ClassID		GetClassID () { return TYPE_OSSMWEATHERER; }
		virtual void		Dispose ();

		virtual void		WeatherLE (LERec *theLE);

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


class TLEList : public TClassID
{
	public:
		long			numOfLEs;
		//short			massUnits;
		LERecH			LEHandle; 	// handle to LE array
		LETYPE 			fLeType;
		UNIQUEID		fOwnersUniqueID; // set if owned by another LE set, i.e this is a mirrored set
		
	public:
						TLEList ();
		virtual		   ~TLEList () { Dispose (); }
					   
		//virtual OSErr	Initialize (long numElements, LERec *representativeLE, short massUnits,
		//							WorldPoint startRelPos, WorldPoint endRelPos,
		//							Seconds startRelTime, Seconds endRelTime,
		//							Boolean bWantEndRelTime, Boolean bWantEndRelPosition) { return noErr; }
		//virtual OSErr	Initialize (long numElements, LERecH array, short massUnits, Seconds fileTime) { return noErr; }
		virtual OSErr	Reset (Boolean newKeys) { return noErr; }

		virtual LETYPE 	GetLEType () { return fLeType; }
		long			GetNumOfLEs(){return numOfLEs;}
		void			SetLEStatus (long leNum, OilStatus status);
		OilStatus		GetLEStatus (long leNum);
		void			ReFloatLE (long leNum);
		void			GetLE (long leNum, LERecP theLE);
		void			SetLE (long leNum, LERecP theLE);
		WorldPoint		GetLEPosition (long leNum);
		void			SetLEPosition (long leNum, WorldPoint p);
		Seconds			GetLEReleaseTime (long leNum);
		void			ReleaseLE (long leNum);
		void			AgeLE (long leNum);
		virtual void	BeachLE (long leNum, WorldPoint beachPosition);
		WorldRect		GetLEBounds ();
		void 			GetLEStatistics(long* numReleased,long* numEvaporated,long* numBeached, long* numOffMap, long* numFloating);
		void 			RecalculateLEStatistics(long* numEvaporated,long* numFloating, long* numRemoved, long* numOffMaps); // 	to account for dispersion
		virtual void 	GetLEAmountStatistics(short desiredMassVolUnits, double *amtTotal,double *amtReleased,double *amtEvaporated,
																double *amtDispersed,double *amtBeached,double *amtOffmap, double *amtFloating, double *amtRemoved){};
		
		void			SetFileDirty (Boolean bFileDirty) { bDirty = bFileDirty; }
		Boolean			IsFileDirty () { return !bDirty; }
		
		virtual long	GetLECount () { return numOfLEs; }  
		
		virtual long	GetMassUnits () { return -1; } // they must override
		virtual short	GetMassUnitType () { return -1; } // they must override
		virtual double	GetTotalMass () { return -1; } ; // they must override

		virtual ClassID GetClassID () { return TYPE_LELIST; }
		virtual Boolean	IAm(ClassID id) { if(id==TYPE_LELIST) return TRUE; return TClassID::IAm(id); }

		virtual	void	Dispose ();

		virtual OSErr UpItem (ListItem item);	// code goes here, decide if this is okay - brought from GNOME_beta
		virtual OSErr DownItem (ListItem item);

		// I/O methods
		virtual OSErr 	Read  (BFPB *bfpb); // read from the current position
		virtual OSErr 	Write (BFPB *bfpb) ; // write to    the current position
		virtual	OSErr	WriteLE (BFPB *bfpb, LERec *theLE);
		virtual	OSErr	ReadLE  (BFPB *bfpb, LERec *theLE, long version);

		// list display methods: base class functionality
		virtual void	Draw (Rect r, WorldRect view) { return; }
		

};
double GetLEMass(LERec theLE);


class TOLEList : public TLEList
{
	public:
		LESetSummary		fSetSummary; // summary of LE's in this list
		Boolean 			bMassBalanceOpen;
		Boolean 			bReleasePositionOpen;
		CMyList	   		   *initialLEs;
		Boolean				binitialLEsVisible;
		Boolean				bShowDispersantArea;
		DispersionRec		fDispersantData;
		AdiosInfoRecH		fAdiosDataH;	// time after spill, amount dispersed, amount evaporated
		BudgetTableDataH	fBudgetTableH;
		char				fOilTypeName[kMaxNameLen];
		WindageRec			fWindageData;
	public:
							TOLEList ();
		virtual			   ~TOLEList () { Dispose (); }
		//virtual OSErr		Initialize (long numElements, LERec *representativeLE, short massUnits,
		//								WorldPoint startRelPos, WorldPoint endRelPos,
		//								Seconds startRelTime, Seconds endRelTime,
		//								Boolean bWantEndRelTime, Boolean bWantEndRelPosition);
		virtual OSErr 		Initialize(LESetSummary * summary,Boolean deleteLERecH);
		virtual OSErr		Initialize (long numElements, LERecH array, short massUnits, Seconds fileTime);
		virtual OSErr		Reset (Boolean newKeys);

		//void				ReleaseLE (long i);
		//void				AgeLE (long i);
		//virtual void		BeachLE (long i, WorldPoint beachPosition);
	
		long				GetLECount () { return fSetSummary.numOfLEs; }

		virtual ClassID 	GetClassID () { return TYPE_OSSMLELIST; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_OSSMLELIST) return TRUE; return TLEList::IAm(id); }
		virtual	void		Dispose ();
		
		virtual long		GetMassUnits () { return fSetSummary.massUnits; } 
		virtual short		GetMassUnitType();
		virtual double		GetTotalMass () { return fSetSummary.totalMass; } 
		virtual OilType		GetOilType(){return fSetSummary.pollutantType;}
		virtual Seconds		GetSpillStartTime() { return fSetSummary.startRelTime; }
		virtual Seconds		GetSpillEndTime() { return fSetSummary.endRelTime; }
		virtual void 		GetLEAmountStatistics(short desiredMassVolUnits, double *amtTotal,double *amtReleased,double *amtEvaporated,
															double *amtDispersed,double *amtBeached,double *amtOffmap, double *amtFloating, double *amtRemoved);
		virtual void 		GetMassBalanceLines(Boolean includePercentage, char* line1,char*line2,char* line3,char* line4,char* line5,char* line6,char* line7);
		virtual OSErr 		CalculateAverageIntrusionDepth(double *avDepth, double *stdDev);

		virtual DispersionRec 		GetDispersionInfo ();
		virtual void 				SetDispersionInfo (DispersionRec info);

		virtual AdiosInfoRecH 		GetAdiosInfo () {return fAdiosDataH;}
		virtual void 				SetAdiosInfo (AdiosInfoRecH adiosInfoH){if (fAdiosDataH && fAdiosDataH != adiosInfoH) {DisposeHandle((Handle)fAdiosDataH);} fAdiosDataH = adiosInfoH;}
		long 				GetNumAdiosBudgetTableItems(void);
		void 				AddToBudgetTableHdl(BudgetTableData *budgetTable);
		OSErr 				ExportBudgetTableHdl(char* path, BFPB *bfpb);
		BudgetTableDataH	GetBudgetTable(){return fBudgetTableH;}

		virtual WindageRec 	GetWindageInfo ();
		virtual void 		SetWindageInfo (WindageRec info);

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

		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
};

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

#include "Map/Map_b.h"
#include "Map/Map_c.h"
#include "Map/Map_g.h"
#include "Map/TMap.h"

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

typedef struct {
	long segNo;
	long startPt;
	long endPt;
	long numBeachedLEs;
	float segmentLengthInKm;
	float	gallonsOnSegment;
	//Seconds time;
} OiledShorelineData,*OiledShorelineDataP,**OiledShorelineDataHdl;

typedef struct ConcTriNumPair
{
	double conc;
	long triNum;
} ConcTriNumPair, *ConcTriNumPairP, **ConcTriNumPairH;

int ConcentrationCompare(void const *x1, void const *x2);

//++ PtCurMap

#include "PtCurMap/PtCurMap_b.h"
#include "PtCurMap/PtCurMap.h"

//-- 


float DistFromWPointToSegment(long pLong, long pLat, long long1, long lat1, 
														long long2, long lat2, long dLong, long dLat);
OSErr AddPtCurMap(char *path, WorldRect bounds);
PtCurMap* CreateAndInitPtCurMap(char *path, WorldRect bounds);
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

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
//class TCompoundMap : public TMap	// maybe PtCurMap ?
class TCompoundMap : public PtCurMap	// maybe PtCurMap ?
{
	/*private:
		LONGH			fBoundarySegmentsH;
		LONGH			fBoundaryTypeH;		// 1 land, 2 water
		LONGH			fBoundaryPointsH;	// for curvilinear grids
		LONGH			fSegSelectedH;
		LONGH			fSelectedBeachHdl;	//not sure if both are needed
		LONGH			fSelectedBeachFlagHdl;	//not sure if both are needed*/
//#ifdef IBM
		//HDIB			fWaterBitmap;
		//HDIB			fLandBitmap;
//#else
		//BitMap			fWaterBitmap; 
		//BitMap			fLandBitmap; 
//#endif
		//Boolean			bDrawLandBitMap;
		//Boolean			bDrawWaterBitMap;
		//Boolean			bShowSurfaceLEs;

 /*	public:
		float			fContourDepth1;
		float			fContourDepth2;
		float			fContourDepth1AtStartOfRun;
		float			fContourDepth2AtStartOfRun;
		DOUBLEH			fContourLevelsH;
		Rect			fLegendRect;
		Boolean			bShowLegend;
		short			fDiagnosticStrType;		// 0 no diagnostic string, 1 tri area, 2 num LEs, 3 conc levels, 4 depths, 5 subsurface particles
		
		long			fWaterDensity;
		double			fMixedLayerDepth;
		double			fBreakingWaveHeight;
		
		double			*fTriAreaArray;
		//long			*fDepthSliceArray;	// number of LEs in each layer (1m) of depth slice
		float			*fDepthSliceArray;	//changed to ppm in each layer (1m) of depth slice 7/21/03
		
		Boolean			bUseSmoothing;
		Boolean			bUseLineCrossAlgorithm;
		float			fMinDistOffshore;	// set how far LEs reflect off shoreline, so they don't get stuck
		//Boolean			bShowElapsedTime;	// should be a model field
		short			fWaveHtInput;	// 0 from wind speed, 1 breaking wave height measure, 2 significant wave height measure
		DropletInfoRecH	fDropletSizesH;
		*/
	public:
		CMyList				*mapList; 			// list of the map's component maps

		Boolean 			bMapsOpen;

							TCompoundMap (char* name, WorldRect bounds);
		virtual			   ~TCompoundMap () { Dispose (); }

		virtual void 		Dispose ();
		virtual OSErr		InitMap ();
		//virtual OSErr		InitContourLevels();
		//virtual OSErr		InitDropletSizes();

		virtual ClassID 	GetClassID () { return TYPE_COMPOUNDMAP; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_COMPOUNDMAP) return TRUE; return PtCurMap::IAm(id); }
		
		
		CMyList				*GetMapList () { return mapList; }
		//virtual OSErr		AddMover (TMover *theMover, short where);
		virtual OSErr		AddMap (TMap *theMap, short where);
		virtual OSErr		DropMap (TMap *theMap);
		virtual OSErr		DropMover (TMover *theMover);
		TCurrentMover*		GetCompoundMover();

		//TMap*					AddNewMap(OSErr *err);	// may not need this
		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb);  	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); 	// write to  current position

		//virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);

		// list display methods
		virtual void		Draw(Rect r, WorldRect view);


		virtual long		GetListLength();
		virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
		virtual OSErr 		UpItem 			(ListItem item);
		virtual OSErr 		DownItem 		(ListItem item);
		//virtual OSErr 		AddItem(ListItem item);
		virtual OSErr 		SettingsItem(ListItem item);
		//virtual OSErr 		DeleteItem(ListItem item);

		virtual	Boolean	HaveMapBoundsLayer (void) { return false; }

		virtual OSErr 	AddMover(TMover *theMover, short where);
		OSErr		MakeBitmaps()	{return 0;}
		virtual	OSErr 	ReplaceMap(){return 0;}
		//virtual	void 	SetBoundarySegs(LONGH boundarySegs) { fBoundarySegmentsH = boundarySegs; }
		//virtual	void 	SetWaterBoundaries(LONGH waterBoundaries) { fBoundaryTypeH = waterBoundaries; }
		//virtual	void 	SetBoundaryPoints(LONGH boundaryPts) { fBoundaryPointsH = boundaryPts; }
		//virtual	LONGH 	GetBoundarySegs() { return fBoundarySegmentsH; }
		//virtual	LONGH 	GetWaterBoundaries() { return fBoundaryTypeH; }
		//virtual	LONGH 	GetBoundaryPoints() { return fBoundaryPointsH; }
		//void 	AddSegmentToSegHdl(long startno);
		//void 	SetSelectedBeach(LONGH *segh, LONGH selh);
		//void 	ClearSelectedBeach();
		//void 	SetBeachSegmentFlag(LONGH *beachBoundaryH, long *numBeachBoundaries);
		//Boolean 	MoreSegments(LONGH segh,long *startIndex, long *endIndex,long *curIndex);
		//void			SetMinDistOffshore(WorldRect wBounds);
		//long 	WhichSelectedSegAmIIn(long index);

		//virtual void 	DrawBoundaries(Rect r);
		//virtual void 	DrawBoundaries2(Rect r);
		void			DrawContoursFromMapIndex(Rect r, WorldRect view, long mapIndex);
		virtual	void 	DrawContours(Rect r, WorldRect view);	// total over all LELists
		virtual void  	DrawContourScale(Rect r, WorldRect view);
		virtual	void 	DrawDepthContourScale(Rect r, WorldRect view);
		//void 			DrawSegmentLabels(Rect r);
		//void 			DrawPointLabels(Rect r);
#ifdef IBM
		void 			EraseRegion(Rect r);
#endif
		virtual Boolean DrawingDependsOnTime(void){return TMap::DrawingDependsOnTime();}
		virtual	Boolean InMap (WorldPoint p);	//  check all maps
		virtual Boolean OnLand (WorldPoint p);	// check all maps
		virtual WorldPoint3D	MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed);	// have to do something here
		virtual long	PointOnWhichSeg(long p);
		virtual	long 	PointOnWhichSeg(long longVal, long latVal, long *startver, long *endver, float *dist);
		Boolean 	ContiguousPoints(long p1, long p2);
		void 	FindNearestBoundary(Point where, long *verNum, long *segNo);
		void  FindNearestBoundary(WorldPoint wp, long *verNum, long *segNo);
		TMover* 		GetMover(ClassID desiredClassID);
		TCurrentMover* 	Get3DCurrentMover();
		//Boolean			ThereIsADispersedSpill();
		Boolean 		ThereAreTrianglesSelected();
		TTriGridVel* 	GetGrid(Boolean wantRefinedGrid);
		virtual TTriGridVel3D* 	GetGrid3D(Boolean wantRefinedGrid);
		double			DepthAtCentroidFromMapIndex(long triNum,long mapIndex);
		LongPointHdl 	GetPointsHdl(Boolean useRefinedGrid);	
		//virtual Boolean		CanReFloat (Seconds time, LERec *theLE);
//		virtual Boolean	CanReFloat (Seconds time, LERec *theLE) { return true; }
		virtual long 	GetLandType (WorldPoint p);
		//Boolean			LEInMap(WorldPoint p);	// not used
		Boolean			InVerticalMap(WorldPoint3D wp);	// check by priority
		//float 			GetMaxDepth(void);
		WorldPoint3D	ReflectPoint(WorldPoint3D fromWPt,WorldPoint3D toWPt,WorldPoint3D wp);	// check by priority
		virtual double			DepthAtPoint(WorldPoint wp);// check by priority
		double 			DepthAtCentroid(long triNum);// check by priority
		WorldPoint3D 	TurnLEAlongShoreLine(WorldPoint3D waterPoint, WorldPoint3D beachedPoint, WorldPoint3D toPoint);
		//WorldPoint3D 	TurnLEAlongShoreLine2(WorldPoint3D waterPoint, WorldPoint3D beachedPoint, WorldPoint3D toPoint);	// Gnome_beta diagnostic stuff
		//WorldPoint3D	SubsurfaceMovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, OilStatus *status);
		//Boolean 		PointOnBoundaryLine(WorldPoint p);
		//double 			GetBreakingWaveHeight(void);
		//void 			TrackOutputData(TOLEList *thisLEList);
		virtual	void 		TrackOutputData(void);
		virtual	void		TrackOutputDataInAllLayers(void);
		void			TrackOutputDataFromMapIndex(long mapIndex);
		//OSErr 			DoAnalysisMenuItems(long menuCodedItemID);

		Rect 			DoArrowTool(long triNum);
		//OSErr 			GetDepthAtMaxTri(TOLEList *thisLEList,long *maxTriIndex,double *depthAtPnt);	
		OSErr 			GetDepthAtMaxTri(long *maxTriIndex, double *depthAtPnt);	
		//OSErr 			CreateDepthSlice(TLEList *thisLEList, long triNum)	;
		OSErr 			CreateDepthSlice(long triNum, float **depthSlice);
		//OSErr 			CreateDepthSlice(long triNum, float *depthSlice);
		void 			DoLassoTool(Point p);
		void 			MarkRect(Point p);

		virtual	long 	GetNumBoundarySegs(void);
		virtual  long 	GetNumPointsInBoundarySeg(long segno);
		virtual	long 	GetNumBoundaryPts(void);
		long 	CountLEsOnSelectedBeach();
		void 	FindStartEndSeg(long ptnum,long *startPt, long *endPt);
		long 	NextPointOnSeg(long segno, long point);
		long 	PrevPointOnSeg(long segno, long point);
		Boolean 	MoreBoundarySegments(long *a,long *b);
		void 	InitBoundaryIter(Boolean clockwise,long segno, long startno, long endno);
		double 	PathLength(Boolean selectionDirection,long segNo, long startno, long endno);
		Boolean 	IsBoundaryPoint(long pt);
		//virtual	long 	GetNumContourLevels(void);
		virtual	float			GetMaxDepth2(void);
		//DropletInfoRecH	GetDropletSizesH(void) {return fDropletSizesH;}		
		long	WhichMapIsPtIn(WorldPoint wp);
		long	WhichMapIsPtInWater(WorldPoint wp);
		TCurrentMover*	Get3DCurrentMoverFromIndex(long moverIndex);
		TTriGridVel3D*	GetGrid3DFromMapIndex(long mapIndex);

};
TCompoundMap* CreateAndInitCompoundMap(char *path, WorldRect bounds);
///////////////////////////////////////////////////////////////////////////

typedef struct RegionInfoRec
			{
					long				rgnEndPt;			/* historical reasons only */
					Boolean				bFilled;
					Boolean				bSelected;
					LongRect			rBounds;
					long				rLinePixels;
					short				rLineColor;
					short				rFillColor;
					Pattern				rFillPat;
					short				rFillCode; 			/* pattern or color fill */
					short				LinkDataType;
					Handle				LinkDataHdl;
//					char				LabelFont			[kFontNameLen];
					short				LabelFontStyle;
					short				LabelFontSize;
					LongPoint			LabelPoint;
//					char				RegionLabel 		[kLabelStrLen];
					Rect				LabelScrRect;
					short				LabelJust;
					LongPoint			**PointsHdl;		/* no longer written to file */
					long				TotalPoints;		/* no longer written to file */
			} RegionInfoRec, *RegionInfoRecPtr, **RegionInfoHandle;

#ifndef IBM
//#pragma options align=reset
#endif

OSErr ScanMatrixPt (char *startChar, LongPoint *MatrixLPtPtr);
OSErr GetHeaderLineInfo (char *StrLine, char *Str1, char *Str2, long *NumOfPtsPtr);
Boolean GetStrExtension (char *labelStr, char *extStr);

extern Boolean gDrawBitmapInBlackAndWhite;

class TVectorMap : public TMap
{
	public:
		CMapLayer			*thisMapLayer;
		CMapLayer			*allowableSpillLayer;
		CMapLayer			*mapBoundsLayer;
		CMapLayer			*esiMapLayer;
		CMap				*map;
#ifdef IBM
		HDIB				fLandWaterBitmap;
		HDIB				fAllowableSpillBitmap;
		HDIB				fMapBoundsBitmap;
		HDIB				fESIBitmap;
#else
		BitMap				fLandWaterBitmap; 
		BitMap				fAllowableSpillBitmap; 
		BitMap				fMapBoundsBitmap; 
		BitMap				fESIBitmap; 
		//CGrafPtr 			fESIBitmap;		// ESI segments bitmap (color)
#endif
		Boolean				bDrawLandBitMap;
		Boolean				bDrawAllowableSpillBitMap;
		Boolean				bSpillableAreaOpen;
		Boolean				bSpillableAreaActive;

		WorldRect			fExtendedMapBounds;
		Boolean				fUseExtendedBounds;
		
		Rect				fLegendRect;
		Boolean				bShowLegend;
		Boolean				bDrawESIBitMap;
			
		//long				fBitMapResMultiple;
		
 	public:
 							TVectorMap (char* name, WorldRect bounds);
 		virtual			   ~TVectorMap () { Dispose (); }
		
		virtual OSErr		InitMap ();
		virtual OSErr		InitMap (char *path);
		void		 		SetExtendedBounds (WorldRect newBounds) {fExtendedMapBounds = newBounds;}
		virtual WorldRect	GetMapBounds () { if (fUseExtendedBounds) return fExtendedMapBounds; else return fMapBounds; }
		virtual ClassID 	GetClassID () { return TYPE_VECTORMAP; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_VECTORMAP) return TRUE; return TMap::IAm(id); }
		virtual void		Dispose ();
		
		virtual void		Draw (Rect r, WorldRect view);
		virtual Boolean 	DrawingDependsOnTime(void){return TMap::DrawingDependsOnTime();}
		virtual	void 		DrawESILegend(Rect r, WorldRect view);
		virtual	Boolean 	InMap (WorldPoint p);
		virtual Boolean 	OnLand (WorldPoint p);
		virtual float		RefloatHalfLifeInHrs(WorldPoint p);
		virtual WorldPoint3D	MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed);
		virtual long 		GetLandType (WorldPoint p);
		virtual Boolean 	HaveAllowableSpillLayer(void);
		virtual Boolean 	HaveMapBoundsLayer(void);
		virtual	Boolean 	HaveLandWaterLayer(void);
		virtual	Boolean 	HaveESIMapLayer(void);
		//virtual	Boolean 	IsIceMap(void);
		virtual Boolean 	IsAllowableSpillPoint(WorldPoint p);
		
		TMover* 			GetMover(ClassID desiredClassID);
		TTriGridVel*		GetGrid();
		virtual	double 				DepthAtPoint(WorldPoint wp);

		virtual long		GetListLength();
		virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);

		virtual OSErr 		CheckAndPassOnMessage(TModelMessage * model);

		// I/O methods
		virtual OSErr 		Read  (BFPB *bfpb); // read from the current position
		virtual OSErr 		Write (BFPB *bfpb); // write to the current position
		OSErr 				ExportAsBNAFileForGnomeAnalyst(char* path);
		virtual void		GetSuggestedFileName(char* suggestedFileName,char* extension);
		virtual OSErr 		ReplaceMap();
		OSErr 				ImportESIData (char *path);
		OSErr 				ChangeMapBox(WorldPoint p, WorldPoint p2);

	private:
		OSErr 				ImportMap (char *path); 
		OSErr 				SelectMapBox (WorldRect bounds);
};

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
TVectorMap* GetNthVectorMap(long desiredNum0Relative);
Boolean IsVectorMap (char *path, Boolean *isESI);
#ifdef MAC
Boolean IsBlackPixel(WorldPoint p,WorldRect mapBounds,BitMap bm);
#else
Boolean IsBlackPixel(WorldPoint p,WorldRect mapBounds,HDIB bm);
#endif
void 	LandBitMapWidthHeight(WorldRect wRect, long *width, long* height);
OSErr	AddMapsDialog ();

///////////////////////////////////////////////////////////////////////////

//++ TMover

#include "Mover/Mover_b.h"
#include "Mover/Mover_c.h"
#include "Mover/Mover_g.h"
#include "Mover/TMover.h"

//--


/////////////////////////////////////////////////


class TCurrentMover : public TMover
{
	protected:
		LONGH			fLESetSizesH;			// cumulative total num le's in each set
		LEUncertainRecH	fUncertaintyListH;		// list of uncertain factors list elements of type LEUncertainRec

	public:
		double			fDownCurUncertainty;	
		double			fUpCurUncertainty;	
		double			fRightCurUncertainty;	
		double			fLeftCurUncertainty;	
		
		Boolean			bIAmPartOfACompoundMover;
		Boolean			bIAmA3DMover;
		
	public:
							TCurrentMover (TMap *owner, char *name);
		virtual			   ~TCurrentMover () { Dispose (); }

		virtual ClassID 	GetClassID () { return TYPE_CURRENTMOVER; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_CURRENTMOVER) return TRUE; return TMover::IAm(id); }
		virtual Boolean		IAmA3DMover(){return false;}
		virtual void		Dispose ();

		virtual OSErr 		UpItem (ListItem item);
		virtual OSErr 		DownItem (ListItem item);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);

		virtual void 		UpdateUncertaintyValues(Seconds elapsedTime);
		virtual OSErr		UpdateUncertainty(void);
		virtual OSErr		AllocateUncertainty ();
		virtual void		DisposeUncertainty ();

		virtual OSErr 		PrepareForModelStep();

		virtual WorldRect GetGridBounds(){return theWorld;}	
		virtual float		GetArrowDepth(){return 0.;}

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb);  // read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		virtual OSErr 		ReadTopology(char* path, TMap **newMap)	{return 2;}
		virtual OSErr 		ExportTopology(char* path) {return 2;}	

		virtual OSErr 		CheckAndPassOnMessage(TModelMessage * model);

		virtual OSErr 		SettingsDialog() {return 0;}
};


//++ TRandom


#include "Random/Random_b.h"
#include "Random/Random_c.h"
#include "Random/Random_g.h"
#include "Random/TRandom.h"


//--


///////////////////////////////////////////////////////////////////////////
class TRandom3D : public TRandom
{
	public:
		double fVerticalDiffusionCoefficient; //cm**2/s
		double fHorizontalDiffusionCoefficient; //cm**2/s
		double fVerticalBottomDiffusionCoefficient; //cm**2/s
		Boolean bUseDepthDependentDiffusion;
		//double fDiffusionCoefficient; //cm**2/s
		//TR_OPTIMZE fOptimize; // this does not need to be saved to the save file
		//double fUncertaintyFactor;		// multiplicative factor applied when uncertainty is on
	
	public:
							TRandom3D (TMap *owner, char *name);
		virtual ClassID 	GetClassID () { return TYPE_RANDOMMOVER3D; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_RANDOMMOVER3D) return TRUE; return TRandom::IAm(id); }
		
		virtual OSErr 		PrepareForModelStep();
		virtual void 		ModelStepIsDone();
		virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
		
		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb);  // read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		virtual void		Draw(Rect r, WorldRect view) { }
		virtual long		GetListLength();
		virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
		//virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
//		virtual OSErr 		AddItem(ListItem item);
		virtual OSErr 		SettingsItem(ListItem item);
		virtual OSErr 		DeleteItem(ListItem item);
};

///////////////////////////////////////////////////////////////////////////
class TGridVel;
///////////////////////////////////////////////////////////////////////////

#define		kUCode			0					// supplied to UorV routine
#define		kVCode			1

typedef struct {
	Boolean isOptimizedForStep;
	Boolean isFirstStep;
	double 	value;
} TCM_OPTIMZE;

class TCATSMover : public TCurrentMover
{
	public:
		WorldPoint 		refP; 					// location of tide station or map-join pin
		TGridVel		*fGrid;					//VelocityH		grid; 
		long 			refZ; 					// meters, positive up
		short 			scaleType; 				// none, constant, or file
		double 			scaleValue; 			// constant value to match at refP
		char 			scaleOtherFile[32]; 	// file to match at refP
		double 			refScale; 				// multiply current-grid value at refP by refScale to match value
		Boolean 		bRefPointOpen;
		Boolean			bUncertaintyPointOpen;
		Boolean 		bTimeFileOpen;
		Boolean			bTimeFileActive;		// active / inactive flag
		Boolean 		bShowGrid;
		Boolean 		bShowArrows;
		double 			arrowScale;
		TOSSMTimeValue *timeDep;
		double			fEddyDiffusion;			// cm**2/s minimum eddy velocity for uncertainty
		double			fEddyV0;			//  in m/s, used for cutoff of minimum eddy for uncertainty
	public:
		TCM_OPTIMZE fOptimize; // this does not need to be saved to the save file

	public:
							TCATSMover (TMap *owner, char *name);
		virtual			   ~TCATSMover () { Dispose (); }
		virtual OSErr		InitMover (TGridVel *grid, WorldPoint p);
		virtual ClassID 	GetClassID () { return TYPE_CATSMOVER; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_CATSMOVER) return TRUE; return TCurrentMover::IAm(id); }
		virtual void		Dispose ();
		virtual Boolean 	OkToAddToUniversalMap();
		virtual	OSErr 		ReplaceMover();
		
		virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
		
		virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
	
		virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
		virtual Boolean 	CurrentUncertaintySame (CurrentUncertainyInfo info);

		void				SetRefPosition (WorldPoint p, long z) { refP = p; refZ = z; }
		void				GetRefPosition (WorldPoint *p, long *z) { (*p) = refP; (*z) = refZ; }

		virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	

		void				SetTimeDep (TOSSMTimeValue *newTimeDep) { timeDep = newTimeDep; }
		TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
		void				DeleteTimeDep ();
		VelocityRec			GetPatValue (WorldPoint p);
		VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
		VelocityRec			GetSmoothVelocity (WorldPoint p);
		//OSErr				ComputeVelocityScale ();
		virtual OSErr		ComputeVelocityScale ();
		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
		virtual OSErr 		PrepareForModelStep();
		virtual void 		ModelStepIsDone();

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		virtual void		Draw (Rect r, WorldRect view);
		virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *velStr);
		virtual long		GetListLength ();
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
//		virtual OSErr 		AddItem (ListItem item);
		virtual OSErr 		SettingsItem (ListItem item);
		virtual OSErr 		DeleteItem (ListItem item);

		virtual	OSErr 	ReadTopology(char* path, TMap **newMap);
		virtual	OSErr 	ExportTopology(char* path);
};

///////////////////////////////////////////////////////////////////////////
class TCATSMover3D : public TCATSMover
{
	public:
		TTriGridVel3D	*fRefinedGrid;			// store a second grid for contouring
		Boolean			bShowDepthContours;	// this should be a map item I think
		Boolean			bShowDepthContourLabels;

		/*WorldPoint 		refP; 					// location of tide station or map-join pin
		TGridVel		*fGrid;					//VelocityH		grid; 
		long 			refZ; 					// meters, positive up
		short 			scaleType; 				// none, constant, or file
		double 			scaleValue; 			// constant value to match at refP
		char 			scaleOtherFile[32]; 	// file to match at refP
		double 			refScale; 				// multiply current-grid value at refP by refScale to match value
		Boolean 		bRefPointOpen;
		Boolean			bUncertaintyPointOpen;
		Boolean 		bTimeFileOpen;
		Boolean			bTimeFileActive;		// active / inactive flag
		Boolean 		bShowGrid;
		Boolean 		bShowArrows;
		double 			arrowScale;
		TOSSMTimeValue *timeDep;
		double			fEddyDiffusion;			// cm**2/s minimum eddy velocity for uncertainty
		double			fEddyV0;			//  in m/s, used for cutoff of minimum eddy for uncertainty
	public:
		TCM_OPTIMZE fOptimize; // this does not need to be saved to the save file
*/
	public:
							TCATSMover3D (TMap *owner, char *name);
		virtual			   ~TCATSMover3D () { Dispose (); }
		//virtual OSErr		InitMover (TGridVel *grid, WorldPoint p);
		virtual ClassID 	GetClassID () { return TYPE_CATSMOVER3D; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_CATSMOVER3D) return TRUE; return TCATSMover::IAm(id); }
		virtual Boolean		IAmA3DMover(){return true;}
		virtual void		Dispose ();
		virtual Boolean 	OkToAddToUniversalMap();
		
		//virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
		
		//virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
	
		//virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);

		/*void				SetRefPosition (WorldPoint p, long z) { refP = p; refZ = z; }
		void				GetRefPosition (WorldPoint *p, long *z) { (*p) = refP; (*z) = refZ; }

		void				SetTimeDep (TOSSMTimeValue *newTimeDep) { timeDep = newTimeDep; }
		TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
		void				DeleteTimeDep ();
		VelocityRec			GetPatValue (WorldPoint p);
		VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
		VelocityRec			GetSmoothVelocity (WorldPoint p);
		OSErr				ComputeVelocityScale ();
		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
*/		virtual OSErr 		PrepareForModelStep();
		//virtual void 		ModelStepIsDone();
		virtual	Boolean 		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);

		//LongPointHdl 		GetPointsHdl(Boolean useRefinedGrid);
		virtual 			LongPointHdl 		GetPointsHdl();
		OSErr				TextRead(char *path, TMap **newMap); 
		OSErr 				ImportGrid(char *path); 
		OSErr 				CreateRefinedGrid (Boolean askForFile, char* givenPath, char* givenFileName);

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		virtual void		Draw (Rect r, WorldRect view);
		//virtual long		GetListLength ();
		//virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		//virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
//		virtual OSErr 		AddItem (ListItem item);
		//virtual OSErr 		SettingsItem (ListItem item);
		//virtual OSErr 		DeleteItem (ListItem item);
		virtual	OSErr 	ReadTopology(char* path, TMap **newMap);
		virtual	OSErr 	ExportTopology(char* path);

};
Boolean IsCATS3DFile (char *path);
void SetShowGrid(Boolean set);
void SetShowSelectedTriangles(Boolean set);
void SetDosageCheckMarks(Boolean set);
void SetDepthContoursCheckMark(Boolean set);
void SetDepthLegendCheckMark(Boolean set);

///////////////////////////////////////////////////////////////////////////
class ADCPMover : public TCurrentMover
{	// code goes here, keep a list of time files or have umbrella move that keeps list of ADCPMovers?
	public:
		WorldPoint 		refP; 					// location of tide station or map-join pin
		TGridVel		*fGrid;					//VelocityH		grid; 
		long 			refZ; 					// meters, positive up
		short 			scaleType; 				// none, constant, or file
		double 			scaleValue; 			// constant value to match at refP
		char 			scaleOtherFile[32]; 	// file to match at refP
		double 			refScale; 				// multiply current-grid value at refP by refScale to match value
		Boolean 		bRefPointOpen;
		Boolean			bUncertaintyPointOpen;
		Boolean 		bTimeFileOpen;
		Boolean			bTimeFileActive;		// active / inactive flag
		Boolean 		bShowGrid;
		Boolean 		bShowArrows;
		double 			arrowScale;
		//ADCPTimeValue *timeDep;
		double			fEddyDiffusion;			// cm**2/s minimum eddy velocity for uncertainty
		double			fEddyV0;			//  in m/s, used for cutoff of minimum eddy for uncertainty
		Rect			fLegendRect;
		long			fBinToUse;
	public:
		TCM_OPTIMZE fOptimize; // this does not need to be saved to the save file
		CMyList		*timeDepList;
	public:
							ADCPMover (TMap *owner, char *name);
		virtual			   ~ADCPMover () { Dispose (); }
		virtual OSErr		InitMover (TGridVel *grid, WorldPoint p);
		virtual OSErr		InitMover ();
		virtual ClassID 	GetClassID () { return TYPE_ADCPMOVER; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_ADCPMOVER) return TRUE; return TCurrentMover::IAm(id); }
		virtual void		Dispose ();
		virtual Boolean 	OkToAddToUniversalMap();
		virtual	OSErr 		ReplaceMover();
		
		OSErr				AddTimeDep(ADCPTimeValue *theTimeDep, short where);
		OSErr				DropTimeDep(ADCPTimeValue *theTimeDep);
		ADCPTimeValue *		AddADCP(OSErr *err);

		virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
		
		virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
	
		virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
		virtual Boolean 	CurrentUncertaintySame (CurrentUncertainyInfo info);

		void				SetRefPosition (WorldPoint p, long z) { refP = p; refZ = z; }
		void				GetRefPosition (WorldPoint *p, long *z) { (*p) = refP; (*z) = refZ; }

		virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	

		//void				SetTimeDep (ADCPTimeValue *newTimeDep) { timeDep = newTimeDep; }
		//ADCPTimeValue		*GetTimeDep () { return (timeDep); }
		//void				DeleteTimeDep ();
		VelocityRec			GetPatValue (WorldPoint p);
		VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);
		VelocityRec			GetVelocityAtPoint(WorldPoint3D p);
		//OSErr				ComputeVelocityScale ();
		virtual OSErr		ComputeVelocityScale ();
		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
		virtual OSErr 		PrepareForModelStep();
		virtual void 		ModelStepIsDone();

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		OSErr				TextRead(char *path);
		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		virtual void		Draw (Rect r, WorldRect view);
		virtual	Boolean		DrawingDependsOnTime(void);
		virtual void		DrawContourScale(Rect r, WorldRect view);
		virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *velStr);
		virtual long		GetListLength ();
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
//		virtual OSErr 		AddItem (ListItem item);
		virtual OSErr 		SettingsItem (ListItem item);
		virtual OSErr 		DeleteItem (ListItem item);

};
Boolean IsADCPFile (char *path);

//++ TWindMover

#include "WindMover/WindMover_b.h"
#include "WindMover/WindMover_c.h"
#include "WindMover/WindMover_g.h"
#include "WindMover/TWindMover.h"

//--

typedef struct {
	Boolean isOptimizedForStep;
	Boolean isFirstStep;
	double 	pat1ValScale;
	double 	pat2ValScale;
} TC_OPTIMZE;

enum {	kLinkToNone = 1, kLinkToTimeFile, kLinkToWindMover };

enum { NONE = 0, WINDSPEED, WINDSTRESS };	// for scaling

class TComponentMover : public TCurrentMover
{
	public:
		TCATSMover			*pattern1;
		TCATSMover			*pattern2;
		Boolean				bPat1Open;
		Boolean				bPat2Open;
		TOSSMTimeValue		*timeFile;

		WorldPoint			refP;
		Boolean 			bRefPointOpen;

		double				pat1Angle;
		double				pat2Angle;
		
		double				pat1Speed;
		double				pat2Speed;
		
		long				pat1SpeedUnits;
		long				pat2SpeedUnits;

		double				pat1ScaleToValue;
		double				pat2ScaleToValue;
		
		long				scaleBy;
		
		Boolean			bUseAveragedWinds;
		Boolean			bExtrapolateWinds;
		Boolean			bUseMainDialogScaleFactor;
		double			fScaleFactorAveragedWinds;
		double			fPowerFactorAveragedWinds;
		long				fPastHoursToAverage;
		TimeValuePairH	fAveragedWindsHdl;
		
//							optimize fields don't need to be saved
		TC_OPTIMZE			fOptimize;

		long				timeMoverCode;
		char 				windMoverName [64]; 	// file to match at refP

							TComponentMover (TMap *owner, char *name);
		virtual			   ~TComponentMover () { Dispose (); }

		virtual void 		Dispose ();
		virtual OSErr		InitMover ();
		virtual ClassID 	GetClassID () { return TYPE_COMPONENTMOVER; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_COMPONENTMOVER) return TRUE; return TCurrentMover::IAm(id); }
		
		virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep);

		virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
		
		virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
		virtual Boolean 	CurrentUncertaintySame (CurrentUncertainyInfo info);
	
		virtual OSErr 		PrepareForModelStep();
		virtual void 		ModelStepIsDone();
		OSErr				SetOptimizeVariables (char *errmsg);
		OSErr				CalculateAveragedWindsHdl(char *errmsg);
		OSErr				GetAveragedWindValue(Seconds time, VelocityRec *avValue);

		virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
		virtual	Boolean 		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
		
		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb);  	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); 	// write to  current position

		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);

		// list display methods
		virtual void		Draw(Rect r, WorldRect view);


		virtual long		GetListLength();
		virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
//		virtual OSErr 		AddItem(ListItem item);
		virtual OSErr 		SettingsItem(ListItem item);
		virtual OSErr 		DeleteItem(ListItem item);
};

///////////////////////////////////////////////////////////////////////////
class TCompoundMover : public TCurrentMover	// maybe just TMover
{
	public:
		//TCATSMover			*pattern1;
		//TCATSMover			*pattern2;
		CMyList				*moverList; 			// list of the mover's component currents
		//Boolean				bPat1Open;
		//Boolean				bPat2Open;
		//TOSSMTimeValue		*timeFile;

		Boolean 			bMoversOpen;
		/*WorldPoint			refP;
		Boolean 			bRefPointOpen;

		double				pat1Angle;
		double				pat2Angle;
		
		double				pat1Speed;
		double				pat2Speed;
		
		long				pat1SpeedUnits;
		long				pat2SpeedUnits;

		double				pat1ScaleToValue;
		double				pat2ScaleToValue;
		
		long				scaleBy;
		
		*/
//							optimize fields don't need to be saved
		//TC_OPTIMZE			fOptimize;

		//long				timeMoverCode;
		//char 				windMoverName [64]; 	// file to match at refP

							TCompoundMover (TMap *owner, char *name);
		virtual			   ~TCompoundMover () { Dispose (); }

		virtual void 		Dispose ();
		virtual OSErr		InitMover ();
		virtual ClassID 	GetClassID () { return TYPE_COMPOUNDMOVER; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_COMPOUNDMOVER) return TRUE; return TCurrentMover::IAm(id); }
		virtual Boolean		IAmA3DMover();
		
		virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep);

		virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
		
		virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
		virtual Boolean 	CurrentUncertaintySame (CurrentUncertainyInfo info);
	
		virtual OSErr 		PrepareForModelStep();
		virtual void 		ModelStepIsDone();

		virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
		virtual	Boolean 		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
		virtual float		GetArrowDepth();
		
		// mover methods
		CMyList				*GetMoverList () { return moverList; }
		//virtual OSErr		AddMover (TMover *theMover, short where);
		virtual OSErr		AddMover (TMover *theMover, short where);
		virtual OSErr		DropMover (TMover *theMover);

		TCurrentMover*		AddCurrent(OSErr *err,TCompoundMap **newMap);	
		virtual	LongPointHdl	GetPointsHdl();
		TTriGridVel*		GetGrid(Boolean wantRefinedGrid);
		TTriGridVel3D*		GetGrid3D(Boolean wantRefinedGrid);
		TCurrentMover*		Get3DCurrentMover();
		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb);  	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); 	// write to  current position

		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);

		// list display methods
		virtual Boolean		DrawingDependsOnTime(void);
		virtual void		Draw(Rect r, WorldRect view);


		virtual long		GetListLength();
		virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
		virtual OSErr 		UpItem 			(ListItem item);
		virtual OSErr 		DownItem 		(ListItem item);
		virtual OSErr 		AddItem(ListItem item);
		virtual OSErr 		SettingsItem(ListItem item);
		virtual OSErr 		DeleteItem(ListItem item);

		TTriGridVel3D*	GetGrid3DFromMoverIndex(long moverIndex);
		TCurrentMover*	Get3DCurrentMoverFromIndex(long moverIndex);

		void			SetShowDepthContours();
		Boolean			ShowDepthContourChecked();
};

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

typedef struct
{
	Seconds		frameTime;
	char		frameLEFName [255];
} LEFrameRec, *LEFrameRecPtr;

typedef struct
{
	Seconds	startTime;			// time to start model run
	Seconds	duration;			// duration of model run
	Seconds computeTimeStep;	// time step for computation
	Boolean	bUncertain; 		// uncertainty on / off
	Boolean preventLandJumping;	// use the code that checks for land jumping
} TModelDialogVariables;

#include "Wizard.h" //JLM
#include "Overlays.h"

class TModel : public TClassID
{
	private:
		TModelDialogVariables fDialogVariables;
		char		fSaveFileName [255];			
		char		fOutputFileName [255];
		Seconds 	fOutputTimeStep;
		Boolean 	fWantOutput;

	public:
		long		nextKeyValue; 	// next available key value for new LE's
		Seconds		modelTime; 		// current model time
		Seconds		lastComputeTime;// the latest time for which the model has computed output that is still valid
		Seconds		LEDumpInterval;	// interval at which LE's are saved to disk for time-bar support
		Boolean		bSaveRunBarLEs;// flag to indicate if model LEs are to be dumped for run-bar support 
		Boolean 	bLEsDirty;		// LE's have changed if true. Used to determine when to initialize uncertainty arrays
									// DO NOT SAVE THIS VARIABLE IN SAVE FILE. MUST BE TRUE WHEN TMODEL IS INSTANTIATED
		
		long		modelMode;		// advanced, novice, etc. modes
		Boolean 	fSettingsOpen;	// model settings being displayed?
		Boolean 	fSpillsOpen;	// spills being displayed?
		Boolean 	bMassBalanceTotalsOpen;	// mass balance totals being displayed?
		Boolean 	mapsOpen;		// individual maps being displayed?
		Boolean 	fOverlaysOpen;		// individual overlays being displayed?
		Boolean 	uMoverOpen;		// individual movers being displayed?
		Boolean 	weatheringOpen;	// weathering being displayed?
		Boolean 	fDrawMovement;
		float		fMaxDuration;	// maybe this should be a global ?

		Boolean 	fRunning; 		// we don't need to save this in the Save file
		Boolean		bMakeMovie;		// flag indicating if movies is being generated during the run
		short		movieFrameIndex;// index of next movie frame
		char		fMoviePicsPath[256];// path specification for folder containing the movie pict frames
		char		fMoviePath[256];	// movie file path


		Boolean 	bSaveSnapshots;	// flag indicating if pictures are being saved during the run
		Seconds 	fTimeOffsetForSnapshots;	// so user can decide when to start the output
		char 		fSnapShotFileName[256];

		Boolean bHindcast, writeNC, ncSnapshot;
		int ncID, ncID_C, stepsCount, currentStep;
		map<string, int> ncVarIDs, ncDimIDs, ncVarIDs_C, ncDimIDs_C;
		char ncPath[256], ncPathConfidence[256];
		
		// bitmap version of map at current view and window size without LEs or movement grid
		#ifdef MAC
			CGrafPtr mapImage;		
		#else 		
			HDIB	mapImage;		
		#endif

		CMyList	*fSquirreledLastComputeTimeLEList;
		CMyList	*LESetsList;
		CMyList	*mapList;
		TMap	*uMap;
		CMyList	*weatherList;
		LocaleWizard *fWizard;
		CMyList	*fOverlayList; // JLM 6/4/10
		
		CMyList *LEFramesList;
		CMyList *frameMapList;		// copy of map list used to detect changes in actual map-list

	public:
						TModel(Seconds start);
		virtual		   ~TModel () { Dispose (); }
		virtual OSErr	InitModel ();
		virtual void	Dispose ();
		virtual ClassID GetClassID () { return TYPE_MODEL; }
		virtual Boolean	IsDirty ();

		void			DisposeLEFrames ();
		void			DisposeModelLEs ();
		void			DisposeAllMoversOfType(ClassID desiredClassID);


		TMover*			GetMover(char* moverName);
		TMover*			GetMover(ClassID desiredClassID);
		TMap* 			GetMap(char* mapName);
		TMap*			GetMap(ClassID desiredClassID);
		TLEList* GetMirroredLEList(TLEList* owner);
		TLEList* GetLEListOwner(TLEList* mirroredLEList);
		long 	GetNumMovers(char* moverName);
		long  GetNumWindMovers();	
		TWindMover* GetWindMover(Boolean createIfDoesNotExist);
		TWindMover* GetNthWindMover(long desiredNum0Relative);
		TRandom* GetDiffusionMover(Boolean createIfDoesNotExist);
		TRandom3D* Get3DDiffusionMover();
		TCurrentMover* GetPossible3DCurrentMover();
		
		virtual OSErr GetTotalAmountStatistics(short desiredMassVolUnits,double *amtTotal,double *amtReleased,double *amtEvaporated,
																	double *amtDispersed,double *amtBeached,double * amtOffmap, double *amtFloating, double *amtRemoved);

		// drawing movement utils
		void MovementString(WorldPoint3D wp,char* str);
		void DrawLEMovement(void);
		Boolean	ThereIsA3DMover(float *arrowDepth);

		void DrawLegends(Rect r, WorldRect wRect); 

		// messages to the model
		void SuppressDirt (long suppressDirtFlags);
		void NewDirtNotification(void);
		void NewDirtNotification (long flags);
		
		OSErr BroadcastMessage(long messageCode, char* targetName, UNIQUEID targetUniqueID, char* dataStr, CHARH dataHdl);
		OSErr BroadcastMessage(long messageCode, char* targetName, char* dataStr, CHARH dataHdl);
		OSErr BroadcastMessage(long messageCode, UNIQUEID uid, char* dataStr, CHARH dataHdl);
		void BroadcastToSelectedItem(long messageCode, char* dataStr, CHARH dataHdl);
		
		virtual OSErr 	CheckAndPassOnMessage(TModelMessage *message);
		long 	GetNumForecastSpills();
		OSErr HandleRunSpillMessage(TModelMessage *message);
		// JLM 6/25/10 break up RUNSPILL in to making createspill and a RUN mesage
		OSErr HandleRunMessage(TModelMessage *message);
		OSErr HandleCreateSpillMessage(TModelMessage *message);
		
		// map methods
		CMyList	*GetMapList () { return mapList; }
		OSErr	AddMap (TMap *theMap, short where);
		OSErr	DropMap (TMap *theMap);
		TMap	*GetBestMap (WorldPoint p);
		Boolean IsWaterPoint(WorldPoint p);
		Boolean IsAllowableSpillPoint(WorldPoint p);
		Boolean HaveAllowableSpillLayer(WorldPoint p);
		Boolean CurrentBeachesLE(WorldPoint3D startPoint, WorldPoint3D *movedPoint, TMap *bestMap);
		WorldPoint3D TurnLEAlongShoreLine(WorldPoint3D waterPoint, WorldPoint3D beachedPoint, TMap *bestMap);
		long	GetMapCount () { return mapList -> GetItemCount (); }
		WorldRect GetMapBounds (void);

		OSErr   AddOverlay(TOverlay *theOverlay, short where);
		OSErr   DropOverlay(TOverlay *theOverlay);
		void	DrawOverlays(Rect r, WorldRect wRect);
		

		// model mode get and set methods
		void	SetModelMode (long theMode);
		long	GetModelMode () { return modelMode; }
		char*  	GetModelModeStr(char *modelModeStr);
		void 	SetLastComputeTime(Seconds time);

		// LE methods
		CMyList	*GetLESetsList () { return (LESetsList); }
		OSErr	AddLEList (TLEList *theLEList, short where);
		OSErr	DropLEList (TLEList *theLEList, Boolean bDispose);

		CMyList	*GetWeatherList () { return weatherList; }
		OSErr	AddWeatherer (TWeatherer *theWeatherer, short where);
		OSErr	DropWeatherer (TWeatherer *theWeatherer);
		
		void	ResetMainKey ();
		long	GetNextLEKey ();
		void	ResetLEKeys ();
		
		Boolean UserIsEditingSplots(void);
		long 	NumEditableSplotObjects(void);
		void 	SelectAnEditableSplotObject(void);
		Boolean 	EditableSplotObjectIsSelected();
		TClassID *ItemBeingEditedInMappingWindow(void);
		OSErr 	DropObject(TClassID  *object, Boolean bDispose);
		OSErr 	EditObjectInMapDrawingRect(TClassID *newObjectToEdit);
		void	CheckEditModeChange(void);
		
		Boolean ThereIsAnEarlierSpill(Seconds timeOfInterest, TLEList *someLEListToIgnore);
		Boolean ThereIsALaterSpill(Seconds timeOfInterest, TLEList *someLEListToIgnore);

		// model run methods
		OSErr				Reset ();
		OSErr				Run (Seconds stopTime);
		OSErr 				FirstStepUserInputChecks(void);
		OSErr				Step ();
		OSErr				StepBackwards ();
		OSErr				StepBack ();
		OSErr 				RunTill(void);
		OSErr				RunTill(Seconds stopTime);
		void				Weather ();
		void				CleanUp ();

		// I/O methods
		virtual OSErr 		Read  (BFPB *bfpb); 		// read from the current position
		virtual OSErr 		Write (BFPB *bfpb); 		// write to  the current position
		virtual OSErr		ReadFromPath (char *path);	// creates a buffer, then calls Read  () 
		virtual OSErr		WriteToPath  (char *path);	// creates a buffer, then calls Write ()
	
		virtual void		Draw(Rect r, WorldRect view);
		virtual long		GetListLength();
		virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
		virtual OSErr 		AddItem(ListItem item);
		virtual OSErr 		SettingsItem(ListItem item);
		virtual OSErr 		DeleteItem(ListItem item);

		void				SetDialogVariables (TModelDialogVariables var);
		TModelDialogVariables GetDialogVariables () { return fDialogVariables; }
		void				SetStartTime (Seconds newStartTime) { fDialogVariables.startTime = newStartTime; }
		Seconds				GetStartTime () { return fDialogVariables.startTime; }
		void				SetDuration (Seconds newDuration) { fDialogVariables.duration = newDuration; }
		Seconds				GetDuration () { return fDialogVariables.duration; }
		Seconds				GetEndTime () { return fDialogVariables.startTime + fDialogVariables.duration; }
		void				SetModelTime (Seconds newModelTime);
		Seconds				GetModelTime () { return modelTime; }
		void				SetTimeStep (Seconds newTimeStep) { fDialogVariables.computeTimeStep = newTimeStep; }
		Seconds				GetTimeStep () { return fDialogVariables.computeTimeStep; }
		Boolean				PreventLandJumping () { return fDialogVariables.preventLandJumping; }
		void				SetPreventLandJumping (Boolean bTrueFalse) { fDialogVariables.preventLandJumping = bTrueFalse; }
		Boolean				WantOutput () { return fWantOutput; }
		void				SetWantOutput (Boolean bTrueFalse) { fWantOutput = bTrueFalse; }
		void				SetOutputStep (Seconds newOutputStep) { fOutputTimeStep = newOutputStep; }
		Seconds				GetOutputStep () { return fOutputTimeStep; }
		Seconds 			GetRunDuration () { return modelTime - fDialogVariables.startTime; }		// zero-based current time
		void				SetUncertain (Boolean bNewUncertain); //JLM 9/1/98 
		Boolean				IsUncertain () { return fDialogVariables.bUncertain; }
		Boolean 			DrawingDependsOnTime(void);
		void				SetSaveFileName (char *newName) { strcpy (fSaveFileName, newName); }
		void				GetSaveFileName (char *theName) { strcpy (theName, fSaveFileName); }
		void				SetOutputFileName (char *newName) { strcpy (fOutputFileName, newName); }
		void				GetOutputFileName (char *theName) { strcpy (theName, fOutputFileName); }
		OSErr				GetOutputFileName (short fileNumber, short typeCode, char *fileName);
		OSErr 				GetTempLEOutputFilePathName(short fileNumber, char* path,short *vRefNumPtr);//JLM
		OSErr 				GetTempLEOutputFilePathNameTake2(short fileNumber, char* path,short *vRefNumPtr);//JLM
		OSErr				SaveOSSMLEFile (Seconds fileTime, short n);
		OSErr				SaveMossLEFile (Seconds fileTime, short n);
		OSErr 			SaveSimpleAsciiLEFile (Seconds fileTime, short fileNumber);
		Seconds				GetLastComputeTime () { return lastComputeTime; }
		OSErr 				GetTotalAmountSpilled(short desiredMassvolUnits,double *amtTotal);
		short 				GetMassUnitsForTotals();
		OSErr 				GetTotalBudgetTableHdl(short desiredMassVolUnits, BudgetTableDataH *totalBudgetTable);
		OSErr 				ExportBudgetTableHdl(char* path);
		void				ReDisperseOil(LERec* thisLE, double breakingWaveHeight);

		OSErr 				CheckMaxModelDuration(float durationHours,char * errStr);

		OSErr				OpenMovieFile ();
		OSErr				CloseMovieFile ();

		OSErr 				SquirrelAwayLastComputedTimeStuff (void);
		OSErr 				ReinstateLastComputedTimeStuff (void);
		OSErr 				DisposeLastComputedTimeStuff (void);
		OSErr 				TemporarilyShowFutureTime (Seconds futureTime);
		OSErr 				SetModelToPastTime (Seconds pastTime);
	
	private:
		long				GetLineIndex (long lineNum);
		void				ReleaseLEs ();
		void				PossiblyReFloatLE (TMap *theMap, TLEList *theLEList, long i, LETYPE leType);
		void 				DisperseOil(TLEList* theLEList, long index);
		void 				UpdateWindage(TLEList* theLEList);
		OSErr 				TellMoversPrepareForStep();
		void 				TellMoversStepIsDone();

		Seconds 			ClosestSavedModelLEsTime(Seconds givenTime);
		Seconds 			NextSavedModelLEsTime(Seconds givenTime);
		Seconds 			PreviousSavedModelLEsTime(Seconds givenTime);
		OSErr				SaveModelLEs (Seconds forTime, short fileNumber);
		OSErr				SaveModelLEs (BFPB *bfpb);
		OSErr				LoadModelLEs (char *LEFileName);
		OSErr				LoadModelLEs (BFPB *bfpb);
		OSErr				LoadModelLEs (Seconds forTime, Seconds *actualTime);
		
		void				UpdateFrameMapList ();
		Boolean				HasFrameMapListChanged ();
		
		OSErr				SaveMovieFrame (WindowPtr movieWindow, Rect frameRect);
		OSErr 				SaveOutputSeriesFiles(Seconds oldTime,Boolean excludeRunBarFile);
		
		long 				NumLEs(LETYPE  leType);
		long 				NumOutputSteps(Seconds outputTimeStep);
		OSErr 				WriteRunSpillOutputFileHeader(BFPB *bfpb,Seconds outputStep,char* noteStr);
		OSErr 				AppendLEsToRunSpillOutputFile(BFPB *bfpb);

};



TModelDialogVariables DefaultTModelDialogVariables(Seconds start);
Boolean EqualTModelDialogVariables(TModelDialogVariables* var1,TModelDialogVariables *var2);
void TellPlayersAboutNewDirt(void);
#define DIRTY_EVERYTHING -1
#define DIRTY_RUNBAR 1
#define DIRTY_LIST 2
#define DIRTY_MAPDRAWINGRECT  4
#define DIRTY_TOOLBAR  8
#define DIRTY_ENTIREWINDOW  16
// powers of 2 so we can do bit checks

//++ TTimeValue

#include "TimeValue/TimeValue_b.h"
#include "TimeValue/TimeValue_c.h"
#include "TimeValue/TimeValue_g.h"
#include "TimeValue/TTimeValue.h"

//--

//++ TOSSMTimeValue

#include "OSSMTimeValue/OSSMTimeValue_b.h"
#include "OSSMTimeValue/OSSMTimeValue_c.h"
#include "OSSMTimeValue/OSSMTimeValue_g.h"
#include "OSSMTimeValue/TOSSMTimeValue.h"


//--


TOSSMTimeValue* LoadTOSSMTimeValue(TMover *theOwner,short unitsIfKnownInAdvance);
TOSSMTimeValue* CreateTOSSMTimeValue(TMover *theOwner,char* path, char* shortFileName, short unitsIfKnownInAdvance); //JLM
OSErr GetScaleFactorFromUser(char *msg, double *scaleFactor);

///////////////////////////////////////////////////////////////////////////

class ADCPTimeValue : public TTimeValue
{
	public:
		TimeValuePairH3D			timeValues;
		char 					fileName [kMaxNameLen];
		short					fFileType; //JLM
		short					fUserUnits; //JLM
		double					fScaleFactor; // user input for scaling height derivatives or hydrology files
		char 					fStationName [kMaxNameLen];
		WorldPoint				fStationPosition;	// will need a handle for a set of positions and a set of depths for each
		double					fStationDepth;
		long					fNumBins;
		double					fBinSize;
		long					fGMTOffset;
		short					fSensorOrientation;
		//Boolean					bOSSMStyle;
		Boolean					bStationPositionOpen;
		Boolean					bStationDataOpen;
		DOUBLEH					fBinDepthsH;

	public:
								ADCPTimeValue (TMover *theOwner);
								ADCPTimeValue (TMover *theOwner,TimeValuePairH3D tvals,short userUnits);
		virtual				   ~ADCPTimeValue () { Dispose (); }

		virtual OSErr 			MakeClone(TClassID **clonePtrPtr);
		virtual OSErr 			BecomeClone(TClassID *clone);

		virtual OSErr			InitTimeFunc ();
		virtual OSErr			ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance);
		virtual OSErr			ReadTimeValues2 (char *path, short format, short unitsIfKnownInAdvance);
		OSErr					ReadMetaDataFile (char *path);
				//OSErr 			ReadHydrologyHeader (char *path);
		virtual ClassID 		GetClassID () { return TYPE_ADCPTIMEVALUES; }
		virtual Boolean			IAm(ClassID id) { if(id==TYPE_ADCPTIMEVALUES) return TRUE; return TTimeValue::IAm(id); }
		virtual void			Dispose ();

		virtual OSErr			GetTimeValue (Seconds time, VelocityRec *value);
		virtual OSErr			GetTimeValueAtDepth (long depthIndex, Seconds time, VelocityRec *value);
		double					GetBinDepth(long depthIndex);
		virtual OSErr			CheckStartTime (Seconds time);
		virtual void			RescaleTimeValues (double oldScaleFactor, double newScaleFactor);
		OSErr					GetDepthIndices(float depthAtPoint, float totalDepth, long *depthIndex1, long *depthIndex2);
		virtual long			GetNumValues ();
		virtual TimeValuePairH3D	GetTimeValueHandle () { return timeValues; }
		virtual void			SetTimeValueHandle (TimeValuePairH3D t) ;
		virtual void 			GetTimeFileName (char *theName) { strcpy (theName, fileName); }
		virtual void 			SetTimeFileName (char *theName) { strcpy (fileName, theName); }
		virtual void 			GetStationName (char *theName) { strcpy (theName, fStationName); }
		virtual void 			SetStationName (char *theName) { strcpy (fStationName, theName); }
		virtual short			GetUserUnits(){return fUserUnits;}
		virtual void			SetUserUnits(short userUnits){fUserUnits=userUnits;}
		virtual short			GetFileType	() { return fFileType; }
		virtual void			SetFileType	(short fileType) { fFileType = fileType; }
		long					GetNumBins(){return fNumBins; }
		double					GetBinSize(){return fBinSize; }
		long					GetSensorOrientation(){return fSensorOrientation; }
		double					GetStationDepth(){return fStationDepth; }
		WorldPoint				GetStationPosition(){return fStationPosition; }

		// I/O methods
		virtual OSErr 			Read  (BFPB *bfpb);  // read from current position
		virtual OSErr 			Write (BFPB *bfpb);  // write to  current position
		virtual double			GetMaxValue();

		virtual long 			GetListLength ();
		virtual ListItem 		GetNthListItem 	(long n, short indent, short *style, char *text);
		virtual Boolean 		ListClick 	  	(ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 		FunctionEnabled (ListItem item, short buttonID);

		virtual OSErr 			CheckAndPassOnMessage(TModelMessage *message);

	private:
		OSErr					GetInterpolatedComponent (Seconds forTime, double *value, short index);
		OSErr					GetInterpolatedComponentAtDepth (long depthIndex, Seconds forTime, double *value, short index);
		OSErr					GetTimeChange (long a, long b, Seconds *dt);
};

ADCPTimeValue* LoadADCPTimeValue(TMover *theOwner,short unitsIfKnownInAdvance);
ADCPTimeValue* CreateADCPTimeValue(TMover *theOwner,char* path, char* shortFileName, short unitsIfKnownInAdvance); //JLM

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
#include "PtCurMover.h"
#include "TriCurMover.h"
#include "NetCDFMover.h"




#ifdef MAC
#ifdef MPW
void MyDebugSysBeep(short x);
#define SysBeep(x)  MyDebugSysBeep(x) // magic macro
#endif
#endif

#endif
