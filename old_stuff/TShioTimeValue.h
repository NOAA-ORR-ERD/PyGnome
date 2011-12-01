

#ifndef		__TSHIOTIMEVALUE__
#define		__TSHIOTIMEVALUE__

#include "Shio.h"

//enum { WIZ_POPUP = 1, WIZ_UNITS , WIZ_EDIT, WIZ_BMP, WIZ_HELPBUTTON };

/////////////////////////////////////////////////
/////////////////////////////////////////////////
#define MAXNUMSHIOYEARS  20
typedef struct
{
	short year;// 1998, etc
	YEARDATAHDL yearDataHdl;
} ShioYearInfo;

typedef struct
{
	Seconds time;
	double speedInKnots;
	short type;	// 0 -> MinBeforeFlood, 1 -> MaxFlood, 2 -> MinBeforeEbb, 3 -> MaxEbb
} EbbFloodData,*EbbFloodDataP,**EbbFloodDataH;

typedef struct
{
	Seconds time;
	double height;
	short type;	// 0 -> Low Tide, 1 -> High Tide
} HighLowData,*HighLowDataP,**HighLowDataH;

YEARDATAHDL GetYearData(short year);

#define MAXSTATIONNAMELEN  128
#define kMaxKeyedLineLength	1024
class TShioTimeValue : public TOSSMTimeValue
{

	public:
								TShioTimeValue (TMover *theOwner);
								TShioTimeValue (TMover *theOwner,TimeValuePairH tvals);
							   ~TShioTimeValue () { this->Dispose (); }
		virtual OSErr 			MakeClone(TShioTimeValue **clonePtrPtr);
		virtual OSErr 			BecomeClone(TShioTimeValue *clone);
		virtual OSErr			InitTimeFunc ();
		virtual OSErr			ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance);
		virtual ClassID 		GetClassID () { return TYPE_SHIOTIMEVALUES; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_SHIOTIMEVALUES) return TRUE; return TOSSMTimeValue::IAm(id); }
		virtual void			Dispose ();

		virtual long			GetNumEbbFloodValues ();	
		virtual long			GetNumHighLowValues ();
		virtual OSErr			GetTimeValue (Seconds time, VelocityRec *value);
		virtual WorldPoint	GetRefWorldPoint (void);

		virtual	double 		GetDeriv (Seconds t1, double val1, Seconds t2, double val2, Seconds theTime);
		virtual	OSErr 		GetConvertedHeightValue(Seconds forTime, VelocityRec *value);
		virtual	OSErr 		GetProgressiveWaveValue(Seconds forTime, VelocityRec *value);
		OSErr 					GetLocationInTideCycle(short *ebbFloodType, float *fraction);

		// I/O methods
		virtual OSErr 			Read  (BFPB *bfpb);  // read from current position
		virtual OSErr 			Write (BFPB *bfpb);  // write to  current position

		virtual long 			GetListLength (); 
		virtual ListItem 		GetNthListItem 	(long n, short indent, short *style, char *text);
		virtual Boolean 		ListClick	(ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 		FunctionEnabled(ListItem item, short buttonID);

		virtual OSErr 			CheckAndPassOnMessage(TModelMessage *message);
	private:
		OSErr 				GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,float *** val);
		OSErr 				GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,DATA * val);
		OSErr 				GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,short * val);
		OSErr 				GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,float * val);
		OSErr 				GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,double * val);
		OSErr					GetInterpolatedComponent (Seconds forTime, double *value, short index);
		OSErr					GetTimeChange (long a, long b, Seconds *dt);

		void 					ProgrammerError(char* routine);
		void 					InitInstanceVariables(void);

		long 					I_SHIOHIGHLOWS(void);
		long 					I_SHIOEBBFLOODS(void);

		// instance variables
		char fStationName[MAXSTATIONNAMELEN];
		char fStationType;
		double fLatitude;
		double fLongitude;
		CONSTITUENT2 fConstituent;
		HEIGHTOFFSET fHeightOffset;
		CURRENTOFFSET fCurrentOffset;
		//
		Boolean fHighLowValuesOpen; // for the list
		Boolean fEbbFloodValuesOpen; // for the list
		EbbFloodDataH fEbbFloodDataHdl;	// values to show on list for tidal currents
		HighLowDataH fHighLowDataHdl;	// values to show on list for tidal heights
};

Boolean IsShioFile(char* path);

#endif