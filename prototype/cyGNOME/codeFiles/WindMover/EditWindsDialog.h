#ifndef __EDITWINDSDIALOG__
#define __EDITWINDSDIALOG__
#include "Earl.h"
#include "TypeDefs.h"

enum{
		EWOK=1,
		EWHILITEDDEFAULT,
		EWCANCEL,
		EWHELP,
		EWDELETEROWS_BTN,
		EWDELETEALL,
		EWREPLACE,
		EWMONTHSPOPUP,
		EWDAY,
		EWYEARSPOPUP,
		EWHOUR,
		EWMINUTE,
		EWSPEED,
		EWDIRECTION,
		EWINCREMENT,
		
		EWDATE_LIST_LABEL,
		EWTIME_LIST_LABEL,
		EWSPEED_LIST_LABEL,
		EWDIRECTION_LIST_LABEL,
	
		EWLIST,
		EWBULLSEYEFRAME,
		EWBULLSEYE,
		EWAUTOINCRTEXT=23,
		EWTIMELABEL=25,
		EWAUTOINCRHOURS=28,

		EWTIMECOLONLABEL=31,
		EWSPEEDPOPUP=32,
		EWWINDATA=34,
		EWSETTINGS,
		EWBUTTONFRAME,
		EWFRAMEINPUT
};

class TWindMover;

// JLM 11/25/98
// structure to help reset stuff when the user cancels from the settings dialog box
typedef struct
{
	Boolean			bActive;
	Seconds			fUncertainStartTime;
	double			fDuration; 				// duration time for uncertainty;
	/////
	double fSpeedScale;
	double fAngleScale;
	//double conversion;
	//double windageA;
	//double windageB;
	// 
	Boolean fIsConstantWind;
	VelocityRec fConstantValue;
} EWDialogNonPtrFields;

void SetEWDialogNonPtrFields(TWindMover * cm,EWDialogNonPtrFields * f);
EWDialogNonPtrFields GetEWDialogNonPtrFields(TWindMover * cm);


OSErr EditWindsDialog(TWindMover *windMover,Seconds startTime,Boolean forWizard,Boolean dontShowSettings);
void Direction2String(double direction,char *s);
void UV2RThetaStrings(double u,double v,long speedUnits,char* text);

void RegisterBullsEye(void);
OSErr CheckWindSpeedLimit(float speed, short speedUnits,char* errStr);
double speedconversion(long speedUnits);
//long StrToSpeedUnits(char* str);

#endif