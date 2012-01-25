
#ifndef __UNCERTAINTY__
#define __UNCERTAINTY__

//////////////
/////////////
typedef struct
{
	Boolean			setEddyValues;
	Seconds			fUncertainStartTime;
	double			fDuration; 				// duration time for uncertainty;
	double			fEddyDiffusion;		
	double			fEddyV0;			
	double			fDownCurUncertainty;	
	double			fUpCurUncertainty;	
	double			fRightCurUncertainty;	
	double			fLeftCurUncertainty;	
} CurrentUncertainyInfo;



OSErr CurrentUncertaintyDialog(CurrentUncertainyInfo *info, WindowPtr parentWindow, Boolean *uncertaintyChanged);


#endif

