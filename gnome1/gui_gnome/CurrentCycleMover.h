
#ifndef __CURRENTCYCLEMOVER__
#define __CURRENTCYCLEMOVER__

#include "CurrentCycleMover_c.h"

#include "GridCurrentMover.h"

Boolean IsCurrentCycleFile (char *path, short *gridType);


class CurrentCycleMover : virtual public CurrentCycleMover_c,  public GridCurrentMover
{

public:
	//WorldPoint 		refP; 					// location of tide station or map-join pin
	long 			refZ; 					// meters, positive up
	short 			scaleType; 				// none, constant, or file
	double 			scaleValue; 			// constant value to match at refP
	char 			scaleOtherFile[32]; 	// file to match at refP
	double 			refScale; 				// multiply current-grid value at refP by refScale to match value
	Boolean 		bRefPointOpen;
	//Boolean			bUncertaintyPointOpen;
	Boolean 		bTimeFileOpen;
	//Boolean			bTimeFileActive;		// active / inactive flag
	//Boolean 		bShowGrid;
	//Boolean 		bShowArrows;
	//double 			arrowScale;
	//float 			arrowDepth;
	Boolean			bApplyLogProfile;
	double			fEddyDiffusion;			// cm**2/s minimum eddy velocity for uncertainty
	double			fEddyV0;			//  in m/s, used for cutoff of minimum eddy for uncertainty
	//TCM_OPTIMZE fOptimize; // this does not need to be saved to the save file	
	
	public:
							CurrentCycleMover (TMap *owner, char *name);
						   ~CurrentCycleMover () { Dispose (); }
	//virtual void		Dispose ();
	//virtual OSErr		InitMover (); //  use TCATSMover version which sets grid ?
	virtual ClassID 	GetClassID () { return TYPE_CURRENTCYCLEMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_CURRENTCYCLEMOVER) return TRUE; return GridCurrentMover::IAm(id); }

	virtual Boolean 	OkToAddToUniversalMap();
	virtual	OSErr 		ReplaceMover();
	virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
	virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
	virtual Boolean 	CurrentUncertaintySame (CurrentUncertainyInfo info);
	// I/O methods
	virtual OSErr		InitMover (TimeGridVel *grid); 

	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	//virtual OSErr		TextRead(char *path, TMap **newMap);
	virtual OSErr		TextRead(char *path, TMap **newMap, char *topFilePath);
	

	// list display methods
	
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
	virtual void		Draw (Rect r, WorldRect view);
	virtual Boolean		DrawingDependsOnTime(void);
	
	virtual long		GetListLength ();
	virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//virtual OSErr 	AddItem (ListItem item);
	virtual OSErr 		SettingsItem (ListItem item);
	virtual OSErr 		DeleteItem (ListItem item);
	
	
	//virtual OSErr 		SettingsDialog();
	

};


typedef struct
{
	Seconds			fUncertainStartTime;
	double			fDuration; 				// duration time for uncertainty;
	/////
	WorldPoint 		refP; 					// location of tide station or map-join pin
	long 				refZ; 					// meters, positive up
	short 			scaleType; 				// none, constant, or file
	double 			scaleValue; 			// constant value to match at refP
	char 				scaleOtherFile[32]; 	// file to match at refP
	double 			refScale; 				// multiply current-grid value at refP by refScale to match value
	Boolean			bTimeFileActive;		// active / inactive flag
	Boolean 			bShowGrid;
	Boolean 			bShowArrows;
	double 			arrowScale;
	double			fEddyDiffusion;		
	double			fEddyV0;			
	double			fDownCurUncertainty;	
	double			fUpCurUncertainty;	
	double			fRightCurUncertainty;	
	double			fLeftCurUncertainty;	
} CurrentCycleDialogNonPtrFields;

#endif //  __CURRENTCYCLEMOVER__
