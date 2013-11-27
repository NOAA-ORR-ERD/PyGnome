/*
 *  OLEList_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __OLEList_c__
#define __OLEList_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "GuiTypeDefs.h"
#include "LEList_c.h"
#include "CMYLIST.H"
#include "MemUtils.h"

class OLEList_c : virtual public LEList_c {

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
	RGBColor			fColor;
	
						OLEList_c ();
	long				GetLECount () { return fSetSummary.numOfLEs; }
	//virtual ClassID 	GetClassID () { return TYPE_OSSMLELIST; }
	//virtual Boolean		IAm(ClassID id) { if(id==TYPE_OSSMLELIST) return TRUE; return LEList_c::IAm(id); }

	
	virtual long		GetMassUnits () { return fSetSummary.massUnits; } 
	virtual short		GetMassUnitType();
	virtual double		GetTotalMass () { return fSetSummary.totalMass; } 
	virtual OilType		GetOilType(){return fSetSummary.pollutantType;}
	virtual Seconds		GetSpillStartTime() { return fSetSummary.startRelTime; }
	virtual Seconds		GetSpillEndTime() { return fSetSummary.endRelTime; }
	virtual void 		GetLEAmountStatistics(short desiredMassVolUnits, double *amtTotal,double *amtReleased,double *amtEvaporated,
											  double *amtDispersed,double *amtBeached,double *amtOffmap, double *amtFloating, double *amtRemoved);
	virtual OSErr 		CalculateAverageIntrusionDepth(double *avDepth, double *stdDev);
	
	virtual DispersionRec 		GetDispersionInfo ();
	virtual void 				SetDispersionInfo (DispersionRec info);
	
	virtual AdiosInfoRecH 		GetAdiosInfo () {return fAdiosDataH;}
	virtual void 				SetAdiosInfo (AdiosInfoRecH adiosInfoH){if (fAdiosDataH && fAdiosDataH != adiosInfoH) {DisposeHandle((Handle)fAdiosDataH);} fAdiosDataH = adiosInfoH;}
	long 				GetNumAdiosBudgetTableItems(void);
	void 				AddToBudgetTableHdl(BudgetTableData *budgetTable);
	BudgetTableDataH	GetBudgetTable(){return fBudgetTableH;}
	
	virtual WindageRec 	GetWindageInfo ();
	virtual void 		SetWindageInfo (WindageRec info);
};


#endif