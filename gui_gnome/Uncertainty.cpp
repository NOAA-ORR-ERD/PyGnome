
#include "Cross.h"

#include "Uncertainty.h"

static CurrentUncertainyInfo sInfo;
static Boolean sDialogValuesChanged;

short CurUncertainClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
#pragma unused (data)
	double eddyV0;

	switch(itemNum)
	{
		case CU_OK:
		{	
			Seconds startTime;
			double duration, downCur, upCur, leftCur, rightCur, eddyDiffusion; 

			// ignore certain fields that are not used by component movers, STH
			if (sInfo.setEddyValues)
			{
				eddyV0 = EditText2Float(dialog, CU_EDDYV0);
				eddyDiffusion = EditText2Float(dialog, CU_EDDYDIFFUSION);
				if(eddyV0 <= 0.0)
				{
					printError("The Eddy V0 value must be greater than zero.");
					MySelectDialogItemText (dialog, CU_EDDYV0, 0, 100);
					break;
				}

				if (sInfo.fEddyV0 != eddyV0 || sInfo.fEddyDiffusion != eddyDiffusion) sDialogValuesChanged = true;

				sInfo.fEddyDiffusion = EditText2Float(dialog, CU_EDDYDIFFUSION);
				sInfo.fEddyV0 = eddyV0;
			}
			
			duration = EditText2Float(dialog, CU_DURATION)*3600;
			downCur = EditText2Float(dialog, CU_DOWNCUR)/100;
			upCur = EditText2Float(dialog, CU_UPCUR)/100;
			rightCur = EditText2Float(dialog, CU_RIGHTCUR)/100;
			leftCur = EditText2Float(dialog, CU_LEFTCUR)/100;
			startTime = round(EditText2Float(dialog,CU_STARTTIME)*3600);

			if (sInfo.fUpCurUncertainty != upCur || sInfo.fRightCurUncertainty != rightCur
				|| sInfo.fDownCurUncertainty != downCur || sInfo.fLeftCurUncertainty != leftCur
				|| sInfo.fUncertainStartTime != startTime || sInfo.fDuration != duration) sDialogValuesChanged = true;

			sInfo.fDuration = EditText2Float(dialog, CU_DURATION)*3600;
			sInfo.fDownCurUncertainty = EditText2Float(dialog, CU_DOWNCUR)/100;
			sInfo.fUpCurUncertainty = EditText2Float(dialog, CU_UPCUR)/100;
			sInfo.fRightCurUncertainty = EditText2Float(dialog, CU_RIGHTCUR)/100;
			sInfo.fLeftCurUncertainty = EditText2Float(dialog, CU_LEFTCUR)/100;
			sInfo.fUncertainStartTime = round(EditText2Float(dialog,CU_STARTTIME)*3600);

			return CU_OK;
		}
			
		case CU_CANCEL:
			return CU_CANCEL;
			break;
			
		case CU_EDDYDIFFUSION:		
			CheckNumberTextItem(dialog, itemNum, FALSE); //  don't allow decimals
			break;

		case CU_DURATION:		
		case CU_UPCUR:		
		case CU_RIGHTCUR:		
		case CU_STARTTIME:		
		case CU_EDDYV0:		
			CheckNumberTextItem(dialog, itemNum, TRUE); //  allow decimals
			break;

		case CU_DOWNCUR:		
		case CU_LEFTCUR:		
			CheckNumberTextItemAllowingNegative(dialog, itemNum, TRUE); //  allow decimals
			break;
	}
	 
	return 0;
}

OSErr CurUncertainInit(DialogPtr dialog, VOIDPTR data)
{
	#pragma unused (data)
	Float2EditText(dialog,CU_DURATION,sInfo.fDuration/3600,2);
	Float2EditText(dialog,CU_DOWNCUR,sInfo.fDownCurUncertainty*100,2);
	Float2EditText(dialog,CU_UPCUR,sInfo.fUpCurUncertainty*100,2);
	Float2EditText(dialog,CU_LEFTCUR,sInfo.fLeftCurUncertainty*100,2);
	Float2EditText(dialog,CU_RIGHTCUR,sInfo.fRightCurUncertainty*100,2);
	Float2EditText(dialog, CU_STARTTIME, sInfo.fUncertainStartTime / 3600.0, 2);
	if (sInfo.setEddyValues)
	{
		Float2EditText(dialog,CU_EDDYDIFFUSION,sInfo.fEddyDiffusion,0);
		Float2EditText(dialog,CU_EDDYV0,sInfo.fEddyV0,0);
	}
	else
	{
		ShowHideDialogItem(dialog,CU_EDDYV0LABEL,false);
		ShowHideDialogItem(dialog,CU_EDDYV0,false);
		ShowHideDialogItem(dialog,CU_EDDYV0UNITS,false);
		
		ShowHideDialogItem(dialog,CU_EDDYDIFFUSIONLABEL,false);
		ShowHideDialogItem(dialog,CU_EDDYDIFFUSION,false);
		ShowHideDialogItem(dialog,CU_EDDYDIFFUSIONUNITS,false);
	}
	MySelectDialogItemText(dialog, CU_DOWNCUR, 0, 255);

	return 0;
}

OSErr CurrentUncertaintyDialog(CurrentUncertainyInfo *info, WindowPtr parentWindow, Boolean *uncertaintyChanged)
{
	short item;
	if(parentWindow == nil) parentWindow = mapWindow; // JLM 6/2/99, we need the parent on the IBM
	if(info == nil) return -1;
	sInfo = *info;
	sDialogValuesChanged = *uncertaintyChanged;
	item = MyModalDialog(CUR_UNCERTAINTY_DLGID, parentWindow, 0, CurUncertainInit, CurUncertainClick);
	if (item == CU_OK) {
		*info = sInfo;
		*uncertaintyChanged = sDialogValuesChanged;
		if(parentWindow == mapWindow) {
			model->NewDirtNotification(); // when a dialog is the parent, we rely on that  dialog's to notify about Dirt 
			// that way we don't get the map redrawing behind the parent dialog on the IBM
		}
	}
	return item ==CU_OK? 0 : -1;
}
