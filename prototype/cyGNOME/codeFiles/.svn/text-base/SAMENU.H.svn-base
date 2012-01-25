
/////////////////////////////////////////////////////////////////////////SAMENU.h


#ifdef IBM
	#define SAMenuActionProcPtr  ProcPtr
#else
	// JLM -- my MPW was messing up the size of the paramaters so I defined this 
	//typedef void (*SAMenuActionProcPtr)(MenuHandle theMenu, short item, Rect *r, char bOn,
	typedef void (*SAMenuActionProcPtr)(MenuHandle theMenu, long item, Rect *r, long bOn,
					   const RGBColor *DefaultFore,
					   const RGBColor *DefaultBack);
#endif

typedef
   struct {
      short       cellHeight;
      short       menuWidth;
      short       bFixedWidth;
      short       numColumns;
     // ProcPtr		drawProc;
     // ProcPtr     invertProc;
     // ProcPtr     disableProc;
      SAMenuActionProcPtr		drawProc;
      SAMenuActionProcPtr     invertProc;
      SAMenuActionProcPtr     disableProc;
      long        storage;
   } MenuDataRec, *MENUDATARECPTR;

typedef
   struct {
      long        dialogID;
      DialogPtr   dialogPtr;
      long        popupItemNum;
      long        titleItemNum;
      long        menuID;
      short       itemWidth;
      short       lastItemSelected;
      Boolean     bStatic;
      //void			(*drawProc)(MenuHandle, short, Rect *, char, const SysEnvRec *, const RGBColor *, const RGBColor *);
      SAMenuActionProcPtr     drawProc;
   } PopInfoRec, *POPINFORECPTR;

#define kCheckMark      0x12
#define kCharWidth      14

#define kUp                1
#define kDown              2

#define ENABLE_BIT      1


short GetPopSelection(DialogPtr theDialog, short item);

void SetPopSelection(DialogPtr theDialog, short item, short selection);

void HideDPopup(DialogPtr theDialog, short item);

int LoadPopupMenu (DialogPtr theDialog, int menuID, short ctlID);

void ShowDPopup(DialogPtr theDialog, short item);

void RegisterPopUpDialog(long dialogID, DialogPtr theDialog);

Boolean MenuItemEnabled(MenuHandle menu, short item);

Boolean PopClick(DialogPtr theDialog, short itemHit, LONGPTR result);

void PopErase(DialogPtr theDialog, short itemNum);

void RegisterPopTable(POPINFORECPTR newTable, short newNumPopUps);

//JLM 8/13/98
typedef struct
{
	POPINFORECPTR popTable;
	short numPopUps;
} PopTableInfo;
PopTableInfo SavePopTable( void );
void RestorePopTableInfo(PopTableInfo pti);

#ifndef IBM
#if TARGET_API_MAC_CARBON
pascal_ifMac void SaMenu_MDEF( short msg, MenuRef menu, Rect* bounds, Point hitPt, short* whichItem );
#endif
//#define SAMenuActionUPP ProcPtr
#define SAMenuActionUPP SAMenuActionProcPtr

#ifdef VERSIONPPC
enum { uppSAMenuActionProcInfo = kCStackBased
		 | STACK_ROUTINE_PARAMETER(1, SIZE_CODE(sizeof(MenuHandle)))
		 | STACK_ROUTINE_PARAMETER(2, SIZE_CODE(sizeof(short))) // int in MPW
		 | STACK_ROUTINE_PARAMETER(3, SIZE_CODE(sizeof(Rect *)))
		 | STACK_ROUTINE_PARAMETER(4, SIZE_CODE(sizeof(char))) // Boolean in MPW
		// | STACK_ROUTINE_PARAMETER(5, SIZE_CODE(sizeof(SysEnvRec *)))
		 | STACK_ROUTINE_PARAMETER(5, SIZE_CODE(sizeof(RGBColor *)))
		 | STACK_ROUTINE_PARAMETER(6, SIZE_CODE(sizeof(RGBColor *))) };
#else
	#define uppSAMenuActionProcInfo 0
#endif

void DefineMenu(int menuID,
				short cellHeight,
				short menuWidth,
				Boolean bFixedWidth,
				short numColumns,
				//ProcPtr drawProc,
				//ProcPtr invertProc,
				//ProcPtr disableProc);
				SAMenuActionProcPtr drawProc,
				SAMenuActionProcPtr invertProc,
				SAMenuActionProcPtr disableProc);

void DrawCheck(RECTPTR r);

void DefaultMenuColors(RGBCOLORPTR DefaultFore, RGBCOLORPTR DefaultBack);

void DrawArrow(RECTPTR rPtr, short upOrDown, Boolean disabled);

#endif

#ifndef IBM
pascal_ifMac void PopDraw(DialogPtr theDialog, short itemNum);
#else
void PopDraw (DialogPtr theDialog, short itemNum);
#endif

void PopInvert(DialogPtr theDialog, short itemNum);
/////////////
#ifdef MAC
	short NewSAMenu_ExtraMenuWidth(short menuID);
	void NewSAMenu_DrawExtra(short menuID, short itemNum, Rect extraRect);
	void NewSAMenu_SetTextToEmptyForAllMenuItems(short menuID);
	short MenuBarEquivalentID(short menuID); // this compensates for the Carbonized MAC version having two different menus, one for the popup and one for the menu bar
#endif
