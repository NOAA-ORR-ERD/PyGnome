
typedef struct {
	//GrafPtr			port;			// List's grafPort
	WindowRef			window;			// List's window, on the MAC either the WindowRef of the window or the WindowRef of the Dialog window,  in Windows, the HWND of the dialog item
	Rect			viewRect;		// List's display rectangle
	short			lineHeight;		// Height of a line in the list in pixels
	long			viewLines;		// Number of lines in view rectangle
	ControlHandle 	scroll;			// Vertical scroll control
	long			clickTime;		// Time of last click
	long			keyTime;		// Time of last key down
	long			selectedRec;	// Last record clicked
	long			curRec;			// Current record number of first record in view
	long			numItems;		// number of items in the list
	long			realControlMaximum;
	long			realControlValue;
	void 			(*draw)(DialogPtr, Rect *, long);	// user draw routine
	// to ease code transition to CARBON,  we have added a separate draw routine in the case the VList is in a window (since almost all VLists are coded for dialogs)
	//void 			(*draw)(DialogRef, Rect *, long);	// user draw routine when list is in a dialog
	void 			(*drawInWindow)(WindowRef, Rect *, long);	// user draw routine when list is in a Window
	//////////////
	void 			(*hRect)(DialogPtr, Rect *, long);	// user-specified hilite rect
} VList, *VLISTPTR;

typedef struct {
	pascal_ifMac Boolean (*userFilter)(DialogPtr, EVENTRECORDPTR, SHORTPTR);
	void (*userSearchText)(DialogPtr, long, CHARPTR);
	Boolean OKflag;
#ifdef IBM
	short sharedListItem;
	short *sharedDialogItem;
	short sharedItemHit;
	long sharedNumItems;
	long sharedChosenItem;
	void (*sharedInit)(DialogPtr, VLISTPTR);
	void (*sharedDraw)(DialogPtr, RECTPTR, long);
	Boolean (*sharedClick)(DialogPtr, VLISTPTR, short, LONGPTR, Boolean);
	VList sharedVList;
#endif
} VListSettings;

#ifdef MAC
	#define LISTTEXTSIZE 9
#else
	#define LISTTEXTSIZE 9
#endif

void VLInvalidate(VLISTPTR L);
void VLUpdate(VLISTPTR L);
void VLUpdateLine(VLISTPTR L, long record);
void VLUpdateSelectedLine(VLISTPTR L);
void VLSetSelect(long newSel, VLISTPTR L);
Boolean VLGetSelect(LONGPTR theRecord, VLISTPTR listHdl);
Boolean VLClick(Point pt, VLISTPTR L);
void ShowHideVList(DialogPtr dialog, VLISTPTR L, Boolean shown);
void VLReset(VLISTPTR L, long numItems);
void VLResetNR(VLISTPTR L, long numItems);
void VLDeleteItem(long item, VLISTPTR L);
void VLAddItem(long item, VLISTPTR L);
Boolean VLAddRecordRowIsSelected(VLISTPTR L);
void VLAutoScroll(VLISTPTR L);
Boolean VListKeyEvent(EVENTRECORDPTR e, VLISTPTR L);
void VLResize(VLISTPTR L, Rect r);
OSErr VLNewInWindow(WindowRef window, RECTPTR r, long numItems,void (*draw)(WindowRef, RECTPTR, long), VLISTPTR L);
OSErr VLNew(DialogPtr diawind, short listItem, RECTPTR r, long numItems,
			void (*draw)(DialogPtr, RECTPTR, long), VLISTPTR L);
void VLDispose(VLISTPTR L);
long SelectFromVListDialog(short dialogID, short listItem, long numItems,
						   void (*init)(DialogPtr, VLISTPTR),
						   void (*searchText)(DialogPtr, long, CHARPTR),
						   pascal_ifMac Boolean (*filter)(DialogPtr, EVENTRECORDPTR, SHORTPTR),
						   void (*draw)(DialogPtr, RECTPTR, long),
						   Boolean (*click)(DialogPtr, VLISTPTR, short, LONGPTR, Boolean),
						   Boolean OKanyTime, short *dialogItem);
void SaveVListSettings(VListSettings *vSettings);
void RestoreVListSettings(VListSettings vSettings);
long myGetControlValue(VLISTPTR L, ControlHandle c);

#ifdef IBM
BOOL CALLBACK SVLWinDlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK VListWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
#endif
#ifdef MAC
void FillRectWithWhite(Rect * r);
#endif
