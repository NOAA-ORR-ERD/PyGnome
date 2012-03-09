
#include "CROSS.h"

// --- SUBROUTINES TO MANANGE USER-DEFINED AND POPUP MENUS ---

#ifdef MAC
	#pragma segment Main
#endif


POPINFORECPTR sa_popTable;
short sa_numPopUps;

// to enable in VU set gVUTrick and gVUTrick to not be equal (JLM)
static long gVUTrick = '_VU_';
static long gVUTrick2 = '_VU_';

// RegisterPopTable() sets the variables sa_popTable and sa_numPopUps.
void RegisterPopTable(POPINFORECPTR newTable, short newNumPopUps)
{
   sa_popTable = newTable;
   sa_numPopUps = newNumPopUps;
}


#ifndef IBM

/****************************************************************
   PopDraw() is the function associated with the user item that
   invokes the popup menu.  We draw the box to look like a menu
   cell, and then call the appropriate drawProc to fill in the
   cell with the actual contents.  If the drawProc is nil, we
   draw the menu item as text.  We gray the cell if the menu
   is disabled.  Finally, we draw a down arrow to indicate that
   the button is a popup menu.

   If the popup item is static, we only draw the cell contents.
 ****************************************************************/
pascal_ifMac void PopDraw(DialogPtr theDialog, short itemNum)
{
   GrafPtr        savePort;
   short          theType;
   Handle         itemHandle;
   Rect           itemBox;
   Rect           cellBox;
   MenuHandle     theMenu;
   //SysEnvRec      theWorld;
   RGBColor       SaveBack, SaveFore, DefaultBack, DefaultFore;
   char           name[256];
   short          i;
   short          drawStringFlag;

   // Added by Glen to code for support of Type-in Pop-Up menus
   // if drawStringFlag  = 1 then we don't draw text string ...
	
   drawStringFlag = 0;

   for ( i = 0 ; i < sa_numPopUps ; i++ )
      if ( (sa_popTable[i].dialogPtr == theDialog) && (sa_popTable[i].popupItemNum == itemNum) )
         break;
                              if (i == sa_numPopUps)
      { SysBeep(1); return; }    // should not happen since the dialog must have been registered
                                             // for PopDraw to have been called

   GetPortGrafPtr(&savePort);
   SetPortDialogPort(theDialog);
  // SysEnvirons(curSysEnvVers,&theWorld);

  // if (theWorld.hasColorQD) {
  if (ColorQDAvailable()) {
      GetForeColor(&SaveFore);
      GetBackColor(&SaveBack);
      DefaultMenuColors(&DefaultFore, &DefaultBack);
      if (sa_popTable[i].bStatic) {
         DefaultFore = SaveFore;
         DefaultBack = SaveBack;
      }
   }

   theMenu = GetMenuHandle(sa_popTable[i].menuID);
   if (!theMenu) { SysBeep(1); return; }
   
   /* change item's width to match the menu */
   GetDialogItem(theDialog,itemNum,&theType,&itemHandle,&itemBox);
   CalcMenuSize(theMenu);
   if (sa_popTable[i].itemWidth == 0) {
      if (sa_popTable[i].drawProc == nil)
        // itemBox.right = itemBox.left + (**theMenu).menuWidth + 20 + 2;
         itemBox.right = itemBox.left + GetMenuWidth(theMenu) + 20 + 2;
      else
         //itemBox.right = itemBox.left + (**theMenu).menuWidth + 2;
         itemBox.right = itemBox.left + GetMenuWidth(theMenu) + 2;
   }
   else if (sa_popTable[i].itemWidth == -1) {  // Type-in Pop-Up Menu
      itemBox.right = itemBox.left + 20 + 2;
      drawStringFlag = 1;
   }
   else
      itemBox.right = itemBox.left + sa_popTable[i].itemWidth + 2;
   SetDialogItem(theDialog,itemNum,theType,itemHandle,&itemBox);

   /* draw the box */
   if (TRUE) { // !sa_popTable[i].bStatic
     // if (theWorld.hasColorQD) RGBBackColor(&DefaultBack);
     if (ColorQDAvailable()) RGBBackColor(&DefaultBack);
      EraseRect( &itemBox );
     // if (theWorld.hasColorQD) RGBForeColor(&SaveFore);
     if (ColorQDAvailable())  RGBForeColor(&SaveFore);
      PenNormal();
	 // if (sa_popTable[i].bStatic) PenPat((ConstPatternParam)&qd.gray);
	  //if (sa_popTable[i].bStatic) PenPat((ConstPatternParam)&GRAY_BRUSH());
	  //if (sa_popTable[i].bStatic) PenPat((ConstPatternParam)&GRAY_BRUSH);
	  if (sa_popTable[i].bStatic) PenPatQDGlobalsGray();
      FrameRect(&itemBox);
      /* draw the shadow */
      MoveTo(itemBox.left + 3, itemBox.bottom);
      Line((itemBox.right - itemBox.left) - 3, 0);
      Line(0, -((itemBox.bottom - itemBox.top) - 2));
      PenNormal();
   }
   else
      EraseRect( &itemBox );

   /* draw the current item in the box */

  // if (theWorld.hasColorQD) RGBForeColor(&DefaultFore);
    if (ColorQDAvailable()) RGBForeColor(&DefaultFore);

   // Draw text if no Type-in Pop-Up
   if(drawStringFlag == 0){
      if (sa_popTable[i].drawProc != nil) {
         cellBox = itemBox;
         InsetRect(&cellBox, 1, 1);
         (* sa_popTable[i].drawProc) (theMenu,
                                                sa_popTable[i].lastItemSelected,
                                                &cellBox,         // so the drawProc gets the same-size rect,
                                                                      // whether it's drawing in the menu or in the pop-box
                                                true,           // since we are indeed drawing a pop box item
                                                //&theWorld,
                                                &DefaultFore,
                                                &DefaultBack);
      }
      else {
         MoveTo(itemBox.left + 15, itemBox.top + 4 + (itemBox.bottom - itemBox.top)/2);
         GetMenuItemText(theMenu, sa_popTable[i].lastItemSelected, (unsigned char *)name);
         DrawString((unsigned char *)name);
      }
   }

   if (TRUE) { // !sa_popTable[i].bStatic
      /* cover the item in gray if the menu is disabled */
     // if (!((**theMenu).enableFlags & ENABLE_BIT)) {
	  #if TARGET_API_MAC_CARBON
	 	 Boolean menuIsEnabled = IsMenuItemEnabled(theMenu,0);
	 #else
	 	 Boolean menuIsEnabled = (**theMenu).enableFlags & ENABLE_BIT;
	 #endif
      if (!menuIsEnabled) {
         //PenPat((ConstPatternParam)&qd.gray);
         //PenPat((ConstPatternParam)&GRAY_BRUSH);
		 PenPatQDGlobalsGray();
         PenMode(patOr);
         //if (theWorld.hasColorQD) RGBForeColor(&DefaultBack);
         if (ColorQDAvailable()) RGBForeColor(&DefaultBack); 
         else ForeColor(whiteColor);
         PaintRect(&itemBox);
         PenNormal();
        // if (theWorld.hasColorQD) RGBForeColor(&DefaultFore);
         if (ColorQDAvailable()) RGBForeColor(&DefaultFore);
         else ForeColor(blackColor);
      }

      /* draw the down arrow */
      itemBox.left = itemBox.right - 20;
      DrawArrow(&itemBox, kDown, sa_popTable[i].bStatic);
   }

   //if (theWorld.hasColorQD) {
   if (ColorQDAvailable()) {
     RGBForeColor(&SaveFore);
     RGBBackColor(&SaveBack);
   }

   SetPortGrafPort(savePort);
}
#endif //IBM

#ifdef IBM
void PopDraw (DialogPtr theDialog, short itemNum)
{
	HWND hCtl;
	short i;
	
	for ( i = 0 ; i < sa_numPopUps ; i++ )
		if ( (sa_popTable[i].dialogPtr == theDialog) && (sa_popTable[i].popupItemNum == itemNum) )
			break;
	if (i == sa_numPopUps)
		{ SysBeep(1); return; } // should not happen since the dialog must have been registered
								// for PopDraw to have been called
	
	MyEnableControl(theDialog, itemNum, !sa_popTable[i].bStatic);
	
	hCtl = GetDlgItem (theDialog, itemNum);
	// ShowWindow (hCtl, SW_SHOW); // Done by KW w/ JM.
	SendMessage (hCtl, CB_SETCURSEL, (sa_popTable[i].lastItemSelected)-1 ,0);
}


void PopErase (DialogPtr theDialog, short itemNum)
{
   // HideDialogItem(theDialog, itemNum);
   // ShowDialogItem(theDialog, itemNum);
}

#endif //IBM



#ifndef IBM

/****************************************************************
  PopErase() erases the given popup and its shadow.
****************************************************************/
void PopErase(DialogPtr theDialog, short itemNum)
{
   GrafPtr           savePort;
   short             itemType;
   Handle            itemHandle;
   Rect              itemBox;

   GetPortGrafPtr(&savePort);
   SetPortDialogPort(theDialog);

   GetDialogItem(theDialog, itemNum, &itemType, &itemHandle, &itemBox);
   itemBox.right++;
   itemBox.bottom++;
   EraseRect(&itemBox);

   SetPortGrafPort(savePort);
}

/****************************************************************
  PopInvert() inverts the given popup.
****************************************************************/
void PopInvert(DialogPtr theDialog, short itemNum)
{
   GrafPtr savePort;
   short itemType;
   Handle itemHandle;
   Rect itemBox;
   //SysEnvRec theWorld;
   RGBColor SaveBack, DefaultBack, DefaultFore, saveHilite;
	
   GetPortGrafPtr(&savePort);
   SetPortDialogPort(theDialog);

   GetDialogItem(theDialog, itemNum, &itemType, &itemHandle, &itemBox);

  // SysEnvirons(curSysEnvVers,&theWorld);

   //if (!theWorld.hasColorQD) {
   if (ColorQDAvailable()) {
      InvertRect( &itemBox );
      return;
   }

   GetBackColor(&SaveBack);
	LMGetHiliteRGB(&saveHilite);
   DefaultMenuColors(&DefaultFore, &DefaultBack);

   RGBBackColor(&DefaultBack);
   HiliteColor(&DefaultFore);
	LMSetHiliteMode(LMGetHiliteMode() & 0x7F); // clear 0th bit = pHiliteBit
   InvertRect( &itemBox );
	
   HiliteColor(&saveHilite);
   RGBBackColor(&SaveBack);
	
   SetPortGrafPort(savePort);
}

#else

void PopInvert(DialogPtr theDialog, short itemNum)
{
}

#endif  //IBM

#ifndef IBM

Boolean MenuItemEnabled(MenuHandle menu, short item)
{
#if TARGET_API_MAC_CARBON
	// first check the entire  menu
	if(!IsMenuItemEnabled(menu,0)) return false;
	return IsMenuItemEnabled(menu,item);
	
#else
	return ((**menu).enableFlags & 1) && // low bit means menu itself is enabled
		   (BitTst((Ptr)&((**menu).enableFlags),
						 31 - (item <= 31 ? item : 0)));
#endif
}

Boolean OverridePopClick(DialogPtr theDialog, short itemHit, long *result)
{
	GrafPtr				savePort;
	MenuHandle			theMenu;
	short				count;
	char				name[256],str[256];
	short				i;
	
	if(gVUTrick == gVUTrick2) return false; // we will not override PopClick
	
	for ( i = 0 ; i < sa_numPopUps ; i++ )
		if ( sa_popTable[i].dialogPtr == theDialog && sa_popTable[i].popupItemNum == itemHit )
			break;
	
	if (i == sa_numPopUps || sa_popTable[i].bStatic) {
		*result = 0;
		return false; // we will not override PopClick
	}
	
	// else we have found the i value
	
	theMenu = GetMenuHandle(sa_popTable[i].menuID);
	GetMenuItemText(theMenu, sa_popTable[i].lastItemSelected, (StringPtr)str); 
	my_p2cstr((StringPtr)str); 
	GetPortGrafPtr(&savePort);
	if(REQUEST("enter value to select",str,str) == ok)
	{
		short numItems = CountMenuItems(theMenu);
		UpperText(str,strlen(str));
		//UppercaseText(str,strlen(str),smSystemScript);	//smCurrentScript
		// find the string	
		for ( count = numItems ; count >= 1 ; count-- ) 
		{
			GetMenuItemText(theMenu, count, (StringPtr)name);
			my_p2cstr((StringPtr)name); 
			UpperText(name,strlen(name));
			//UppercaseText(name,strlen(name),smSystemScript);	//smCurrentScript
			if (strcmp(str,name) == 0) {
				if (!MenuItemEnabled(theMenu, count)) {
					REQUEST("Item found but disabled",str,str);
					*result = 0;
					SetPortGrafPort(savePort);
					return false;
				}
				else
					break; // we found it
			}
		}
		if(count == 0) 
		{
			// we did not find the string
			REQUEST("Could not find string in Menu",str,str);
			*result = 0;
			SetPortGrafPort(savePort);
			return false;
		}
		else
		{
			// count holds the menuItem Num we want
			SetItemMark(theMenu, sa_popTable[i].lastItemSelected, noMark);
			sa_popTable[i].lastItemSelected = count;
			SetItemMark(theMenu, sa_popTable[i].lastItemSelected, kCheckMark);
			*result = sa_popTable[i].menuID*0x10000 + sa_popTable[i].lastItemSelected;
		}
	
	}
	SetPortGrafPort(savePort);
	PopDraw(sa_popTable[i].dialogPtr, sa_popTable[i].popupItemNum);
	return(true); //we have overriden PopClick
}
#endif

/****************************************************************
   PopClick() gets called when there is a click in a dialog item
   that might be a popup item.  It determines if the click is
   in a popup and if so handles the click.  It brings up the
   popup on the item that was last selected for this menu.  It
   returns true if the item was a popup item and the mouse was
   released on a valid item.
   In that case, it also stores the value returned
   by PopUpMenuSelect in *result (a long containing the menu and
   item ID selected).  If the item selected was in this menu (as
   opposed to one of its submenus), that item is stored in the
   lastItemSelected field for this popup in the popTable.  In
   that case, the item is also checked, and the previously
   selected item is unchecked.

   For a click on a static popup item, PopClick returns false,
   returns 0 in *result and doesn't call PopUpMenuSelect.

   -- On the IBM, this simply gets the index value of the selected item.

****************************************************************/
Boolean PopClick(DialogPtr theDialog, short itemHit, LONGPTR result)
{
#ifndef IBM
   GrafPtr        savePort;
   short          theType;
   Rect           popBox;
   Rect           titleBox;
   MenuHandle     theMenu;
   long           res;
   short          count;
   char           name[256];
   Handle         itemHandle;
#else
	HANDLE itemHandle;
#endif
   short          i;
   
   #ifndef IBM
   if (OverridePopClick(theDialog, itemHit, result)) return true;
   #endif
   
   for ( i = 0 ; i < sa_numPopUps ; i++ )
      if ( (sa_popTable[i].dialogPtr == theDialog) && (sa_popTable[i].popupItemNum == itemHit) )
         break;

   if (i == sa_numPopUps || sa_popTable[i].bStatic) {
      *result = 0;
      return false;
   }

 #ifdef IBM
   itemHandle = GetDlgItem (theDialog, itemHit);
   sa_popTable[i].lastItemSelected =
    // (SendMessage (itemHandle, CB_GETCURSEL, 0, 0L)+1);  //Combo box item ID's begin at 0
     (SendMessage ((HWND)itemHandle, CB_GETCURSEL, 0, 0L)+1);  //Combo box item ID's begin at 0
	
	return FALSE;
 #else
   GetPortGrafPtr(&savePort);
   SetPortDialogPort(theDialog);
   theMenu = GetMenuHandle(sa_popTable[i].menuID);

   if (sa_popTable[i].drawProc == nil)       // grow items to match size of pop-box
      for ( count = CountMenuItems(theMenu) ; count >= 1 ; count-- ) {
         GetMenuItemText(theMenu, count, (StringPtr)name);
         my_p2cstr((StringPtr)name);
         strcat(name, "     ");
         my_c2pstr(name);
         SetMenuItemText(theMenu, count, (StringPtr)name);
         my_p2cstr((StringPtr)name);
      }

   if (sa_popTable[i].titleItemNum) {
      GetDialogItem(theDialog,sa_popTable[i].titleItemNum,&theType,&itemHandle,&titleBox);
      InvertRect(&titleBox);
   }
   GetDialogItem(theDialog,sa_popTable[i].popupItemNum,&theType,&itemHandle,&popBox);
   LocalToGlobal((Point *)(&popBox.top));
   LocalToGlobal((Point *)(&popBox.bottom));
   res = PopUpMenuSelect( theMenu, popBox.top + 1, popBox.left + 1, sa_popTable[i].lastItemSelected );
   if (sa_popTable[i].titleItemNum)
      InvertRect(&titleBox);
   if (LoWord(res) && (HiWord(res) == sa_popTable[i].menuID)) {
      SetItemMark(theMenu, sa_popTable[i].lastItemSelected, noMark);
      sa_popTable[i].lastItemSelected = LoWord(res);
      SetItemMark(theMenu, sa_popTable[i].lastItemSelected, kCheckMark);
   }

   if (sa_popTable[i].drawProc == nil)       // shrink items back to normal
      for ( count = CountMenuItems(theMenu) ; count >= 1 ; count-- ) {
         GetMenuItemText(theMenu, count, (StringPtr)name);
         my_p2cstr((StringPtr)name);
         name[strlen(name) - 5] = 0;
         my_c2pstr(name);
         SetMenuItemText(theMenu, count, (StringPtr)name);
         my_p2cstr((StringPtr)name);
      }

   *result = res;

   SetPortGrafPort(savePort);

   if (LoWord(res)) {
      PopDraw(sa_popTable[i].dialogPtr, sa_popTable[i].popupItemNum);
      return true;
   }

   return false;
 #endif //IBM
}

/****************************************************************
   RegisterPopUpDialog() is used to fill in the dialogPtr field
   of a PopTable entry and to set the user item procs of that
   dialog's popup items once you have loaded the dialog and have
   the dialog pointer.
 ****************************************************************/
void RegisterPopUpDialog(long dialogID, DialogPtr theDialog)
{
   short i;

 #ifndef IBM
   short                theType;
   Handle               itemHandle;
   Rect                 itemBox;
 #endif

   for ( i = 0 ; i < sa_numPopUps ; i++ )
      if (sa_popTable[i].dialogID == dialogID) {
			sa_popTable[i].dialogPtr = theDialog;
			
			#ifndef IBM
			GetDialogItem(theDialog, sa_popTable[i].popupItemNum, &theType, &itemHandle, &itemBox);
			SetDialogItem(theDialog, sa_popTable[i].popupItemNum, userItem, nil, &itemBox);
			SetDialogItemHandle(theDialog, sa_popTable[i].popupItemNum, (Handle)PopDraw);
			#else
			PopDraw(theDialog, sa_popTable[i].popupItemNum);
			LoadPopupMenu(theDialog, sa_popTable[i].menuID, sa_popTable[i].popupItemNum);
			#endif
      }
}

/****************************************************************
   UnRegisterPopUpDialog() is used when you have a popup that you
   want to turn off, not in the sense of being static but in the
   sense of not being there at all.  You should erase the popup
   on the screen with PopErase, and then call this procedure to
   set the handle to nil (instead of PopDraw) so it won't get
   redrawn.
 ****************************************************************/
void UnRegisterPopUpDialog(long  dialogID, DialogPtr theDialog)
{
 #ifndef IBM
   short                theType;
   Handle               itemHandle;
   Rect                 itemBox;
   short                i;
 #endif

 #ifdef IBM
   return;
 #else
   for ( i = 0 ; i < sa_numPopUps ; i++ )
      if (sa_popTable[i].dialogID == dialogID) {
         GetDialogItem(theDialog, sa_popTable[i].popupItemNum, &theType, &itemHandle, &itemBox);
         SetDialogItem(theDialog, sa_popTable[i].popupItemNum, userItem, nil, &itemBox);
      }
 #endif
}

/****************************************************************
   GetPopSelection() returns the lastItemSelected for the given
   popup.
 ****************************************************************/
short GetPopSelection(DialogPtr theDialog, short item)
{
   short          i;

   for ( i = 0 ; i < sa_numPopUps ; i++ )
      if ( (sa_popTable[i].dialogPtr == theDialog) && (sa_popTable[i].popupItemNum == item) )
         return(sa_popTable[i].lastItemSelected);

   return 0;      // invalid DialogPtr or item
}

/****************************************************************
   SetPopSelection() sets the lastItemSelected for the given
   popup.
 ****************************************************************/
void SetPopSelection(DialogPtr theDialog, short item, short selection)
{
      short          i;
   
   #ifndef IBM   
      short          j, c;
   #endif
   
      for ( i = 0 ; i < sa_numPopUps ; i++ )
         if ( (sa_popTable[i].dialogPtr == theDialog) && (sa_popTable[i].popupItemNum == item) )
            break;

      if (i < sa_numPopUps)
      {
      #ifndef IBM
	  
         MenuHandle  theMenu;

         theMenu = GetMenuHandle(sa_popTable[i].menuID);
         for (j = 1, c = CountMenuItems(theMenu); j <= c ; j++)
            SetItemMark(theMenu, j, noMark);
         SetItemMark(theMenu, selection, kCheckMark);
		 
      #endif //IBM
	  
         sa_popTable[i].lastItemSelected = selection;
     
     #ifdef IBM //{/////////
       
		 // This forces an update of the 'combo box' so that it displays the 
	     //   currently selected menu item.
		 PopDraw( theDialog, item );
       
     #endif // IBM //}/////////
	 
      }

}

/****************************************************************
   HideDPopup() hides the given popup item and its label item.
 ****************************************************************/
void HideDPopup(DialogPtr theDialog, short item)
{
      short          i;

      for ( i = 0 ; i < sa_numPopUps ; i++ )
         if ( (sa_popTable[i].dialogPtr == theDialog) && (sa_popTable[i].popupItemNum == item) )
            break;

      if (i < sa_numPopUps) {
         HideDialogItem(theDialog, sa_popTable[i].popupItemNum);
         if (sa_popTable[i].titleItemNum)
            HideDialogItem(theDialog, sa_popTable[i].titleItemNum);
      }
}

/****************************************************************
   ShowDPopup() shows the given popup item and its label item.
 ****************************************************************/
void ShowDPopup(DialogPtr theDialog, short item)
{
      short          i;

      for ( i = 0 ; i < sa_numPopUps ; i++ )
         if ( (sa_popTable[i].dialogPtr == theDialog) && (sa_popTable[i].popupItemNum == item) )
            break;

      if (i < sa_numPopUps) {
         ShowDialogItem(theDialog, sa_popTable[i].popupItemNum);
         if (sa_popTable[i].titleItemNum)
            ShowDialogItem(theDialog, sa_popTable[i].titleItemNum);
      }
}

#ifndef IBM

/******************************************************
 DrawArrow() - This subroutine draws an up or down arrow
 in the given rect.
 *****************************************************/
void DrawArrow(RECTPTR rPtr, short upOrDown, Boolean disabled)
{
   short    top;
   short    mid;
   short    i;

   PenNormal();
   //if (disabled) PenPat((ConstPatternParam)&qd.gray);
   //if (disabled) PenPat((ConstPatternParam)&GRAY_BRUSH);
   if (disabled) PenPatQDGlobalsGray();

   top = ( rPtr->bottom - rPtr->top ) / 2 + rPtr->top - 3;
   mid = ( rPtr->right - rPtr->left ) / 2 + rPtr->left;

   for ( i = 0; i < 6; i++ ) {
      MoveTo( mid - (6 - i), top + ( (upOrDown == kUp) ? (6 - i) : i ) );
      LineTo( mid + (5 - i), top + ( (upOrDown == kUp) ? (6 - i) : i ) );
   }
   
   PenNormal();
}

/******************************************************
 DefaultMenuColors() - This subroutine finds the current
 default menu foreground and background colors.
 *****************************************************/
void DefaultMenuColors(RGBCOLORPTR DefaultFore, RGBCOLORPTR DefaultBack)
{
   MCEntry        *mce;

   /* the default menu colors are stored in the current mctb resource,
      within the menu bar's entry (the one with ID = 0, item = 0)
      [IM V p. 234] */

   mce = GetMCEntry(0, 0);
   if (mce == nil) {
      DefaultFore->red = 0;
      DefaultFore->blue = 0;
      DefaultFore->green = 0;
      DefaultBack->red = 0xFFFF;
      DefaultBack->blue = 0xFFFF;
      DefaultBack->green = 0xFFFF;
   }
   else {
      *DefaultFore = mce->mctRGB3;
      *DefaultBack = mce->mctRGB2;
   }
}

/******************************************************
 DrawCheck() - This subroutine draws a check mark
 at the chosen item.
 *****************************************************/
void DrawCheck(RECTPTR r)
{
   PenSize(1, 2);

   MoveTo(r->left + 2, (r->top + r->bottom)/2);
   Line(2, 2);
   Line(6, -6);

   PenNormal();
}

/****************************************************************
 DefineMenu() reads the menu's 'MDAT' resource into memory and
 fills in the fields.  It creates the resource if it doesn't
 exist.
 ****************************************************************/
void DefineMenu(int menuID,
          		short cellHeight,
				short menuWidth,
				Boolean bFixedWidth,
				short numColumns,
//JLM				ProcPtr drawProc,
//JLM				ProcPtr invertProc,
//JLM				ProcPtr disableProc)
				SAMenuActionProcPtr drawProc,
				SAMenuActionProcPtr invertProc,
				SAMenuActionProcPtr disableProc)
{
   MenuDataRec **menuDataHandle;
   char message[100];

   menuDataHandle = (MenuDataRec **)GetResource('MDAT', menuID);
   if (menuDataHandle == nil) {
      // try to create the resource
      menuDataHandle = (MenuDataRec **)_NewHandle(sizeof (MenuDataRec));
      if (menuDataHandle == nil) {
         sprintf(message, "Could not create info structure for menu ID %d.", menuID);
         // debugstr(message);
		 SysBeep(1);
         return;
      }
      AddResource((Handle)menuDataHandle, 'MDAT', menuID, "\p");
      if (ResError() != 0) {
         sprintf(message, "Could not add 'MDAT' resource for menu ID %d.", menuID);
         // debugstr(message);
		 SysBeep(1);
         return;
      }
   }

   (**menuDataHandle).cellHeight = cellHeight;
   (**menuDataHandle).menuWidth = menuWidth;
   (**menuDataHandle).bFixedWidth = bFixedWidth;
   (**menuDataHandle).numColumns = numColumns;
   (**menuDataHandle).drawProc = drawProc;
   (**menuDataHandle).invertProc = invertProc;
   (**menuDataHandle).disableProc = disableProc;
}
#endif

int LoadPopupMenu (DialogPtr theDialog, int menuID, short ctlID)
{
#ifdef IBM
   HWND  hCtl;
   char  buffer[256];
   // int i=1;
   WORD  resNum;

   hCtl = GetDlgItem (theDialog, ctlID);
   resNum = menuID;
   resNum *= 100;
   resNum++;

   while (TRUE)
   {
      buffer[0] = 0;
      LoadString(hInst, resNum, buffer, 255);
      // getindstring (buffer, menuID, i);
      if (!strlen(buffer))
         return (0);  //Sucessfully found end of list
      else
      {
         SendMessage (hCtl, CB_ADDSTRING, 0, (LPARAM)(LPCSTR)buffer);
         resNum++;
      }
   }

#else
	#pragma unused (theDialog)
	#pragma unused (menuID)
	#pragma unused (ctlID)
	
	// we assume the menu is already loaded and inserted
	
	/*
	MenuHandle menuHdl;

	menuHdl = GetMenu( menuID );
	if ( menuHdl != nil ) {
	  InsertMenu( menuHdl, -1 );
	  return (0);
	}
	*/
	
	return 0;
#endif
	
	return 1;
}

///****************************************************************
//   CurrentPopTable() returns a pointer to the current pop up table array.
// ****************************************************************/
//POPINFORECPTR CurrentPopTable( void )
//{
//   return ( sa_popTable );
//}

PopTableInfo SavePopTable( void )
{	//JLM
	PopTableInfo pti;
	pti.popTable = sa_popTable;
	pti.numPopUps = sa_numPopUps;
	return pti;
}

void RestorePopTableInfo(PopTableInfo pti)
{	//JLM
	sa_popTable = pti.popTable;
	sa_numPopUps = pti.numPopUps;
}

