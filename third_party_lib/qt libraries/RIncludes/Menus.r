/*
     File:       Menus.r
 
     Contains:   Menu Manager Interfaces.
 
     Version:    Technology: Mac OS 9.0
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1985-2001 by Apple Computer, Inc., all rights reserved.
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __MENUS_R__
#define __MENUS_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#define kMenuStdMenuProc 				63
#define kMenuStdMenuBarProc 			63

#define kMenuNoModifiers 				0					/*  Mask for no modifiers */
#define kMenuShiftModifier 				0x01				/*  Mask for shift key modifier */
#define kMenuOptionModifier 			0x02				/*  Mask for option key modifier */
#define kMenuControlModifier 			0x04				/*  Mask for control key modifier */
#define kMenuNoCommandModifier 			0x08				/*  Mask for no command key modifier */

#define kMenuNoIcon 					0					/*  No icon */
#define kMenuIconType 					1					/*  Type for ICON */
#define kMenuShrinkIconType 			2					/*  Type for ICON plotted 16 x 16 */
#define kMenuSmallIconType 				3					/*  Type for SICN */
#define kMenuColorIconType 				4					/*  Type for cicn */
#define kMenuIconSuiteType 				5					/*  Type for Icon Suite */
#define kMenuIconRefType 				6					/*  Type for Icon Ref */

#define kMenuAttrExcludesMarkColumn 	0x01				/*  No space is allocated for the mark character  */
#define kMenuAttrAutoDisable 			0x04				/*  Menu title is automatically disabled when all items are disabled  */

#define kMenuItemAttrSubmenuParentChoosable  0x04			/*  Parent item of a submenu is still selectable by the user  */
#define gestaltContextualMenuAttr 		'cmnu'
#define gestaltContextualMenuUnusedBit 	0
#define gestaltContextualMenuTrapAvailable  1

#define kCMHelpItemNoHelp 				0
#define kCMHelpItemAppleGuide 			1
#define kCMHelpItemOtherHelp 			2

#define kCMNothingSelected 				0
#define kCMMenuItemSelected 			1
#define kCMShowHelpSelected 			3


/*----------------------------MENU ¥ Menu-----------------------------------------------*/
type 'MENU' {
      integer;                                                /* Menu ID              */
     fill word[2];
      integer         textMenuProc = 0;                       /* ID of menu def proc  */
     fill word;
     unsigned hex bitstring[31]
                     allEnabled = 0x7FFFFFFF;                /* Enable flags         */
     boolean         disabled, enabled;                      /* Menu enable          */
     pstring         apple = "\0x14";                        /* Menu Title           */
     wide array {
               pstring;                                        /* Item title           */
             byte            noIcon;                         /* Icon number          */
             char            noKey = "\0x00",                /* Key equivalent or    */
                             hierarchicalMenu = "\0x1B";     /* hierarchical menu    */
             char            noMark = "\0x00",               /* Marking char or id   */
                             check = "\0x12";                /* of hierarchical menu */
             fill bit;
              unsigned bitstring[7]
                              plain;                          /* Style                */
     };
     byte = 0;
};

/*----------------------------MBAR ¥ Menu Bar-------------------------------------------*/
type 'MBAR' {
      integer = $$CountOf(MenuArray);                         /* Number of menus      */
     wide array MenuArray{
              integer;                                        /* Menu resource ID     */
     };
};

/*----------------------------mctb ¥ Menu Color Lookup Table----------------------------*/
type 'mctb' {
      integer = $$CountOf(MCTBArray);                         /* Color table count    */
     wide array MCTBArray {
         integer             mctbLast = -99;                 /* Menu resource ID     */
         integer;                                            /* Menu Item            */
         wide array [4] {
                   unsigned integer;                           /* RGB: red             */
                 unsigned integer;                           /*      green           */
                 unsigned integer;                           /*      blue            */
         };
         fill word;                                          /* Reserved word        */
     };
};


/*-------------------------------xmnu ¥ Extended menu resource---------------------------*/
type 'xmnu'
{
 switch
 {
      case versionZero:
          key integer = 0;    /* version */

         integer = $$Countof(ItemExtensions);
           array ItemExtensions
           {
              switch
             {
                  case skipItem:
                     key integer=0;
                     
                   case dataItem:
                     key integer=1;
                     unsigned longint;                       /* Command ID */
                       unsigned hex byte;                      /* modifiers */
                        fill byte;                              /* icon type placeholder */
                        fill long;                              /* icon handle placeholder */
                      unsigned longint sysScript=-1,          /* text encoding */
                                         currScript=-2;         /*  (use currScript for default)*/
                     unsigned longint;                       /* refCon */
                       unsigned longint;                       /* refCon2 */
                      unsigned integer noHierID=0;            /* hierarchical menu ID */
                     unsigned integer sysFont=0;             /* font ID */
                      integer naturalGlyph=0;                 /* keyboard glyph */
               };
         };
 };
};

#if TARGET_OS_WIN32

/*----------------------------MENA ¥ Menu Accessory Key---------------------------------*/
type 'MENA' {
      integer = $$CountOf(MenuArray);                         /* Number of keys       */
     longint;                                                /* flags                */
     wide array MenuArray{
              char;                                           /* key                  */
     };
};

#endif  /* TARGET_OS_WIN32 */


#endif /* __MENUS_R__ */

