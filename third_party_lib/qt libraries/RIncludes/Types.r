/*
     File:       Types.r
 
     Contains:   Basic Macintosh data types.
 
     Version:    Technology: System 7.5
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1985-2001 by Apple Computer, Inc., all rights reserved.
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __TYPES_R__
#define __TYPES_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#ifndef __MACTYPES_R__
#include "MacTypes.r"        /* Basic types moved to MacTypes.r */
#endif
#ifndef __CONTROLS_R__
#include "Controls.r"        /* Types.r used to define 'CNTL' and 'cctb' */
#endif
#ifndef __CONTROLDEFINITIONS_R__
#include "ControlDefinitions.r"
#endif
#ifndef __MACWINDOWS_R__
#include "MacWindows.r"      /* Types.r used to define 'WIND' and 'wctb' */
#endif
#ifndef __DIALOGS_R__
#include "Dialogs.r"         /* Types.r used to define 'DLOG', 'ALRT', 'DITL', 'actb', and 'dctb' */
#endif
#ifndef __MENUS_R__
#include "Menus.r"           /* Types.r used to define 'MENU', 'MBAR', and 'mctb' */
#endif
#ifndef __ICONS_R__
#include "Icons.r"           /* Types.r used to define 'ICON', 'ICN#', 'SICN', 'ics#', 'icm#', 'icm8', 'icm4', 'icl8', etc. */
#endif
#ifndef __FINDER_R__
#include "Finder.r"          /* Types.r used to define 'BNDL', 'FREF', 'open', and 'kind' */
#endif
#ifndef __QUICKDRAW_R__
#include "Quickdraw.r"       /* Types.r used to define 'CURS', 'PAT', 'ppat', 'PICT', 'acur', 'clut', 'crsr', and 'PAT#' */
#endif
#ifndef __PROCESSES_R__
#include "Processes.r"       /* Types.r used to define 'SIZE' */
#endif
#ifndef __APPLEEVENTS_R__
#include "AppleEvents.r"     /* AppleEvents.r used to define 'aedt' */
#endif


#endif /* __TYPES_R__ */

