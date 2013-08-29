/*
     File:       PCCardEnablerPlugin.r
 
     Contains:   Interfacer for PCCard Manager 3.0
 
     Version:    Technology: Mac OS 8.5
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1997-2001 by Apple Computer, Inc. and SystemSoft Corporation.  All rights reserved.
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __PCCARDENABLERPLUGIN_R__
#define __PCCARDENABLERPLUGIN_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#ifndef __CARDSERVICES__
type 'pccd' {
   longint;                                     /* MUST BE ZERO */
    integer dontShowIcon = -1, noCustomIcon = 0;    /* customIconID             */
    integer noCustomStrings = 0;                    /* customStringsID          */
    unsigned integer;                               /*  customTypeIndex         */
    unsigned integer;                               /*  customHelpIndex         */
    literal longint noCustomAction = 0;             /* customAction             */
    longint;                                        /*  customActionParam1      */
    longint;                                        /*  customActionParam2      */
};
#endif  /* !defined(__CARDSERVICES__) */


#endif /* __PCCARDENABLERPLUGIN_R__ */

