/*
     File:       Movies.r
 
     Contains:   QuickTime Interfaces.
 
     Version:    Technology: QuickTime 6.0
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1990-2001 by Apple Computer, Inc., all rights reserved
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __MOVIES_R__
#define __MOVIES_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#if CALL_NOT_IN_CARBON

type 'qter' {
 longint = $$Countof(ErrorSpec);
    wide array ErrorSpec {
 longint;                            // error code used to find this error
  longint                             // error type
      kQuickTimeErrorNotice = 1,
     kQuickTimeErrorWarning = 2,
        kQuickTimeErrorError = 3;
  // In the following strings, ^FILENAME, ^APPNAME, ^0, ^1, etc will be replaced as appropriate.
 pstring;                            // main error string
   pstring;                            // explaination error string
   pstring;                            // technical string (not displayed to user except in debug cases)
  align long;
    };
};

#endif  /* CALL_NOT_IN_CARBON */


#endif /* __MOVIES_R__ */

