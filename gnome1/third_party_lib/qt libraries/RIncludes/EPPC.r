/*
     File:       EPPC.r
 
     Contains:   High Level Event Manager Interfaces.
 
     Version:    Technology: System 7.5
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1988-2001 by Apple Computer, Inc., all rights reserved
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __EPPC_R__
#define __EPPC_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif


/*----------------------------------------eppc -----------------------------------------*/
type 'eppc' {
    unsigned longint;   /* flags word   */
    unsigned longint;   /* reserved     */
    unsigned integer;   /* scriptCode   */
    pstring[32];
};

#endif /* __EPPC_R__ */

