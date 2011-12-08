/*
     File:       IAExtractor.r
 
     Contains:   Interfaces to Find by Content Plugins that scan files
 
     Version:    Technology: Mac OS 8.6
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1999-2001 by Apple Computer, Inc., all rights reserved.
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __IAEXTRACTOR_R__
#define __IAEXTRACTOR_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#if CALL_NOT_IN_CARBON

#define kIACurrentMIMEMappingVersion    1

type 'mimp' {
    longint;        /* resource version */
    longint;        /* file type        */
    longint;        /* file creator     */
    pstring;        /* file extension   */
    pstring;        /* MIME type    */
    pstring;        /* description  */
};

#endif  /* CALL_NOT_IN_CARBON */


#endif /* __IAEXTRACTOR_R__ */

