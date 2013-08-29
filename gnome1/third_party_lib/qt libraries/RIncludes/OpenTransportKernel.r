/*
     File:       OpenTransportKernel.r
 
     Contains:   Definitions for Open Transport kernel code, such as drivers and protocol modules.
 
     Version:    Technology: 2.5
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1993-2001 by Apple Computer, Inc. and Mentat Inc., all rights reserved.
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __OPENTRANSPORTKERNEL_R__
#define __OPENTRANSPORTKERNEL_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#define OTKERNEL 1
    
   /*
  * Interface ID for STREAMS Modules for ASLM.
  */
#define kOTModuleInterfaceID     kOTModulePrefix "StrmMod"

#if CALL_NOT_IN_CARBON
#define kOTPortScannerPrefix        "ot:pScnr$"

#define kOTPortScannerInterfaceID          kOTKernelPrefix "pScnr"
#define kOTPseudoPortScannerInterfaceID     kOTKernelPrefix "ppScnr"
#define kOTCompatScannerInterfaceID            kOTKernelPrefix "cpScnr,1.0"

#define kOTPortScannerCFMTag              kOTKernelPrefix "pScnr"
#define kOTPseudoPortScannerCFMTag          kOTKernelPrefix "ppScnr"
#define kOTCompatPortScannerCFMTag         kOTKernelPrefix "cpScnr"

#endif  /* CALL_NOT_IN_CARBON */


#endif /* __OPENTRANSPORTKERNEL_R__ */

