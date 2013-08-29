/*
     File:       IsochronousDataHandler.r
 
     Contains:   The defines the client API to an Isochronous Data Handler, which is
 
     Version:    Technology: xxx put version here xxx
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1997-2001 by Apple Computer, Inc., all rights reserved.
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/
//
// Check for Prior Inclusion of IsochronousDataHandler.h
//  If this header is trying to be included via a C path, make it act
//  as a NOP.  This will allow both Rez & C files to get to use the
//  contants for the component type, subtype, and interface version.
#ifndef __ISOCHRONOUSDATAHANDLER__

#ifndef __ISOCHRONOUSDATAHANDLER_R__
#define __ISOCHRONOUSDATAHANDLER_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#define kIDHComponentType 				'ihlr'				/*  Component type */
#define kIDHSubtypeDV 					'dv  '				/*  Subtype for DV (over FireWire) */
#define kIDHSubtypeFireWireConference 	'fwc '				/*  Subtype for FW Conference */

#define kIDHInterfaceVersion1 			0x0001				/*  Initial relase (Summer '99) */
#endif /* ifndef __ISOCHRONOUSDATAHANDLER__ */

#endif /* __ISOCHRONOUSDATAHANDLER_R__ */

