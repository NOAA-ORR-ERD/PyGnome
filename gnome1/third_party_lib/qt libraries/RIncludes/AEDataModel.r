/*
     File:       AEDataModel.r
 
     Contains:   AppleEvent Data Model Interfaces.
 
     Version:    Technology: System 7.5
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1996-2001 by Apple Computer, Inc., all rights reserved
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __AEDATAMODEL_R__
#define __AEDATAMODEL_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

/* Apple event descriptor types */
#define typeBoolean 					'bool'
#define typeChar 						'TEXT'

/* Preferred numeric Apple event descriptor types */
#define typeSInt16 						'shor'
#define typeSInt32 						'long'
#define typeUInt32 						'magn'
#define typeSInt64 						'comp'
#define typeIEEE32BitFloatingPoint 		'sing'
#define typeIEEE64BitFloatingPoint 		'doub'
#define type128BitFloatingPoint 		'ldbl'
#define typeDecimalStruct 				'decm'

/* Non-preferred Apple event descriptor types */
#define typeSMInt 						'shor'
#define typeShortInteger 				'shor'
#define typeInteger 					'long'
#define typeLongInteger 				'long'
#define typeMagnitude 					'magn'
#define typeComp 						'comp'
#define typeSMFloat 					'sing'
#define typeShortFloat 					'sing'
#define typeFloat 						'doub'
#define typeLongFloat 					'doub'
#define typeExtended 					'exte'

/* More Apple event descriptor types */
#define typeAEList 						'list'
#define typeAERecord 					'reco'
#define typeAppleEvent 					'aevt'
#define typeEventRecord 				'evrc'
#define typeTrue 						'true'
#define typeFalse 						'fals'
#define typeAlias 						'alis'
#define typeEnumerated 					'enum'
#define typeType 						'type'
#define typeAppParameters 				'appa'
#define typeProperty 					'prop'
#define typeFSS 						'fss '
#define typeKeyword 					'keyw'
#define typeSectionH 					'sect'
#define typeWildCard 					'****'
#define typeApplSignature 				'sign'
#define typeQDRectangle 				'qdrt'
#define typeFixed 						'fixd'
#define typeSessionID 					'ssid'
#define typeTargetID 					'targ'
#define typeProcessSerialNumber 		'psn '
#define typeKernelProcessID 			'kpid'
#define typeDispatcherID 				'dspt'
#define typeNull 						'null'				/*  null or nonexistent data  */

/* Keywords for Apple event attributes */
#define keyTransactionIDAttr 			'tran'
#define keyReturnIDAttr 				'rtid'
#define keyEventClassAttr 				'evcl'
#define keyEventIDAttr 					'evid'
#define keyAddressAttr 					'addr'
#define keyOptionalKeywordAttr 			'optk'
#define keyTimeoutAttr 					'timo'
#define keyInteractLevelAttr 			'inte'				/*  this attribute is read only - will be set in AESend  */
#define keyEventSourceAttr 				'esrc'				/*  this attribute is read only  */
#define keyMissedKeywordAttr 			'miss'				/*  this attribute is read only  */
#define keyOriginalAddressAttr 			'from'				/*  new in 1.0.1  */


#endif /* __AEDATAMODEL_R__ */

