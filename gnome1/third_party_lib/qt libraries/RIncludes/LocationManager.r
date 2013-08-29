/*
     File:       LocationManager.r
 
     Contains:   LocationManager (manages groups of settings)
 
     Version:    Technology: Mac OS 8
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1995-2001 by Apple Computer, Inc., all rights reserved.
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __LOCATIONMANAGER_R__
#define __LOCATIONMANAGER_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

/* Location Manager API Support -------------------------------------------------------------------- */
/* A Location Token uniquely identifies a Location on a machine... */

#define kALMNoLocationToken 			(-1)				/*  ALMToken of "off" Location... */
#define kALMLocationNameMaxLen 			31					/*  name (actually imposed by file system)...  */
#define kALMNoLocationIndex 			(-1)				/*  index for the "off" Location (kALMNoLocationToken)...  */

/* Returned from ALMConfirmName... */
#define kALMConfirmRename 				1
#define kALMConfirmReplace 				2

/* ALMConfirmName dialog item numbers for use in callbacks (ALM 2.0)... */

#define kALMDuplicateRenameButton 		1					/*  if Window refcon is kALMDuplicateDialogRefCon...  */
#define kALMDuplicateReplaceButton 		2
#define kALMDuplicateCancelButton 		3
#define kALMDuplicatePromptText 		5

#define kALMRenameRenameButton 			1					/*  if Window refcon is kALMRenameDialogRefCon...  */
#define kALMRenameCancelButton 			2
#define kALMRenameEditText 				3
#define kALMRenamePromptText 			4

/* Refcons of two windows in ALMConfirmName (ALM 2.0)... */

#define kALMDuplicateDialogRefCon 		'dupl'
#define kALMRenameDialogRefCon 			'rnam'

/* Callback routine for Location awareness (mimics AppleEvents) in non-application code... */

/* Notification AppleEvents sent to apps/registered code...  */
#define kAELocationChangedNoticeKey 	'walk'				/*  Current Location changed...  */
#define kAELocationRescanNoticeKey 		'trip'				/*  Location created/renamed/deleted...  */

/* ALMSwitchToLocation masks... */
#define kALMDefaultSwitchFlags 			0x00000000			/*  No special action to take...  */
#define kALMDontShowStatusWindow 		0x00000001			/*  Suppress "switching" window...  */
#define kALMSignalViaAE 				0x00000002			/*  Switch by sending Finder AppleEvent...  */

/* Parameters for Get/Put/Merge Location calls... */

#define kALMAddAllOnSimple 				0					/*  Add all single-instance, non-action modules...  */
#define kALMAddAllOff 					(-1)				/*  Add all modules but turn them off...  */

/* Item numbers for use in Get/Put/Merge Location filters... */

#define kALMLocationSelectButton 		1
#define kALMLocationCancelButton 		2
#define kALMLocationBalloonHelp 		3
#define kALMLocationLocationList 		7
#define kALMLocationLocationNameEdit 	10
#define kALMLocationPromptText 			11

#define kALMLocationSaveButton 			1
/* Location Manager Module API Support ------------------------------------------------------------- */

/* ALMGetScriptInfo stuff... */

#define kALMScriptInfoVersion 			2					/*  Customarily put in resource for localization...  */
/*
   Alternate form of ScriptInfo is easier to localize in resources; it is used extensively in
   samples and internally, so....
*/
#define kALMAltScriptManagerInfoRsrcType  'trip'
#define kALMAltScriptManagerInfoRsrcID 	0

type kALMAltScriptManagerInfoRsrcType {
    integer;    // version = kALMScriptInfoVersion
    integer;    // scriptCode (eg. smRoman)
    integer;    // regionCode (eg. versUS)
    integer;    // langCode (eg. langEnglish)
    integer;    // fontSize
    pstring;    // fontName
};
/* Reboot information used on ALMSetCurrent (input/output parameter)... */
#define kALMNoChange 					0
#define kALMAvailableNow 				1
#define kALMFinderRestart 				2
#define kALMProcesses 					3
#define kALMExtensions 					4
#define kALMWarmBoot 					5
#define kALMColdBoot 					6
#define kALMShutdown 					7

/*
   File types and signatures...
   Note: auto-routing of modules will not be supported for 'thng' files...
*/

#define kALMFileCreator 				'fall'				/*  Creator of Location Manager files...  */
#define kALMComponentModuleFileType 	'thng'				/*  Type of a Component Manager Module file [v1.0]...  */
#define kALMComponentStateModuleFileType  'almn'			/*  Type of a CM 'state' Module file...  */
#define kALMComponentActionModuleFileType  'almb'			/*  Type of a CM 'action' Module file...  */
#define kALMCFMStateModuleFileType 		'almm'				/*  Type of a CFM 'state' Module file...  */
#define kALMCFMActionModuleFileType 	'alma'				/*  Type of a CFM 'action' Module file...  */

/* Component Manager 'thng' info... */

#define kALMComponentRsrcType 			'thng'
#define kALMComponentType 				'walk'

/* CFM Modules require a bit of information (replacing some of the 'thng' resource)... */

#define kALMModuleInfoRsrcType 			'walk'
#define kALMModuleInfoOriginalVersion 	0

type kALMModuleInfoRsrcType {
    switch {
        case Original:
            key longint = kALMModuleInfoOriginalVersion;
            literal longint;        // Subtype
            literal longint;        // Manufacturer
            unsigned hex longint;   // Flags
    };
};
/* These masks apply to the "Flags" field in the 'thng' or 'walk' resource... */

#define kALMMultiplePerLocation 		0x00000001			/*  Module can be added more than once to a Location...  */
#define kALMDescriptionGetsStale 		0x00000002			/*  Descriptions may change though the setting didn't...   */

/* Misc stuff for older implementations ------------------------------------------------------------ */

/* Old error codes for compatibility - new names are in Errors interface... */
#if OLDROUTINENAMES
#define ALMInternalErr 					(-30049)			/*  use kALMInternalErr  */
#define ALMLocationNotFound 			(-30048)			/*  use kALMLocationNotFoundErr  */
#define ALMNoSuchModuleErr 				(-30047)			/*  use kALMNoSuchModuleErr  */
#define ALMModuleCommunicationErr 		(-30046)			/*  use kALMModuleCommunicationErr  */
#define ALMDuplicateModuleErr 			(-30045)			/*  use kALMDuplicateModuleErr  */
#define ALMInstallationErr 				(-30044)			/*  use kALMInstallationErr  */
#define ALMDeferSwitchErr 				(-30043)			/*  use kALMDeferSwitchErr  */

/* Old ALMConfirmName constants... */

#define ALMConfirmRenameConfig 			1
#define ALMConfirmReplaceConfig 		2

/* Old AppleEvents... */

#define kAELocationNotice 				'walk'
#endif  /* OLDROUTINENAMES */


#endif /* __LOCATIONMANAGER_R__ */

