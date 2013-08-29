/*
     File:       AVComponents.r
 
     Contains:   Standard includes for standard AV panels
 
     Version:    Technology: System 7.5
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1989-2001 by Apple Computer, Inc., all rights reserved
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __AVCOMPONENTS_R__
#define __AVCOMPONENTS_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif


/*
    The subtypes listed here are for example only.  The display manager will find _all_ panels
    with the appropriate types.  These panels return class information that is used to devide them
    up into groups to be displayed in the AV Windows (class means "geometry" or "color" or other groupings
    like that.
*/
#define kAVPanelType 					'avpc'				/*  Panel subtypes         */
#define kBrightnessPanelSubType 		'brit'
#define kContrastPanelSubType 			'cont'
#define kBitDepthPanelSubType 			'bitd'
#define kAVEngineType 					'avec'				/*  Engine subtypes           */
#define kBrightnessEngineSubType 		'brit'
#define kContrastEngineSubType 			'cont'				/*     kBitDepthEngineSubType     = 'bitd',       // Not used               */
#define kAVPortType 					'avdp'				/*  subtypes are defined in each port's public .h file  */
#define kAVUtilityType 					'avuc'
#define kAVBackChannelSubType 			'avbc'
#define kAVCommunicationType 			'avcm'
#define kAVDialogType 					'avdg'

/* PortComponent subtypes are up to the port and display manager does not use the subtype
    to find port components.  Instead, display manager uses an internal cache to search for portcompoennts.
    It turns out to be useful to have a unique subtype so that engines can see if they should apply themselves to
    a particular port component.
  
   PortKinds are the "class" of port.  When a port is registered with display manager (creating a display ID), the
    caller of DMNewDisplayIDByPortComponent passes a portKind.  Ports of this type are returned by
    DMNewDevicePortList.
  
   PortKinds are NOT subtypes of components
   PortKinds ARE used to register and find port components with Display Manager.  Here are the basic port kinds:
  
   Video displays are distinct from video out because there are some video out ports that are not actaully displays.
    if EZAV is looking to configure displays, it needs to look for kAVVideoDisplayPortKind not kAVVideoOutPortKind.
*/
#define kAVVideoDisplayPortKind 		'pkdo'				/*  Video Display (CRT or panel display)           */
#define kAVVideoOutPortKind 			'pkvo'				/*  Video out port (camera output).                 */
#define kAVVideoInPortKind 				'pkvi'				/*  Video in port (camera input)                */
#define kAVSoundOutPortKind 			'pkso'				/*  Sound out port (speaker or speaker jack)       */
#define kAVSoundInPortKind 				'pksi'				/*  Sound in port (microphone or microphone jack)   */
#define kAVDeviceType 					'avdc'				/*  Device Component subtypes are up to the manufacturor since each device may contain multiple function types (eg telecaster)  */
#define kAVDisplayDeviceKind 			'dkvo'				/*  Display device */
															/*  Device Component subtypes are up to the manufacturor since each device may contain multiple function types (eg telecaster) */
#define kAVCategoryType 				'avcc'
#define kAVSoundInSubType 				'avao'
#define kAVSoundOutSubType 				'avai'
#define kAVVideoInSubType 				'vdin'
#define kAVVideoOutSubType 				'vdou'
#define kAVInvalidType 					'badt'				/*  Some calls return a component type, in case of errors, these types are set to kAVInvalidComponentType  */

/*
   Interface Signatures are used to identify what kind of component
   calls can be made for a given component. Today this applies only
   to ports, but could be applied to other components as well.
*/
#define kAVGenericInterfaceSignature 	'dmgr'
#define kAVAppleVisionInterfaceSignature  'avav'

/* =============================                    */
/* Panel Class Constants                            */
/* =============================                    */
#define kAVPanelClassDisplayDefault 	'cdsp'
#define kAVPanelClassColor 				'cclr'
#define kAVPanelClassGeometry 			'cgeo'
#define kAVPanelClassSound 				'csnd'
#define kAVPanelClassPreferences 		'cprf'
#define kAVPanelClassLCD 				'clcd'
#define kAVPanelClassMonitorSound 		'cres'
#define kAVPanelClassAlert 				'calr'
#define kAVPanelClassExtras 			'cext'
#define kAVPanelClassRearrange 			'crea'


/* =============================                    */
/* AV Notification Types                            */
/* =============================                    */
/*
   This notification will be sent whenever a
   device has been reset, for whatever reason.
*/
#define kAVNotifyDeviceReset 			'rset'

/* =============================                    */
/* Component interface revision levels and history  */
/* =============================                    */
#define kAVPanelComponentInterfaceRevOne  1
#define kAVPanelComponentInterfaceRevTwo  2
#define kAVEngineComponentInterfaceRevOne  1
#define kAVPortComponentInterfaceRevOne  1
#define kAVDeviceComponentInterfaceRevOne  1
#define kAVUtilityComponentInterfaceRevOne  1


/* =============================                    */
/* Adornment Constants                              */
/* =============================                    */
#define kAVPanelAdornmentNoBorder 		0
#define kAVPanelAdornmentStandardBorder  1

#define kAVPanelAdornmentNoName 		0
#define kAVPanelAdornmentStandardName 	1


/* =============================                    */
/* Selector Ranges                                  */
/* =============================                    */
#define kBaseAVComponentSelector 		256					/*  First apple-defined selector for AV components  */
#define kAppleAVComponentSelector 		512					/*  First apple-defined type-specific selector for AV components  */


/* =============================                */
/* Panel Standard component selectors           */
/* =============================                */
#define kAVPanelFakeRegisterSelect 		(-5)				/*  -5  */
#define kAVPanelSetCustomDataSelect 	0
#define kAVPanelGetDitlSelect 			1
#define kAVPanelGetTitleSelect 			2
#define kAVPanelInstallSelect 			3
#define kAVPanelEventSelect 			4
#define kAVPanelItemSelect 				5
#define kAVPanelRemoveSelect 			6
#define kAVPanelValidateInputSelect 	7
#define kAVPanelGetSettingsIdentifiersSelect  8
#define kAVPanelGetSettingsSelect 		9
#define kAVPanelSetSettingsSelect 		10
#define kAVPanelSelectorGetFidelitySelect  256
#define kAVPanelSelectorTargetDeviceSelect  257
#define kAVPanelSelectorGetPanelClassSelect  258
#define kAVPanelSelectorGetPanelAdornmentSelect  259
#define kAVPanelSelectorGetBalloonHelpStringSelect  260
#define kAVPanelSelectorAppleGuideRequestSelect  261


/* =============================                */
/* Engine Standard component selectors          */
/* =============================                */
#define kAVEngineGetEngineFidelitySelect  256
#define kAVEngineTargetDeviceSelect 	257


/* =============================                    */
/* Video Port Specific calls                        */
/* =============================                    */
#define kAVPortCheckTimingModeSelect 	0
#define kAVPortReserved1Select 			1					/*  Reserved */
#define kAVPortReserved2Select 			2					/*  Reserved */
#define kAVPortGetDisplayTimingInfoSelect  512
#define kAVPortGetDisplayProfileCountSelect  513
#define kAVPortGetIndexedDisplayProfileSelect  514
#define kAVPortGetDisplayGestaltSelect 	515


/* =============================                    */
/* AV Port Specific calls                           */
/* =============================                    */
#define kAVPortGetAVDeviceFidelitySelect  256				/*  Port Standard Component selectors  */
#define kAVPortGetWiggleSelect 			257
#define kAVPortSetWiggleSelect 			258
#define kAVPortGetNameSelect 			259
#define kAVPortGetGraphicInfoSelect 	260
#define kAVPortSetActiveSelect 			261
#define kAVPortGetActiveSelect 			262
#define kAVPortUnsed1Select 			263					/*  Selector removed as part of API change.  We don't want to mess up the following selectors, so we put in this spacer (ie kPadSelector).  */
#define kAVPortGetAVIDSelect 			264
#define kAVPortSetAVIDSelect 			265
#define kAVPortSetDeviceAVIDSelect 		266					/*  For registrar to set device (instead of hitting global directly) -- should only be called once  */
#define kAVPortGetDeviceAVIDSelect 		267					/*  Called by display mgr for generic ports  */
#define kAVPortGetPowerStateSelect 		268
#define kAVPortSetPowerStateSelect 		269
#define kAVPortGetMakeAndModelSelect 	270					/*  Get Make and model information */
#define kAVPortGetInterfaceSignatureSelect  271				/*  To determine what VideoPort-specific calls can be made */
#define kAVPortReserved3Select 			272					/*  Reserved */
#define kAVPortGetManufactureInfoSelect  273				/*  Get more Make and model information   */




/* =============================                    */
/* Device Component Standard Component selectors    */
/* =============================                    */
#define kAVDeviceGetNameSelect 			256
#define kAVDeviceGetGraphicInfoSelect 	257
#define kAVDeviceGetPowerStateSelect 	258
#define kAVDeviceSetPowerStateSelect 	259
#define kAVDeviceGetAVIDSelect 			260
#define kAVDeviceSetAVIDSelect 			261

/* =============================                    */
/* AV Back-Channel Selectors                        */
/* =============================                    */
#define kAVBackChannelReservedSelector 	1
#define kAVBackChannelPreModalFilterSelect  2
#define kAVBackChannelModalFilterSelect  3
#define kAVBackChannelAppleGuideLaunchSelect  4








#endif /* __AVCOMPONENTS_R__ */

