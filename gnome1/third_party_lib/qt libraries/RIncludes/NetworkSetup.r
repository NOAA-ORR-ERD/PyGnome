/*
     File:       NetworkSetup.r
 
     Contains:   Network Setup Interfaces
 
     Version:    Technology: 1.1.0
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1998-2001 by Apple Computer, Inc., all rights reserved
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __NETWORKSETUP_R__
#define __NETWORKSETUP_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#if CALL_NOT_IN_CARBON
#define kCfgErrDatabaseChanged 			(-3290)				/*  database has changed since last call - close and reopen DB */
#define kCfgErrAreaNotFound 			(-3291)				/*  Area doesn't exist */
#define kCfgErrAreaAlreadyExists 		(-3292)				/*  Area already exists */
#define kCfgErrAreaNotOpen 				(-3293)				/*  Area needs to open first */
#define kCfgErrConfigLocked 			(-3294)				/*  Access conflict - retry later */
#define kCfgErrEntityNotFound 			(-3295)				/*  An entity with this name doesn't exist */
#define kCfgErrEntityAlreadyExists 		(-3296)				/*  An entity with this name already exists */
#define kCfgErrPrefsTypeNotFound 		(-3297)				/*  An record with this PrefsType doesn't exist */
#define kCfgErrDataTruncated 			(-3298)				/*  Data truncated when read buffer too small */
#define kCfgErrFileCorrupted 			(-3299)				/*  The database format appears to be corrupted. */

#define kCfgTypefree 					'free'
#define kCfgClassAnyEntity 				'****'
#define kCfgClassUnknownEntity 			'????'
#define kCfgTypeAnyEntity 				'****'
#define kCfgTypeUnknownEntity 			'????'

#define kCfgIgnoreArea 					1
#define kCfgDontIgnoreArea 				0

#define kOTCfgIgnoreArea 				1
#define kOTCfgDontIgnoreArea 			0

#define kOTCfgTypeStruct 				'stru'
#define kOTCfgTypeElement 				'elem'
#define kOTCfgTypeVector 				'vect'

#define kOTCfgClassNetworkConnection 	'otnc'
#define kOTCfgClassGlobalSettings 		'otgl'
#define kOTCfgClassServer 				'otsv'
#define kOTCfgTypeGeneric 				'otan'
#define kOTCfgTypeAppleTalk 			'atlk'
#define kOTCfgTypeTCPv4 				'tcp4'
#define kOTCfgTypeTCPv6 				'tcp6'
#define kOTCfgTypeRemote 				'ara '
#define kOTCfgTypeDial 					'dial'
#define kOTCfgTypeModem 				'modm'
#define kOTCfgTypeInfrared 				'infr'
#define kOTCfgClassSetOfSettings 		'otsc'
#define kOTCfgTypeSetOfSettings 		'otst'
#define kOTCfgTypeDNS 					'dns '

#define kOTCfgIndexSetsActive 			0
#define kOTCfgIndexSetsEdit 			1
#define kOTCfgIndexSetsLimit 			2					/*     last value, no comma */

															/*     connection  */
#define kOTCfgTypeConfigName 			'cnam'
#define kOTCfgTypeConfigSelected 		'ccfg'				/*     transport options   */
#define kOTCfgTypeUserLevel 			'ulvl'
#define kOTCfgTypeWindowPosition 		'wpos'

															/*     connection  */
#define kOTCfgTypeAppleTalkPrefs 		'atpf'
#define kOTCfgTypeAppleTalkVersion 		'cvrs'
#define kOTCfgTypeAppleTalkLocks 		'lcks'
#define kOTCfgTypeAppleTalkPort 		'port'
#define kOTCfgTypeAppleTalkProtocol 	'prot'
#define kOTCfgTypeAppleTalkPassword 	'pwrd'
#define kOTCfgTypeAppleTalkPortFamily 	'ptfm'				/*     transport options   */

#define kOTCfgIndexAppleTalkAARP 		0
#define kOTCfgIndexAppleTalkDDP 		1
#define kOTCfgIndexAppleTalkNBP 		2
#define kOTCfgIndexAppleTalkZIP 		3
#define kOTCfgIndexAppleTalkATP 		4
#define kOTCfgIndexAppleTalkADSP 		5
#define kOTCfgIndexAppleTalkPAP 		6
#define kOTCfgIndexAppleTalkASP 		7
#define kOTCfgIndexAppleTalkLast 		7

#define kOTCfgTypeInfraredPrefs 		'atpf'
#define kOTCfgTypeInfraredGlobal 		'irgo'

															/*     connection  */
#define kOTCfgTypeTCPalis 				'alis'
#define kOTCfgTypeTCPcvrs 				'cvrs'
#define kOTCfgTypeTCPdcid 				'dcid'
#define kOTCfgTypeTCPdclt 				'dclt'
#define kOTCfgTypeTCPdtyp 				'dtyp'
#define kOTCfgTypeTCPidns 				'idns'
#define kOTCfgTypeTCPihst 				'ihst'
#define kOTCfgTypeTCPiitf 				'iitf'
#define kOTCfgTypeTCPara 				'ipcp'
#define kOTCfgTypeTCPirte 				'irte'
#define kOTCfgTypeTCPisdm 				'isdm'
#define kOTCfgTypeTCPstng 				'stng'
#define kOTCfgTypeTCPunld 				'unld'
#define kOTCfgTypeTCPVersion 			'cvrs'				/*     Version  */
#define kOTCfgTypeTCPDevType 			'dvty'
#define kOTCfgTypeTCPPrefs 				'iitf'
#define kOTCfgTypeTCPServersList 		'idns'
#define kOTCfgTypeTCPSearchList 		'ihst'
#define kOTCfgTypeTCPRoutersList 		'irte'
#define kOTCfgTypeTCPDomainsList 		'isdm'
#define kOTCfgTypeTCPPort 				'port'				/*     Ports  */
#define kOTCfgTypeTCPProtocol 			'prot'
#define kOTCfgTypeTCPPassword 			'pwrd'				/*     Password  */
#define kOTCfgTypeTCPLocks 				'stng'				/*     locks  */
#define kOTCfgTypeTCPUnloadType 		'unld'				/*     transport options   */

															/*     connection  */
#define kOTCfgTypeDNSidns 				'idns'
#define kOTCfgTypeDNSisdm 				'isdm'
#define kOTCfgTypeDNSihst 				'ihst'
#define kOTCfgTypeDNSstng 				'stng'
#define kOTCfgTypeDNSPassword 			'pwrd'				/*     transport options   */

															/*     connection  */
#define kOTCfgTypeModemModem 			'ccl '				/*     Type for Modem configuration resource */
#define kOTCfgTypeModemLocks 			'lkmd'				/*     Types for lock resources */
#define kOTCfgTypeModemAdminPswd 		'mdpw'				/*     Password */
															/*     transport options   */
#define kOTCfgTypeModemApp 				'mapt'

															/*     connection  */
#define kOTCfgTypeRemoteARAP 			'arap'
#define kOTCfgTypeRemoteAddress 		'cadr'
#define kOTCfgTypeRemoteChat 			'ccha'
#define kOTCfgTypeRemoteDialing 		'cdia'
#define kOTCfgTypeRemoteExtAddress 		'cead'
#define kOTCfgTypeRemoteClientLocks 	'clks'
#define kOTCfgTypeRemoteClientMisc 		'cmsc'
#define kOTCfgTypeRemoteConnect 		'conn'
#define kOTCfgTypeRemoteUser 			'cusr'
#define kOTCfgTypeRemoteDialAssist 		'dass'
#define kOTCfgTypeRemoteIPCP 			'ipcp'
#define kOTCfgTypeRemoteLCP 			'lcp '				/*  trailing space is important!  */
#define kOTCfgTypeRemoteLogOptions 		'logo'
#define kOTCfgTypeRemotePassword 		'pass'
#define kOTCfgTypeRemotePort 			'port'
#define kOTCfgTypeRemoteServerLocks 	'slks'
#define kOTCfgTypeRemoteServer 			'srvr'
#define kOTCfgTypeRemoteUserMode 		'usmd'
#define kOTCfgTypeRemoteX25 			'x25 '				/*  trailing space is important!  */
															/*     transport options   */
#define kOTCfgTypeRemoteApp 			'capt'

#define kOTCfgRemoteMaxAddressSize 		256
#define kOTCfgRemoteMaxPasswordLength 	255
#define kOTCfgRemoteMaxPasswordSize 	256
#define kOTCfgRemoteMaxUserNameLength 	255
#define kOTCfgRemoteMaxUserNameSize 	256
#define kOTCfgRemoteMaxAddressLength 	255					/*     kOTCfgRemoteMaxAddressSize           = (255 + 1), */
#define kOTCfgRemoteMaxServerNameLength  32
#define kOTCfgRemoteMaxServerNameSize 	33
#define kOTCfgRemoteMaxMessageLength 	255
#define kOTCfgRemoteMaxMessageSize 		256
#define kOTCfgRemoteMaxX25ClosedUserGroupLength  4
#define kOTCfgRemoteInfiniteSeconds 	0xFFFFFFFF
#define kOTCfgRemoteMinReminderMinutes 	1
#define kOTCfgRemoteChatScriptFileCreator  'ttxt'
#define kOTCfgRemoteChatScriptFileType 	'TEXT'
#define kOTCfgRemoteMaxChatScriptLength  0x8000

#define kOTCfgRemoteStatusIdle 			1
#define kOTCfgRemoteStatusConnecting 	2
#define kOTCfgRemoteStatusConnected 	3
#define kOTCfgRemoteStatusDisconnecting  4

#endif  /* CALL_NOT_IN_CARBON */


#endif /* __NETWORKSETUP_R__ */

