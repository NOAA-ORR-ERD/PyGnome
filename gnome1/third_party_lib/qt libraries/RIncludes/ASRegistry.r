/*
     File:       ASRegistry.r
 
     Contains:   AppleScript Registry constants.
 
     Version:    Technology: AppleScript 1.3
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1991-2001 by Apple Computer, Inc., all rights reserved
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __ASREGISTRY_R__
#define __ASREGISTRY_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#define keyAETarget 					'targ'				/*   0x74617267   */
#define keySubjectAttr 					'subj'				/*   0x7375626a   */
															/*  Magic 'returning' parameter:  */
#define keyASReturning 					'Krtn'				/*   0x4b72746e   */
															/*  AppleScript Specific Codes:  */
#define kASAppleScriptSuite 			'ascr'				/*   0x61736372   */
#define kASScriptEditorSuite 			'ToyS'				/*  AppleScript 1.3 added from private headers  */
#define kASTypeNamesSuite 				'tpnm'				/*   0x74706e6d   */
															/*  dynamic terminologies  */
#define typeAETE 						'aete'				/*   0x61657465   */
#define typeAEUT 						'aeut'				/*   0x61657574   */
#define kGetAETE 						'gdte'				/*   0x67647465   */
#define kGetAEUT 						'gdut'				/*   0x67647574   */
#define kUpdateAEUT 					'udut'				/*   0x75647574   */
#define kUpdateAETE 					'udte'				/*   0x75647465   */
#define kCleanUpAEUT 					'cdut'				/*   0x63647574   */
#define kASComment 						'cmnt'				/*   0x636d6e74   */
#define kASLaunchEvent 					'noop'				/*   0x6e6f6f70   */
#define keyScszResource 				'scsz'				/*   0x7363737A   */
#define typeScszResource 				'scsz'				/*   0x7363737A   */
															/*  subroutine calls  */
#define kASSubroutineEvent 				'psbr'				/*   0x70736272   */
#define keyASSubroutineName 			'snam'				/*   0x736e616d   */
#define kASPrepositionalSubroutine 		'psbr'				/*  AppleScript 1.3 added from private headers  */
#define keyASPositionalArgs 			'parg'				/*  AppleScript 1.3 added from private headers  */

															/*  Miscellaneous AppleScript commands  */
#define kASStartLogEvent 				'log1'				/*  AppleScript 1.3 Script Editor Start Log  */
#define kASStopLogEvent 				'log0'				/*  AppleScript 1.3 Script Editor Stop Log  */
#define kASCommentEvent 				'cmnt'				/*  AppleScript 1.3 magic "comment" event  */

															/*  Binary:  */
#define kASAdd 							'+   '				/*   0x2b202020   */
#define kASSubtract 					'-   '				/*   0x2d202020   */
#define kASMultiply 					'*   '				/*   0x2a202020   */
#define kASDivide 						'/   '				/*   0x2f202020   */
#define kASQuotient 					'div '				/*   0x64697620   */
#define kASRemainder 					'mod '				/*   0x6d6f6420   */
#define kASPower 						'^   '				/*   0x5e202020   */
#define kASEqual 						'=   '
#define kASNotEqual 					'­   '				/*   0xad202020   */
#define kASGreaterThan 					'>   '
#define kASGreaterThanOrEqual 			'>=  '
#define kASLessThan 					'<   '
#define kASLessThanOrEqual 				'<=  '
#define kASComesBefore 					'cbfr'				/*   0x63626672   */
#define kASComesAfter 					'cafr'				/*   0x63616672   */
#define kASConcatenate 					'ccat'				/*   0x63636174   */
#define kASStartsWith 					'bgwt'
#define kASEndsWith 					'ends'
#define kASContains 					'cont'

#define kASAnd 							'AND '
#define kASOr 							'OR  '				/*  Unary:  */
#define kASNot 							'NOT '
#define kASNegate 						'neg '				/*   0x6e656720   */
#define keyASArg 						'arg '				/*   0x61726720   */

															/*  event code for the 'error' statement  */
#define kASErrorEventCode 				'err '				/*   0x65727220   */
#define kOSAErrorArgs 					'erra'				/*   0x65727261   */
#define keyAEErrorObject 				'erob'				/*      Added in AppleScript 1.3 from AppleScript private headers  */
															/*  Properties:  */
#define pLength 						'leng'				/*   0x6c656e67   */
#define pReverse 						'rvse'				/*   0x72767365   */
#define pRest 							'rest'				/*   0x72657374   */
#define pInherits 						'c@#^'				/*   0x6340235e   */
#define pProperties 					'pALL'				/*  User-Defined Record Fields:  */
#define keyASUserRecordFields 			'usrf'				/*   0x75737266   */
#define typeUserRecordFields 			'list'

#define keyASPrepositionAt 				'at  '				/*   0x61742020   */
#define keyASPrepositionIn 				'in  '				/*   0x696e2020   */
#define keyASPrepositionFrom 			'from'				/*   0x66726f6d   */
#define keyASPrepositionFor 			'for '				/*   0x666f7220   */
#define keyASPrepositionTo 				'to  '				/*   0x746f2020   */
#define keyASPrepositionThru 			'thru'				/*   0x74687275   */
#define keyASPrepositionThrough 		'thgh'				/*   0x74686768   */
#define keyASPrepositionBy 				'by  '				/*   0x62792020   */
#define keyASPrepositionOn 				'on  '				/*   0x6f6e2020   */
#define keyASPrepositionInto 			'into'				/*   0x696e746f   */
#define keyASPrepositionOnto 			'onto'				/*   0x6f6e746f   */
#define keyASPrepositionBetween 		'btwn'				/*   0x6274776e   */
#define keyASPrepositionAgainst 		'agst'				/*   0x61677374   */
#define keyASPrepositionOutOf 			'outo'				/*   0x6f75746f   */
#define keyASPrepositionInsteadOf 		'isto'				/*   0x6973746f   */
#define keyASPrepositionAsideFrom 		'asdf'				/*   0x61736466   */
#define keyASPrepositionAround 			'arnd'				/*   0x61726e64   */
#define keyASPrepositionBeside 			'bsid'				/*   0x62736964   */
#define keyASPrepositionBeneath 		'bnth'				/*   0x626e7468   */
#define keyASPrepositionUnder 			'undr'				/*   0x756e6472   */

#define keyASPrepositionOver 			'over'				/*   0x6f766572   */
#define keyASPrepositionAbove 			'abve'				/*   0x61627665   */
#define keyASPrepositionBelow 			'belw'				/*   0x62656c77   */
#define keyASPrepositionApartFrom 		'aprt'				/*   0x61707274   */
#define keyASPrepositionGiven 			'givn'				/*   0x6769766e   */
#define keyASPrepositionWith 			'with'				/*   0x77697468   */
#define keyASPrepositionWithout 		'wout'				/*   0x776f7574   */
#define keyASPrepositionAbout 			'abou'				/*   0x61626f75   */
#define keyASPrepositionSince 			'snce'				/*   0x736e6365   */
#define keyASPrepositionUntil 			'till'				/*   0x74696c6c   */

															/*  Terminology & Dialect things:  */
#define kDialectBundleResType 			'Dbdl'				/*   0x4462646c   */
															/*  AppleScript Classes and Enums:  */
#define cConstant 						'enum'
#define cClassIdentifier 				'pcls'
#define cObjectBeingExamined 			'exmn'
#define cList 							'list'
#define cSmallReal 						'sing'
#define cReal 							'doub'
#define cRecord 						'reco'
#define cReference 						'obj '
#define cUndefined 						'undf'				/*   0x756e6466   */
#define cMissingValue 					'msng'				/*   AppleScript 1.3 newly created */
#define cSymbol 						'symb'				/*   0x73796d62   */
#define cLinkedList 					'llst'				/*   0x6c6c7374   */
#define cVector 						'vect'				/*   0x76656374   */
#define cEventIdentifier 				'evnt'				/*   0x65766e74   */
#define cKeyIdentifier 					'kyid'				/*   0x6b796964   */
#define cUserIdentifier 				'uid '				/*   0x75696420   */
#define cPreposition 					'prep'				/*   0x70726570   */
#define cKeyForm 						'kfrm'
#define cScript 						'scpt'				/*   0x73637074   */
#define cHandler 						'hand'				/*   0x68616e64   */
#define cProcedure 						'proc'				/*   0x70726f63   */

#define cClosure 						'clsr'				/*   0x636c7372   */
#define cRawData 						'rdat'				/*   0x72646174   */
#define cStringClass 					'TEXT'
#define cNumber 						'nmbr'				/*   0x6e6d6272   */
#define cListElement 					'celm'				/*  AppleScript 1.3 added from private headers  */
#define cListOrRecord 					'lr  '				/*   0x6c722020   */
#define cListOrString 					'ls  '				/*   0x6c732020   */
#define cListRecordOrString 			'lrs '				/*   0x6c727320   */
#define cNumberOrString 				'ns  '				/*  AppleScript 1.3 for Display Dialog  */
#define cNumberOrDateTime 				'nd  '				/*   0x6e642020   */
#define cNumberDateTimeOrString 		'nds '				/*   0x6e647320   */
#define cAliasOrString 					'sf  '
#define cSeconds 						'scnd'				/*   0x73636e64   */
#define typeSound 						'snd '
#define enumBooleanValues 				'boov'				/*   Use this instead of typeBoolean to avoid with/without conversion   */
#define kAETrue 						'true'
#define kAEFalse 						'fals'
#define enumMiscValues 					'misc'				/*   0x6d697363   */
#define kASCurrentApplication 			'cura'				/*   0x63757261   */
															/*  User-defined property ospecs:  */
#define formUserPropertyID 				'usrp'				/*   0x75737270   */

															/*  Global properties:  */
#define pASIt 							'it  '				/*   0x69742020   */
#define pASMe 							'me  '				/*   0x6d652020   */
#define pASResult 						'rslt'				/*   0x72736c74   */
#define pASSpace 						'spac'				/*   0x73706163   */
#define pASReturn 						'ret '				/*   0x72657420   */
#define pASTab 							'tab '				/*   0x74616220   */
#define pASPi 							'pi  '				/*   0x70692020   */
#define pASParent 						'pare'				/*   0x70617265   */
#define kASInitializeEventCode 			'init'				/*   0x696e6974   */
#define pASPrintLength 					'prln'				/*   0x70726c6e   */
#define pASPrintDepth 					'prdp'				/*   0x70726470   */
#define pASTopLevelScript 				'ascr'				/*   0x61736372   */

															/*  Considerations  */
#define kAECase 						'case'				/*   0x63617365   */
#define kAEDiacritic 					'diac'				/*   0x64696163   */
#define kAEWhiteSpace 					'whit'				/*   0x77686974   */
#define kAEHyphens 						'hyph'				/*   0x68797068   */
#define kAEExpansion 					'expa'				/*   0x65787061   */
#define kAEPunctuation 					'punc'				/*   0x70756e63   */
#define kAEZenkakuHankaku 				'zkhk'				/*   0x7a6b686b   */
#define kAESmallKana 					'skna'				/*   0x736b6e61   */
#define kAEKataHiragana 				'hika'				/*   0x68696b61   */
															/*  AppleScript considerations:  */
#define kASConsiderReplies 				'rmte'				/*   0x726d7465   */
#define enumConsiderations 				'cons'				/*   0x636f6e73   */

#define cCoercion 						'coec'				/*   0x636f6563   */
#define cCoerceUpperCase 				'txup'				/*   0x74787570   */
#define cCoerceLowerCase 				'txlo'				/*   0x74786c6f   */
#define cCoerceRemoveDiacriticals 		'txdc'				/*   0x74786463   */
#define cCoerceRemovePunctuation 		'txpc'				/*   0x74787063   */
#define cCoerceRemoveHyphens 			'txhy'				/*   0x74786879   */
#define cCoerceOneByteToTwoByte 		'txex'				/*   0x74786578   */
#define cCoerceRemoveWhiteSpace 		'txws'				/*   0x74787773   */
#define cCoerceSmallKana 				'txsk'				/*   0x7478736b   */
#define cCoerceZenkakuhankaku 			'txze'				/*   0x74787a65   */
#define cCoerceKataHiragana 			'txkh'				/*   0x74786b68   */
															/*  Lorax things:  */
#define cZone 							'zone'				/*   0x7a6f6e65   */
#define cMachine 						'mach'				/*   0x6d616368   */
#define cAddress 						'addr'				/*   0x61646472   */
#define cRunningAddress 				'radd'				/*   0x72616464   */
#define cStorage 						'stor'				/*   0x73746f72   */

															/*  DateTime things:  */
#define pASWeekday 						'wkdy'				/*   0x776b6479   */
#define pASMonth 						'mnth'				/*   0x6d6e7468   */
#define pASDay 							'day '				/*   0x64617920   */
#define pASYear 						'year'				/*   0x79656172   */
#define pASTime 						'time'				/*   0x74696d65   */
#define pASDateString 					'dstr'				/*   0x64737472   */
#define pASTimeString 					'tstr'				/*   0x74737472   */
															/*  Months  */
#define cMonth 							'mnth'
#define cJanuary 						'jan '				/*   0x6a616e20   */
#define cFebruary 						'feb '				/*   0x66656220   */
#define cMarch 							'mar '				/*   0x6d617220   */
#define cApril 							'apr '				/*   0x61707220   */
#define cMay 							'may '				/*   0x6d617920   */
#define cJune 							'jun '				/*   0x6a756e20   */
#define cJuly 							'jul '				/*   0x6a756c20   */
#define cAugust 						'aug '				/*   0x61756720   */
#define cSeptember 						'sep '				/*   0x73657020   */
#define cOctober 						'oct '				/*   0x6f637420   */
#define cNovember 						'nov '				/*   0x6e6f7620   */
#define cDecember 						'dec '				/*   0x64656320   */

															/*  Weekdays  */
#define cWeekday 						'wkdy'
#define cSunday 						'sun '				/*   0x73756e20   */
#define cMonday 						'mon '				/*   0x6d6f6e20   */
#define cTuesday 						'tue '				/*   0x74756520   */
#define cWednesday 						'wed '				/*   0x77656420   */
#define cThursday 						'thu '				/*   0x74687520   */
#define cFriday 						'fri '				/*   0x66726920   */
#define cSaturday 						'sat '				/*   0x73617420   */
															/*  AS 1.1 Globals:  */
#define pASQuote 						'quot'				/*   0x71756f74   */
#define pASSeconds 						'secs'				/*   0x73656373   */
#define pASMinutes 						'min '				/*   0x6d696e20   */
#define pASHours 						'hour'				/*   0x686f7572   */
#define pASDays 						'days'				/*   0x64617973   */
#define pASWeeks 						'week'				/*   0x7765656b   */
															/*  Writing Code things:  */
#define cWritingCodeInfo 				'citl'				/*   0x6369746c   */
#define pScriptCode 					'pscd'				/*   0x70736364   */
#define pLangCode 						'plcd'				/*   0x706c6364   */
															/*  Magic Tell and End Tell events for logging:  */
#define kASMagicTellEvent 				'tell'				/*   0x74656c6c   */
#define kASMagicEndTellEvent 			'tend'				/*   0x74656e64   */


#endif /* __ASREGISTRY_R__ */

