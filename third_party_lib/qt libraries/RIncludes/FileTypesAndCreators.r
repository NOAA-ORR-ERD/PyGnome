/*
     File:       FileTypesAndCreators.r
 
     Contains:   Symbolic constants for FileTypes and signatures of popular documents.
 
     Version:    Technology: Macintosh Easy Open 1.1
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1992-2001 by Apple Computer, Inc., all rights reserved.
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __FILETYPESANDCREATORS_R__
#define __FILETYPESANDCREATORS_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

															/*  Mac word processors  */
#define sigWord 						'MSWD'
#define ftWord3Document 				'MSW3'
#define ftWord4Document 				'MSW4'
#define ftWord5Document 				'MSW5'
#define ftWordDocument 					'WDBN'
#define ftWordDocumentPC 				'MWPC'				/*  not registered  */
#define ftWord1DocumentWindows 			'WW1 '				/*  not registered  */
#define ftWord2DocumentWindows 			'WW2 '				/*  not registered  */
#define ftRTFDocument 					'RTF '				/*  not registered  */
#define sigWordPerfect 					'SSIW'
#define ftWordPerfectDocument 			'WPD0'
#define sigWordPerfect2 				'WPC2'
#define ftWordPerfect2Document 			'WPD1'
#define ftWordPerfect21Document 		'WPD2'
#define ftWordPerfect42DocumentPC 		'.WP4'				/*  not registered  */
#define ftWordPerfect50DocumentPC 		'.WP5'				/*  not registered  */
#define ftWordPerfect51DocumentPC 		'WP51'				/*  not registered  */
#define ftWordPerfectGraphicsPC 		'WPGf'				/*  not registered  */
#define sigMacWriteII 					'MWII'
#define ftMacWriteIIDocument 			'MW2D'
#define sigWriteNow 					'nX^n'
#define ftWriteNow2Document 			'nX^d'
#define ftWriteNow3Document 			'nX^2'
#define sigMacWrite 					'MACA'
#define ftMacWrite5Document 			'WORD'
#define sigFrameMaker 					'Fram'
#define ftFrameMakerDocument 			'FASL'
#define ftFrameMakerMIFDocument 		'MIF '
#define ftFrameMakerMIF2Document 		'MIF2'
#define ftFrameMakerMIF3Document 		'MIF3'
#define sigMSWrite 						'MSWT'
#define sigActa 						'ACTA'
#define sigTHINKPascal 					'PJMM'
#define sigTHINKC 						'KAHL'
#define sigFullWrite 					'FWRT'
#define sigTeachText 					'ttxt'
#define ftTeachTextDocument 			'ttro'
#define sigSimpleText 					'ttxt'
#define ftSimpleTextDocument 			'ttro'
#define sigMPWShell 					'MPS '
#define sigQuarkXPress 					'XPR3'
#define sigNisus 						'NISI'
#define sigOmniPage 					'PRTC'
#define sigPersonalPress 				'SCPG'
#define sigPublishItEZ 					'2CTY'
#define sigReadySetGo 					'MEMR'
#define sigRagTime 						'R#+A'
#define sigLetraStudio 					'LSTP'
#define sigLetterPerfect 				'WPCI'
#define sigTheWritingCenter 			0x0A1A5750			/*  this 'unprintable unprintable WP' One of the unprintables is a line feed.   */
#define sigInstantUpdate 				'IUA0'

															/*  databases  */
#define sig4thDimension 				'4D03'
#define ft4thDimensionDB 				'BAS3'
#define sigFileMakerPro 				'FMPR'
#define ftFileMakerProDatabase 			'FMPR'
#define sigHyperCard 					'WILD'
#define ftHyperCard 					'STAK'
#define sigSmartFormAsst 				'KCFM'
#define ftSmartFormAsst 				'STCK'
#define sigSmartFormDesign 				'KCFD'
#define ftSmartFormDesign 				'CFRM'
#define sigFileForce 					'4D93'
#define ftFileForceDatabase 			'FIL3'
#define sigFileMaker2 					'FMK4'
#define ftFileMaker2Database 			'FMK$'
#define sigSuperCard 					'RUNT'
#define sigDoubleHelix 					'HELX'
#define sigGeoQuery 					'RGgq'
#define sigFoxBASE 						'FOX+'
#define sigINSPIRATION 					'CER3'
#define sigPanorama 					'KAS1'
#define sigSilverrunLDM 				'CDML'
#define sigSilverrunDFD 				'CDDF'				/*  finance  */
#define sigQuicken 						'INTU'
#define sigMacInTax91 					'MIT1'
#define ftMacInTax91 					'MITF'
#define sigAccountantInc 				'APRO'
#define sigAtOnce 						'KISS'
#define sigCAT3 						'tCat'
#define sigDollarsNSense 				'EAGP'
#define sigInsightExpert 				'LSGL'
#define sigMYOB 						'MYOB'
#define sigMacMoney 					'SSLA'
#define sigManagingYourMoney 			'MYMC'
#define sigPlainsAndSimple 				'PEGG'				/*  scheduling  */
#define sigMacProject2 					'MPRX'
#define ftMacProject 					'MPRD'
#define sigMSProject 					'MSPJ'
#define sigMacProjectPro 				'MPRP'				/*  utilities  */
#define sigStuffIt 						'SIT!'
#define ftStuffItArchive 				'SIT!'
#define sigCompactPro 					'CPCT'
#define ftCompactProArchive 			'PACT'
#define sigFontographer 				'aCa2'
#define sigMetamorphosis 				'MEtP'
#define sigCorrectGrammar 				'LsCG'
#define sigDynodex 						'DYNO'
#define sigMariah 						'MarH'
#define sigAddressBook 					'AdBk'
#define sigThePrintShop 				'PSHP'
#define sigQuicKeys2 					'Qky2'
#define sigReadStar2Plus 				'INOV'
#define sigSoftPC 						'PCXT'
#define sigMacMenlo 					'MNLO'
#define sigDisinfectant 				'D2CT'				/*  communications  */
#define sigSmartcom2 					'SCOM'
#define sigVersaTermPRO 				'VPRO'
#define sigVersaTerm 					'VATM'
#define sigWhiteKnight 					'WK11'
#define sigNCSATelnet 					'NCSA'
#define sigDynaComm 					'PAR2'
#define sigQMForms 						'MLTM'				/*  math and statistics  */
#define sigMathematica 					'OMEG'
#define sigMathCAD 						'MCAD'
#define sigStatView2 					'STAT'
#define sigDataDesk 					'DDSK'
#define sigPowerMath2 					'MATH'
#define sigSuperANOVA 					'SupA'
#define sigSystat 						'SYT1'
#define sigTheorist 					'Theo'

															/*  spreadsheets  */
#define sigExcel 						'XCEL'
#define ftExcel2Spreadsheet 			'XLS '
#define ftExcel2Macro 					'XLM '
#define ftExcel2Chart 					'XLC '
#define ftExcel3Spreadsheet 			'XLS3'
#define ftExcel3Macro 					'XLM3'
#define ftExcel3Chart 					'XLC3'
#define ftExcel4Spreadsheet 			'XLS4'
#define ftExcel4Macro 					'XLM4'
#define ftSYLKSpreadsheet 				'SYLK'
#define sigLotus123 					'L123'
#define ft123Spreadsheet 				'LWKS'
#define sigWingz 						'WNGZ'
#define ftWingzSpreadsheet 				'WZSS'
#define ftWingzScript 					'WZSC'
#define sigResolve 						'Rslv'
#define ftResolve 						'RsWs'
#define ftResolveScript 				'RsWc'
#define sigFullImpact2 					'Flv2'

															/*  graphics  */
#define sigIllustrator 					'ART3'
#define ftPostScriptMac 				'EPSF'
#define sigMacPaint 					'MPNT'
#define ftMacPaintGraphic 				'PNTG'
#define sigSuperPaint 					'SPNT'
#define ftSuperPaintGraphic 			'SPTG'
#define sigCanvas 						'DAD2'
#define ftCanvasGraphic 				'drw2'
#define sigUltraPaint 					'ULTR'
#define ftUltraPaint 					'UPNT'
#define sigPhotoshop 					'8BIM'
#define ftPhotoshopGraphic 				'8BIM'
#define sigMacDrawPro 					'dPro'
#define ftMacDrawProDrawing 			'dDoc'
#define sigPageMaker 					'ALD4'
#define ftPageMakerPublication 			'ALB4'
#define sigFreeHand 					'FHA3'
#define ftFreeHandDrawing 				'FHD3'
#define sigClarisCAD 					'CCAD'
#define ftClarisCAD 					'CAD2'
#define sigMacDrawII 					'MDPL'
#define ftMacDrawIIDrawing 				'DRWG'
#define sigMacroMindDirector 			'MMDR'
#define ftMMDirectorMovie 				'VWMD'
#define ftMMDirectorSound 				'MMSD'
#define sigOptix 						'PIXL'				/*  was previously PixelPerfect  */
#define sigPixelPaint 					'PIXR'
#define ftPixelPaint 					'PX01'
#define sigAldusSuper3D 				'SP3D'
#define ftSuper3DDrawing 				'3DBX'
#define sigSwivel3D 					'SWVL'
#define ftSwivel3DDrawing 				'SMDL'
#define sigCricketDraw 					'CRDW'
#define ftCricketDrawing 				'CKDT'
#define sigCricketGraph 				'CGRF'
#define ftCricketChart 					'CGPC'
#define sigDesignCAD 					'ASBC'
#define ftDesignCADDrawing 				'DCAD'
#define sigImageStudio 					'FSPE'
#define ftImageStudioGraphic 			'RIFF'
#define sigVersaCad 					'VCAD'
#define ftVersaCADDrawing 				'2D  '
#define sigAdobePremier 				'PrMr'
#define ftAdobePremierMovie 			'MooV'
#define sigAfterDark 					'ADrk'
#define ftAfterDarkModule 				'ADgm'
#define sigClip3D 						'EZ3E'
#define ftClip3Dgraphic 				'EZ3D'
#define sigKaleidaGraph 				'QKPT'
#define ftKaleidaGraphGraphic 			'QPCT'
#define sigMacFlow 						'MCFL'
#define ftMacFlowChart 					'FLCH'
#define sigMoviePlayer 					'TVOD'
#define ftMoviePlayerMovie 				'MooV'
#define sigMacSpin 						'D2SP'
#define ftMacSpinDataSet 				'D2BN'
#define sigAutoCAD 						'ACAD'
#define sigLabVIEW 						'LBVW'
#define sigColorMacCheese 				'CMCÆ'
#define sigMiniCad 						'CDP3'
#define sigDreams 						'PHNX'
#define sigOmnis5 						'Q2$$'
#define sigPhotoMac 					'PMAC'
#define sigGraphMaster 					'GRAM'
#define sigInfiniD 						'SI°D'
#define sigOfoto 						'APLS'
#define sigMacDraw 						'MDRW'
#define sigDeltagraphPro 				'DGRH'
#define sigDesign2 						'DESG'
#define sigDesignStudio 				'MRJN'
#define sigDynaperspective 				'PERS'
#define sigGenericCADD 					'CAD3'
#define sigMacDraft 					'MD20'
#define sigModelShop 					'MDSP'
#define sigOasis 						'TAOA'
#define sigOBJECTMASTER 				'BROW'
#define sigMovieRecorder 				'mrcr'
#define sigPictureCompressor 			'ppxi'
#define sigPICTViewer 					'MDTS'
#define sigSmoothie 					'Smoo'
#define sigScreenPlay 					'SPLY'
#define sigStudio1 						'ST/1'
#define sigStudio32 					'ST32'
#define sigStudio8 						'ST/8'
#define sigKidPix 						'Kid2'
#define sigDigDarkroom 					'DIDR'

															/*  presentations  */
#define sigMore 						'MOR2'
#define ftMore3Document 				'MOR3'
#define ftMore2Document 				'MOR2'
#define sigPersuasion 					'PLP2'
#define ftPersuasion1Presentation 		'PRS1'
#define ftPersuasion2Presentation 		'PRS2'
#define sigPowerPoint 					'PPNT'
#define ftPowerPointPresentation 		'SLDS'
#define sigCricketPresents 				'CRPR'
#define ftCricketPresentation 			'PRDF'				/*  works  */
#define sigMSWorks 						'PSI2'
#define sigMSWorks3 					'MSWK'
#define ftMSWorksWordProcessor 			'AWWP'
#define ftMSWorksSpreadsheet 			'AWSS'
#define ftMSWorksDataBase 				'AWDB'
#define ftMSWorksComm 					'AWDC'
#define ftMSWorksMacros 				'AWMC'
#define ftMSWorks1WordProcessor 		'AWW1'				/*  not registered  */
#define ftMSWorks1Spreadsheet 			'AWS1'				/*  not registered  */
#define ftMSWorks1DataBase 				'AWD1'				/*  not registered  */
#define ftMSWorks2WordProcessor 		'AWW2'				/*  not registered  */
#define ftMSWorks2Spreadsheet 			'AWS2'				/*  not registered  */
#define ftMSWorks2DataBase 				'AWD2'				/*  not registered  */
#define ftMSWorks3WordProcessor 		'AWW3'				/*  not registered  */
#define ftMSWorks3Spreadsheet 			'AWS3'				/*  not registered  */
#define ftMSWorks3DataBase 				'AWD3'				/*  not registered  */
#define ftMSWorks3Comm 					'AWC3'				/*  not registered  */
#define ftMSWorks3Macro 				'AWM3'				/*  not registered  */
#define ftMSWorks3Draw 					'AWR3'				/*  not registered  */
#define ftMSWorks2WordProcessorPC 		'PWW2'				/*  not registered  */
#define ftMSWorks2DatabasePC 			'PWDB'				/*  not registered  */
#define sigGreatWorks 					'ZEBR'
#define ftGreatWorksWordProcessor 		'ZWRT'
#define ftGreatWorksSpreadsheet 		'ZCAL'
#define ftGreatWorksPaint 				'ZPNT'
#define sigClarisWorks 					'BOBO'
#define ftClarisWorksWordProcessor 		'CWWP'
#define ftClarisWorksSpreadsheet 		'CWSS'
#define ftClarisWorksGraphics 			'CWGR'
#define sigBeagleWorks 					'BWks'
#define ftBeagleWorksWordProcessor 		'BWwp'
#define ftBeagleWorksDatabase 			'BWdb'
#define ftBeagleWorksSpreadsheet 		'BWss'
#define ftBeagleWorksComm 				'BWcm'
#define ftBeagleWorksDrawing 			'BWdr'
#define ftBeagleWorksGraphic 			'BWpt'
#define ftPICTFile 						'PICT'

															/*  entertainment  */
#define sigPGATourGolf 					'gOLF'
#define sigSimCity 						'MCRP'
#define sigHellCats 					'HELL'				/*  education  */
#define sigReaderRabbit3 				'RDR3'				/*  Translation applications  */
#define sigDataVizDesktop 				'DVDT'
#define sigSotwareBridge 				'mdos'
#define sigWordForWord 					'MSTR'
#define sigAppleFileExchange 			'PSPT'				/*  Apple software  */
#define sigAppleLink 					'GEOL'
#define ftAppleLinkAddressBook 			'ADRS'
#define ftAppleLinkImageFile 			'SIMA'
#define ftAppleLinkPackage 				'HBSF'
#define ftAppleLinkConnFile 			'PETE'
#define ftAppleLinkHelp 				'HLPF'
#define sigInstaller 					'bjbc'
#define ftInstallerScript 				'bjbc'
#define sigDiskCopy 					'dCpy'
#define ftDiskCopyImage 				'dImg'
#define sigResEdit 						'RSED'
#define ftResEditResourceFile 			'rsrc'
#define sigAardvark 					'AARD'
#define sigCompatibilityChkr 			'wkrp'
#define sigMacTerminal 					'Term'
#define sigSADE 						'sade'
#define sigCurare 						'Cura'
#define sigPCXChange 					'dosa'
#define sigAtEase 						'mfdr'
#define sigStockItToMe 					'SITM'
#define sigAppleSearch 					'asis'
#define sigAppleSearchToo 				'hobs'				/*  the following are files types for system files  */
#define ftScriptSystemResourceCollection  'ifil'
#define ftSoundFile 					'sfil'
#define ftFontFile 						'ffil'
#define ftTrueTypeFontFile 				'tfil'
#define ftKeyboardLayout 				'kfil'
#define ftFontSuitcase 					'FFIL'
#define ftDASuitcase 					'DFIL'
#define ftSystemExtension 				'INIT'
#define ftDAMQueryDocument 				'qery'

#define ftApplicationName 				'apnm'				/*  this is the type used to define the application name in a kind resource  */
#define sigIndustryStandard 			'istd'				/*  this is the creator used to define a kind string in a kind resource for a FileType that has many creators   */
#define ftXTND13TextImport 				'xt13'				/*  this is a pseduo-format used by "XTND for Apps". The taDstIsAppTranslation bit is set  */

#define sigAppleProDOS 					'pdos'				/*  not registered  */
#define ftAppleWorksWordProcessor 		'1A  '				/*  not registered  */
#define ftAppleWorks1WordProcessor 		'1A1 '				/*  not registered  */
#define ftAppleWorks2WordProcessor 		'1A2 '				/*  not registered  */
#define ftAppleWorks3WordProcessor 		'1A3 '				/*  not registered  */
#define ftAppleWorksDataBase 			'19  '				/*  not registered  */
#define ftAppleWorks1DataBase 			'191 '				/*  not registered  */
#define ftAppleWorks2DataBase 			'192 '				/*  not registered  */
#define ftAppleWorks3DataBase 			'193 '				/*  not registered  */
#define ftAppleWorksSpreadsheet 		'1B  '				/*  not registered  */
#define ftAppleWorks1Spreadsheet 		'1B1 '				/*  not registered  */
#define ftAppleWorks2Spreadsheet 		'1B2 '				/*  not registered  */
#define ftAppleWorks3Spreadsheet 		'1B3 '				/*  not registered  */
#define ftAppleWorksWordProcessorGS 	'50  '				/*  not registered  */
#define ftApple2GS_SuperHiRes 			'A2SU'				/*  not registered  */
#define ftApple2GS_SuperHiResPacked 	'A2SP'				/*  not registered  */
#define ftApple2GS_PaintWorks 			'A2PW'				/*  not registered  */
#define ftApple2_DoubleHiRes 			'A2DU'				/*  not registered  */
#define ftApple2_DoubleHiResPacked 		'A2DP'				/*  not registered  */
#define ftApple2_DoubleHiRes16colors 	'A2DC'				/*  not registered  */
#define ftApple2_SingleHiRes 			'A2HU'				/*  not registered  */
#define ftApple2_SingleHiResPacked 		'A2HP'				/*  not registered  */
#define ftApple2_SingleHiRes8colors 	'A2HC'				/*  not registered  */

#define sigPCDOS 						'mdos'				/*  not registered  */
#define ftGenericDocumentPC 			'TEXT'				/*     word processor formats  */
#define ftWordStarDocumentPC 			'WStr'				/*  not registered  */
#define ftWordStar4DocumentPC 			'WSt4'				/*  not registered  */
#define ftWordStar5DocumentPC 			'WSt5'				/*  not registered  */
#define ftWordStar55DocumentPC 			'WS55'				/*  not registered  */
#define ftWordStar6DocumentPC 			'WSt6'				/*  not registered  */
#define ftWordStar2000DocumentPC 		'WS20'				/*  not registered  */
#define ftXyWriteIIIDocumentPC 			'XyWr'				/*  registered???  */
#define ftDecDXDocumentPC 				'DX  '				/*  registered???  */
#define ftDecWPSPlusDocumentPC 			'WPS+'				/*  registered???  */
#define ftDisplayWrite3DocumentPC 		'DW3 '				/*  registered???  */
#define ftDisplayWrite4DocumentPC 		'DW4 '				/*  registered???  */
#define ftDisplayWrite5DocumentPC 		'DW5 '				/*  registered???  */
#define ftIBMWritingAsstDocumentPC 		'ASST'				/*  registered???  */
#define ftManuscript1DocumentPC 		'MAN1'				/*  registered???  */
#define ftManuscript2DocumentPC 		'MAN2'				/*  registered???  */
#define ftMass11PCDocumentPC 			'M11P'				/*  registered???  */
#define ftMass11VaxDocumentPC 			'M11V'				/*  registered???  */
#define ftMultiMateDocumentPC 			'MMAT'				/*  registered???  */
#define ftMultiMate36DocumentPC 		'MM36'				/*  registered???  */
#define ftMultiMate40DocumentPC 		'MM40'				/*  registered???  */
#define ftMultiMateAdvDocumentPC 		'MMAD'				/*  registered???  */
#define ftMultiMateNoteDocumentPC 		'MMNT'				/*  registered???  */
#define ftOfficeWriterDocumentPC 		'OFFW'				/*  registered???  */
#define ftPCFileLetterDocumentPC 		'PCFL'				/*  registered???  */
#define ftPFSWriteADocumentPC 			'PFSA'				/*  registered???  */
#define ftPFSWriteBDocumentPC 			'PFSB'				/*  registered???  */
#define ftPFSPlanDocumentPC 			'PFSP'				/*  registered???  */
#define ftProWrite1DocumentPC 			'PW1 '				/*  registered???  */
#define ftProWrite2DocumentPC 			'PW2 '				/*  registered???  */
#define ftProWritePlusDocumentPC 		'PW+ '				/*  registered???  */
#define ftFirstChoiceDocumentPC 		'FCH '				/*  registered???  */
#define ftFirstChoice3DocumentPC 		'FCH3'				/*  registered???  */
#define ftDCARFTDocumentPC 				'RFT '				/*  registered???  */
#define ftSamnaDocumentPC 				'SAMN'				/*  registered???  */
#define ftSmartDocumentPC 				'SMRT'				/*  registered???  */
#define ftSprintDocumentPC 				'SPRT'				/*  registered???  */
#define ftTotalWordDocumentPC 			'TOTL'				/*  registered???  */
#define ftVolksWriterDocumentPC 		'VOLK'				/*  registered???  */
#define ftWangWPSDocumentPC 			'WPS '				/*  registered???  */
#define ftWordMarcDocumentPC 			'MARC'				/*  registered???  */
#define ftAmiDocumentPC 				'AMI '				/*  registered???  */
#define ftAmiProDocumentPC 				'APRO'				/*  registered???  */
#define ftAmiPro2DocumentPC 			'APR2'				/*  registered???  */
#define ftEnableDocumentPC 				'ENWP'				/*  registered???  */
															/*     data base formats  */
#define ftdBaseDatabasePC 				'DBF '				/*  registered???  */
#define ftdBase3DatabasePC 				'DB3 '				/*  registered???  */
#define ftdBase4DatabasePC 				'DB4 '				/*  registered???  */
#define ftDataEaseDatabasePC 			'DTEZ'				/*  registered???  */
#define ftFrameWorkIIIDatabasePC 		'FWK3'				/*  registered???  */
#define ftRBaseVDatabasePC 				'RBsV'				/*  registered???  */
#define ftRBase5000DatabasePC 			'RB50'				/*  registered???  */
#define ftRBaseFile1DatabasePC 			'RBs1'				/*  registered???  */
#define ftRBaseFile3DatabasePC 			'RBs3'				/*  registered???  */
#define ftReflexDatabasePC 				'RFLX'				/*  registered???  */
#define ftQAWriteDatabasePC 			'QAWT'				/*  registered???  */
#define ftQADBaseDatabasePC 			'QADB'				/*  registered???  */
#define ftSmartDataBasePC 				'SMTD'				/*  registered???  */
#define ftFirstChoiceDataBasePC 		'FCDB'				/*  registered???  */

															/*     spread sheet formats  */
#define ftDIFSpreadsheetPC 				'DIF '				/*  registered???  */
#define ftEnableSpreadsheetPC 			'ENAB'				/*  registered???  */
#define ft123R1SpreadsheetPC 			'WKS1'				/*  registered???  */
#define ft123R2SpreadsheetPC 			'WKS2'				/*  registered???  */
#define ft123R3SpreadsheetPC 			'WKS3'				/*  registered???  */
#define ftParadox3SpreadsheetPC 		'PDX3'				/*  registered???  */
#define ftParadox35SpreadsheetPC 		'PD35'				/*  registered???  */
#define ftQuattroSpreadsheetPC 			'QTRO'				/*  registered???  */
#define ftQuattroProSpreadsheetPC 		'QTR5'				/*  registered???  */
#define ftSuperCalc5SpreadsheetPC 		'SPC5'				/*  registered???  */
#define ftSymphony1SpreadsheetPC 		'SYM1'				/*  registered???  */
#define ftTwinSpreadsheetPC 			'TWIN'				/*  registered???  */
#define ftVPPlannerSpreadsheetPC 		'VPPL'				/*  registered???  */
#define ftSmartSpeadsheetPC 			'SMSH'				/*  registered???  */
#define ftFirstChoiceSpeadsheetPC 		'FCSS'				/*  registered???  */
															/*     graphics formats  */
#define ftPCPaintBrushGraphicPC 		'PCX '				/*  not registered  */
#define ftLotusPICGraphicPC 			'.PIC'				/*  not registered  */
#define ftCGMGraphicPC 					'.CGM'				/*  not registered  */
#define ftGEMGraphicPC 					'.GEM'				/*  not registered  */
#define ftIMGGraphicPC 					'.IMG'				/*  not registered  */
#define ftDXFGraphicPC 					'.DXF'				/*  not registered  */
#define ftBitmapWindows 				'.BMP'				/*  not registered  */
#define ftMetaFileWindows 				'.WMF'				/*  not registered  */
#define ftTIFFGraphic 					'TIFF'				/*  not registered  */
#define ftPostScriptPC 					'EPSP'
#define ftPostScriptWindows 			'EPSW'				/*  not registered  */
#define ftDigitalFX_TitleMan 			'TDIM'				/*  registered???  */
#define ftDigitalFX_VideoFX 			'GRAF'				/*  registered???  */
#define ftAutodeskFLIandFLC 			'FLIC'				/*  registered???  */
#define ftGIF 							'GIFf'				/*  registered???  */
#define ftIFF 							'ILBM'				/*  registered???  */
#define ftMicrosoftPaint 				'.MSP'				/*  registered???  */
#define ftPixar 						'PXAR'				/*  registered???  */
#define ftQDV 							'.QDV'				/*  registered???  */
#define ftRLE_Compuserve 				'RLEC'				/*  registered???  */
															/*     Generic vector formats  */
#define ftIGESGraphicPC 				'IGES'				/*  not registered  */
#define ftDDES2GraphicPC 				'DDES'				/*  not registered  */
#define ft3DGFGraphicPC 				'3DGF'				/*  not registered  */
															/*     Plotter formats  */
#define ftHPGLGraphicPC 				'HPGL'				/*  not registered  */
#define ftDMPLGraphicPC 				'DMPL'				/*  not registered  */
#define ftCalComp906GraphicPC 			'C906'				/*  not registered  */
#define ftCalComp907GraphicPC 			'C907'				/*  not registered  */
															/*     Vendor-specific formats  */
#define ftStereoLithographyGraphicPC 	'STL '				/*     3D Systems     - not registered  */
#define ftZoomGraphicPC 				'ZOOM'				/*     Abvent          - not registered  */
#define ftFocusGraphicPC 				'FOCS'				/*     Abvent          - not registered  */
#define ftWaveFrontGraphicPC 			'WOBJ'				/*     WaveFront      - not registered  */
#define ftSculpt4DGraphicPC 			'Scn2'				/*     Byte By Byte   - not registered  */
#define ftMiniPascal3GraphicPC 			'MPT3'				/*     Graphsoft      - not registered  */
#define ftMiniPascal4GraphicPC 			'MPT4'				/*     Graphsoft      - not registered  */
#define ftWalkThroughGraphicPC 			'VWLK'				/*     Virtus          - not registered  */
#define ftSiliconGraphics 				'.SGI'				/*  registered???  */
#define ftSunRaster 					'.SUN'				/*  registered???  */
#define ftTarga 						'TPIC'				/*  registered???  */
															/*  misc DOS   */
#define ftDOSComPC 						'.COM'				/*  registered???  */
#define ftDOSExecutablePC 				'.EXE'				/*  registered???  */
#define ftDOSArcPC 						'.ARC'				/*  registered???  */
#define ftAbekas 						'ABEK'				/*  registered???  */
#define ftDrHaloCUT 					'.CUT'				/*  registered???  */
															/*  misc Atari  */
#define ftDegas 						'DEGA'				/*  not registered  */
#define ftNEO 							'.NEO'				/*  not registered  */


#endif /* __FILETYPESANDCREATORS_R__ */

