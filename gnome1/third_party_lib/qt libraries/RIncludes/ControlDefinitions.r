/*
     File:       ControlDefinitions.r
 
     Contains:   Definitions of controls used by Control Mgr
 
     Version:    Technology: Mac OS 9
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1999-2001 by Apple Computer, Inc., all rights reserved.
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __CONTROLDEFINITIONS_R__
#define __CONTROLDEFINITIONS_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#ifndef __CONTROLS_R__
#include "Controls.r"
#endif

#define kControlTabListResType 			'tab#'				/*  used for tab control (Appearance 1.0 and later) */
#define kControlListDescResType 		'ldes'				/*  used for list box control (Appearance 1.0 and later) */

#define kControlCheckBoxUncheckedValue 	0
#define kControlCheckBoxCheckedValue 	1
#define kControlCheckBoxMixedValue 		2

#define kControlRadioButtonUncheckedValue  0
#define kControlRadioButtonCheckedValue  1
#define kControlRadioButtonMixedValue 	2

#define popupFixedWidth 				0x01
#define popupVariableWidth 				0x02
#define popupUseAddResMenu 				0x04
#define popupUseWFont 					0x08

#define popupTitleBold 					0x0100
#define popupTitleItalic 				0x0200
#define popupTitleUnderline 			0x0400
#define popupTitleOutline 				0x0800
#define popupTitleShadow 				0x1000
#define popupTitleCondense 				0x2000
#define popupTitleExtend 				0x4000
#define popupTitleNoStyle 				0x8000

#define popupTitleLeftJust 				0x00000000
#define popupTitleCenterJust 			0x00000001
#define popupTitleRightJust 			0x000000FF

#define pushButProc 					0
#define checkBoxProc 					1
#define radioButProc 					2
#define scrollBarProc 					16
#define popupMenuProc 					1008

#define kControlLabelPart 				1
#define kControlMenuPart 				2
#define kControlTrianglePart 			4
#define kControlEditTextPart 			5					/*  Appearance 1.0 and later */
#define kControlPicturePart 			6					/*  Appearance 1.0 and later */
#define kControlIconPart 				7					/*  Appearance 1.0 and later */
#define kControlClockPart 				8					/*  Appearance 1.0 and later */
#define kControlListBoxPart 			24					/*  Appearance 1.0 and later */
#define kControlListBoxDoubleClickPart 	25					/*  Appearance 1.0 and later */
#define kControlImageWellPart 			26					/*  Appearance 1.0 and later */
#define kControlRadioGroupPart 			27					/*  Appearance 1.0.2 and later */
#define kControlButtonPart 				10
#define kControlCheckBoxPart 			11
#define kControlRadioButtonPart 		11
#define kControlUpButtonPart 			20
#define kControlDownButtonPart 			21
#define kControlPageUpPart 				22
#define kControlPageDownPart 			23
#define kControlClockHourDayPart 		9					/*  Appearance 1.1 and later */
#define kControlClockMinuteMonthPart 	10					/*  Appearance 1.1 and later */
#define kControlClockSecondYearPart 	11					/*  Appearance 1.1 and later */
#define kControlClockAMPMPart 			12					/*  Appearance 1.1 and later */
#define kControlDataBrowserPart 		24					/*  CarbonLib 1.0 and later */
#define kControlDataBrowserDraggedPart 	25					/*  CarbonLib 1.0 and later */

#define kControlBevelButtonSmallBevelProc  32
#define kControlBevelButtonNormalBevelProc  33
#define kControlBevelButtonLargeBevelProc  34

#define kControlBevelButtonSmallBevelVariant  0
#define kControlBevelButtonNormalBevelVariant  0x01
#define kControlBevelButtonLargeBevelVariant  0x02
#define kControlBevelButtonMenuOnRightVariant  0x04

#define kControlBevelButtonSmallBevel 	0
#define kControlBevelButtonNormalBevel 	1
#define kControlBevelButtonLargeBevel 	2

#define kControlBehaviorPushbutton 		0
#define kControlBehaviorToggles 		0x0100
#define kControlBehaviorSticky 			0x0200
#define kControlBehaviorSingleValueMenu  0
#define kControlBehaviorMultiValueMenu 	0x4000				/*  only makes sense when a menu is attached. */
#define kControlBehaviorOffsetContents 	0x8000

#define kControlBehaviorCommandMenu 	0x2000				/*  menu holds commands, not choices. Overrides multi-value bit. */
#define kControlBevelButtonMenuOnBottom  0
#define kControlBevelButtonMenuOnRight 	0x04

#define kControlBevelButtonAlignSysDirection  (-1)			/*  only left or right */
#define kControlBevelButtonAlignCenter 	0
#define kControlBevelButtonAlignLeft 	1
#define kControlBevelButtonAlignRight 	2
#define kControlBevelButtonAlignTop 	3
#define kControlBevelButtonAlignBottom 	4
#define kControlBevelButtonAlignTopLeft  5
#define kControlBevelButtonAlignBottomLeft  6
#define kControlBevelButtonAlignTopRight  7
#define kControlBevelButtonAlignBottomRight  8

#define kControlBevelButtonAlignTextSysDirection  0
#define kControlBevelButtonAlignTextCenter  1
#define kControlBevelButtonAlignTextFlushRight  (-1)
#define kControlBevelButtonAlignTextFlushLeft  (-2)

#define kControlBevelButtonPlaceSysDirection  (-1)			/*  if graphic on right, then on left */
#define kControlBevelButtonPlaceNormally  0
#define kControlBevelButtonPlaceToRightOfGraphic  1
#define kControlBevelButtonPlaceToLeftOfGraphic  2
#define kControlBevelButtonPlaceBelowGraphic  3
#define kControlBevelButtonPlaceAboveGraphic  4

#define kControlBevelButtonContentTag 	'cont'				/*  ButtonContentInfo */
#define kControlBevelButtonTransformTag  'tran'				/*  IconTransformType */
#define kControlBevelButtonTextAlignTag  'tali'				/*  ButtonTextAlignment */
#define kControlBevelButtonTextOffsetTag  'toff'			/*  SInt16 */
#define kControlBevelButtonGraphicAlignTag  'gali'			/*  ButtonGraphicAlignment */
#define kControlBevelButtonGraphicOffsetTag  'goff'			/*  Point */
#define kControlBevelButtonTextPlaceTag  'tplc'				/*  ButtonTextPlacement */
#define kControlBevelButtonMenuValueTag  'mval'				/*  SInt16 */
#define kControlBevelButtonMenuHandleTag  'mhnd'			/*  MenuHandle */
#define kControlBevelButtonCenterPopupGlyphTag  'pglc'		/*  Boolean: true = center, false = bottom right */

#define kControlBevelButtonLastMenuTag 	'lmnu'				/*  SInt16: menuID of last menu item selected from */
#define kControlBevelButtonMenuDelayTag  'mdly'				/*  SInt32: ticks to delay before menu appears */

															/*  Boolean: True = if an icon of the ideal size for */
															/*  the button isn't available, scale a larger or */
															/*  smaller icon to the ideal size. False = don't */
															/*  scale; draw a smaller icon or clip a larger icon. */
															/*  Default is false. Only applies to IconSuites and */
#define kControlBevelButtonScaleIconTag  'scal'				/*  IconRefs. */
#define kControlSliderProc 				48
#define kControlSliderLiveFeedback 		0x01
#define kControlSliderHasTickMarks 		0x02
#define kControlSliderReverseDirection 	0x04
#define kControlSliderNonDirectional 	0x08

#define kControlTriangleProc 			64
#define kControlTriangleLeftFacingProc 	65
#define kControlTriangleAutoToggleProc 	66
#define kControlTriangleLeftFacingAutoToggleProc  67

#define kControlTriangleLastValueTag 	'last'				/*  SInt16 */
#define kControlProgressBarProc 		80
#define kControlProgressBarIndeterminateTag  'inde'			/*  Boolean */
#define kControlLittleArrowsProc 		96
#define kControlChasingArrowsProc 		112
#define kControlTabLargeProc 			128					/*  Large tab size, north facing    */
#define kControlTabSmallProc 			129					/*  Small tab size, north facing    */
#define kControlTabLargeNorthProc 		128					/*  Large tab size, north facing    */
#define kControlTabSmallNorthProc 		129					/*  Small tab size, north facing    */
#define kControlTabLargeSouthProc 		130					/*  Large tab size, south facing    */
#define kControlTabSmallSouthProc 		131					/*  Small tab size, south facing    */
#define kControlTabLargeEastProc 		132					/*  Large tab size, east facing     */
#define kControlTabSmallEastProc 		133					/*  Small tab size, east facing     */
#define kControlTabLargeWestProc 		134					/*  Large tab size, west facing     */
#define kControlTabSmallWestProc 		135					/*  Small tab size, west facing     */

#define kControlTabContentRectTag 		'rect'				/*  Rect */
#define kControlTabEnabledFlagTag 		'enab'				/*  Boolean */
#define kControlTabFontStyleTag 		'font'				/*  ControlFontStyleRec */

#define kControlTabInfoTag 				'tabi'				/*  ControlTabInfoRec */
#define kControlTabInfoVersionZero 		0
#define kControlSeparatorLineProc 		144
#define kControlGroupBoxTextTitleProc 	160
#define kControlGroupBoxCheckBoxProc 	161
#define kControlGroupBoxPopupButtonProc  162
#define kControlGroupBoxSecondaryTextTitleProc  164
#define kControlGroupBoxSecondaryCheckBoxProc  165
#define kControlGroupBoxSecondaryPopupButtonProc  166

#define kControlGroupBoxMenuHandleTag 	'mhan'				/*  MenuHandle (popup title only) */
#define kControlGroupBoxFontStyleTag 	'font'				/*  ControlFontStyleRec */

#define kControlGroupBoxTitleRectTag 	'trec'				/*  Rect. Rectangle that the title text/control is drawn in. (get only) */
#define kControlImageWellProc 			176
#define kControlImageWellContentTag 	'cont'				/*  ButtonContentInfo */
#define kControlImageWellTransformTag 	'tran'				/*  IconTransformType */

#define kControlPopupArrowEastProc 		192
#define kControlPopupArrowWestProc 		193
#define kControlPopupArrowNorthProc 	194
#define kControlPopupArrowSouthProc 	195
#define kControlPopupArrowSmallEastProc  196
#define kControlPopupArrowSmallWestProc  197
#define kControlPopupArrowSmallNorthProc  198
#define kControlPopupArrowSmallSouthProc  199

#define kControlPopupArrowOrientationEast  0
#define kControlPopupArrowOrientationWest  1
#define kControlPopupArrowOrientationNorth  2
#define kControlPopupArrowOrientationSouth  3

#define kControlPlacardProc 			224
#define kControlClockTimeProc 			240
#define kControlClockTimeSecondsProc 	241
#define kControlClockDateProc 			242
#define kControlClockMonthYearProc 		243

#define kControlClockTypeHourMinute 	0
#define kControlClockTypeHourMinuteSecond  1
#define kControlClockTypeMonthDay 		2
#define kControlClockTypeMonthDayYear 	3

#define kControlClockFlagStandard 		0					/*  editable, non-live */
#define kControlClockNoFlags 			0
#define kControlClockFlagDisplayOnly 	1					/*  add this to become non-editable */
#define kControlClockIsDisplayOnly 		1
#define kControlClockFlagLive 			2					/*  automatically shows current time on idle. only valid with display only. */
#define kControlClockIsLive 			2

#define kControlClockLongDateTag 		'date'				/*  LongDateRec */
#define kControlClockFontStyleTag 		'font'				/*  ControlFontStyleRec */

#define kControlUserPaneProc 			256
#define kControlUserItemDrawProcTag 	'uidp'				/*  UserItemUPP */
#define kControlUserPaneDrawProcTag 	'draw'				/*  ControlUserPaneDrawingUPP */
#define kControlUserPaneHitTestProcTag 	'hitt'				/*  ControlUserPaneHitTestUPP */
#define kControlUserPaneTrackingProcTag  'trak'				/*  ControlUserPaneTrackingUPP */
#define kControlUserPaneIdleProcTag 	'idle'				/*  ControlUserPaneIdleUPP */
#define kControlUserPaneKeyDownProcTag 	'keyd'				/*  ControlUserPaneKeyDownUPP */
#define kControlUserPaneActivateProcTag  'acti'				/*  ControlUserPaneActivateUPP */
#define kControlUserPaneFocusProcTag 	'foci'				/*  ControlUserPaneFocusUPP */
#define kControlUserPaneBackgroundProcTag  'back'			/*  ControlUserPaneBackgroundUPP */

#define kControlEditTextProc 			272
#define kControlEditTextPasswordProc 	274

#define kControlEditTextInlineInputProc  276				/*  Can't combine with the other variants */
#define kControlEditTextStyleTag 		'font'				/*  ControlFontStyleRec */
#define kControlEditTextTextTag 		'text'				/*  Buffer of chars - you supply the buffer */
#define kControlEditTextTEHandleTag 	'than'				/*  The TEHandle of the text edit record */
#define kControlEditTextKeyFilterTag 	'fltr'
#define kControlEditTextSelectionTag 	'sele'				/*  EditTextSelectionRec */
#define kControlEditTextPasswordTag 	'pass'				/*  The clear text password text */

#define kControlEditTextKeyScriptBehaviorTag  'kscr'		/*  ControlKeyScriptBehavior. Defaults to "PrefersRoman" for password fields, */
															/*        or "AllowAnyScript" for non-password fields. */
#define kControlEditTextLockedTag 		'lock'				/*  Boolean. Locking disables editability. */
#define kControlEditTextFixedTextTag 	'ftxt'				/*  Like the normal text tag, but fixes inline input first */
#define kControlEditTextValidationProcTag  'vali'			/*  ControlEditTextValidationUPP. Called when a key filter can't be: after cut, paste, etc. */
#define kControlEditTextInlinePreUpdateProcTag  'prup'		/*  TSMTEPreUpdateUPP and TSMTEPostUpdateUpp. For use with inline input variant... */
#define kControlEditTextInlinePostUpdateProcTag  'poup'		/*  ...The refCon parameter will contain the ControlHandle. */

#define kControlStaticTextProc 			288
#define kControlStaticTextStyleTag 		'font'				/*  ControlFontStyleRec */
#define kControlStaticTextTextTag 		'text'				/*  Copy of text */
#define kControlStaticTextTextHeightTag  'thei'				/*  SInt16 */

#define kControlStaticTextTruncTag 		'trun'				/*  TruncCode (-1 means no truncation) */
#define kControlPictureProc 			304
#define kControlPictureNoTrackProc 		305					/*  immediately returns kControlPicturePart */

#define kControlIconProc 				320
#define kControlIconNoTrackProc 		321					/*  immediately returns kControlIconPart */
#define kControlIconSuiteProc 			322
#define kControlIconSuiteNoTrackProc 	323					/*  immediately returns kControlIconPart */

															/*  icon ref controls may have either an icon, color icon, icon suite, or icon ref. */
															/*  for data other than icon, you must set the data by passing a */
															/*  ControlButtonContentInfo to SetControlData */
#define kControlIconRefProc 			324
#define kControlIconRefNoTrackProc 		325					/*  immediately returns kControlIconPart */

#define kControlIconTransformTag 		'trfm'				/*  IconTransformType */
#define kControlIconAlignmentTag 		'algn'				/*  IconAlignmentType */

#define kControlIconResourceIDTag 		'ires'				/*  SInt16 resource ID of icon to use */
#define kControlIconContentTag 			'cont'				/*  accepts a ControlButtonContentInfo */

#define kControlWindowHeaderProc 		336					/*  normal header */
#define kControlWindowListViewHeaderProc  337				/*  variant for list views - no bottom line */

#define kControlListBoxProc 			352
#define kControlListBoxAutoSizeProc 	353

#define kControlListBoxListHandleTag 	'lhan'				/*  ListHandle */
#define kControlListBoxKeyFilterTag 	'fltr'				/*  ControlKeyFilterUPP */
#define kControlListBoxFontStyleTag 	'font'				/*  ControlFontStyleRec */

#define kControlListBoxDoubleClickTag 	'dblc'				/*  Boolean. Was last click a double-click? */
#define kControlListBoxLDEFTag 			'ldef'				/*  SInt16. ID of LDEF to use. */

#define kControlPushButtonProc 			368
#define kControlCheckBoxProc 			369
#define kControlRadioButtonProc 		370
#define kControlPushButLeftIconProc 	374					/*  Standard pushbutton with left-side icon */
#define kControlPushButRightIconProc 	375					/*  Standard pushbutton with right-side icon */

#define kControlCheckBoxAutoToggleProc 	371
#define kControlRadioButtonAutoToggleProc  372

#define kControlPushButtonDefaultTag 	'dflt'				/*  default ring flag */
#define kControlPushButtonCancelTag 	'cncl'				/*  cancel button flag (1.1 and later) */

#define kControlScrollBarProc 			384					/*  normal scroll bar */
#define kControlScrollBarLiveProc 		386					/*  live scrolling variant */

#define kControlPopupButtonProc 		400
#define kControlPopupFixedWidthVariant 	0x01
#define kControlPopupVariableWidthVariant  0x02
#define kControlPopupUseAddResMenuVariant  0x04
#define kControlPopupUseWFontVariant 	0x08

#define kControlPopupButtonMenuHandleTag  'mhan'			/*  MenuHandle */
#define kControlPopupButtonMenuIDTag 	'mnid'				/*  SInt16 */

#define kControlPopupButtonExtraHeightTag  'exht'			/*  SInt16 extra vertical whitespace within the button */
#define kControlRadioGroupProc 			416
#define kControlScrollTextBoxProc 		432
#define kControlScrollTextBoxAutoScrollProc  433

#define kControlScrollTextBoxDelayBeforeAutoScrollTag  'stdl' /*  UInt32 (ticks until autoscrolling starts) */
#define kControlScrollTextBoxDelayBetweenAutoScrollTag  'scdl' /*  UInt32 (ticks between scrolls) */
#define kControlScrollTextBoxAutoScrollAmountTag  'samt'	/*  UInt16 (pixels per scroll) -- defaults to 1 */
#define kControlScrollTextBoxContentsTag  'tres'			/*  SInt16 (resource ID of 'TEXT'/'styl') -- write only! */


/*--------------------------ldes ¥ List Box Description Template------------------------*/
/*  Used in conjunction with the list box control.                                    */

type 'ldes'
{
    switch
 {
      case versionZero:
          key integer = 0;    /* version */

         integer;                                                /* Rows                 */
         integer;                                                /* Columns              */
         integer;                                                /* Cell Height          */
         integer;                                                /* Cell Width           */
         byte            noVertScroll, hasVertScroll;            /* Vert Scroll          */
         fill byte;                                              /* Filler Byte          */
         byte            noHorizScroll, hasHorizScroll;          /* Horiz Scroll         */
         fill byte;                                              /* Filler Byte          */
         integer;                                                /* LDEF Res ID          */
         byte            noGrowSpace, hasGrowSpace;              /* HasGrow?             */
         fill byte;
 };
};


/*-------------------------------tab# ¥ Tab Control Template-----------------------------*/
type 'tab#'
{
 switch
 {
      case versionZero:
          key integer = 0;    /* version */

         integer = $$Countof(TabInfo);
          array TabInfo
          {
              integer;                                            /* Icon Suite ID        */
             pstring;                                            /* Tab Name             */
             fill long;                                          /* Reserved             */
             fill word;                                          /* Reserved             */
         };
 };
};


#endif /* __CONTROLDEFINITIONS_R__ */

