
#ifndef __APPTYPES__
#define __APPTYPES__

#define		PI					3.14159265359

#define		kMaxShort			32767
#define		kMinShort	   	   -32767
#define		kMaxLong			2147483647
#define		kMinLong		   -2147483647

#define		kFileNameLen		64
#define		kSetLineWidth		182		// picComment constant for used to insert a SetLineWidth command
#define		kMultFindEvt		15		// code for multifinder suspend/resume message

#define		kScreenMode			0		// mode indicating where a screen image is being drawn
#define		kPictMode			1		
#define		kPrintMode			2
#define		kDragMode			3

#define 	kControlInvisible	0		// used to hide controls
#define 	kControlVisible		0xFF	// used to show controls

#define		kNoColorInd			-1
#define		kWhiteColorInd	 	0
#define		kBlackColorInd	 	1
#define		kLandColorInd	 	8
#define		kWaterColorInd	 	36
#define		kRedColorInd		35
#define		kBlueColorInd		198
#define		kLgtBlueColorInd 	36
#define		kLgtGrayColorInd	249
#define		kVLgtGrayColorInd 	247
#define		kGrayColorInd	 	250
#define		kDkGrayColorInd		252
// colors for ESI Map
#define		kESIDkPurpleInd		301
#define		kESILgtPurpleInd	302
#define		kESIBlueInd			303
#define		kESILgtBlueInd		304
#define		kESILgtBlueGreenInd	305
#define		kESIGreenInd		306
#define		kESILgtGreenInd		307
#define		kESIOliveInd		308
#define		kESIYellowInd		309
#define		kESIPeachInd		310
#define		kESILgtOrangeInd	311
#define		kESIOrangeInd		312
#define		kESIRedInd			313
#define		kESILgtMagentaInd	314
#define		kESIDkRedInd		315
#define		kESIBrownInd		316

void Our_PmForeColor(long colorInd); //JLM 10/19/98

#define		kStdGrayPat		"\pAA55AA55AA55AA55"
#define 	kLgtGrayPat		"\p8822882288228822"
#define		kVeryLgtGrayPat	"\p8800220088002200"
#define		kDarkGrayPat	"\pDD77DD77DD77DD77"
#define		kBlackPat		"\pFFFFFFFFFFFFFFFF"
#define		kWhitePat		"\p0000000000000000"

/**************************************** macros *****************************************/
// 		enumerated key codes for frequently used keys (obtained with a charCodeMask)	

#ifndef MYKEYS
#define MYKEYS

enum { ENTER = 3, DELETE = 8, TAB = 9, RETURN = 13, ESCAPE = 27, LEFT = 28,
	   RIGHT = 29, UP = 30, DOWN = 31 };

// enumerated key codes for frequently used keys (obtained with a keyCodeMask)
enum { ESCAPEKEY = 0X35, ENTERKEY = 0X4C, RETURNKEY = 0X24, PERIODKEY = 0X2F, UPKEY = 0X7E,
	   UPKEYPLUS = 0X4D, DOWNKEY = 0X7D, DOWNKEYPLUS = 0X48, DELETEKEY = 0x335, AKEY = 0X00,
	   IKEY = 0X22, LEFTARROW = 0x7B, RIGHTARROW = 0x7C, TABKEY = 0x30};

#endif
/**************************************** macros *****************************************/
#ifndef __CROSSH__
#define _min(a, b) ((a) < (b) ? (a) : (b))
#define _max(a, b) ((a) > (b) ? (a) : (b))
#endif

#define myabs(n) ((n) >= 0 ? (n) : -(n))



#endif
