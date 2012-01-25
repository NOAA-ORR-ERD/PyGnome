#ifndef __VECTORCONTOUR__
#define __VECTORCONTOUR__

#include "cross.h"

#define CONTOUR_DLGID		5150

//Contour dialog items
#define	CONT_OKAY				1		/* Select btn id.	<Must be item 1> */
#define	CONT_HILITE				2		/* Select ring id.	<Must be item 2> */
#define	CONT_CANCEL				3		/* Cancel btn id.	<Must be item 3> */
#define	CONT_LISTID				4
#define	CONT_INITVALUE			5
#define	CONT_FINALVALUE			6
#define CONT_INCR				13
#define	CONT_ADD				7
#define	CONT_DEL				8
#define	CONT_DELALL				9
#define CONT_ADDONE				17
#define CONT_LEVELVALUE			16
#define CONT_FRAME_MAKELEVELS	14
#define CONT_FRAMEONE			18	
#define CONT_UNITS				19
#define CONT_SHOWVALBOX			20	  
#define CONT_DEFAULT			21	  

void MyDrawString(/*short dir,*/ short h, short v, char *s,Boolean framed,short selectMode);
void GetStringRect(char text[],short h, short v, Rect *r);
void GetTextOffsets(/*short dir,*/char s[], short *h, short *v);

OSErr ContourDialog(DOUBLEH *contourLevels,short contourType);	// concentration 0, depth 1
OSErr SetDefaultContours(DOUBLEH contourLevels, short contourType);
//OSErr SetDefaultDepthContours(DOUBLEH contourLevels);
Boolean AutoContour(DOUBLEH contourValues, DOUBLEH *contourLevels);
//Boolean AutoContour(FLOATH contourValues, DOUBLEH *contourLevels);
RGBColor GetRGBColor(double val);
void  SetRGBColor(RGBColor *rgb,long red, long green, long blue);
long GetNumDoubleHdlItems(DOUBLEH h);
long GetNumHandleItems(Handle h, long itemSize);
void MyDisposeHandle(Handle *g);
char *MyNumToStr(double x, char *s);

#endif
