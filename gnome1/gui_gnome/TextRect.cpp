#include "TextRect.h"


TextRect::TextRect(Rect r,short fontID,short FontSize,short loffset,short toffset)
{
	short offSetToBaseLine;
	FontInfo finfo;
	
	printRect = r;
	fFontID = fontID;
	fontSize = FontSize;
	leftOffset = loffset;
	topOffset= toffset;
	curLeft = printRect.left+leftOffset;

	//the IBM aligns the text on the baseline  ?? center ??
	// the MAC aligns the text on the center ?? baseline 
	TextFontSize(fFontID,fontSize);
	GetFontInfo(&finfo);
	offSetToBaseLine = finfo.ascent;

	curBottom  = printRect.top + offSetToBaseLine + topOffset;
}

void TextRect::PrintLine(char *s,short perCentOffset )
{
	TextFontSize(fFontID,fontSize);
	MyMoveTo(curLeft+(perCentOffset*RectWidth(printRect)/100),curBottom);
	drawstring(s);
	
	FontInfo finfo;
	GetFontInfo(&finfo);
	curBottom+=finfo.ascent+finfo.descent+finfo.leading;// JLM add leading
}

void TextRect::FrameBounds()
{

		Rect r=printRect;

		#ifdef MAC
			SetPenPat(GRAY);
			MyFrameRect(&r);
		#else
			r.right--;
			MyFrameRect(&r);
		#endif
		SetPenPat(BLACK);
}