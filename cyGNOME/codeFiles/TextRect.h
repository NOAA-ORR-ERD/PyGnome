#ifndef __TEXTRECT__
#define __TEXTRECT__

#ifndef __CROSS__
#include "Cross.h"
#endif

class TextRect
{
	protected:
		Rect printRect;
		short fFontID;
		short fontSize;
		short curLeft;
		short curBottom;
		short leftOffset;
		short topOffset;
		Boolean frameRect;
		
	public:
		TextRect(Rect r,short fontID,short FontSize,short loffset,short toffset);
		void PrintLine(char *s,short percentOffset=0);
		void FrameBounds();
};


#endif
