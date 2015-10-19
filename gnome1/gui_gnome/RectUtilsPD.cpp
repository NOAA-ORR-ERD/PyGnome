/*
 *  RectUtilsPD.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "RectUtilsPD.h"

/**************************************************************************************************/
void AlignRect (Rect *rectToAlign, Rect *frameRect, short justCode)
/* this subroutine modifies the coordinates of rect-To-Align to justify it horizontally around the
 frame rect. The just-Code parameter can contain teJustCenter, teJustLeft, or teJustRight. */
{
	short	DeltaX, RectWidth;
	
	if (justCode == teJustCenter)
	{
		RectWidth = rectToAlign -> right - rectToAlign -> left;
		DeltaX = rectToAlign -> left  - frameRect -> left +
		frameRect -> right - rectToAlign -> right;
		rectToAlign -> left  = frameRect -> left + (DeltaX / 2);
		rectToAlign -> right = rectToAlign -> left + RectWidth;
	}
	else if (justCode == teJustLeft)
	{
		DeltaX = rectToAlign -> left - frameRect -> left;
		rectToAlign -> left -= DeltaX;
		rectToAlign -> right -= DeltaX;
	}
	else if (justCode == teJustRight)
	{
		DeltaX = frameRect -> right - rectToAlign -> right;
		rectToAlign -> left += DeltaX;
		rectToAlign -> right += DeltaX;
	}
	//	else
	//		Debug ("justCode = %hd\n", justCode);
	
	return;
}