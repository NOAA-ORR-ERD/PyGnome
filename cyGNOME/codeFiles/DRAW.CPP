
#include "CROSS.h"
#include "GenDefs.h"
#include "ObjectUtils.h"
#include "GridVel.h"

void DrawArrowHead (Point p1, Point p2, VelocityRec velocity);
void MyDrawArrow(short x0, short y0, short x1, short y1);
////////////////////////////////////////////////////////////////////////////////

void PrepareToDraw(Rect r, WorldRect view, short dX, short dY)
{
	long i;
	float ratio;

	AX = view.loLong;
	BX = view.hiLong;
	aX = r.left;
	bX = r.right;

	AY = view.loLat;
	BY = view.hiLat;
	aY = r.top;
	bY = r.bottom;

	// force (bX - aX) and (bY - aY) to be a power of 2, and scale BX and BY accordingly

	for (i = 2 ; i < (bX - aX) ; i = i << 1) ;
	ratio = (float)i / (float)(bX - aX);
	bX = aX + i;
	BX = AX + (BX - AX) * ratio;

	for (i = 2 ; i < (bY - aY); i = i << 1) ;
	ratio = (float)i / (float)(bY - aY);
	bY = aY + i;
	BY = AY + (BY - AY) * ratio;

	DX = dX;
	DY = dY;
}

////////////////////////////////////////////////////////////////////////////////

#define ARROWSIZE 8

void DrawArrowHead(Point to, VelocityRec velocity)
{
	double angle;
	Rect r;
	
	MySetRect(&r, to.h - ARROWSIZE, to.v - ARROWSIZE, to.h + ARROWSIZE, to.v + ARROWSIZE);
	angle = atan2(-velocity.v, velocity.u);
	PaintArc(&r, (angle * 180) / PI - (20 + 90), 40);
}

OSErr DrawMaps(WindowPtr w, Rect r, WorldRect view, Boolean printing)
{
	long i;
	
	OSErr err = 0;
	sharedPrinting = printing;
	sharedView = view;
	PrepareToDraw(r, view, 0, 0);
	
	if (model)
		model->Draw(r, view);
	
	sharedPrinting = FALSE;
		
	return err;
}

void DrawMarkers(Boolean invalidate)
{
#pragma unused(invalidate)
}

////////////////////////////////////////////////////////////////////////////////

/******** UNUSED CODE ************
/////////////////////////////////////////////////
/////////////////////////////////////////////////
Boolean SaveScreenImage(MyGWorldRec	frameGWRec)
{
	Rect r = MapDrawingRect();
	Handle h;
	
	// dispose of old screen images if any
	if (screenImage) { BitMapDispose(screenImage, mapWindow); screenImage = nil; }

	if (WindowNeedsUpdating(mapWindow)) return FALSE;
	if (RectOffScreen(mapWindow, r) || WindowObscured(mapWindow)) return FALSE;
	if (settings.inBackground) return FALSE;
	if (mapWindow != FrontWindow()) return FALSE;
	
#ifdef MAC
	screenImage = PixMapCreate(frameGWRec.sourcePort, &r, &h, &err);
	if (!screenImage) return FALSE;
	CopyBitsFromWindow(mapWindow, screenImage, r, r);
#else
	CopyBitsFromWindowB(mapWindow, &screenImage, r, r);
#endif
	screenImageRect = r; // JLM 10/13/98
	return TRUE;
}

Boolean SaveScreenImageFromGWorld (CGrafPtr mapGWorld)
{
	Rect r = MapDrawingRect();
	Handle h;
	
	// dispose of old screen images if any
	if (screenImage) { BitMapDispose(screenImage, mapWindow); screenImage = nil; }

	if (WindowNeedsUpdating(mapWindow)) return FALSE;
	if (RectOffScreen(mapWindow, r) || WindowObscured(mapWindow)) return FALSE;
	if (settings.inBackground) return FALSE;
	if (mapWindow != FrontWindow()) return FALSE;
	
#ifdef MAC
	screenImage = PixMapCreate(mapGWorld, &r, &h, &err);
	if (!screenImage) return FALSE;
	CopyBitsFromWindow(mapWindow, screenImage, r, r);
#else
	CopyBitsFromWindowB(mapWindow, &screenImage, r, r);
#endif
	screenImageRect = r; // JLM 10/13/98
	return TRUE;
}

////////////////////////////////////////////////////////////////////////////////

Boolean RestoreScreenImage()
{	// returns FALSE if it did not restore the image
	Rect r = MapDrawingRect();

	if(!EqualRects(r,screenImageRect)) return FALSE; // JLM 10/13/98
#ifdef MAC
	//return TRUE;
	if (screenImage)
		CopyBitsToWindow(screenImage, mapWindow, r, r);
#else
	CopyBitsToWindowB(screenImage, mapWindow, r, r);
#endif
	// DoneDrawingToScreen(saveClip);
	
	return TRUE;
}

////////////////////////////////////////////////////////////////////////////////

Boolean MakeScreenImageFromGWorld(CGrafPtr sourceGWorld)
{
	Rect r = MapDrawingRect();
	Handle h;

	InvalidateMapImage()	; // dispose of old screen images if any
	if (WindowNeedsUpdating(mapWindow)) return FALSE;
	if (RectOffScreen(mapWindow, r) || WindowObscured(mapWindow)) return FALSE;
	if (settings.inBackground) return FALSE;
	if (mapWindow != FrontWindow()) return FALSE;
	
#ifdef MAC
	if (!sourceGWorld) return FALSE;
	screenImage = PixMapCreate((CGrafPort *)mapWindow, &r, &h, &err);
	if (!screenImage) return FALSE;
	CopyBitsFromGWorld(sourceGWorld, screenImage, r, r);
#else
	// code goes here STH
#endif
	screenImageRect = r; // JLM 10/13/98
	return TRUE;
}

void InvalidateScreenImage()
{
	SetRect(&screenImageRect,0,0,0,0);
#ifdef MAC
	if (screenImage) { BitMapDispose(screenImage, mapWindow); screenImage = nil; }
#else
	if (screenImage) { BitMapDisposeB(screenImage, mapWindow); screenImage = nil; }
#endif
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////
***END UNUSED CODE ********/

void InvalidateMapImage()
{
	MySetRect(&mapImageRect,0,0,0,0);
#ifdef MAC
	if (model -> mapImage)		// test line/body STH
	{
		KillGWorld (model -> mapImage);
		model -> mapImage = nil;
	}
#else
	if (model -> mapImage)		// test line/body STH
	{
		DestroyDIB(model -> mapImage);
		model -> mapImage = nil;
	}
#endif
}

///////////////////////

/*void Swap2IfDefIBM(char *ptr)
{
	#ifdef SWAP_BINARY  
		char	ch;
		ch = ptr[0];
		ptr[0] = ptr[1];
		ptr[1] = ch;
	#else
		#pragma unused(ptr)
	#endif
}

//////////////////
void SwapShortIfDefIBM(short* p)
{
	if(sizeof(*p) == 2) 
		Swap2IfDefIBM((char*)p);
}*/

/////////////////
////////////////////////////////////////////////////////////////////////////////

/*Rect GetPicHandleFrame(PicHandle pic)
{	
	Rect frame; 
	#ifdef _MACINTOSH
		// on the intel mac, we need to swap the binary
		frame = (**pic).picFrame;
		SwapShortIfDefIBM(&frame.left);
		SwapShortIfDefIBM(&frame.right);
		SwapShortIfDefIBM(&frame.top);
		SwapShortIfDefIBM(&frame.bottom);
	#else
	    PicPtr picPtr = (PicPtr)GlobalLock((HANDLE)pic);
		frame = picPtr->picFrame;
		(void)GlobalUnlock((HANDLE)pic);
	#endif
	return frame;
}*/


/////////////////////////////////////////////////
void DrawTriangle(Rect box, short bullet)
{
#ifdef MAC
	Rect frame;
	PicHandle p;

	p = GetPicture(bullet == BULLET_OPENTRIANGLE ? OPENLEPICT : CLOSEDLEPICT);
	if (!p) { SysBeep(1); return; }

	//frame = (**p).picFrame;
	frame = GetPICTFrame(p);
	MyOffsetRect(&frame, box.left - 1, box.top - 2);
	if (bullet == BULLET_OPENTRIANGLE)
		MyOffsetRect(&frame, 0, 2);
	else
		MyOffsetRect(&frame, 2, 0);
	DrawPicture(p, &frame);
#else
	POINT points[3];

	if (bullet == BULLET_OPENTRIANGLE) {
		points[0] = MakePOINT(box.left - 1, (box.top + box.bottom) / 2 - 1);
		points[1] = MakePOINT(box.right + 1, (box.top + box.bottom) / 2 - 1);
		points[2] = MakePOINT((box.right + box.left) / 2, box.bottom - 1);
		Polygon(currentHDC, points, 3);
	}
	else {
		MyMoveTo(box.left, box.top - 2);
		MyLineTo(box.right - 1, (box.top + box.bottom) / 2);
		MyLineTo(box.left, box.bottom + 1);
		MyLineTo(box.left, box.top - 2);
	}
#endif
}

////////////////////////////////////////////////////////////////////////////////
void DrawTimeTriangle(short h)
{
	// don't draw points in the lower corners
	Rect triRect = TimeTriangleRect(h);
	short midH = (triRect.right + triRect.left)/2;
	#ifdef MAC
		short rightStop = triRect.right-2;
		short bottomStop = triRect.bottom-2;
	#else //IBM
		short rightStop = triRect.right-1;
		short bottomStop = triRect.bottom-1;
	#endif
	//
	// bottom line
	MyMoveTo (triRect.left+1,triRect.bottom-1);
	MyLineTo (rightStop,triRect.bottom-1);
	//upper part
	MyMoveTo (triRect.left+1,triRect.bottom-2);
	MyLineTo (midH,triRect.top);
	MyLineTo (rightStop,bottomStop);
}

float DrawTimeBarTriangle(short h, Boolean bErasePrevious)
// returns the ratio corresponding to the last triangle X-pos
{
	Rect 			r = TimeLineRect();
	float			ratio = 0;
	static short	sPrevH =-100;

	if (h > r.right)
		h = r.right;
	if (h < r.left + 1)
		h = r.left + 1;
	
	if (bErasePrevious)
	{
		Rect rect;
		RGBForeColor(&colors[LIGHTGRAY]);
		DrawTimeTriangle(sPrevH);
		RGBForeColor(&colors[BLACK]);
	}
	
	{
		//PenState	penStatus;
		//GetPenState (&penStatus);	// save original pen state / mode
		//PenMode (srcXor);			// set mode to xor
//		PenPat((ConstPatternParam)&qd.gray);
		DrawTimeTriangle(h);
		
		//SetPenState (&penStatus);	// restore original pen state / mode
		
		// save last one used
		sPrevH = h;
	}
		
	// calculate and return ratio
	ratio = (float) (h - r.left) / (float) (r.right - r.left);

	return ratio;
}

short DrawBullet(ListItem item, RECTPTR r, Boolean draw, RECTPTR box)
{
	Rect b;
	Rect frame;
	PicHandle p = nil;
	
#ifdef MAC
	MySetRect(&b, 5 + item.indent * 12,
				  r->top + 4,
				  5 + item.indent * 12 + 6,
				  r->bottom - 4);
#else
	MySetRect(&b, 7 + item.indent * 12,
				  r->top + 4, 
				  7 + item.indent * 12 + 6,
				  r->bottom - 5);  
#endif

	if (draw) {
		PenNormal();
		RGBForeColor(&colors[BLACK]);
		switch (item.bullet) {
			case BULLET_DASH:
				MyMoveTo(b.left, (b.top + b.bottom) / 2);
				MyLineTo(b.right, (b.top + b.bottom) / 2);
				break;
			case BULLET_EMPTYBOX:
				b.left+=1;b.right+=1;
#ifdef MAC
				p = GetPicture (EMPTYBOXPICT);
				if (!p) { SysBeep(1);break; }

				//frame = (**p).picFrame;
				frame = GetPICTFrame(p);
				MyOffsetRect(&frame, b.left, b.top);
				DrawPicture(p, &frame);
#else
				MyFrameRect(&b);
#endif
				b.left-=1;b.right-=1;
				break;
			case BULLET_FILLEDBOX:
				b.left+=1;b.right+=1;
#ifdef MAC
				p = GetPicture (FILLEDBOXPICT);
				if (!p) { SysBeep(1);break; }

				//frame = (**p).picFrame;
				frame = GetPICTFrame(p);
				MyOffsetRect(&frame, b.left, b.top);
				DrawPicture(p, &frame);
#else
				PaintRect(&b);
#endif
				
			//	MyFrameRect(&b);
			//	MyMoveTo(b.left,b.top);
			//	MyLineTo(b.right-1,b.bottom-1);
			//	MyMoveTo(b.right-1,b.top);
			//	MyLineTo(b.left,b.bottom-1);
				
				b.left-=1;b.right-=1;
				break;
			case BULLET_OPENTRIANGLE:
			case BULLET_CLOSEDTRIANGLE:
				DrawTriangle(b, item.bullet);
				break;
		}
	}

	if (box)
		(*box) = b;
		
	// line up the text with an indented bullet
	b.right += 3;	// mac tweek by sohail
	
	return b.right;
}

void DrawObjectsList(DialogPtr w, RECTPTR r, long n)
// called once for each item in VList to draw nth line
{
	char s[256];
	short right, style = normal;
	Point p;
	ListItem item = model->GetNthListItem(n, 0, &style, s);
	
	if (item.owner == 0) return;
	
	GetPen(&p);	
	PenNormal();
	
	right = DrawBullet(item, r, FALSE, 0);
	
	MyMoveTo(right + 4, p.v - OBJECTLINEOFFSET);

	// set text-face according to style requested
	TextFontSizeFace(kFontIDGeneva,LISTTEXTSIZE,style);
	drawstring(s);		
	DrawBullet(item, r, TRUE, 0);
	TextFontSizeFace(kFontIDGeneva,LISTTEXTSIZE,normal);
}

void DrawObjectsList2(WindowRef w, RECTPTR r, long n)
// called once for each item in VList to draw nth line
{
	char s[256];
	short right, style = normal;
	Point p;
	ListItem item = model->GetNthListItem(n, 0, &style, s);
	
	if (item.owner == 0) return;
	
	GetPen(&p);	
	PenNormal();
	
	right = DrawBullet(item, r, FALSE, 0);
	
	MyMoveTo(right + 4, p.v - OBJECTLINEOFFSET);

	// set text-face according to style requested
	TextFontSizeFace(kFontIDGeneva,LISTTEXTSIZE,style);
	drawstring(s);		
	DrawBullet(item, r, TRUE, 0);
	TextFontSizeFace(kFontIDGeneva,LISTTEXTSIZE,normal);
}

void HiliteObjectsRect(DialogPtr w, RECTPTR r, long n)
// limits the horizontal selection based on text string width
{
	char s[256];
	short left, right, style = normal;
	ListItem item = model->GetNthListItem(n, 0, &style, s);
	
	if (item.owner == 0) return;
	
	TextFontSizeFace(kFontIDGeneva,LISTTEXTSIZE,style);
	
	right = DrawBullet(item, r, FALSE, 0);
	
	left = right + 3;
	if (left > r->left) r->left = left;
	right = r->left + stringwidth(s) + 3;
	if (right < r->right) r->right = right;		// selects only upto right edge of text

	TextFontSizeFace(kFontIDGeneva,LISTTEXTSIZE,normal);
}
void MyDrawArrow(short x0, short y0, short x1, short y1)
{
	double headLen,headWidth,dh,dv,dhB,dvB,h,v,len;
	


	if(x0 != x1 || y0 != y1)
	{
		MyMoveTo(x0,y0);MyLineTo(x1,y1);
		dh = x1 - x0;
		dv = y1 - y0;
		len = sqrt(dh*dh+dv*dv);
		headLen= len/3;
		headLen = headLen > 10 ? 10/len: headLen/len;
		headWidth = headLen/3;
		dhB = headWidth * dh;
		dvB = headWidth * dv;
		h = x1 - headLen * dh;
		v = y1 - headLen * dv;
		MyMoveTo(x1,y1); MyLineTo(round(h - dvB), round( v + dhB));
		MyMoveTo(x1,y1); MyLineTo(round( h + dvB), round( v - dhB));
	}
	else
	{
	#ifdef IBM
		MyMoveTo(x0,y0);MyLineTo(x0+1,y0+1); // JLM
	#else
		MyMoveTo(x0,y0);MyLineTo(x0,y0);
	#endif
	}

}
void DrawArrowHead (Point p1, Point p2, VelocityRec velocity)
{
#define				kArrowHang 150 * PI / 180
#define				kHeadRatio 0.25	

	double		angle1, angle2, cs, sn, result, angle, x;
	
	velocity = velocity;
	velocity.v = -velocity.v;	// compensate for top-to-bottom screen pixels
	
	if (velocity.u == 0)
	{
		if (velocity.v == 0)
			angle = 0;
		else if (velocity.v > 0)
			angle = 90 * PI / 180;
		else
			angle = 270 * PI / 180;
	}
	else
	{
		angle = atan (velocity.v / velocity.u);
		if (angle < 0)
			angle += 360 * PI / 180;
		if (velocity.u < 0)
			angle += 180 * PI / 180;
	}

	x =   ((p2.h - p1.h) * (p2.h - p1.h) +
				   (p2.v - p1.v) * (p2.v - p1.v));

	result = sqrt (x);
										// now draw the arrow head wings */
	angle1 = angle - kArrowHang;				
	cs = cos (angle1);					// calculate the horizontal component */
	sn = sin (angle1);					// calculate the vertical 	component */
	
	p1.h = cs * (result * kHeadRatio) + p2.h;
	p1.v = sn * (result * kHeadRatio) + p2.v;

	MyMoveTo (p2.h, p2.v);				// move pen back to head */
	MyLineTo (p1.h, p1.v);					// line to the end of wing #1 */
	
	angle2 = angle + kArrowHang;
	cs = cos (angle2);					// calculate the horizontal component */
	sn = sin (angle2);					// calculate the vertical 	component */
	
	p1.h = cs * (result * kHeadRatio) + p2.h;
	p1.v = sn * (result * kHeadRatio) + p2.v;

										// move pen back to head origin */
	MyMoveTo (p2.h, p2.v);				// move pen back to head */
	MyLineTo (p1.h, p1.v);				// line to the end of wing #2 */

	return;
}
