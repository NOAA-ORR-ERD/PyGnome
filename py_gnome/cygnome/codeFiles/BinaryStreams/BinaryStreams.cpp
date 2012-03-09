/*
 *  BinaryStreams.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/6/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Earl.h"
#include "TypeDefs.h"
#include "CROSS.H"
#include <iostream>
#include <fstream>

using std::ios;

// For what, I am not sure:

static char readWriteProc[256];

void StartReadWriteSequence(char *procName)
{
	strcpy(readWriteProc, procName);
}

char *SwapN(char *s, short n)
{
	char c;
	short i;
	
	for (i = 0 ; i < n / 2 ; i++) {
		c = s[i];
		s[i] = s[n - (i + 1)];
		s[n - (i + 1)] = c;
	}
	
	return s;
}


//++ Read / Write Functions

OSErr openPath(CHARPTR path, BFPBP bfpb) {
	
	int err = 0;
	
	try {
		bfpb = new fstream(path, ios::binary | ios::in | ios::out );
	}

	catch (...) {
		return !err;
	}
	
	return err;
}


OSErr WriteValue(BFPB *bfpb, char *data, short length, Boolean doSwap)
{
	int err = 0;
	
#ifdef SWAP_BINARY
	//#ifdef IBM
	if (doSwap) SwapN(data, length);	//????!?!?!??!?!?!
#endif
	
	try {
		bfpb->write(data, length);
	}
	catch(...) {
		return !err;
	}
	
	return err;
}

OSErr ReadValue(BFPB *bfpb, char *data, short length, Boolean doSwap)
{
	int err = 0;
	try {
		bfpb->read(data, length);
	}
	catch(...) {
		return !err;
	}
#ifdef SWAP_BINARY
	//#ifdef IBM
	if (doSwap) SwapN(data, length);
#endif
	
	return err;
}

OSErr ReadMacValue(BFPB *bfpb, Seconds* val)
{
	return ReadValue(bfpb,(char*)val,sizeof(*val),TRUE);
}

OSErr ReadMacValue(BFPB *bfpb, char* val)
{
	return ReadValue(bfpb,val,sizeof(*val),FALSE);
}

OSErr ReadMacValue(BFPB *bfpb, long* val)
{
	return ReadValue(bfpb,(char*)val,sizeof(*val),TRUE);
}

OSErr ReadMacValue(BFPB *bfpb, short* val)
{
	return ReadValue(bfpb,(char*)val,sizeof(*val),TRUE);
}

OSErr ReadMacValue(BFPB *bfpb, float* val)
{
	return ReadValue(bfpb,(char*)val,sizeof(*val),TRUE);
}

OSErr ReadMacValue(BFPB *bfpb, double* val)
{
	return ReadValue(bfpb,(char*)val,sizeof(*val),TRUE);
}

OSErr ReadMacValue(BFPB *bfpb, Boolean* val)
{
	char c = 0;
	OSErr err = ReadValue(bfpb,&c,sizeof(c),FALSE);
	*val= c;
	return err;
}


OSErr ReadMacValue(BFPB *bfpb, UNIQUEID* val)
{
	OSErr err = ReadMacValue(bfpb,&val->ticksAtCreation);
	if(!err) err = ReadMacValue(bfpb,&val->counter);
	return err;
}

OSErr ReadMacValue(BFPB *bfpb, LongPoint *lp)
{
	OSErr err = 0;
	if (err = ReadMacValue(bfpb,&(lp->h))) return err;
	if (err = ReadMacValue(bfpb,&(lp->v))) return err;
	
	return 0;
}

OSErr ReadMacValue(BFPB *bfpb, WorldRect *wRect)
{
	OSErr err = 0;
	if (err = ReadMacValue(bfpb,&(wRect->hiLat))) return err;
	if (err = ReadMacValue(bfpb,&(wRect->loLat))) return err;
	if (err = ReadMacValue(bfpb,&(wRect->hiLong))) return err;
	if (err = ReadMacValue(bfpb,&(wRect->loLong))) return err;
	
	return 0;
}

OSErr ReadMacValue(BFPB *bfpb, Rect *theRect)
{
	OSErr err = 0;
	if (err = ReadMacValue(bfpb,&(theRect->left))) return err;
	if (err = ReadMacValue(bfpb,&(theRect->top))) return err;
	if (err = ReadMacValue(bfpb,&(theRect->right))) return err;
	if (err = ReadMacValue(bfpb,&(theRect->bottom))) return err;
	
	return 0;
}

OSErr ReadMacValue(BFPB *bfpb, WorldPoint *wp)
{
	OSErr err = 0;
	if (err = ReadMacValue(bfpb,&(wp->pLong))) return err;
	if (err = ReadMacValue(bfpb,&(wp->pLat))) return err;
	
	return 0;
}

OSErr ReadMacValue(BFPB *bfpb, char* str, long len)
{
	return ReadValue(bfpb,str,len,FALSE);
}


OSErr WriteMacValue(BFPB *bfpb, Seconds val)
{
	return WriteValue(bfpb,(char*)&val,sizeof(val),TRUE);
}

OSErr WriteMacValue(BFPB *bfpb, char val)
{
	return WriteValue(bfpb,&val,sizeof(val),FALSE);
}

OSErr WriteMacValue(BFPB *bfpb, long val)
{
	return WriteValue(bfpb,(char*)&val,sizeof(val),TRUE);
}

OSErr WriteMacValue(BFPB *bfpb, short val)
{
	return WriteValue(bfpb,(char*)&val,sizeof(val),TRUE);
}

OSErr WriteMacValue(BFPB *bfpb, float val)
{
	return WriteValue(bfpb,(char*)&val,sizeof(val),TRUE);
}

OSErr WriteMacValue(BFPB *bfpb, double val)
{
	return WriteValue(bfpb,(char*)&val,sizeof(val),TRUE);
}

OSErr WriteMacValue(BFPB *bfpb, Boolean val)
{
	char c = val;
	OSErr err = WriteValue(bfpb,&c,sizeof(c),FALSE);
	return err;
}

OSErr WriteMacValue(BFPB *bfpb, UNIQUEID val)
{
	OSErr err = WriteMacValue(bfpb,val.ticksAtCreation);
	if(!err) err = WriteMacValue(bfpb,val.counter);
	return err;
}

OSErr WriteMacValue(BFPB *bfpb, LongPoint lp)
{
	OSErr err = 0;
	if (err = WriteMacValue(bfpb,lp.h)) return err;
	if (err = WriteMacValue(bfpb,lp.v)) return err;
	
	return 0;
}
OSErr WriteMacValue(BFPB *bfpb, WorldRect wRect)
{
	OSErr err = 0;
	if (err = WriteMacValue(bfpb,wRect.hiLat)) return err;
	if (err = WriteMacValue(bfpb,wRect.loLat)) return err;
	if (err = WriteMacValue(bfpb,wRect.hiLong)) return err;
	if (err = WriteMacValue(bfpb,wRect.loLong)) return err;
	
	return 0;
}

OSErr WriteMacValue(BFPB *bfpb, Rect theRect)
{
	OSErr err = 0;
	if (err = WriteMacValue(bfpb,theRect.left)) return err;
	if (err = WriteMacValue(bfpb,theRect.top)) return err;
	if (err = WriteMacValue(bfpb,theRect.right)) return err;
	if (err = WriteMacValue(bfpb,theRect.bottom)) return err;
	
	return 0;
}

OSErr WriteMacValue(BFPB *bfpb, WorldPoint wp)
{
	OSErr err = 0;
	if (err = WriteMacValue(bfpb,wp.pLong)) return err;
	if (err = WriteMacValue(bfpb,wp.pLat)) return err;
	
	return 0;
}

OSErr WriteMacValue(BFPB *bfpb, char* str, long len)
{
	return WriteValue(bfpb,str,len,FALSE);
}

