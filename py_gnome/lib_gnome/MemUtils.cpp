/*
 *  MemUtils.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "Basics.h"
#include "TypeDefs.h"
#include "MemUtils.h"


#ifndef hubris

///// SYSTEM  STUFF /////////////////////////

OSErr DeleteHandleItem(CHARH h, long i, long size)
{
	long total = _GetHandleSize((Handle)h),
	elements = total / size;
	
	_BlockMove(&INDEXH(h, (i + 1) * size),
			   &INDEXH(h, i * size),
			   (elements - (i + 1)) * size);
	
	_SetHandleSize((Handle)h, total - size);
	
	return _MemError();
}

/////////// UNIVERSAL MEMORY UTILS

long _handleCount = 0;
static long masterPointerCount = 0;
static OSErr memoryError = 0;
static Ptr *masterPointers = 0;
static Handle freeMasterPointers[10];



void _MyHLock(Handle h)
{
	_HLock(h);
}

Handle _MySetHandleSize(Handle h, long newSize)
{
	_SetHandleSize(h, newSize);
	
	return h;
}


OSErr _InitAllHandles()
{
	long kNumToAllocate = 6000; // was 1000 , JLM 6/8/10

	if ((masterPointers = (Ptr *)_NewPtr(kNumToAllocate * sizeof(Ptr))) == NULL)
		return -1;

	for (long i = 0; i < kNumToAllocate; i++)
		masterPointers[i] = 0;

	for (long i = 0; i < 10 ;i++)
		freeMasterPointers[i] = &masterPointers[i];

	masterPointerCount = kNumToAllocate;

	return 0;
}

void _DeleteAllHandles()
{
	//delete[] masterPointers;
	_DisposePtr((Ptr)masterPointers);
}

Ptr _NewPtr(long size)
{
	memoryError = 0;
	Ptr p;

	try {
		p = new char[size + sizeof(long)]();
	}
	catch(...) {
		memoryError = -1; 
		return 0; 
	}

	((long *)p)[0] = size;

	p += sizeof(long);

	memset(p, 0, size);

	_handleCount++;
	
	return(p);
}


Ptr _NewPtrClear(long size)
{
	return _NewPtr(size);
}

long _GetPtrSize(Ptr p)
{
	return ((long *)(p - sizeof(long)))[0];
}

// ok this is a bit tricky
// these memory pointers are managed as
// a (sort of)struct of the type
//    struct Data {
//	    long size;
//	    char data[];
//    };
// and this function manages the resizing of
// the data *and* the value which reports the
// size.
//
// Basically any pointer p passed in is assumed to
// be immediately preceded in memory by a long
// value representing the size of allocated memory.
Ptr _SetPtrSize(Ptr p, long newSize)
{
	Ptr p2 = 0;
	memoryError = 0;

	if (p > (Ptr)sizeof(long)) {
		// we have a valid buffer coming in
		try {
			long *currentSize = (long *)(p - sizeof(long));

			p2 = new char[newSize + sizeof(long)]();

			((long *)p2)[0] = newSize;

			if (newSize < currentSize[0])
				memmove(p2 + sizeof(long), p, newSize);
			else
				memmove(p2 + sizeof(long), p, *currentSize);

			// this seems a bit brittle, but...ok.
			p -= sizeof(long);
			delete[] p;
		}
		catch(...) {
			memoryError = -1;
			return p;
		}
	}
	else {
		memoryError = -1;
		return p;
	}

	return (p2 + sizeof(long));
}

void _DisposePtr(Ptr p)
{
	if ((size_t)p > sizeof(long)) {
		p -= sizeof(long);
		delete[] p;
		_handleCount--;
	}
	else {
		printf("_DisposePtr(): Warning: trying to dispose an invalid pointer(%lX)", (size_t)p);
	}
}

void _DisposPtr(Ptr p)
{
	_DisposePtr(p);
}

Handle _NewHandle(long size)
{
	Ptr p;
	Handle h = 0;
	
	// look for a free space
	for (long i = 0; i < 10; i++)
		if (freeMasterPointers[i] != 0) {
			// freeMasterPointers holds the easy places to look
			h = freeMasterPointers[i];
			break;
		} // we found an easy one


	if (!h) {
		// did not find a free place for our handle
		// we need to reset the freeMasterPointers
		for (long i = 0, j = 0 ; i < masterPointerCount ; i++)
			if (masterPointers[i] == 0) {
				// we found a free unused place
				if (!h)
					h = &masterPointers[i];

				if (j < 10)
					freeMasterPointers[j++] = &masterPointers[i];
				else
					break;// we have found all ten
			}

		if (!h) {
			// JLM 6/8/10, the InitWindowsHandles below will loose track of hte previously allocated "Handles".
			// as a work around to help avoid this, I've increase the number allocated in InitWindowsHandles 
			_InitAllHandles();
			h = &masterPointers[0];
			// code goes here, JLM 3/30/99, why was this code was commented out ?
			// Note: This code is still in TAT
			/*
			 masterPointers = (Ptr *)Win_SetPtrSize(masterPointers, (masterPointerCount + 1000) * sizeof(Ptr));
			 if (memoryError) return 0;
			 for (i = 0 ; i < 10 ; i++)
			 freeMasterPointers[i] = &masterPointers[masterPointerCount + i];
			 h = &masterPointers[masterPointerCount];
			 masterPointerCount += 1000;
			 */
		}
	}

	// we have found all ten
	if (!(p = _NewPtr(size)))
		return 0; // unable to allocate

	(*h) = p; // record the pointer in the MAC-like "Handle"

	for (long i = 0; i < 10; i++)
		if (freeMasterPointers[i] == h) {
			freeMasterPointers[i] = 0; // mark this place as no longer free
			break;
		}

	return h;
}


Handle _NewHandleClear(long size)
{
	return _NewHandle(size);
}


Handle _TempNewHandle(long size, LONGPTR err)
{
	Handle h = _NewHandle(size);
	
	(*err) = memoryError;
	
	return h;
}


Handle _RecoverHandle(Ptr p)
{
	long i;
	
	memoryError = 0;
	
	for (i = 0 ; i < masterPointerCount ; i++)
		if (masterPointers[i] == p)
			return &masterPointers[i];
	
	memoryError = -1;
	
	return 0;
}


OSErr _HandToHand(HANDLEPTR hp)
{
	long size = _GetHandleSize(*hp);
	Handle h2;
	
	h2 = _NewHandle(size);
	if (!h2) return -1;
	
	memcpy(*h2, **hp, size);
	// CopyMemory(h2, *hp, size);
	
	*hp = h2;
	
	return 0;
}

void _HLock(Handle h)
{
	return; // all handles are always locked in Windows
}

void _HUnlock(Handle h)
{
	return; // all handles are always locked in Windows
}

void _SetHandleSize(Handle h, long newSize)
{
	(*h) = _SetPtrSize(*h, newSize);
}

long _GetHandleSize(Handle h)
{
	return _GetPtrSize(*h);
}

void _DisposeHandleReally(Handle h)
{
	short i;
	
	_DisposePtr(*h);
	
	*h = 0;
	
	for (i = 0 ; i < 10 ; i++)
		if (freeMasterPointers[i] == 0)
		{ freeMasterPointers[i] = h; break; }
}

void _DisposeHandle2(HANDLEPTR hp)
{
	_DisposeHandleReally(*hp);
	*hp = 0;
}

short _ZeroHandleError(Handle h)
{
	if (h == 0)
		std::cerr << "Dereference of a NULL handle.";
	return 0;
}

void _BlockMove(VOIDPTR sourcePtr, VOIDPTR destPtr, long count)
{
	memmove(destPtr, sourcePtr, count);
}

void _MyBlockMove(VOIDPTR sourcePtr, VOIDPTR destPtr, long count)
{
	memmove(destPtr, sourcePtr, count);
}

#define THE_UPPER_BOUND 2147483647L

long _MaxBlock (void)
// We'll do this until we find a better way,
{
	static long N = THE_UPPER_BOUND/2;
	
	try {
		char *p;
		int m = N;
		p = new char[m];
		delete[] p;
		N = THE_UPPER_BOUND/2;
		return m;
	}
	
	catch(...) {
		N /= 2;
		return _MaxBlock();
	}
	
}

OSErr _MemError (void)
{
	return memoryError;
}

#endif

/*long MyTempMaxMem() {
	return _MaxBlock();
}*/

Handle MyNewHandleTemp(long size)
{
	return _NewHandle(size);
	
}


long GetNumHandleItems(Handle h, long itemSize)
{
	return h ? _GetHandleSize(h)/itemSize : 0;
}

long GetNumDoubleHdlItems(DOUBLEH h)
{
	return GetNumHandleItems((Handle)h,sizeof(double));
}
