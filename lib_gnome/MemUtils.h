/*
 *  MemUtils.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Basics.h"
#include "TypeDefs.h"

#ifndef __MemUtils__
#define __MemUtils__

#ifndef IBM
#ifndef pyGNOME
#define hubris
#endif
#endif

// If either pyGNOME or IBM is defined:

#ifndef hubris

OSErr _InitAllHandles();
void _DeleteAllHandles();

Ptr _NewPtr(long size);
Ptr _NewPtrClear(long size);
long _GetPtrSize(Ptr p);
Ptr _SetPtrSize(Ptr p, long newSize);
//void DisposPtr(Ptr p);
void _DisposePtr(Ptr p);

Handle _NewHandle(long size);
Handle _NewHandleClear(long size);
Handle _TempNewHandle(long size, LONGPTR err);
OSErr _HandToHand(HANDLEPTR hp);
void _HLock(Handle h);
void _HUnlock(Handle h);
long _GetHandleSize(Handle h);
void _SetHandleSize(Handle h, long newSize);

//Handle RecoverHandle(Ptr p);

void _MyBlockMove(VOIDPTR sourcePtr, VOIDPTR destPtr, long count);
void _BlockMove(VOIDPTR sourcePtr, VOIDPTR destPtr, long count);
long _MaxBlock(void);
OSErr _MemError(void);

/// IF DEBUGGING...
// #define DisposeHandle(p) _DisposeHandle2((&(p)))
// void _DisposeHandle2(HANDLEPTR hp);
// #define INDEXH(h, i) ((h)[i + _ZeroHandleError(h)])
// short _ZeroHandleError(Handle h);

/// ELSE...
//#define DisposHandle(p) _DisposeHandleReally(p)
void _DisposeHandleReally(Handle p);
// #define INDEXH(h, i) ((h)[i])
#define DisposeHandle(p) _DisposeHandleReally(p)
#define _DisposeHandle(p) _DisposeHandleReally(p)

#else
#define _InitAllHandles InitAllHandles
#define _DeleteAllHandles DeleteAllHandles
#define _NewPtr NewPtr
#define _NewPtrClear NewPtrClear
#define _GetPtrSize GetPtrSize
#define _SetPtrSize SetPtrSize
#define _DisposePtr DisposePtr
#define _DisposPtr DisposPtr
#define _NewHandle NewHandle
#define _NewHandleClear NewHandleClear
#define _TempNewHandle TempNewHandle
#define _HandToHand HandToHand
#define _HLock HLock
#define _HUnlock HUnlock
#define _GetHandleSize GetHandleSize
#define _SetHandleSize SetHandleSize
#define _MyBlockMove BlockMove
#define _BlockMove BlockMove
#define _MaxBlock MaxBlock
#define _MemError MemError
#define _DisposeHandleReally DisposeHandleReally
#define _DisposeHandle DisposeHandle
#define _MyHLock MyHLock
#define _MySetHandleSize MySetHandleSize
#define _RecoverHandle RecoverHandle
#define _ZeroHandleError ZeroHandleError
#endif

long GetNumDoubleHdlItems(DOUBLEH h);
long GetNumHandleItems(Handle h, long itemSize);

#endif
