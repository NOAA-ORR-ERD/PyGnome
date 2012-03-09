#include "MemUtils/MemUtils.h"

#include "CROSS.H"
#include "CMyList.h"

///////////////////////////////////////////////////////////////////////////
CMyList::CMyList(long elemSize)
{
	n = 0;
	L = 0;
	elementSize = elemSize;
	iteratorIndex = 0;
	iterIncrement = 1;
}

OSErr CMyList::IList()
{
	L = (CHARH)_NewHandleClear(0);
	
	return L ? 0 : -1;
}

void CMyList::Dispose()
{
	if (L)	{
		DisposeHandle((Handle)L);
		L = 0;
	}
}

Handle CMyList::GetListDataHdl()
{
	return (Handle)L;
}

void CMyList::SetListDataHdl(Handle newListHdl, long newItemCount)
{
	Dispose();
	L = (CHARH)newListHdl;
	n = newItemCount;
}

long CMyList::GetItemSize()
{
	return elementSize;
}

void CMyList::ClearList()
{
	n = 0;
	
	if (L)
		_SetHandleSize((Handle)L, 0);
}

OSErr CMyList::InsertAtIndex(Ptr itemPtr, long i)
{
	if (!L) return -1;
	
	if (i == -1) i = n;
	
	if (i < 0 || i > n) return -1; // out of range
	
	_SetHandleSize((Handle)L, (n + 1) * elementSize);
	if (_MemError()) return -1;
	
	if (i < n)
		_BlockMove(ITEM_PTR(i), ITEM_PTR(i + 1), (n - i) * elementSize);
	
	_BlockMove(itemPtr, ITEM_PTR(i), elementSize);
	
	n++;
	
	return 0;
}

OSErr CMyList::AppendItem(Ptr itemPtr)
{
	return InsertAtIndex(itemPtr, n);
}

OSErr CMyList::DeleteItem(long i)
{
	long numElementsToMove;
	if (!L) return -1;
	
	if (i == -1) i = n - 1;
	
	if (i < 0 || i >= n) return -1; // out of range
	
	//_BlockMove(ITEM_PTR(i + 1), ITEM_PTR(i), (n - i) * elementSize);
	// JLM, 1/20/99, this was moving one too many elements
	numElementsToMove = n - i - 1;
	if(numElementsToMove > 0) _BlockMove(ITEM_PTR(i + 1), ITEM_PTR(i), numElementsToMove * elementSize);
	
	n--;
	_SetHandleSize((Handle)L, n * elementSize);
	if (_MemError()) return -1;
	
	return 0;
}

OSErr CMyList::SetListItem(Ptr itemPtr, long i)
{
	if (!L) return -1;
	
	if (i < 0 || i >= n) return -1; // out of range
	
	_BlockMove(itemPtr, ITEM_PTR(i), elementSize);
	
	return 0;
}

OSErr CMyList::GetListItem(Ptr itemPtr, long i)
{
	if (!L) return -1;
	
	if (i < 0 || i >= n) return -1; // out of range
	
	_BlockMove(ITEM_PTR(i), itemPtr, elementSize);
	
	return 0;
}

OSErr CMyList::SwapItems(long iA, long iB)
{
	char *p, *q, c;
	long i;
	
	if (!L) return -1;
	
	if (iA < 0 || iA >= n) return -1; // out of range
	if (iB < 0 || iB >= n) return -1; // out of range
	
	p = ITEM_PTR(iA);
	q = ITEM_PTR(iB);
	
	for (i = 0 ; i < elementSize ; i++) {
		c = p[i];
		p[i] = q[i];
		q[i] = c;
	}
	
	return 0;
}

OSErr CMyList::MoveToTop(long i)
{
	return SwapItems(i, 0);
}

OSErr CMyList::MoveToBottom(long i)
{
	return SwapItems(i, n - 1);
}

long CMyList::GetItemCount()
{
	return n;
}

Boolean CMyList::IsItemInList(Ptr itemPtr, long *itemIndexPtr)
{
	long i;
	
	(*itemIndexPtr) = 0;
	
	if (!L) return FALSE;
	
	for (i = 0 ; i < n ; i++) {
		if (!memcmp(itemPtr, ITEM_PTR(i), elementSize)) {
			(*itemIndexPtr) = i;
			break;
		}
	}
	
	return i < n;
}

OSErr CMyList::GetFirstItem(Ptr itemPtr)
{
	iteratorIndex = 0;
	iterIncrement = 0;
	
	return GetListItem(itemPtr, iteratorIndex);
}

OSErr CMyList::GetLastItem(Ptr itemPtr)
{
	iteratorIndex = n - 1;
	iterIncrement = -1;
	
	return GetListItem(itemPtr, iteratorIndex);
}

OSErr CMyList::GetNextItem(Ptr itemPtr)
{
	iteratorIndex += iterIncrement;
	
	return GetListItem(itemPtr, iteratorIndex);
}

long CMyList::GetIteratorIndex()
{
	return iteratorIndex;
}

OSErr CMyList::InsertSorted(Ptr itemPtr,long *itemnum,Boolean allowDuplicates)
{
	return 0;
}

///////////////////////////////////////////////////////////////////////////////



OSErr CTimeValuePairList::InsertSorted(Ptr p,long *itemnum,Boolean allowDuplicates)
{
	long i;
	
	if (!L) return -1;
		
	_SetHandleSize((Handle)L, (n + 1) * elementSize);
	if (_MemError()) return -1;
	if(n==0 || ((*(TimeValuePairH)L)[n-1].time < ((TimeValuePair *)p)->time))
	{
		AppendItem(p);
		*itemnum=n-1;
		return 0;
	}
	else if(!allowDuplicates && ((*(TimeValuePairH)L)[n-1].time == ((TimeValuePair *)p)->time))
	{
		*itemnum=n-1;
		return -2;
	}
	else
	{
		for(i=n-1;i>=0;i--)
		{
			if(!allowDuplicates)
			{
				if((*(TimeValuePairH)L)[i].time == ((TimeValuePair *)p)->time)
				{
					*itemnum = i;
					return -2;
				}
			}
			if((*(TimeValuePairH)L)[i].time <= ((TimeValuePair *)p)->time)break;
		}
		i++;
		_BlockMove(ITEM_PTR(i), ITEM_PTR(i + 1), (n - i) * elementSize);
		_BlockMove(p, ITEM_PTR(i), elementSize);
		n++;
		*itemnum=i;
		return 0;
	}
}

/////////////////////////////////////////////////
OSErr CProfilesList::InsertSorted(Ptr p,long *itemnum,Boolean allowDuplicates)
{
	// should be able to create a template
	long i;
	
	if (!L) return -1;
		
	_SetHandleSize((Handle)L, (n + 1) * elementSize);
	if (_MemError()) return -1;
	if(n==0 || ((*(DepthValuesSetH)L)[n-1].depth < ((DepthValuesSet *)p)->depth))
	{
		AppendItem(p);
		*itemnum=n-1;
		return 0;
	}
	else if(!allowDuplicates && ((*(DepthValuesSetH)L)[n-1].depth == ((DepthValuesSet *)p)->depth))
	{
		*itemnum=n-1;
		return -2;
	}
	else
	{
		for(i=n-1;i>=0;i--)
		{
			if(!allowDuplicates)
			{
				if((*(DepthValuesSetH)L)[i].depth == ((DepthValuesSet *)p)->depth)
				{
					*itemnum = i;
					return -2;
				}
			}
			if((*(DepthValuesSetH)L)[i].depth <= ((DepthValuesSet *)p)->depth)break;
		}
		i++;
		_BlockMove(ITEM_PTR(i), ITEM_PTR(i + 1), (n - i) * elementSize);
		_BlockMove(p, ITEM_PTR(i), elementSize);
		n++;
		*itemnum=i;
		return 0;
	}
}/////////////////////////////////////////////////
OSErr CDischargeTimes::InsertSorted(Ptr p,long *itemnum,Boolean allowDuplicates)
{
	// should be able to create a template
	long i;
	
	if (!L) return -1;
		
	_SetHandleSize((Handle)L, (n + 1) * elementSize);
	if (_MemError()) return -1;
	if(n==0 || ((*(DischargeDataH)L)[n-1].time < ((DischargeData *)p)->time))
	{
		AppendItem(p);
		*itemnum=n-1;
		return 0;
	}
	else if(!allowDuplicates && ((*(DischargeDataH)L)[n-1].time == ((DischargeData *)p)->time))
	{
		*itemnum=n-1;
		return -2;
	}
	else
	{
		for(i=n-1;i>=0;i--)
		{
			if(!allowDuplicates)
			{
				if((*(DischargeDataH)L)[i].time == ((DischargeData *)p)->time)
				{
					*itemnum = i;
					return -2;
				}
			}
			if((*(DischargeDataH)L)[i].time <= ((DischargeData *)p)->time)break;
		}
		i++;
		_BlockMove(ITEM_PTR(i), ITEM_PTR(i + 1), (n - i) * elementSize);
		_BlockMove(p, ITEM_PTR(i), elementSize);
		n++;
		*itemnum=i;
		return 0;
	}
}