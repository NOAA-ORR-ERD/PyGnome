
#ifndef __CMYLIST__
#define __CMYLIST__

#include "Earl.h"
#include "TypeDefs.h"

class CMyList
{
	protected:

		long	n;
		long	elementSize;
		long	iteratorIndex;
		long	iterIncrement;
		
		CHARH	L;

	public:
		
				CMyList (long elemSize);
		virtual ~CMyList () { Dispose (); }

		OSErr	IList ();

		Handle	GetListDataHdl ();
		void	SetListDataHdl (Handle newListHdl, long newItemCount);

		long	GetItemSize ();
		void	ClearList ();
		//void	Dispose ();
		virtual void	Dispose ();

		OSErr	AppendItem (Ptr itemPtr);
		OSErr	SetListItem (Ptr itemPtr, long i);
		OSErr	GetListItem (Ptr itemPtr, long i);
		OSErr	InsertAtIndex (Ptr itemPtr, long i);
		OSErr	DeleteItem (long i);
		Boolean	IsItemInList (Ptr itemPtr, long *itemIndexPtr);
		Boolean	IsTempMemList ();
		OSErr	SwapItems (long iA, long iB);
		OSErr	MoveToTop (long i);
		OSErr	MoveToBottom (long i);
		long	GetItemCount ();

		OSErr	GetFirstItem (Ptr itemPtr); // resets iterator and returns first item
		OSErr	GetLastItem (Ptr itemPtr); // resets iterator and returns last  item
		OSErr	GetNextItem (Ptr itemPtr); // returns an error when done
		long	GetIteratorIndex ();
		virtual OSErr InsertSorted(Ptr p,long *itemnum,Boolean allowDuplicates);
		
};

class CTimeValuePairList : public CMyList
{
	public:
		CTimeValuePairList(long n) : CMyList(n) {};
		virtual	~CTimeValuePairList () { Dispose (); }
		OSErr InsertSorted(Ptr p,long *itemnum,Boolean allowDuplicates);
		virtual void		Dispose () { return; }
};
// for CDOG model
class CProfilesList : public CMyList
{
	public:
		CProfilesList(long n) : CMyList(n) {};
		virtual	~CProfilesList () { Dispose (); }
		OSErr InsertSorted(Ptr p,long *itemnum,Boolean allowDuplicates);
		virtual void		Dispose () { return; }
};

class CDischargeTimes : public CMyList
{
	public:
		CDischargeTimes(long n) : CMyList(n) {};
		virtual	~CDischargeTimes () { Dispose (); }
		OSErr InsertSorted(Ptr p,long *itemnum,Boolean allowDuplicates);
		virtual void		Dispose () { return; }
};

#endif
