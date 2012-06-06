IF not HEADERS.count("_CMYLIST_"):
    DEF HEADERS = HEADERS + ["_CMYLIST_"]
    cdef extern from "CMYLIST.H":
        cdef cppclass CMyList:
            CMyList(long)
            OSErr AppendItem(char *)
            OSErr IList()
