IF not HEADERS.count("_LIST_"):
    DEF HEADERS = HEADERS +  ["_LE_LIST_"]
    cdef extern from "LEList_c.h":
        cdef cppclass LEList_c:
            long numOfLEs
            LETYPE fLeType
        
