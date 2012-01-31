#!/usr/bin/env python2.3

"""
A small module with assorted handy utilities:


sort_by_other_list(list_to_sort,list_to_sort_by):

        returns a list of the elements of "list_to_sort", sorted by the
        elements of "list_to_sort_by".

        Example:
        >>> sort_by_other_list(['a,','b','c','d'],[4,1,3,2])
        ['b', 'd', 'c', 'a,']

"""
__version__ = "0.1.1"


def sort_by_other_list(list_to_sort,list_to_sort_by):
    """
    sort_by_other list(list_to_sort,list_to_sort_by)
    
    function that sorts one list by the contents of another list.
    
    the list that is being sorted does not have to be sortable
    """
    pairs = map(None, list_to_sort_by,range(len(list_to_sort_by)))
    pairs.sort()
    out_list = []
    for i in map(lambda x: x[1],pairs):
        out_list.append(list_to_sort[i])
    return out_list
    

