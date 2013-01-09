#!/usr/bin/env python

class OrderedCollection(object):
    '''
    Generalized Container for a set of objects of a particular type which
    preserves the order of insertion and supports replacement of not only
    an object in the list, but the key/id that references it. (a normal
    OrderedDict can't do this)
    - The order of insertion is preserved.
    - Objects are accessed by id, as if in a dictionary.
    - Objects can be replaced in order.  The objects will be referenced 
      by a new id, and still be in the correct order.
    '''
    def __init__(self, elems=[], dtype=None):
        if elems and not isinstance(elems, list):
            raise TypeError('OrderedCollection: needs a list of objects')

        if not dtype and len(elems) == 0:
            raise TypeError('OrderedCollection: specify a data type if list is empty')
        elif not dtype:
            self.dtype = type(elems[0])
        else:
            self.dtype = dtype

        if not all([isinstance(e, self.dtype) for e in elems]):
            raise TypeError('Group needs a list of type: %s' % (self.dtype))
        # a bunch of Gnome classes have an id property defined, which we will prefer
        # otherwise, we just take the id(e) value
        self._index = dict([(e.id if hasattr(e, 'id') else id(e), idx) for e, idx in zip(elems, range(len(elems)))])
        self._elems = elems[:]
        pass

    def __len__(self):
        return len(self._index.keys())

    def __iter__(self):
        vals = self._index.values()
        for i in sorted(vals):
            yield self._elems[i]

    def __getitem__(self, ident):
        return self.get(ident)

    def __setitem__(self, ident, new_elem):
        self.replace(ident, new_elem)

    def __delitem__(self, ident):
        self.remove(ident)

    def __iadd__(self, rop):
        self.add(rop)
        return self

    def get(self, ident):
        return self._elems[self._index[ident]]

    def add(self, elem):
        ''' Add an object to the collection '''
        if isinstance(elem, self.dtype):
            if hasattr(elem, 'id'):
                # a bunch of Gnome classes have an id property defined, which we will prefer
                l__id = elem.id
            else:
                l__id = id(elem)
            if l__id not in self._index.keys():
                self._index[l__id] = len(self._elems)
                self._elems.append(elem)
            pass
        elif isinstance(elem, list) and all([isinstance(e, self.dtype) for e in elem]):
            for e in elem:
                self.add(e)
            pass
        else:
            raise TypeError('OrderedCollection: expected type %s, got type %s' % (self.dtype, type(elem)))

    def remove(self, ident):
        ''' Remove an object from the collection '''
        self._elems[self._index[ident]] = None
        del self._index[ident]

    def replace(self, ident, new_elem):
        ''' Replace an object in the collection '''
        if ident in self._index.keys():
            # we have an existing object
            idx = self._index[ident]
            del self._index[ident]
            if hasattr(new_elem, 'id'):
                # a bunch of Gnome classes have an id property defined, which we will prefer
                self._index[new_elem.id] = idx
            else:
                self._index[id(new_elem)] = idx
            self._elems[idx] = new_elem
        else:
            # right now we just add it at the end.
            # should we throw a key error instead?
            self.add(new_elem)


