#!/usr/bin/env python
# OilLibParse - program to parse the OilLib flat file
#               from the ADIOS2 application

import sys
from optparse import OptionParser

class LibFile(object):
    ''' LibFile - A specialized file reader for the
                  OilLib and CustLib flat datafiles.
                - These files use a different character to designate
                  a line of text ('\r').
                - Additionally, each line contains a number of fields
                  separated by a tab ('\t').  In this way it attempts
                  to represent tabular data.
                - The first line in the file contains a file format
                  version number ('N.N'), followed by a date ('d/m/YY'),
                  and finally the product ('adios').
                - The second line in the file contains a table header,
                  where each field represents the "long" name of a
                  tabular column.
                - The rest of the lines in the file contain table data.

    '''
    def __init__(self, name, lineDelim='\r', fieldDelim='\t'):
        self.fileobj = open(name, 'r')
        self.lineDelim = lineDelim
        self.fieldDelim = fieldDelim

        # we will consume the header upon opening.
        self.__version__ = self.readline(32)
        self._check_version_hdr()

        # we will next consume the column fields upon opening.
        self.fileColumns = self.readline()
        self.fileColumnsLU = dict(zip(self.fileColumns, range(len(self.fileColumns))))
        self.numFileColumns = len(self.fileColumns)

    def _check_version_hdr(self):
        ''' check that the file has a proper header.
            right now we are just checking for adios
            specific fields.
        '''
        if len(self.__version__) != 3:
            raise Exception('Bad file header: did not find 3 fields for version!!')
        elif self.__version__[-1] != 'adios':
            raise Exception('Bad file header: did not find product field!!')

    def _parse_row(self, buff):
        ret = unicode(str(buff), encoding='utf_8', errors='replace').split(self.fieldDelim)  # split fields
        ret = [c if len(c) > 0 else None for c in ret]  # replace empty fields with None
        return ret


    def readline(self, size=None):
        ''' Read the next line in the file that is delimited by our
            specified line delimiter
            - We split the lines based on our field delimiter and
              return a list of fields.
        '''
        buff = bytearray()
        while True:
            c = self.fileobj.read(1)
            if (len(c) < 1 or c == self.lineDelim) or (size and len(buff) >= size):
                return self._parse_row(buff)
            else:
                buff += c

    def readlines(self):
        ''' Sequentially read the lines that are delimited by our
            specified line delimiter.
            - This works as a generator so we can iterate over the lines.
            - We split the lines based on our field delimiter and
              return a list of fields.
        '''
        buff = bytearray()
        while True:
            c = self.fileobj.read(1)
            if len(c) < 1:
                if len(buff) > 0:
                    yield self._parse_row(buff)
                break
            elif c == self.lineDelim:
                yield self._parse_row(buff)
                buff = bytearray()
            else:
                buff += c



if __name__ == '__main__':
    # parse our command line options
    usage = '''Usage: OilLibParse.py FILE [options]
    
Required:
  FILE\t\t\tan input file containing Oil Library table data'''
    parser = OptionParser(usage=usage)
    parser.add_option('-f','--field',
            dest='fields',
            help='list of fields to be displayed',
            metavar='Field1,...,FieldN' )
    parser.add_option('-v','--verbose',
            action='store_true', default=False,
            dest='verbose',
            help='verbose output')
    parser.add_option('-n','--noprompt',
            action='store_false', default=True,
            dest='prompt',
            help='do not prompt for the next row')
    parser.add_option('-r','--raw',
            action='store_true', default=False,
            dest='raw',
            help='just display the raw row data')
    (options, args) = parser.parse_args()

    # open our OilLib file
    if len(args) < 1:
        parser.error("please specify an input file")
    else:
        filename = args[0]

    if options.verbose:
        print 'opening:', (filename,)
    fd = LibFile(filename)
    if options.verbose:
        print fd.__version__

    for r in fd.readlines():
        matchingFields = []

        if options.verbose:
            print '-'*50
            print 'Number of Fields/Header Columns: %d/%d' % (len(r), fd.numFileColumns)

        if options.fields:
            fields = options.fields.split(',')
            matchingFields = set(fields).intersection(fd.fileColumns)
            if options.verbose:
                print 'fields specified:', fields
                print 'fields matching columns:', matchingFields

        if options.raw:
            print '\t%s' % (r,)
        elif len(matchingFields) > 0:
            # we just display the fields we want
            for f in matchingFields:
                print '\t%-20s\t%s' % (f + ':', (r[fd.fileColumnsLU[f]],))
        else:
            # we display all fields
            for f, i in zip(r, range(len(r))):
                if i < fd.numFileColumns:
                    fieldName = fd.fileColumns[i]
                    print '\t%-20s\t%s' % (fieldName + ':', (f,))
                else:
                    print '\t%-20s\t%s' % ('extra field:', (f,))
        if options.prompt:
            sys.stdin.readline()
        else:
            print
