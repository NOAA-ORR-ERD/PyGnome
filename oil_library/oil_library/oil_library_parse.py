#!/usr/bin/env python
# OilLibParse - program to parse the OilLib flat file
#               from the ADIOS2 application

import sys
from optparse import OptionParser


class OilLibraryFile(object):
    ''' A specialized file reader for the OilLib and CustLib
        flat datafiles.
        - We will use universal newline support to designate
          a line of text.
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
    def __init__(self, name, field_delim='\t'):
        self.file_columns = None
        self.file_columns_lu = None
        self.num_columns = None

        self.fileobj = open(name, 'rU')
        self.field_delim = field_delim

        self.__version__ = self.readline()
        self._check_version_hdr()
        self._set_table_columns()

    def _check_version_hdr(self):
        ''' check that the file has a proper header.
            right now we are just checking for adios
            specific fields.
        '''
        print 'checking version header:', self.__version__
        if len(self.__version__) != 3:
            raise Exception('Bad file header: did not find 3 fields ' \
                            'for version!!')
        elif self.__version__[-1] != 'adios':
            raise Exception('Bad file header: did not find product field!!')

    def _set_table_columns(self):
        self.file_columns = self.readline()
        self.file_columns_lu = dict(zip(self.file_columns,
                                        range(len(self.file_columns))))
        self.num_columns = len(self.file_columns)

    def _parse_row(self, line):
        line = line.strip()
        if len(line) > 0:
            row = unicode(str(line), encoding='utf_8',
                          errors='replace').split(self.field_delim)
            row = [c if len(c) > 0 else None for c in row]
        else:
            row = []
        return row

    def readline(self):
        return self._parse_row(self.fileobj.readline())

    def readlines(self):
        while True:
            line = self.readline()
            if len(line) > 0:
                yield line
            else:
                break


if __name__ == '__main__':
    # parse our command line options
    usage = '''Usage: OilLibParse.py FILE [options]

Required:
  FILE\t\t\tan input file containing Oil Library table data'''
    parser = OptionParser(usage=usage)
    parser.add_option('-f', '--field',
            dest='fields',
            help='list of fields to be displayed',
            metavar='Field1,...,FieldN')
    parser.add_option('-v', '--verbose',
            action='store_true', default=False,
            dest='verbose',
            help='verbose output')
    parser.add_option('-n', '--noprompt',
            action='store_false', default=True,
            dest='prompt',
            help='do not prompt for the next row')
    parser.add_option('-r', '--raw',
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
    fd = OilLibraryFile(filename)
    if options.verbose:
        print fd.__version__

    for r in fd.readlines():
        matchingFields = []

        if options.verbose:
            print '-' * 50
            print 'Number of Fields/Header Columns: %d/%d' % (len(r), fd.num_columns)

        if options.fields:
            fields = options.fields.split(',')
            matchingFields = set(fields).intersection(fd.file_columns)
            if options.verbose:
                print 'fields specified:', fields
                print 'fields matching columns:', matchingFields

        if options.raw:
            print '\t%s' % (r,)
        elif len(matchingFields) > 0:
            # we just display the fields we want
            for f in matchingFields:
                print '\t%-20s\t%s' % (f + ':', (r[fd.file_columns_lu[f]],))
        else:
            # we display all fields
            for f, i in zip(r, range(len(r))):
                if i < fd.num_columns:
                    fieldName = fd.file_columns[i]
                    print '\t%-20s\t%s' % (fieldName + ':', (f,))
                else:
                    print '\t%-20s\t%s' % ('extra field:', (f,))
        if options.prompt:
            sys.stdin.readline()
        else:
            print
