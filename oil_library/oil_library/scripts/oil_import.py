import os
import sys

import numpy as np

from ..oil_library_parse import OilLibraryFile


def usage(argv):
    cmd = os.path.basename(argv[0])
    print('usage: {0} <import_file_1> <import_file_2>\n'
          '(example: "{0} OilLib ADIOS2Export.txt")'.format(cmd))
    sys.exit(1)


def diff_import_files(file1, file2):
    print 'opening file: {0} ...'.format(file1)
    fd1 = OilLibraryFile(file1, ignore_version=True)

    print 'opening file: {0} ...'.format(file2)
    fd2 = OilLibraryFile(file2, ignore_version=True)

    lines1, lines2 = list(fd1.readlines()), list(fd2.readlines())
    print 'line lengths = ', (len(lines1), len(lines2))
    print 'matching slices for these files:'
    get_diffs(lines1, lines2, fd1.file_columns)


def get_diffs(a, b, field_names):
    '''
        We are using a slightly modified Hunt-McIlroy algorithm here
        to generate our diff.
        - to compare rows(lines), we use np.isclose() when comparing
          numerical fields.  Precision is set to around 3 decimal places.
        - we setup a hash of compare results so we do not redundantly
          compare rows that have been previously compared.
        - when we have a pair of rows that we think are modified versions
          of the same row, we display only the fields that are different
          between them.
    '''
    matching_lines = {}
    ia = ib = 0
    slices = matching_slices(a, 0, len(a),
                             b, 0, len(b),
                             matching_lines)
    slices.append((len(a), len(b), 0))
    print
    for sa, sb, n in slices:
        print (ia, sa), (ib, sb), n
        sa_len, sb_len = sa - ia, sb - ib
        for idx in range(max([sa_len, sb_len])):
            if idx < sa_len and idx < sb_len:
                # we will diff sa and sb as lists
                print '<-> {0}: {1}'.format(a[ia + idx][1],
                                            row_diff(a[ia + idx],
                                                     b[ib + idx],
                                                     field_names))
            elif idx < sa_len:
                # display sa but not sb
                print '- {0}'.format(a[ia + idx][:2])
            elif idx < sb_len:
                # display sb but not sa
                print '+ {0}'.format(b[ib + idx][:2])

        for line in a[sa:sa + min(n, 4)]:
            print '  {0}'.format(line[:2])

        if n > 4:
            print '  ...'

        ia = sa + n
        ib = sb + n


def matching_slices(a, a0, a1,
                    b, b0, b1,
                    matching_lines):
    sa, sb, n = longest_matching_slice(a, a0, a1,
                                       b, b0, b1,
                                       matching_lines)

    if n == 0:
        return []

    return (matching_slices(a, a0, sa, b, b0, sb, matching_lines) +
            [(sa, sb, n)] +
            matching_slices(a, sa+n, a1, b, sb+n, b1, matching_lines))


def longest_matching_slice(a, a0, a1,
                           b, b0, b1,
                           matching_lines):
    sa, sb, n = a0, b0, 0
    runs = {}

    for i in range(a0, a1):
        new_runs = {}

        for j in range(b0, b1):
            # if a[i] == b[j]:
            if (i, j) not in matching_lines:
                matching_lines[(i, j)] = all([is_equal(i1, i2)
                                              for i1, i2 in zip(a[i], b[j])])

            if matching_lines[(i, j)]:
                sys.stderr.write('.')
                k = new_runs[j] = runs.get(j - 1, 0) + 1
                if k > n:
                    sa, sb, n = i - k + 1, j - k + 1, k

        runs = new_runs

    # assert a[sa:sa + n] == b[sb:sb + n]
    return sa, sb, n


def is_equal(item1, item2):
    try:
        # try to evaluate as numeric
        f1, f2 = float(item1), float(item2)

        if np.isclose(f1, f2, rtol=0.005):
            return True
        else:
            return False
    except:
        # try to evaluate as non-numeric
        return item1 == item2


def row_diff(a, b, field_names):
    return [(fn, v1, v2)
            for fn, v1, v2 in zip(field_names, a, b)
            if not is_equal(v1, v2)]


def diff_import_files_cmd(argv=sys.argv, proc=diff_import_files):
    if len(argv) < 3:
        usage(argv)

    f1, f2 = argv[1:3]

    try:
        proc(f1, f2)
    except:
        print "{0} FAILED\n".format(proc)
        raise
