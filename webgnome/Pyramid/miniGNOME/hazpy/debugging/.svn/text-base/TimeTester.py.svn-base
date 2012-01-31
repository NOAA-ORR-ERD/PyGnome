#!/usr/bin/env python
"""TimeTester -- Record the time required for several discrete operations.

Usage:
tt = TimeTester()
# Do something here whose time you want to record.
tt.record("Task 1")
# Do something here whose time you want to record.
tt.record("Task 2")
# Do things here whose time doesn't matter.
tt.reset_time()
# Do something here whose time you want to record.
tt.record("Task 3")
# Print a report.
print
tt.report(sys.stdout)

The argument to .record() should be formatted to fit into
the sentence "Took N seconds to ___."
"""
import sys, time

class TimeTester(object):
    def __init__(self):
        self.results  = []
        self.last_call = None
        self.reset_time()

    def reset_time(self):
        self.last_time_called = time.time()

    def record(self, label):
        """Call function and record its execution time."""
        start = self.last_time_called
        interval = time.time() - start
        self.results.append((label, interval))
        self.reset_time()

    def print_report(self, f=None, decimal_digits=1):
        """Write a multi-line report to file 'f'."""
        if f is None:
            f = sys.stdout
        print >>f
        for label, interval in self.results:
            tup = decimal_digits, interval, label
            print >>f, "Took %0.*f seconds to %s." % tup
        print >>f
