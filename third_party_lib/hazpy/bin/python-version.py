#!/usr/bin/env python
"""Show the installed version of each package."""

import optparse
import sys

import pkg_resources

USAGE = "%prog [options] PACKAGE_NAME ..."
DESCRIPTION = __doc__.splitlines()[0]

parser = optparse.OptionParser(usage=USAGE, description=DESCRIPTION)
opts, args = parser.parse_args()
if not args:
    parser.error("which packages?")
for name in args:
    try:
        dist = pkg_resources.get_distribution(name)
        version = dist.version
        location = " (%s)" % dist.location
        print "%s: %s%s" % (name, version, location)
    except pkg_resources.DistributionNotFound:
        print "%s: None" % name
