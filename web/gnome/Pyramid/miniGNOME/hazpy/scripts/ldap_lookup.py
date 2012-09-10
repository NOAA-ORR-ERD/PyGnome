import getpass
import logging
import os
import pprint

import argparse
import ldap

import hazpy.ldap_util as ldap_util

DESCRIPTION = "Display a user's NOAA LDAP properties."

log = logging.getLogger()

def get_arg_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    paa = parser.add_argument
    paa("--user", action="store", metavar="USERNAME",
        help="authentication username (default: don't authenticate)")
    paa("--password", action="store",
        help="authentication password (default: prompt for it)")
    paa("-a", "--attr", action="append", dest="attributes",
        metavar="ATTRIBUTE", 
        help="""\
LDAP attribute (property) to display. May be specified multiple times.
(default: all attributes)""")
    paa("--ou", action="store_true",
        help="shortcut for '-a ou -a ou1'")
    paa("--server", action="store", default=ldap_util.SEATTLE, 
        help="LDAP server to use")
    paa("--list-servers", action="store_true", 
        help="list known LDAP servers and their abbreviations")
    paa("--debug", action="store_true",  
        help="debug Python code")
    paa("--debug-c", action="store", type=int, default=0, metavar="N", 
        help="debug C library (int 0-3, default 0)")
    paa("targets", action="store", metavar="TARGET_USER", nargs="*",
        help="target users (default: same as --user)")
    return parser

parser = get_arg_parser()

def list_servers():
    my_servers = sorted(ldap_util.servers.iteritems())
    maxlen = max(len(x[0]) for x in my_servers)
    for abbrev, url in my_servers:
        print "%-*s   %s" % (maxlen, abbrev, url)

def get_server_url(server_opt):
    if ":" in server_opt:
        return server_opt
    try:
        return ldap_util.servers[server_opt]
    except KeyError:
        msg = "unknown server '{0}' (try --list-servers)"
        msg = msg.format(server_opt)
        parser.error(msg)

def log_in(server, username, password):
    dn = ldap_util.get_user_dn(username)
    if not password:
        password = getpass.getpass("LDAP password: ")
    try:
        server.simple_bind_s(dn, password)
    except ldap.NO_SUCH_OBJECT:
        msg = "no such authenticating user '{0}".format(opts.user)
        parser.error(msg)
    except ldap.INVALID_CREDENTIALS:
        parser.error("incorrect password")

def lookup_user(server, username, attributes=None):
    log = logging.getLogger()
    log.debug("lookup_user args: %r", locals())
    print "{0}: ".format(username)
    dn = ldap_util.get_user_dn(username)
    scope = ldap.SCOPE_BASE
    attrlist = attributes or None
    try:
        records = server.search_s(dn, scope, attrlist=attrlist)
    except ldap.NO_SUCH_OBJECT:
        print "    no such user"
        return
    if records:
        for dn, attrs in records:
            print "    {0}:".format(dn)
            for key, value in sorted(attrs.iteritems()):
                print "        {0}: {1!r}".format(key, value)
    else:
        print "    no results"

def main():
    logging.basicConfig()
    log = logging.getLogger()
    opts = parser.parse_args()
    if opts.debug:
        log.setLevel(logging.DEBUG)
    log.debug("Command-line args: %s", opts)
    if opts.list_servers:
        list_servers()
        return
    if opts.targets:
        targets = opts.targets
    elif opts.user:
        targets = [opts.user]
    else:
        parser.error("must specify TARGET_USER if not using --user")
    attributes = None
    if opts.attributes and opts.ou:
        parser.error("can't specify both --attr and --ou")
    elif opts.attributes:
        attributes = opts.attributes
    elif opts.ou:
        attributes = ["ou", "ou1"]
    server_url = get_server_url(opts.server)
    server = ldap_util.init_ldap(server_url, debug_level=opts.debug_c)
    if opts.user:
        log_in(server, opts.user, opts.password)
    for target in targets:
        lookup_user(server, target, attributes)
    

if __name__ == "__main__":  main()
