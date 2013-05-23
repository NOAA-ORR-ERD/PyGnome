import ldap
import os

__all__ = ["init_ldap", "get_user_dn", "log_in", "SEATTLE", "SILVER_SPRING"]

#### NOAA LDAP authentication servers
servers = {
    "seattle": "ldaps://ldap-west.nems.noaa.gov:636",
    "silver_spring": "ldaps://ldap-east.nems.noaa.gov:636",
    "boulder": "ldaps://ldap-mountain.nems.noaa.gov:636",
    }
SEATTLE = servers["seattle"]
SILVER_SPRING = servers["silver_spring"]

#### File containing TLS site certificates for all above servers
certfile_name = "noaa-ldap-certs.crt"
my_directory = os.path.abspath(os.path.dirname(__file__))
CERTFILE = os.path.join(my_directory, certfile_name)

#### Exceptions
class NoSuchUser(Exception):
    pass

class InvalidPassword(Exception):
    pass


#### init_ldap function
def init_ldap(url=SEATTLE, certfile=CERTFILE, debug_level=0, **initialize_kw):
    """Initialize a NOAA LDAP server.

    Arguments:
    
    * ``url``: The server's URL. See ``servers`` constant for examples.

    * ``certfile``: Absolute path of the file containing the server's
    TLS site certificates.

    * ``debug_level``: Debug level for the LDAP C library. Default 0. Other
      values are undocumented at this time.

    * ``\*\*initialize_kw``: Keyword args to pass to ``ldap.initialize()``.

    Return a server object, the same that ``ldap.initialize()`` returns.

    Exceptions:

    * ``ldap.SERVER_DOWN``: Can't contact server, or site certificate
      rejected.

    * Other ``ldap`` exceptions may also be raised.
    
    May be used with non-NOAA servers.
    """
    # Must set cert options globally or they're not recognized.
    ldap.set_option(ldap.OPT_DEBUG_LEVEL, debug_level)
    if certfile:
        ldap.set_option(ldap.OPT_X_TLS_CACERTFILE, certfile)
    else:
        ldap.set_option(ldap.OPT_X_TLS_REQUIRE_CERT, ldap.OPT_X_TLS_ALLOW)
    server = ldap.initialize(url, **initialize_kw)
    server.protocol = ldap.VERSION3
    return server

def get_user_dn(username):
    """Return the NOAA LDAP Distinguished Name for the specified person.

    ``username`` is the user ID (the first part of the person's email
    address (e.g., john.doe). 
    """
    return "uid=%s,ou=People,o=noaa.gov" % username
