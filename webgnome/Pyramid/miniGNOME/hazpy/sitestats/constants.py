import datetime
import re

# Cutoff date for Access table.  Earlier records are in LegacyAccess.
ACCESS_CUTOFF = datetime.date(2009, 3, 1)

IGNORE_REFERER_DOMAINS = [
    "cameochemicals.noaa.gov",
    "chemresponsetool.noaa.gov",
    "incidentnews.gov",
    "www.cameochemicals.noaa.gov",
    "www.chemresponsetool.noaa.gov",
    "www.incidentnews.gov",
    "127.0.0.1",
    "localhost",
    "10.55.66",
    ]

STATIC_SECTIONS = [
    "robots.txt",
    "favicon.ico",
    "images",
    "stylesheets",
    "javascripts",
    ]

# Sites for which we track USCG usage.
USCG_SITES = ["cameo", "crt"]

# SQL LIKE expression matching USCG remote addresses.
USCG_REMOTE_ADDR_LIKE = "152.121.%"

# Regular expressions to extract a record ID from a details page URL.
RLINK_INCIDENT_RX = re.compile( R"/hotline/(?:incidents?/)?(\d+)/?$" )
INEWS_INCIDENT_RX = re.compile( R"/(?:incidents?/)?(\d+)/?$" )
CAMEO_CHEMICAL_RX = NotImplemented
CAMEO_UNNA_RX = NotImplemented
CAMEO_REACT_RX = NotImplemented
