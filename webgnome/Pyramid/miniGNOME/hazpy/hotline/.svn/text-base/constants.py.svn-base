"""This module is imported via "from hotline.constants import *", so it should
contain only constants named with CAPITAL_LETTERS only.  Any private variables
should have a leading _underscore.
"""

CAUSES = {
    1: "Collision",
    2: "Discharge / Dumping",
    3: "Grounding",
    4: "Mystery Spill",
    5: "Sunken Vessel",
    6: "Transfer",
    7: "Leaking Shipping Container",
    8: "Leaking Tank",
    9: "Pipeline Leak",
    10: "Well Blowout",
    11: "Plane Crash",
    12: "Search & Rescue",
    13: "Train Derailment",
    14: "Fire / Explosion",
    15: "Other / Unknown",
    16: "Capsized Vessel"
    }
OTHER_CAUSE = 15    # If this cause is chosen, 'other_cause' field is active.
DEFAULT_CAUSE = 15   # Default cause for new incidents.

class Category(object):
    def __init__(self, id, label, rlink_order, inews_order, orr_private, 
        default_thumb, description):
        self.id = id
        self.label = label
        self.rlink_order = rlink_order
        self.inews_order = inews_order
        self.orr_private = orr_private
        self.default_thumb = default_thumb
        self.description = description

    def __repr__(self):
        return "<Category %d '%s'>" % (self.id, self.label)

CATEGORIES = dict((x.id, x) for x in [
    Category(1, "Situation Reports", 1, 0, False, False, 
"""Situation Reports include any entries that provide an overview of the
current state of an incident or response activities. Examples: SSC Evening
Reports, Pollution Reports (POLREPS), Site Status documents. Exceptions are
Incident Action Plans, which have their own category."""),
    Category(2, "Overflights", 3, 0, False, False, 
"""Overflights are products generated following an aerial reconnaissance of an
incident. Generally, they are one-page maps charting the locations of
uncontained oil. Photos taken during an over-flight are posted to the Photos
category."""),
    Category(3, "Trajectories & Chemistry", 2, 0, False, False, 
"""Trajectory Forecasts predict the likely movement of, in most cases, floating
oil on a body of water, but could also be dispersed chemicals in air, etc. This
category also contains oil weathering information and other pollutant
properties."""),
    Category(4, "Weather & Water", 4, 0, False, False, 
"""This category contains information about environmental conditions. Examples:
weather forecasts, weather observations, tide heights, tidal current forecasts,
tidal current observations, river flow rates, river gauge levels."""),
    Category(5, "Resources at Risk", 5, 0, False, False, 
"""Resources at Risk are specific documents that review the environmental
resources (for example, shoreline resources, biological resources like fish and
birds, and human-use resources) in the geographic area potentially impacted by
an incident and discuss how those resources might be negatively affected by the
incident."""),
    Category(6, "Other Products", 11, 0, False, False, 
"""Other Products is the final "catch-all" category when an entry has no other
logical placement. Examples: in-situ burn plans, vessel diagrams, oil recovery
plans."""),
    Category(7, "Photos", 7, 0, False, False, 
"""The Photo category is a general category where all photo images are located.
Since it is a general category, care should be taken to provide adequate photo
captions. Please note that full-resolution photos are sometimes too large to
post, so size-reduced photos are often posted instead, and contacting the
photographer is the only way to acquire the full resolution image."""),
    Category(8, "Shoreline Assessments", 6, 0, False, False, 
"""This category is for shoreline assessment processes (sometimes referred to
as SCAT). Teams survey shorelines that may be impacted by an incident and
provide first hand observations in order to inform the cleanup planning
process."""),
    Category(9, "Maps", 10, 0, False, False, 
"""The Maps category is a "catch-all" for map-based products that don't fit
into another category. Other categories, like Overflights, Chemistry &
Trajectories, and Shoreline Assessments might also have map products."""),
    Category(10, "Logistics & Safety", 9, 0, False, False, 
"""This is a category for logistical or administrative information that may be
of interest to responders, like directions to the command post, the command
post FAX number or site safety information."""),
    Category(11, "Incident Action Plans", 8, 0, False, False, 
"""Incident Action Plans (IAP) are created and used by the US Coast Guard. They
document response actions during the active response stage of an incident."""),
    Category(12, "Press Releases", 12, 0, False, False, 
"""Public advisories and press releases.  These give timely and sometimes
urgent information to the public during an incident."""),
    Category(13, "After-Incident Documents", 13, 0, False, False, 
"""Reports and analyses generated after ERD's participation in an incident has
finished."""),
    Category(14, "NRDA", 14, 0, False, False, 
"""Damage assessment information available to all responders."""),
    Category(15, "OR&R Private", 15, 0, True, False, 
"""Private notes visible only to OR&R staff."""),
    Category(16, "FAQs", 16, 0, True, False, 
"""Talking points (OR&R access only.) (NOTE: This is a temporary category for
the Deepwater Horizon incident only. It appears in other incidents due to
limitations in ResponseLINK. Please do not put anything in it for other
incidents."""),
    Category(17, "Chemistry", 17, 0, False, False,
"""Chemistry data. (NOTE: This is a temporary category for the Deepwater
Horizon incident only. It appears in other incidents due to limitations in
ResponseLINK. Please do not put anything in it for other incidents.)"""),
    Category(18, "Deepwater Operations", 18, 0, False, False,
"""Deepwater monitoring ship operation. (NOTE: This is a temporary category for
the Deepwater Horizon incident only. It appears in other incidents due to
limitations in ResponseLINK. Please do not put anything in it for other
incidents.)"""),
    ])

DEFAULT_CATEGORY = 1
ORR_PRIVATE_CATEGORIES = [x.id for x in CATEGORIES.itervalues() if
    x.orr_private]


EFFORT = {
    1: "Notification",
    2: "Phone support",
    3: "Products generated",
    4: "On-scene support",
    }

COUNTERMEASURES = {
    0: "No information",
    2: "Evaluated but not used",
    3: "Evaluated and applied",
    4: "Not applicable",
    }
    # Countermeasure 1 was deleted.
DEFAULT_COUNTERMEASURE = 0
APPLIED_COUNTERMEASURE = 3

MEASURE_SKIM_TITLE = "On-Water Recovery"
MEASURE_SHORE_TITLE = "Shoreline Cleanup"
MEASURE_DISPERSE_TITLE = "Dispersants"
MEASURE_BURN_TITLE = "In-Situ Burn"
MEASURE_BIO_TITLE = "Bioremediation"

# Images we create thumbnails for.
IMAGE_TYPES = ['jpg', 'png', 'gif']

# Units for input/display.  Internal units are gallons and pounds.
VOLUME_UNITS = ['gallons', 'barrels', 'cubic meters']
MASS_UNITS = ['pounds', 'kilograms', 'tons', 'metric tons']

# OR&R divisions for Locator.
DIVISIONS = [
    "ARD",
    "BSG", 
    "ERD",
    "HQ",
    "Marine Debris",
    "Pribs",
    ]
