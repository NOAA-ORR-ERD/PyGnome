#!/usr/bin/env python
import textwrap

def pretty_name(incident, use_keys=False):
    if incident is None:
        return ""
    if use_keys:  # Backward compatibility for Quixote rlink/inews.
        name = incident['name']
        location = incident['location']
        is_drill = incident['is_type_drill']
    else:
        name = incident.name
        location = incident.location
        is_drill = incident.is_type_drill
    if name and location:
        ret = "%s, %s" % (name, location)
    elif name:
        ret = name
    elif location:
        ret = location
    if is_drill:
        ret = "DRILL -- " + ret
    return ret
    

def format_release_range(s1, s2, unit, v1, v2, is_mass):
    """Format a potential/actual release range for display.
       `s1`, `s2`, and `unit` are the values the user entered.
       `v1`, `v2`, and `is_mass` are examined for boundary cases.
    """
    if not s1:
        return ""
    if v1 == 0.0 and v2 in [0.0, None]:
        return "0"
    if not s2:
        return "%s %s" % (s1, unit)
    return "%s - %s %s" % (s1, s2, unit)
        
def wrap_entry_content(text, width=90, wrap_if_longer_than=None):
    if wrap_if_longer_than is None:
        wrap_if_longer_than = width
    wrapper = textwrap.TextWrapper(width=width)
    lines = text.splitlines()
    ret = []
    for lin in lines:
        if len(lin) <= wrap_if_longer_than:
            ret.append(lin)
        else:
            wrapped = wrapper.wrap(lin)
            ret.extend(wrapped)
            ret.append("")
    return "\n".join(ret) + "\n"
        
            
