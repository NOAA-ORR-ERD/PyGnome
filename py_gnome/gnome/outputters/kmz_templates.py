"""
templates for the kmz  outputter
"""

caveat = "This trajectory was produced by GNOME (General NOAA Operational Modeling Environment), and should be used for educational and planning purposes only--not for a real response. In the event of an oil or chemical spill in U.S. waters, contact the U.S. Coast Guard National Response Center at 1-800-424-8802."


### The kml templates:
header_template="""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{kml_name}</name>
    <open>1</open>
    <description><![CDATA[<b>Valid for:</b> {valid_timestring}<br>
                          <b>Issued:</b>{issued_timestring} <br>
                          {caveat}]]>
    </description>

    <Style id="RedDotIcon">
      <IconStyle>
         <scale>0.2</scale>
         <color>ff0000ff</color>
         <Icon>
            <href>dot.png</href>
         </Icon>
          <hotSpot x="0.5"  y="0.5" xunits="fraction" yunits="fraction"/>
      </IconStyle>
      <LabelStyle>
         <color>00000000</color>
      </LabelStyle>
    </Style>

    <Style id="BlackDotIcon">
      <IconStyle>
         <scale>0.2</scale>
         <Icon>
            <href>dot.png</href>
         </Icon>
         <color>ff000000</color>
         <hotSpot x="0.5"  y="0.5" xunits="fraction" yunits="fraction"/>
      </IconStyle>
      <LabelStyle>
         <color>00000000</color>
      </LabelStyle>
    </Style>

    <Style id="YellowDotIcon">
      <IconStyle>
         <scale>0.2</scale>
         <Icon>
            <href>dot.png</href>
         </Icon>
         <color>ff00ffff</color>
         <hotSpot x="0.5"  y="0.5" xunits="fraction" yunits="fraction"/>
      </IconStyle>
      <LabelStyle>
         <color>00000000</color>
      </LabelStyle>
    </Style>

    <Style id="RedXIcon">
      <IconStyle>
         <scale>0.2</scale>
         <color>ff0000ff</color>
         <Icon>
            <href>x.png</href>
         </Icon>
          <hotSpot x="0.5"  y="0.5" xunits="fraction" yunits="fraction"/>
      </IconStyle>
      <LabelStyle>
         <color>00000000</color>
      </LabelStyle>
    </Style>

    <Style id="BlackXIcon">
      <IconStyle>
         <scale>0.2</scale>
         <Icon>
            <href>x.png</href>
         </Icon>
         <color>ff000000</color>
         <hotSpot x="0.5"  y="0.5" xunits="fraction" yunits="fraction"/>
      </IconStyle>
      <LabelStyle>
         <color>00000000</color>
      </LabelStyle>
    </Style>

    <Style id="YellowXIcon">
      <IconStyle>
         <scale>0.2</scale>
         <Icon>
            <href>x.png</href>
         </Icon>
         <color>ff00ffff</color>
         <hotSpot x="0.5"  y="0.5" xunits="fraction" yunits="fraction"/>
      </IconStyle>
      <LabelStyle>
         <color>00000000</color>
      </LabelStyle>
    </Style>
"""


point_template="""             <Point>
                     <altitudeMode>relativeToGround</altitudeMode>
                     <coordinates>{:.6f},{:.6f},1.000000</coordinates>
             </Point>
"""


timestep_header_template = """<Folder>
  <name>{date_string}:{certain}</name>
"""

one_run_header = """    <Placemark>
      <name>{certain} {status} Splots </name>
      <styleUrl>{style}</styleUrl>
      <TimeSpan id="ID">
        <begin>{start_time}</begin>     <!-- kml:dateTime -->
        <end>{end_time}</end>         <!-- kml:dateTime -->
      </TimeSpan>
      <MultiGeometry>
"""
one_run_footer = """      </MultiGeometry>
    </Placemark>
"""
timestep_footer = """
</Folder>
"""
def build_one_timestep(floating_positions,
                       beached_positions,
                       start_time,
                       end_time,
                       uncertain,
                      ):

    data = {'certain' : "Uncertainty" if uncertain else "Best Guess",
            'start_time': start_time,
            'end_time' : end_time,
            'date_string': start_time,
           }
    kml = []
    kml.append(timestep_header_template.format(**data))

    for status, positions in [('Floating',floating_positions),
                              ('Beached',beached_positions)]:
        color = "Red" if uncertain else "Yellow"
        data['style'] = "#"+color+"DotIcon" if status == "Floating" else "#"+color+"XIcon"

        data['status'] = status
        kml.append(one_run_header.format(**data))

        for point in positions:
            kml.append(point_template.format(*point[:2]))
        kml.append(one_run_footer)
    kml.append(timestep_footer)

    return "".join(kml)





footer = """
  </Document>
</kml>
"""




