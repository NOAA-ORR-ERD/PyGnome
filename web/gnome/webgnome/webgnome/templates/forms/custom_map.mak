<%namespace name="defs" file="../defs.mak"/>

<div class="custom-map form page hide" id="add-custom-map" title="Add Custom Map">
    <form action="" class="form-horizontal" method="POST">
        <div class="page-body">
            <div class="col-md-5">
                <%
                    name = h.text('name', data_value='map.name', class_='input-small')
                    refloat_halflife = h.text('refloat_halflife',
                                              data_value='map.refloat_halflife',
                                              class_='input-small')
                %>

                ${defs.form_control(name, label="Name")}
                ${defs.form_control(refloat_halflife, label="Refloat Halflife",
                                    help_text="hours")}
                ${defs.form_control(h.text('north_lat', class_='input-small', data_value='map.north_lat'), label='North Latitude')}
                ${defs.form_control(h.text('east_lon', class_='input-small', data_value='map.east_lon'), label='East Longitude')}
                ${defs.form_control(h.text('south_lat', class_='input-small', data_value='map.south_lat'), label='South Latitude')}
                ${defs.form_control(h.text('west_lon', class_='input-small', data_value='map.west_lon'), label='West Longitude')}
                ${defs.form_control(h.checkbox('cross_dateline', class_='input-small', data_value='map.cross_dateline'), label='Cross Dateline?')}
                <%
                    options = [('c', 'Course'), ('l', 'Low'), ('i', 'Intermediate'),
                                ('h', 'High'), ('f', 'Full')]
                %>
                ${defs.form_control(h.select('resolution', 'i', options, data_value='map.resolution', class_='map-resolution'),
                                    label='Resolution')}
            </div>
            <div class="col-md-6">
                <div id="custom-map"></div>
            </div>
        </div>
    </form>
</div>
