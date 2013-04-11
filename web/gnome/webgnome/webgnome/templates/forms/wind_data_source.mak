<%namespace name="defs" file="../defs.mak"/>

<% from webgnome.model_manager import WebWind %>

<div class="span3">
    ${defs.form_control(h.select('source_type', 'manual',
        WebWind.source_types, class_='input-medium',
        data_value='wind.source_type'),
        label="Data Source")}
    ${defs.form_control(h.text('source_id', class_='input-small',
        data_class_required='wind:sourceIdRequired < wind.source_type',
        data_value='wind.source_id'), label='Source ID')}

## Coming soon ...
##     <div data-show='wind:isBuoy < wind.source_type'>
##     ${defs.form_control(h.select('station_id', '',
##        (('', 'None'),),  class_='input-medium',
##        data_value='wind.source_id',), label="Station ID")}
##    </div>

    ${defs.form_control(h.text('latitude', class_='input-small',
        data_class_required='wind:latitudeRequired < wind.source_type .latitude',
        data_value='wind.latitude'), label='Latitude')}
    ${defs.form_control(h.text('longitude', class_='input-small',
        data_class_required='wind:longitudeRequired < wind.source_type .longitude',
        data_value='wind.longitude'), label='Longitude')}
    ${defs.datetime_control('updated_at', date_label="Last Updated")}
    ${defs.form_control(h.textarea('description', class_='input-medium', data_value='wind.description'), label='Description')}

    <div class='control-group'>
        <div class="controls">
            <button class="btn query-source" data-disabled='wind:isManual < wind.source_type'>Get Latest</button>
        </div>
    </div>
</div>

<div class="span3">
    <div class="nws-map-container" data-show='wind:isNws < wind.source_type'>
        <div class="nws-map-canvas"></div>
    </div>
</div>
