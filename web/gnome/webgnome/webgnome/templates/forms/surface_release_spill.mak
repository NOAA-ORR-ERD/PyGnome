<%namespace name="defs" file="../defs.mak"/>
<%page args="form_id=None"/>

<div class="spill form page hide" id='${form_id}'>
    <form action="" class="form-horizontal" method="POST">
        <div class="page-body">
            <%
                name = h.text('name', data_value='spill.name')
                is_active = h.checkbox("is_active", data_checked='spill.is_active')
                num_elements = h.text('num_elements', data_value='spill.num_elements')
                start_position_x = h.text('start_position_x', data_value='spill:start_position_x < .start_position', class_="coordinate")
                start_position_y = h.text('start_position_y', data_value='spill:start_position_y < .start_position', class_="coordinate")
                start_position_z = h.text('start_position_z', data_value='spill:start_position_z < .start_position', class_="coordinate")
                windage_min = h.text('windage_min', data_value='spill:windage_range_min < .windage_range')
                windage_max = h.text('windage_max', data_value='spill:windage_range_max < .windage_range')
                windage_persist = h.text('windage_persist', id=None, data_value='spill.windage_persist')
            %>

            ${defs.form_control(name, label="Name")}
            ${defs.form_control(is_active, label="Active")}
            ${defs.form_control(num_elements, label="Number of Elements")}
            ${defs.datetime_control('release_time', date_label='Release Start')}

            <div class="control-group">
                <label class="control-label">Start Position (X, Y, Z)</label>

                <div class="controls start-coordinates">
                    ${start_position_x}
                    ${start_position_y}
                    ${start_position_z}
                    <span class="help"> </span>
                </div>
            </div>

            ${defs.form_control(windage_min, label="Windage Min")}
            ${defs.form_control(windage_max, label="Windage Max")}
            ${defs.form_control(windage_persist, label="Windage persist")}
        </div>
    </form>
</div>
