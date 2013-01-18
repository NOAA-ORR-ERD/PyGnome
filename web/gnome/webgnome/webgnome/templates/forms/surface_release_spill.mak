<%namespace name="defs" file="../defs.mak"/>
<%page args="spill, spill_id=None, form_id=None"/>

<div class="spill form page hide" id='${form_id}'>
    <form action="" class="form-horizontal" method="POST">
        <div class="page-body">
            <%
                name = h.text('name', spill.name)
                is_active = h.checkbox("is_active", checked=spill.is_active)
                num_elements = h.text('num_elements', spill.num_elements, 'num_elements')
                start_position_x = h.text('start_position_x', spill.start_position[0], class_="coordinate")
                start_position_y = h.text('start_position_y', spill.start_position[1], class_="coordinate")
                start_position_z = h.text('start_position_z', spill.start_position[2], class_="coordinate")
                windage_min = h.text('windage_min', spill.windage_range[0])
                windage_max = h.text('windage_max', spill.windage_range[1])
                windage_persist = h.text('windage_persist', spill.windage_persist)
            %>

            ${defs.form_control(name, label="Name")}
            ${defs.form_control(is_active, label="Active")}
            ${defs.form_control(num_elements, label="Number of LEs")}
            ${defs.datetime_control(spill.release_time, 'release_time',
                                    date_label='Release Start')}

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
