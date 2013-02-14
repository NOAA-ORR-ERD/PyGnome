<%namespace name="defs" file="../defs.mak"/>

<div class="form page hide" id="model-settings" title="Model Settings">
    <div class="page-body">
        <form action="" class="form-horizontal model-settings" method="POST">
            <%
                duration_days = h.text('duration_days', data_value='model.duration_days', class_="input-extra-small")
                duration_hours = h.text('duration_hours', data_value='model.duration_hours', class_="input-extra-small")
                uncertain = h.checkbox('uncertain', data_checked='model.uncertain')
                computation_time_step = h.text('time_step', data_value='model.time_step', class_="input-extra-small")
            %>

            ${defs.datetime_control('start_time', date_label='Model Start Date', hour_label='Model Start Time')}

            <div class="control-group">
                <label class="control-label" for="duration_days">Model
                    Run Duration:</label>

                <div class="controls">
                    ${duration_days} days and ${duration_hours} hours
                </div>
            </div>

            ${defs.form_control(uncertain, label='Include Uncertainty Solution')}
            ${defs.form_control(computation_time_step, 'hours', label='Time Step')}

        </form>
    </div>
</div>
