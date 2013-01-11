<%namespace name="defs" file="../defs.mak"/>
<%page args="model"/>

<div class="form page hide" id="model_settings" title="Model Settings">
    <div class="page-body">
        <form action="" class="form-horizontal model-settings" method="POST">
            <%
                start_time = h.text('start_time', model.start_time, class_="date")
                duration_days = h.text('duration_days', model.duration_days, class_="input-extra-small")
                duration_hours = h.text('duration_hours', model.duration_hours, class_="input-extra-small")
                uncertain = h.checkbox('uncertain', checked=model.uncertain)
                computation_time_step = h.text('time_step', model.time_step, class_="input-extra-small")
            %>

            ${defs.form_control(start_time, label="Model Start Date:")}
            ${defs.time_control("Model Start Time",  help_text="(24-hour)")}

            <div class="control-group">
                <label class="control-label" for="duration_days">Model
                    Run Duration:</label>

                <div class="controls">
                    ${duration_days} days and ${duration_hours} hours
                </div>
            </div>

            ${defs.form_control(uncertain, label='Uncertain')}
            ${defs.form_control(computation_time_step, 'hours', label='Time Step')}

        </form>
    </div>
</div>
