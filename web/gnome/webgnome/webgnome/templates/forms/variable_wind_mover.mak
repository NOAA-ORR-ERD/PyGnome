<%namespace name="defs" file="../defs.mak"/>
<%page args="form, action_url"/>

<div class="modal hide fade" id="variable_wind_mover" tabindex="-1"
     role="dialog" aria-labelledby="modal-label" aria-hidden="true"
     data-backdrop="static">
    <div class="modal-header">
        <button type="button" class="close" data-dismiss="modal"
                aria-hidden="true">×
        </button>
        <h3 id="modal-label">Variable Wind Mover</h3>
    </div>
    <div class="modal-body">
        <form action="${action_url}" class="form-horizontal multistep" method="POST">
            ${form.type}
            <div data-step="1" class="step active">
                ${defs.form_control(form.date)}
                ${defs.time_control(form, "Time (24 hour)")}
                ${defs.form_control(form.direction, 'Enter degrees true or text (e.g., "NNW").')}
                ${defs.form_control(form.speed)}
                ${defs.form_control(form.speed_type)}
            </div>

            <div data-step="2" class="hidden step">
                <fieldset>
                    <legend>Settings</legend>
                    ${defs.form_control(form.is_active)}
                </fieldset>
            </div>

            <div data-step="3" class="hidden step">
                <fieldset>
                    <legend>Uncertainty</legend>
                    ${defs.form_control(form.start_time, "hours")}
                    ${defs.form_control(form.duration, "hours")}
                    ${defs.form_control(form.speed_scale)}
                    ${defs.form_control(form.total_angle_scale)}
                    ${defs.form_control(form.total_angle_scale_type)}
                </fieldset>
            </div>
        </form>
    </div>
    <div class="modal-footer">
        <button class="btn" data-dismiss="modal" aria-hidden="true"> Cancel </button>
        <button class="btn btn-prev hidden">Previous</button>
        <button class="btn btn-next">Next</button>
        <button class="btn btn-primary hidden">Save</button>
    </div>
</div>

