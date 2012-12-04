<%namespace name="defs" file="../defs.mak"/>
<%page args="form, action_url"/>

<div class="form page hidden" id="${form.id}">
    <div class="page-header">
        <h3>Model Settings</h3>
    </div>
    <div class="page-body">
        <form action="${action_url}" class="form-horizontal model-settings" method="POST">

            ${defs.form_control(form.date, label="Model Start Date:")}
            ${defs.time_control(form, "Model Start Time",  help_text="(24-hour)")}

            <!-- Model duration control group -->
            <div class="control-group ${'error' if form.duration_days.errors or form.duration_hours.errors else ''}">
                <label class="control-label" for="duration_days">Model
                    Run Duration:</label>

                <div class="controls">
                    ${form.duration_days} days and ${form.minute} hours
                </div>
            </div>

            ${defs.form_control(form.include_minimum_regret)}
            ${defs.form_control(form.show_currents)}
            ${defs.form_control(form.computation_time_step, "hours")}
            ${defs.form_control(form.prevent_land_jumping)}
            ${defs.form_control(form.run_backwards)}
        </form>
    </div>
        <div class="control-group form-buttons">
            <div class="form-actions">
                <button class="btn cancel"> Cancel </button>
                <button class="btn btn-primary">Save</button>
            </div>
        </div>
</div>
