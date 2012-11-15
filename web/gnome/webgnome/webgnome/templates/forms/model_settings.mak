<%namespace name="defs" file="../defs.mak"/>
<%page args="form, action_url"/>

<div class="modal hide fade" id="${form.id}" tabindex="-1"
     data-backdrop="static" role="dialog" aria-labelledby="modal-label"
     aria-hidden="true">
    <div class="modal-header">
        <button type="button" class="close" data-dismiss="modal"
                aria-hidden="true">×
        </button>
        <h3 id="modal-label">Model Settings</h3>
    </div>
    <div class="modal-body">
        <form action="${action_url}" class="form-horizontal multistep" method="POST">

             <ul class="nav nav-tabs">
                <li class="active"><a href="#model_start_time" data-toggle="tab">
                    Start Time and Duration</a></li>
                <li><a href="#model_additional_settings" data-toggle="tab">
                    Additional Settings</a></li>
            </ul>

            <div class="tab-content">
                <div class="tab-pane active" id="model_start_time">
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
                </div>

                <div class="tab-pane" id="model_additional_settings">
                    ${defs.form_control(form.include_minimum_regret)}
                    ${defs.form_control(form.show_currents)}
                    ${defs.form_control(form.computation_time_step, "hours")}
                    ${defs.form_control(form.prevent_land_jumping)}
                    ${defs.form_control(form.run_backwards)}
                </div>
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
