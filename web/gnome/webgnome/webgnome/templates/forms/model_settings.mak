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
            <div data-step="1" class="step active">
                 <fieldset>
                    <legend>Model Start Time and Duration</legend>
                     ${defs.form_control(form.date, "Model Start Date:")}
                     ${defs.time_control(form, "Model Start Time: (24-hour)")}

                     <!-- Model duration control group -->
                    <div class="control-group ${'error' if form.duration_days.errors or form.duration_hours.errors else ''}">
                    <label class="control-label" for="duration_days">Model Run Duration:</label>
                    <div class="controls">
                        ${form.duration_days} days and ${form.minute} hours
                    </div>
                </fieldset>
            </div>

            <div data-step="2" class="hidden step">
                <fieldset>
                    <legend>Additional Settings</legend>
                     ${defs.form_control(form.include_minimum_regret)}
                     ${defs.form_control(form.show_currents)}
                     ${defs.form_control(form.computation_time_step, "hours")}
                     ${defs.form_control(form.prevent_land_jumping)}
                     ${defs.form_control(form.run_backwards)}
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
