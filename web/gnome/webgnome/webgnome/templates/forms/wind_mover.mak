<%namespace name="defs" file="../defs.mak"/>
<%page args="form, action_url"/>

<div class="modal hide fade" id="${form.id}" tabindex="-1"
     role="dialog" aria-labelledby="modal-label" aria-hidden="true"
     data-backdrop="static">
    <div class="modal-header">
        <button type="button" class="close" data-dismiss="modal"
                aria-hidden="true">×
        </button>
        <h3 id="modal-label">Wind Mover</h3>
    </div>
    <div class="modal-body">
        <form action="${action_url}" id="wind_mover" class="form-horizontal"
              method="POST">

            <%
                settings_id = defs.uid('basic-settings', form)
                uncertainty_id = defs.uid('uncertainty', form)
            %>

            <ul class="nav nav-tabs">
                <li class="active"><a href="#${settings_id}" data-toggle="tab">Basic Settings</a></li>
                <li><a href="#${uncertainty_id}" data-toggle="tab">Uncertainty</a></li>
            </ul>

            <div class="tab-content">
                <div class="tab-pane active" id="${settings_id}">
                    ${defs.form_control(form.date)}
                    ${defs.time_control(form, "Time (24 hour)")}
                    ${defs.form_control(form.direction,
                                        'Select "Degrees true" to enter degrees')}
                    ${defs.form_control(form.direction_degrees, hidden=True)}
                    ${defs.form_control(form.speed)}
                    ${defs.form_control(form.speed_type)}
                    ${defs.form_control(form.is_active)}
                </div>
                 <div class="tab-pane" id="${uncertainty_id}">
                    ${defs.form_control(form.start_time, "hours")}
                    ${defs.form_control(form.duration, "hours")}
                    ${defs.form_control(form.speed_scale)}
                    ${defs.form_control(form.total_angle_scale)}
                    ${defs.form_control(form.total_angle_scale_type)}
                </div>
            </div>
        </form>
    </div>
    <div class="modal-footer">
        <button class="btn" data-dismiss="modal" aria-hidden="true"> Cancel </button>
        <button class="btn btn-primary">Save</button>
    </div>

    <script type="text/javascript">
            % if form:
                // Inline event handler to hide or show the direction degrees input.
                $(document).ready(function() {
                    var formEl = "#${form.id}";
                    $(formEl).on('change', '#direction', function(event) {
                        var selected_direction = $(this).val();
                        var degreesControl = $(formEl).find(
                                '#direction_degrees').closest('.control-group');

                        if (selected_direction === 'Degrees true') {
                            degreesControl.removeClass('hidden');
                        } else {
                            degreesControl.addClass('hidden');
                        }
                    });
                });
            % endif
    </script>
</div>

