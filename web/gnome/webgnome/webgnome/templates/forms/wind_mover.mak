<%namespace name="defs" file="../defs.mak"/>
<%page args="form, action_url"/>

<div class="modal wind hide fade" id="${form.id}" tabindex="-1"
     role="dialog" data-backdrop="static" data-keyboard="false">

    <form action="${action_url}" id="wind_mover" class="form-horizontal"
          method="POST">

    <div class="modal-header">
        <button type="button" class="close" data-dismiss="modal" aria-hidden="true">×
        </button>
        <span class="modal-label"><h3>${form.name.data}</h3></span>
        <span class="hidden edit-mover-name-field">${form.name}</span>
        <a class="edit-mover-name" href="#"><i class="icon-edit"></i></a>
        <a class="save-mover-name hidden" href="#"><i class="icon-ok"></i></a>
    </div>
    <div class="modal-body">
            <%
                time_series_id = defs.uid('time-series', form)
                uncertainty_id = defs.uid('uncertainty', form)
            %>

            <ul class="nav nav-tabs">
                <li class="active"><a href="#${time_series_id}" data-toggle="tab">Time Series</a></li>
                <li><a href="#${uncertainty_id}" data-toggle="tab">Uncertainty</a></li>
            </ul>

            <div class="tab-content">
                <div class="tab-pane active" id="${time_series_id}">
                    
                    <%
                        add_form = form.timeseries[-1]
                    %>

                    <div class="span6 add-time-forms">
                        <div class='time-form add-time-form'>
                            <%include file="wind_form.mak" args="form=add_form"/>
                            ${defs.form_control(form.auto_increment_time_by, "hours",
                                opts={'class_': 'auto_increment_by'})}
                            ${defs.form_control(form.is_active)}

                            <div class="control-group add-time-buttons">
                                <div class="controls">
                                    <button class="btn btn-success add-time">
                                        Add Time
                                    </button>
                                </div>
                            </div>

                            <div class="control-group edit-time-buttons hidden">
                                    <div class="controls">
                                        <button class="btn cancel">
                                            Cancel
                                        </button>
                                        <button class="btn btn-success save">
                                            Save
                                        </button>
                                    </div>
                            </div>
                        </div>
                    </div>

                    <div class="span6 edit-time-forms">
                        <table class="table table-striped time-list">
                            <thead>
                            <tr class='table-header'>
                                <th>Date (m/d/y)</th>
                                <th>Time</th>
                                <th>Speed</th>
                                <th>Wind From</th>
                                <th>&nbsp;</th>
                            </tr>
                            </thead>
                            <tbody>
                            </tbody>
                        </table>

                        % for wind_form in form.timeseries[:-1]:
                            <div class="edit-time-form time-form hidden">
                            <%include file="wind_form.mak" args="form=wind_form"/>

                                <div class="control-group edit-time-buttons">
                                    <div class="controls">
                                         <button class="btn btn-success save">
                                            Save
                                        </button>
                                        <button class="btn cancel">
                                            Cancel
                                        </button>
                                    </div>
                                </div>
                            </div>
                        % endfor
                    </div>
                </div>
                <div class="tab-pane" id="${uncertainty_id}">
                    ${defs.form_control(form.uncertain_time_delay, "hours")}
                    ${defs.form_control(form.uncertain_duration, "hours")}
                    ${defs.form_control(form.uncertain_speed_scale)}
                    ${defs.form_control(form.uncertain_angle_scale)}
                    ${defs.form_control(form.uncertain_angle_scale_type)}
                </div>
            </div>
    </div>
    <div class="modal-footer">
        <button class="btn" data-dismiss="modal" aria-hidden="true"> Cancel </button>
        <button class="btn btn-primary">Save</button>
    </div>

    <!-- A template for time series item rows. -->
     <script type="text/template" id="time-series-row">
         <tr class="{{ error }}">
             <td class="time-series-date">{{ date }}</td>
             <td class="time-series-time">{{ time }}</td>
             <td class="time-series-speed">{{ speed }}</td>
             <td class="time-series-direction">{{ direction }}</td>
             <td><a href="#" class="edit-time"><i class="icon-edit"></i></a>
                 <a href="#" class="delete-time"><i class="icon-trash"></i></a>
             </td>
         </tr>
     </script>

    </form>
</div>

