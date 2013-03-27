<%namespace name="defs" file="../defs.mak"/>
<%page args="form_id"/>

<div class="wind form page hide" id="${form_id}">
    <form action="" class="form-horizontal" method="POST">
        <div class="wind-mover-header clearfix">
            ${defs.form_control(h.text('name', data_value='mover.name'),
                                label='Name', inline=True)}
            <%
                from webgnome.util import velocity_unit_values
                velocity_unit_options = [(value, value) for value in velocity_unit_values]
            %>
            ${defs.form_control(h.select('type', 'constant', (
                                        ('constant-wind', 'Constant'),
                                        ('variable-wind', 'Variable')),
                                        class_='type input-small', data_value='mover:type'),
                                label='Type', inline=True)}
            ${defs.form_control(h.select('units', 'knots', velocity_unit_options,
                                         class_='units input-small', data_value='wind.units'),
                                label='Units', inline=True)}
            ${defs.form_control(h.checkbox('on', data_checked='mover.on'),
                                label_class='checkbox', label='On', inline=True)}
        </div>
    <div class="page-body">
        <ul class="nav nav-tabs">
            <li class="active wind-data-link"><a href="#${form_id}_wind" data-toggle="tab">Wind Data</a></li>
            <li class="data-source-link"><a href="#${form_id}_data_source" data-toggle="tab">Data Source</a></li>
            <li><a href="#${form_id}_uncertainty" data-toggle="tab">Uncertainty</a></li>
            <li><a href="#${form_id}_active_range" data-toggle="tab">Active Time Range</a></li>
        </ul>

        <div class="tab-content">
            <div class="tab-pane active wind" id="${form_id}_wind">
                <div class="constant-wind">
                    <div class="span3 add-time-forms">
                        <div class='time-form add-time-form'>
                        <%include file="timeseries_value.mak"/>
                        </div>
                    </div>

                    <div class="span2">
                        <div id="${form_id}_compass_add_constant" class="compass"></div>
                    </div>
                </div>

                <div class="variable-wind hidden">
                    <div class="span3 add-time-forms">
                        <div class='time-form add-time-form'>
                            <%
                                auto_increment_by = h.text('auto_increment_by', 6,
                                                        class_='auto_increment_by')
                            %>
                            <%include file="timeseries_value.mak", args="is_variable=True, compass_link=True"/>
                            ${defs.form_control(auto_increment_by, "hours", label="Auto-increment By")}

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

                    <div class="edit-time-forms">
                        <div class="wind-values">
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
                        </div>
                    </div>
                </div>
            </div>

            <div class="tab-pane data-source" id="${form_id}_data_source">
                <div class="span4">
                    <% from webgnome.model_manager import WebWind %>
                    ${defs.form_control(h.select('source_type', 'manual',
                                                 WebWind.source_types, label='Source Type',
                                                 class_='input-medium', data_value='wind.source_type'),
                                        label="Data Source")}
                    ${defs.form_control(h.text('source_id', class_='input-small', data_class_required='wind:isBuoy < wind.source_type',
                                        data_value='wind.source_id'), label='Source ID')}
                    ${defs.form_control(h.text('latitude', class_='input-small', data_class_required='wind:isNws < wind.source_type',
                                        data_value='wind.latitude'), label='Latitude')}
                    ${defs.form_control(h.text('longitude', class_='input-small', data_class_required='wind:isNws < wind.source_type',
                                        data_value='wind.longitude'), label='Longitude')}
                    ${defs.datetime_control('updated_at', date_label="Last Updated")}
                    ${defs.form_control(h.textarea('description', class_='input-medium', data_value='wind.description'), label='Description')}
                    <div class='control-group'>
                        <div class="controls">
                            <button class="btn query-source" data-disabled='wind:isManual < wind.source_type'>Get Latest</button>
                        </div>
                    </div>
                </div>

                <div class="nws-map-container">
                    <div class="nws-map-canvas"></div>
                </div>
            </div>

            <div class="tab-pane uncertainty" id="${form_id}_uncertainty">
                <%
                    uncertain_time_delay = h.text('uncertain_time_delay', data_value='mover.uncertain_time_delay')
                    uncertain_duration = h.text('uncertain_duration', data_value='mover.uncertain_duration')
                    uncertain_speed_scale = h.text('uncertain_speed_scale', data_value='mover.uncertain_speed_scale')
                    uncertain_angle_scale = h.text('uncertain_angle_scale', data_value='mover.uncertain_angle_scale')
                    uncertain_angle_scale_units = h.select('uncertain_angle_scale_units', 'rad',
                                                            (('rad', 'Radians'), ('deg', 'Degrees')),
                                                            data_value='mover.uncertain_angle_scale_units')
                %>

                ${defs.form_control(uncertain_time_delay, "hours", label="Time Delay")}
                ${defs.form_control(uncertain_duration, "hours", label="Duration")}
                ${defs.form_control(uncertain_speed_scale, label="Speed Scale")}
                ${defs.form_control(uncertain_angle_scale, label="Angle Scale")}
                ${defs.form_control(uncertain_angle_scale_units, label="Angle Scale Units")}
            </div>
            <div class="tab-pane active-range" id="${form_id}_active_range">
                ${defs.datetime_control('active_start', date_label="Active Start")}
                ${defs.datetime_control('active_stop', date_label="Active Stop")}
            </div>
        </div>
    </div>

    <div class="compass-container">
        <div id="${form_id}_compass_add" class="compass"></div>
    </div>

    <!-- A template for time series item rows. -->
     <script type="text/template" id="time-series-row">
         <tr class="{{ error }}">
             <td class="time-series-date">{{ date }}</td>
             <td class="time-series-time">{{ time }}</td>
             <td class="time-series-speed">{{ speed }}</td>
             <td class="time-series-direction">{{ direction }}</td>
             <td><a href="javascript:" class="edit-time"><i class="icon-edit"></i></a>
                 <a href="javascript:" class="delete-time"><i class="icon-trash"></i></a>
             </td>
         </tr>
     </script>

    </form>
</div>

