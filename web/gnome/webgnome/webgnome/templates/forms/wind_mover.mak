<%namespace name="defs" file="../defs.mak"/>
<%page args="form_id"/>

<div class="wind-mover form page hide" id="${form_id}">
    <form action="" class="form-horizontal" method="POST">
        <div class="page-body">

            <%
                wind_form_id = '%s_wind' % form_id
                constant_id = '%s_constant' % wind_form_id
                variable_id = '%s_variable' % wind_form_id
            %>

            <ul class="nav nav-tabs">
                <li class="active"><a href="#${wind_form_id}" data-toggle="tab">Wind</a></li>
                <li><a href="#${form_id}_general" data-toggle="tab">General</a></li>
                <li><a href="#${form_id}_uncertainty" data-toggle="tab">Uncertainty</a></li>
                <li><a href="#${form_id}_active_range" data-toggle="tab">Active Time Range</a></li>
            </ul>

            <div class="tab-content">
                <div class="tab-pane active wind" id="${wind_form_id}">
                    <ul class="nav nav-tabs">
                        <li class="active wind-data-link"><a href="#${wind_form_id}_data" data-toggle="tab">Wind Data</a></li>
                        <li class="data-source-link"><a href="#${wind_form_id}_data_source" data-toggle="tab">Data Source</a></li>
                    </ul>

                    <div class="tab-content">
                        <div class="tab-pane active"  id="${wind_form_id}_data">
                            <div class="row">
                                <div class="span7 wind-header">
                                  ${defs.form_control(h.select('wind_id', 'new', (('new', 'New Wind'),),
                                    class_='input-medium',
                                    data_value='mover.wind_id'),
                                    label='Wind', inline=True)}
                                 <%
                                   from webgnome.util import velocity_unit_values
                                   velocity_unit_options = [(value, value) for value in velocity_unit_values]
                                %>
                                ${defs.form_control(h.select('type', 'constant', (
                                    ('constant-wind', 'Constant'),
                                    ('variable-wind', 'Variable')),
                                    class_='type input-small', data_value='wind:type'),
                                    label='Type', inline=True)}
                                ${defs.form_control(h.select('units', 'knots', velocity_unit_options,
                                    class_='units input-small', data_value='wind.units'),
                                    label='Units', inline=True)}
                                </div>
                            </div>
                            <div class="row">
                                <hr style="margin: 10px 10px 10px;"/>

                                <div id="${constant_id}" class="constant-wind">
                                    <div class="span3 add-time-forms">
                                        <div class='time-form add-time-form'>
                                                <%include file="timeseries_value.mak"/>
                                        </div>
                                    </div>

                                    <div class="span2">
                                        <div id="${constant_id}_compass"
                                             class="compass"></div>
                                    </div>
                                </div>

                                <div id="${variable_id}"
                                     class="variable-wind hidden">
                                    <div class="span3 add-time-forms">
                                        <div class='time-form add-time-form'>
                                                <%include file="add_wind_timeseries_form.mak" args="form_id=variable_id"/>
                                        </div>
                                    </div>

                                    <div class="edit-time-forms">
                                            <%include file="wind_timeseries_table.mak"/>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="tab-pane data-source" id="${wind_form_id}_data_source">
                            <%include file="wind_data_source.mak"/>
                        </div>
                    </div>
                </div>

                <div class="tab-pane general" id="${form_id}_general">
                     ${defs.form_control(h.text('name', data_value='mover.name'), label='Name')}
                     ${defs.form_control(h.checkbox('on', data_checked='mover.on'), label='On')}
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
    </form>
</div>

<!-- A template for wind drop-down items. -->
<script type="text/template" id="wind-select">
    <option value="{{ id }}">{{ name }}</option>
</script>