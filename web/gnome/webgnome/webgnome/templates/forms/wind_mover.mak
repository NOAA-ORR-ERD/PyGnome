<%namespace name="defs" file="../defs.mak"/>
<%page args="mover, default_wind, default_wind_value, form_id"/>

<div class="wind form page hide" id="${form_id}">
    <form action="" class="form-horizontal" method="POST">
        <div class="page-header form-inline">
            <label>Name</label> ${h.text('name', mover.name, class_='input-small')}
            <%
                from webgnome.util import velocity_unit_options
                units = mover.wind.units if mover.wind else default_wind.units
            %>
            <label>Type</label> ${h.select('type', 'constant', (
                                           ('constant-wind', 'Constant'),
                                           ('variable-wind', 'Variable')),
                                           class_='type input-small')}
            <label>Units</label> ${h.select('units', units, velocity_unit_options, class_='units')}
            <label>Active from</label>${h.text('is_active_start', mover.is_active_start, class_='date input-small')}
            <label>Until</label>${h.text('is_active_end', mover.is_active_stop, class_='date input-small')}
        </div>
    <div class="page-body">
        <%
            winds = mover.wind.timeseries if mover.wind else []
        %>

        <ul class="nav nav-tabs">
            <li class="active">
                <a href="#${form_id}_wind" data-toggle="tab">Wind Data</a>
            </li>
            <li><a href="#${form_id}_uncertainty" data-toggle="tab">Uncertainty</a></li>
        </ul>

        <div class="tab-content">
            <div class="tab-pane active wind" id="${form_id}_wind">
                <div class="constant-wind">
                    <div class="span4 add-time-forms">
                        <%
                            wind = winds[0] if winds else default_wind_value
                        %>
                        <div class='time-form add-time-form'>
                        <%include file="wind_form.mak" args="wind=wind"/>
                        </div>
                    </div>
                    <div class="compass-container span3 offset1">
                        <div id="${form_id}_compass_add" class="compass"></div>
                    </div>
                </div>

                <div class="variable-wind hidden">
                    <div class="span4 add-time-forms">
                        <div class='time-form add-time-form'>
                            <%
                                auto_increment_by = h.text('auto_increment_by', 6,
                                                        class_='auto_increment_by')
                            %>
                            <%include file="wind_form.mak" args="wind=default_wind_value, is_variable=True"/>
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

                        <div class="compass-container">
                            <div id="${form_id}_compass_edit" class="compass"></div>
                        </div>
                    </div>

                    <div class="span4 edit-time-forms">
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
            <div class="tab-pane uncertainty" id="${form_id}_uncertainty">
                <%
                    uncertain_time_delay = h.text('uncertain_time_delay', mover.uncertain_time_delay)
                    uncertain_duration = h.text('uncertain_duration', mover.uncertain_duration)
                    uncertain_speed_scale = h.text('uncertain_speed_scale', mover.uncertain_speed_scale)
                    uncertain_angle_scale = h.text('uncertain_angle_scale', mover.uncertain_angle_scale)
                    uncertain_angle_scale_units = h.select('uncertain_angle_scale_units',
                                                            mover.uncertain_angle_scale_units,
                                                            (('rad', 'Radians'), ('deg', 'Degrees')))
                %>

                ${defs.form_control(uncertain_time_delay, "hours", label="Time Delay")}
                ${defs.form_control(uncertain_duration, "hours", label="Duration")}
                ${defs.form_control(uncertain_speed_scale, label="Speed Scale")}
                ${defs.form_control(uncertain_angle_scale, label="Angle Scale")}
                ${defs.form_control(uncertain_angle_scale_units, label="Angle Scale Units")}
            </div>
        </div>
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

