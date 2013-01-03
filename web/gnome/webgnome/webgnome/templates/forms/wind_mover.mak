<%namespace name="defs" file="../defs.mak"/>
<%page args="mover"/>

<div class="wind form page hide" id="wind_mover"
     data-id="${mover_id if mover_id else ''}">

    <form action="" class="form-horizontal" method="POST">

    <div class="page-header form-inline">
        ${h.text('name', mover.name)}
        <%
            from webgnome.util import velocity_unit_options
            units = mover.wind['units'] if mover.wind else 'mps'
        %>
        <label>Units ${h.select('units', units, velocity_unit_options, class_='units')}</label>
        <label class="checkbox">${h.checkbox('is_active', checked=mover.is_active)} Active </label>
    </div>
    <div class="page-body">
        <%
            is_constant = hasattr(mover, 'id') is False or mover.is_constant
            is_variable = is_constant is False
            winds = mover.wind.timeseries if mover.wind else []
        %>

        <ul class="nav nav-tabs">
            <li class="${"active" if is_constant else ""}">
                <a href="#constant" data-toggle="tab">Constant Wind</a>
            </li>
            <li class="${"active" if is_variable else ""}">
                <a href="#variable" data-toggle="tab">Variable Wind</a>
            </li>
            <li><a href="#uncertainty" data-toggle="tab">Uncertainty</a></li>
        </ul>

        <div class="tab-content">
            <div class="tab-pane constant-wind ${"active" if is_constant else ""}"
                 id="constant">

                <div class="span4 add-time-forms">
                    <%
                        wind = winds[0] if winds else default_wind
                    %>
                    <div class='time-form add-time-form'>
                        ${h.checkbox("is_constant", checked=True, class_="hidden")}
                        <%include file="wind_form.mak" args="wind=wind"/>
                    </div>
                </div>
                <div class="compass-container span3 offset1">
                    <div id="compass_add"
                         class="compass"></div>
                </div>
            </div>

            <div class="tab-pane variable-wind ${"active" if is_variable else ""}"
                 id="variable">

                <div class="span4 add-time-forms">
                    <div class='time-form add-time-form'>
                        <%
                            auto_increment_by = h.text('auto_increment_by',
                                                        class_='auto_increment_by')
                        %>
                        <%include file="wind_form.mak" args="wind=default_wind, is_variable=True"/>
                        ${defs.form_control(auto_increment_by, "hours")}

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

                    <div class="compass-container offset1">
                        <div id="compass_edit"
                             class="compass"></div>
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

            <div class="tab-pane" id="uncertainty">
                <%
                    uncertain_time_delay = h.text('uncertain_time_delay', mover.uncertain_time_delay)
                    uncertain_duration = h.text('uncertain_duration', mover.uncertain_duration)
                    uncertain_speed_scale = h.text('uncertain_speed_scale', mover.uncertain_speed_scale)
                    uncertain_angle_scale = h.text('uncertain_angle_scale', mover.uncertain_angle_scale)
                    uncertain_angle_scale_units = h.text('uncertain_angle_scale_units', mover.uncertain_angle_scale_units)
                %>

                ${defs.form_control(uncertain_time_delay, "hours")}
                ${defs.form_control(uncertain_duration, "hours")}
                ${defs.form_control(uncertain_speed_scale)}
                ${defs.form_control(uncertain_angle_scale)}
                ${defs.form_control(uncertain_angle_scale_units)}
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

