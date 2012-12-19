<%namespace name="defs" file="../defs.mak"/>
<%page args="form, action_url"/>

<div class="wind form hidden" id="${form.id}">

    <form action="${action_url}" id="wind_mover" class="form-horizontal"
          method="POST">

    <div class="page-header form-inline">
        ${form.name}
        <label>Units ${form.units(class_='units', id='')} </label>
        <label class="checkbox">${form.is_active} Active </label>
    </div>
    <div class="page-body">
        <%
            constant_id = defs.uid('constant', form)
            variable_id = defs.uid('variable', form)
            uncertainty_id = defs.uid('uncertainty', form)
            is_constant = form.instance is None or form.is_constant.data is True
            is_variable = is_constant is False
        %>

        <ul class="nav nav-tabs">
            <li class="${"active" if is_constant else ""}">
                <a href="#${constant_id}" data-toggle="tab">Constant Wind</a>
            </li>
            <li class="${"active" if is_variable else ""}">
                <a href="#${variable_id}" data-toggle="tab">Variable Wind</a>
            </li>
            <li><a href="#${uncertainty_id}" data-toggle="tab">Uncertainty</a></li>
        </ul>

        <div class="tab-content">
                <div class="tab-pane constant-wind ${"active" if is_constant else ""}"
                     id="${constant_id}">
                <%
                    add_form = form.timeseries[0]
                %>

                <div class="span6 add-time-forms">
                    <div class='time-form add-time-form'>
                        ${form.is_constant(class_='hidden')}
                        <%include file="wind_form.mak" args="form=add_form"/>
                    </div>
                </div>
                <div class="compass-container span6">
                    <div id="${defs.uid('compass_add', form)}" class="compass"></div>
                </div>
            </div>

            <div class="tab-pane variable-wind ${"active" if is_variable else ""}"
                 id="${variable_id}">
                <%
                    add_form = form.timeseries[-1]
                %>

                <div class="span6 add-time-forms">
                    <div class='time-form add-time-form'>
                        <%include file="wind_form.mak" args="form=add_form, is_variable=True"/>
                        ${defs.form_control(form.auto_increment_time_by, "hours",
                                            opts={'class_': 'auto_increment_by'})}

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

                    <div class="compass-container offset3">
                        <div id="${defs.uid('compass_edit', form)}" class="compass"></div>
                    </div>
                </div>

                <div class="span6 edit-time-forms">

                    <div class="span11 wind-values">
                        <hr>
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

                    % for wind_form in form.timeseries[:-1]:
                        <div class="edit-time-form time-form hidden">
                        <%include file="wind_form.mak" args="form=wind_form, is_variable=True"/>

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

        <div class="control-group form-buttons">
            <div class="form-actions">
                <button class="btn cancel"> Cancel </button>
                <button class="btn btn-primary">Save</button>
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

