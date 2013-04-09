<%namespace name="defs" file="../defs.mak"/>
<%page args="form_id"/>

<ul class="nav nav-tabs">
    <li class="active wind-data-link"><a href="#${form_id}_wind" data-toggle="tab">Wind Data</a></li>
    <li class="data-source-link"><a href="#${form_id}_data_source" data-toggle="tab">Data Source</a></li>
</ul>

<div class="tab-content">
    <div class="tab-pane active wind" id="${form_id}_wind">
        <%
            constant_id = '%s_constant' % form_id
            variable_id = '%s_variable' % form_id
        %>
        <div id="${constant_id}" class="constant-wind">
            <div class="span3 add-time-forms">
                <div class='time-form add-time-form'>
                    <%include file="timeseries_value.mak"/>
                </div>
            </div>

            <div class="span2">
                <div id="${constant_id}_compass" class="compass"></div>
            </div>
        </div>

        <div id="${variable_id}" class="variable-wind hidden">
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

    <div class="tab-pane data-source" id="${form_id}_data_source">
        <%include file="wind_data_source.mak"/>
    </div>
</div>