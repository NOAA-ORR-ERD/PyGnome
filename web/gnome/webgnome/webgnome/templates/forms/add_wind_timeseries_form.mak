<%namespace name="defs" file="../defs.mak"/>
<%page args="form_id"/>

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

<div class="compass-container hidden">
    <div id="${form_id}_compass" class="compass"></div>
</div>