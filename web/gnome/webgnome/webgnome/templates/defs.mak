### Mako defs.

<%def name="is_active(url)">
    % if request.path == url:
        active
    % endif
</%def>

<%def name="form_control(field, help_text=None)">
    <div class="control-group ${'error' if field.errors else ''}">
        <label class="control-label">${field.label.text}</label>

        <div class="controls">
            ${field}
            % if help_text:
                <span class="help-inline">${help_text}</span>
            % endif
             % if field.errors:
                <span class="help-inline">
                    ${field.errors[0]}
                </span>
            % endif
        </div>
    </div>
</%def>

<%def name="time_control(form, hour_label='Time (24-hour): ', minute_label='')">
    <div class="control-group ${'error' if form.hour.errors or form.minute.errors else ''}">
        % if hour_label:
            <label class="control-label" for="hour">${hour_label}</label>
        % endif

        <div class="controls">
        ${form.hour} : ${form.minute}
            % if form.hour.errors:
                    <span class="help">
                    ${form.hour.errors[0]}
                    </span>
            % endif
            % if form.minute.errors:
                    <span class="help">
                    ${form.minute.errors[0]}
                    </span>
            % endif
        </div>
    </div>
</%def>
