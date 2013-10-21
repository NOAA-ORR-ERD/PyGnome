### Mako defs.

<%def name="is_active(url)">
    <%doc>
         Return True if the current request path is equal to ``url``.
    </%doc>
    % if request.path == url:
        active
    % endif
</%def>


<%def name="btn(text, classes='', function_name='', deferred=False)">
    <button type="button" class="ui-button ui-widget ui-corner-all ui-button-text-only ${classes}"
            ${'data-function-name="%s"' % function_name if function_name else ''}
            data-deferred="${'true' if deferred else 'false'}" role="button">
        <span class="ui-button-text">
            ${text}
        </span>
    </button>
</%def>


<%def name="cancel_btn(text='Cancel', **kwargs)">
    ${btn(text, 'cancel', **kwargs)}
</%def>


<%def name="next_btn(text='Next', **kwargs)">
    ${btn(text, 'next ui-button-primary', **kwargs)}
</%def>


<%def name="back_btn(text='Back', **kwargs)">
    ${btn(text, 'back ui-button-primary', **kwargs)}
</%def>


<%def name="finish_btn(text='To the map window', **kwargs)">
    ${btn(text, 'finish ui-button-primary', **kwargs)}
</%def>


<%def name="references_btn(text='Show references', **kwargs)">
    ${btn(text, 'references ui-button-primary', **kwargs)}
</%def>


<%def name="buttons()">
    <div class="ui-dialog-buttonset custom-dialog-buttons hidden">
        ${caller.body()}
    </div>
</%def>


<%def name="step(height=None, width=None, show_form=None)">
    <%doc>
        Render a singles "step" in a multi-step form.

        Specifying a ``height`` or ``width`` will add these as data- attributes,
        and the JavaScript application will later set the height on the jQuery
        UI Dialog for this form to the correct size for this step.

        Specifying the name of a form that exists in the app for ``show_form``
        will show that form as the step, hiding the current form until the user
        either submits or cancels the form named by ``show_form``. The value
        of ``show_form`` should be the ID of the form.
    </%doc>
    <div class="step hidden"
        ${'data-height=%s' % height if height else ''}
        ${'data-width=%s' % width if width else ''}
        ${'data-show-form=%s' % show_form if show_form else ''}>

        ## Let the caller use this def like a tag with body content.
        ${caller.body()}
    </div>
</%def>


<%def name="form_control(field, field_id=None, help_text=None,
                         label=None, label_class=None,
                         hidden=False, extra_classes=None, inline=False)">
    <%doc>
        Render a Bootstrap form control around ``field``.
    </%doc>
    % if inline:
        <span class="control-group ${'hidden' if hidden else ''} ${'form-inline' if inline else ''}">
             % if label:
                <label class="${label_class}"> ${label} </label>
            % endif

            ${field}

            <span class="help-inline">
                % if help_text:
                    ${help_text | n}
                % endif
                <a href="#" class="glyphicon glyphicon-warning-sign error" title="error"></a>
            </span>
        </span>
    % else:
        <div class="form-group ${'hidden' if hidden else ''}
                    % if extra_classes:
                        % for cls in extra_classes:
                            ${cls}
                        % endfor
                    % endif
                    ">

            % if label:
                % if field_id:
                    <label class="col-md-2 control-label" for="${field_id}"> ${label} </label>
                % else:
                    <label class="col-md-2 control-label"> ${label} </label>
                % endif
            % endif

            <div class="col-md-6">
                ${field | n}
                <span class="help-inline">
                     % if help_text:
                        ${help_text | n}
                     % endif

                    <a href="#" class="glyphicon glyphicon-warning-sign error" title="error"></a>
                </span>
            </div>
        </div>
    % endif
</%def>


<%def name="datetime_control(date_name, value=None, date_label=None,
                             date_class='date input-small',
                             date_help_text=None, hour_value=None,
                             hour_label='Time (24-hr): ', hour_name='hour', hour_class='hour',
                             minute_value=None, minute_label=None,
                             minute_class='minute', minute_name='minute',
                             time_help_text=None, date_id=None)">
    <%doc>
        Render a date input for ``value``, by splitting it into a date input
        and a set of hour and minute time inputs.
    </%doc>
    <%
        hour = value.hour if value and hasattr(value, 'hour') else None
        minute = value.minute if value and hasattr(value, 'minute') else None
        field = h.text(date_name, value=value, class_=date_class, id=date_id)
    %>

    <div class="${date_name}_container">
     ${form_control(field, help_text=date_help_text, label=date_label)}
     ${time_control(hour_value if hour_value else hour,
                    minute_value if minute_value else minute,
                    hour_label, hour_name, hour_class, minute_label,
                    minute_name, minute_class, time_help_text)}
    </div>
</%def>


<%def name="time_control(hour=None, minute=None, hour_label='Time (24-hour): ',
                         hour_name='hour', hour_class='hour', minute_label=None,
                         minute_name='minute', minute_class='minute',
                         help_text=None)">
    <%doc>
        Render a Bootstrap form control for a :class:`datetime.datetime` value,
        displaying only the time values (hour and minute).
    </%doc>
    <div class="form-group">
        % if hour_label:
            <label class="col-md-2 control-label">${hour_label}</label>
        % endif

        <div class="controls">
        ${h.text(hour_label, value=hour, class_=hour_class)} : ${h.text(minute_label, value=minute, class_=minute_class)}
            <span class="help-inline">
                    % if help_text:
                    ${help_text}
                    % endif
            </span>
            <span class="help-inline">
                % if help_text:
                    ${help_text}
                % endif
            </span>
        </div>
    </div>
</%def>
