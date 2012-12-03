### Mako defs.

## Make a unique element ID by combining ``form.id`` and ``id``.
<%def name="uid(id, form)"><% return "%s_%s" % (id, form.id) %></%def>

<%def name="is_active(url)">
    % if request.path == url:
        active
    % endif
</%def>

## Render a Bootstrap form control for ``field``.
<%def name="form_control(field, help_text=None, label=None, hidden=False,
                         extra_classes=None, use_id=False, opts=None)">
    <div class="control-group  ${'error' if field.errors else ''}
                % if hidden and not field.errors:
                    hidden
                % endif
                % if extra_classes:
                    % for cls in extra_classes:
                        ${cls}
                    % endfor
                % endif
                ">
        <label class="control-label">
            % if label is None:
                ${field.label.text}
            % else:
                ${label}
            % endif
        </label>

        <div class="controls">
            <%
                # Use blank IDs for form controls by default to avoid stomping
                # on reusable form components.
                if use_id is False and opts and 'id' not in opts:
                    opts['id'] = ''
            %>
            ${field(**opts) if opts else field(id='')}
            % if help_text:
                <span class="help-inline">${help_text | n}</span>
            % endif
             % if field.errors:
                <span class="help-inline">
                    ${field.errors[0]}
                </span>
            % endif
        </div>
    </div>
</%def>

## Render Bootstrap form controls for the fields on a
## :class:`webgnome.forms.base.DateTimeForm` form.
"""
<%def name="time_control(form, hour_label='Time (24-hour): ', minute_label='', help_text='')">
    <div class="control-group ${'error' if form.hour.errors or form.minute.errors else ''}">
        % if hour_label:
            <label class="control-label">${hour_label}</label>
        % endif

        <div class="controls">
        ${form.hour(id='', class_='hour')} : ${form.minute(id='', class_="minute")}
            % if help_text:
                <span class="help-inline">${help_text}</span>
            % endif
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
