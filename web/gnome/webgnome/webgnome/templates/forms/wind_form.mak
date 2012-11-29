<%namespace name="defs" file="../defs.mak"/>
<%page args="form"/>

${defs.form_control(form.date, opts={'class_': 'date'}, use_id=True)}
${defs.time_control(form, "Time (24 hour)")}
${defs.form_control(form.direction, 'Select "Degrees true" to enter degrees',
                    opts={'class_': 'direction'})}
${defs.form_control(form.direction_degrees, hidden=True,
                    help_text='<a href="javascript:" class="show-compass">Use Compass</a>',
                    opts={'class_': 'direction_degrees'})}


<div class="control-group ${'error' if form.speed.errors or form.speed_type.errors else ''}">
    <label class="control-label">Speed</label>

    <div class="controls">
    ${form.speed(class_='speed')} ${form.speed_type(class_='speed_type')}
    % if form.speed.errors:
        <span class="help">
        ${form.speed.errors[0]}
        </span>
    % endif
    % if form.speed_type.errors:
        <span class="help">
        ${form.speed_type.errors[0]}
        </span>
    % endif
    </div>
</div>

