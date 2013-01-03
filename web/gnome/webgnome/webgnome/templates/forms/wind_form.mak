<%namespace name="defs" file="../defs.mak"/>
<%page args="wind, is_variable=False"/>

% if is_variable:
    ${defs.form_control(
        h.text(wind.datetime, class_='date'))}
    ${defs.time_control()}
% endif

## Are we actually getting a wind value here or what?

${defs.form_control(
    h.text(wind.direction),
    help_text='Enter cardinal direction or degrees true.')}

<div class="control-group">
    <label class="control-label">Speed</label>

    <div class="controls">
    ${h.text(wind.speed, classes="speed")}
    <span class="help"> </span>
    </div>
</div>

