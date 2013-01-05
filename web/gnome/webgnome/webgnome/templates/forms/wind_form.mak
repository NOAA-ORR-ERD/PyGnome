<%namespace name="defs" file="../defs.mak"/>
<%page args="wind, is_variable=False"/>

% if is_variable:
    ${defs.form_control(
        h.text('datetime', wind.datetime, class_='date'), label="Date")}
    ${defs.time_control()}
% endif

${defs.form_control(
    h.text('direction', wind.direction, class_="direction"),
    label='Direction',
    help_text='Enter cardinal direction or degrees true.')}

<div class="control-group">
    <label class="control-label">Speed</label>

    <div class="controls">
    ${h.text('speed', wind.speed, class_="speed")}
    <span class="help"> </span>
    </div>
</div>

