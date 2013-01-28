<%namespace name="defs" file="../defs.mak"/>
<%page args="wind, is_variable=False"/>

% if is_variable:
    ${defs.datetime_control(wind.datetime, 'datetime', date_label="Date")}
% endif

${defs.form_control(
    h.text('direction', wind.direction, class_="direction input-small"),
    label='Direction',
    help_text='Enter cardinal direction or degrees true. ' \
        '<a href="javascript:" class="show-compass">Show compass.</a>')}

${defs.form_control(
    h.text('speed', wind.speed, class_="speed"), label='Speed')}
