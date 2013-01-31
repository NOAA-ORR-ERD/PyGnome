<%namespace name="defs" file="../defs.mak"/>
<%page args="is_variable=False, compass_link=False"/>

% if is_variable:
    ${defs.datetime_control('datetime', date_label="Date")}
% endif

<%
    if compass_link:
        help_text='Enter cardinal direction or degrees true. ' \
            '<a href="javascript:" class="show-compass">Show compass.</a>'
    else:
        help_text =''
%>

${defs.form_control(
    h.text('direction', class_="direction input-small"),
    label='Direction', help_text=help_text)}

${defs.form_control(
    h.text('speed', class_="speed"), label='Speed')}
