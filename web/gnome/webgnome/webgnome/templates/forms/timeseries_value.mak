<%namespace name="defs" file="../defs.mak"/>
<%page args="is_variable=False, compass_link=False"/>

% if is_variable:
    ${defs.datetime_control('datetime', date_label="Date")}
    <%
        direction_data = 'mover:constantDirection < mover.wind'
        speed_data = 'mover:constantSpeed < mover.wind'
    %>
% else:
     <%
        direction_data = ''
        speed_data = ''
    %>
% endif

<%
    if compass_link:
        help_text='Enter cardinal direction or degrees true. ' \
            '<a href="javascript:" class="show-compass">Show compass.</a>'
    else:
        help_text =''
%>

${defs.form_control(
    h.text('direction', class_="direction input-small", data_value=direction_data),
    label='Direction', help_text=help_text)}

${defs.form_control(
    h.text('speed', class_="speed", data_value=speed_data), label='Speed')}
