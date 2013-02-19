<%namespace name="defs" file="../defs.mak"/>
<%page args="is_variable=False, compass_link=False"/>

% if is_variable:
    ${defs.datetime_control('datetime', date_label="Date")}
    <%
        direction = h.text('direction', class_="direction input-small",
                           data_value='mover:constantDirection < mover.wind')
        speed = h.text('speed', class_="speed",
                       data_value='mover:constantSpeed < mover.wind')
    %>
% else:
    <%
        direction = h.text('direction', class_="direction input-small")
        speed = h.text('speed', class_="speed")
    %>
% endif

<%
    if compass_link:
        help_text='Enter cardinal direction or degrees true. ' \
            '<a href="javascript:" class="show-compass">Show compass.</a>'
    else:
        help_text =''
%>

${defs.form_control(direction,  label='Direction', help_text=help_text)}
${defs.form_control(speed, label='Speed')}
