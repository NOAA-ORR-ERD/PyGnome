<%namespace name="defs" file="../defs.mak"/>
<%page args="is_variable=False, compass_link=False"/>

% if is_variable:
    ${defs.datetime_control('datetime', date_label="Date")}
    <%
        direction = h.text('direction', class_="form-control direction",
                           data_value='mover:constantDirection < mover.wind')
        speed = h.text('speed', class_="form-control speed",
                       data_value='mover:constantSpeed < mover.wind')
    %>
% else:
    <%
        direction = h.text('direction', class_="form-control direction")
        speed = h.text('speed', class_="form-control speed")
    %>
% endif

<%
    if compass_link:
        help_text='<button type="button" class="btn btn-default btn-xs show-compass">' \
        	'<i class="glyphicon glyphicon-screenshot"></i>' \
	        '</button>' \
	        '<br>Enter cardinal direction or degrees true.'
    else:
        help_text =''
%>

${defs.form_control(direction, field_id='direction', label='Direction', help_text=help_text)}
${defs.form_control(speed, field_id='speed', label='Speed')}
