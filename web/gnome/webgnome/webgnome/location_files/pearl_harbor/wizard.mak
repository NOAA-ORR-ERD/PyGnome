<%inherit file='webgnome:location_files/templates/base.mak'/>
<%namespace name='defs' file='webgnome:templates/defs.mak'/>

<%def name='title()'>Welcome to Pearl Harbor</%def>
<%def name='form_id()'>pearl-harbor-wizard</%def>
<%def name='references()'>
    References show up in a dialog.
</%def>

<%def name='intro()'>
    <p>You will need to do the following in order to set up the General
        NOAA Operational Modeling Environment for Pearl Harbor:</p>

    <ol>
        <li>Set the model run parameters</li>
        <li>Input the wind conditions</li>
    </ol>
</%def>

<%defs:step show_form='model-settings'>
    <%defs:buttons>
        ${defs.back_btn(function_name="cancel")}
        ${defs.next_btn(function_name="submit", deferred=True)}
    </%defs:buttons>
</%defs:step>

<%defs:step show_form='add-wind-mover'>
    <%defs:buttons>
        ${defs.back_btn(function_name="cancel")}
        ${defs.next_btn(function_name="submit", deferred=True)}
    </%defs:buttons>
</%defs:step>
