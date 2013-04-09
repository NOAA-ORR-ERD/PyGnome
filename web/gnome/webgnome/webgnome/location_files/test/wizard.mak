<%inherit file='webgnome:location_files/templates/base.mak'/>
<%namespace name='defs' file='webgnome:templates/defs.mak'/>

<%def name='title()'>Test Location File</%def>
<%def name='form_id()'>test_wizard</%def>
<%def name='references()'>
    Some references.
</%def>

<%def name='intro()'>
    <p>You will need to do the following in order to set up the General
        NOAA Operational Modeling Environment for Test Location:</p>

    <ol>
        <li>Set the model run parameters</li>
        <li>Input the wind conditions</li>
        <li>Set some custom stuff</li>
    </ol>
</%def>

<%defs:step show_form='model-settings'>
    <%defs:buttons>
        ${defs.back_btn()}
        ${defs.next_btn()}
    </%defs:buttons>
</%defs:step>

<%defs:step show_form='add-wind-mover'>
    <%defs:buttons>
        ${defs.back_btn()}
        ${defs.next_btn()}
    </%defs:buttons>
</%defs:step>

<%defs:step>
    <h5>Custom Stuff</h5>
    <p>Here's some custom stuff:</p>

    ${defs.form_control(h.select('custom_stuff', 'no', (
        ('no', 'Do not use custom stuff'),
        ('yes', 'Use custom stuff')),
        class_='type input-small', data_value='wizard.custom_stuff'))}

    <%defs:buttons>
        ${defs.back_btn()}
        ${defs.next_btn()}
    </%defs:buttons>
</%defs:step>