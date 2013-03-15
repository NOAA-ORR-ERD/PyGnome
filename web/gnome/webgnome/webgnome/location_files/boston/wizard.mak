<%inherit file='webgnome:location_files/templates/base.mak'/>
<%namespace name='defs' file='webgnome:templates/defs.mak'/>

<%def name='title()'>Welcome to Boston and vicinity</%def>
<%def name='form_id()'>boston-wizard</%def>
<%def name='references()'>
    References show up in a dialog.
</%def>

<%def name='intro()'>
    <p>You will need to do the following in order to set up the General
        NOAA Operational Modeling Environment for Boston and vicinity:</p>

    <ol>
        <li>Set the model run parameters</li>
        <li>Input the wind conditions</li>
        <li>Choose whether or not to add effects from sewage outfall</li>
    </ol>
</%def>

<%defs:step reference_form='model-settings'>
    <%defs:buttons>
        ${defs.back_btn()}
        ${defs.next_btn()}
    </%defs:buttons>
</%defs:step>

<%defs:step reference_form='add-wind-mover'>
    <%defs:buttons>
        ${defs.back_btn()}
        ${defs.next_btn()}
    </%defs:buttons>
</%defs:step>

<%defs:step height="432" width="420">
    <h5>Sewage Outflow</h5>

    <p>Wastewater effluent will begin to flow into Massachusetts Bay through
        the Massachusetts Water Resource Authority's Effluent Outfall Tunnel
        in September, 2000.</p>

    <img style="float:right;" src='/static/location_file/boston/outfall.jpg'/>

    <p> The sewage outfall is predicted to have very minimal effects limited
        to the highlighted area in the picture. Any outfall driven currents
        would most likely be weak and noticed only in winter during low wind
        conditions.</p>

    <p>You can choose whether or not to simulate these effects:</p>

    ${defs.form_control(h.select('sewage_outfall', 'no', (
        ('no', 'No surface outfall effects'),
        ('yes', 'Add surface outfall effects')),
        data_value='wizard.add_sewage_outfall'))}

    <%defs:buttons>
        ${defs.back_btn()}
        ${defs.next_btn()}
    </%defs:buttons>
</%defs:step>
