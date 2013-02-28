<%inherit file='webgnome:location_files/templates/base.mak'/>
<%namespace name='defs' file='webgnome:templates/defs.mak'/>

<%block name='title'>Welcome to Boston and vicinity</%block>

<%block name='intro'>
    <p>You will need to do the following in order to set up the General
        NOAA Operational Modeling Environment for Boston and vicinity:</p>

    <ol>
        <li>Set the model run parameters</li>
        <li>Input the wind conditions</li>
        <li>Choose whether or not to add effects from sewage outfall</li>
    </ol>
</%block>

<%defs:step num='2' form_reference='model-settings'></%defs:step>
<%defs:step num='3' form_reference='add-wind-mover'></%defs:step>

<%defs:step num='4'>
    <h2>Sewage Outflow</h2>

    <p>Wastewater effluent will begin to flow into Massachusetts Bay through
        the Massachusetts Water Resource Authority's Effluent Outfall Tunnel
        in September, 2000.</p>

    <img class='align-left' src='/static/location/boston/outfall.jpg'/>

    <p>The sewage outfall is predicted to have very minimal effects limited
        to the highlighted area in the picture. Any outfall driven currents
        would most likely be weak and noticed only in winter during low wind
        conditions.</p>

    <p>You can choose whether or not to simulate these effects:</p>

    ${defs.form_control(h.select('sewage_outfall', 'no', (
        ('no', 'No surface outfall effects'),
        ('yes', 'Add surface outfall effects')),
        class_='type input-small', data_value='wizard.add_sewage_outfall'))}
</%defs:step>
