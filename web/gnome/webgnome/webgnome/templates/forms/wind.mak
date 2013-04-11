<%namespace name="defs" file="../defs.mak"/>
<%page args="form_id"/>

<div class="wind form page hide" id="${form_id}">
    <form action="" class="form-horizontal" method="POST">
        <div class="wind-header clearfix">
            ${defs.form_control(h.text('name', data_value='wind.name'),
                label='Name', inline=True)}
            <%
               from webgnome.util import velocity_unit_values
               velocity_unit_options = [(value, value) for value in velocity_unit_values]
            %>
            ${defs.form_control(h.select('type', 'constant', (
                ('constant-wind', 'Constant'),
                ('variable-wind', 'Variable')),
                class_='type input-small', data_value='wind:type'),
                label='Type', inline=True)}
            ${defs.form_control(h.select('units', 'knots', velocity_unit_options,
                class_='units input-small', data_value='wind.units'),
                label='Units', inline=True)}
        </div>
        <div class="page-body">
            <%include file="wind_form.mak" args="form_id=form_id"/>
        </div>
    </form>
</div>
