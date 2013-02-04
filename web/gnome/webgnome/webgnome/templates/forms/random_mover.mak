<%namespace name="defs" file="../defs.mak"/>
<%page args="form_id"/>

<div class="random form page hide" id="${form_id}">
    <form action="" class="form-horizontal" method="POST">
        <div class="page-body">
            ${defs.form_control(h.checkbox('on', data_checked='mover.on'), label='On')}
            ${defs.form_control(h.text('name', data_value='mover.name'), label='Name')}
            ${defs.form_control(h.text('diffusion_coef', data_value='mover.diffusion_coef'),
                                label='Diffusion Coefficient')}
            ${defs.datetime_control('active_start', date_label="Active Start")}
            ${defs.datetime_control('active_stop', date_label="Active Stop")}
        </div>
    </form>
</div>
