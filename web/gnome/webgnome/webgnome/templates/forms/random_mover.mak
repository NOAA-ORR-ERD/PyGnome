<%namespace name="defs" file="../defs.mak"/>
<%page args="mover, form_id"/>

<div class="random form page hide" id="${form_id}">
    <form action="" class="form-horizontal" method="POST">
        <div class="page-body">
            ${defs.form_control(h.checkbox('on', checked=mover.on), label='On')}
            ${defs.form_control(h.text('name', mover.name), label='Name')}
            ${defs.form_control(h.text('diffusion_coef', mover.diffusion_coef),
                                label='Diffusion Coefficient')}
            ${defs.datetime_control(mover.active_start, 'active_start', date_label="Active Start")}
            ${defs.datetime_control(mover.active_stop, 'active_stop', date_label="Active Stop")}
        </div>
    </form>
</div>
