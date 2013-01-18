<%namespace name="defs" file="../defs.mak"/>

<div class="form page hide" id="add_mover" title="Add Mover">
    <div class="page-body">
        <form action="" class="form-horizontal" method="POST">
            <div class="control-group ">
                <label class="control-label">
                    Mover Type
                </label>

                <div class="controls">
                    <%
                        mover_types = (('add_wind_mover', 'Wind Mover'),
                                       ('add_random_mover', 'Random Mover'))
                    %>
                    ${h.select('mover_type', 'add_wind_mover', mover_types)}
                </div>
            </div>
        </form>
    </div>
</div>
