<%namespace name="defs" file="../defs.mak"/>

<div class="form page hide" id="add-mover" title="Add Mover">
    <div class="page-body">
        <form action="" class="form-horizontal" method="POST">
            <div class="control-group ">
                <label class="control-label">
                    Mover Type
                </label>

                <div class="controls">
                    <%
                        mover_types = (('add-wind-mover', 'Wind Mover'),
                                       ('add-random-mover', 'Random Mover'))
                    %>
                    ${h.select('mover-type', 'add-wind-mover', mover_types)}
                </div>
            </div>
        </form>
    </div>
</div>
