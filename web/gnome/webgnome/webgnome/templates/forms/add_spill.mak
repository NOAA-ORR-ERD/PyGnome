<%namespace name="defs" file="../defs.mak"/>

<div class="page form hide" id="add_spill" title="Add Spill">
    <div class="page-body">
        <form action="" class="form-horizontal" method="POST">
            <div class="control-group ">
                <label class="control-label">
                    Spill Type
                </label>

                <div class="controls">
                    ${h.select('spill_type', 'add_point_release_spill',
                               (('add_point_release_spill', 'Point Release Spill'),))}
                </div>
            </div>
        </form>
    </div>
</div>
