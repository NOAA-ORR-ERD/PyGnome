<%namespace name="defs" file="../defs.mak"/>

<div class="modal form hide fade" id="add_spill" tabindex="-1"
     data-backdrop="static" role="dialog" aria-labelledby="modal-label"
     aria-hidden="true">
    <div class="modal-header">
        <button type="button" class="close" data-dismiss="modal"
                aria-hidden="true">×
        </button>
        <h3 id="modal-label">Add Spill</h3>
    </div>
    <div class="modal-body">
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
    <div class="modal-footer">
        <button class="btn cancel" data-dismiss="modal" aria-hidden="true">Cancel</button>
        <button class="btn btn-primary">Create</button>
    </div>
</div>
