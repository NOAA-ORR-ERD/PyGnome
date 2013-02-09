<%namespace name="defs" file="../defs.mak"/>

<div class="page form hide" id="add-spill" title="Add Spill">
    <div class="page-body">
        <form action="" class="form-horizontal" method="POST">
            <div class="control-group ">
                <label class="control-label">
                    Spill Type
                </label>

                <div class="controls">
                    ${h.select('spill-type', 'add-surface-release-spill',
                               (('add-surface-release-spill', 'Surface Release Spill'),))}
                </div>
            </div>
        </form>
    </div>
</div>
