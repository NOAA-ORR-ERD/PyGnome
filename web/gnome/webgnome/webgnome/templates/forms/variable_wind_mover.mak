<div class="modal hide fade" id="variable-wind-mover-modal" tabindex="-1"
     role="dialog" aria-labelledby="modal-label" aria-hidden="true">
    <div class="modal-header">
        <button type="button" class="close" data-dismiss="variable-wind-mover-modal"
                aria-hidden="true">×
        </button>
        <h3 id="modal-label">Variable Wind Mover</h3>
    </div>
    <div class="modal-body">
        <form action="" class="form-horizontal" method="POST">
            <div class="control-group">
                <label class="control-label">${ form.speed.label.text }</label>
                <div class="controls">
                    ${ form.speed } ${ form.speed_type }
                </div>
                <label class="control-label">${ form.direction.label.text }</label>
                <div class="controls">
                    ${ form.direction }
                    <p>Enter degrees true or text (e.g., "NNW").</p>
                </div>
            </div>
        </form>
    </div>
    <div class="modal-footer">
        <button class="btn" data-dismiss="modal" aria-hidden="true"> Cancel </button>
        <button class="btn btn-primary">Save</button>
    </div>
</div>
