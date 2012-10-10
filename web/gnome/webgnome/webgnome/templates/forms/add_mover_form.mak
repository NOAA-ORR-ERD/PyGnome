<div class="modal hide fade" id="constant-wind-mover-modal" tabindex="-1"
     role="dialog" aria-labelledby="modal-label" aria-hidden="true">
    <div class="modal-header">
        <button type="button" class="close" data-dismiss="modal"
                aria-hidden="true">×
        </button>
        <h3 id="modal-label">Add Mover</h3>
    </div>
    <div class="modal-body">
        <form action="${ action_url }" class="form-horizontal" method="POST">
            <div class="control-group ${ 'error' if form.mover_type.errors else ''}">
                <label class="control-label">${ form.mover_type.label.text }</label>
                <div class="controls">
                    ${ form.mover_type }

                     % if form.mover_type.errors:
                         <span class="help-inline">
                         %for error in form.mover_type.errors:
                             ${ error }
                         %endfor
                         </span>
                    % endif
                </div>
            </div>
        </form>
    </div>
    <div class="modal-footer">
        <button class="btn btn-primary">Create</button>
        <button class="btn" data-dismiss="modal" aria-hidden="true">Cancel</button>
    </div>
</div>
