<%namespace name="defs" file="../defs.mak"/>

<div class="modal hide fade" id="run-model-until-modal" tabindex="-1"
     data-backdrop="static" role="dialog" aria-labelledby="modal-label"
     aria-hidden="true">
    <div class="modal-header">
        <button type="button" class="close" data-dismiss="modal"
                aria-hidden="true">×
        </button>
        <h3 id="modal-label">Run Until</h3>
    </div>
    <div class="modal-body">
        <form action="${ action_url }" class="form-horizontal" method="POST">
            ${ defs.form_control(form.run_until) }
        </form>
    </div>
    <div class="modal-footer">
        <button class="btn" data-dismiss="modal" aria-hidden="true">Cancel</button>
        <button class="btn btn-primary">Run</button>
    </div>
</div>
