<%namespace name="defs" file="../defs.mak"/>
<%page args="form, action_url"/>

<div class="spill form hidden" id="${form.id}">

    <form action="${action_url}" id="point_release_spill"
          class="form-horizontal"
          method="POST">

        <div class="page-header">
            <h3><a href="javascript:"
                   class="edit-spill-name">${form.name.data}</a></h3>

            <div class="top-form form-inline hidden">
                ${form.name}
                <label class="checkbox">${form.is_active} Active </label>
                <button class="save-spill-name btn btn-success">Save</button>
            </div>
        </div>

        <div class="page-body">
            Test!
        </div>

        <div class="control-group form-buttons">
            <div class="form-actions">
                <button class="btn cancel"> Cancel</button>
                <button class="btn btn-primary">Save</button>
            </div>
        </div>
    </form>
</div>
