<%namespace name="defs" file="../defs.mak"/>
<%page args="form, action_url"/>

<div class="spill form hidden"  id="${form.id}">

    <form action="${action_url}"
          id="point_release_spill"
          data-type="spill"
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
            ${defs.form_control(form.num_LEs)}

            ${defs.form_control(form.date, label='Release Start',
                                opts={'class_': 'date'}, use_id=True)}
            ${defs.time_control(form, "Time (24 hour)")}

            <div class="control-group
                ${'error' if form.start_position_x.errors \
                             or form.start_position_y.errors \
                             or form.start_position_z.errors else ''}">
                <label class="control-label">Start Position (X, Y, Z)</label>

                <div class="controls start-coordinates">
                    ${form.start_position_x(class_='coordinate')}
                     % if form.start_position_x.errors:
                            <span class="help">
                            ${form.start_position_x.errors[0]}
                            </span>
                    % endif
                    ${form.start_position_y(class_='coordinate')}
                    % if form.start_position_y.errors:
                            <span class="help">
                            ${form.start_position_y.errors[0]}
                            </span>
                    % endif
                    ${form.start_position_z(class_='coordinate')}
                    % if form.start_position_z.errors:
                            <span class="help">
                            ${form.start_position_z.errors[0]}
                            </span>
                    % endif
                </div>
            </div>

            ${defs.form_control(form.windage_min)}
            ${defs.form_control(form.windage_max)}
            ${defs.form_control(form.windage_persist)}
            ${defs.form_control(form.is_uncertain)}
        </div>

        <div class="control-group form-buttons">
            <div class="form-actions">
                <button class="btn cancel"> Cancel</button>
                <button class="btn btn-primary">Save</button>
            </div>
        </div>
    </form>
</div>
