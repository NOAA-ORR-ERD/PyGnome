<%include file="forms/add_mover.mak" args="form=add_mover_form"/>
<%include file="forms/run_model_until.mak" args="form=run_model_until_form, action_url=run_model_until_form_url"/>
<%include file="forms/model_settings.mak" args="form=settings_form, action_url=settings_form_url"/>
<%include file="forms/wind_mover.mak" args="form=wind_mover_form, action_url=wind_mover_form_url"/>

% for url, form, form_id in mover_update_forms:
    <% import webgnome.forms %>
    % if form.__class__ == webgnome.forms.WindMoverForm:
        <%include file="forms/wind_mover.mak" args="form=form, action_url=url, form_id=form_id"/>
    % endif
% endfor

% for url, form, form_id in mover_delete_forms:
    <%include file="forms/delete_mover.mak" args="form=form, action_url=url, form_id=form_id"/>
% endfor