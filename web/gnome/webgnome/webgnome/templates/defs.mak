### Mako defs.

<%def name="is_active(url)">
    % if request.path == url:
        active
    % endif
</%def>