<%namespace name="defs" file="../defs.mak"/>

<div class="form page hide" id="add-environment" title="Add Environment">
    <div class="page-body">
        <form action="" class="form-horizontal" method="POST">
            <div class="control-group ">
                <label class="control-label">
                    Environment Type
                </label>

                <div class="controls">
                    <%
                        environment_types = (('add-wind', 'Wind'),)
                    %>
                    ${h.select('environment-type', 'add-wind', environment_types)}
                </div>
            </div>
        </form>
    </div>
</div>
