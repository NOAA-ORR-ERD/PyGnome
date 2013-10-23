<%namespace name="defs" file="../defs.mak"/>

<div class="map form page hide" id='edit-map' title="Edit Map">
    <form class="form-horizontal" role="form" action="" method="POST">
        <div class="page-body">
            <div class="col-md-9">
                <%
                    name = h.text('name', data_value='map.name')
                    refloat_halflife = h.text('refloat_halflife',
                                              data_value='map.refloat_halflife',
                                              class_='input-small')
                %>

                ${defs.form_control(name, label="Name")}
                ${defs.form_control(refloat_halflife, label="Refloat Halflife",
                                    help_text="hours")}
            </div>
        </div>
    </form>
</div>
