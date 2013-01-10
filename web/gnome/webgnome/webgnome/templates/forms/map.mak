<%namespace name="defs" file="../defs.mak"/>

<div class="map form page hide" id='map'>
    <form action="" class="form-horizontal" method="POST">
        <div class="page-body">
            <%
                name = h.text('name', disabled=True)
                refloat_halflife = h.text('refloat_halflife')
            %>

            ${defs.form_control(name, label="Name")}
            ${defs.form_control(refloat_halflife, label="Refloat Halflife",
                                help_text="hours")}
        </div>
    </form>
</div>
