<%namespace name="defs" file="../defs.mak"/>

<div class="map form page hide" id='add-map-from-upload' title="Add Map">
    <form action="" class="form-horizontal" method="POST" enctype="multipart/form-data">
        <div class="page-body">
            <div class="span4">
                <%
                    name = h.text('name', data_value='map.name')
                    refloat_halflife = h.text('refloat_halflife',
                                              data_value='map.refloat_halflife',
                                              class_='input-small')
                %>

                ${defs.form_control(name, label="Name")}
                ${defs.form_control(refloat_halflife, label="Refloat Halflife", help_text="hours")}

                ## We want the file control to receive any error messages, so its
                ## name is 'filename', which matches the model's 'filename' field.
                ${defs.form_control(h.file(name='filename', class_='fileupload btn'), label='File')}
            </div>
        </div>
    </form>
</div>
