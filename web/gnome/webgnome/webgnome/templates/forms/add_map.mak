<%namespace name="defs" file="../defs.mak"/>

<div class="form page hide" id='add-map' title="Add Map">
    <form action="" class="form-horizontal" method="POST">
        <div class="page-body">
            <div class="control-group ">
                <label class="control-label">
                    Map
                </label>

                <div class="controls">
                    ${h.select('map-source', 'add-custom-map', (('add-custom-map', 'Custom'),
                                                                ('add-map-from-upload', 'From Upload')))}
                </div>
            </div>
        </div>
    </form>
</div>
