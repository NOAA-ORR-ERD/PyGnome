<%namespace name="defs" file="../defs.mak"/>

<div class="form page hide" id='add_map' title="Add Map">
    <form action="" class="form-horizontal" method="POST">
        <div class="page-body">
            <div class="control-group ">
                <label class="control-label">
                    Map
                </label>

                <div class="controls">
                    ${h.select('map_file', 'eerie', (('/data/lakeerie.bna', 'Lake Eerie'),
                                                     ('/data/lakehuron.bna', 'Lake Huron'),
                                                     ('/data/lakemichigan.bna', 'Lake Michigan'),
                                                     ('/data/lakeontario.bna', 'Lake Ontario'),
                                                     ('/data/newyork.bna', 'New York')))}
                </div>
            </div>
        </div>
    </form>
</div>
