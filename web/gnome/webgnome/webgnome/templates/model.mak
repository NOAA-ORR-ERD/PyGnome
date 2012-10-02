<%inherit file="base.mak"/>

<%block name="css">
    <link rel='stylesheet' type='text/css' href='/static/css/skin/ui.dynatree.css'>
    <link rel='stylesheet' type='text/css' href='/static/css/model.css'>
</%block>

<%block name="content">
    <div class="container-fluid" id="content">
        <div class="row-fluid">
            <div class="span2 offset1">
                <!--Sidebar content-->
                <div id="tree">
                </div>
            </div>
            <div class="span8">
                <!--Body content-->
                <img id="map" src="/static/img/map_zoom_3.png">
            </div>
        </div>
    </div>
</%block>

<%block name="javascript">
    <script src="/static/js/map.js"></script>
    <script src="/static/js/jquery.dynatree.min.js"></script>
    <script src='/static/js/jquery.cookie.js' type="text/javascript"></script>
</%block>