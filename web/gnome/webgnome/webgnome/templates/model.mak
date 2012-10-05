<%inherit file="base.mak"/>

<%block name="css">
    <link rel='stylesheet' type='text/css' href='/static/css/skin/ui.dynatree.css'>
    <link rel='stylesheet' type='text/css' href='/static/css/model.css'>
</%block>

<%block name="navbar">
    <ul class="nav" role="navigation">
        <li class="dropdown">
            <a id="file-drop" href="#" role="button" class="dropdown-toggle" data-toggle="dropdown">Model<b class="caret"></b></a>
            <ul class="dropdown-menu" role="menu" aria-labelledby="file-drop">
                <li><a tabindex="-1" href="#">Start New</a></li>
                <li><a tabindex="-1" href="#">Load from file</a></li>
                <li><a tabindex="-1" href="#">Save</a></li>
                <li class="divider"></li>
                <li><a tabindex="-1" href="#">Preferences</a></li>
            </ul>
        </li>
        <li class="dropdown">
            <a id="run-drop" href="#" role="button" class="dropdown-toggle" data-toggle="dropdown">Run<b class="caret"></b></a>
            <ul class="dropdown-menu" role="menu" aria-labelledby="run-drop">
                <li><a tabindex="-1" href="#">Run</a></li>
                <li><a tabindex="-1" href="#">Step</a></li>
                <li><a tabindex="-1" href="#">Run Until...</a></li>
            </ul>
        </li>
        <li class="dropdown">
            <a id="help-drop" href="#" role="button" class="dropdown-toggle" data-toggle="dropdown">Help<b class="caret"></b></a>
            <ul class="dropdown-menu" role="menu" aria-labelledby="help-drop">
                <li><a tabindex="-1" target="window" href="#">About</a></li>
                <li><a tabindex="-1" href="#">Tutorial</a></li>
            </ul>
        </li>
    </ul>
</%block>

<%block name="sidebar">
     <div class="container" id="sidebar-toolbar">
      <div class="btn-toolbar">
        <div class="btn-group">
            <a class="btn" id="open-button" href="#"><i class="icon-folder-open"></i></a>
            <a class="btn" id="open-save" href="#"><i class="icon-download-alt"></i></a>
        </div>
        <div class="btn-group">
            <a class="btn" id="up-button" href="#"><i class="icon-arrow-up"></i></a>
            <a class="btn" id="down-button" href="#"><i class="icon-arrow-down"></i></a>
            <a class="btn" id="settings-button" href="#"><i class="icon-wrench"></i></a>
            <a class="btn" id="delete-button" href="#"><i class="icon-trash"></i></a>
        </div>
      </div>
    </div>
    <div id="tree">
        <ul id="tree-list">
            <li id="settings" title="Model Settings">
                Model Settings
                <ul>
                    <li>Configure</li>
                    % for setting in model.get_settings():
                            <li id="${setting['name']}">${setting['name']}
                                : ${setting['value']}</li>
                    % endfor
                </ul>
            </li>
            <li id="movers" title="Universal Movers">
                Universal Movers
                <ul>
                    <li>Add mover</li>
                    % for mover in model.get_universal_movers():
                            <li>${mover['name']}</li>
                    % endfor
                </ul>
            </li>
            <li id="maps" title="Maps">
                Maps
                <ul>
                    <li>Add map</li>
                    % for map in model.get_maps():
                            <li>${map['name']}</li>
                    % endfor
                </ul>
            </li>
            <li id="spills" title="Spills">
                Spills
                <ul>
                    <li>Add spill</li>
                    % for spill in model.get_spills():
                            <li>${spill['name']}</li>
                    % endfor
                </ul>
            </li>
        </ul>
    </div>
</%block>

<%block name="content">
    <div class="container">
      <div class="btn-toolbar">
        <div class="btn-group">
            <a class="btn" id="zoom-in-button" href="#"><i class="icon-zoom-in"></i></a>
            <a class="btn" id="zoom-out-button" href="#"><i class="icon-zoom-out"></i></a>
            <a class="btn" id="move-button" href="#"><i class="icon-move"></i></a>
        </div>
        <div class="btn-group">
            <a class="btn" id="back-button" href="#"><i class="icon-fast-backward"></i></a>
            <div class="btn disabled" id="slider-container"><span id="time">00:00</span> <div id="slider"></div></div>
            <a class="btn" id="play-button" href="#"><i class="icon-play"></i></a>
            <a class="btn" id="pause-button" href="#"><i class="icon-pause"></i></a>
            <a class="btn" id="forward-button" href="#"><i class="icon-fast-forward"></i></a>
        </div>
      </div>
    </div>
    <div id="map">
        <img class="frame active" data-position="0" src="/static/img/placeholder.gif">
    </div>
</%block>

<%block name="javascript">
    <script src='/static/js/jquery.imagesloaded.min.js' type="text/javascript"></script>
    <script src='/static/js/jquery.cycle.all.latest.js' type="text/javascript"></script>
    <script src='/static/js/jquery.cookie.js' type="text/javascript"></script>
    <script src="/static/js/jquery.dynatree.min.js"></script>
    <script src="/static/js/underscore-min.js"></script>
    <script src="/static/js/map.js"></script>
</%block>