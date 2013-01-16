<%inherit file="base.mak"/>

<%block name="extra_head">
    <link rel='stylesheet' type='text/css' href='/static/css/skin/ui.dynatree.css'>
    <link rel='stylesheet' type='text/css' href='/static/css/model.css'>

    <script src="/static/js/require-jquery.js"></script>
    <script src="/static/js/config.js"></script>
</%block>

<%block name="navbar">
    <ul class="nav" role="navigation">
        <li class="dropdown">
            <a id="file-drop" href="javascript:" role="button" class="dropdown-toggle" data-toggle="dropdown">Model<b class="caret"></b></a>
            <ul class="dropdown-menu" role="menu" aria-labelledby="file-drop">
                <li><a tabindex="-1" id="menu-new" href="javascript:">New</a></li>
                <li><a tabindex="-1" href="javascript:">Load from file</a></li>
                <li class="dropdown-submenu"><a tabindex="-1" href="javascript:">Load example...</a>
                    <ul class="dropdown-menu">
                        <li><a tabindex="-1" id="long-island" href="javascript:">Long Island Sound</a></li>
                    </ul>
                </li>
                <li><a tabindex="-1" href="javascript:">Save</a></li>
                <li class="divider"></li>
                <li><a tabindex="-1" href="javascript:">Preferences</a></li>
            </ul>
        </li>
        <li class="dropdown">
            <a id="run-drop" href="javascript:" role="button" class="dropdown-toggle" data-toggle="dropdown">Run<b class="caret"></b></a>
            <ul class="dropdown-menu" role="menu" aria-labelledby="run-drop">
                <li><a tabindex="-1" id="menu-run" href="javascript:">Run</a></li>
                <li><a tabindex="-1" id="menu-step" href="javascript:">Step</a></li>
                <li><a tabindex="-1" id="menu-run-until" href="javascript:">Run Until...</a></li>
            </ul>
        </li>
        <li class="dropdown">
            <a id="help-drop" href="javascript:" role="button" class="dropdown-toggle" data-toggle="dropdown">Help<b class="caret"></b></a>
            <ul class="dropdown-menu" role="menu" aria-labelledby="help-drop">
                <li><a tabindex="-1" target="window" href="javascript:">About</a></li>
                <li><a tabindex="-1" href="javascript:">Tutorial</a></li>
            </ul>
        </li>
    </ul>
</%block>

<%block name="sidebar">
     <div class="container sidebar-toolbar">
      <div class="btn-toolbar">
        <div class="btn-group">
            <a class="btn" id="add-button" href="javascript:"><i class="icon-plus-sign"></i></a>
            <a class="btn disabled" id="remove-button" href="javascript:"><i class="icon-minus-sign"></i></a>
            <a class="btn disabled" id="settings-button" href="javascript:"><i class="icon-wrench"></i></a>
        </div>
      </div>
    </div>
    <div id="tree"> </div>
</%block>

<%block name="content">
    <div class="container">
      <div class="messages">
          <div class="alert alert-success ${'' if success else 'hidden'}">
              <button type="button" class="close" data-dismiss="alert">× </button>
              <span class='message'>${success if success else ''}</span>
          </div>
          <div class="alert alert-warning ${'' if warning else 'hidden'}">
              <button type="button" class="close" data-dismiss="alert">× </button>
              <strong>Warning!</strong> <span class="message">${warning if warning else ''}</span>
          </div>
           <div class="alert alert-error ${'' if error else 'hidden'}">
              <button type="button" class="close" data-dismiss="alert">× </button>
              <strong>Error!</strong> <span class="message">${error if error else ''}</span>
          </div>
      </div>
    </div>

    <div id="main-content" class="row">
        <div class="btn-toolbar">
            <div class="btn-group">
                <a class="btn" id="fullscreen-button" href="javascript:"><i class="icon-fullscreen"></i></a>
            </div>
            <div class="btn-group">
                <a class="btn" id="resize-button" href="javascript:"><i class="icon-resize-small"></i></a>
            </div>
            <div class="btn-group">
                <a class="btn disabled" id="zoom-in-button" href="javascript:"><i class="icon-zoom-in"></i></a>
                <a class="btn disabled" id="zoom-out-button" href="javascript:"><i class="icon-zoom-out"></i></a>
                <a class="btn disabled" id="move-button" href="javascript:"><i class="icon-move"></i></a>
                <a class="btn disabled" id="spill-button" href="javascript:"><i class="icon-tint"></i></a>
            </div>
            <div class="btn-group">
                <a class="btn disabled" id="back-button" href="javascript:"><i class="icon-fast-backward"></i></a>

                <div class="btn disabled" id="slider-container">
                    <span id="time">00:00</span>

                    <div id="slider"></div>
                </div>
                <a class="btn" id="play-button" href="javascript:"><i class="icon-play"></i></a>
                <a class="btn disabled" id="pause-button" href="javascript:"><i class="icon-pause"></i></a>
                <a class="btn disabled" id="forward-button" href="javascript:"><i class="icon-fast-forward"></i></a>
            </div>
        </div>


        <div id="map">
        </div>

        <div class="placeholder" class="hidden">
            <div id="map_canvas"></div>
        </div>
    </div>

    <div id="modal-container">
        <%include file="forms/add_mover.mak"/>
        <%include file="forms/add_spill.mak"/>
        <%include file="forms/add_map.mak"/>
        <%include file="forms/map.mak" args="map=_map"/>
        <%include file="forms/model_settings.mak" args="model=model"/>
        <%include file="forms/wind_mover.mak"
            args="mover=default_wind_mover, default_wind=default_wind,
                  default_wind_value=default_wind_value, form_id='add_wind_mover'"/>
        <%include file="forms/wind_mover.mak"
            args="mover=default_wind_mover, default_wind=default_wind,
                  default_wind_value=default_wind_value, form_id='edit_wind_mover'"/>
        <%include file="forms/point_release_spill.mak"
            args="spill=default_point_release_spill, form_id='add_point_release_spill'"/>
         <%include file="forms/point_release_spill.mak"
            args="spill=default_point_release_spill, form_id='edit_point_release_spill'"/>
    </div>
</%block>

<%block name="javascript">
    <script type="text/javascript">

        // App entry-point
        require([
            'jquery',
            'lib/underscore',
            'app_view',
            'util',
            'lib/jquery.imagesloaded.min',
        ], function($, _, app_view, util) {
            "use strict";

            // Use Django-style templates.
            _.templateSettings = {
                interpolate: /\{\{(.+?)\}\}/g
            };

            $('#map').imagesLoaded(function() {
                new app_view.AppView({
                    generatedTimeSteps: ${generated_time_steps_json or '[]' | n},
                    expectedTimeSteps: ${expected_time_steps_json or '[]' | n},
                    backgroundImageUrl: "${background_image_url | n}",
                    currentTimeStep: ${current_time_step},
                    pointReleaseSpills: ${point_release_spills | n},
                    windMovers: ${wind_movers | n},
                    modelId: "${model_id}",
                    modelSettings: ${model_settings | n},
                    map: ${map_data | n},
                    mapIsLoaded: ${"true" if map_is_loaded else "false"}
                });
            });
        });
    </script>
</%block>