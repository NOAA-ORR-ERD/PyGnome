<%inherit file="base.mak"/>


<%block name="extra_head">
    <link rel="stylesheet" href="http://cdn.leafletjs.com/leaflet-0.5.1/leaflet.css"/>
    <link rel='stylesheet' type='text/css' href='/static/css/skin/ui.dynatree.css'>
    <link rel='stylesheet' type='text/css' href='/static/css/model.css'>

    <script src="/static/js/lib/leaflet-src.js"></script>
    <script src="/static/js/lib/proj4js-compressed.js"></script>
    <script src="/static/js/lib/proj4leaflet.js"></script>
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
                        % for location_file in location_files:
                            <li><a tabindex="-1" class='location-file-item'
                                   data-location='${location_file['filename']}'
                                   href="javascript:">${location_file['name']}</a>
                            </li>
                        % endfor
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

    <div id="main-content" class="row expand">
        <div id="model" class='section hidden expand'>
            <div class="btn-toolbar">
                <div class="btn-group">
                    <a class="btn" id="fullscreen-button" href="javascript:"><i class="icon-fullscreen"></i></a>
                </div>
                <div class="btn-group">
                    <a class="btn" id="resize-button" href="javascript:"><i class="icon-resize-small"></i></a>
                </div>
                <div class="btn-group">
                    <a class="btn disabled" id="hand-button" href="javascript:"><i class="icon-hand-up"></i></a>
                    <a class="btn disabled" id="zoom-in-button" href="javascript:"><i class="icon-zoom-in"></i></a>
                    <a class="btn disabled" id="zoom-out-button" href="javascript:"><i class="icon-zoom-out"></i></a>
                    <a class="btn disabled" id="move-button" href="javascript:"><i class="icon-move"></i></a>
                    <a class="btn disabled" id="spill-button" href="javascript:"><i class="icon-tint"></i></a>
                </div>
                <div class="btn-group">
                    <a class="btn disabled" id="back-button" href="javascript:"><i class="icon-fast-backward"></i></a>

                    <div class="btn disabled" id="slider-container">
                        <span id="time">00:00</span>

                        <div id="slider"><div id="slider-shaded"></div></div>
                    </div>
                    <a class="btn" id="play-button" href="javascript:"><i class="icon-play"></i></a>
                    <a class="btn disabled" id="pause-button" href="javascript:"><i class="icon-pause"></i></a>
                    <a class="btn disabled" id="forward-button" href="javascript:"><i class="icon-fast-forward"></i></a>
                </div>
            </div>

            <div id="leaflet-map"> </div>
            <div id="map" class="hidden"></div>
            <div class="placeholder"></div>
        </div>

        <div id="splash-page" class='section hidden'>
             <img alt="GNOME model output depicting relative distribution of oil."
                 src="http://response.restoration.noaa.gov/sites/default/files/gnome_output_0.png"
                    style="float:right;">

            <h3>About Gnome</h3>

            <p>
                GNOME (General NOAA Operational Modeling Environment) is the
                modeling tool the Office of Response and Restoration's (OR&R)
                Emergency Response Division uses to predict the possible route,
                or trajectory, a pollutant might follow in or on a body of
                water, such as in an oil spill.
            </p>

            <h3>Get Started Now</h3>

            <div>
                <p>
                    <a class="choose-location btn btn-large btn-primary" href="javascript:">Choose a location</a>
                </p>
            </div>
            <div>
                <p>
                    <a class="build-model btn btn-large btn-primary" href="javascript:">Build your own model</a>
                </p>
            </div>
            <div>
                <p>
                    <a class="btn btn-large btn-primary">Load a save file</a>
                </p>
            </div>
        </div>

        <div id="location-file-map" class='section hidden'>
            <div id="map_canvas"></div>
        </div>
    </div>

    <div id="modal-container">
        <%include file="forms/add_mover.mak"/>
        <%include file="forms/add_environment.mak"/>
        <%include file="forms/add_spill.mak"/>
        <%include file="forms/add_map.mak"/>
        <%include file="forms/model_settings.mak"/>

        <%include file="forms/map.mak"/>
        <%include file="forms/custom_map.mak"/>
        <%include file="forms/map_from_upload.mak"/>

        ## Mover forms
        <%include file="forms/wind_mover.mak" args="form_id='add-wind-mover'"/>
        <%include file="forms/wind_mover.mak" args="form_id='edit-wind-mover'"/>
        <%include file="forms/wind.mak" args="form_id='add-wind'"/>
        <%include file="forms/wind.mak" args="form_id='edit-wind'"/>
        <%include file="forms/random_mover.mak" args="form_id='add-random-mover'"/>
        <%include file="forms/random_mover.mak" args="form_id='edit-random-mover'"/>

        ## Spill forms
        <%include file="forms/surface_release_spill.mak" args="form_id='add-surface-release-spill'"/>
        <%include file="forms/surface_release_spill.mak", args="form_id='edit-surface-release-spill'"/>

        % for location_file in location_files:
            % if 'wizard_html' in location_file:
                ${location_file['wizard_html'] | n}
            % endif
        % endfor
    </div>
</%block>

<%block name="javascript">
    <script type="text/javascript">

        // App entry-point
        require([
            'jquery',
            'lib/underscore',
            'lib/backbone',
            'router',
            'util',
            'lib/rivets',
            'lib/jquery.imagesloaded.min',
        ], function($, _, Backbone, router, util, rivets) {
            "use strict";

            // Configure a Rivets adapter to work with Backbone
            // per http://rivetsjs.com/
            rivets.configure({
                adapter: {
                    subscribe: function(obj, keypath, callback) {
                        callback.wrapped = function(m, v) {
                            callback(v)
                        };
                        obj.on('change:' + keypath, callback.wrapped);
                    },
                    unsubscribe: function(obj, keypath, callback) {
                        obj.off('change:' + keypath, callback.wrapped);
                    },
                    read: function(obj, keypath) {
                        return obj.get(keypath);
                    },
                    publish: function(obj, keypath, value) {
                        obj.set(keypath, value);
                    }
                }
            });

            // Use Django-style templates with Underscore.
            _.templateSettings = {
                interpolate: /\{\{(.+?)\}\}/g
            };

            var appOptions = {
                el: $('#app'),
                modelId: "${model_id}",
                map: ${map_data | n},
                renderer: ${renderer_data | n},
                gnomeSettings: ${model_settings | n},
                generatedTimeSteps: ${generated_time_steps_json or '[]' | n},
                expectedTimeSteps: ${expected_time_steps_json or '[]' | n},
                currentTimeStep: ${current_time_step},
                surfaceReleaseSpills: ${surface_release_spills | n},
                windMovers: ${wind_movers | n},
                winds: ${winds | n},
                randomMovers: ${random_movers | n},
                mapIsLoaded: ${"true" if map_is_loaded else "false"},
                locationFilesMeta: ${location_file_json | n},
                animationThreshold: 30, // Milliseconds
                defaultSurfaceReleaseSpill: ${default_surface_release_spill | n},
                defaultWindMover: ${default_wind_mover | n},
                defaultWind: ${default_wind | n},
                defaultRandomMover: ${default_random_mover | n},
                defaultMap: ${default_map | n},
                defaultCustomMap: ${default_custom_map | n},
                jsonSchema: ${json_schema | n}
            };

            $('#map').imagesLoaded(function() {
                new router.Router({
                    newModel: ${"true" if created else "false"},
                    appOptions: appOptions
                });

                Backbone.history.start();
            });
        });
    </script>

    <!-- A template Location File content windows. -->
     <script type="text/template" id="location-file-template">
        <h4>{{ name }}</h4>
        <p>Latitude: {{ latitude }}</p>
        <p>Longitude: {{ longitude }}</p>
        <a class="btn btn-primary load-location-file" data-location="{{ filename }}">Load Location File</a>
     </script>
</%block>