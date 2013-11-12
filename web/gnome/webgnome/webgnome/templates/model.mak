<%inherit file="base.mak"/>

<%block name="third_party_css">
    <link href="/static/css/custom-theme/jquery-ui-1.8.16.custom.css" rel="stylesheet"/>

	<link href="//netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.css" rel="stylesheet">
    <link href="//netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css" rel="stylesheet">
    <link href="//netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-theme.css" rel="stylesheet">

    <link href='/static/css/leaflet.css' rel="stylesheet"/>
    <link href='/static/css/leaflet.draw.css' rel="stylesheet"/>

    <link href='/static/css/skin/ui.dynatree.css' rel='stylesheet' type='text/css'/>
</%block>

<%block name="third_party_js">
    ## HTML 5 and IE-specific shims. Not sure why they're called "shivs" ...
    <!--[if lt IE 9]>
        <!--<script src="/static/js/lib/excanvas.compiled.js"></script>-->
        <script src="/static/js/lib/excanvas.js"></script>
        <script src="/static/js/lib/html5shiv.js"></script>
        <script src="/static/js/lib/indexOfShiv.js"></script>
    <![endif]-->

	<script src="//netdna.bootstrapcdn.com/bootstrap/3.0.0/js/bootstrap.min.js"></script>

    ## TODO: Add these as RequireJS dependencies.
    <script src="/static/js/lib/leaflet-src.js"></script>
    <script src="/static/js/lib/leaflet.draw.js"></script>
    <script src="/static/js/lib/leaflet.label.js"></script>
    <script src="/static/js/lib/L.Graticule.js"></script>

    <script src="/static/js/require-jquery.js"></script>
    <script src="/static/js/config.js"></script>
    
    <script src="//dygraphs.com/dygraph-dev.js"></script>
</%block>

<%block name="extra_head">
    <link href='/static/css/model.css' rel='stylesheet' type='text/css'/>
</%block>

<%block name="navbar">
	<div class="collapse navbar-collapse navbar-ex1-collapse">
	    <ul class="nav navbar-nav">
	        <li class="dropdown">
	            <a id="file-drop" class="dropdown-toggle" href="javascript:" data-toggle="dropdown">Model <b class="caret"></b></a>
	            <ul class="dropdown-menu">
	                <li><a id="menu-new" href="javascript:">New</a></li>
	                <li><a href="javascript:">Load from file</a></li>
	                <li class="dropdown-submenu">
		                	<a href="javascript:" >Load example</a>
		                    <ul class="dropdown-menu">
		                        % for location_file in location_files:
		                            <li>
		                            	<a class='location-file-item'
		                                   data-location='${location_file['filename']}'
		                                   href="javascript:">${location_file['name']}</a>
		                            </li>
		                        % endfor
		                    </ul>
	                </li>
	                <li><a href="javascript:">Save</a></li>
	                <li class="divider"></li>
	                <li><a href="javascript:">Preferences</a></li>
	            </ul>
	        </li>
	        <li class="dropdown">
	            <a id="run-drop" class="dropdown-toggle" href="javascript:" data-toggle="dropdown">Run <b class="caret"></b></a>
	            <ul class="dropdown-menu">
	                <li><a id="menu-run" href="javascript:">Run</a></li>
	                <li><a id="menu-step" href="javascript:">Step</a></li>
	                <li><a id="menu-run-until" href="javascript:">Run Until...</a></li>
	            </ul>
	        </li>
	        <li class="dropdown">
	            <a id="help-drop" class="dropdown-toggle" href="javascript:" data-toggle="dropdown">Help <b class="caret"></b></a>
	            <ul class="dropdown-menu">
	                <li><a target="window" href="javascript:">About</a></li>
	                <li><a href="javascript:">Tutorial</a></li>
	            </ul>
	        </li>
	    </ul>
	</div>
</%block>

<%block name="sidebar">
      <div class="btn-toolbar">
        <div class="btn-group">
            <button type="button" class="btn btn-default" id="collapse-sidebar-button">
            	<i class="glyphicon glyphicon-resize-small"></i>
            </button>
            <button type="button" class="btn btn-default" id="expand-sidebar-button">
            	<i class="glyphicon glyphicon-fullscreen"></i>
            </button>
        </div>
        <div class="btn-group pull-right">
            <button type="button" class="btn btn-default" id="add-button">
            	<i class="glyphicon glyphicon-plus-sign"></i>
            </button>
            <button type="button" class="btn btn-default disabled" id="remove-button">
            	<i class="glyphicon glyphicon-minus-sign"></i>
            </button>
            <button type="button" class="btn btn-default disabled" id="settings-button">
            	<i class="glyphicon glyphicon-wrench"></i>
            </button>
        </div>
      </div>
    <div id="tree" class="panel"> </div>
</%block>

<%block name="content">
    <div id="main-content" class="row">

    	<div id="carousel-app-views" class="carousel slide" data-interval="0">
    		<!-- Indicators -->
    		<ol class="carousel-indicators">
    			<li data-target="#carousel-app-views" data-slide-to="0" class="active"></li>
    			<li data-target="#carousel-app-views" data-slide-to="1"></li>
    		</ol>

    		<!-- Wrapper for slides -->
    		<div class="carousel-inner">
    			<div class="item active">

			        <div id="model" class='section hidden'>

					    <div class="btn-toolbar" >
					        <div class="btn-group">
					            <button type="button" class="btn btn-default disabled" id="back-button">
					            	<i class="glyphicon glyphicon-fast-backward"></i>
					            </button>

					            <div type="button" class="btn btn-default disabled" id="slider-container">
					                        <span id="time">00:00</span>

					                        <div id="slider"><div id="slider-shaded"></div></div>
					            </div>

					            <button type="button" class="btn btn-default" id="play-button">
					            	<i class="glyphicon glyphicon-play"></i>
					            </button>
					            <button type="button" class="btn btn-default disabled" id="pause-button">
					            	<i class="glyphicon glyphicon-pause"></i>
					            </button>
					            <button type="button" class="btn btn-default disabled" id="forward-button">
					            	<i class="glyphicon glyphicon-fast-forward"></i>
					            </button>
					        </div>
					    </div>

			            <div class="panel" id="leaflet-map"> </div>
			            <div class="current-coordinates"></div>
			            <div id="map" class="hidden"></div>
			            <div class="placeholder"></div>
			        </div>

    				<div class="carousel-caption">
    					Map View
    				</div>
    			</div> <!-- end first item -->
				<div class="item active">
					<ul class="nav nav-tabs">
					    <li class="active"><a href="#remaining-graph" data-toggle="tab">Remaining</a></li>
					    <li><a href="#dispersed-graph" data-toggle="tab">Dispersed</a></li>
					    <li><a href="#evaporated-graph" data-toggle="tab">Evaporated</a></li>
					</ul>
					<div id="myTabContent" class="tab-content">
					    <!-- It is important for all panes to be initially active -->
					    <!-- when loading dynamic content like dygraphs.          -->
					    <!-- After the page load, we can use JS to remove the     -->
					    <!-- 'in active' classes from our inactive panes.         -->
					    <div class="tab-pane fade in active" id="remaining-graph">
					    </div>
					    <div class="tab-pane fade in active" id="dispersed-graph">
					    </div>
					    <div class="tab-pane fade in active" id="evaporated-graph">
					    </div>
					</div>
					<div class="carousel-caption">
						Reports View (None Yet...)
					</div>
				</div>  <!-- end second item -->
    		</div>

    		<!-- Controls -->
##    		<a class="left carousel-control" href="#carousel-app-views" data-slide="prev">
##    			<span class="icon-prev"></span>
##    		</a>
    		<a class="right carousel-control" href="#carousel-app-views" data-slide="next">
    			<span class="icon-next"></span>
    		</a>
    	</div> <!-- end carousel -->

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

            // Ask jQuery to add a cache-buster to AJAX requests, so that
            // IE's aggressive caching doesn't break everything.
            $.ajaxSetup({
                cache: false
            });

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
                    /*
                     When setting a value, if it's parsable as a float, use a
                     float value instead. This is to support JSON Schema
                     validation of float types.
                     */
                    publish: function(obj, keypath, value) {
                        var floatVal = parseFloat(value);
                        if (!isNaN(floatVal)) {
                            value = floatVal;
                        }
                        obj.set(keypath, value);
                    }
                }
            });

            // Use Django-style templates semantics with Underscore's _.template.
            _.templateSettings = {
                // {{- variable_name }} -- Escapes unsafe output (e.g. user
                // input) for security.
                escape: /\{\{-(.+?)\}\}/g,

                // {{ variable_name }} -- Does not escape output.
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
                animationThreshold: 25, // Milliseconds
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

    <!-- A template for Location File popups. -->
    <script type="text/template" id="location-file-template">
        <h4>{{- name }}</h4>
        <p>Latitude: {{- latitude }}</p>
        <p>Longitude: {{- longitude }}</p>
        <a class="btn btn-primary load-location-file" data-location="{{- filename }}">Load Location File</a>
    </script>


    <!-- A template for Surface Release Spill popups. -->
    <script type="text/template" id="surface-release-spill-template">
        <h4>{{- name }}</h4>
        <p>Latitude: {{- lat }}</p>
        <p>Longitude: {{- lng }}</p>
    </script>
</%block>