define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'lib/geo',
    'lib/jquery.imagesloaded.min',
], function($, _, Backbone, models, Geo) {

    var GnomeImageOverlay = L.ImageOverlay.extend({
        _initImage: function() {
            var _this = this;
            this._image = L.DomUtil.create('img', 'leaflet-image-layer');

            if (this._map.options.zoomAnimation && L.Browser.any3d) {
                L.DomUtil.addClass(this._image, 'leaflet-zoom-animated');
            } else {
                L.DomUtil.addClass(this._image, 'leaflet-zoom-hide');
            }

            this._updateOpacity();

            $(this._image).imagesLoaded(function() {
                _this._onImageLoad();
            });

            L.extend(this._image, {
                galleryimg: 'no',
                onselectstart: L.Util.falseFn,
                onmousemove: L.Util.falseFn,
                src: this._url
            });

            this._updateZIndex();
        },

        update: function(url) {
            $(this._image).attr('src', url);
            this._reset();
        },

        _updateZIndex: function() {
            if (this._image && this.options.zIndex !== undefined) {
                this._image.style.zIndex = this.options.zIndex;
            }
        },

        setZIndex: function(zIndex) {
            this.options.zIndex = zIndex;
            this._updateZIndex();

            return this;
        }
    });

    var imageOverlay = function(url, bounds, options) {
        return new GnomeImageOverlay(url, bounds, options);
    };


    function MapViewException(message) {
        this.message = message;
        this.name = "MapViewException";
    }


    /*
     `MapView` represents the visual map and is responsible for animating frames
     for each time step rendered by the server
     */
    var MapView = Backbone.View.extend({
        initialize: function() {
            var _this = this;
            _.bindAll(this);

            this.mapEl = this.options.mapEl;
            this.placeholderClass = this.options.placeholderClass;
            this.animationThreshold = this.options.animationThreshold;
            this.renderer = this.options.renderer;
            this.drawToolsInitialized = false;
            this.spillMarkers = [];
            this.surfaceReleaseSpills = this.options.surfaceReleaseSpills;
            this.router = this.options.router;

            this.surfaceReleaseSpills.on('sync', this.drawSpills);
            this.surfaceReleaseSpills.on('add', this.drawSpills);
            this.surfaceReleaseSpills.on('remove', this.drawSpills);

            this.makeImagesClickable();

            this.state = this.options.state;
            this.listenTo(this.state, 'change:animation', this.animationStateChanged);

            this.map = $(this.mapEl);

            this.stepGenerator = this.options.stepGenerator;
            this.stepGenerator.on(models.StepGenerator.NEXT_TIME_STEP_READY, this.nextTimeStepReady);
            this.stepGenerator.on(models.StepGenerator.RUN_ERROR, this.stepGeneratorError);
            this.stepGenerator.on(models.StepGenerator.RUN_FINISHED, this.stepGeneratorFinished);
            this.stepGenerator.on(models.StepGenerator.CREATED, this.reset);

            this.model = this.options.model;
            this.model.on('destroy', function() {
                _this.reset().then(function() {
                    _this.leafletMap.removeLayer(_this.backgroundOverlay);
                });
            });
            this.model.on('sync', this.mapSynced);

            this.customMap = this.options.customMap;
            this.customMap.on('sync', this.mapSynced);

            if (this.stepGenerator.hasCachedTimeStep(this.stepGenerator.getCurrentTimeStep())) {
                this.nextTimeStepReady();
            }

            this.cursorClasses = ['zooming-in', 'zooming-out', 'moving', 'spill'];
            this.currentCoordinates = $('.current-coordinates');

            this.setupMap();
        },

        /*
         If a new map was added, fetch the renderer and model data and then reset
         the Leaflet map.

         Because the view looks at `this.model` instead of a collection -- since
         there is always only one map -- we don't have an 'add' event we could
         watch, so this method exists as a 'sync' callback. To differentiate a
         sync caused by saving a new model versus one caused by fetching or
         saving an existing one, we check for the 'added' flag and bail if it
         wasn't set. We pass a manual flag to `this.model.save()` when saving a
         new map, and watch for that flag in this method.
         */
        mapSynced: function(model, attrs, opts) {
            var _this = this;
            if (!opts.added) {
                return;
            }

            this.renderer.fetch().done(function() {
                _this.model.fetch().done(function() {
                    _this.reset();
                });
            });
        },

        setupDrawingTools: function() {
            var _this = this;
            if (this.drawToolsInitialized) {
                return;
            }

            var OilMarker = L.Icon.extend({
                options: {
                    iconUrl: 'http://upload.wikimedia.org/wikipedia/commons/7/75/Oil_drop.png',
                    iconSize: new L.Point(18.12, 28)
                }
            });

            var drawnItems = new L.FeatureGroup();
            this.leafletMap.addLayer(drawnItems);

            var drawControl = new L.Control.Draw({
                draw: {
                    polygon: null,
                    circle: null,
                    rectangle: null,
                    polyline: {
                        title: "Add a point release spill with a start and end point."
                    },
                    marker: {
                        title: "Add a point release spill with a single origin."
                    }
                },
                edit: false
            });
            this.leafletMap.addControl(drawControl);

            this.leafletMap.on('draw:created', function(e) {
                var type = e.layerType,
                    layer = e.layer;

                _this.spillMarkers.push(e.layer);
                drawnItems.addLayer(layer);

                var latLng, coords;

                if (type == 'polyline') {
                    latLng = e.layer.getLatLngs();
                    coords = [
                        [latLng[0].lng, latLng[0].lat],
                        [latLng[1].lng, latLng[1].lat]
                    ];
                } else {
                    latLng = e.layer.getLatLng();
                    coords = [
                        [latLng.lng, latLng.lat],
                        [latLng.lng, latLng.lat]
                    ];
                }

                _this.trigger(MapView.SPILL_DRAWN, coords[0], coords[1]);
            });

            this.drawToolsInitialized = true;
        },

        setupMap: function() {
            var _this = this;
            this.leafletMap = L.map('leaflet-map', {
                crs: L.CRS.EPSG4326,
                minZoom: 9,
                worldCopyJump: false,
                attribution: false,
                inertia: false
            });

            this.leafletMap.on('zoomend', _.debounce(function() {
                _this.setNewViewport();
            }, 400));

            this.leafletMap.on('dragend', _.debounce(function() {
                _this.setNewViewport();
            }, 300));

            this.leafletMap.on('mousemove', function(event) {
                _this.updateCoordinates(event.latlng);
            });

            this.leafletMap.on('mouseout', function(event) {
                _this.updateCoordinates();
            });

            this.setLeafletMapSize();

            this.graticule = L.graticule({
                interval: 1,
                style: {
                    weight: 0.5,
                    color: '#333'
                },
                onEachFeature: function(feature, layer) {
                    layer.bindLabel(feature.properties.name);
                }
            }).addTo(this.leafletMap);

            $(window).resize(this.setLeafletMapSize);
            
            this.drawSpills();
        },

        updateCoordinates: function(latlng) {
            var text = '';
            if (latlng) {
                text = Geo.toDMS(latlng.lat) + ' ' + Geo.toDMS(latlng.lng);
            }
            this.currentCoordinates.html(text);
        },

        setLeafletMapSize: function() {
            var mapHeight = $(window).height() - $('.navbar').height() - $('.model .btn-toolbar').height() - 90;
            $('#leaflet-map').height(mapHeight);
            this.updateSize();
        },

        setNewViewport: function() {
            var _this = this;
            if (this.isSettingViewport) {
                return;
            }

            this.isSettingViewport = true;
            var newBounds = this.leafletMap.getBounds();
            var sw = newBounds.getSouthWest();
            var ne = newBounds.getNorthEast();
            var size = this.leafletMap.getSize();

            this.renderer.set({
                viewport: [
                    [sw.lng, sw.lat],
                    [ne.lng, ne.lat]
                ],
                image_size: [size.x, size.y]
            });

            this.saveRenderer().then(function() {
                _this.trigger(MapView.VIEWPORT_CHANGED);
                _this.isSettingViewport = false;
            });
        },

        /**
         * Save the renderer settings and get the latest background image from
         * the map model.
         */
        saveRenderer: function() {
            var _this = this;
            return this.renderer.save()
                .then(function() {
                    _this.model.fetch({reloadTree: false});
                })
        },

        animationStateChanged: function(animationState) {
            switch (animationState) {
                case this.state.animation.STOPPED:
//                    this.setStopped();
                    break;
                case this.state.animation.PLAYING:
//                    this.setPlaying();
                    break;
                case this.state.animation.PAUSED:
//                    this.setPaused();
                    break;
            }
        },

        show: function() {
            this.setBackground();
        },

        addBackgroundLayer: function() {
            var _this = this;
            var savingRenderer;

            // If we're in the process of setting a new viewport, part of
            // which involves saving the renderer, then we don't need to save
            // the renderer -- otherwise we do.
            if (this.isSettingViewport) {
                savingRenderer = this.model.fetch({reloadTree: false});
            } else {
                var size = this.leafletMap.getSize();
                this.renderer.set('image_size', [size.x, size.y]);
                savingRenderer = this.saveRenderer();
            }

            return savingRenderer.then(function() {
                var viewport = _this.renderer.getLatLongViewport();
                var url = _this.model.get('background_image_url');
                _this.viewport = new L.LatLngBounds([viewport.sw, viewport.ne]);

                if (!_this.backgroundOverlay) {
                    // Fit the map to the viewport bounds if we're adding the back-
                    // ground for the first time.
                    _this.leafletMap.fitBounds(_this.viewport);
                }

                _this.backgroundOverlay = imageOverlay(url, _this.viewport, {
                    zIndex: -100
                });
                _this.leafletMap.addLayer(_this.backgroundOverlay);
                _this.setupDrawingTools();
            });
        },

        /*
         Set the background image on the map.

         First add the background image as a map layer. When that operation
         completes, remove the old background by fading it out and then fading
         the new image on top of it.

         Returns the promise returned by `this.addBackgroundLayer`.
         */
        setBackground: function() {
            var _this = this;
            var url = this.model.get('background_image_url');

            if (url) {
                this.hidePlaceholder();
            } else {
                this.showPlaceholder();
                return $.Deferred().resolve();
            }

            var oldOverlay = _this.backgroundOverlay;

            function removeBackgroundLayer() {
                if (oldOverlay) {
                    // Close over the old background layer, fade it out and
                    // remove it.
                    $(oldOverlay._image).fadeOut(400, function() {
                        _this.leafletMap.removeLayer(oldOverlay);
                    });
                    $(_this.backgroundOverlay._image).fadeIn(400);
                }
            }

            return _this.addBackgroundLayer().then(removeBackgroundLayer);
        },

        showPlaceholder: function() {
            $('.' + this.placeholderClass).removeClass('hidden');
        },

        hidePlaceholder: function() {
            $('.' + this.placeholderClass).addClass('hidden');
        },

        makeImagesClickable: function() {
            var _this = this;
            $(this.mapEl).on('click', 'img', function(event) {
                if ($(this).data('clickEnabled')) {
                    _this.trigger(MapView.MAP_WAS_CLICKED, {
                        x: event.pageX,
                        y: event.pageY
                    });
                }
            });
        },

        makeActiveImageClickable: function() {
            var image = this.getActiveImage();
            image.data('clickEnabled', true);
        },

        makeActiveImageSelectable: function() {
            var _this = this;
            var image = this.getActiveImage();
            image.selectable({
                start: function(event) {
                    _this.startPosition = {x: event.pageX, y: event.pageY};
                },
                stop: function(event) {
                    if (!$(this).selectable('option', 'disabled')) {
                        _this.trigger(MapView.DRAGGING_FINISHED, [
                            _this.startPosition,
                            {x: event.pageX, y: event.pageY}
                        ]);
                    }
                }
            });
        },

        getActiveImage: function() {
            return $(this.mapEl + " > img.active");
        },

        getImageForTimeStep: function(stepNum) {
            return $('img[data-id="' + (stepNum) + '"]');
        },

        /*
         Given the length of time the last timestep request took `requestTime`,
         calculate the timeout value for displaying that step.

         If `requestTime` was less than the threshold, then use the difference
         between the threshold value and `requestTime` as the timeout.
         Otherwise, use a timeout of 0, since the request has taken long enough.
          */
        getAnimationTimeout: function(requestTime) {
            // Get the number of MS the last request took.
            requestTime = requestTime || 0;
            var threshold = this.animationThreshold;
            return requestTime < threshold ? threshold - requestTime : 0;
        },

        addTimeStep: function(timeStep) {
            var _this = this;
            if (!this.leafletMap) {
                return;
            }

            function addLayer(timeStepLayer) {
                if (_this.timeStepLayer) {
                    _this.leafletMap.removeLayer(_this.timeStepLayer);
                }
                _this.timeStepLayer = timeStepLayer;
                timeStepLayer.on('load', function() {
                    _this.trigger(MapView.FRAME_CHANGED);
                });
                _this.leafletMap.addLayer(timeStepLayer);
            }

            function addImageOverlay(url) {
                var timeStepLayer = imageOverlay(url, _this.viewport, {
                    zIndex: -50
                });
                setTimeout(addLayer, _this.getAnimationTimeout(requestTime),
                           timeStepLayer);
            }

            var url = timeStep.get('url');
            var imageExists = this.getImageForTimeStep(timeStep.id).length;

            if (imageExists) {
                addImageOverlay(url);
                return;
            }

            var requestTime = timeStep.get('requestTime');
            var imageCache = $('<img>').attr({
                'class': 'hidden frame',
                'data-id': timeStep.id,
                'src': timeStep.get('url')
            }).addClass('hidden');

            imageCache.appendTo(map);
            imageCache.imagesLoaded(function() {
                addImageOverlay(url);
            });
        },

        // Clear out <img> DOM elements used to cache frame images.
        clearImageCache: function() {
            var map = $(this.mapEl);
            map.find('img').remove();
        },

        getBackground: function() {
            return $(this.mapEl).find('img.background')[0];
        },

        getSize: function() {
            var image = $('.background');
            return {height: image.height(), width: image.width()};
        },

        getPosition: function() {
            return this.getActiveImage().position();
        },

        getBoundingBox: function() {
            var pos = this.getPosition();
            var size = this.getSize();

            return [
                {x: pos.left, y: pos.top},
                {x: pos.left + size.width, y: pos.top + size.height}
            ];
        },

        removeCursorClasses: function() {
            for (var i = 0; i < this.cursorClasses.length; i++) {
                var cls = this.cursorClasses[i];
                $(this.mapEl).removeClass(cls);
            }
        },

        setZoomingInCursor: function() {
            $(this.mapEl).addClass('zooming-in-cursor');
        },

        setZoomingOutCursor: function() {
            $(this.mapEl).addClass('zooming-out-cursor');
        },

        setRegularCursor: function() {
            $(this.mapEl).addClass('regular-cursor');
        },

        setMovingCursor: function() {
            $(this.mapEl).addClass('moving-cursor');
        },

        updateSize: function() {
            if (!this.leafletMap) {
                return;
            }

            this.leafletMap.invalidateSize(true);

            if (this.viewport) {
                this.setNewViewport();
            }
        },

        nextTimeStepReady: function() {
            this.addTimeStep(this.stepGenerator.getCurrentTimeStep());
        },

        drawSpill: function(spill) {
            var _this = this;
            var startX, startY, startZ, endX, endY, endZ;
            var start = spill.get('start_position');
            var end = spill.get('end_position');

            if (!this.leafletMap) {
                return;
            }

            startX = endX = start[0];
            startY = endY = start[1];
            startZ = endZ = start[2];

            if (!startX) {
                return;
            }

            if (end) {
                endX = end[0];
                endY = end[1];
                endZ = end[2];
            }

            var marker;

            if (startX == endX || !endX) {
                // Create a marker
                marker = L.marker([startY, startX]).addTo(this.leafletMap);
            } else {
                // Create a polyline
                var coords = [
                    [startY, startX],
                    [endY, endX]
                ];
                marker = L.polyline(coords).addTo(this.leafletMap);
            }

            // XXX: Leaflet doesn't propagate click events within popups, so we
            // can't use a single template with an .on() delegated listener here.
            var templ = _.template($('#surface-release-spill-template').html());
            var html = templ({
                name: spill.get('name'),
                lng: startY,
                lat: startX
            });

            templ = _.template('<a href="javascript:" class="btn btn-primary open-spill" ' +
                'data-id="{{- id }}">Edit</a>');

            var link = $(templ({
                id: spill.id
            }));

            link.click(function(event) {
                var spillId = $(this).data('id');
                _this.router.navigate('#/spill/' + spillId, true);
                marker.closePopup();
            });

            var div = $('<div>').html(html).append(link);
            marker.bindPopup(div[0]);

            this.spillMarkers.push(marker);
        },

        // Draw a mark on the map for each existing spill.
        drawSpills: function() {
            var _this = this;

            if (this.surfaceReleaseSpills === undefined || !this.surfaceReleaseSpills.length) {
                return;
            }

            for (var i = 0; i < this.spillMarkers.length; i++) {
                this.leafletMap.removeLayer(this.spillMarkers[i]);
            }

            this.surfaceReleaseSpills.forEach(function(spill) {
                _this.drawSpill(spill);
            });
        },

        // TODO: This may belong in app.js
        stepGeneratorError: function() {
            this.state.animation.setStopped();
        },

        // TODO: This probably belongs in app.js
        stepGeneratorFinished: function() {
            this.state.animation.setStopped();
        },

        reset: function() {
            if (this.timeStepLayer) {
                this.leafletMap.removeLayer(this.timeStepLayer);
            }
            this.clearImageCache();
            // Return a promise to set the background image
            return this.setBackground();
        },

        pixelsFromCoordinates: function(point) {
            var size = this.getSize();
            var bounds = this.model.get('map_bounds');

            if (!size.height || !size.width) {
                throw new MapViewException('No current image size detected.');
            }

            if (!bounds) {
                throw new MapViewException('Map is missing boundary data.');
            }

            var minLat = bounds[0][1];
            var minLong = bounds[0][0];
            var maxLat = bounds[1][1];
            var maxLong = bounds[2][0];

            var x = ((point.long - minLong) / (maxLong - minLong)) * size.width;
            var y = ((point.lat - minLat) / (maxLat - minLat)) * size.height;

            // Adjust for different origin
            y = -y + size.height;

            return {x: Math.round(x), y: Math.round(y)};
        },

        coordinatesFromPixels: function(point) {
            var size = this.getSize();
            var bounds = this.model.get('map_bounds');

            if (!size.height || !size.width) {
                throw new MapViewException('No current image size detected.');
            }

            var minLat = bounds[0][1];
            var minLong = bounds[0][0];
            var maxLat = bounds[1][1];
            var maxLong = bounds[2][0];

            // Adjust for different origin
            point.y = -point.y + size.height;

            var lat = (maxLat - minLat) * (point.y / size.height) + minLat;
            var lng = (maxLong - minLong) * (point.x / size.width) + minLong;

            return {lat: lat, long: lng};
        }
    }, {
        // Event constants
        DRAGGING_FINISHED: 'mapView:draggingFinished',
        REFRESH_FINISHED: 'mapView:refreshFinished',
        PLAYING_FINISHED: 'mapView:playingFinished',
        FRAME_CHANGED: 'mapView:frameChanged',
        MAP_WAS_CLICKED: 'mapView:mapWasClicked',
        SPILL_DRAWN: 'mapView:spillDrawn',
        READY: 'mapView:ready',
        VIEWPORT_CHANGED: 'mapView:viewportChanged'
    });

    return {
        MapViewException: MapViewException,
        MapView: MapView
    }
});