
define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'util',
    'forms',
    'map_generator',
    'lib/jquery.imagesloaded.min',
    'lib/jquery.dynatree',
    'lib/bootstrap-collapse',
    'lib/bootstrap-dropdown',
    'async!http://maps.googleapis.com/maps/api/js?key=AIzaSyATcDk4cEYobGp9mq75DeZKaEdeppPnSlk&sensor=false&libraries=drawing'
], function($, _, Backbone, models, util, forms) {
     /*
     `MessageView` is responsible for displaying messages sent back from the server
     during AJAX form submissions. These are non-form error conditions, usually,
     but can also be success messages.
     */
    var MessageView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);

            this.options.gnomeRun.on(
                models.GnomeRun.MESSAGE_RECEIVED, this.displayMessage);
            this.options.gnomeModel.on(
                models.GnomeModel.MESSAGE_RECEIVED, this.displayMessage);
            this.options.surfaceReleaseSpills.on(
                models.SurfaceReleaseSpill.MESSAGE_RECEIVED, this.displayMessage);
            this.options.windMovers.on(
                models.WindMover.MESSAGE_RECEIVED, this.displayMessage);

            this.hideAll();
        },

        displayMessage: function(message) {
            if (!_.has(message, 'type') || !_.has(message, 'text')) {
                return false;
            }

            var alertDiv = $('div .alert-' + message.type);

            if (message.text && alertDiv) {
                alertDiv.find('span.message').text(message.text);
                alertDiv.fadeIn(10);
                // The hidden class makes the div hidden on page load. After
                // we fade out the first time, jQuery sets the display: none;
                // value on the element directly.
                alertDiv.removeClass('hidden');
            }

            this.hideAll();
            return true;
        },

        // Hide alerts automatically after a timeout.
        hideAll: function() {
            setTimeout(function() {
                $('.alert').fadeOut();
            }, 2000);
        }
    });

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
            this.frameClass = this.options.frameClass;
            this.activeFrameClass = this.options.activeFrameClass;
            this.placeholderClass = this.options.placeholderClass;
            this.latLongBounds = this.options.latLongBounds;
            this.animationThreshold = this.options.animationThreshold;
            this.locationFilesMeta = this.options.locationFilesMeta;
            this.canDrawSpill = false;

            this.makeImagesClickable();
            this.status = MapView.STOPPED;
            this.map = $(this.mapEl);

            this.gnomeRun = this.options.gnomeRun;
            this.gnomeRun.on(models.GnomeRun.NEXT_TIME_STEP_READY, this.nextTimeStepReady);
            this.gnomeRun.on(models.GnomeRun.RUN_BEGAN, this.gnomeRunBegan);
            this.gnomeRun.on(models.GnomeRun.RUN_ERROR, this.gnomeRunError);
            this.gnomeRun.on(models.GnomeRun.RUN_FINISHED, this.gnomeRunFinished);
            this.gnomeRun.on(models.GnomeRun.CREATED, this.reset);

            this.model = this.options.model;
            this.model.on('change:background_image_url', function() {
                _this.reset();
            });
            this.model.on('destroy', function () {
                _this.reset();
                _this.map.empty();
            });

            if (this.gnomeRun.hasCachedTimeStep(this.gnomeRun.getCurrentTimeStep())) {
                this.nextTimeStepReady();
            }
        },

        show: function() {
            this.setBackground();
        },

        setBackground: function() {
            var backgroundImageUrl = this.model.get('background_image_url');

            if (backgroundImageUrl) {
                this.loadMapFromUrl(backgroundImageUrl);
            } else {
                this.showPlaceholder();
            }
        },

        isPaused: function() {
            return this.status === MapView.PAUSED;
        },

        isStopped: function() {
            return this.status === MapView.STOPPED;
        },

        isPlaying: function() {
            return this.status === MapView.PLAYING;
        },

        setPaused: function() {
            this.status = MapView.PAUSED;
        },

        setStopped: function() {
            this.status = MapView.STOPPED;
        },

        setPlaying: function() {
            this.status = MapView.PLAYING;
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

        timeStepIsLoaded: function(stepNum) {
            var step = this.getImageForTimeStep(stepNum);
            return step && step.length;
        },

        /*
         Show the image for time step with ID `stepNum`.

         Triggers:
            - `MapView.FRAME_CHANGED` after the image has loaded.
         */
        showImageForTimeStep: function(stepNum) {
            var mapImages = this.map.find('img');

            // Show the map div if this is the first image of the run.
            if (mapImages.length === 1) {
                this.map.show();
            }

            var stepImage = this.getImageForTimeStep(stepNum);

            // The image isn't loaded.
            if (stepImage.length === 0) {
                alert("An animation error occurred.");
                console.log('Could not load image for timestep: ' + stepNum);
                return;
            }

            var imagesToHide = mapImages.not(stepImage).not('.background');

            // Hide all images in the map div other than the background and the
            // image for the current step.
            imagesToHide.addClass('hidden');
            imagesToHide.removeClass(this.activeFrameClass);

            stepImage.addClass(this.activeFrameClass);
            stepImage.removeClass('hidden');

            this.trigger(MapView.FRAME_CHANGED);
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

        addImageForTimeStep: function(timeStep) {
            var _this = this;
            var map = $(this.mapEl);
            var requestTime = timeStep.get('requestTime');

            var img = $('<img>').attr({
                'class': 'frame',
                'data-id': timeStep.id,
                'src': timeStep.get('url')
            }).addClass('hidden');

            img.appendTo(map);

            $(img).imagesLoaded(function() {
                // TODO: Check how much time has passed after next time
                // // step is ready. If longer than N, show the image
                // immediately. Otherwise, set a delay and then show image.

                // TODO: Make the timeout value configurable.
                setTimeout(_this.showImageForTimeStep,
                           _this.getAnimationTimeout(requestTime),
                           [timeStep.id]);
            });
        },

        addTimeStep: function(timeStep) {
            var imageExists = this.getImageForTimeStep(timeStep.id).length;

            // We must be playing a cached model run because the image already
            // exists. In all other cases the image should NOT exist.
            if (imageExists) {
                setTimeout(this.showImageForTimeStep,
                           // Use 0 since this was a cached time step.
                           this.getAnimationTimeout(0),
                           [timeStep.id]);
                return;
            }

            this.addImageForTimeStep(timeStep);
        },

        // Clear out the current frames.
        clear: function(opts) {
            var map = $(this.mapEl);
            opts = opts || {};

            if (opts.clearBackground) {
                map.find('img').remove();
            } else {
                map.find('img').not('.background').remove();
            }
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

        setZoomingInCursor: function() {
            $(this.mapEl).addClass('zooming-in');
        },

        setZoomingOutCursor: function() {
            $(this.mapEl).addClass('zooming-out');
        },

        setRegularCursor: function() {
            $(this.mapEl).removeClass('zooming-out');
            $(this.mapEl).removeClass('zooming-in');
        },

        getRect: function(rect) {
            var newStartPosition, newEndPosition;

            // Do a shallow object copy, so we don't modify the original.
            if (rect.end.x > rect.start.x || rect.end.y > rect.start.y) {
                newStartPosition = $.extend({}, rect.start);
                newEndPosition = $.extend({}, rect.end);
            } else {
                newStartPosition = $.extend({}, rect.end);
                newEndPosition = $.extend({}, rect.start);
            }

            return {start: newStartPosition, end: newEndPosition};
        },

        // Adjust a selection rectangle so that it fits within the bounding box.
        getAdjustedRect: function(rect) {
            var adjustedRect = this.getRect(rect);
            var bbox = this.getBoundingBox();

            // TOOD: This looks wrong. Add tests.
            if (adjustedRect.start.x > bbox[0].x) {
                adjustedRect.start.x = bbox[0].x;
            }
            if (adjustedRect.start.y < bbox[0].y) {
                adjustedRect.start.y = bbox[0].y;
            }

            if (adjustedRect.end.x < bbox[1].x) {
                adjustedRect.end.x = bbox[1].x;
            }
            if (adjustedRect.end.y > bbox[1].y) {
                adjustedRect.end.y = bbox[1].y;
            }

            return adjustedRect;
        },

        isPositionInsideMap: function(position) {
            var bbox = this.getBoundingBox();
            return (position.x > bbox[0].x && position.x < bbox[1].x &&
                position.y > bbox[0].y && position.y < bbox[1].y);
        },

        isRectInsideMap: function(rect) {
            var _rect = this.getRect(rect);

            return this.isPositionInsideMap(_rect.start) &&
                this.isPositionInsideMap(_rect.end);
        },

        nextTimeStepReady: function() {
            this.addTimeStep(this.gnomeRun.getCurrentTimeStep());
        },

        loadMapFromUrl: function(url) {
            var _this = this;

            this.hidePlaceholder();

            var map = $(this.mapEl);
            map.find('.background').remove();

            var background = $('<img>').attr({
                class: 'background',
                src: url
            });

            background.imagesLoaded(function() {
                _this.createCanvases();
                _this.trigger(MapView.READY);
            });

            background.appendTo(map);
        },

        drawLine: function(ctx, start_x, start_y, end_x, end_y) {
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(start_x, start_y);
            ctx.lineTo(end_x, end_y);
            ctx.stroke();
            ctx.closePath();
            ctx.beginPath();
            ctx.closePath();
        },

        drawSpill: function(spill) {
            var ctx = this.backgroundCanvas[0].getContext('2d');
            var startX, startY, startZ, endX, endY, endZ;
            var start = spill.get('start_position');
            var end = spill.get('end_position');

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

            var pixelStart = this.pixelsFromCoordinates({
                lat: startY,
                long: startX
            });

            var pixelEnd = this.pixelsFromCoordinates({
                lat: endY,
                long: endX
            });

            if (startX === endX && startY === endY) {
                pixelEnd.x += 2;
                pixelEnd.y += 2;
            }

            this.drawLine(ctx, pixelStart.x, pixelStart.y, pixelEnd.x, pixelEnd.y);
        },
        
        clearCanvas: function(canvas) {
            var ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);           
        },

        // Draw a mark on the map for each existing spill.
        drawSpills: function(spills) {
            var _this = this;

            if (!this.backgroundCanvas || !this.foregroundCanvas) {
                return;
            }

            this.clearCanvas(this.backgroundCanvas[0]);
            this.clearCanvas(this.foregroundCanvas[0]);

            if (spills === undefined || !spills.length) {
                return;
            }

            spills.forEach(function(spill) {
                _this.drawSpill(spill);
            });
        },

        /*
         Create a foreground canvas and setup event handlers to capture new
         spills added to the map. This canvas is cleared entirely during line
         additions (as the line position changes) and when the form container
         refreshes.

         TODO: Update canvas sizes when window changes.
         */
        createCanvases: function() {
            var _this = this;
            var background = $(this.mapEl).find('.background');

            if (this.backgroundCanvas) {
                $(this.backgroundCanvas).remove();
            }

            if (this.foregroundCanvas) {
                $(this.foregroundCanvas).remove();
            }

            this.backgroundCanvas = $('<canvas>').attr({
                id: 'canvas-background',
                class: 'drawable-background',
                height: background.height(),
                width: background.width()
            });

            this.foregroundCanvas = $('<canvas>').attr({
                id: 'canvas-foreground',
                class: 'drawable-foreground',
                height: background.height(),
                width: background.width()
            });

            this.foregroundCanvas.mousedown(function(ev) {
                if (!_this.canDrawSpill) {
                    return;
                }

                this.pressed = true;
                if (ev.originalEvent['layerX'] != undefined) {
                    this.x0 = ev.originalEvent.layerX;
                    this.y0 = ev.originalEvent.layerY;
                }
                else {
                    // in IE, we use this property
                    this.x0 = ev.originalEvent.x;
                    this.y0 = ev.originalEvent.y;
                }
            });

            // Event handlers to draw new spills
            this.foregroundCanvas.mousemove(function(ev) {
                if (!this.pressed) {
                    return;
                }
                this.moved = true;
                var ctx = this.getContext('2d');
                var xcurr, ycurr;
                if (ev.originalEvent['layerX'] != undefined) {
                    xcurr = ev.originalEvent.layerX;
                    ycurr = ev.originalEvent.layerY;
                }
                else {
                    // in IE, we use this property
                    xcurr = ev.originalEvent.x;
                    ycurr = ev.originalEvent.y;
                }

                ctx.clearRect(0, 0, this.width, this.height);
                _this.drawLine(ctx, this.x0, this.y0, xcurr, ycurr);
            });

            $(this.foregroundCanvas).mouseup(function(ev) {
                var canvas = _this.backgroundCanvas[0];
                var ctx = canvas.getContext('2d');
                var offset = $(this).offset();
                var endX = ev.pageX - offset.left;
                var endY = ev.pageY - offset.top;

                _this.drawLine(ctx, this.x0, this.y0, endX, endY);
                _this.clearCanvas(_this.foregroundCanvas[0]);

                if (this.pressed && this.moved) {
                    var start = _this.coordinatesFromPixels({
                        x: this.x0,
                        y: this.y0
                    });
                    var end = _this.coordinatesFromPixels({
                        x: endX,
                        y: endY
                    });

                    _this.trigger(MapView.SPILL_DRAWN, [start.long, start.lat],
                                  [end.long, end.lat]);
                }
                this.pressed = this.moved = false;
            });

            this.backgroundCanvas.appendTo(map);
            this.foregroundCanvas.appendTo(map);
        },

        gnomeRunBegan: function() {
            this.loadMapFromUrl(this.model.get('background_image_url'));
        },

        gnomeRunError: function() {
            this.setStopped();
        },

        gnomeRunFinished: function() {
            this.setStopped();
        },

        reset: function() {
            this.clear({clearBackground: true});
            this.setBackground();
            this.setStopped();
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
        // Statuse constants
        PAUSED: 1,
        STOPPED: 2,
        PLAYING: 3,

        // Event constants
        DRAGGING_FINISHED: 'mapView:draggingFinished',
        REFRESH_FINISHED: 'mapView:refreshFinished',
        PLAYING_FINISHED: 'mapView:playingFinished',
        FRAME_CHANGED: 'mapView:frameChanged',
        MAP_WAS_CLICKED: 'mapView:mapWasClicked',
        SPILL_DRAWN: 'mapView:spillDrawn',
        READY: 'mapView:ready'
    });


    /*
     `TreeView` is a representation of the user's current model displayed as a tree
     of items that the user may click or double-click on to display add/edit forms
     for model settings, movers and spills.
     */
    var TreeView = Backbone.View.extend({
        initialize: function() {
            var _this = this;
            _.bindAll(this);
            this.treeEl = this.options.treeEl;
            this.gnomeModel = this.options.gnomeModel;
            this.url = this.gnomeModel.url() + "/tree";

            // Turn off node icons. A [+] icon will still appear for nodes
            // that have children.
            $.ui.dynatree.nodedatadefaults["icon"] = false;
            this.tree = this.setupDynatree();

            _.each(this.options.collections, function(collection) {
                collection.on('sync', _this.reload);
                collection.on('add', _this.reload);
                collection.on('destroy', _this.reload);
            });

            this.gnomeModel.on('sync', this.reload);
            this.options.map.on('sync', this.reload);

            $(window).bind('resize', function() {
                _this.resize();
            });

            _this.resize();
        },

        /*
         Adjust the sidebar height to stay at 100% of the page minus the navbar.
         */
        resize: function() {
            var windowHeight = $(window).height();
            var navbarHeight = $('.navbar').height();
            var tree = $('#tree');
            var sidebar = $('#sidebar');
            var treeHeight = tree.height();
            var treeHeightDiff = sidebar.height() - treeHeight;
            var newSidebarHeight = windowHeight - navbarHeight;
            sidebar.height(newSidebarHeight);
            tree.height(newSidebarHeight - treeHeightDiff);
        },

        setupDynatree: function() {
            var _this = this;

            return $(this.treeEl).dynatree({
                onActivate: function(node) {
                    _this.trigger(TreeView.ITEM_ACTIVATED, node);
                },
                onPostInit: function(isReloading, isError) {
                    // Fire events for a tree that was reloaded from cookies.
                    // isReloading is true if status was read from existing cookies.
                    // isError is only used in Ajax mode
                    this.reactivate();

                     // Expand all items
                    this.getRoot().visit(function (node) {
                        node.expand(true);
                    });
                },
                onDblClick: function(node, event) {
                    _this.trigger(TreeView.ITEM_DOUBLE_CLICKED, node);
                },
                initAjax: {
                    url: _this.url
                },
                windage_persist: true
            });
        },

        getActiveItem: function() {
            return this.tree.dynatree("getActiveNode");
        },

        reload: function() {
            var _this = this;
            if (this.gnomeModel && this.gnomeModel.wasDeleted) {
                return;
            }
            this.tree.dynatree('getTree').reload(function () {
                _this.trigger(TreeView.RELOADED);
                _this.resize();
            });
        }
    }, {
        ITEM_ACTIVATED: 'treeView:treeItemActivated',
        ITEM_DOUBLE_CLICKED: 'treeView:treeItemDoubleClicked',
        RELOADED: 'treeView:reloaded'
    });


    /*
     `TreeControlView` is a button bar that sits above the tree view and allows
     the user to add, edit and remove settings values, movers and spills using
     button clicks.
     */
    var TreeControlView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.addButtonEl = this.options.addButtonEl;
            this.removeButtonEl = this.options.removeButtonEl;
            this.settingsButtonEl = this.options.settingsButtonEl;

            // Controls that require the user to select an item in the TreeView.
            this.itemControls = [this.removeButtonEl, this.settingsButtonEl];
            this.disableControls();
            this.setupClickEvents();

            this.options.treeView.on(TreeView.ITEM_ACTIVATED, this.treeItemActivated);
        },

        setupClickEvents: function() {
            var _this = this;
            var clickEvents = [
                [this.addButtonEl, TreeControlView.ADD_BUTTON_CLICKED],
                [this.removeButtonEl, TreeControlView.REMOVE_BUTTON_CLICKED],
                [this.settingsButtonEl, TreeControlView.SETTINGS_BUTTON_CLICKED]
            ];

            _.each(_.object(clickEvents), function(customEvent, element) {
                $(element).click(function(event) {
                    if ($(_this).hasClass('disabled')) {
                        return false;
                    }
                    _this.trigger(customEvent);
                    return true;
                });
            });
        },

        treeItemActivated: function() {
            this.enableControls();
        },

        enableControls: function() {
            _.each(this.itemControls, function(buttonEl) {
                $(buttonEl).removeClass('disabled');
            });
        },

        disableControls: function() {
            _.each(this.itemControls, function(buttonEl) {
                $(buttonEl).addClass('disabled');
            });
        }
    }, {
        // Event constants 
        ADD_BUTTON_CLICKED: 'gnome:addItemButtonClicked',
        REMOVE_BUTTON_CLICKED: 'gnome:removeItemButtonClicked',
        SETTINGS_BUTTON_CLICKED: 'gnome:itemSettingsButtonClicked'
    });


    /*
     `MapControlView` is a button toolbar that sits above the map and allows the
     user to stop, start, skip to the end, skip to the beginning, and scrub between
     frames of an animation generated during a model run.
     */
    var MapControlView = Backbone.View.extend({
        initialize: function() {
            var _this = this;
            _.bindAll(this);
            this.containerEl = this.options.containerEl;
            this.sliderEl = this.options.sliderEl;
            this.playButtonEl = this.options.playButtonEl;
            this.pauseButtonEl = this.options.pauseButtonEl;
            this.backButtonEl = this.options.backButtonEl;
            this.forwardButtonEl = this.options.forwardButtonEl;
            this.zoomInButtonEl = this.options.zoomInButtonEl;
            this.zoomOutButtonEl = this.options.zoomOutButtonEl;
            this.moveButtonEl = this.options.moveButtonEl;
            this.fullscreenButtonEl = this.options.fullscreenButtonEl;
            this.resizeButtonEl = this.options.resizeButtonEl;
            this.spillButtonEl = this.options.spillButtonEl;
            this.sliderShadedEl = this.options.sliderShadedEl;
            this.timeEl = this.options.timeEl;
            this.mapView = this.options.mapView;
            this.gnomeRun = this.options.gnomeRun;
            this.model = this.options.model;

            this.animationControls = [
                this.backButtonEl, this.forwardButtonEl, this.playButtonEl,
                this.pauseButtonEl
            ];

            this.mapControls = [
                this.moveButtonEl, this.zoomInButtonEl, this.zoomOutButtonEl,
                this.spillButtonEl
            ];

            if (this.model.id) {
                this.enableControls(this.mapControls);
            }

            this.model.on('sync', function() {
                if (_this.model.id) {
                    _this.enableControls(_this.mapControls);
                }
            });

            this.controls = this.animationControls.concat(this.mapControls);
            this.status = MapControlView.STATUS_STOPPED;

            $(this.pauseButtonEl).hide();
            $(this.resizeButtonEl).hide();

            $(this.sliderEl).slider({
                start: this.sliderStarted,
                change: this.sliderChanged,
                slide: this.sliderMoved,
                disabled: true
            });

            if (this.gnomeRun.expectedTimeSteps.length) {
                this.setTimeSteps(this.gnomeRun.expectedTimeSteps);
                this.enableControls();
            }

            this.setupClickEvents();
            this.updateCachedPercentage();

            this.gnomeRun.on(models.GnomeRun.RUN_BEGAN, this.runBegan);
            this.gnomeRun.on(models.GnomeRun.RUN_ERROR, this.gnomeRunError);
            this.gnomeRun.on(models.GnomeRun.NEXT_TIME_STEP_READY, this.updateCachedPercentage);
            this.gnomeRun.on(models.GnomeRun.RUN_FINISHED, this.gnomeRunFinished);
            this.gnomeRun.on(models.GnomeRun.CREATED, this.modelCreated);

            this.options.mapView.on(MapView.FRAME_CHANGED, this.mapViewFrameChanged);
        },

        setupClickEvents: function() {
            var _this = this;

            var clickEvents = [
                [this.playButtonEl, MapControlView.PLAY_BUTTON_CLICKED],
                [this.backButtonEl, MapControlView.BACK_BUTTON_CLICKED],
                [this.forwardButtonEl, MapControlView.FORWARD_BUTTON_CLICKED],
                [this.zoomInButtonEl, MapControlView.ZOOM_IN_BUTTON_CLICKED],
                [this.zoomOutButtonEl, MapControlView.ZOOM_OUT_BUTTON_CLICKED],
                [this.moveButtonEl, MapControlView.MOVE_BUTTON_CLICKED],
                [this.fullscreenButtonEl, MapControlView.FULLSCREEN_BUTTON_CLICKED],
                [this.resizeButtonEl, MapControlView.RESIZE_BUTTON_CLICKED],
                [this.pauseButtonEl, MapControlView.PAUSE_BUTTON_CLICKED],
                [this.spillButtonEl, MapControlView.SPILL_BUTTON_CLICKED]
            ];

            // TODO: This probably leaks memory due to closing around `button`.
            _.each(_.object(clickEvents), function(customEvent, button) {
                $(button).click(function(event) {
                    if ($(button).hasClass('disabled')) {
                        return false;
                    }
                    _this.trigger(customEvent);
                    return true;
                });
            });
        },

        sliderStarted: function(event, ui) {
            this.trigger(MapControlView.PAUSE_BUTTON_CLICKED);
        },

        sliderChanged: function(event, ui) {
            this.trigger(MapControlView.SLIDER_CHANGED, ui.value);
        },

        sliderMoved: function(event, ui) {
            var timestamp = this.gnomeRun.getTimestampForExpectedStep(ui.value);

            if (timestamp) {
                this.setTime(timestamp);
            } else {
                util.log('Slider changed to invalid time step: ' + ui.value);
                return false;
            }

            this.trigger(MapControlView.SLIDER_MOVED, ui.value);
        },

        updateCachedPercentage: function() {
            this.setCachedPercentage(
                100*(this.gnomeRun.length / this.gnomeRun.expectedTimeSteps.length))
        },

        /*
         Set the slider to `value`.

         This triggers the "change" event on the widget.
         */
        setValue: function(value) {
            this.sliderMoved(null, {value: value});
            $(this.sliderEl).slider('value', value);
        },

        runBegan: function() {
            if (this.gnomeRun.dirty) {
                // TODO: Is this really what we want to do here?
                this.reset();
            }

            this.setTimeSteps(this.gnomeRun.expectedTimeSteps);
        },

        mapViewFrameChanged: function() {
            var timeStep = this.gnomeRun.getCurrentTimeStep();
            this.setTimeStep(timeStep.id);
            this.setTime(timeStep.get('timestamp'));
        },

        stop: function() {
            this.setStopped();
            this.enableControls();
        },

        gnomeRunError: function() {
            this.stop();
        },

        gnomeRunFinished: function() {
            this.disableControls();
            this.stop();
        },

        modelCreated: function() {
            this.reset();
        },

        setStopped: function() {
            this.status = MapControlView.STATUS_STOPPED;
            $(this.pauseButtonEl).hide();
            $(this.playButtonEl).show();
        },

        setPlaying: function() {
            this.status = MapControlView.STATUS_PLAYING;
            $(this.pauseButtonEl).show();
            $(this.playButtonEl).hide();
        },

        setPaused: function() {
            this.status = MapControlView.STATUS_PAUSED;
            $(this.pauseButtonEl).hide();
            $(this.playButtonEl).show();
        },

        setForward: function() {
            this.status = MapControlView.STATUS_FORWARD;
        },

        setBack: function() {
            this.status = MapControlView.STATUS_BACK;
        },

        setZoomingIn: function() {
            this.status = MapControlView.STATUS_ZOOMING_IN;
        },

        setZoomingOut: function() {
            this.status = MapControlView.STATUS_ZOOMING_OUT;
        },

        setTimeStep: function(stepNum) {
            $(this.sliderEl).slider('value', stepNum);
        },

        setTime: function(time) {
            $(this.timeEl).text(time);
        },

        setTimeSteps: function(timeSteps) {
            $(this.sliderEl).slider('option', 'max', timeSteps.length - 1);
        },

        switchToFullscreen: function() {
            $(this.fullscreenButtonEl).hide();
            $(this.resizeButtonEl).show();
        },

        switchToNormalScreen: function() {
            $(this.resizeButtonEl).hide();
            $(this.fullscreenButtonEl).show();
        },

        isPlaying: function() {
            return this.status === MapControlView.STATUS_PLAYING;
        },

        isStopped: function() {
            return this.status === MapControlView.STATUS_STOPPED;
        },

        isPaused: function() {
            return this.status === MapControlView.STATUS_PAUSED;
        },

        isForward: function() {
            return this.status === MapControlView.STATUS_PLAYING;
        },

        isBack: function() {
            return this.status === MapControlView.STATUS_BACK;
        },

        isZoomingIn: function() {
            return this.status === MapControlView.STATUS_ZOOMING_IN;
        },

        isZoomingOut: function() {
            return this.status === MapControlView.STATUS_ZOOMING_OUT;
        },

        // Toggle the slider. `toggleOn` should be either `MapControlView.ON`
        // or `MapControlView.OFF`.
        toggleSlider: function(toggle) {
            var value = toggle !== MapControlView.ON;
            $(this.sliderEl).slider('option', 'disabled', value);
        },

        // Toggle a control. `toggleOn` should be either `MapControlView.ON`
        // or `MapControlView.OFF`.
        toggleControl: function(buttonEl, toggle) {
            if (toggle === MapControlView.ON) {
                $(buttonEl).removeClass('disabled');
                return;
            }

            $(buttonEl).addClass('disabled');
        },

        /*
         Enable or disable specified controls.

         If `this.sliderEl` is present in `controls`, use the `this.toggleSlider`
         function to toggle it.

         If `controls` is empty, apply `toggle` to all controls in `this.controls`.

         Options:
         - `controls`: an array of HTML elements to toggle
         - `toggle`: either `MapControlView.OFF` or `MapControlView.ON`.
         */
        toggleControls: function(controls, toggle) {
            var _this = this;

            if (controls && controls.length) {
                if (_.contains(controls, this.sliderEl)) {
                    this.toggleSlider(toggle);
                }
                _.each(_.without(controls, this.sliderEl), function(button) {
                    _this.toggleControl(button, toggle);
                });
                return;
            }

            this.toggleSlider(toggle);
            _.each(this.controls, function(button) {
                _this.toggleControl(button, toggle);
            });
        },

        enableControls: function(controls) {
            this.toggleControls(controls, MapControlView.ON);
        },

        disableControls: function(controls) {
            this.toggleControls(controls, MapControlView.OFF);
        },

        getTimeStep: function() {
            $(this.sliderEl).slider('value');
        },

        setCachedPercentage: function(percentage) {
            $(this.sliderShadedEl).css('width', percentage + '%');
        },

        reset: function() {
            this.setTime('00:00');
            this.disableControls();
            this.setStopped();
            this.setTimeStep(0);
            $(this.sliderEl).slider('values', null);
            this.enableControls([this.playButtonEl]);
        }
    }, {
        // Constants
        ON: true,
        OFF: false,

        // Event constants
        PLAY_BUTTON_CLICKED: "mapControlView:playButtonClicked",
        PAUSE_BUTTON_CLICKED: "mapControlView:pauseButtonClicked",
        BACK_BUTTON_CLICKED: "mapControlView:backButtonClicked",
        FORWARD_BUTTON_CLICKED: "mapControlView:forwardButtonClicked",
        ZOOM_IN_BUTTON_CLICKED: "mapControlView:zoomInButtonClicked",
        ZOOM_OUT_BUTTON_CLICKED: "mapControlView:zoomOutButtonClicked",
        MOVE_BUTTON_CLICKED: "mapControlView:moveButtonClicked",
        FULLSCREEN_BUTTON_CLICKED: "mapControlView:fullscreenButtonClicked",
        RESIZE_BUTTON_CLICKED: "mapControlView:resizeButtonClicked",
        SPILL_BUTTON_CLICKED: "mapControllView:spillButtonClicked",
        SLIDER_CHANGED: "mapControlView:sliderChanged",
        SLIDER_MOVED: "mapControlView:sliderMoved",

        // Status constants
        STATUS_STOPPED: 0,
        STATUS_PLAYING: 1,
        STATUS_PAUSED: 2,
        STATUS_BACK: 3,
        STATUS_FORWARD: 4,
        STATUS_ZOOMING_IN: 5,
        STATUS_ZOOMING_OUT: 6
    });


    /*
     `MenuView` handles the drop-down menus on the top of the page. The object
     listens for click events on menu items and fires specialized events, like
      RUN_ITEM_CLICKED, which an `AppView` object listens for.

      Most of these functions exist elsewhere in the application and `AppView`
      calls the appropriate method for whatever functionality the user invoked.
     */
    var MenuView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            // Top-level drop-downs
            this.modelDropdownEl = this.options.modelDropdownEl;
            this.runDropdownEl = this.options.runDropdownEl;
            this.helpDropdownEl = this.options.helpDropdownEl;

            // Drop-down children
            this.newItemEl = this.options.newItemEl;
            this.runItemEl = this.options.runItemEl;
            this.stepItemEl = this.options.stepItemEl;
            this.runUntilItemEl = this.options.runUntilItemEl;
            this.longIslandItemEl = this.options.longIslandItemEl;

            $(this.newItemEl).click(this.newItemClicked);
            $(this.runItemEl).click(this.runItemClicked);
            $(this.runUntilItemEl).click(this.runUntilItemClicked);

            $('ul.nav').on('click', '.location-file-item', this.locationFileItemClicked);
        },

        hideDropdown: function() {
            $(this.modelDropdownEl).dropdown('toggle');
        },

        newItemClicked: function(event) {
            this.hideDropdown();
            this.trigger(MenuView.NEW_ITEM_CLICKED);
        },

        runItemClicked: function(event) {
            this.hideDropdown();
            this.trigger(MenuView.RUN_ITEM_CLICKED);
        },

        runUntilItemClicked: function(event) {
            this.hideDropdown();
            this.trigger(MenuView.RUN_UNTIL_ITEM_CLICKED);
        },

        locationFileItemClicked: function(event) {
            event.preventDefault();
            this.hideDropdown();
            var location = $(event.target).data('location');
            this.trigger(MenuView.LOCATION_FILE_ITEM_CLICKED, location);
        }
    }, {
        // Event constants
        NEW_ITEM_CLICKED: "menuView:newMenuItemClicked",
        RUN_ITEM_CLICKED: "menuView:runMenuItemClicked",
        RUN_UNTIL_ITEM_CLICKED: "menuView:runUntilMenuItemClicked",
        LOCATION_FILE_ITEM_CLICKED: "menuView:locationFileItemClicked"
    });


    var SplashView = Backbone.View.extend({
        initialize: function() {
            this.router = this.options.router;
        },

        events: {
            'click .choose-location': 'chooseLocation',
            'click .build-model': 'buildModel'
        },

        chooseLocation: function(event) {
            event.preventDefault();
            this.router.navigate('location_map', true);
        },

        buildModel: function(event) {
            event.preventDefault();
            this.router.navigate('model', true);
        }
    });


    var LocationFileMapView = Backbone.View.extend({
        events: {
            'click .load-location-file': 'locationChosen'
        },

        initialize: function() {
            _.bindAll(this);
            this.mapCanvas = $(this.options.mapCanvas);
            this.locationFilesMeta = this.options.locationFilesMeta;

            this.setupLocationFileMap();
        },

        locationChosen: function(event) {
            event.preventDefault();
            var location = $(event.target).data('location');
            if (location) {
                this.trigger(LocationFileMapView.LOCATION_CHOSEN, location);
            }
            this.infoWindow.close();
        },

        setupLocationFileMap: function() {
            var _this = this;
            this.center = new google.maps.LatLng(-34.397, 150.644);
            this.infoWindow = new google.maps.InfoWindow();
            var gmapOptions = {
                center: this.center,
                backgroundColor: '#212E68',
                zoom: 1,
                scrollwheel: true,
                scaleControl: true,
                mapTypeId: google.maps.MapTypeId.HYBRID,
            };

            this.locationFileMap = new google.maps.Map(
                this.mapCanvas[0], gmapOptions);

            this.locationFilesMeta.each(function(location) {
                var latLng = new google.maps.LatLng(
                    location.get('latitude'), location.get('longitude'));

                var marker = new google.maps.Marker({
                    position: latLng,
                    map: _this.locationFileMap
                });

                google.maps.event.addListener(marker, 'click', function() {
                    var template = _.template(
                        $('#location-file-template').text());
                    _this.infoWindow.setContent(template(location.toJSON()));
                    _this.infoWindow.open(_this.locationFileMap, marker);
                });
            });
        },

        centerMap: function() {
            google.maps.event.trigger(this.locationFileMap, 'resize');
            this.locationFileMap.setCenter(this.center);
        },

        show: function() {
            this.$el.imagesLoaded(this.centerMap);
        }
    }, {
        LOCATION_CHOSEN: 'locationFileMapView:locationChosen'
    });


    return {
        MessageView: MessageView,
        MapView: MapView,
        TreeView: TreeView,
        TreeControlView: TreeControlView,
        MapControlView: MapControlView,
        MenuView: MenuView,
        SplashView: SplashView,
        LocationFileMapView: LocationFileMapView
    };

});