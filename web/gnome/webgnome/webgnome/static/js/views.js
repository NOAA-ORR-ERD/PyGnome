
define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'util',
    'lib/jquery.imagesloaded.min',
    'lib/jquery.dynatree.min',
    'lib/bootstrap',
], function($, _, Backbone, models, util) {

     /*
     `MessageView` is responsible for displaying messages sent back from the server
     during AJAX form submissions. These are non-form error conditions, usually,
     but can also be success messages.
     */
    var MessageView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);

            this.options.model.on(
                models.Model.MESSAGE_RECEIVED, this.displayMessage);
            this.options.ajaxForms.on(
                models.AjaxForm.MESSAGE_RECEIVED, this.displayMessage);

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
            _.bindAll(this);
            this.mapEl = this.options.mapEl;
            this.frameClass = this.options.frameClass;
            this.activeFrameClass = this.options.activeFrameClass;
            this.placeholderEl = this.options.placeholderEl;
            this.backgroundImageUrl = this.options.backgroundImageUrl;
            this.latLongBounds = this.options.latLongBounds;

            this.createPlaceholderCopy();
            this.makeImagesClickable();
            this.status = MapView.STOPPED;
            this.map = $(this.mapEl);

            this.model = this.options.model;
            this.model.on(models.Model.NEXT_TIME_STEP_READY, this.nextTimeStepReady);
            this.model.on(models.Model.RUN_BEGAN, this.modelRunBegan);
            this.model.on(models.Model.RUN_ERROR, this.modelRunError);
            this.model.on(models.Model.RUN_FINISHED, this.modelRunFinished);
            this.model.on(models.Model.CREATED, this.modelCreated);

            if (this.backgroundImageUrl) {
                this.loadMapFromUrl(this.backgroundImageUrl);
            }

            if (this.model.hasCachedTimeStep(this.model.getCurrentTimeStep())) {
                this.nextTimeStepReady();
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

        createPlaceholderCopy: function() {
            this.placeholderCopy = $(this.placeholderEl).find(
                'img').clone().appendTo($(this.mapEl)).show();
        },

        removePlaceholderCopy: function() {
            this.placeholderCopy.remove();
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
            // Show the map div if this is the first image of the run.
            if (this.map.find('img').length === 1) {
                this.map.show();
            }

            var stepImage = this.getImageForTimeStep(stepNum);
            var otherImages = this.map.find('img').not(stepImage).not('.background');

            // Hide all other images in the map div.
            otherImages.css('display', 'none');
            otherImages.removeClass(this.activeFrameClass);

            // The image isn't loaded.
            if (stepImage.length === 0) {
                alert("An animation error occurred. Please refresh.");
            }

            stepImage.addClass(this.activeFrameClass);
            stepImage.css('display', 'block');

            this.trigger(MapView.FRAME_CHANGED);
        },

        addImageForTimeStep: function(timeStep) {
            var _this = this;
            var map = $(this.mapEl);

            var img = $('<img>').attr({
                'class': 'frame',
                'data-id': timeStep.id,
                'src': timeStep.get('url')
            }).css('display', 'none');

            img.appendTo(map);

            $(img).imagesLoaded(function() {
                setTimeout(_this.showImageForTimeStep, 50, [timeStep.id]);
            });
        },

        addTimeStep: function(timeStep) {
            var imageExists = this.getImageForTimeStep(timeStep.id).length;

            // We must be playing a cached model run because the image already
            // exists. In all other cases the image should NOT exist.
            if (imageExists) {
                setTimeout(this.showImageForTimeStep, 50, [timeStep.id]);
                return;
            }

            this.addImageForTimeStep(timeStep);
        },

        // Clear out the current frames.
        clear: function() {
            $(this.mapEl).not('.background').empty();
        },

        getBackground: function() {
            return $(this.mapEl).find('img.background')[0];
        },

        getSize: function() {
            var image = this.getActiveImage();
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
            this.addTimeStep(this.model.getCurrentTimeStep());
        },

        loadMapFromUrl: function(url) {
            if (this.placeholderCopy.length) {
                this.removePlaceholderCopy();
            }

            var map = $(this.mapEl);
            map.find('.background').remove();

            var background = $('<img>').attr({
                class: 'background',
                src: url
            });

            background.imagesLoaded(this.createCanvas);
            background.appendTo(map);
        },

        createCanvas: function() {
            var _this = this;
            var background = $(this.mapEl).find('.background');

            var canvas = $('<canvas>').attr({
                id: 'canvas-background',
                class: 'drawable',
                height: background.height(),
                width: background.width()
            });

            // TODO: Update canvas size when window changes.

            canvas.mousedown(function(ev) {
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

            canvas.mousemove(function(ev) {
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

                // draw a line from the center
                ctx.lineWidth = 2;
                ctx.clearRect(0, 0, this.width, this.height);
                ctx.beginPath();
                ctx.moveTo(this.width / 2, this.height / 2);
                ctx.lineTo(xcurr, ycurr);
                ctx.stroke();
                ctx.closePath();

                ctx.beginPath();
                ctx.closePath();
            });

            $(canvas).mouseup(function(ev) {
                if (this.pressed && this.moved) {
                    var coords = _this.coordinatesFromPixels({
                        x: ev.clientX,
                        y: ev.clientY
                    });
                    _this.trigger(MapView.SPILL_DRAWN, coords.lat, coords.long);
                }
                this.pressed = this.moved = false;
            });

            canvas.appendTo(map);
        },

        modelRunBegan: function(data) {
            this.loadMapFromUrl(data.background_image);
        },

        modelRunError: function() {
            this.setStopped();
        },

        modelRunFinished: function() {
            this.setStopped();
        },

        modelCreated: function() {
            this.clear();
            this.createPlaceholderCopy();
            this.setStopped();
        },

        pixelsFromCoordinates: function(point) {
            var size = this.getSize();

            if (!size.height || !size.width) {
                throw new MapViewException('No current image size detected.');
            }

            var minLat = this.latLongBounds[0][1];
            var minLong = this.latLongBounds[0][0];
            var maxLat = this.latLongBounds[1][1];
            var maxLong = this.latLongBounds[2][0];

            var x = ((point.long - minLong) / (maxLong - minLong)) * size.width;
            var y = ((point.lat - minLat) / (maxLat - minLat)) * size.height;

            return {x: Math.round(x), y: Math.round(y)};
        },

        coordinatesFromPixels: function(point) {
            var size = this.getSize();

            if (!size.height || !size.width) {
                throw new MapViewException('No current image size detected.');
            }

            var minLat = this.latLongBounds[0][1];
            var minLong = this.latLongBounds[0][0];
            var maxLat = this.latLongBounds[1][1];
            var maxLong = this.latLongBounds[2][0];

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
        SPILL_DRAWN: 'mapView:spillDrawn'
    });


    /*
     `TreeView` is a representation of the user's current model displayed as a tree
     of items that the user may click or double-click on to display add/edit forms
     for model settings, movers and spills.
     */
    var TreeView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.treeEl = this.options.treeEl;
            this.url = this.options.url;
            this.tree = this.setupDynatree();

            // Event handlers
            this.options.ajaxForms.on(models.AjaxForm.SUCCESS, this.ajaxFormSuccess);
            this.options.model.on(models.Model.CREATED, this.reload);
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
                },
                onDblClick: function(node, event) {
                    _this.trigger(TreeView.ITEM_DOUBLE_CLICKED, node);
                },
                initAjax: {
                    url: _this.url
                },
                persist: true
            });
        },

        /*
         Reload the tree view in case new items were added in an `AjaxForm` submit.
         Called when an `AjaxForm` submits successfully.
         */
        ajaxFormSuccess: function(ajaxForm) {
            util.log('tree view success')
            this.reload();
        },

        getActiveItem: function() {
            return this.tree.dynatree("getActiveNode");
        },

        reload: function() {
            this.tree.dynatree('getTree').reload();
        }
    }, {
        ITEM_ACTIVATED: 'gnome:treeItemActivated',
        ITEM_DOUBLE_CLICKED: 'gnome:treeItemDoubleClicked'
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
            this.url = this.options.url;

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
            this.timeEl = this.options.timeEl;
            this.mapView = this.options.mapView;
            this.model = this.options.model;

            // Controls whose state, either enabled or disabled, is related to whether
            // or not an animation is playing. The resize and full screen buttons
            // are intentionally excluded.
            this.controls = [
                this.backButtonEl, this.forwardButtonEl, this.playButtonEl,
                this.pauseButtonEl, this.moveButtonEl, this.zoomInButtonEl,
                this.zoomOutButtonEl
            ];

            this.status = MapControlView.STATUS_STOPPED;

            $(this.pauseButtonEl).hide();
            $(this.resizeButtonEl).hide();

            $(this.sliderEl).slider({
                start: this.sliderStarted,
                change: this.sliderChanged,
                slide: this.sliderMoved,
                disabled: true
            });

            if (this.model.expectedTimeSteps.length) {
                this.setTimeSteps(this.model.expectedTimeSteps);
                this.enableControls();
            }

            this.setupClickEvents();

            this.model.on(models.Model.RUN_BEGAN, this.runBegan);
            this.model.on(models.Model.RUN_ERROR, this.modelRunError);
            this.model.on(models.Model.RUN_FINISHED, this.modelRunFinished);
            this.model.on(models.Model.CREATED, this.modelCreated);

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
                [this.pauseButtonEl, MapControlView.PAUSE_BUTTON_CLICKED]
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
            var timestamp = this.model.getTimestampForExpectedStep(ui.value);

            if (timestamp) {
                this.setTime(timestamp);
            } else {
                util.log('Slider changed to invalid time step: ' + ui.value);
                return false;
            }

            this.trigger(MapControlView.SLIDER_MOVED, ui.value);
        },

        runBegan: function() {
            if (this.model.dirty) {
                // TODO: Is this really what we want to do here?
                this.reset();
            }

            this.setTimeSteps(this.model.expectedTimeSteps);
        },

        mapViewFrameChanged: function() {
            var timeStep = this.model.getCurrentTimeStep();
            this.setTimeStep(timeStep.id);
            this.setTime(timeStep.get('timestamp'));
        },

        stop: function() {
            this.setStopped();
            this.enableControls();
        },

        modelRunError: function() {
            this.stop();
        },

        modelRunFinished: function() {
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

            $(this.newItemEl).click(this.newItemClicked);
            $(this.runItemEl).click(this.runItemClicked);
            $(this.runUntilItemEl).click(this.runUntilItemClicked);
        },

        newItemClicked: function(event) {
            $(this.modelDropdownEl).dropdown('toggle');
            this.trigger(MenuView.NEW_ITEM_CLICKED);
        },

        runItemClicked: function(event) {
            $(this.runDropdownEl).dropdown('toggle');
            this.trigger(MenuView.RUN_ITEM_CLICKED);
        },

        runUntilItemClicked: function(event) {
            $(this.runDropdownEl).dropdown('toggle');
            this.trigger(MenuView.RUN_UNTIL_ITEM_CLICKED);
        }
    }, {
        // Event constants
        NEW_ITEM_CLICKED: "menuView:newMenuItemClicked",
        RUN_ITEM_CLICKED: "menuView:runMenuItemClicked",
        RUN_UNTIL_ITEM_CLICKED: "menuView:runUntilMenuItemClicked"
    });

    return {
        MessageView: MessageView,
        MapView: MapView,
        TreeView: TreeView,
        TreeControlView: TreeControlView,
        MapControlView: MapControlView,
        MenuView: MenuView
    };

});