
define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'lib/jquery.imagesloaded.min',
    'lib/jquery.dynatree.min',
    'lib/moment'
], function($, _, Backbone, models) {

     /*
     `MessageView` is responsible for displaying messages sent back from the server
     during AJAX form submissions. These are non-form error conditions, usually,
     but can also be success messages.
     */
    var MessageView = Backbone.View.extend({
        initialize: function() {
            this.options.model.on(
                models.Model.MESSAGE_RECEIVED, this.displayMessage);
            this.options.ajaxForms.on(
                models.AjaxForm.MESSAGE_RECEIVED, this.displayMessage);
        },

        displayMessage: function(message) {
            if (!_.has(message, 'type') || !_.has(message, 'text')) {
                return false;
            }

            var alertDiv = $('div .alert-' + message.type);

            if (message.text && alertDiv) {
                alertDiv.find('span.message').text(message.text);
                alertDiv.removeClass('hidden');
            }

            return true;
        }
    });


    /*
     `MapView` represents the visual map and is reponsible for animating frames
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

            this.createPlaceholderCopy();
            this.makeImagesClickable();
            this.status = MapView.STOPPED;

            this.$map = $(this.mapEl);

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
            if (this.$map.find('img').length === 1) {
                this.$map.show();
            }

            var stepImage = this.getImageForTimeStep(stepNum);
            var otherImages = this.$map.find('img').not(stepImage).not('.background');

            // Hide all other images in the map div.
            otherImages.css('display', 'none');
            otherImages.removeClass(this.activeFrameClass);

            // The image isn't loaded.
            if (stepImage.length === 0) {
                window.alert("An animation error occurred. Please refresh.");
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
                src: timeStep.get('url')
            }).css('display', 'none');

            img.appendTo(map);

            $(img).imagesLoaded(function() {
                window.setTimeout(_this.showImageForTimeStep, 50, [timeStep.id]);
            });
        },

        addTimeStep: function(timeStep) {
            var imageExists = this.getImageForTimeStep(timeStep.id).length;

            // We must be playing a cached model run because the image already
            // exists. In all other cases the image should NOT exist.
            if (imageExists) {
                window.setTimeout(this.showImageForTimeStep, 50, [timeStep.id]);
                return;
            }

            this.addImageForTimeStep(timeStep);
        },

        // Clear out the current frames.
        clear: function() {
            $(this.mapEl).not('.background').empty();
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

            var img = $('<img>').attr({
                'class': 'background',
                src: url
            });

            img.appendTo(map);
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
        }
    }, {
        // Statuses
        PAUSED: 1,
        STOPPED: 2,
        PLAYING: 3,

        // Events
        DRAGGING_FINISHED: 'mapView:draggingFinished',
        REFRESH_FINISHED: 'mapView:refreshFinished',
        PLAYING_FINISHED: 'mapView:playingFinished',
        FRAME_CHANGED: 'mapView:frameChanged',
        MAP_WAS_CLICKED: 'mapView:mapWasClicked'
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
            log('tree view success')
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
        // Events
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
            }

            this.setupClickEvents();

            this.model = this.options.model;
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

            // TODO: This probably leaks memory, so do something else here, like
            // looking up the right `customEvent` for the element.
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
                console.log('Slider changed to invalid time step: ' + ui.value);
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

        // Events
        PLAY_BUTTON_CLICKED: "gnome:playButtonClicked",
        PAUSE_BUTTON_CLICKED: "gnome:pauseButtonClicked",
        BACK_BUTTON_CLICKED: "gnome:backButtonClicked",
        FORWARD_BUTTON_CLICKED: "gnome:forwardButtonClicked",
        ZOOM_IN_BUTTON_CLICKED: "gnome:zoomInButtonClicked",
        ZOOM_OUT_BUTTON_CLICKED: "gnome:zoomOutButtonClicked",
        MOVE_BUTTON_CLICKED: "gnome:moveButtonClicked",
        FULLSCREEN_BUTTON_CLICKED: "gnome:fullscreenButtonClicked",
        RESIZE_BUTTON_CLICKED: "gnome:resizeButtonClicked",
        SLIDER_CHANGED: "gnome:sliderChanged",
        SLIDER_MOVED: "gnome:sliderMoved",

        // Statuses
        STATUS_STOPPED: 0,
        STATUS_PLAYING: 1,
        STATUS_PAUSED: 2,
        STATUS_BACK: 3,
        STATUS_FORWARD: 4,
        STATUS_ZOOMING_IN: 5,
        STATUS_ZOOMING_OUT: 6
    });


    var ModalFormViewContainer = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.options.ajaxForms.on(models.AjaxForm.SUCCESS, this.refresh);
            this.formViews = {};
        },

        /*
         Refresh all forms from the server.

         Called when any `AjaxForm` on the page has a successful submit, in case
         additional forms should appear for new items.
         */
        refresh: function() {
            var _this = this;

            $.ajax({
                type: 'GET',
                url: this.options.url,
                tryCount: 0,
                retryLimit: 3,
                success: function(data) {
                    if (_.has(data, 'html')) {
                        _this.$el.html(data.html);
                        _this.trigger(ModalFormViewContainer.REFRESHED);
                    }
                },
                error: handleAjaxError
            });
        },

        formIdChanged: function(newId, oldId) {
            this.formViews[newId] = this.formViews[oldId];
            delete this.formViews[oldId];
        },

        add: function(opts, obj) {
            if (typeof opts === "number" || typeof opts === "string") {
                this.formViews[opts] = obj;
                return;
            }

            if (typeof opts === "object" &&
                    (_.has(opts, 'id') && opts.id)) {
                var view = new ModalFormView(opts);
                this.formViews[opts.id] = view;
                view.on(ModalFormView.ID_CHANGED, this.formIdChanged);
                return;
            }

            throw "Must pass ID and object or an options object.";
        },

        get: function(formId) {
            return this.formViews[formId];
        },

        deleteAll: function() {
            var _this = this;
             _.each(this.formViews, function(formView, key) {
                formView.remove();
                delete _this.formViews[key];
            });
        }
    }, {
        REFRESHED: 'modalFormViewContainer:refreshed'
    });


    /*
     `ModalFormView` is responsible for displaying HTML forms retrieved
     from and submitted to the server using an `AjaxForm object. `ModalFormView`
     displays an HTML form in a modal "window" over the page using the rendered HTML
     returned by the server. It listens to 'change' events on a bound `AjaxForm` and
     refreshes itself when that event fires.

     The view is designed to handle multi-step forms implemented purely in
     JavaScript (and HTML) using data- properties on DOM elements. The server
     returns one rendered form, but may split its content into several <div>s, each
     with a `data-step` property. If a form is structured this way, the user of the
     JavaScript application will see it as a multi-step form with "next," "back"
     and (at the end) a "save" or "create" button (the label is given by the server,
     but whatever it is, this is the button that signals final submission of the
     form).

     Submitting a form from `ModalFormView` serializes the form HTML and sends it to
     a bound `AjaxForm` model object, which then handles settings up the AJAX
     request for a POST.
     */
    var ModalFormView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.$container = $(this.options.formContainerEl);
            this.ajaxForm = this.options.ajaxForm;
            this.ajaxForm.on(models.AjaxForm.CHANGED, this.ajaxFormChanged);
            this.setupEventHandlers();
        },

        /*
         Bind listeners to the form container using `on()`, so they persist if
         the underlying form elements are replaced.
         */
        setupEventHandlers: function() {
            this.id = '#' + this.$el.attr('id');
            this.$container.on('click', this.id + ' .btn-primary', this.submit);
            this.$container.on('click', this.id + ' .btn-next', this.goToNextStep);
            this.$container.on('click', this.id + ' .btn-prev', this.goToPreviousStep);
        },

        ajaxFormChanged: function(ajaxForm) {
            var formHtml = ajaxForm.form_html;
            if (formHtml) {
                this.refresh(formHtml);
                this.show();
            }
        },

        /*
         Hide any other visible modals and show this one.
         */
        show: function() {
            $('div.modal').modal('hide');
            this.$el.modal();
        },

        /*
         Reload this form's HTML by initiating an AJAX request via this view's
         bound `AjaxForm`. If the request is successful, this `ModelFormView` will
         fire its `ajaxFormChanged` event handler.
         */
        reload: function(id) {
            this.ajaxForm.get({id: id});
        },

        getForm: function() {
            return this.$el.find('form');
        },

        getFirstTabWithError: function() {
            if (this.getForm().find('.nav-tabs').length === 0) {
                return null;
            }

            var errorDiv = $('div.control-group.error').first();
            var tabDiv = errorDiv.closest('.tab-pane');

            if (tabDiv.length) {
                return tabDiv.attr('id');
            }
        },

        getFirstStepWithError: function() {
            var step = 1;

            if (!this.getForm().hasClass('multistep')) {
                return null;
            }

            var errorDiv = $('div.control-group.error').first();
            var stepDiv = errorDiv.closest('div.step');

            if (stepDiv === false) {
                step = stepDiv.attr('data-step');
            }

            return step;
        },

        getStep: function(stepNum) {
            return this.getForm().find('div[data-step="' + stepNum  + '"]').length > 0;
        },

        previousStepExists: function(stepNum) {
           return this.getStep(stepNum - 1);
        },

        nextStepExists: function(stepNum) {
            stepNum = parseInt(stepNum, 10);
            return this.getStep(stepNum + 1);
        },

        goToStep: function(stepNum) {
            var $form = this.getForm();

            if (!$form.hasClass('multistep')) {
                return;
            }

            var stepDiv = $form.find('div.step[data-step="' + stepNum + '"]');

            if (stepDiv.length === 0) {
                return;
            }

            var otherStepDivs = $form.find('div.step');
            otherStepDivs.addClass('hidden');
            otherStepDivs.removeClass('active');
            stepDiv.removeClass('hidden');
            stepDiv.addClass('active');

            var prevButton = this.$container.find('.btn-prev');
            var nextButton = this.$container.find('.btn-next');
            var saveButton = this.$container.find('.btn-primary');

            if (this.previousStepExists(stepNum)) {
                prevButton.removeClass('hidden');
            } else {
                prevButton.addClass('hidden');
            }

            if (this.nextStepExists(stepNum)) {
                nextButton.removeClass('hidden');
                saveButton.addClass('hidden');
                return;
            }

            nextButton.addClass('hidden');
            saveButton.removeClass('hidden');
        },

        goToNextStep: function() {
            var $form = this.getForm();

            if (!$form.hasClass('multistep')) {
                return;
            }

            var activeStepDiv = $form.find('div.step.active');
            var currentStep = parseInt(activeStepDiv.attr('data-step'), 10);
            this.goToStep(currentStep + 1);
        },

        goToPreviousStep: function(event) {
            var $form = this.getForm();

            if (!$form.hasClass('multistep')) {
                return;
            }

            var activeStep = $form.find('div.step.active');
            var currentStep = parseInt(activeStep.attr('data-step'), 10);
            this.goToStep(currentStep - 1);
        },

        submit: function(event) {
            event.preventDefault();
            var $form = this.getForm();
            this.ajaxForm.submit({
                data: $form.serialize(),
                url: $form.attr('action')
            });
            this.hide();
            return false;
        },

        /*
         Replace this form with the form in `html`, an HTML string rendered by the
         server. Recreate any jQuery UI datepickers on the form if necessary.
         If there is an error in the form, load the step with errors.
         */
        refresh: function(html) {
            var oldId = this.$el.attr('id');

            this.remove();

            var $html = $(html);
            $html.appendTo(this.$container);

            this.$el = $('#' + $html.attr('id'));

             // Setup datepickers
            _.each(this.$el.find('.date'), function(field) {
                $(field).datepicker({
                    changeMonth: true,
                    changeYear: true
                });
            });

            var stepWithError = this.getFirstStepWithError();
            if (stepWithError) {
                this.goToStep(stepWithError);
            }

            var tabWithError = this.getFirstTabWithError();
            if (tabWithError) {
                $('a[href="#' + tabWithError + '"]').tab('show');
            }

            this.setupEventHandlers();
            window.noaa.erd.util.fixModals();

            var newId = this.$el.attr('id');
            if (oldId !== newId) {
                this.trigger(ModalFormView.ID_CHANGED, newId, oldId);
            }
        },

        hide: function() {
            this.$el.modal('hide');
        },

        remove: function() {
            this.hide();
            this.$el.empty();
            this.$el.remove();
            this.$container.off('click', this.id + ' .btn-primary', this.submit);
            this.$container.off('click', this.id + ' .btn-next', this.goToNextStep);
            this.$container.off('click', this.id + ' .btn-prev', this.goToPreviousStep);
        }
    }, {
        ID_CHANGED: 'modalFormView:idChanged'
    });


    /*
     This is a non-AJAX-enabled modal form object to support the "add mover" form,
     which asks the user to choose a type of mover to add. We then use the selection
     to disply another, more-specific form.
     */
    var AddMoverFormView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.$container = $(this.options.formContainerEl);

            // Bind listeners to the container, using `on()`, so they persist.
            this.id = '#' + this.$el.attr('id');
            this.$container.on('click', this.id + ' .btn-primary', this.submit);
        },

        getForm: function() {
            return this.$el.find('form');
        },

        show: function() {
            this.$el.modal();
        },

        hide: function() {
            this.$el.modal('hide');
        },

        submit: function(event) {
            event.preventDefault();
            var $form = this.getForm();
            var moverType = $form.find('select[name="mover_type"]').val();

            if (moverType) {
                this.trigger(AddMoverFormView.MOVER_CHOSEN, moverType);
            }

            return false;
        }
    }, {
        // Events
        MOVER_CHOSEN: 'addMoverFormView:moverChosen'
    });


    /*
     `WindMoverFormView` handles the WindMover form.
     */
    var WindMoverFormView = ModalFormView.extend({
        initialize: function(options) {
            this.constructor.__super__.initialize.apply(this, arguments);
            var _this = this;

            this.$timesTable = this.$el.find('.time-list');

            this.$el.on('change', '.direction', function() {
                _this.toggleDegreesInput(this);
            });

            this.$el.on('click', '.add-time', function(event) {
                event.preventDefault();
                _this.addTime();
            });

            this.$el.on('click', '.icon-edit', function(event) {
                event.preventDefault();
                _this.showEditForm(this);
            });

            // TODO: Move into function
            this.$el.on('click', '.cancel', function(event) {
                event.preventDefault();
                var form = $(this).closest('.time-form');
                form.addClass('hidden');
                _this.clearInputs(form);
                form.detach().appendTo('.times-list');
                $('.add-time-form').find('.time-form').removeClass('hidden');
            });

            // TODO: Move into function
            this.$el.on('click', '.save', function(event) {
                event.preventDefault();
                var $form = $(this).closest('.time-form');
                $form.addClass('hidden');
                // Delete the "original" form that we're replacing.
                $form.data('form-original').detach().empty().remove();
                $form.detach().appendTo('.times-list');
                $('.add-time-form').find('.time-form').removeClass('hidden');
                _this.$timesTable.append($form);
                _this.renderTimeTable();
            });

            this.$el.on('click', '.icon-trash', function(event) {
                event.preventDefault();
                var $form = $(this).closest('tr').data('data-form');
                $form.detach().empty().remove();
                _this.renderTimeTable();
            });

            _this.renderTimeTable();
        },

        showEditForm: function(editIcon) {
            var $form = $(editIcon).closest('tr').data('data-form');
            var addFormContainer = $('.add-time-form');
            var addTimeForm = addFormContainer.find('.time-form');
            addTimeForm.addClass('hidden');
            var $formCopy = $form.clone().appendTo(addFormContainer);
            $formCopy.data('form-original', $form);
            $formCopy.removeClass('hidden');
        },

        toggleDegreesInput: function(directionInput) {
            var $dirInput = $(directionInput);
            var selected_direction = $dirInput.val();
            var $formDiv = $dirInput.closest('.time-form');
            var $degreesControl = $formDiv.find(
                '.direction_degrees').closest('.control-group');

            if (selected_direction === 'Degrees true') {
                $degreesControl.removeClass('hidden');
            } else {
                $degreesControl.addClass('hidden');
            }
        },

        clearInputs: function(form) {
            $(form).find(':input').each(function() {
                $(this).val('').removeAttr('checked');
            });
        },

        /*
         Clone the add time form and add an item to the table of time series.
         */
        addTime: function() {
            var $addForm = this.$el.find('.add-time-form').find('.time-form');
            var $newForm = $addForm.clone(true).addClass('hidden');
            var formId = $addForm.find(':input')[0].id;
            var formNum = parseInt(formId.replace(/.*-(\d{1,4})-.*/m, '$1')) + 1;

            // There are no edit forms, so this is the first time series.
            if (!formNum) {
                formNum = 0;
            }

            // Select all of the options selected on the original form.
            _.each($addForm.find('select option:selected'), function(opt) {
                var $opt = $(opt);
                var name = $opt.closest('select').attr('name');
                var $newOpt = $newForm.find(
                    'select[name="' + name + '"] option[value="' + $opt.val() + '"]');
                $newOpt.attr('selected', true);
            });

            // Increment the IDs of the add form elements -- it should always be
            // the last form in the list of edit forms.
            $addForm.find(':input').each(function() {
                var id = $(this).attr('id');
                if (id) {
                    id = id.replace('-' + (formNum - 1) + '-', '-' + formNum + '-');
                    $(this).attr({'name': id, 'id': id});
                }
            });

            $newForm.find('.add-time-buttons').addClass('hidden');
            $newForm.find('.edit-time-buttons').removeClass('hidden');

            this.$timesTable.after($newForm);
            this.renderTimeTable();


            var autoIncrementBy = $addForm.find('.auto_increment_by').val();

            // Increase the date and time on the Add form if 'auto increase by'
            // value was provided.
            if (autoIncrementBy) {
                var $date = $addForm.find('.date');
                var $hour = $addForm.find('.hour');
                var $minute = $addForm.find('.minute');
                var time = $hour.val()  + ':' + $minute.val();

                // TODO: Handle a date-parsing error here.
                var dateTime = moment($date.val() + ' ' + time);
                dateTime.add('hours', autoIncrementBy);

                $date.val(dateTime.format("MM/DD/YYYY"));
                $hour.val(dateTime.hours());
                $minute.val(dateTime.minutes());
            }
        },

        renderTimeTable: function() {
            var _this = this;
            var $forms = this.$el.find('.edit-time-forms .time-form');
            var rows = [];

            this.$timesTable.find('tr').not('.table-header').remove();

            _.each($forms, function(form) {
                var $form = $(form);
                var tmpl = _.template($("#time-series-row").html());
                var speedType = $form.find('.speed_type option:selected').val();
                var direction = $form.find('.direction').val();

                if (direction === 'Degrees true') {
                    direction = $form.find('.direction_degrees').val() + ' &deg;';
                }

                var dateTime = moment(
                    $form.find('.date').val() + ' ' +
                    $form.find('.hour').val() + ':' +
                    $form.find('.minute').val());

                rows.push($(tmpl({
                    date: dateTime.format('MM/DD/YYYY'),
                    time: dateTime.format('HH:mm'),
                    direction: direction,
                    speed: $form.find('.speed').val() + ' ' + speedType
                })).data('data-form', $form));
            });

            // Sort table by date and time of each item.
            rows = _.sortBy(rows, function($tr) {
                var date = $tr.find('.time-series-date').text();
                var time = $tr.find(
                    '.time-series-time').text().replace(' ', '', 'g');
                return Date.parse(date + ' ' + time)
            });

            _.each(rows, function($row) {
                $row.appendTo(_this.$timesTable);
            });
        },

        /*
         Remove the "Add" form inputs and submit the form.
         */
        submit: function() {
            this.$el.find('.add-time-form .time-form').empty().remove();
            WindMoverFormView.__super__.submit.apply(this, arguments);
        }
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
        // Events
        NEW_ITEM_CLICKED: "gnome:newMenuItemClicked",
        RUN_ITEM_CLICKED: "gnome:runMenuItemClicked",
        RUN_UNTIL_ITEM_CLICKED: "gnome:runUntilMenuItemClicked"
    });

    return {
        MessageView: MessageView,
        MapView: MapView,
        TreeView: TreeView,
        TreeControlView: TreeControlView,
        MapControlView: MapControlView,
        AddMoverFormView: AddMoverFormView,
        WindMoverFormView: WindMoverFormView,
        ModalFormView: ModalFormView,
        ModalFormViewContainer: ModalFormViewContainer,
        MenuView: MenuView
    };

});