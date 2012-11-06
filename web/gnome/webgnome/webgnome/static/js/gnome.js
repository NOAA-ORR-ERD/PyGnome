// gnome.js: The WebGNOME JavaScript application.
"use strict";


// Aliases.
var log = window.noaa.erd.util.log;
var handleAjaxError = window.noaa.erd.util.handleAjaxError;


/*
  Retrieve a message object from the object `data` if the `message` key
  exists, annotate the message object ith an `error` value set to true
  if the message is an error type, and return the message object.
 */
var parseMessage = function(data) {
    var message;

    if (data === null || data === undefined) {
        return false;
    }

    if (_.has(data, 'message')) {
        message = data.message;

        if (data.message.type === 'error') {
            message.error = true;
        }

        return message;
    }

    return false;
};


/*
 Return a UTC date string for `timestamp`, which should be in a format
 acceptable to `Date.parse`.
 */
var getUTCStringForTimestamp = function(timestamp) {
    var date = new Date(Date.parse(timestamp));
    if (date) {
        timestamp = date.toUTCString();
    }
    return timestamp;
};


/*
 `TimeStep` represents a single time step of the user's actively-running
 model on the server.
 */
var TimeStep = Backbone.Model.extend({
    get: function(attr) {
        var value = Backbone.Model.prototype.get.call(this, attr);

        if (attr === 'timestamp') {
            value = getUTCStringForTimestamp(value);
        }

        return value;
    }
});


/*
 `Model` is a collection of `TimeStep` objects representing a run of
 the user's active model.
 */
var Model = Backbone.Collection.extend({
    model: TimeStep,

    initialize: function(timeSteps, opts) {
        _.bindAll(this);
        this.url = opts.url;
        this.currentTimeStep = opts.currentTimeStep || 0;
        this.nextTimeStep = this.currentTimeStep ? this.currentTimeStep + 1 : 0;
        // An array of timestamps, one for each step we expect the server to
        // make, passed back when we initiate a model run.
        this.expectedTimeSteps = opts.expectedTimeSteps || [];
        // Optionally specify the zoom level.
        this.zoomLevel = opts.zoomLevel === undefined ? 4 : opts.zoomLevel;
        // If true, `Model` will request a new set of time steps from the server
        // on the next run. Assume we want to do this by default (i.e., at
        // construction time) if there are no time steps.
        this.dirty = timeSteps.length === 0;

        // When initializing the model at the last time step of a generated
        // series, rewind to the beginning so the user can play the series
        // again.
        if (this.currentTimeStep === timeSteps.length -1) {
            this.rewind();
        }
    },

    hasData: function() {
        return this.expectedTimeSteps.length > 0;
    },

    /*
     Return true if the model has time step data for the step numbered
     `stepNum`.
     */
    hasCachedTimeStep: function(stepNum) {
        return this.get(stepNum) !== undefined;
    },

    /*
     Return true if the server gave us a time step for step number `stepNum`.
     */
    serverHasTimeStep: function(stepNum) {
        return this.expectedTimeSteps[stepNum] !== undefined;
    },

    /*
     Return the timestamp the server returned for the expected step `stepNum`.
     Unlike `this.getTimeStep()`, this function may be called for steps that
     the model has not yet received from the server.
     */
    getTimestampForExpectedStep: function(stepNum) {
        var timestamp;

        if (this.serverHasTimeStep(stepNum)) {
            timestamp = getUTCStringForTimestamp(this.expectedTimeSteps[stepNum]);
        }

        return timestamp;
    },

    /*
     Handle a successful request to the server to start the model run.
     Events:

     - Triggers:
        - `Model.MESSAGE_RECEIVED` if the server sent a message.
        - `Model.RUN_BEGAN` unless we received an error message.
     */
    runSuccess: function(data) {
        var message = parseMessage(data);

        if (message) {
            this.trigger(Model.MESSAGE_RECEIVED, message);

            if (message.error) {
                this.trigger(Model.RUN_ERROR);
                return false;
            }
        }

        this.dirty = false;
        this.expectedTimeSteps = data.expected_time_steps;
        this.trigger(Model.RUN_BEGAN, data);
        this.getNextTimeStep();
        return true;
    },

    /*
     Helper that performs an AJAX request to start ("run") the model.

     Receives back an array of timestamps, one for each step the server
     expects to generate on subsequent requests.
     */
    doRun: function(opts) {
        var isInvalid = function(obj) {
            return obj === undefined || obj === null || typeof(obj) !== "object";
        };

        // Abort if we were asked to zoom without a valid `opts.rect` or
        // `opts.point`.
        if (opts.zoomLevel !== this.zoomLevel &&
            isInvalid(opts.rect) && isInvalid(opts.point)) {
            window.alert("Invalid zoom level. Please try again.");
            return;
        }

        this.expectedTimeSteps = [];

        $.ajax({
            type: 'POST',
            url: this.url + '/run',
            data: opts,
            tryCount: 0,
            retryLimit: 3,
            success: this.runSuccess,
            error: handleAjaxError
        });
    },

    /*
     Run the model.

     If the model is dirty, make an AJAX request to the server to initiate a
     model run. Otherwise request the next time step.

     Options:
     - `zoomLevel`: the user's chosen zoom level
     - `zoomDirection`: if the user is zooming, `Model.ZOOM_IN`,
         `Model.ZOOM_OUT`, otherwise `Model.ZOOM_NONE` (the default)
     - `runUntilTimeStep`: the time step to stop running. This value is
         passed to the server-side model and running will stop after the
         client requests the step with this number.
     */
    run: function(opts) {
        var options = $.extend({}, {
            zoomLevel: this.zoomLevel,
            zoomDirection: Model.ZOOM_NONE,
            runUntilTimeStep: this.runUntilTimeStep
        }, opts);

        var needToGetRunUntilStep = false;

        if (options.runUntilTimeStep) {
            this.runUntilTimeStep = options.runUntilTimeStep;
            needToGetRunUntilStep = options.runUntilTimeStep &&
                !this.hasCachedTimeStep(options.runUntilTimeStep);
        }

        if (this.dirty || needToGetRunUntilStep) {
            this.doRun(options);
            return;
        }

        this.getNextTimeStep();
    },

    /*
     Return the `TimeStep` object whose ID matches `self.currentTimeStep`.
     */
    getCurrentTimeStep: function() {
        return this.get(this.currentTimeStep);
    },

    /*
     Set the current time step to `newStepNum`.
     */
    addTimeStep: function(timeStepJson) {
        var timeStep = new TimeStep(timeStepJson);
        this.add(timeStep);
        this.setCurrentTimeStep(timeStep.id);
    },

    /*
     Set the current time step to `stepNum`.

     Triggers:
     - `Model.NEXT_TIME_STEP_READY` with the time step object for the new step.
     - `Model.RUN_FINISHED` if this was the step chosen to run until
     */
    setCurrentTimeStep: function(stepNum) {
        this.currentTimeStep = stepNum;
        this.nextTimeStep = stepNum + 1;

         if (this.runUntilTimeStep &&
                this.currentTimeStep === this.runUntilTimeStep) {
             this.trigger(Model.RUN_FINISHED);
             this.runUntilTimeStep = null;
             return;
         }

         this.trigger(Model.NEXT_TIME_STEP_READY, this.getCurrentTimeStep());
    },

    /*
     Makes a request to the server for the next time step.

     Triggers:
     - `Model.RUN_FINISHED` if the server has no more time steps to run.
     */
    getNextTimeStep: function() {
        if (!this.serverHasTimeStep(this.nextTimeStep)) {
            this.rewind();
            this.trigger(Model.RUN_FINISHED);
            return;
        }

        // The time step has already been generated and we have it.
        if (this.hasCachedTimeStep(this.nextTimeStep)) {
            this.setCurrentTimeStep(this.nextTimeStep);
            return;
        }

        // Request the next step from the server.
        $.ajax({
            // Block until finished, so this and subsequent requests come
            // back in order.
            async: false,
            type: "GET",
            url: this.url + '/next_step',
            success: this.timeStepRequestSuccess,
            error: handleAjaxError
        });
    },

    timeStepRequestSuccess: function(data) {
        var message = parseMessage(data);

        if (message) {
            this.trigger(Model.MESSAGE_RECEIVED, message);

            if (message.error) {
                this.trigger(Model.RUN_ERROR);
                return;
            }
        }

        if (!data.time_step) {
            this.trigger(Model.RUN_ERROR);
            return;
        }

        this.addTimeStep(data.time_step);
   },

    /*
     Zoom the map from `point` in direction `direction`.

     Options:
     - `point`: an x, y coordinate, where the user clicked the map
     - `direction`: either `Model.ZOOM_IN` or `Model.ZOOM_OUT`
     */
    zoomFromPoint: function(point, direction) {
        this.dirty = true;
        this.run({point: point, zoom: direction});
    },

    /*
     Zoom the map from a rectangle `rect` in direction `direction`.

     Options:
     - `rect`: a rectangle consisting of two (x, y) coordinates that the
     user selected for the zoom operation. TODO: This should be
     constrained to the aspect ratio of the background image.
     - `direction`: either `Model.ZOOM_IN` or `Model.ZOOM_OUT`
     */
    zoomFromRect: function(rect, direction) {
        this.dirty = true;
        this.run({rect: rect, zoom: direction});
    },

    /*
     Set the current time step to 0.
     */
    rewind: function() {
        this.currentTimeStep = 0;
        this.nextTimeStep = 0;
    },

    /*
     Clear all time step data. Used when creating a new server-side model.
     */
    clearData: function() {
        this.rewind();
        this.timeSteps = [];
        this.expectedTimeSteps = [];
    },

    /*
     Request a new model. This destroys the current model.
     */
    create: function() {
        $.ajax({
            url: this.url + "/create",
            data: "confirm_new=1",
            type: "POST",
            tryCount: 0,
            retryLimit: 3,
            success: this.createSuccess,
            error: handleAjaxError
        });
    },

     /*
     Handle a successful request to the server to create a new model.
     */
    createSuccess: function(data) {
        var message = parseMessage(data);

        if (message) {
            this.trigger(Model.MESSAGE_RECEIVED, message);

            if (message.error) {
                // TODO: Separate error event?
                this.trigger(Model.RUN_ERROR);
                return;
            }
        }

        this.clearData();
        this.dirty = true;
        this.trigger(Model.CREATED);
    }
}, {
    // Class constants
    ZOOM_IN: 'zoom_in',
    ZOOM_OUT: 'zoom_out',
    ZOOM_NONE: 'zoom_none',

    // Class events
    CREATED: 'model:Created',
    RUN_BEGAN: 'model:modelRunBegan',
    RUN_FINISHED: 'model:modelRunFinished',
    RUN_ERROR: 'model:runError',
    NEXT_TIME_STEP_READY: 'model:nextTimeStepReady',
    MESSAGE_RECEIVED: 'model:messageReceived'
});


/*
 `AjaxForm` is a helper object that handles requesting rendered form HTML from
 the server and posting submitted forms. Form HTML, including error output, is
 rendered on the server. By convention, if a form submission returns `form_html`
 then the form contains errors and should be displayed again. Otherwise, we
 assume that submission succeeded.

 This object handles the GET and POST requests made when a user clicks on a
 control, typically using one of the control views (e.g., `TreeControlView`),
 that displays a form, or when the user submits a form. The form HTML is
 displayed in a modal view using `ModalFormView`.
 */
var AjaxForm = function(opts) {
    _.bindAll(this);
    this.url = opts.url;
    this.collection = opts.collection;

    // Mix Backbone.js event methods into `AjaxForm`.
    _.extend(this, Backbone.Events);
};

// Events
AjaxForm.MESSAGE_RECEIVED = 'ajaxForm:messageReceived';
AjaxForm.CHANGED = 'ajaxForm:changed';

AjaxForm.prototype = {
    /*
     Refresh this form from the server's JSON response.
     */
    parse: function(response) {
        var message = parseMessage(response);
        if (message) {
            this.trigger(AjaxForm.MESSAGE_RECEIVED, message);
        }

        if (_.has(response, 'form_html')) {
            this.form_html = response.form_html;
            this.trigger(AjaxForm.CHANGED, this);
            this.collection.trigger(AjaxForm.CHANGED, this);
        }
    },

    /*
     Make an AJAX request for this `AjaxForm`, merging `opts` into the options
     object passed to $.ajax. By default, this method uses a GET operation.
     */
    makeRequest: function(opts) {
        var options = $.extend({}, opts || {}, {
            url: this.url,
            tryCount: 0,
            retryLimit: 3,
            success: this.parse,
            error: handleAjaxError
        });

        if (options.id) {
            options.url = options.url + '/' + options.id;
        }

        $.ajax(options);
    },

    /*
     Get the HTML for this form.
     */
    get: function(opts) {
        var options = $.extend({}, opts || {}, {
            type: 'GET'
        });
        this.makeRequest(options);
    },

    /*
     Submit using `opts` and refresh this `AjaxForm` from JSON in the response.
     The assumption here is that `data` and `url` have been provided in `opts`
     and we're just passing them along to the `makeRequest()` method.
     */
    submit: function(opts) {
         var options = $.extend({}, opts, {
            type: 'POST'
        });

        this.makeRequest(options);
    }
};


/*
 `MessageView` is responsible for displaying messages sent back from the server
 during AJAX form submissions. These are non-form error conditions, usually,
 but can also be success messages.
 */
var MessageView = Backbone.View.extend({
    initialize: function() {
        this.options.model.on(
            Model.MESSAGE_RECEIVED, this.displayMessage);
        this.options.ajaxForms.on(
            AjaxForm.MESSAGE_RECEIVED, this.displayMessage);
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

        this.createPlaceholderCopy();
        this.makeImagesClickable();
        this.status = MapView.STOPPED;

        this.model = this.options.model;
        this.model.on(Model.NEXT_TIME_STEP_READY, this.nextTimeStepReady);
        this.model.on(Model.RUN_ERROR, this.modelRunError);
        this.model.on(Model.RUN_FINISHED, this.modelRunFinished);
        this.model.on(Model.CREATED, this.modelCreated);
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
        var map = $(this.mapEl);

        // Show the map div if this is the first image of the run.
        if (map.find('img').length === 1) {
            map.show();
        }

        var stepImage = this.getImageForTimeStep(stepNum);
        var otherImages = $(this.mapEl).find('img').not(stepImage);

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
            setTimeout(_this.showImageForTimeStep, 150, [timeStep.id]);
        });
    },

    addTimeStep: function(timeStep) {
        if (timeStep.id === 0 && this.placeholderCopy) {
            this.removePlaceholderCopy();
        }

        var imageExists = this.getImageForTimeStep(timeStep.id).length;

        // We must be playing a cached model run because the image already
        // exists. In all other cases the image should NOT exist.
        if (imageExists) {
            setTimeout(this.showImageForTimeStep, 150, [timeStep.id]);
            return;
        }

        this.addImageForTimeStep(timeStep);
    },

    // Clear out the current frames.
    clear: function() {
        $(this.mapEl).empty();
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
        this.options.ajaxForms.on(AjaxForm.CHANGED, this.ajaxFormChanged);
        this.options.model.on(Model.CREATED, this.reload);
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
     An event handler called when an `AjaxForm` in `this.forms` changes.

     If a form was submitted successfully, we need to reload the tree view in
     case new items were added.
     */
    ajaxFormChanged: function(ajaxForm) {
        var formHtml = ajaxForm.form_html;

        // This field will be null on a successful submit.
        if (!formHtml) {
            this.reload();
        }
    },

    getActiveItem: function() {
        return this.tree.dynatree("getActiveNode");
    },

    hasItem: function(data) {
        return this.tree.dynatree('getTree').selectKey(data.id) !== null;
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
        this.model.on(Model.RUN_BEGAN, this.runBegan);
        this.model.on(Model.RUN_ERROR, this.modelRunError);
        this.model.on(Model.RUN_FINISHED, this.modelRunFinished);
        this.model.on(Model.CREATED, this.modelCreated);

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
            log('Slider changed to invalid time step: ' + ui.value);
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


/*
 `ModalFormView` is responsible for displaying HTML forms retrieved
 from and submitted to the server using an `AjaxForm object. `ModalFormView`
 displays an HTML form in a modal "window" over the page using the rendered HTML
 returned by the server. It listens to 'change'events on a bound `AjaxForm` and
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
        this.ajaxForm.on(AjaxForm.CHANGED, this.ajaxFormChanged);

        // Bind listeners to the container, using `on()`, so they persist if
        // the underlying form elements are replaced.
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
        this.clear();
        this.$el = $(html);
        this.$el.prependTo(this.$container);

        this.$el.find('.date').datepicker({
            changeMonth: true,
            changeYear: true
        });

        var stepWithError = this.getFirstStepWithError();
        if (stepWithError) {
            this.goToStep(stepWithError);
        }
    },

    hide: function() {
        this.$el.modal('hide');
    },

    clear: function() {
        this.hide();
        $(this.id).empty().remove();
    }
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


/*
 `AppView` acts as a controller, listening to delegate views and models for
 events and coordinating any necessary changes.

 As a design principle, `AppView` should only handle events triggered by models
 and views that *require* coordination. Otherwise, views should listen directly
 to a specific model (or another view) and handle updating themselves without
 assistance from `AppView`. This convention is in-progress and could be better
 enforced.
 */
var AppView = Backbone.View.extend({
    initialize: function() {
        var _this = this;
        _.bindAll(this);

        this.apiRoot = "/model";

        // Initialize the model with any previously-generated time step data the
        // server had available.
        this.model = new Model(this.options.generatedTimeSteps, {
            url: this.apiRoot,
            expectedTimeSteps: this.options.expectedTimeSteps,
            currentTimeStep: this.options.currentTimeStep
        });

        this.addMoverFormView = new AddMoverFormView({
            el: $('#' + this.options.addMoverFormId),
            formContainerEl: this.options.formContainerEl
        });

        // All of the `ModelFormView` instances, keyed to form ID.
        this.formViews = {};
        this.formViews[this.options.addMoverFormId] = this.addMoverFormView;

        // An object holding all of our `AjaxForm` instances, keyed to the name
        // of the form as passed by the server in `this.options.formUrls`.
        this.forms = {};
        _.extend(this.forms, Backbone.Events);

        // Create an `AjaxForm` and bind it to a `ModalFormView` for each modal
        // form on the page, other than the Add Mover form, which we handled.
        _.each($('div.modal'), function(modalDiv) {
            var $div = $(modalDiv);
            var $form = $div.find('form');
            var modalFormId = $div.attr('id');

            if (modalFormId === _this.options.addMoverFormId) {
                return;
            }

            _this.forms[modalFormId] = new AjaxForm({
                url: $form.attr('action'),
                collection: _this.forms
            });

            _this.formViews[modalFormId] = new ModalFormView({
                ajaxForm: _this.forms[modalFormId],
                el: $('#' + modalFormId),
                formContainerEl: _this.options.formContainerEl
            });
        });

        this.menuView = new MenuView({
            modelDropDownEl: "#file-drop",
            runDropdownEl: "#run-drop",
            helpDropdownEl: "#help-drop",
            newItemEl: "#menu-new",
            runItemEl: "#menu-run",
            stepItemEl: "#menu-step",
            runUntilItemEl: "#menu-run-until"
        });

        this.sidebarEl = this.options.sidebarEl;

        this.treeView = new TreeView({
            treeEl: "#tree",
            url: "/tree",
            ajaxForms: this.forms,
            model: this.model
        });

        this.treeControlView = new TreeControlView({
            addButtonEl: "#add-button",
            removeButtonEl: "#remove-button",
            settingsButtonEl: "#settings-button",
            treeView: this.treeView
        });

        this.mapView = new MapView({
            mapEl: this.options.mapEl,
            placeholderEl: this.options.mapPlaceholderEl,
            frameClass: 'frame',
            activeFrameClass: 'active',
            model: this.model
        });

        this.mapControlView = new MapControlView({
            sliderEl: "#slider",
            playButtonEl: "#play-button",
            pauseButtonEl: "#pause-button",
            backButtonEl: "#back-button",
            forwardButtonEl: "#forward-button",
            zoomInButtonEl: "#zoom-in-button",
            zoomOutButtonEl: "#zoom-out-button",
            moveButtonEl: "#move-button",
            fullscreenButtonEl: "#fullscreen-button",
            resizeButtonEl: "#resize-button",
            timeEl: "#time",
            url: this.apiRoot + '/time_steps',
            model: this.model,
            mapView: this.mapView
        });

        this.messageView = new MessageView({
            model: this.model,
            ajaxForms: this.forms
        });

        this.setupEventHandlers();
    },

    setupEventHandlers: function() {
        this.model.on(Model.RUN_ERROR, this.modelRunError);

        this.treeView.on(TreeView.ITEM_DOUBLE_CLICKED, this.treeItemDoubleClicked);

        this.treeControlView.on(TreeControlView.ADD_BUTTON_CLICKED, this.addButtonClicked);
        this.treeControlView.on(TreeControlView.REMOVE_BUTTON_CLICKED, this.removeButtonClicked);
        this.treeControlView.on(TreeControlView.SETTINGS_BUTTON_CLICKED, this.settingsButtonClicked);

        this.mapControlView.on(MapControlView.PLAY_BUTTON_CLICKED, this.playButtonClicked);
        this.mapControlView.on(MapControlView.PAUSE_BUTTON_CLICKED, this.pause);
        this.mapControlView.on(MapControlView.ZOOM_IN_BUTTON_CLICKED, this.enableZoomIn);
        this.mapControlView.on(MapControlView.ZOOM_OUT_BUTTON_CLICKED, this.enableZoomOut);
        this.mapControlView.on(MapControlView.SLIDER_CHANGED, this.sliderChanged);
        this.mapControlView.on(MapControlView.SLIDER_MOVED, this.sliderMoved);
        this.mapControlView.on(MapControlView.BACK_BUTTON_CLICKED, this.jumpToFirstFrame);
        this.mapControlView.on(MapControlView.FORWARD_BUTTON_CLICKED, this.jumpToLastFrame);
        this.mapControlView.on(MapControlView.FULLSCREEN_BUTTON_CLICKED, this.useFullscreen);
        this.mapControlView.on(MapControlView.RESIZE_BUTTON_CLICKED, this.disableFullscreen);

        this.mapView.on(MapView.PLAYING_FINISHED, this.stopAnimation);
        this.mapView.on(MapView.DRAGGING_FINISHED, this.zoomIn);
        this.mapView.on(MapView.FRAME_CHANGED, this.frameChanged);
        this.mapView.on(MapView.MAP_WAS_CLICKED, this.zoomOut);

        this.menuView.on(MenuView.NEW_ITEM_CLICKED, this.newMenuItemClicked);
        this.menuView.on(MenuView.RUN_ITEM_CLICKED, this.runMenuItemClicked);
        this.menuView.on(MenuView.RUN_UNTIL_ITEM_CLICKED, this.runUntilMenuItemClicked);

        this.addMoverFormView.on(AddMoverFormView.MOVER_CHOSEN, this.moverChosen);
    },

    modelRunError: function() {
        this.messageView.displayMessage({
            type: 'error',
            text: 'Model run failed.'
        });
    },

    isValidFormType: function(formType) {
        return _.contains(this.formTypes, formType);
    },

    runMenuItemClicked: function() {
        this.play({});
    },

    runUntilMenuItemClicked: function() {
        this.fetchForm({type: 'run_until'});
    },

    newMenuItemClicked: function() {
        if (!window.confirm("Reset model?")) {
            return;
        }

        this.model.create();
    },

    play: function(opts) {
        this.mapControlView.disableControls();
        this.mapControlView.enableControls([this.mapControlView.pauseButtonEl]);
        this.mapControlView.setPlaying();
        this.mapView.setPlaying();
        this.model.run(opts);
    },

    playButtonClicked: function() {
        this.play({});
    },

    enableZoomIn: function() {
        if (this.model.hasData() === false) {
            return;
        }

        this.mapControlView.setZoomingIn();
        this.mapView.makeActiveImageClickable();
        this.mapView.makeActiveImageSelectable();
        this.mapView.setZoomingInCursor();
    },

    enableZoomOut: function() {
        if (this.model.hasData() === false) {
            return;
        }

        this.mapControlView.setZoomingOut();
        this.mapView.makeActiveImageClickable();
        this.mapView.setZoomingOutCursor();
    },

    stopAnimation: function() {
        this.mapControlView.setStopped();
    },

    zoomIn: function(startPosition, endPosition) {
        this.mapView.setPaused();
        this.mapControlView.setPaused();
        this.model.rewind();

        if (endPosition) {
            var rect = {start: startPosition, end: endPosition};
            var isInsideMap = this.mapView.isRectInsideMap(rect);

            // If we are at zoom level 0 and there is no map portion outside of
            // the visible area, then adjust the coordinates of the selected
            // rectangle to the on-screen pixel bounds.
            if (!isInsideMap && this.model.zoomLevel === 0) {
                rect = this.mapView.getAdjustedRect(rect);
            }

            this.model.zoomFromRect(rect, Model.ZOOM_IN);
        } else {
            this.model.zoomFromPoint(startPosition, Model.ZOOM_IN);
        }

        this.mapView.setRegularCursor();
    },

    zoomOut: function(point) {
        this.model.rewind();
        this.mapView.setPaused();
        this.mapControlView.setPaused();
        this.model.zoomFromPoint(point, Model.ZOOM_OUT);
        this.mapView.setRegularCursor();
    },

    pause: function() {
        this.mapView.setPaused();
        this.mapControlView.setPaused();
        this.mapControlView.enableControls();
    },

    sliderChanged: function(newStepNum) {
        // We're advancing in an animation, so ignore the change. Otherwise
        // the user just dragged the slider to a new position.
        if (newStepNum === this.model.currentTimeStep) {
            return;
        }

        // If the model and map view have the time step, display it.
        if (this.model.hasCachedTimeStep(newStepNum) &&
                this.mapView.timeStepIsLoaded(newStepNum)) {
            this.model.setCurrentTimeStep(newStepNum);
            return;
        }

        // Otherwise, we need to run until the new time step.
        this.play({
            runUntilTimeStep: newStepNum
        });
    },

    frameChanged: function() {
        if (this.mapView.isPaused() || this.mapView.isStopped()) {
            return;
        }
        this.model.getNextTimeStep();
    },

    jumpToFirstFrame: function() {
        this.model.setCurrentTimeStep(0);
    },

    /*
     Jump to the last LOADED frame of the animation. This will stop at
     whatever frame was the last received from the server.

     TODO: This should probably do something fancier, like block and load
     all of the remaining frames if they don't exist, until the end.
     */
    jumpToLastFrame: function() {
        var lastFrame = this.model.length - 1;
        this.model.setCurrentTimeStep(lastFrame);
    },

    useFullscreen: function() {
        this.mapControlView.switchToFullscreen();
        $(this.sidebarEl).hide('slow');
    },

    disableFullscreen: function() {
        this.mapControlView.switchToNormalScreen();
        $(this.sidebarEl).show('slow');
    },

    showFormForNode: function(node) {
        var formView = this.formViews[node.data.form_id];
        log(formView, node.data)

        if (formView === undefined) {
            return;
        }

        if (node.data.id) {
            formView.reload(node.data.id);
        } else {
            formView.show();
        }
    },

    /*
     Show the `ModalFormView` for the active tree item.

     If showing an add form, display the `ModalFormView` for the active node.

     If showing an edit form, perform a `fetch` using the `AjaxForm` for the
     selected node first, which will trigger the bound `ModalFormView` to display.

     The distinction of "add" versus "edit" is made on whether or not the node
     has an `id` property with a non-null value.
     */
    showFormForActiveTreeItem: function() {
        var node = this.treeView.getActiveItem();
        this.showFormForNode(node);
    },

    addButtonClicked: function() {
        this.showFormForActiveTreeItem();
    },

    treeItemDoubleClicked: function(node) {
        this.showFormForNode(node);
    },

    settingsButtonClicked: function() {
        this.showFormForActiveTreeItem();
    },

    removeButtonClicked: function() {
        var node = this.treeView.getActiveItem();

        if (!node.data.form_id || !node.data.id) {
            return;
        }

        var type = node.data.form_id.replace('_', ' ');

        if (window.confirm('Remove ' + type + '?') === false) {
            return;
        }

        this.ajaxForm.submit({
            url: this.ajaxForm.get('url') + '/' + node.data.form_id + '/delete',
            data: "mover_id=" + node.data.id,
            error: function() {
                window.alert('Could not remove item.');
            }
        });
    },

    moverChosen: function(moverType) {
        var formView = this.formViews[moverType];
        log(this.formViews)

        if (formView === undefined) {
            return;
        }

        formView.show();
    }
});


window.noaa.erd.gnome.AppView = AppView;

