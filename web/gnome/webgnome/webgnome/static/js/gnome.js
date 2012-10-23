// gnome.js: The WebGNOME JavaScript application.

/*
 Generic AJAX error handler.
 Retry on error if the request specified tryCount.
 */
window.noaa.erd.gnome.handleAjaxError  = function(xhr, textStatus, errorThrown) {
    if (textStatus == 'timeout') {
        this.tryCount++;
        if (this.tryCount <= this.retryLimit) {
            // Retry count is below the limit, so try the request again.
            $.ajax(this);
            return;
        }
        return;
    }

    alert('Could not connect to server.');
    console.log(xhr, textStatus, errorThrown);
};


/*
 `TimeStep` represents a single time step of the user's actively-running
 model on the server.
 */
var TimeStep = Backbone.Model.extend({});


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
        // An array of timestamps, one for each step we expect the server to
        // make, passed back when we initiate a model run.
        this.expectedTimeSteps = opts.expectedTimeSteps || [];
        // When running the model, start from this time step.
        this.startFromTimeStep = opts.startFromTimeStep || 0;
        // Optionally specify the zoom level.
        this.zoomLevel = opts.zoomLevel === undefined ? 4 : opts.zoomLevel;
        // If true, `Model` will request a new set of time steps from the server
        // on the next run. Assume we want to do this by default (i.e., at
        // construction time) if there are no time steps.
        this.dirty = this.length === 0;
    },

    getTimestampForStep: function(stepNum) {
        var timestamp = this.expectedTimeSteps[stepNum];
        if (timestamp) {
            var date = new Date(Date.parse(timestamp));
            if (date) {
                timestamp = date.toUTCString();
            }
        }
        return timestamp;
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
     Handle a successful request to the server to start the model run.
     Events:
        - Triggers `Model.MESSAGE_RECEIVED` if the server sent a message.
        - Triggers `Model.RUN_BEGAN` unless we received an error message.
     */
    runBegan: function(data) {
        var message = this.parseMessage(data);

        if (message) {
            this.trigger(Model.MESSAGE_RECEIVED, message);

            if (message.error) {
                return false;
            }
        }

        this.dirty = false;
        this.isFirstStep = true;
        this.expectedTimeSteps = data.expected_time_steps; 
        this.trigger(Model.RUN_BEGAN, data);
        return true;
    },

    /*
     Run the model.

     Makes an AJAX request to the server to initiate a model run.
     Receives back an array of timestamps, one for each step the server
     expects to generate on subsequent requests.

     Options:
     - `zoomLevel`: the user's chosen zoom level
     - `zoomDirection`: if the user is zooming, `Model.ZOOM_IN`,
     `Model.ZOOM_OUT`, otherwise `Model.ZOOM_NONE` (the default)
     - `startFromTimeStep`: the time step to start the model run at. This
     is used during cached runs, when the user chooses a time step,
     say with the slider, and runs the model, but the user has not
     changed any values (i.e., the model is not `dirty`) and the
     model already has time step data in its internal arra.
     - `runUntilTimeStep`: the time step to stop running. This value is
     passed to the server-side model and running will stop after the
     client requests the step with this number.

     Events:
     - Triggers `Model.RUN_BEGAN` after the run begins (i.e., the AJAX
     request was successful).
     */
    run: function(opts) {
        opts = $.extend({
            zoomLevel: this.zoomLevel,
            zoomDirection: Model.ZOOM_NONE,
            startFromTimeStep: null,
            runUntilTimeStep: null
        }, opts);

        this.startFromTimeStep = opts.startFromTimeStep;
        this.runUntilTimeStep = opts.runUntilTimeStep;

        if (this.dirty === false) {
            if ((this.startFromTimeStep &&
                this.hasCachedTimeStep(this.startFromTimeStep)) ||
                (this.runUntilTimeStep &&
                    this.hasCachedTimeStep(this.runUntilTimeStep))) {

                // We have the time steps needed and assume that any in-
                // between are also loaded.
                this.isFirstStep = true;
                $(this).trigger(Model.RUN_BEGAN);
                return;
            }
        }

        var isInvalid = function(obj) {
            return obj === undefined || obj === null || typeof(obj) != "object";
        };

        // Abort if we were asked to zoom without a valid `opts.rect` or
        // `opts.point`.
        if (opts.zoomLevel != this.zoomLevel &&
            isInvalid(opts.rect) && isInvalid(opts.point)) {
            alert("Invalid zoom level. Please try again.");
            return;
        }

        this.expectedTimeSteps = [];

        $.ajax({
            type: 'POST',
            url: this.url + '/run',
            data: opts,
            tryCount: 0,
            retryLimit: 3,
            success: this.runBegan,
            error: window.noaa.erd.gnome.handleAjaxError
        });
    },

    /*
     * Retrieve a message object from the object `data` if the `message` key
     * exists, annotate the message object ith an `error` value set to true
     * if the message is an error type, and return the message object.
     */
    parseMessage: function(data) {
        var message;

        if (data === null || data === undefined) {
            return false;
        }

        if (_.has(data, 'message')) {
            message = data.message;
            if (data.message.type == 'error') {
                message.error = true;
            }

            return message;
        }

        return false;
    },

    /*
     Set the current time step to `newStepNum`.
     Assumes that the internal timesteps array has the new time step object.

     Triggers:
     - `Model.NEXT_TIME_STEP_READY` with the time step object for the new
     step.
     */
    addTimeStep: function(timeStepJson) {
        console.log('wtf')
        var timeStep = new TimeStep(timeStepJson);
        this.add(timeStep);
        this.currentTimeStep = timeStep.get('step_number');
        console.log('add time step', this.models, this.currentTimeStep, this.get(this.currentTimeStep));
        this.trigger(Model.NEXT_TIME_STEP_READY, this.get(this.currentTimeStep));
    },

    /*
     Makes a request to the server for the next time step.

     Triggers:
     - `Model.RUN_FINISHED` if the server has no more time steps to run.
     */
    getNextTimeStep: function() {
        var currStepNum, nextStepNum;
        var _this = this;

        if (this.startFromTimeStep) {
            currStepNum = this.startFromTimeStep;
            this.startFromTimeStep = null;
        } else {
            currStepNum = this.currentTimeStep;
        }

        // If this is step 0 and we don't have it yet, then we're requesting step 0.
        nextStepNum = this.hasData() ? currStepNum + 1 : currStepNum;

        if (this.serverHasTimeStep(nextStepNum) === false) {
            $(this).trigger(Model.RUN_FINISHED);
            return;
        }

        // The time step has already been generated and we have it.
        if (this.hasCachedTimeStep(nextStepNum)) {
            this.setTimeStep(nextStepNum);
            return;
        }

        // Request the next step from the server.
        $.ajax({
            // Block until finished, so this and subsequent
            // requests come back in order.
            async: false,
            type: "GET",
            url: this.url + '/next_step',
            success: function(data) {
                _this.addTimeStep(data.time_step);
            },
            error: window.noaa.erd.gnome.handleAjaxError
        });
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
     Reset the current time step to 0.
     */
    resetCurrentTimeStep: function() {
        this.currentTimeStep = 0;
        this.startFromTimeStep = null;
    },

    /*
     Clear all time step data. Used when creating a new server-side model.
     */
    clearData: function() {
        this.resetCurrentTimeStep();
        this.reset([]);
        this.timeSteps = [];
        this.expectedTimeSteps = [];
        this.startFromTimeStep = 0;
    }
}, {
    // Class constants
    ZOOM_IN: 'zoom_in',
    ZOOM_OUT: 'zoom_out',
    ZOOM_NONE: 'zoom_none',

    // Class events
    MESSAGE_RECEIVED: 'model:messageReceived',
    RUN_BEGAN: 'model:modelRunBegan',
    RUN_FINISHED: 'model:modelRunFinished',
    NEXT_TIME_STEP_READY: 'model:nextTimeStepReady'
});


/*
 `MessageView` is responsible for displaying messages sent from the server.
 */
var MessageView = Backbone.View.extend({
    initialize: function(opts) {
        opts.model.on(Model.MESSAGE_RECEIVED, this.displayMessage);
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
    },
});


/*
 `MapView` represents the visual map and is reponsible for animating frames
 for each time step rendered by the server
 */
function MapView(opts) {
    this.mapEl = opts.mapEl;
    this.frameClass = opts.frameClass;
    this.activeFrameClass = opts.activeFrameClass;
    this.placeholderEl = opts.placeholderEl;
    this.currentTimeStep = 0;
}

MapView.PAUSED = 1;
MapView.STOPPED = 2;
MapView.PLAYING = 3;

// `MapView` events
MapView.DRAGGING_FINISHED = 'gnome:draggingFinished';
MapView.REFRESH_FINISHED = 'gnome:refreshFinished';
MapView.PLAYING_FINISHED = 'gnome:playingFinished';
MapView.FRAME_CHANGED = 'gnome:frameChanged';
MapView.MAP_WAS_CLICKED = 'gnome:mapWasClicked';

MapView.prototype = {
    initialize: function() {
        this.createPlaceholderCopy();
        this.makeImagesClickable();
        this.status = MapView.STOPPED;
        return this;
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
                $(_this).trigger(
                    MapView.MAP_WAS_CLICKED,
                    {x: event.pageX, y: event.pageY});
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
                    $(_this).trigger(
                        MapView.DRAGGING_FINISHED,
                        [_this.startPosition, {x: event.pageX, y: event.pageY}]);
                }
            }
        });
    },

    getActiveImage: function() {
        return $(this.mapEl + " > img.active");
    },

    getImageForTimeStep: function(stepNum) {
        return $('img[data-position="' + (stepNum) + '"]');
    },

    timeStepIsLoaded: function(stepNum) {
        var step = this.getImageForTimeStep(stepNum);
        return step && step.length;
    },

    // Set `stepNum` as the current time step and display an image if one
    // exists.
    setTimeStep: function(stepNum) {
        var nextImage = this.getImageForTimeStep(stepNum);
        var otherImages = $(this.mapEl).find('img').not(nextImage);

        // Hide all other images in the map div.
        otherImages.css('display', 'none');
        otherImages.removeClass(this.activeFrameClass);

        if (nextImage.length === 0) {
            alert("An animation error occurred. Please refresh.");
        }

        nextImage.addClass(this.activeFrameClass);
        nextImage.css('display', 'block');
        this.currentTimeStep = stepNum;
    },

    // Advance to time step `stepNum` and trigger FRAME_CHANGED to continue
    // forward animation.
    advanceTimeStep: function(stepNum) {
        var map = $(this.mapEl);
        var _this = this;

        // Show the map div if this is the first image of the run.
        if (map.find('img').length == 1) {
            map.show();
        }

        setTimeout(function() {
            if (_this.isPaused()) {
                return;
            }
            _this.setTimeStep(stepNum);
            $(_this).trigger(MapView.FRAME_CHANGED);
        }, 150);
    },

    addImageForTimeStep: function(timeStep) {
        var _this = this;
        var map = $(this.mapEl);

        var img = $('<img>').attr({
            'class': 'frame',
            'data-position': timeStep.step_number,
            src: timeStep.url
        }).css('display', 'none');

        img.appendTo(map);

        $(img).imagesLoaded(function() {
            _this.advanceTimeStep(timeStep.step_number);
        });
    },

    addTimeStep: function(timeStep) {
        var imageExists = this.getImageForTimeStep(timeStep.step_number).length;

        if (timeStep.step_number === 0 && this.placeholderCopy) {
            this.removePlaceholderCopy();
        }

        // We must be playing a cached model run because the image already
        // exists. In all other cases the image should NOT exist.
        if (imageExists) {
            this.advanceTimeStep(timeStep.step_number);
        } else {
            this.addImageForTimeStep(timeStep);
        }
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

    getFrameCount: function() {
        return $(this.mapEl).find('img').length - 1;
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
    }
};


function TreeView(opts) {
    this.treeEl = opts.treeEl;
    this.url = opts.url;
    return this;
}

TreeView.ITEM_ACTIVATED = 'gnome:treeItemActivated';
TreeView.ITEM_DOUBLE_CLICKED = 'gnome:treeItemDoubleClicked';

TreeView.prototype = {
    initialize: function() {
        var _this = this;

        this.tree = $(this.treeEl).dynatree({
            onActivate: function(node) {
                $(_this).trigger(TreeView.ITEM_ACTIVATED, node);
            },
            onPostInit: function(isReloading, isError) {
                // Fire events for a tree that was reloaded from cookies.
                // isReloading is true if status was read from existing cookies.
                // isError is only used in Ajax mode
                this.reactivate();
            },
            onDblClick: function(node, event) {
                $(_this).trigger(TreeView.ITEM_DOUBLE_CLICKED, node);
            },
            initAjax: {
                url: _this.url
            },
            persist: true
        });

        return this;
    },

    getActiveItem: function() {
        return this.tree.dynatree("getActiveNode");
    },

    hasItem: function(data) {
        return this.tree.dynatree('getTree').selectKey(data.id) !== null;
    },

    addItem: function(data, parent) {
        if (!_.has(data, 'id') || !_.has(data, 'type')) {
            alert('An error occurred. Try refreshing the page.');
            console.log(data);
        }

        var rootNode = this.tree.dynatree('getTree').selectKey(data.parent);
        rootNode.addChild({
            title: data.id,
            key: data.id,
            parent: parent,
            type: data.type
        });
    },

    reload: function() {
        this.tree.dynatree('getTree').reload();
    }
};


function TreeControlView(opts) {
    this.addButtonEl = opts.addButtonEl;
    this.removeButtonEl = opts.removeButtonEl;
    this.settingsButtonEl = opts.settingsButtonEl;
    this.url = opts.url;

    // Controls that require the user to select an item in the TreeView.
    this.itemControls = [this.removeButtonEl, this.settingsButtonEl];
}

TreeControlView.ADD_BUTTON_CLICKED = 'gnome:addItemButtonClicked';
TreeControlView.REMOVE_BUTTON_CLICKED = 'gnome:removeItemButtonClicked';
TreeControlView.SETTINGS_BUTTON_CLICKED = 'gnome:itemSettingsButtonClicked';

TreeControlView.prototype = {
    initialize: function() {
        var _this = this;
        this.disableControls();

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
                $(_this).trigger(customEvent);
                return true;
            });
        });
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
};


var MapControlView = Backbone.View.extend({
    initialize: function() {
        var _this = this;
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

        $(this.pauseButtonEl).click(this.pauseButtonClicked);

        var clickEvents = [
            [this.playButtonEl, MapControlView.PLAY_BUTTON_CLICKED],
            [this.backButtonEl, MapControlView.BACK_BUTTON_CLICKED],
            [this.forwardButtonEl, MapControlView.FORWARD_BUTTON_CLICKED],
            [this.zoomInButtonEl, MapControlView.ZOOM_IN_BUTTON_CLICKED],
            [this.zoomOutButtonEl, MapControlView.ZOOM_OUT_BUTTON_CLICKED],
            [this.moveButtonEl, MapControlView.MOVE_BUTTON_CLICKED],
            [this.fullscreenButtonEl, MapControlView.FULLSCREEN_BUTTON_CLICKED],
            [this.resizeButtonEl, MapControlView.RESIZE_BUTTON_CLICKED]
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
        
        this.model = this.options.model;
        this.model.on(Model.RUN_BEGAN, this.runBegan);

        return this;
    },

    sliderStarted: function(event, ui) {
        this.trigger(MapControlView.PAUSE_BUTTON_CLICKED);
    },

    sliderChanged: function(event, ui) {
        this.trigger(MapControlView.SLIDER_CHANGED, ui.value);
    },

    sliderMoved: function(event, ui) {
        this.trigger(MapControlView.SLIDER_MOVED, ui.value);
    },

    pauseButtonClicked: function(event) {
        if ($(this.pauseButtonEl).hasClass('disabled')) {
            return false;
        }
        if (this.status === MapControlView.STATUS_PLAYING) {
            this.trigger(MapControlView.PAUSE_BUTTON_CLICKED);
        }
        return true;
    },
    
    runBegan: function() {
         if (this.model.dirty === true) {
            this.reload();
        }

        this.setTimeSteps(this.model.expectedTimeSteps);
        this.enableControls([this.pauseButtonEl]);
    },

    setStopped: function() {
        this.status = MapControlView.STATUS_STOPPED;
        $(this.pauseButtonEl).hide();
        $(this.playButtonEl).show();
    },

    setPlaying: function() {
        this.status = MapControlView.STATUS_PLAYING;
        $(this.playButtonEl).hide();
        $(this.pauseButtonEl).show();
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
        } else {
            $(buttonEl).addClass('disabled');
        }
    },

    toggleControls: function(controls, toggle) {
        var _this = this;

        if (controls) {
            if (_.indexOf(controls, this.sliderEl)) {
                this.toggleSlider(MapControlView.ON);
            }
            _.each(_.without(controls, this.sliderEl), function(button) {
                _this.toggleControl(button, toggle);
            });
        } else {
            this.toggleSlider(toggle);
            _.each(this.controls, function(button) {
                _this.toggleControl(button, toggle);
            });
        }
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
        this.disableControls();
        this.setStopped();
        $(this.sliderEl).slider('values', null);
    },

    reload: function() {
        $.ajax({
            type: "GET",
            url: this.url,
            success: this.handleReloadSuccess,
            error: window.noaa.erd.gnome.handleAjaxError
        });
    },

    handleReloadSuccess: function(data) {
        $(this.containerEl).html(data.result);
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
   


function ModalFormView(opts) {
    _.bindAll(this);
    this.formContainerEl = opts.formContainerEl;
    this.urls = opts.urls;
}

ModalFormView.SAVE_BUTTON_CLICKED = 'gnome:formSaveButtonClicked';
ModalFormView.FORM_SUBMITTED = 'gnome:formSubmitted';

ModalFormView.prototype = {

    initialize: function() {
        var _this = this;

        $(this.formContainerEl).on('click', '.btn-primary', function(event) {
            $(_this).trigger(ModalFormView.SAVE_BUTTON_CLICKED);
        });

        $(this.formContainerEl).on('click', '.btn-next', this.goToNextStep);
        $(this.formContainerEl).on('click', '.btn-prev', this.goToPreviousStep);

        $(this.formContainerEl).on('submit', 'form', function(event) {
            event.preventDefault();
            $(_this).trigger(ModalFormView.FORM_SUBMITTED, [this]);
            return false;
        });
    },

    getFirstStepWithError: function() {
        var form = $(this.formContainerEl).find('form');
        var step = 1;

        if (!form.hasClass('multistep')) {
            return null;
        }

        var errorDiv = $('div.control-group.error').first();
        var stepDiv = errorDiv.closest('div.step');

        if (stepDiv) {
            step = stepDiv.attr('data-step');
        }

        return step;
    },

    previousStepExists: function(stepNum) {
        var form = $(this.formContainerEl).find('form');
        return form.find('div[data-step="' + (stepNum - 1) + '"]').length > 0;
    },

    nextStepExists: function(stepNum) {
        stepNum = parseInt(stepNum, 10);
        var form = $(this.formContainerEl).find('form');
        return form.find('div[data-step="' + (stepNum + 1) + '"]').length > 0;
    },

    goToStep: function(stepNum) {
        var form = $(this.formContainerEl).find('form');

        if (!form.hasClass('multistep')) {
            return;
        }

        var stepDiv = form.find('div.step[data-step="' + stepNum + '"]');

        if (stepDiv.length === 0) {
            return;
        }

        var otherSteps = form.find('div.step');
        otherSteps.addClass('hidden');
        otherSteps.removeClass('active');
        stepDiv.removeClass('hidden');
        stepDiv.addClass('active');

        var prevButton = $(this.formContainerEl).find('.btn-prev');
        var nextButton = $(this.formContainerEl).find('.btn-next');
        var saveButton = $(this.formContainerEl).find('.btn-primary');

        if (this.previousStepExists(stepNum)) {
            prevButton.removeClass('hidden');
        } else {
            prevButton.addClass('hidden');
        }

        if (this.nextStepExists(stepNum)) {
            nextButton.removeClass('hidden');
            saveButton.addClass('hidden');
        } else {
            nextButton.addClass('hidden');
            saveButton.removeClass('hidden');
        }
    },

    goToNextStep: function(event) {
        var form = $(this.formContainerEl).find('form');

        if (!form.hasClass('multistep')) {
            return;
        }

        var activeStep = form.find('div.step.active');
        var currentStep = parseInt(activeStep.attr('data-step'), 10);
        this.goToStep(currentStep + 1);
    },

    goToPreviousStep: function(event) {
        var form = $(this.formContainerEl).find('form');

        if (!form.hasClass('multistep')) {
            return;
        }

        var activeStep = form.find('div.step.active');
        var currentStep = parseInt(activeStep.attr('data-step'), 10);
        this.goToStep(currentStep - 1);
    },

    reloadForm: function(html) {
        this.clearForm();
        var container = $(this.formContainerEl);
        container.html(html);
        container.find('div.modal').modal();
        container.find('.date').datepicker({
            changeMonth: true,
            changeYear: true
        });

        var stepWithError = this.getFirstStepWithError();
        if (stepWithError) {
            this.goToStep(stepWithError);
        }
    },

    clearForm: function() {
        $(this.formContainerEl + ' div.modal').modal('hide');
        $(this.formContainerEl).empty();
    },

    submitForm: function() {
        $(this.formContainerEl + ' form').submit();
        this.clearForm();
    }
};


function MenuView(opts) {
    // Top-level drop-downs
    this.modelDropdownEl = opts.modelDropdownEl;
    this.runDropdownEl = opts.runDropdownEl;
    this.helpDropdownEl = opts.hepDropdownEl;

    this.newItemEl = opts.newItemEl;
    this.runItemEl = opts.runItemEl;
    this.stepItemEl = opts.stepItemEl;
    this.runUntilItemEl = opts.runUntilItemEl;
}

MenuView.NEW_ITEM_CLICKED = "gnome:newMenuItemClicked";
MenuView.RUN_ITEM_CLICKED = "gnome:runMenuItemClicked";
MenuView.RUN_UNTIL_ITEM_CLICKED = "gnome:runUntilMenuItemClicked";

MenuView.prototype = {
    initialize: function() {
        var _this = this;

        // TODO: Initialize these events from a data structure.
        $(this.newItemEl).click(function(event) {
            $(_this.modelDropdownEl).dropdown('toggle');
            $(_this).trigger(MenuView.NEW_ITEM_CLICKED);
        });

        $(this.runItemEl).click(function(event) {
            $(_this.runDropdownEl).dropdown('toggle');
            $(_this).trigger(MenuView.RUN_ITEM_CLICKED);
        });

        $(this.runUntilItemEl).click(function(event) {
            $(_this.runDropdownEl).dropdown('toggle');
            $(_this).trigger(MenuView.RUN_UNTIL_ITEM_CLICKED);
        });
    }
};


function MapController(opts) {
    _.bindAll(this);

    // TODO: Obtain from server via template variable passed into controller.
    this.urls = {
        model: "/model",
        timeSteps: '/model/time_steps',
        runUntil: '/model/run_until',
        setting: '/model/setting',
        mover: '/model/mover',
        spill: '/model/spill',
        tree: '/tree'
    };

    this.model = new Model(opts.generatedTimeSteps, {
        url: this.urls.model,
        expectedTimeSteps: opts.expectedTimeSteps
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

    this.formView = new ModalFormView({
        formContainerEl: opts.formContainerEl,
        urls: this.urls
    });

    this.sidebarEl = opts.sidebarEl;

    this.treeView = new TreeView({
        treeEl: "#tree",
        url: this.urls.tree
    });

    this.treeControlView = new TreeControlView({
        addButtonEl: "#add-button",
        removeButtonEl: "#remove-button",
        settingsButtonEl: "#settings-button"
    });

    this.mapView = new MapView({
        mapEl: opts.mapEl,
        placeholderEl: opts.mapPlaceholderEl,
        frameClass: 'frame',
        activeFrameClass: 'active'
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
        url: this.urls.timeSteps,
        model: this.model
    });

    this.setupEventHandlers();
    this.initializeViews();

    return this;
}


MapController.prototype = {
    setupEventHandlers: function() {
        this.model.on(Model.RUN_BEGAN, this.modelRunBegan);
        this.model.on(Model.NEXT_TIME_STEP_READY, this.nextTimeStepReady);

        $(this.model).bind(Model.RUN_FINISHED, this.modelRunFinished);

        $(this.formView).bind(ModalFormView.SAVE_BUTTON_CLICKED, this.saveForm);
        $(this.formView).bind(ModalFormView.FORM_SUBMITTED, this.formSubmitted);

        $(this.treeView).bind(TreeView.ITEM_ACTIVATED, this.treeItemActivated);
        $(this.treeView).bind(TreeView.ITEM_DOUBLE_CLICKED, this.treeItemDoubleClicked);

        $(this.treeControlView).bind(
            TreeControlView.ADD_BUTTON_CLICKED, this.addButtonClicked);
        $(this.treeControlView).bind(
            TreeControlView.REMOVE_BUTTON_CLICKED, this.removeButtonClicked);
        $(this.treeControlView).bind(
            TreeControlView.SETTINGS_BUTTON_CLICKED, this.settingsButtonClicked);

        $(this.mapControlView).bind(
            MapControlView.PLAY_BUTTON_CLICKED, this.playButtonClicked);
        $(this.mapControlView).bind(
            MapControlView.PAUSE_BUTTON_CLICKED, this.pauseButtonClicked);
        $(this.mapControlView).bind(
            MapControlView.ZOOM_IN_BUTTON_CLICKED, this.enableZoomIn);
        $(this.mapControlView).bind(
            MapControlView.ZOOM_OUT_BUTTON_CLICKED, this.enableZoomOut);
        $(this.mapControlView).bind(
            MapControlView.SLIDER_CHANGED, this.sliderChanged);
        $(this.mapControlView).bind(
            MapControlView.SLIDER_MOVED, this.sliderMoved);
        $(this.mapControlView).bind(
            MapControlView.BACK_BUTTON_CLICKED, this.jumpToFirstFrame);
        $(this.mapControlView).bind(
            MapControlView.FORWARD_BUTTON_CLICKED, this.jumpToLastFrame);
        $(this.mapControlView).bind(
            MapControlView.FULLSCREEN_BUTTON_CLICKED, this.useFullscreen);
        $(this.mapControlView).bind(
            MapControlView.RESIZE_BUTTON_CLICKED, this.disableFullscreen);

        $(this.mapView).bind(MapView.PLAYING_FINISHED, this.stopAnimation);
        $(this.mapView).bind(MapView.DRAGGING_FINISHED, this.zoomIn);
        $(this.mapView).bind(MapView.FRAME_CHANGED, this.frameChanged);
        $(this.mapView).bind(MapView.MAP_WAS_CLICKED, this.zoomOut);

        $(this.menuView).bind(
            MenuView.NEW_ITEM_CLICKED, this.newMenuItemClicked);
        $(this.menuView).bind(
            MenuView.RUN_ITEM_CLICKED, this.runMenuItemClicked);
        $(this.menuView).bind(
            MenuView.RUN_UNTIL_ITEM_CLICKED, this.runUntilMenuItemClicked);
    },

    initializeViews: function() {
        this.formView.initialize();
        this.treeControlView.initialize();
        this.treeView.initialize();
        this.mapControlView.initialize();
        this.mapView.initialize();
        this.menuView.initialize();
    },

    runMenuItemClicked: function(event) {
        var stepNum = this.mapView.currentTimeStep;
        var opts = this.mapControlView.isPaused() ? {startFromTimeStep: stepNum} : {};
        this.play(opts);
    },

    runUntilMenuItemClicked: function(event) {
        this.showForm(this.urls.runUntil);
    },

    newMenuItemClicked: function() {
        var _this = this;

        if (!confirm("Reset model?")) {
            return;
        }

        $.ajax({
            url: this.urls.model + "/create",
            data: "confirm=1",
            type: "POST",
            tryCount: 0,
            retryLimit: 3,
            success: function(data) {
                if ('message' in data) {
                    _this.displayMessage(data.message);
                }
                _this.model.clearData();
                _this.treeView.reload();
                _this.mapView.clear();
                _this.mapView.createPlaceholderCopy();
                _this.mapView.setStopped();
                _this.mapControlView.setStopped();
            },
            error: window.noaa.erd.gnome.handleAjaxError
        });
    },

    play: function(opts) {
        if (this.model.dirty) {
            this.mapControlView.reset();
        }

        if (this.mapControlView.isStopped()) {
            this.model.resetCurrentTimeStep();
        }

        this.mapView.setPlaying();
        this.mapControlView.setPlaying();
        this.mapControlView.disableControls();
        this.model.run(opts);
    },

    pause: function() {
        // TODO: Should the mapView have a reference to mapControlView?
        this.mapControlView.setPaused();
        this.mapView.setPaused();
        this.mapControlView.enableControls();
    },

    playButtonClicked: function(event) {
        var stepNum = this.mapView.currentTimeStep;
        var opts = this.mapControlView.isPaused() ? {startFromTimeStep: stepNum} : {};
        this.play(opts);
    },

    pauseButtonClicked: function(event) {
        this.pause();
    },

    enableZoomIn: function(event) {
        if (this.model.hasData() === false) {
            return;
        }

        this.mapControlView.setZoomingIn();
        this.mapView.makeActiveImageClickable();
        this.mapView.makeActiveImageSelectable();
        this.mapView.setZoomingInCursor();
    },

    enableZoomOut: function(event) {
        if (this.model.hasData() === false) {
            return;
        }

        this.mapControlView.setZoomingOut();
        this.mapView.makeActiveImageClickable();
        this.mapView.setZoomingOutCursor();
    },

    stopAnimation: function(event) {
        this.mapControlView.setStopped();
    },

    zoomIn: function(event, startPosition, endPosition) {
        this.setTimeStep(0);

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

    zoomOut: function(event, point) {
        this.setTimeStep(0);
        this.model.zoomFromPoint(point, Model.ZOOM_OUT);
        this.mapView.setRegularCursor();
    },

    sliderMoved: function(event, stepNum) {
        var timestamp = this.model.getTimestampForStep(stepNum);
        if (timestamp) {
            this.mapControlView.setTime(timestamp);
        } else {
            alert("Time step does not exist.");
            console.log("Step number: ", stepNum, "Model timestamps: ",
                this.model.expectedTimeSteps);
        }
    },

    sliderChanged: function(event, newStepNum) {
        // If the model is dirty, we need to run until the new time step.
        if (this.model.dirty) {
            if (confirm("You have changed settings. Re-run the model now?")) {
                this.play({
                    startFromTimeStep: this.mapView.currentTimeStep,
                    runUntilTimeStep: newStepNum
                });
            } else {
                this.pause();
            }
            return;
        }

        if (newStepNum == this.mapView.currentTimeStep) {
            return;
        }

        // If the model and map view have the time step, display it.
        if (this.model.hasCachedTimeStep(newStepNum) &&
            this.mapView.timeStepIsLoaded(newStepNum)) {

            var timestamp = this.model.getTimestampForStep(newStepNum);
            this.mapControlView.setTime(timestamp);
            this.mapView.setTimeStep(newStepNum);
            this.mapControlView.setTimeStep(newStepNum);
            this.mapControlView.setTime(timestamp);
            return;
        }

        // Otherwise, we need to run until the new time step.
        this.play({
            startFromTimeStep: this.mapView.currentTimeStep,
            runUntilTimeStep: newStepNum
        });
    },

    frameChanged: function(event) {
        this.mapControlView.setTimeStep(this.mapView.currentTimeStep);
        this.mapControlView.setTime(
            this.model.getTimestampForStep(this.mapView.currentTimeStep));
        this.model.getNextTimeStep(false);

        if (this.model.runUntilTimeStep &&
            this.mapView.currentTimeStep == this.model.runUntilTimeStep) {

            this.pause();
        }
    },

    // Set the model and all controls to `stepNum`.
    setTimeStep: function(stepNum) {
        this.model.setTimeStep(stepNum);
        this.mapControlView.setPaused();
        this.mapControlView.setTimeStep(stepNum);
        this.mapView.setPaused();
        this.mapView.setTimeStep(stepNum);
    },

    jumpToFirstFrame: function(event) {
        this.setTimeStep(0);
    },

    // Jump to the last LOADED frame of the animation. This will stop at
    // whatever frame was the last received from the server.
    //
    // TODO: This should probably do something fancier, like block and load
    // all of the remaining frames if they don't exist, until the end.
    jumpToLastFrame: function(event) {
        var lastFrame = this.mapView.getFrameCount();
        this.setTimeStep(lastFrame);
    },

    useFullscreen: function(event) {
        this.mapControlView.switchToFullscreen();
        $(this.sidebarEl).hide('slow');
    },

    disableFullscreen: function(event) {
        this.mapControlView.switchToNormalScreen();
        $(this.sidebarEl).show('slow');
    },

    treeItemActivated: function(event) {
        this.treeControlView.enableControls();
    },

    /*
     * Get the URL path for the currently-selected item in the tree. The URL 
     * is looked up in `this.urls`.
     */
    getUrlForTreeItem: function(node, mode) {
        var urlKey = null;
        var url = null;

        // TODO: Only activate 'Add item' button when a root node is selected.
        if (node === null) {
            console.log('Failed to get active node');
            return false;
        }

        if (node.data.key == 'setting' || node.data.key == 'spill') {
            return false;
        }

        // If this is a top-level node, then its `data.key` value will
        // match a URL in `this.urls`. Otherwise it's a child node and its
        // parent (in `node.parent` will have a `key` value  set to
        // 'setting', 'spill' or 'mover' and `node` will have a `data.type`
        // value specific to its server-side representation, e.g.
        // 'constant_wind'.
        if (node.data.key in this.urls) {
            urlKey = node.data.key;
        } else if (node.parent.data.key in this.urls) {
            urlKey = node.parent.data.key;
        }

        if (!urlKey) {
            return false;
        }

        var rootUrl = this.urls[urlKey];

        if (!rootUrl) {
            return false;
        }

        switch (mode) {
            case 'add':
                url = rootUrl + '/add';
                break;
            case 'edit':
                url = rootUrl + '/' + node.data.type + '/' + mode + '/' + node.data.key;
                break;
            case 'delete':
                url = rootUrl + '/' + mode;
                break;
            default:
                // Default to the add URL.
                url = rootUrl + '/add';
                break;
        }

        return url;
    },

    showFormForActiveTreeItem: function(mode) {
        var node = this.treeView.getActiveItem();
        var url = this.getUrlForTreeItem(node, mode);

        if (url) {
            this.showForm(url);
        }
    },

    addButtonClicked: function(event) {
        this.showFormForActiveTreeItem('add');
    },

    treeItemDoubleClicked: function(event, node) {
        if (node.data.key in this.urls) {
            // A root item
            this.showFormForActiveTreeItem('add');
        } else {
            // A child item
            this.showFormForActiveTreeItem('edit');
        }
    },

    settingsButtonClicked: function(event) {
        this.showFormForActiveTreeItem('edit');
    },

    saveForm: function() {
        this.formView.submitForm();
    },

    removeButtonClicked: function(event) {
        var node = this.treeView.getActiveItem();
        var url = this.getUrlForTreeItem(node, 'delete');
        var _this = this;

        if (!node.data.key) {
            return;
        }

        if (confirm('Remove mover?') === false) {
            return;
        }

        $.ajax({
            type: "POST",
            url: url,
            tryCount: 0,
            retryLimit: 3,
            data: "mover_id=" + node.data.key,
            success: function(event, data) {
                _this.treeView.reload();
            },
            error: function() {
                alert('Could not remove item.');
            }
        });
    },

    showForm: function(url) {
        $.ajax({
            type: 'GET',
            url: url,
            tryCount: 0,
            retryLimit: 3,
            success: this.submitSuccess,
            error: window.noaa.erd.gnome.handleAjaxError
        });
    },

    formSubmitted: function(event, form) {
        $.ajax({
            type: 'POST',
            tryCount: 0,
            retryLimit: 3,
            data: $(form).serialize(),
            url: $(form).attr('action'),
            success: this.submitSuccess,
            error: window.noaa.erd.gnome.handleAjaxError
        });
    },

    submitSuccess: function(data, textStatus, xhr) {
        if ('form_html' in data) {
            this.formView.reloadForm(data.form_html);
        } else {
            this.treeView.reload();
            this.formView.clearForm();
        }

        if ('message' in data) {
            this.displayMessage(data.message);
        }
    },

    modelRunBegan: function(event, data) {
        this.model.getNextTimeStep(true);
        return true;
    },

    stop: function() {
        this.mapControlView.setStopped();
        this.mapView.setStopped();
        this.mapControlView.enableControls();
    },

    modelRunFinished: function(event) {
        this.stop();
    },

    nextTimeStepReady: function(event, step) {
        this.mapView.addTimeStep(step);
    }
};


/*
 The `Router` initializes all views and handles application routing.
 */
window.noaa.erd.gnome.Router = Backbone.Router.extend({
});

window.noaa.erd.gnome.MapController = MapController;
    

