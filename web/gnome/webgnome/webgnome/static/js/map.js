function MapModel(opts) {
    // Optionally specify the current frame the user is in.
    this.frame = opts.frame == undefined ? 0 : opts.frame;

    // Optionally specify the zoom level of the map.
    this.zoomLevel = opts.zoomLevel == undefined ? 4 : opts.zoomLevel;

    this.data = null;

    // If true, `MapModel` will request a new set of frames from the server
    // when the user runs the model.
    this.dirty = true;
}


// `MapModel` events
MapModel.RUN_FINISHED = 'gnome:runFinished';
MapModel.RUN_FAILED = 'gnome:runFailed';

MapModel.RUN_URL = '/model/run';
MapModel.ZOOM_IN = 'zoom_in';
MapModel.ZOOM_OUT = 'zoom_out';
MapModel.ZOOM_NONE = 'zoom_none';


MapModel.prototype = {
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

        // TOOD: This looks wrong. Add tests.
        if (adjustedRect.start.x > this.bbox[0].x) {
            adjustedRect.start.x = this.bbox[0].x;
        }
        if (adjustedRect.start.y < this.bbox[0].y) {
            adjustedRect.start.y = this.bbox[0].y;
        }

        if (adjustedRect.end.x < this.bbox[1].x) {
            adjustedRect.end.x = this.bbox[1].x;
        }
        if (adjustedRect.end.y > this.bbox[1].y) {
            adjustedRect.end.y = this.bbox[1].y;
        }

        return adjustedRect;
    },

    isPositionInsideMap: function(position) {
        return (position.x > this.bbox[0].x && position.x < this.bbox[1].x
            && position.y > this.bbox[0].y && position.y < this.bbox[1].y);
    },

    isRectInsideMap: function(rect) {
        var _rect = this.getRect(rect);

        return this.isPositionInsideMap(_rect.start) &&
            this.isPositionInsideMap(_rect.end);
    },

    handleFailedRun: function(data) {
        if ('message' in data && data.message.type == 'error') {
            this.error = data.message;
        } else {
            this.error = {text: 'The model failed to run. Please try again.'}
        }
        $(this).trigger(MapModel.RUN_FAILED);
    },

    run: function(opts) {
        var _this = this;

        if (this.dirty === false) {
            $(_this).trigger(MapModel.RUN_FINISHED);
            return;
        }

        opts = $.extend({
            zoomLevel: this.zoomLevel,
            zoomDirection: MapModel.ZOOM_NONE
        }, opts);

        var isInvalid = function(obj) {
            return obj === undefined || obj === null || typeof(obj) != "object";
        };

        if ((opts.zoomLevel != this.zoomLevel) &&
            (isInvalid(opts.rect) && isInvalid(opts.point))) {
            alert("Could not zoom. Please try again.");
        }

        $.ajax({
            type: 'POST',
            url: MapModel.RUN_URL,
            data: opts,
            success: function(data) {
                if ('message' in data) {
                    _this.handleFailedRun(data);
                    return false;
                }
                _this.dirty = false;
                _this.data = data;
                $(_this).trigger(MapModel.RUN_FINISHED);
                return false;
            },
            error: function(data) {
                _this.handleFailedRun(data);
                return false;
            }
        });
    },

    zoomFromPoint: function(point, direction) {
        this.dirty = true;
        this.run({point: point, zoom: direction});
    },

    zoomFromRect: function(rect, direction) {
        var isInsideMap = this.isRectInsideMap(rect);

        // If we are at zoom level 0 and there is no map portion outside of
        // the visible area, then adjust the coordinates of the selected
        // rectangle to the on-screen pixel bound.
        if (!isInsideMap && this.zoomLevel === 0) {
            rect = this.getAdjustedRect(rect);
        }

        this.dirty = true;
        this.run({rect: rect, zoom: direction});
    },

    getFrames: function() {
        if (this.data === undefined) {
            return null;
        } else {
            return this.data.result;
        }
    },

    getTimestampForFrame: function(frameNum) {
        var timestamp = this.data.result[frameNum].timestamp;
        var date = new Date(Date.parse(timestamp));
        if (date) {
            return date.toUTCString();
        }
        return null;
    },

    hasData: function() {
        return this.data != null;
    },

    setBoundingBox: function(bbox) {
        this.bbox = bbox;
    }
};


function MapView(opts) {
    this.mapEl = opts.mapEl;
    this.frameClass = opts.frameClass;
    this.activeFrameClass = opts.activeFrameClass;
    this.currentFrame = 0;
}

// `MapView` events
MapView.INIT_FINISHED = 'gnome:mapInitFinished';
MapView.DRAGGING_FINISHED = 'gnome:draggingFinished';
MapView.REFRESH_FINISHED = 'gnome:refreshFinished';
MapView.PLAYING_FINISHED = 'gnome:playingFinished';
MapView.FRAME_CHANGED = 'gnome:frameChanged';
MapView.MAP_WAS_CLICKED = 'gnome:mapWasClicked';


MapView.prototype = {
    initialize: function() {
        // Only have to do this once.
        this.makeImagesClickable();
        return this;
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
        return $(this.mapEl + " > img.active")
    },

    // Refresh the map from a new set of rendered images.
    refresh: function(frames) {
        var _this = this;

        $(this.mapEl).empty();
        $(this.mapEl).hide();

        $.each(frames, function(index) {
            var img = $('<img>').attr({
                class: 'frame',
                'data-position': index,
                src: this.url
            });

            img.appendTo($(_this.mapEl));
        });

        this.cycle = $(this.mapEl).cycle({
            slideResize: true,
            containerResize: false,
            width: '100%',
            fit: 1,
            nowrap: true,
            speed: 1,
            timeout: 1,
            delay: 300,
            end: function(opts) {
                _this.pause();
                var image = _this.getActiveImage();
                _this.currentFrame = image.attr('data-position');
                $(_this).trigger(MapView.PLAYING_FINISHED);
            },
            before: function(currSlideElement, nextSlideElement, options, forwardFlag) {
                $(currSlideElement).removeClass('active');
                $(nextSlideElement).addClass('active');
            },
            after: function(currSlideElement, nextSlideElement, options, forwardFlag) {
                _this.currentFrame = $(nextSlideElement).attr('data-position');
                $(_this).trigger(MapView.FRAME_CHANGED);
            }
        });

        this.cycle.cycle('pause')
        $(this.mapEl).show();

        $(this).trigger(MapView.REFRESH_FINISHED);
    },

    play: function() {
        this.cycle.cycle('resume');
    },

    pause: function() {
        this.cycle.cycle('pause');
    },

    restart: function() {
        this.cycle.cycle(0);
        this.cycle.cycle('resume');
    },

    getSize: function() {
        var image = this.getActiveImage();
        return {height: image.height(), width: image.width()}
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

    setCurrentFrame: function(frameNum) {
        this.cycle.cycle(frameNum);
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
    }
};


function TreeView(opts) {
    this.treeEl = opts.treeEl;
    return this;
}

TreeView.ITEM_ACTIVATED = 'gnome:treeItemActivated';

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
            initAjax: {
                url: '/tree'
            },
            persist: true
        });

        return this;
    },

    getActiveItem: function() {
        return this.tree.dynatree("getActiveNode");
    },

    hasItem: function(data) {
        return this.tree.dynatree('getTree').selectKey(data.id) != null;
    },

    addItem: function(data, parent) {
        if (!'id' in data || !'type' in data) {
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
            [this.settingsButtonEl, TreeControlView.SETTINGS_BUTTON_CLICKED],
        ];

        _.each(_.object(clickEvents), function(customEvent, element) {
            $(element).click(function(event) {
                $(_this).trigger(customEvent);
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
    },
};


function AnimationControlView(opts) {
    this.sliderEl = opts.sliderEl;
    this.playButtonEl = opts.playButtonEl;
    this.pauseButtonEl = opts.pauseButtonEl;
    this.backButtonEl = opts.backButtonEl;
    this.forwardButtonEl = opts.forwardButtonEl;
    this.zoomInButtonEl = opts.zoomInButtonEl;
    this.zoomOutButtonEl = opts.zoomOutButtonEl;
    this.moveButtonEl = opts.moveButtonEl;
    this.fullscreenButtonEl = opts.fullscreenButtonEl;
    this.resizeButtonEl = opts.resizeButtonEl;
    this.timeEl = opts.timeEl;

    this.controls = [
        this.backButtonEl, this.forwardButtonEl, this.playButtonEl,
        this.pauseButtonEl, this.moveButtonEl, this.fullscreenButtonEl,
        this.resizeButtonEl
    ];

    return this;
}

// Events for `AnimationControlView`
AnimationControlView.PLAY_BUTTON_CLICKED = "gnome:playButtonClicked";
AnimationControlView.PAUSE_BUTTON_CLICKED = "gnome:pauseButtonClicked";
AnimationControlView.BACK_BUTTON_CLICKED = "gnome:backButtonClicked";
AnimationControlView.FORWARD_BUTTON_CLICKED = "gnome:forwardButtonClicked";
AnimationControlView.ZOOM_IN_BUTTON_CLICKED = "gnome:zoomInButtonClicked";
AnimationControlView.ZOOM_OUT_BUTTON_CLICKED = "gnome:zoomOutButtonClicked";
AnimationControlView.MOVE_BUTTON_CLICKED = "gnome:moveButtonClicked";
AnimationControlView.FULLSCREEN_BUTTON_CLICKED = "gnome:fullscreenButtonClicked";
AnimationControlView.RESIZE_BUTTON_CLICKED = "gnome:resizeButtonClicked";
AnimationControlView.SLIDER_CHANGED = "gnome:sliderChanged";

// Statuses
AnimationControlView.STATUS_STOPPED = 0;
AnimationControlView.STATUS_PLAYING = 1;
AnimationControlView.STATUS_PAUSED = 2;
AnimationControlView.STATUS_BACK = 3;
AnimationControlView.STATUS_FORWARD = 4;
AnimationControlView.STATUS_ZOOMING_IN = 5;
AnimationControlView.STATUS_ZOOMING_OUT = 6;

AnimationControlView.prototype = {
    initialize: function() {
        var _this = this;
        this.status = AnimationControlView.STATUS_STOPPED;

        $(this.pauseButtonEl).hide();
        $(this.resizeButtonEl).hide();

        $(this.sliderEl).slider({
            start: function(event, ui) {
                $(_this).trigger(AnimationControlView.PAUSE_BUTTON_CLICKED);
            },
            change: function(event, ui) {
                $(_this).trigger(AnimationControlView.SLIDER_CHANGED, ui.value);
            },
            disabled: true
        });

        $(this.pauseButtonEl).click(function() {
            if (_this.status === AnimationControlView.STATUS_PLAYING) {
                $(_this).trigger(AnimationControlView.PAUSE_BUTTON_CLICKED);
            }
        });

        var clickEvents = [
            [this.playButtonEl, AnimationControlView.PLAY_BUTTON_CLICKED],
            [this.backButtonEl, AnimationControlView.BACK_BUTTON_CLICKED],
            [this.forwardButtonEl, AnimationControlView.FORWARD_BUTTON_CLICKED],
            [this.zoomInButtonEl, AnimationControlView.ZOOM_IN_BUTTON_CLICKED],
            [this.zoomOutButtonEl, AnimationControlView.ZOOM_OUT_BUTTON_CLICKED],
            [this.moveButtonEl, AnimationControlView.MOVE_BUTTON_CLICKED],
            [this.fullscreenButtonEl, AnimationControlView.FULLSCREEN_BUTTON_CLICKED],
            [this.resizeButtonEl, AnimationControlView.RESIZE_BUTTON_CLICKED]
        ];

        _.each(_.object(clickEvents), function(customEvent, element) {
            $(element).click(function(event) {
                $(_this).trigger(customEvent);
            });
        });

        return this;
    },

    setStopped: function() {
        this.status = AnimationControlView.STATUS_STOPPED;
    },

    setPlaying: function() {
        this.status = AnimationControlView.STATUS_PLAYING;
        $(this.playButtonEl).hide();
        $(this.pauseButtonEl).show();
    },

    setPaused: function() {
        this.status = AnimationControlView.STATUS_PAUSED;
        $(this.pauseButtonEl).hide();
        $(this.playButtonEl).show();
    },

    setForward: function() {
        this.status = AnimationControlView.STATUS_FORWARD;
    },

    setBack: function() {
        this.status = AnimationControlView.STATUS_BACK;
    },

    setZoomingIn: function() {
        this.status = AnimationControlView.STATUS_ZOOMING_IN;
    },

    setZoomingOut: function() {
        this.status = AnimationControlView.STATUS_ZOOMING_OUT;
    },

    setFrameCount: function(frameCount) {
        $(this.sliderEl).slider('option', 'max', frameCount);
    },

    setCurrentFrame: function(frameNum) {
        $(this.sliderEl).slider('value', frameNum);
    },

    setTime: function(time) {
        $(this.timeEl).text(time);
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
        return this.status === AnimationControlView.STATUS_PLAYING;
    },

    isStopped: function() {
        return this.status === AnimationControlView.STATUS_STOPPED;
    },

    isPaused: function() {
        return this.status === AnimationControlView.STATUS_PAUSED;
    },

    isForward: function() {
        return this.status === AnimationControlView.STATUS_PLAYING;
    },

    isBack: function() {
        return this.status === AnimationControlView.STATUS_BACK;
    },

    isZoomingIn: function() {
        return this.status === AnimationControlView.STATUS_ZOOMING_IN;
    },

    isZoomingOut: function() {
        return this.status === AnimationControlView.STATUS_ZOOMING_OUT;
    },

    enableControls: function() {
        $(this.sliderEl).slider('option', 'disabled', false);
        _.each(this.controls, function(buttonEl) {
            $(buttonEl).removeClass('disabled');
        });
    },

    disableControls: function() {
        $(this.sliderEl).slider('option', 'disabled', true);
        _.each(this.controls, function(item) {
            $(item).removeClass('enabled');
        });
    }
};


function ModalFormView(opts) {
    _.bindAll(this);
    this.formContainerEl = opts.formContainerEl;
    this.rootApiUrls = opts.rootApiUrls;
}

ModalFormView.SAVE_BUTTON_CLICKED = 'gnome:formSaveButtonClicked';

ModalFormView.SUBMIT_SUCCESS = 'gnome:formSubmitSuccess';

// These events will fire if the server included a message in the JSON response
// containing the rendered HTML of a submitted form. They are not fired for
// regular form validation errors, which appear as HTML markup.
ModalFormView.FORM_SUCCESS = 'gnome:formSuccess';
ModalFormView.FORM_WARNING = 'gnome:formWarning';
ModalFormView.FORM_ERROR = 'gnome:formError';

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
            _this.doAjaxFormSubmit(this);
            return false;
        });
    },

    getFirstStepWithError: function() {
        var form = $(this.formContainerEl).find('form');
        var step = 1;

        if (!form.hasClass('multistep')) {
            return;
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
        stepNum = parseInt(stepNum);
        var form = $(this.formContainerEl).find('form');
        return form.find('div[data-step="' + (stepNum + 1) + '"]').length > 0;
    },

    goToStep: function(stepNum) {
        var form = $(this.formContainerEl).find('form');

        if (!form.hasClass('multistep')) {
            return;
        }

        var stepDiv = form.find('div.step[data-step="' + stepNum + '"]');

        if (stepDiv.length == 0) {
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
        var currentStep = parseInt(activeStep.attr('data-step'));
        this.goToStep(currentStep + 1);
    },
    
    goToPreviousStep: function(event) {
        var form = $(this.formContainerEl).find('form');

        if (!form.hasClass('multistep')) {
            return;
        }

        var activeStep = form.find('div.step.active');
        var currentStep = parseInt(activeStep.attr('data-step'));
        this.goToStep(currentStep - 1);
    },

    showForm: function(urlKey, type, id) {
        var _this = this;
        var rootUrl = this.rootApiUrls[urlKey];
        var url = id && type ? rootUrl + '/' + type + '/edit/' + id : rootUrl + '/add';

        $.ajax({
            type: 'GET',
            url: url,
            success: this.handleAjaxSuccess,
            error: this.handleAjaxError
        });
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
    },

    doAjaxFormSubmit: function(form) {
        var _this = this;

        $.ajax({
            type: 'POST',
            data: $(form).serialize(),
            url: $(form).attr('action'),
            success: this.handleAjaxSuccess,
            error: this.handleAjaxError
        });
    },

    handleAjaxSuccess: function(data, textStatus, xhr) {
        if ('form_html' in data) {
            this.reloadForm(data.form_html);
        } else {
            $(this).trigger(ModalFormView.SUBMIT_SUCCESS, data);
            this.clearForm();
        }

        if ('message' in data) {
            var messageEvent;

            switch (data.message.type) {
                case 'success':
                    messageEvent = ModalFormView.FORM_SUCCESS;
                    break;
                case 'error':
                    messageEvent = ModalFormView.FORM_ERROR;
                    break;
                case 'warning':
                    messageEvent = ModalFormView.FORM_WARNING;
                    break;
            }

            if (messageEvent) {
                $(this).trigger(messageEvent, data.message);
            }
        }
    },
    
    handleAjaxError: function(xhr, textStatus, errorThrown) {
        alert('Could not connect to server.');
        console.log(xhr, textStatus, errorThrown);
    }
};


function MapController(opts) {
    _.bindAll(this);

    // TODO: Obtain from server via template variable passed into controller.
    this.rootApiUrls = {
        setting: '/model/setting',
        mover: '/model/mover',
        spill: '/model/spill'
    };

    this.formView = new ModalFormView({
        formContainerEl: opts.formContainerEl,
        rootApiUrls: this.rootApiUrls
    });

    this.sidebarEl = opts.sidebarEl;

    this.treeView = new TreeView({
        treeEl: "#tree"
    });

    this.treeControlView = new TreeControlView({
        addButtonEl: "#add-button",
        removeButtonEl: "#remove-button",
        settingsButtonEl: "#settings-button"
    });

    this.mapView = new MapView({
        mapEl: opts.mapEl,
        frameClass: 'frame',
        activeFrameClass: 'active'
    });

    this.animationControlView = new AnimationControlView({
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
        timeEl: "#time"
    });

    this.mapModel = new MapModel({});

    this.setupEventHandlers();
    this.initializeViews();

    return this;
}

MapController.prototype = {
    setupEventHandlers: function() {
        $(this.formView).bind(ModalFormView.SAVE_BUTTON_CLICKED, this.saveForm);
        $(this.formView).bind(ModalFormView.FORM_WARNING, this.receivedMessage);
        $(this.formView).bind(ModalFormView.FORM_SUCCESS, this.receivedMessage);
        $(this.formView).bind(ModalFormView.FORM_ERROR, this.receivedMessage);
        $(this.formView).bind(ModalFormView.SUBMIT_SUCCESS, this.formSubmitted);

        $(this.treeView).bind(TreeView.ITEM_ACTIVATED, this.treeItemActivated);

        $(this.treeControlView).bind(
            TreeControlView.ADD_BUTTON_CLICKED, this.addButtonClicked);
        $(this.treeControlView).bind(
            TreeControlView.REMOVE_BUTTON_CLICKED, this.removeButtonClicked);
        $(this.treeControlView).bind(
            TreeControlView.SETTINGS_BUTTON_CLICKED, this.settingsButtonClicked);

        $(this.animationControlView).bind(
            AnimationControlView.PLAY_BUTTON_CLICKED, this.play);
        $(this.animationControlView).bind(
            AnimationControlView.PAUSE_BUTTON_CLICKED, this.pause);
        $(this.animationControlView).bind(
            AnimationControlView.ZOOM_IN_BUTTON_CLICKED, this.enableZoomIn);
        $(this.animationControlView).bind(
            AnimationControlView.ZOOM_OUT_BUTTON_CLICKED, this.enableZoomOut);
        $(this.animationControlView).bind(
            AnimationControlView.SLIDER_CHANGED, this.sliderChanged);
        $(this.animationControlView).bind(
            AnimationControlView.BACK_BUTTON_CLICKED, this.jumpToFirstFrame);
        $(this.animationControlView).bind(
            AnimationControlView.FORWARD_BUTTON_CLICKED, this.jumpToLastFrame);
        $(this.animationControlView).bind(
            AnimationControlView.FULLSCREEN_BUTTON_CLICKED, this.useFullscreen);
        $(this.animationControlView).bind(
            AnimationControlView.RESIZE_BUTTON_CLICKED, this.disableFullscreen);

        $(this.mapModel).bind(MapModel.RUN_FINISHED, this.restart);
        $(this.mapModel).bind(MapModel.RUN_FAILED, this.runFailed);

        $(this.mapView).bind(MapView.INIT_FINISHED, this.mapInitFinished);
        $(this.mapView).bind(MapView.REFRESH_FINISHED, this.refreshFinished);
        $(this.mapView).bind(MapView.PLAYING_FINISHED, this.stopAnimation);
        $(this.mapView).bind(MapView.DRAGGING_FINISHED, this.zoomIn);
        $(this.mapView).bind(MapView.FRAME_CHANGED, this.frameChanged);
        $(this.mapView).bind(MapView.MAP_WAS_CLICKED, this.zoomOut);
    },

    initializeViews: function() {
        this.formView.initialize();
        this.treeControlView.initialize();
        this.treeView.initialize();
        this.animationControlView.initialize();
        this.mapView.initialize();
    },

    mapInitFinished: function() {
        // XXX: Get bbox from the server?
        var bbox = this.mapView.getBoundingBox();
        this.mapModel.setBoundingBox(bbox);
    },

    play: function(event) {
        if (this.animationControlView.isPaused()) {
            this.animationControlView.setPlaying();
            this.mapView.play();
        } else {
            this.animationControlView.disableControls();
            this.animationControlView.setPlaying();
            this.mapModel.run();
        }
    },

    pause: function(event) {
        this.animationControlView.setPaused();
        this.mapView.pause();
    },

    enableZoomIn: function(event) {
        if (this.mapModel.hasData() === false) {
            return;
        }

        this.animationControlView.setZoomingIn();
        this.mapView.makeActiveImageClickable();
        this.mapView.makeActiveImageSelectable();
        this.mapView.setZoomingInCursor();
    },

    enableZoomOut: function(event) {
        if (this.mapModel.hasData() === false) {
            return;
        }

        this.animationControlView.setZoomingOut();
        this.mapView.makeActiveImageClickable();
        this.mapView.setZoomingOutCursor();
    },

    restart: function(event) {
        var frames = this.mapModel.getFrames();

        if (frames === null) {
            this.mapView.restart();
            return;
        }
        this.mapView.refresh(frames);
    },

    runFailed: function(event) {
        alert(this.mapModel.error.text);
    },

    refreshFinished: function(event) {
        this.animationControlView.enableControls();
        this.animationControlView.setFrameCount(this.mapView.getFrameCount());
        if (this.animationControlView.isPlaying()) {
            this.mapView.play();
        }
    },

    stopAnimation: function(event) {
        this.animationControlView.setStopped();
    },

    zoomIn: function(event, startPosition, endPosition) {
        this.animationControlView.setStopped();

        if (endPosition) {
            this.mapModel.zoomFromRect(
                {start: startPosition, end: endPosition},
                MapModel.ZOOM_IN
            );
        } else {
            this.mapModel.zoomFromPoint(
                startPosition,
                MapModel.ZOOM_OUT
            );
        }

        this.mapView.setRegularCursor();
    },

    zoomOut: function(event, point) {
        this.mapModel.zoomFromPoint(
            point, MapModel.ZOOM_OUT
        );
        this.mapView.setRegularCursor();
    },

    sliderChanged: function(event, newFrame) {
        if (newFrame != this.mapView.currentFrame) {
            this.mapView.setCurrentFrame(newFrame);
        }
    },

    frameChanged: function(event) {
        var timestamp = this.mapModel.getTimestampForFrame(this.mapView.currentFrame);
        this.animationControlView.setCurrentFrame(this.mapView.currentFrame);
        this.animationControlView.setTime(timestamp);
    },

    jumpToFirstFrame: function(event) {
        this.mapView.setCurrentFrame(0);
    },

    jumpToLastFrame: function(event) {
        var lastFrame = this.mapView.getFrameCount();
        this.mapView.setCurrentFrame(lastFrame);
        this.animationControlView.setCurrentFrame(lastFrame);
    },

    useFullscreen: function(event) {
        this.animationControlView.switchToFullscreen();
        $(this.sidebarEl).hide('slow');
    },

    disableFullscreen: function(event) {
        this.animationControlView.switchToNormalScreen();
        $(this.sidebarEl).show('slow');
    },

    treeItemActivated: function(event) {
        this.treeControlView.enableControls();
    },

    showFormForActiveTreeItem: function(opts) {
        opts = $.extend({
            editing: false
        }, opts);

        console.log(opts)

        var node = this.treeView.getActiveItem();
        var urlKey = null;
        var id = null;
        var type = null;

        // TODO: Only activate 'Add item' button when a root node is selected.
        if (node === null) {
            console.log('Failed to get active node');
            return;
        }

        // If this is a top-level node, then its `data.key` value will match a
        // URL in `this.rootApiUrl`. Otherwise it's a child node and its parent
        // (in `node.parent` will have a `key` value  set to 'setting', 'spill'
        // or 'mover' and `node` will have a `data.type` value specific to its
        // server-side representation, e.g. 'constant_wind'.
        if (node.data.key in this.rootApiUrls) {
            urlKey = node.data.key;
        } else if (node.parent.data.key in this.rootApiUrls) {
            urlKey = node.parent.data.key;

            if (opts.editing) {
                id = node.data.key;
                type = node.data.type;
            }
        }

        console.log(node.data.key)
        console.log(urlKey, id, type)

        if (urlKey) {
            this.formView.showForm(urlKey, type, id);
        }
    },

    addButtonClicked: function(event) {
        this.showFormForActiveTreeItem();
    },

    settingsButtonClicked: function(event) {
        this.showFormForActiveTreeItem({ editing: true });
    },

    saveForm: function() {
        this.formView.submitForm();
    },

    removeButtonClicked: function(event) {
        console.log(event.data.key);
    },

    receivedMessage: function(event, message) {
        if (message) {
            this.displayMessage(message);
        }
    },

    displayMessage: function(message) {
        if (!'type' in message || !'text' in message)  {
            return;
        }

        var alertDiv = $('div .alert-' + message.type);

        if (message.text && alertDiv) {
            alertDiv.find('span.message').text(message.text);
            alertDiv.removeClass('hidden');
        }
    },

    formSubmitted: function(event, data) {
        if ('message' in data) {
            this.displayMessage(data.message);
        }

        this.treeView.reload();
    }
};


// Get an alias to the `window.noaa.erd.gnome` namespace.
gnome = window.noaa.erd.gnome;


// Wait until the placeholder image has loaded before loading controls.
$('#map').imagesLoaded(function() {
    new MapController({
        mapEl: '#map',
        sidebarEl: '#sidebar',
        'formContainerEl': '#modal-container'
    });
});

