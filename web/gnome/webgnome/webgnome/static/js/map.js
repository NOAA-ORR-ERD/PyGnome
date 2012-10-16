// map.js: The WebGNOME JavaScript application.
(function() {

    // Generic AJAX error handler.
    // Retry on error if the request specified tryCount.
    function handleAjaxError(xhr, textStatus, errorThrown) {
        if (textStatus == 'timeout') {
            this.tryCount++;
            if (this.tryCount <= this.retryLimit) {
                //try again
                $.ajax(this);
                return;
            }
            return;
        }

        alert('Could not connect to server.');
        console.log(xhr, textStatus, errorThrown);
    }


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
            $(this).trigger(MapView.INIT_FINISHED);
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


    function MapControlView(opts) {
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
            this.zoomInButtonEl, this.zoomOutButtonEl, this.resizeButtonEl
        ];

        return this;
    }


    // Events for `mapControlView`
    MapControlView.PLAY_BUTTON_CLICKED = "gnome:playButtonClicked";
    MapControlView.PAUSE_BUTTON_CLICKED = "gnome:pauseButtonClicked";
    MapControlView.BACK_BUTTON_CLICKED = "gnome:backButtonClicked";
    MapControlView.FORWARD_BUTTON_CLICKED = "gnome:forwardButtonClicked";
    MapControlView.ZOOM_IN_BUTTON_CLICKED = "gnome:zoomInButtonClicked";
    MapControlView.ZOOM_OUT_BUTTON_CLICKED = "gnome:zoomOutButtonClicked";
    MapControlView.MOVE_BUTTON_CLICKED = "gnome:moveButtonClicked";
    MapControlView.FULLSCREEN_BUTTON_CLICKED = "gnome:fullscreenButtonClicked";
    MapControlView.RESIZE_BUTTON_CLICKED = "gnome:resizeButtonClicked";
    MapControlView.SLIDER_CHANGED = "gnome:sliderChanged";

    // Statuses
    MapControlView.STATUS_STOPPED = 0;
    MapControlView.STATUS_PLAYING = 1;
    MapControlView.STATUS_PAUSED = 2;
    MapControlView.STATUS_BACK = 3;
    MapControlView.STATUS_FORWARD = 4;
    MapControlView.STATUS_ZOOMING_IN = 5;
    MapControlView.STATUS_ZOOMING_OUT = 6;

    MapControlView.prototype = {
        initialize: function() {
            var _this = this;
            this.status = MapControlView.STATUS_STOPPED;

            $(this.pauseButtonEl).hide();
            $(this.resizeButtonEl).hide();

            $(this.sliderEl).slider({
                start: function(event, ui) {
                    $(_this).trigger(MapControlView.PAUSE_BUTTON_CLICKED);
                },
                change: function(event, ui) {
                    $(_this).trigger(MapControlView.SLIDER_CHANGED, ui.value);
                },
                disabled: true
            });

            $(this.pauseButtonEl).click(function() {
                if (_this.status === MapControlView.STATUS_PLAYING) {
                    $(_this).trigger(MapControlView.PAUSE_BUTTON_CLICKED);
                }
            });

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

            _.each(_.object(clickEvents), function(customEvent, element) {
                $(element).click(function(event) {
                    $(_this).trigger(customEvent);
                });
            });

            return this;
        },

        setStopped: function() {
            this.status = MapControlView.STATUS_STOPPED;
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
        this.modelItemEl = opts.modelItemEl;
        this.newItemEl = opts.newItemEl;
        this.runItemEl = opts.runItemEl;
        this.stepItemEl = opts.stepItemEl;
        this.runUntilItemEl = opts.runUntilItemEl;
    }


    MenuView.NEW_ITEM_CLICKED = "gnome:newMenuItemClicked";


    MenuView.prototype = {
        initialize: function() {
            var _this = this;

            $(this.newItemEl).click(function(event) {
                $(_this.modelItemEl).dropdown('toggle');
                $(_this).trigger(MenuView.NEW_ITEM_CLICKED);
            });
        }
    };


    function MapController(opts) {
        _.bindAll(this);

        // TODO: Obtain from server via template variable passed into controller.
        this.rootApiUrls = {
            model: "/model",
            setting: '/model/setting',
            mover: '/model/mover',
            spill: '/model/spill'
        };

        this.menuView = new MenuView({
            modelItemEl: "#file-drop",
            newItemEl: "#menu-new",
            runItemEl: "#menu-run",
            stepItemEl: "#menu-step",
            runUntilItemEl: "#mnu-run-until"
        });

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
                MapControlView.PLAY_BUTTON_CLICKED, this.play);
            $(this.mapControlView).bind(
                MapControlView.PAUSE_BUTTON_CLICKED, this.pause);
            $(this.mapControlView).bind(
                MapControlView.ZOOM_IN_BUTTON_CLICKED, this.enableZoomIn);
            $(this.mapControlView).bind(
                MapControlView.ZOOM_OUT_BUTTON_CLICKED, this.enableZoomOut);
            $(this.mapControlView).bind(
                MapControlView.SLIDER_CHANGED, this.sliderChanged);
            $(this.mapControlView).bind(
                MapControlView.BACK_BUTTON_CLICKED, this.jumpToFirstFrame);
            $(this.mapControlView).bind(
                MapControlView.FORWARD_BUTTON_CLICKED, this.jumpToLastFrame);
            $(this.mapControlView).bind(
                MapControlView.FULLSCREEN_BUTTON_CLICKED, this.useFullscreen);
            $(this.mapControlView).bind(
                MapControlView.RESIZE_BUTTON_CLICKED, this.disableFullscreen);

            $(this.mapView).bind(MapView.INIT_FINISHED, this.mapInitFinished);
            $(this.mapView).bind(MapView.REFRESH_FINISHED, this.refreshFinished);
            $(this.mapView).bind(MapView.PLAYING_FINISHED, this.stopAnimation);
            $(this.mapView).bind(MapView.DRAGGING_FINISHED, this.zoomIn);
            $(this.mapView).bind(MapView.FRAME_CHANGED, this.frameChanged);
            $(this.mapView).bind(MapView.MAP_WAS_CLICKED, this.zoomOut);

            $(this.menuView).bind(MenuView.NEW_ITEM_CLICKED, this.newMenuItemClicked);
        },

        initializeViews: function() {
            this.formView.initialize();
            this.treeControlView.initialize();
            this.treeView.initialize();
            this.mapControlView.initialize();
            this.mapView.initialize();
            this.menuView.initialize();
        },

        mapInitFinished: function() {
            // XXX: Get bbox from the server?
            var bbox = this.mapView.getBoundingBox();
            this.mapModel.setBoundingBox(bbox);
        },

        newMenuItemClicked: function() {
            var _this = this;

            if (!confirm("Reset model?")) {
                return;
            }

            $.ajax({
                url: this.rootApiUrls.model + "/create",
                data: "confirm=1",
                type: "POST",
                tryCount: 0,
                retryLimit: 3,
                success: function(data) {
                    if ('message' in data) {
                        _this.displayMessage(data.message);
                    }
                    _this.treeView.reload();
                },
                error: handleAjaxError
            });
        },

        play: function(event) {
            if (this.mapControlView.isPaused()) {
                this.mapControlView.setPlaying();
                this.mapView.play();
            } else {
                this.mapControlView.disableControls();
                this.mapControlView.setPlaying();
                this.runModel();
            }
        },

        pause: function(event) {
            this.mapControlView.setPaused();
            this.mapView.pause();
        },

        enableZoomIn: function(event) {
            if (this.mapModel.hasData() === false) {
                return;
            }

            this.mapControlView.setZoomingIn();
            this.mapView.makeActiveImageClickable();
            this.mapView.makeActiveImageSelectable();
            this.mapView.setZoomingInCursor();
        },

        enableZoomOut: function(event) {
            if (this.mapModel.hasData() === false) {
                return;
            }

            this.mapControlView.setZoomingOut();
            this.mapView.makeActiveImageClickable();
            this.mapView.setZoomingOutCursor();
        },

        restart: function() {
            var frames = this.mapModel.getFrames();

            if (frames === null) {
                this.mapView.restart();
                return;
            }
            this.mapView.refresh(frames);
        },

        runFailed: function(event) {
        },

        refreshFinished: function(event) {
            this.mapControlView.enableControls();
            this.mapControlView.setFrameCount(this.mapView.getFrameCount());
            if (this.mapControlView.isPlaying()) {
                this.mapView.play();
            }
        },

        stopAnimation: function(event) {
            this.mapControlView.setStopped();
        },

        zoomIn: function(event, startPosition, endPosition) {
            this.mapControlView.setStopped();

            if (endPosition) {
                this.zoomFromRect(
                    {start: startPosition, end: endPosition},
                    MapModel.ZOOM_IN
                );
            } else {
                this.zoomFromPoint(startPosition, MapModel.ZOOM_OUT);
            }

            this.mapView.setRegularCursor();
        },

        zoomOut: function(event, point) {
            this.zoomFromPoint(
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
            this.mapControlView.setCurrentFrame(this.mapView.currentFrame);
            this.mapControlView.setTime(timestamp);
        },

        jumpToFirstFrame: function(event) {
            this.mapView.setCurrentFrame(0);
        },

        jumpToLastFrame: function(event) {
            var lastFrame = this.mapView.getFrameCount();
            this.mapView.setCurrentFrame(lastFrame);
            this.mapControlView.setCurrentFrame(lastFrame);
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

            // If this is a top-level node, then its `data.key` value will match a
            // URL in `this.rootApiUrl`. Otherwise it's a child node and its parent
            // (in `node.parent` will have a `key` value  set to 'setting', 'spill'
            // or 'mover' and `node` will have a `data.type` value specific to its
            // server-side representation, e.g. 'constant_wind'.
            if (node.data.key in this.rootApiUrls) {
                urlKey = node.data.key;
            } else if (node.parent.data.key in this.rootApiUrls) {
                urlKey = node.parent.data.key;
            }

            if (!urlKey) {
                return;
            }

            var rootUrl = this.rootApiUrls[urlKey];

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
            if (node.data.key in this.rootApiUrls) {
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

            if (confirm('Remove mover?') == false) {
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
                    alert('Could not remove item.')
                }
            });
        },

        displayMessage: function(message) {
            if (!'type' in message || !'text' in message) {
                return;
            }

            var alertDiv = $('div .alert-' + message.type);

            if (message.text && alertDiv) {
                alertDiv.find('span.message').text(message.text);
                alertDiv.removeClass('hidden');
            }
        },

        showForm: function(url) {
            $.ajax({
                type: 'GET',
                url: url,
                tryCount: 0,
                retryLimit: 3,
                success: this.submitSuccess,
                error: handleAjaxError
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
                error: handleAjaxError
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

        handleFailedModelRun: function(data) {
            if ('message' in data && data.message.type == 'error') {
                this.error = data.message;
            } else {
                this.error = {text: 'The model failed to run. Please try again.'}
            }

            alert(this.mapModel.error.text);
        },

        runModel: function(opts) {
            var _this = this;

            if (this.dirty === false) {
                this.restart();
                return;
            }

            opts = $.extend({
                zoomLevel: this.mapModel.zoomLevel,
                zoomDirection: MapModel.ZOOM_NONE
            }, opts);

            var isInvalid = function(obj) {
                return obj === undefined || obj === null || typeof(obj) != "object";
            };

            if ((opts.zoomLevel != this.mapModel.zoomLevel) &&
                (isInvalid(opts.rect) && isInvalid(opts.point))) {
                alert("Could not zoom. Please try again.");
            }

            $.ajax({
                type: 'POST',
                url: this.rootApiUrls.model + '/run',
                data: opts,
                tryCount: 0,
                retryLimit: 3,
                success: function(data) {
                    if ('message' in data) {
                        _this.handleFailedModelRun(data);
                        return false;
                    }
                    _this.mapModel.dirty = false;
                    _this.mapModel.data = data;
                    _this.restart();
                    return false;
                },
                error: function(data) {
                    _this.handleFailedModelRun(data);
                    return false;
                }
            });
        },

        zoomFromPoint: function(point, direction) {
            this.dirty = true;
            this.runModel({point: point, zoom: direction});
        },

        zoomFromRect: function(rect, direction) {
            var isInsideMap = this.mapModel.isRectInsideMap(rect);

            // If we are at zoom level 0 and there is no map portion outside of
            // the visible area, then adjust the coordinates of the selected
            // rectangle to the on-screen pixel bound.
            if (!isInsideMap && this.mapModel.zoomLevel === 0) {
                rect = this.mapModel.getAdjustedRect(rect);
            }

            this.dirty = true;
            this.runModel({rect: rect, zoom: direction});
        }
    };


    // Export `MapController`
    window.noaa.erd.gnome.MapController = MapController;


    // Wait until the placeholder image has loaded before loading controls.
    $('#map').imagesLoaded(function() {
        new MapController({
            mapEl: '#map',
            sidebarEl: '#sidebar',
            'formContainerEl': '#modal-container'
        });
    });
})();
