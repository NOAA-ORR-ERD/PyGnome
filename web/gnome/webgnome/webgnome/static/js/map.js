
// Get an alias to the `window.noaa.erd.gnome` namespace.
gnome = window.noaa.erd.gnome;

gnome.MapModel = function (opts) {
    this.bbox = opts.bbox;

    // Optionally specify the current frame the user is in.
    this.frame = opts.frame == undefined ? 0 : opts.frame;

    // Optionally specify the zoom level of the map.
    this.zoomLevel = opts.zoomLevel == undefined ? 4 : opts.zoomLevel;

    this.data = null;

    // If true, `MapModel` will request a new set of frames from the server
    // when the user runs the model.
    this.dirty = true;
};


// `MapModel` events
gnome.MapModel.RUN_FINISHED = 'gnome:runFinished';
gnome.MapModel.RUN_FAILED = 'gnome:runFailed';

gnome.MapModel.RUN_URL = '/model/run';
gnome.MapModel.ZOOM_IN = 'zoom_in';
gnome.MapModel.ZOOM_OUT = 'zoom_out';
gnome.MapModel.ZOOM_NONE = 'zoom_none'


gnome.MapModel.prototype = {
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

    isRectInsideMap: function (rect) {
        var _rect = this.getRect(rect);

        return this.isPositionInsideMap(_rect.start) &&
               this.isPositionInsideMap(_rect.end);
    },

    run: function(opts) {
        var _this = this;

        if (this.dirty === false) {
            $(_this).trigger(gnome.MapModel.RUN_FINISHED);
            return;
        }

        opts = $.extend(opts, {
            zoomLevel: this.zoomLevel,
            zoomDirection: gnome.MapModel.ZOOM_NONE
        });

        var isInvalid = function(obj) {
            return obj === undefined || obj === null || typeof(obj) != "object"
        };

        if ((opts.zoomLevel != this.zoomLevel) &&
            (isInvalid(opts.rect) && isInvalid(opts.point))) {
            alert("Could not zoom. Please try again.");
        }

        $.ajax({
            type: 'POST',
            url: gnome.MapModel.RUN_URL,
            data: opts,
            success: function(data) {
                _this.dirty = false;
                _this.data = data;
                $(_this).trigger(gnome.MapModel.RUN_FINISHED);
            },
            error: function(data) {
                _this.error = data;
                $(_this).trigger(gnome.MapModel.RUN_FAILED);
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
    }
};


gnome.MapView = function(opts) {
    this.mapEl = opts.mapEl;
    this.frameClass = opts.frameClass;
    this.activeFrameClass = opts.activeFrameClass;
    this.currentFrame = 0;
};

// `MapView` events
gnome.MapView.DRAGGING_FINISHED = 'gnome:draggingFinished';
gnome.MapView.REFRESH_FINISHED = 'gnome:refreshFinished';
gnome.MapView.PLAYING_FINISHED = 'gnome:playingFinished';
gnome.MapView.FRAME_CHANGED = 'gnome:frameChanged';
gnome.MapView.MAP_WAS_CLICKED = 'gnome:mapWasClicked';


gnome.MapView.prototype = {
    initialize: function() {
        // Only have to do this once.
        this.makeImagesClickable();
        return this;
    },

    makeImagesClickable: function() {
        var _this = this;
        $(this.mapEl).on('click', 'img', function(event){
            if ($(this).data('clickEnabled')) {
                $(_this).trigger(
                    gnome.MapView.MAP_WAS_CLICKED,
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
            start: function (event) {
                _this.startPosition = {x: event.pageX, y: event.pageY};
            },
            stop: function (event) {
                if (!$(this).selectable('option', 'disabled')) {
                    $(_this).trigger(
                        gnome.MapView.DRAGGING_FINISHED,
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
                $(_this).trigger(gnome.MapView.PLAYING_FINISHED);
            },
            before: function(currSlideElement, nextSlideElement, options, forwardFlag) {
                $(currSlideElement).removeClass('active');
                $(nextSlideElement).addClass('active');
            },
            after: function(currSlideElement, nextSlideElement, options, forwardFlag) {
                _this.currentFrame = $(nextSlideElement).attr('data-position');
                $(_this).trigger(gnome.MapView.FRAME_CHANGED);
            }
        });

        this.cycle.cycle('pause')
        $(this.mapEl).show();

        $(this).trigger(gnome.MapView.REFRESH_FINISHED);
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

    getSize: function () {
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


gnome.TreeView = function(opts) {
    this.treeEl = opts.treeEl;
    return this;
};

gnome.TreeView.prototype = {
    initialize: function() {
        $(this.treeEl).dynatree({
            onActivate:function (node) {
                console.log(node);
            },
            persist:true
        });

        return this;
    }
};


gnome.AnimationControlView = function(opts) {
    this.sliderEl = opts.sliderEl;
    this.playButtonEl = opts.playButtonEl;
    this.pauseButtonEl = opts.pauseButtonEl;
    this.backButtonEl = opts.backButtonEl;
    this.forwardButtonEl = opts.forwardButtonEl;
    this.zoomInButtonEl = opts.zoomInButtonEl;
    this.zoomOutButtonEl = opts.zoomOutButtonEl;
    this.moveButtonEl = opts.moveButtonEl;
    this.timeEl = opts.timeEl;
    return this;
};

// Events for `AnimationControlView`
gnome.AnimationControlView.PLAY_BUTTON_CLICKED = "gnome:playButtonClicked";
gnome.AnimationControlView.PAUSE_BUTTON_CLICKED = "gnome:pauseButtonClicked";
gnome.AnimationControlView.BACK_BUTTON_CLICKED = "gnome:backButtonClicked";
gnome.AnimationControlView.FORWARD_BUTTON_CLICKED = "gnome:forwardButtonClicked";
gnome.AnimationControlView.ZOOM_IN_BUTTON_CLICKED = "gnome:zoomInButtonClicked";
gnome.AnimationControlView.ZOOM_OUT_BUTTON_CLICKED = "gnome:zoomOutButtonClicked";
gnome.AnimationControlView.MOVE_BUTTON_CLICKED = "gnome:moveButtonClicked";
gnome.AnimationControlView.SLIDER_CHANGED = "gnome:sliderChanged";

// Statuses
gnome.AnimationControlView.STATUS_STOPPED = 0;
gnome.AnimationControlView.STATUS_PLAYING = 1;
gnome.AnimationControlView.STATUS_PAUSED = 2;
gnome.AnimationControlView.STATUS_BACK = 3;
gnome.AnimationControlView.STATUS_FORWARD = 4;
gnome.AnimationControlView.STATUS_ZOOMING_IN = 5;
gnome.AnimationControlView.STATUS_ZOOMING_OUT = 6;

gnome.AnimationControlView.prototype = {
    initialize: function() {
        var _this = this;
        this.status = gnome.AnimationControlView.STATUS_STOPPED;

        $(this.sliderEl).slider({
            start: function(event, ui) {
                $(_this).trigger(gnome.AnimationControlView.PAUSE_BUTTON_CLICKED);
            },
            change: function(event, ui) {
                $(_this).trigger(gnome.AnimationControlView.SLIDER_CHANGED, ui.value);
            },
            disabled: true
        });

        $(this.playButtonEl).click(function () {
            $(_this).trigger(gnome.AnimationControlView.PLAY_BUTTON_CLICKED);
        });

        $(this.pauseButtonEl).click(function () {
            if (_this.status === gnome.AnimationControlView.STATUS_PLAYING) {
                $(_this).trigger(gnome.AnimationControlView.PAUSE_BUTTON_CLICKED);
            }
        });

        $(this.backButtonEl).click(function () {
            $(_this).trigger(gnome.AnimationControlView.BACK_BUTTON_CLICKED);
        });

        $(this.forwardButtonEl).click(function () {
            $(_this).trigger(gnome.AnimationControlView.FORWARD_BUTTON_CLICKED);
        });

        $(this.zoomInButtonEl).click(function () {
            $(_this).trigger(gnome.AnimationControlView.ZOOM_IN_BUTTON_CLICKED);
        });

        $(this.zoomOutButtonEl).click(function () {
            $(_this).trigger(gnome.AnimationControlView.ZOOM_OUT_BUTTON_CLICKED);
        });

        $(this.moveButtonEl).click(function () {
            $(_this).trigger(gnome.AnimationControlView.MOVE_BUTTON_CLICKED);
        });

        return this;
    },

    setStopped: function() {
        this.status = gnome.AnimationControlView.STATUS_STOPPED;
    },

    setPlaying: function() {
        this.status = gnome.AnimationControlView.STATUS_PLAYING;
    },

    setPaused: function() {
        this.status = gnome.AnimationControlView.STATUS_PAUSED;
    },

    setForward: function() {
        this.status = gnome.AnimationControlView.STATUS_FORWARD;
    },
    
    setBack: function() {
        this.status = gnome.AnimationControlView.STATUS_BACK;
    },

    setZoomingIn: function() {
        this.status = gnome.AnimationControlView.STATUS_ZOOMING_IN;
    },

    setZoomingOut: function() {
        this.status = gnome.AnimationControlView.STATUS_ZOOMING_OUT;
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
    
    isPlaying: function() {
        return this.status === gnome.AnimationControlView.STATUS_PLAYING;
    },

    isStopped: function() {
        return this.status === gnome.AnimationControlView.STATUS_STOPPED;
    },

    isPaused: function() {
        return this.status === gnome.AnimationControlView.STATUS_PAUSED;
    },

    isForward: function() {
        return this.status === gnome.AnimationControlView.STATUS_PLAYING;
    },

    isBack: function() {
        return this.status === gnome.AnimationControlView.STATUS_BACK;
    },

    isZoomingIn: function() {
        return this.status === gnome.AnimationControlView.STATUS_ZOOMING_IN;
    },

    isZoomingOut: function() {
        return this.status === gnome.AnimationControlView.STATUS_ZOOMING_OUT;
    },

    enableControls: function() {
        $(this.sliderEl).slider('option', 'disabled', false);
        _.each([this.backButtonEl, this.forwardButtonEl, this.playButtonEl,
            this.pauseButtonEl, this.moveButtonEl], function (buttonEl) {
            $(buttonEl).removeClass('disabled');
        });
    },

    disableControls: function() {
        $(this.sliderEl).slider('option', 'disabled', true);
        _.each([this.backButtonEl, this.forwardButtonEl, this.playButtonEl,
            this.pauseButtonEl, this.moveButtonEl], function (item) {
            $(item).removeClass('enabled');
        });
    }
};


gnome.MapController = function(opts) {
    var _this = this;

    _.bindAll(this);

    this.treeView = new gnome.TreeView({
        treeEl: "#tree"
    }).initialize();

    this.mapView = new gnome.MapView({
        mapEl: opts.mapEl,
        frameClass: 'frame',
        activeFrameClass: 'active'
    }).initialize();

    this.animationControlView = new gnome.AnimationControlView({
        sliderEl: "#slider",
        playButtonEl: "#play-button",
        pauseButtonEl: "#pause-button",
        backButtonEl: "#back-button",
        forwardButtonEl: "#forward-button",
        zoomInButtonEl: "#zoom-in-button",
        zoomOutButtonEl: "#zoom-out-button",
        moveButtonEl: "#move-button",
        timeEl: "#time"
    }).initialize();

    this.mapModel = new gnome.MapModel({
        // XXX: Get bbox from the server?
        bbox: this.mapView.getBoundingBox()
    });

    // Event handlers
    $(this.animationControlView).bind(
        gnome.AnimationControlView.PLAY_BUTTON_CLICKED, this.play);
    $(this.animationControlView).bind(
        gnome.AnimationControlView.PAUSE_BUTTON_CLICKED, this.pause);
    $(this.animationControlView).bind(
        gnome.AnimationControlView.ZOOM_IN_BUTTON_CLICKED, this.enableZoomIn);
    $(this.animationControlView).bind(
        gnome.AnimationControlView.ZOOM_OUT_BUTTON_CLICKED, this.enableZoomOut);
    $(this.animationControlView).bind(
        gnome.AnimationControlView.SLIDER_CHANGED, this.sliderChanged);
    $(this.mapModel).bind(gnome.MapModel.RUN_FINISHED, this.restart);
    $(this.mapView).bind(gnome.MapView.REFRESH_FINISHED, this.refreshFinished);
    $(this.mapView).bind(gnome.MapView.PLAYING_FINISHED, this.stopAnimation);
    $(this.mapView).bind(gnome.MapView.DRAGGING_FINISHED, this.zoomIn);
    $(this.mapView).bind(gnome.MapView.FRAME_CHANGED, this.frameChanged);
    $(this.mapView).bind(gnome.MapView.MAP_WAS_CLICKED, this.zoomOut);

    return this;
};

gnome.MapController.prototype = {
    play: function (event) {
        if (this.animationControlView.isPaused()) {
            this.animationControlView.setPlaying();
            this.mapView.play();
        } else {
            this.animationControlView.disableControls();
            this.animationControlView.setPlaying();
            this.mapModel.run();
        }
    },

    pause: function (event) {
        this.animationControlView.setPaused();
        this.mapView.pause();
    },

    enableZoomIn: function (event) {
        if (this.mapModel.hasData() === false) {
            return;
        }

        this.animationControlView.setZoomingIn();
        this.mapView.makeActiveImageClickable();
        this.mapView.makeActiveImageSelectable();
        this.mapView.setZoomingInCursor();
    },

    enableZoomOut: function (event) {
        if (this.mapModel.hasData() === false) {
            return;
        }

        this.animationControlView.setZoomingOut();
        this.mapView.makeActiveImageClickable();
        this.mapView.setZoomingOutCursor();
    },

    restart: function (event) {
        var frames = this.mapModel.getFrames();

        if (frames === null) {
            this.mapView.restart();
            return;
        }
        this.mapView.refresh(frames);
    },

    refreshFinished: function (event) {
        this.animationControlView.enableControls();
        this.animationControlView.setFrameCount(this.mapView.getFrameCount());
        if (this.animationControlView.isPlaying()) {
            this.mapView.play();
        }
    },

    stopAnimation: function (event) {
        this.animationControlView.setStopped();
    },

    zoomIn: function(event, startPosition, endPosition) {
        this.animationControlView.setStopped();

        if (endPosition) {
             this.mapModel.zoomFromRect(
                {start: startPosition, end: endPosition},
                gnome.MapModel.ZOOM_IN
            );
        } else {
            this.mapModel.zoomFromPoint(
                startPosition,
                gnome.MapModel.ZOOM_OUT
            );
        }

        this.mapView.setRegularCursor();
    },

    zoomOut: function(event, point) {
        this.mapModel.zoomFromPoint(
            point, gnome.MapModel.ZOOM_OUT
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
    }
};


$('#map').imagesLoaded(function() {
    new gnome.MapController({
        mapEl: '#map'
    });
});

