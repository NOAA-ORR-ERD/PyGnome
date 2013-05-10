define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'views/map'
], function($, _, Backbone, models, map) {
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
            this.restingButtonEl = this.options.restingButtonEl;
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
            this.stepGenerator = this.options.stepGenerator;
            this.model = this.options.model;

            this.animationControls = [
                this.backButtonEl, this.forwardButtonEl, this.playButtonEl,
                this.pauseButtonEl
            ];

            this.mapControls = [
                this.restingButtonEl, this.moveButtonEl, this.zoomInButtonEl,
                this.zoomOutButtonEl, this.spillButtonEl
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

            this.state = this.options.state;
            this.listenTo(this.state, 'change:animation', this.renderAnimationControls);
            this.listenTo(this.state, 'change:cursor', this.renderCursorControls);
            this.listenTo(this.state, 'change:fullscreen', this.renderFullscreenControls);

            $(this.sliderEl).slider({
                start: this.sliderStarted,
                change: this.sliderChanged,
                slide: this.sliderMoved,
                disabled: true
            });

            if (this.stepGenerator.expectedTimeSteps.length) {
                this.setTimeSteps(this.stepGenerator.expectedTimeSteps);
                this.enableControls();
            }

            this.setPaused();
            this.setupClickEvents();
            this.updateCachedPercentage();

            this.stepGenerator.on(models.StepGenerator.RUN_BEGAN, this.runBegan);
            this.stepGenerator.on(models.StepGenerator.RUN_ERROR, this.stepGeneratorError);
            this.stepGenerator.on(models.StepGenerator.NEXT_TIME_STEP_READY, this.updateCachedPercentage);
            this.stepGenerator.on(models.StepGenerator.RUN_FINISHED, this.stepGeneratorFinished);
            this.stepGenerator.on(models.StepGenerator.CREATED, this.modelCreated);

            this.options.mapView.on(map.MapView.FRAME_CHANGED, this.mapViewFrameChanged);
        },

        renderAnimationControls: function(animationState) {
            switch (animationState) {
                case models.AnimationState.STOPPED:
                    this.setStopped();
                    break;
                case models.AnimationState.PLAYING:
                    this.setPlaying();
                    break;
                case models.AnimationState.PAUSED:
                    this.setPaused();
                    break;
            }
        },

        renderCursorControls: function(cursorState) {
            switch(cursorState) {
                case models.CursorState.ZOOMING_IN:
                    this.setZoomingIn();
                    break;
                case models.CursorState.ZOOMING_OUT:
                    this.setZoomingOut();
                    break;
                case models.CursorState.RESTING:
                    this.setResting();
                    break;
                case models.CursorState.MOVING:
                    this.setMoving();
                    break;
                case models.CursorState.DRAWING_SPILL:
                    this.setDrawingSpill();
                    break;
            }
        },

        renderFullscreenControls: function(fullscreenState) {
            switch (fullscreenState) {
                case models.FullscreenState.DISABLED:
                    this.switchToNormalScreen();
                    break;
                case models.FullscreenState.ENABLED:
                    this.switchToFullscreen();
                    break;
            }
        },

        setupClickEvents: function() {
            var _this = this;

            var clickEvents = [
                [this.restingButtonEl, MapControlView.HAND_BUTTON_CLICKED],
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

            // TODO: Use a named method here instead of a closure?
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
            var timestamp = this.stepGenerator.getTimestampForExpectedStep(ui.value);

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
                100*(this.stepGenerator.length / this.stepGenerator.expectedTimeSteps.length))
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
            if (this.stepGenerator.dirty) {
                // TODO: Is this really what we want to do here?
                this.reset();
            }

            this.setTimeSteps(this.stepGenerator.expectedTimeSteps);
        },

        mapViewFrameChanged: function() {
            var timeStep = this.stepGenerator.getCurrentTimeStep();
            this.setTimeStep(timeStep.id);
            this.setTime(timeStep.get('timestamp'));
        },

        stepGeneratorError: function() {
            this.disableControls();
        },

        stepGeneratorFinished: function() {
            this.disableControls();
            this.enableControls(this.playButtonEl);
        },

        modelCreated: function() {
            this.reset();
        },

        setStopped: function() {
            $(this.pauseButtonEl).hide();
            $(this.playButtonEl).show();
            this.enableControls();
        },

        setPlaying: function() {
            $(this.playButtonEl).hide();
            $(this.pauseButtonEl).show();
            this.disableControls();
            this.enableControls(this.pauseButtonEl);
        },

        setPaused: function() {
            $(this.pauseButtonEl).hide();
            $(this.playButtonEl).show();
            this.enableControls();
        },

        setZoomingIn: function() {
            this.deactivateControl();
            this.activateControl(this.zoomInButtonEl);
        },

        setZoomingOut: function() {
            this.deactivateControl();
            this.activateControl(this.zoomOutButtonEl);
        },

        setResting: function() {
            this.deactivateControl();
            this.activateControl(this.restingButtonEl);
        },

        setMoving: function() {
            this.deactivateControl();
            this.activateControl(this.moveButtonEl);
        },

        setDrawingSpill: function() {
            this.deactivateControl();
            this.activateControl(this.spillButtonEl);
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

            if (controls && typeof controls !== 'string' && controls.length) {
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

        toggleControlClass: function(controls, cls, toggle) {
            if (controls === undefined) {
                controls = this.mapControls;
            }

            function doToggle(control, toggle) {
                control = $(control);
                if (toggle === MapControlView.OFF) {
                    control.removeClass(cls);
                } else {
                    control.addClass(cls);
                }
            }

            if (controls && typeof controls !== 'string' && controls.length) {
                for (var i = 0; i < controls.length; i++) {
                    doToggle(controls[i], toggle)
                }
                return;
            }

            doToggle(controls);
        },

        activateControl: function(control) {
            this.toggleControlClass(control, 'active', MapControlView.ON);
        },

        deactivateControl: function(control) {
            this.toggleControlClass(control, 'active', MapControlView.OFF);
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
            this.setTimeStep(0);
            $(this.sliderEl).slider('values', null);
            this.enableControls([this.playButtonEl]);
        }
    }, {
        // Constants
        ON: true,
        OFF: false,

        // Event constants
        HAND_BUTTON_CLICKED: "mapControlView:restingButtonClicked",
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
        SLIDER_MOVED: "mapControlView:sliderMoved"
    });

    return {
        MapControlView: MapControlView
    }
});