
define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'util'
], function($, _, Backbone, util) {

     /*
     `TimeStep` represents a single time step of the user's actively-running
     model on the server.
     */
    var TimeStep = Backbone.Model.extend({
        get: function(attr) {
            var value = Backbone.Model.prototype.get.call(this, attr);

            if (attr === 'timestamp') {
                value = util.formatTimestamp(value);
            }

            return value;
        }
    });


    /*
     `ModelRun` is a collection of `TimeStep` objects representing a run of
     the user's active model.
     */
    var ModelRun = Backbone.Collection.extend({
        model: TimeStep,

        initialize: function(timeSteps, opts) {
            _.bindAll(this);
            this.url = opts.url;
            this.currentTimeStep = opts.currentTimeStep || 0;
            this.nextTimeStep = this.currentTimeStep ? this.currentTimeStep + 1 : 0;
            // An array of timestamps, one for each step we expect the server to
            // make, passed back when we initiate a model run.
            this.expectedTimeSteps = opts.expectedTimeSteps || [];
            this.bounds = opts.bounds;
            // Optionally specify the zoom level.
            this.zoomLevel = opts.zoomLevel === undefined ? 4 : opts.zoomLevel;
            // If true, `Model` will request a new set of time steps from the server
            // on the next run. Assume we want to do this by default (i.e., at
            // construction time) if there are no time steps.
            this.dirty = timeSteps.length === 0;

            // When initializing the model at the last time step of a generated
            // series, rewind to the beginning so the user can play the series
            // again.
            if (this.isOnLastTimeStep()) {
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
                timestamp = util.formatTimestamp(this.expectedTimeSteps[stepNum]);
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
            var message = util.parseMessage(data);

            if (message) {
                this.trigger(ModelRun.MESSAGE_RECEIVED, message);

                if (message.error) {
                    this.trigger(ModelRun.RUN_ERROR);
                    return false;
                }
            }

            this.dirty = false;
            this.expectedTimeSteps = data.expected_time_steps;
            this.bounds = data.map_bounds;

            if (_.has(data, 'time_step')) {
                this.addTimeStep(data.time_step)                ;
            }

            this.trigger(ModelRun.RUN_BEGAN, data);
            this.getNextTimeStep();
            return true;
        },

        /*
         Helper that performs an AJAX request to start ("run") the model.

         Receives back the background image for the map and an array of timestamps,
         one for each step the server expects to generate on subsequent requests.
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
                error: util.handleAjaxError
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
                zoomDirection: ModelRun.ZOOM_NONE,
                runUntilTimeStep: this.runUntilTimeStep
            }, opts);

            var needToGetRunUntilStep = false;

            if (options.runUntilTimeStep) {
                this.runUntilTimeStep = options.runUntilTimeStep;
                needToGetRunUntilStep = options.runUntilTimeStep &&
                    !this.hasCachedTimeStep(options.runUntilTimeStep);
            }

            if (this.dirty) {
                options['no_cache'] = true;
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
         - `Model.RUN_FINISHED` if the model has run until `this.runUntilTimeStep`.
         */
        setCurrentTimeStep: function(stepNum) {
            this.currentTimeStep = stepNum;
            this.nextTimeStep = stepNum + 1;

            if (this.currentTimeStep === this.runUntilTimeStep ||
                    this.currentTimeStep === _.last(this.expectedTimeSteps)) {
                this.trigger(ModelRun.RUN_FINISHED);
                this.runUntilTimeStep = null;
                return;
             }

             this.trigger(ModelRun.NEXT_TIME_STEP_READY, this.getCurrentTimeStep());
        },

        isOnLastTimeStep: function() {
            return this.currentTimeStep === this.length - 1;
        },

         /*
         Finish the current run.

         Triggers:
         - `Model.RUN_FINISHED`
         */
        finishRun: function() {
            this.rewind();
            this.runUntilTimeStep = null;
            this.trigger(ModelRun.RUN_FINISHED);
        },

        /*
         Makes a request to the server for the next time step.

         Triggers:
         - `Model.RUN_FINISHED` if the server has no more time steps to run.
         */
        getNextTimeStep: function() {
            if (!this.serverHasTimeStep(this.nextTimeStep)) {
                this.finishRun();
                return;
            }

            // The time step has already been generated and we have it.
            if (this.hasCachedTimeStep(this.nextTimeStep)) {
                this.setCurrentTimeStep(this.nextTimeStep);
                return;
            }

            // Request the next step from the server.
            $.ajax({
                type: "GET",
                url: this.url + '/next_step',
                success: this.timeStepRequestSuccess,
                error: this.timeStepRequestFailure
            });
        },

        timeStepRequestSuccess: function(data) {
            var message = util.parseMessage(data);

            if (message) {
                this.trigger(ModelRun.MESSAGE_RECEIVED, message);

                if (message.error) {
                    this.trigger(ModelRun.RUN_ERROR);
                    return;
                }
            }

            if (!data.time_step) {
                this.trigger(ModelRun.RUN_ERROR);
                return;
            }

            this.addTimeStep(data.time_step);
       },

       timeStepRequestFailure: function(xhr, textStatus, errorThrown) {
           if (xhr.status === 500) {
               // TODO: Inform user of more information.
               alert('The run failed due to a server-side error.');
           } if (xhr.status === 404) {
               // TODO: Maybe we shouldn't return 404 when finished? Seems wrong.
               alert('The model you were working with is no longer available.');
           }
           this.finishRun();
           util.log(xhr);
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
            this.dirty = true;
            this.rewind();
            this.reset();
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
                error: util.handleAjaxError
            });
        },

         /*
         Handle a successful request to the server to create a new model.
         */
        createSuccess: function(data) {
            var message = util.parseMessage(data);

            if (message) {
                this.trigger(ModelRun.MESSAGE_RECEIVED, message);

                if (message.error) {
                    // TODO: Is this the error event we want to use here?
                    this.trigger(ModelRun.RUN_ERROR);
                    return;
                }
            }

            this.clearData();
            this.dirty = true;
            this.trigger(ModelRun.CREATED);
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


    var BaseModel = Backbone.Model.extend({
        parse: function(response) {
            var message = util.parseMessage(response);
            if (message) {
                this.trigger(this.prototype.MESSAGE_RECEIVED, message);
            }
            return BaseModel.__super__.parse.apply(this, arguments);
        }
    }, {
        MESSAGE_RECEIVED: 'ajaxForm:messageReceived'
    });


    var Model = BaseModel.extend({
        initialize: function(attrs, opts) {
            this.url = opts.url;
        }
    });


    // Spills

    var PointReleaseSpill = BaseModel.extend({});


    var PointReleaseSpillCollection = Backbone.Collection.extend({
        model: PointReleaseSpill,

        initialize: function(spills, opts) {
            this.url = opts.url;
        }
    });


    // Movers
    var WindValue = BaseModel.extend({});

    var WindValueCollection = Backbone.Collection.extend({
        model: WindValue
    });

    var Wind = BaseModel.extend({
        initialize: function(attrs) {
            var timeseries = [];
            // TODO: Move into a utility function.
            if (attrs && attrs.hasOwnProperty('timeseries')) {
                timeseries = attrs['timeseries'];
                delete(attrs['timeseries']);
            }
            this.set('timeseries', new WindValueCollection(timeseries));
        },

        toJSON: function() {
            var attrs = this.attributes;
            var timesteps = attrs['timeseries'];
            attrs['timeseries'] = timesteps.sortBy(function(item) {
                return item.datetime;
            });
            return attrs;
        }
    });

    var WindMover = BaseModel.extend({
        initialize: function(attrs) {
            var wind = {};
            if (attrs && attrs.hasOwnProperty('wind')) {
                wind = attrs['wind'];
                delete(attrs['wind']);
            }
            this.set('wind', new Wind(wind));
        },

        // Return a `moment` object for the date field.
        get: function(attr) {
            if (attr === 'date') {
                return moment(this.attributes[attr]).local();
            } else {
                return WindMover.__super__.get.apply(this, arguments);
            }
        }
    });

    var WindMoverCollection = Backbone.Collection.extend({
        model: WindMover,

        initialize: function(movers, opts) {
            this.url = opts.url;
        }
    });


    return {
        TimeStep: TimeStep,
        ModelRun: ModelRun,
        Model: Model,
        PointReleaseSpill: PointReleaseSpill,
        PointReleaseSpillCollection: PointReleaseSpillCollection,
        WindMover: WindMover,
        WindMoverCollection: WindMoverCollection
    };

});
