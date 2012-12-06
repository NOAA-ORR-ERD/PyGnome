
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
                value = util.getUTCStringForTimestamp(value);
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
                timestamp = util.getUTCStringForTimestamp(
                    this.expectedTimeSteps[stepNum]);
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
         - `Model.RUN_FINISHED` if the model has run until `this.runUntilTimeStep`.
         */
        setCurrentTimeStep: function(stepNum) {
            this.currentTimeStep = stepNum;
            this.nextTimeStep = stepNum + 1;

            if (this.currentTimeStep === this.runUntilTimeStep ||
                    this.currentTimeStep === _.last(this.expectedTimeSteps)) {
                this.trigger(Model.RUN_FINISHED);
                this.runUntilTimeStep = null;
                return;
             }

             this.trigger(Model.NEXT_TIME_STEP_READY, this.getCurrentTimeStep());
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
            this.trigger(Model.RUN_FINISHED);
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

       timeStepRequestFailure: function(xhr, textStatus, errorThrown) {
           if (xhr.status === 404) {
               // TODO: Maybe we shouldn't return 404 when finished? Seems wrong.
               this.finishRun();
           }
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
                error: util.handleAjaxError
            });
        },

         /*
         Handle a successful request to the server to create a new model.
         */
        createSuccess: function(data) {
            var message = util.parseMessage(data);

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
    AjaxForm.SUCCESS = 'ajaxForm:success';

    AjaxForm.prototype = {
        /*
         Refresh this form from the server's JSON response.
         */
        parse: function(response) {
            var message = util.parseMessage(response);
            if (message) {
                this.trigger(AjaxForm.MESSAGE_RECEIVED, message);
            }

            if (_.has(response, 'form_html') && response.form_html) {
                this.form_html = response.form_html;
                this.trigger(AjaxForm.CHANGED, this);
            } else {
                this.trigger(AjaxForm.SUCCESS, this);
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
                error: util.handleAjaxError
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
            util.log(this)
             var options = $.extend({}, opts, {
                type: 'POST'
            });

            this.makeRequest(options);
        }
    };


    /*
     A collection of `AjaxForm` instances.

     Listen for SUBMIT_SUCCESS and SUBMIT_ERROR events on all instances and
     rebroadcast them.
     */
    var AjaxFormCollection = function() {
        _.bindAll(this);
        _.extend(this, Backbone.Events);
        this.forms = {};
    };


    AjaxFormCollection.prototype = {
        add: function(formOpts) {
            var _this = this;

            if (!_.has(formOpts, 'collection')) {
                formOpts.collection = this;
            }

            this.forms[formOpts.id] = new AjaxForm(formOpts);

            this.forms[formOpts.id].on(AjaxForm.CHANGED,  function(ajaxForm) {
                _this.trigger(AjaxForm.CHANGED, ajaxForm);
            });

            this.forms[formOpts.id].on(AjaxForm.SUCCESS,  function(ajaxForm) {
                _this.trigger(AjaxForm.SUCCESS, ajaxForm);
            });
        },

        get: function(id) {
            return this.forms[id];
        },

        deleteAll: function() {
            var _this = this;
            _.each(this.forms, function(form, key) {
                delete _this.forms[key];
            });
        }
    };

    return {
        TimeStep: TimeStep,
        Model: Model,
        AjaxForm: AjaxForm,
        AjaxFormCollection: AjaxFormCollection
    };

});
