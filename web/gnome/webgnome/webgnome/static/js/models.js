
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
     `GnomeRun` is a collection of `TimeStep` objects representing a run of
     the user's active Gnome model.
     */
    var GnomeRun = Backbone.Collection.extend({
        model: TimeStep,

        initialize: function(timeSteps, opts) {
            _.bindAll(this, 'runSuccess', 'timeStepRequestFailure');

            this.url = opts.url + '/runner';
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
                this.trigger(GnomeRun.MESSAGE_RECEIVED, message);

                if (message.error) {
                    this.trigger(GnomeRun.RUN_ERROR);
                    return false;
                }
            }

            this.dirty = false;

            var isFirstStep = data.time_step.id === 0;

            // The Gnome model was reset on the server without our knowledge,
            // so reset the client-side model to stay in sync.
            if (isFirstStep && this.currentTimeStep) {
                this.rewind();
                this.trigger(GnomeRun.SERVER_RESET);
            }

            this.addTimeStep(data.time_step);

            if (isFirstStep) {
                this.expectedTimeSteps = data.expected_time_steps;
                this.trigger(GnomeRun.RUN_BEGAN, data);
            }

            return true;
        },

        /*
         Helper that performs an AJAX request to start ("run") the model.

         Receives an array of timestamps, one for each step the server expects
         to generate on subsequent requests.
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
                url: this.url,
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
                zoomDirection: GnomeRun.ZOOM_NONE,
                runUntilTimeStep: this.runUntilTimeStep
            }, opts);

            this.runUntilTimeStep = options.runUntilTimeStep || null;

            if (this.dirty || this.runUntilTimeStep) {
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
            var now = new Date().getTime();
            var requestBegan = this.timeStepRequestBegin || now;
            timeStep.set('requestTime', now - requestBegan);
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
                this.trigger(GnomeRun.RUN_FINISHED);
                this.runUntilTimeStep = null;
                return;
             }

             this.trigger(GnomeRun.NEXT_TIME_STEP_READY, this.getCurrentTimeStep());
        },

        isOnLastTimeStep: function() {
            return this.currentTimeStep === this.expectedTimeSteps.length - 1;
        },

         /*
         Finish the current run.

         Triggers:
         - `Model.RUN_FINISHED`
         */
        finishRun: function() {
            this.rewind();
            this.runUntilTimeStep = null;
            this.trigger(GnomeRun.RUN_FINISHED);
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

            this.timeStepRequestBegin = new Date().getTime();

            // Request the next step from the server.
            $.ajax({
                type: "GET",
                url: this.url,
                success: this.runSuccess,
                error: this.timeStepRequestFailure
            });
        },

       timeStepRequestFailure: function(xhr, textStatus, errorThrown) {
           if (xhr.status === 500) {
               // TODO: Inform user of more information.
               alert('The run failed due to a server-side error.');
           } if (xhr.status === 404) {
               // The run finished. We already check if the server is expected
               // to have a time step before th in a local cache of
               // expected time steps for the run, so we should not reach
               // this point in normal operation. That is, assuming the local
               // cache of time steps matches the server's -- which it always
               // should.
               this.finishRun();
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
        }
    }, {
        // Class constants
        ZOOM_IN: 'zoom_in',
        ZOOM_OUT: 'zoom_out',
        ZOOM_NONE: 'zoom_none',

        // Class events
        CREATED: 'model:Created',
        RUN_BEGAN: 'model:gnomeRunBegan',
        RUN_FINISHED: 'model:gnomeRunFinished',
        RUN_ERROR: 'model:runError',
        NEXT_TIME_STEP_READY: 'model:nextTimeStepReady',
        MESSAGE_RECEIVED: 'model:messageReceived',
        SERVER_RESET: 'model:serverReset'
    });


    var BaseModel = Backbone.Model.extend({
        /*
         Add an array of field names here that should be converted to strings
         during `toJSON` calls and to `moment` objects during `get` calls.
         */
        dateFields: null,

        initialize: function(attrs, opts) {
            this.dirty = false;
            this.bind('change', this.change, this);
            this.bind('change:id', this.onIndexChange, this);

            if (opts && opts.gnomeModel) {
                this.gnomeModel = opts.gnomeModel;
            }
            BaseModel.__super__.initialize.apply(this, arguments)
        },

        onIndexChange: function() {
            if (this.collection) {
                this.collection.sort({silent: true});
            }
        },

        /*
         Keep an array field on the model in sync with a field that represents
         one item in the array.

         E.g. given a `start_position` field that is an array, the model might
         have a `start_position_x` field that we use for easier data binding
         in views. The following usage of `syncArrayField` will keep
         `start_position` up to date with the latest value of `start_position_x`:
         at index 0 and vice versa.

                syncArrayField('start_position', 'start_position_x', 0);
         */
        syncArrayField: function(arrayFieldName, arrayItemFieldName, index) {

            function setArrayField(model) {
                var arrayField = model.get(arrayFieldName);
                arrayField[index] = model.get(arrayItemFieldName);
                this.set(arrayFieldName, arrayField);
            }

            function setArrayItemField(model) {
                var arrayField = model.get(arrayFieldName);
                model.set(arrayItemFieldName, arrayField[index]);
            }

            this.on('change:' + arrayItemFieldName, setArrayField);
            this.on('change:' + arrayFieldName, setArrayItemField);

            var arrayField = this.get(arrayFieldName);

            if (arrayField && arrayField.length) {
                setArrayItemField(this);
            }
        },

        change: function() {
            this.dirty = true;
        },

        save: function(attrs, options) {
            options = options || {};
            var _this = this;

            if (!_.has(options, 'wait')) {
                options.wait = true;
            }

            if (_.has(options, 'success')) {
                var success = options.success;

                options.success = function(model, response, options) {
                    _this.success(model, response, options);
                    success(model, response, options);
                }
            } else {
                options.success = this.success;
            }

            if (!_.has(options, 'error')) {
                options.error = this.error;
            }

            return BaseModel.__super__.save.apply(this, [attrs, options]);
        },

        success: function(model, response, options) {
            model.errors = null;
            model.dirty = false;
        },

        destroy: function(options) {
            options = options || {};

            if (!_.has(options, 'wait')) {
                options.wait = true;
            }

            BaseModel.__super__.destroy.apply(this, [options]);
        },

        error: function(model, response, options) {
            try {
                response = $.parseJSON(response.responseText);
            } catch(e) {
                response.errors = [{
                    name: 'server',
                    description: 'A server error prevented saving the model.'
                }];
            }

            if (response.errors.length) {
                model.errors = response.errors;
                model.set(model.previousAttributes());
            }
        },

        fetch: function(options) {
            options = options || {};

            if (!_.has(options, 'success')) {
                options.success = this.success
            }

            BaseModel.__super__.fetch.apply(this, [options]);
        },

        parse: function(response) {
            var message = util.parseMessage(response);
            if (message) {
                this.trigger(BaseModel.MESSAGE_RECEIVED, message);
            }

            var data = BaseModel.__super__.parse.apply(this, arguments);

            // Convert date fields from strings into `moment` objects.
            if (this.dateFields) {
                _.each(this.dateFields, function(field) {
                    if (typeof(data[field] === "string")) {
                        data[field] = moment(data[field]);
                    }
                });
            }

            return data;
        },

         // Return a `moment` object for any date field.
        get: function(attr) {
            var val = BaseModel.__super__.get.apply(this, arguments);

            if(this.dateFields && _.contains(this.dateFields, attr)) {
                var date = moment(val);
                if (date && date.isValid()) {
                    return date;
                } else {
                    return val;
                }
            }

            return val;
        },

        // Call .format() on any date fields when preparing them for JSON
        // serialization.
        toJSON: function() {
            var data = BaseModel.__super__.toJSON.apply(this, arguments);

            if (this.dateFields) {
                _.each(this.dateFields, function(field) {
                    if (typeof(data[field]) === "string") {
                        return;
                    }

                    if (data[field]) {
                        data[field] = data[field].format();
                    }
                });
            }

            return data;
        }
    }, {
        MESSAGE_RECEIVED: 'ajaxForm:messageReceived'
    });
    
    
    var BaseCollection = Backbone.Collection.extend({
        initialize: function (objs, opts) {
            if (opts && opts.gnomeModel) {
                this.gnomeModel = opts.gnomeModel;
            }
        }
    });


    var GnomeModel = BaseModel.extend({
        dateFields: ['start_time'],

        url: function() {
            var id = this.id ? '/' + this.id : '';
            return '/model' + id;
        }
    });


    // Spills
    var SurfaceReleaseSpill = BaseModel.extend({
        dateFields: ['release_time'],

        initialize: function() {
            this.syncArrayField('start_position', 'start_position_x', 0);
            this.syncArrayField('start_position', 'start_position_y', 1);
            this.syncArrayField('start_position', 'start_position_z', 2);

            this.syncArrayField('end_position', 'end_position_x', 0);
            this.syncArrayField('end_position', 'end_position_y', 1);
            this.syncArrayField('end_position', 'end_position_z', 2);
            
            this.syncArrayField('windage_range', 'windage_range_min', 0);
            this.syncArrayField('windage_range', 'windage_range_max', 1);

            SurfaceReleaseSpill.__super__.initialize.apply(this, arguments);
        }
    });


    var SurfaceReleaseSpillCollection = BaseCollection.extend({
        model: SurfaceReleaseSpill,

        comparator: function(spill) {
            return moment(spill.get('release_time')).valueOf();
        },

        url: function() {
            return this.gnomeModel.url() + '/spill/surface_release';
        }
    });


    // Movers

    var Wind = BaseModel.extend({
        dateFields: ['updated_at'],

        initialize: function(attrs, options) {
            if (!attrs || !attrs.timeseries) {
                this.set('timeseries', []);
            }

            Wind.__super__.set.apply(this, arguments);
        },

        /*
         Whenever `timeseries` is set, sort it by datetime.
         */
        set: function(key, val, options) {
            if (key.timeseries && key.timeseries.length) {
                key.timeseries = _.sortBy(key.timeseries, function(item) {
                    return item[0];
                });
            } else if (key === 'timeseries' && val && val.length) {
                val = _.sortBy(val, function(item) {
                    return item[0];
                });
            }

            return Wind.__super__.set.apply(this, [key, val, options]);
        },

        isManual: function() {
            return this.get('source_type') === 'manual';
        },

        isNws: function() {
            return this.get('source_type') === 'nws';
        },

        isBuoy: function() {
            return this.get('source_type') === 'buoy';
        }
    });


    var BaseMover = BaseModel.extend({
        dateFields: ['active_start', 'active_stop']
    });


    var WindMover = BaseMover.extend({

        /*
         Make sure the 'wind' field is a `Wind` object.

         Assume an object with a `get` method is a Wind object; otherwise
         assume it is an object of attrs that we can use to create a new Wind.
         */
        set: function(key, val, options) {
            if (key && key.wind) {
                if (key.wind.get === undefined) {
                    key.wind = new Wind(key.wind);
                }
            } else if (key === 'wind' && val && val.get === undefined) {
                key.wind = new Wind(val);
            } else if (this.get('wind') === undefined) {
                key.wind = new Wind();
            }

            WindMover.__super__.set.apply(this, [key, val, options]);
            return this;
        },

        getTimeseries: function() {
            return this.get('wind').get('timeseries');
        },

        type: function() {
            var timeseries = this.getTimeseries();

            if (timeseries && timeseries.length > 1) {
                return 'variable-wind';
            } else {
                return 'constant-wind';
            }
        },

        constantSpeed: function() {
            var timeseries = this.getTimeseries();

            if (timeseries && timeseries.length) {
                return timeseries[0][1];
            } else {
                return 0;
            }
        },

        constantDirection: function() {
            var timeseries = this.getTimeseries();

            if (timeseries && timeseries.length) {
                return timeseries[0][2];
            } else {
                return 0;
            }
        }
    });


    var WindMoverCollection = BaseCollection.extend({
        model: WindMover,

        comparator: function(mover) {
            var wind = mover.get('wind');
            var timeseries = wind.get('timeseries');

            if (timeseries.length) {
                return moment(timeseries[0][0]).valueOf();
            }
        },

        url: function() {
            return this.gnomeModel.url() + '/mover/wind';
        }
    });
    
    
    var RandomMover = BaseMover.extend({});
    
    
    var RandomMoverCollection = BaseCollection.extend({
        model: RandomMover,

        url: function() {
            return this.gnomeModel.url() + '/mover/random';
        },

        comparator: function(mover) {
            return this.get('active_start');
        }
    });


    var Map = BaseModel.extend({
        url: function() {
            return this.gnomeModel.url() + '/map';
        }
    });


    var CustomMap = BaseModel.extend({
        url: function() {
            return this.gnomeModel.url() + '/custom_map';
        }
    });


    var LocationFile = BaseModel.extend({
        idAttribute: 'filename'
    });

    var LocationFileCollection = BaseCollection.extend({
        model: LocationFile,

        url: function() {
            return this.gnomeModel.url() + '/location_file';
        }
    });

    var LocationFileWizard = BaseModel.extend({
        url: function() {
            return this.collection.url() + '/' + this.id + '/wizard';
        }
    });

    var LocationFileWizardCollection = BaseCollection.extend({
        model: LocationFileWizard,

        url: function() {
            return this.gnomeModel.url() + '/location_file';
        }

    });


    var GnomeModelFromLocationFile = BaseModel.extend({
        url: function() {
            return '/model/from_location_file/' + this.get('location_name')
        }
    });


    function nwsWindError(resp) {
        var error;
        var defaultError = 'Could not contact the National Weather Service.';
        try {
            error = JSON.parse(resp.responseText).error;
        } catch (e) {
            error = defaultError;
        }
        window.alert(error);
    }


    function getNwsWind(coords, opts) {
        var url = '/nws/wind?lat=' + coords.latitude + '&lon=' + coords.longitude;
        var error = opts.error | nwsWindError;
        var success = opts.success;
        $.ajax({
            url: url,
            success: success,
            error: error,
            dataType: 'json'
        });
    }


    return {
        TimeStep: TimeStep,
        GnomeRun: GnomeRun,
        GnomeModel: GnomeModel,
        GnomeModelFromLocationFile: GnomeModelFromLocationFile,
        BaseModel: BaseModel,
        BaseCollection: BaseCollection,
        SurfaceReleaseSpill: SurfaceReleaseSpill,
        SurfaceReleaseSpillCollection: SurfaceReleaseSpillCollection,
        Wind: Wind,
        WindMover: WindMover,
        WindMoverCollection: WindMoverCollection,
        RandomMover: RandomMover,
        RandomMoverCollection: RandomMoverCollection,
        Map: Map,
        CustomMap: CustomMap,
        LocationFile: LocationFile,
        LocationFileCollection: LocationFileCollection,
        LocationFileWizard: LocationFileWizard,
        LocationFileWizardCollection: LocationFileWizardCollection,
        getNwsWind: getNwsWind
    };

});
