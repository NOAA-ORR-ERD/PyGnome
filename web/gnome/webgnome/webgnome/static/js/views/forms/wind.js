define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'views/base',
    'views/forms/base',
    'views/forms/modal',
    'views/forms/timeseries',
    'lib/moment',
    '//netdna.bootstrapcdn.com/bootstrap/3.0.0/js/bootstrap.min.js',
    'lib/jquery.imagesloaded.min',
    'lib/gmaps'
], function($, _, Backbone, models, base_view, base_form, modal, timeseries) {
     var ExternalWindDataView = base_view.BaseView.extend({
        initialize: function() {
            _.bindAll(this);

            this.map = this.options.map;

            // Setup this.$el, which we need for `setupWindMap`.
            ExternalWindDataView.__super__.initialize.apply(this, arguments);
            this.events = _.extend({}, Backbone.View.prototype.events, this.events);

            this.setupWindMap();
        },

        events: {
            'click .query-source': 'querySource'
        },

        resizeNwsMap: function() {
            google.maps.event.trigger(this.windMap, 'resize');
        },

        zoomNwsMap: function(zoomLevel) {
            this.windMap.makeZoomable(zoomLevel);
        },

        /*
         Center the NWS winds map on the center of the user's chosen map, if a
         map has been added to the model and has a bounding box.
         */
        centerNwsMap: function() {
            if (!this.map) {
                return;
            }

            var center = this.map.getLatLongCenter();

            if (!center) {
                return;
            }

            this.windMap.setCenter(center);
            this.zoomNwsMap(6);
        },

        prepareNwsMap: function() {
            this.resizeNwsMap();
            this.centerNwsMap();
        },

        setModel: function(model) {
            this.model = model;
            this.listenTo(model, 'change:source_type', this.sourceTypeChanged);
        },

        close: function() {
            this.stopListening(this.model);
            this.model = null;
        },

        nwsWindsReceived: function(data) {
            this.model.set(data);
            this.setDateFields('.updated_at_container', moment());
            this.sendMessage({
                type: 'success',
                text: 'Wind data refreshed from current NWS forecasts.'
            });
            this.trigger('dataReceived');
        },

        /*
         Run a function to query an external data source for wind data, given
         a valid data source chosen for the 'source' field.
         */
        querySource: function(event) {
            event.preventDefault();

            var dataSourceFns = {
                nws: this.queryNws
            };

            var source = this.getElementByName('source_type').find('option:selected').val();

            if (dataSourceFns[source]) {
                dataSourceFns[source].apply(this);
            } else {
                window.alert('That data source does not exist.');
            }
        },

        queryNws: function() {
            var lat = this.getElementByName('latitude');
            var lon = this.getElementByName('longitude');
            var coords = {
                latitude: lat.val(),
                longitude: lon.val()
            };

            if (!coords.latitude || !coords.longitude) {
                alert('Please enter both a latitude and longitude value.');
                return;
            }

            if (!window.confirm('Reset wind data from current NWS forecasts?')) {
                return;
            }

            models.getNwsWind(coords, {
                success: this.nwsWindsReceived
            });
        },

        setupWindMap: function() {
            var lat = this.getElementByName('latitude');
            var lon = this.getElementByName('longitude');

            this.windMapCenter = new google.maps.LatLngBounds(
                new google.maps.LatLng(13, 144),
                new google.maps.LatLng(40, -30)
            );

            var myOptions = {
                center: this.windMapCenter.getCenter(),
                zoom: 2,
                mapTypeId: google.maps.MapTypeId.HYBRID,
                streetViewControl: false
            };

            var latlngInit = new google.maps.LatLng(lat.val(), lon.val());

            var map = new google.maps.Map(
                this.$el.find('.nws-map-canvas')[0], myOptions);

            var point = new google.maps.Marker({
                position: latlngInit,
                editable: true,
                draggable: true
            });

            point.setMap(map);
            point.setVisible(false);

            google.maps.event.addListener(map, 'click', function(event) {
                var ulatlng = event.latLng;
                point.setPosition(ulatlng);
                point.setVisible(true);

                lat.val(Math.round(ulatlng.lat() * 1000) / 1000);
                lon.val(Math.round(ulatlng.lng() * 1000) / 1000);

                lat.change();
                lon.change();
            });

            google.maps.event.addListener(point, 'dragend', function(event) {
                var ulatlng = event.latLng;

                point.setPosition(ulatlng);
                point.setVisible(true);

                lat.val(Math.round(ulatlng.lat() * 1000) / 1000);
                lon.val(Math.round(ulatlng.lng() * 1000) / 1000);

                lat.change();
                lon.change();
            });

            this.windMap = map;
        },

        nwsCoordinatesChanged: function() {
            var ulatlng = new google.maps.LatLng(
                this.getElementByName('latitude').val(),
                this.getElementByName('longitude').val());
            this.nwsPoint.setPosition(ulatlng);
            this.nwsPoint.setVisible(true);
        },

        sourceTypeChanged: function() {
            var _this = this;
            if (this.model.isNws()) {
                this.$el.find('.nws-map-container').imagesLoaded(function() {
                    _this.prepareNwsMap();
                });
            }
        }
    });

    var WindFormView = modal.JQueryUIModalFormView.extend({
        initialize: function() {
            this.options.dialog = _.extend({
                width: 750,
                height: 550,
                title: "Edit Wind"
            }, this.options.dialog);

            WindFormView.__super__.initialize.apply(this, arguments);

            // Extend prototype's events with ours.
            this.events = _.extend({}, base_form.FormView.prototype.events, this.events);

            this.externalDataView = new ExternalWindDataView({
                id: this.id + '_data_source',
                map: this.options.map
            });

            this.externalDataView.on('dataReceived', this.externalDataReceived);

            this.variableTimeseriesView = new timeseries.VariableWindTimeseriesView({
                id: this.id + '_variable'
            });

            this.constantTimeseriesView = new timeseries.ConstantWindTimeseriesView({
                id: this.id + '_constant'
            });
        },

        validator: models.WindValidator,

        events: {
            'change .units': 'render',
            'change .type': 'typeChanged'
        },

        render: function() {
            this.variableTimeseriesView.render();
            this.constantTimeseriesView.render();
        },

        externalDataReceived: function() {
            var type = this.getElementByName('type');

            type.find('option[value="variable-wind"]').attr('selected', 'selected');
            type.change();

            this.render();
        },

        getDataBindings: function() {
            return {
                wind: this.model
            };
        },

        close: function() {
            WindFormView.__super__.close.apply(this, arguments);
            this.variableTimeseriesView.close();
            this.constantTimeseriesView.close();
            this.externalDataView.close();
        },

        prepareSubmitData: function() {
            // Clear the add time form in the variable wind div as those
            // values must be "saved" in order to mean anything.
            this.variableTimeseriesView.clearAddForm();

            var wind = this.model;
            var windUpdatedAt = this.getDate(
                this.$el.find('.updated_at_container'));

            if (windUpdatedAt) {
                wind.set('updated_at', windUpdatedAt);
            }

            var timeseries = wind.get('timeseries');
            var constantWindSelected = this.$el.find('.type').find(
                'option:selected').val() === 'constant-wind';

            if (constantWindSelected && timeseries.length > 1) {
                var message = 'Changing this mover to use constant wind will ' +
                    'delete variable wind data. Go ahead?';

                if (!window.confirm(message)) {
                    return;
                }
            }

            // A constant wind mover has these values.
            if (constantWindSelected) {
                var windData = this.constantTimeseriesView.getData();

                if (timeseries.length === 1) {
                    // Update an existing time series value.
                    timeseries[0] = windData
                } else {
                    // Add the first (and only) time series value.
                    timeseries = [windData];
                }

                wind.set('timeseries', timeseries);
            }
        },

        typeChanged: function() {
            var type = this.$el.find('.type').val();

            if (type === 'variable-wind') {
                this.constantTimeseriesView.hide();
                this.variableTimeseriesView.show();
            } else {
                this.constantTimeseriesView.show();
                this.variableTimeseriesView.hide();
            }
        },

        /*
         Set all fields with the current values of `self.model`.
         */
        setInputsFromModel: function() {
            var wind = this.model;
            this.setDateFields('.updated_at_container', wind.get('updated_at'));

            var windType = this.$el.find('.type');
            var timeseries = wind.get('timeseries');

            if (timeseries.length > 1) {
                windType.val('variable-wind');
            } else {
                windType.val('constant-wind');
            }

            this.typeChanged();
        },

        prepareForm: function() {
            this.externalDataView.setModel(this.model);
            this.variableTimeseriesView.setModel(this.model);
            this.constantTimeseriesView.setModel(this.model);
            this.render();

            if (this.model && this.model.id) {
                this.setInputsFromModel();
            } else {
                this.typeChanged();
            }
        },

        /*
         Return an object of timeseries errors keyed to their index in the
         Wind's timeseries array.

         This method has a side effect -- it consumes timeseries-related errors
         from `this.model.errors` and removes them from that array.
         */
        getTimeseriesErrors: function() {
            var errors = {};
            var newErrors = [];

            if (!this.model.errors) {
               return errors;
            }

            _.each(this.model.errors, function(error) {
                var parts = error.name.split('.');

                if (parts.length > 1 && parts[1] === 'timeseries') {
                    errors[parts[2]] = error;
                    return;
                }

                newErrors.push(error);
            });

            this.model.errors = newErrors;

            return errors;
        },

        handleFieldError: function(error) {
            if (error.name.indexOf('wind.') === 0) {
                var parts = error.name.split('.');
                var fieldName = parts[1];
                var field = this.$el.find('*[name="' + fieldName + '"]').not('.hidden');

                this.showErrorForField(field, error);
                return;
            }

            WindFormView.__super__.handleFieldError.apply(this, arguments);
        },

        /*
         Restore the model's wind value and its timeseries values to their
         previous state if there was a server-side error, and render the wind
         values table, in case one of the wind values is erroneous.
         */
        handleServerError: function() {
            var wind = this.model;
            var timeseries = wind.get('timeseries');
            var timeseriesErrors = this.getTimeseriesErrors();
            var timeseriesIdsWithErrors = _.keys(timeseriesErrors).sort();

            if (timeseriesIdsWithErrors.length) {
                if (timeseries.length > 1) {
                    window.alert('Your wind data has errors. The errors have been' +
                        ' highlighted. Please resolve them and save again.');

                    this.$el.find('.wind-data-link').find('a').tab('show');

                    this.showEditFormForWind(timeseriesIdsWithErrors[0]);

                    // XXX: Do we need to make the dialog larger anymore?
                    // This was to accommodate the new space needed for error
                    // messages.
                    this.$el.dialog('option', 'height', 600);
                }

                // Save timeseries errors on the wind object.
                wind.timeseriesErrors = timeseriesErrors;
            }

            this.render();

            // After this is called, model.errors will be null.
            WindFormView.__super__.handleServerError.apply(this, arguments);
        }
    });


    var AddWindFormView = WindFormView.extend({
        initialize: function() {
            this.options.dialog = _.extend({
                width: 750,
                height: 550,
                title: "Add Wind"
            }, this.options.dialog);

            AddWindFormView.__super__.initialize.apply(this, arguments);
        },

        /*
         Use a new Wind every time the form is opened.

         This breaks any event handlers called in superclasses before this
         method is called, so we need to reapply them.
         */
        show: function() {
            this.model = new models.Wind(this.defaults, {
                gnomeModel: this.gnomeModel
            });
            this.setupModelEvents();
            this.listenTo(this.model, 'sync', this.closeDialog);
            AddWindFormView.__super__.show.apply(this);
        }
    });


    // A mixin that overrides jQuery UI dialog related actions.
    var EmbeddedWindFormMixin = {
        setupDialog: function() {
            // Do nothing
        },

        openDialog: function() {
            // Do nothing
        },

        closeDialog: function() {
            // Do nothing
        },

        hide: function() {
            this.$el.addClass('hidden');
        }
    };


    var EmbeddedWindFormView = WindFormView.extend({});
    _.extend(EmbeddedWindFormView.prototype, EmbeddedWindFormMixin);


    var EmbeddedAddWindFormView = AddWindFormView.extend({});
    _.extend(EmbeddedAddWindFormView.prototype, EmbeddedWindFormMixin);


    return {
        EmbeddedWindFormView: EmbeddedWindFormView,
        EmbeddedAddWindFormView: EmbeddedAddWindFormView,
        AddWindFormView: AddWindFormView,
        WindFormView: WindFormView,
        ExternalWindDataView: ExternalWindDataView
    }
});
