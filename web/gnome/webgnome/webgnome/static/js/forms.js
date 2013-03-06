
define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'util',
    'lib/geo',
    'lib/rivets',
    'lib/moment',
    'lib/compass-ui',
    'lib/bootstrap-tab',
    'lib/jquery.ui',
    'lib/jquery.fileupload',
    'lib/jquery.iframe-transport',
    'lib/bootstrap.file-input'
], function($, _, Backbone, models, util, geo, rivets) {

    var FormViewContainer = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.formViews = {};
        },

        add: function(view) {
            var _this = this;
            this.formViews[view.id] = view;

            view.on(FormView.CANCELED, function(form) {
                _this.trigger(FormView.CANCELED, form);
            });
            view.on(FormView.REFRESHED, function(form) {
                _this.trigger(FormView.REFRESHED, form);
            });
            view.on(FormView.MESSAGE_READY, function(message) {
                _this.trigger(FormView.MESSAGE_READY, message);
            });

            view.on(FormView.SHOW_FORM, this.show);

            return view;
        },

        remove: function(id) {
            if (_.has(id, 'id')) {
                id = id.id;
            }
            var view = this.formViews[id];
            delete this.formViews[id];
            return view;
        },

        get: function(formId) {
            return this.formViews[formId];
        },

        hideAll: function() {
            _.each(this.formViews, function(formView, key) {
                formView.hide();
            });
        },

        show: function(formId, success, cancel) {
            var formView = this.get(formId);

            if (formView) {
                formView.once(FormView.SUBMITTED, success);
                formView.once(FormView.CANCELED, cancel);
                formView.reload();
                formView.show();
            }
        },
    });

    function ModelNotFoundException(message) {
        this.message = message;
        this.name = "ModelNotFoundException";
    }

    /*
     `FormView` is the base class for forms intended to wrap Backbone models.

     Submitting a form from `FormView` sends the value of its inputs to the
     model object passed into the form's constructor, which syncs it with the
     server.
     */
    var FormView = Backbone.View.extend({
        initialize: function() {
            var _this = this;
            _.bindAll(this);
            this.wasCancelled = false;
            this.id = this.options.id;

            this.model = this.options.model;

            if (this.model) {
                this.model.on('destroy', function() {
                    _this.model.clear();
                });
            }

            this.collection = this.options.collection;

            if (this.options.id) {
                this.$el = $('#' + this.options.id);
            }

            this.defaults = this.options.defaults;
            this.setupDatePickers();
            $('.error').tooltip({selector: "a"});
        },

        showErrorForField: function(field, error) {
            var errorDiv;

            if (!field.length) {
                alert(error.description);
                return;
            }

            var group = field.closest('.control-group');

            if (group.length) {
                group.addClass('error');
                errorDiv = group.find('a.error')
            }

            // If there is no error div, then report the error to the user.
            if (errorDiv.length) {
                errorDiv.attr('title', error.description);
                errorDiv.removeClass('hidden');
            } else {
                alert(error.description);
            }
        },

        handleFieldError: function(error) {
            var field = this.$el.find('*[name="' + error.name + '"]').not('.hidden');
            this.showErrorForField(field, error);
        },

        /*
         Transform an error object into an input ID.
         */
        getFieldIdForError: function(error) {
            return '#' + error.name;
        },

        /*
         Handle a server-side error.

         This will find any fields that directly match `item.location`. Form
         classes can provide their own more advanced error handling with
         `handleFieldError` and `getFieldIdForError`.
         */
        handleServerError: function() {
            var _this = this;
            this.clearErrors();

            _.each(this.model.errors, function(error) {
                _this.handleFieldError(error);
            });

            // Clear out the errors now that we've handled them.
            this.model.errors = null;
        },

        setupDatePickers: function() {
            // Setup datepickers
            _.each(this.$el.find('.date'), function(field) {
                $(field).datepicker({
                    changeMonth: true,
                    changeYear: true
                });
            });
        },

        getForm: function() {
            return this.$el.find('form');
        },

        getByName: function(name, container, type) {
            container = container && container.length ? container : this.$el;
            type = type ? type : '*';
            return container.find(type + '[name="' + name + '"]');
        },

        clearErrors: function() {
            var groups = this.$el.find('.control-group');
            var errors = this.$el.find('a.error');

            if (groups.length) {
                groups.removeClass('error');
            }

            if (errors.length) {
                errors.attr('title', '');
                errors.addClass('hidden');
            }
        },

        prepareForm: function() {
            // Override
        },

        bindData: function() {
            if (this.rivets) {
                this.rivets.unbind();
            }

            var bindings = this.getDataBindings();

            if (bindings) {
                this.rivets = rivets.bind(this.$el, bindings);
            }
        },

        show: function() {
            throw new Error('You must override show() in a subclass');
        },

        hide: function() {
            throw new Error('You must override hide() in a subclass');
        },

        submit: function() {
            var _this = this;
            if (this.collection) {
                this.collection.add(this.model);
            }
            this.model.save(null, {
                success: function() {
                    _this.trigger(FormView.SUBMITTED);
                }
            });
        },

        cancel: function() {
            this.resetModel();
            this.trigger(FormView.CANCELED);
        },

        resetModel: function() {
            if (this.model && this.model.id) {
                this.model.fetch();
                this.model = null;
            }
        },

        sendMessage: function(message) {
            this.trigger(FormView.MESSAGE_READY, message);
        },

        hasDateFields: function(target) {
            var fields = this.getDateFields(target);
            return fields.date && fields.hour && fields.minute;
        },

        setForm: function(form, data) {
            var _this = this;
            _.each(data, function(dataVal, fieldName) {
                var input = _this.getByName(fieldName, form);

                if (!input.length) {
                    return;
                }

                if (input.is(':checkbox')) {
                    input.prop('checked', dataVal);
                } else {
                    input.val(dataVal);
                }
            });
        },

        getDateFields: function(target) {
            target = $(target);

            if (target.length === 0) {
                return;
            }

            return {
                date: target.find('.date'),
                hour: target.find('.hour'),
                minute: target.find('.minute')
            }
        },

        getFormDate: function(target) {
            if (target.length === 0) {
                return;
            }

            var fields = this.getDateFields(target);
            var date = fields.date.val();
            var hour = fields.hour.val();
            var minute = fields.minute.val();

            if (hour && minute) {
                date = date + ' ' + hour + ':' + minute;
            }

            // TODO: Handle a date-parsing error here.
            if (date) {
                return moment(date);
            }
        },

        setDateFields: function(target, datetime) {
            if (typeof target === 'string') {
                target = this.$el.find(target);
            }

            if (!datetime || target.length === 0) {
                return;
            }

            var fields = this.getDateFields(target);
            fields.date.val(datetime.format("MM/DD/YYYY"));
            fields.hour.val(datetime.format('HH'));
            fields.minute.val(datetime.format('mm'));
        },

        clearForm: function() {
            var inputs = this.$el.find('select,input,textarea');

            if (!inputs.length) {
                return;
            }

            _.each(inputs, function(field) {
                $(field).val('');
            });
        },

        setupModelEvents: function() {
            this.model.unbind('error');
            this.model.bind('error', this.handleServerError);
        },

        getDataBindings: function() {
            return {model: this.model};
        },

        getModel: function(id) {
            if (!id) {
                return;
            }

            var model = null;

            if (this.collection) {
                model = this.collection.get(id);
            }

            return model;
        },

        reload: function(id) {
            if (!id) {
                return;
            }

            this.model = this.getModel(id);

            if (!this.model) {
                throw new ModelNotFoundException('Model missing on reload.');
            }

            this.setupModelEvents();
        }
    }, {
        SUBMITTED: 'formView.submitted',
        CANCELED: 'formView:canceled',
        REFRESHED: 'formView:refreshed',
        MESSAGE_READY: 'formView:messageReady',
        SHOW_FORM: 'formView:showForm'
    });


     /*
     An `FormView` subclass that displays in a JQuery UI modal window.
     */
    var JQueryUIModalFormView = FormView.extend({
        initialize: function(options) {
            JQueryUIModalFormView.__super__.initialize.apply(this, [options]);
            var _this = this;

            // The default set of UI Dialog widget options. A 'dialog' field
            // may be passed in with `options` containing additional options,
            // or subclasses may provide a 'dialog' field with the same.
            var opts = $.extend({
                zIndex: 5000,
                autoOpen: false,
                buttons: {
                    Cancel: function() {
                        _this.cancel();
                    },

                    Save: function() {
                        _this.submit();
                    }
                },
                close: this.close,
                beforeClose: this.beforeClose
            }, options.dialog || this.dialog || {});

            this.$el.dialog(opts);

            // Workaround the default JQuery UI Dialog behavior of auto-
            // focusing on the first input element by adding an invisible one.
            $('<span class="ui-helper-hidden-accessible">' +
                '<input type="text"/></span>').prependTo(this.$el);
        },

        beforeClose: function() {
            if (!this.wasCancelled && this.model && this.model.dirty) {
                if (!window.confirm('You have unsaved changes. Really close?')) {
                    return false;
                }
            }

            return true;
        },

        cancel: function() {
            this.wasCancelled = true;
            this.closeDialog();
            JQueryUIModalFormView.__super__.cancel.apply(this, arguments);
        },

        submit: function() {
            // Close if the form isn't using a model. If the form IS using a
            // model, it will close when the model saves without error.
            if (!this.model) {
                this.$el.dialog("close");
            }
            JQueryUIModalFormView.__super__.submit.apply(this, arguments);
        },

        setupModelEvents: function() {
            if (this.model) {
                this.model.on('sync', this.closeDialog);
            }
            if (this.collection) {
                this.collection.on('sync', this.closeDialog);
            }
            JQueryUIModalFormView.__super__.setupModelEvents.apply(this, arguments);
        },

        /*
         Hide any other visible modals and show this one.
         */
        show: function() {
            this.wasCancelled = false;
            this.clearErrors();
            this.prepareForm();
            this.$el.dialog('open');
            this.$el.removeClass('hide');
            this.bindData();
        },

        hide: function() {
            this.$el.dialog('close');
        },

        close: function() {
            this.closeDialog();
            this.resetModel();
        },

        closeDialog: function() {
            this.$el.dialog('close');
        }
    });


    var MultiStepFormView = JQueryUIModalFormView.extend({
        events: {
            'click .ui-button.next': 'next',
            'click .ui-button.back': 'back'
        },

        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    height: 350,
                    width: 350
                }
            }, options);

            // Extend prototype's events with ours.
            this.events = _.extend({}, FormView.prototype.events, this.events);

            // Have to initialize super super before showing current step.
            MultiStepFormView.__super__.initialize.apply(this, [opts]);

            this.annotateSteps();
        },

        /*
         Annotate each step div with its step number. E.g., the first step will
         have step number 0, second step number 1, etc.
         */
        annotateSteps: function() {
            _.each(this.$el.find('.step'), function(step, idx) {
                $(step).attr('data-step', idx);
            });
        },

        getCurrentStep: function() {
            return this.$el.find('.step').not('.hidden');
        },

        getCurrentStepNum: function() {
            return this.getCurrentStep().data('step');
        },

        getStep: function(stepNum) {
            return this.$el.find('.step[data-step="' + stepNum + '"]');
        },

        getNextStep: function() {
            var currentStep = this.getCurrentStep();
            var nextStepNum = currentStep.data('step') + 1;
            return this.getStep(nextStepNum);
        },

        getPreviousStep: function() {
            var currentStep = this.getCurrentStep();
            var previousStepNum = currentStep.data('step') - 1;
            // Minimum step number is 0.
            previousStepNum = previousStepNum < 0 ? 0 : previousStepNum;
            return this.getStep(previousStepNum);
        },

        show: function() {
            this.showStep(this.getStep(0));
            MultiStepFormView.__super__.show.apply(this, arguments);
        },

        showStep: function(step) {
            if (step.length) {
                this.$el.find('.step').not(step).addClass('hidden');
                step.removeClass('hidden');
            }
        },

        next: function() {
            var nextStep = this.getNextStep();
            this.showStep(nextStep);
        },

        back: function() {
            var previousStep = this.getPreviousStep();
            this.showStep(previousStep);
        }
    });


    var LocationFileWizardFormView = MultiStepFormView.extend({
        initialize: function(options) {
            LocationFileWizardFormView.__super__.initialize.apply(this, arguments);

            _.bindAll(this);

            this.widget = this.$el.dialog('widget');
            this.widget.on('click', '.ui-button.next', this.next);
            this.widget.on('click', '.ui-button.back', this.back);
            this.widget.on('click', '.ui-button.cancel', this.cancel);
            this.widget.on('click', '.ui-button.finish', this.finish);
            this.widget.on('click', '.ui-button.references', this.showReferences);

            this.references = this.$el.find('div.references').dialog({
                autoOpen: false,
                buttons: {
                    Ok: function() {
                        $(this).dialog("close");
                    }
                }
            });

            this.setCustomButtons();
        },

        getDataBindings: function() {
            return {wizard: this.model};
        },

        finish: function() {
            console.log('Finished');
        },

        close: function() {
            LocationFileWizardFormView.__super__.close.apply(this, arguments);
            this.remove();
        },

        showReferences: function() {
            this.references.dialog('open');
        },

        setCustomButtons: function() {
            var step = this.getCurrentStep();
            var buttons = step.find('.custom-dialog-buttons');
            if (buttons.length) {
                var buttonPane = this.widget.find('.ui-dialog-buttonpane');
                buttonPane.empty();
                buttons.clone().appendTo(buttonPane).removeClass('hidden');
            }
        },

        showStep: function(step) {
            LocationFileWizardFormView.__super__.showStep.apply(this, [step]);
            this.setCustomButtons();

            var referenceForm = step.data('reference-form');
            if (referenceForm) {
                this.widget.hide();
                this.showReferenceForm(referenceForm);
            } else {
                this.widget.show();
            }
        },

        showReferenceForm: function(referenceForm) {
            var _this = this;
            function showNextForm() {
                _this.next();
            }
            function cancel() {
                _this.back();
            }
            this.trigger(FormView.SHOW_FORM, referenceForm, showNextForm, cancel);
        }
    });


    /*
     A base class for modal forms that ask the user to choose from a list of
     object types that are themselves represented by a `FormView` instance.

     TODO: Should this extend a non-FormView class?
     */
    var ChooseObjectTypeFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var _this = this;
            var opts = _.extend({
                dialog: {
                    height: 175,
                    width: 400,
                    buttons: {
                        Cancel: function() {
                            _this.cancel();
                            $(this).dialog("close");
                        },

                        Choose: function() {
                            _this.submit();
                            $(this).dialog("close");
                        }
                    }
                }
            }, options);

            ChooseObjectTypeFormView.__super__.initialize.apply(this, [opts]);
        }
    });


    /*
     This is a non-AJAX-enabled modal form object to support the "add mover" form,
     which asks the user to choose a type of mover to add. We then use the selection
     to display another, mover-specific form.
     */
    var AddMoverFormView = ChooseObjectTypeFormView.extend({
        submit: function() {
            var moverType = this.getByName('mover-type').val();

            if (moverType) {
                this.trigger(AddMoverFormView.MOVER_CHOSEN, moverType);
                this.hide();
            }

            return false;
        }
    }, {
        // Events
        MOVER_CHOSEN: 'addMoverFormView:moverChosen'
    });


     /*
     This is a non-AJAX-enabled modal form object to support the "add spill"
     form, which asks the user to choose a type of spill to add. We then use the
     selection to display another, spill-specific form.
     */
    var AddSpillFormView = ChooseObjectTypeFormView.extend({
        show: function(startCoords, endCoords) {
            this.startCoords = startCoords;
            this.endCoords = endCoords;

            AddSpillFormView.__super__.show.apply(this);
        },

        submit: function() {
            var spillType = this.getByName('spill-type').val();

            if (spillType) {
                this.trigger(AddSpillFormView.SPILL_CHOSEN, spillType, this.startCoords, this.endCoords);
                this.coords = null;
                this.hide();
            }

            return false;
        },

        cancel: function() {
            this.trigger(AddSpillFormView.CANCELED, this);
        }
    }, {
        // Event constants
        SPILL_CHOSEN: 'addSpillFormView:spillChosen',
        CANCELED: 'addSpillFormView:canceled'
    });


    var AddMapFormView = ChooseObjectTypeFormView.extend({
        submit: function() {
            var source = this.getByName('map-source').val();
            if (source) {
                this.trigger(AddMapFormView.SOURCE_CHOSEN, source);
            }
        }
    }, {
        // Event constants
        SOURCE_CHOSEN: 'addMapFormView:sourceChosen'
    });


    var MapFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    height: 200,
                    width: 425
                }
            }, options);

            MapFormView.__super__.initialize.apply(this, [opts]);

            this.setupModelEvents();
        },

        // Always use the same model
        getModel: function(id) {
            return this.model;
        },

        // Never reset the model
        resetModel: function() {
            return;
        },

        getDataBindings: function() {
            return {map: this.model};
        }
    });


    var AddCustomMapFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    height: 450,
                    width: 700
                }
            }, options);

            AddCustomMapFormView.__super__.initialize.apply(this, [opts]);

            this.map = this.$el.find('#custom-map').mapGenerator({
                change: this.updateSelection
            });
        },

        updateSelection: function(rect) {
            var _this = this;
            _.each(rect, function(value, key) {
                var field = _this.$el.find('#' + key);
                if (field.length) {
                    field.val(value);
                    field.change();
                }
            });
        },

        // Always use the same model
        getModel: function(id) {
            return this.model
        },

        // Never reset the model.
        resetModel: function() {
            return;
        },

        getDataBindings: function() {
            return {map: this.model};
        },

        show: function() {
            this.model.clear();
            this.model.set(this.defaults);
            this.map.clearSelection();
            // Have to set these manually since reload() doesn't get called
            this.setupModelEvents();
            AddCustomMapFormView.__super__.show.apply(this);
            this.map.resize();
        }
    });


    var AddMapFromUploadFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var _this = this;
            var opts = _.extend({
                dialog: {
                    height: 250,
                    width: 500
                }
            }, options);

            AddMapFromUploadFormView.__super__.initialize.apply(this, [opts]);

            this.uploadUrl = options.uploadUrl;
            var uploadInput = this.$el.find('.fileupload');
            var saveButton = this.$el.parent().find('button:contains("Save")');
            uploadInput.attr('data-url', this.uploadUrl);

            // The user will not be able to submit the form until the file they
            // chose has finished uploading.
            this.uploader = uploadInput.fileupload({
                dataType: 'json',
                submit: function(e, data) {
                    saveButton.button('disable');
                },
                done: function(e, data) {
                    _this.model.set('filename', data.result.filename);
                    saveButton.button('enable');
                }
            });

            this.setupModelEvents();
        },

        // Always use the same model
        getModel: function(id) {
            return this.model;
        },

        getDataBindings: function() {
            return {map: this.model};
        }
    });



    /*
     `WindMoverFormView` handles the WindMover form.
     */
    var WindMoverFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    width: 750,
                    height: 550,
                    title: "Edit Wind Mover"
                }
            }, options);

            WindMoverFormView.__super__.initialize.apply(this, [opts]);

            this.defaultWindTimeseriesValue = options.defaultWindTimeseriesValue;
            this.windMovers = options.windMovers;
            this.setupCompass();
            this.setupWindMap();

            this.$el.find('.data-source-link').on('shown', this.resizeWindMap);

            // Extend prototype's events with ours.
            this.events = _.extend({}, FormView.prototype.events, this.events);
        },

        events: {
            'click .add-time': 'addButtonClicked',
            'click .edit-time': 'editButtonClicked',
            'click .show-compass': 'showCompass',
            'click .cancel': 'cancelButtonClicked',
            'click .save': 'saveButtonClicked',
            'click .delete-time': 'trashButtonClicked',
            'click .query-source': 'querySource',
            'change .units': 'renderTimeTable',
            'change .type': 'typeChanged'
        },

        getDataBindings: function() {
            return {
                mover: this.model,
                wind: this.model.get('wind')
            };
        },

        close: function() {
            WindMoverFormView.__super__.close.apply(this, arguments);
            this.compassDialog.dialog('close');
        },

        resizeWindMap: function() {
            google.maps.event.trigger(this.windMap, 'resize');
            this.windMap.setCenter(this.windMapCenter.getCenter());
        },

        nwsWindsReceived: function(data) {
            var desc = this.getByName('description');
            var type = this.getByName('type');

            desc.val(data.description);
            desc.change();

            type.find('option[value="variable-wind"]').attr('selected', 'selected');
            type.change();

            this.setDateFields('.updated_at_container', moment());

            var wind = this.model.get('wind');
            var timeseries = [];

            _.each(data.results, function(windData) {
                timeseries.push([windData[0], windData[1], windData[2]]);
            });

            wind.set('timeseries', timeseries);

            // NWS data is in knots, so the entire wind mover data set will have
            // to use knots, since we don't have a way of setting it per value.
            wind.set('units', 'knots');

            this.renderTimeTable();

            this.sendMessage({
                type: 'success',
                text: 'Wind data refreshed from current NWS forecasts.'
            });
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

            var source = this.getByName('source_type').find('option:selected').val();

            if (dataSourceFns[source]) {
                dataSourceFns[source]();
            } else {
                window.alert('That data source does not exist.');
            }
        },

        queryNws: function() {
            var lat = this.getByName('latitude');
            var lon = this.getByName('longitude');
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

            models.getNwsWind(coords, this.nwsWindsReceived);
        },

        setupWindMap: function() {
            var lat = this.getByName('latitude');
            var lon = this.getByName('longitude');

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
                this.getByName('latitude').val(),
                this.getByName('longitude').val());
            this.nwsPoint.setPosition(ulatlng);
            this.nwsPoint.setVisible(true);
        },

        typeChanged: function() {
            var type = this.$el.find('.type').val();
            var typeDiv = this.$el.find('.' + type);
            var otherDivs = this.$el.find(
                '.tab-pane.wind > div').not(typeDiv);

            typeDiv.removeClass('hidden');
            otherDivs.addClass('hidden');
        },

        showCompass: function() {
            this.compass.compassUI('reset');
            this.compassDialog.dialog('open');
        },

        compassChanged: function(magnitude, direction) {
            this.compassMoved(magnitude, direction);
        },

        compassMoved: function(magnitude, direction) {
            var form = this.getAddForm();
            form.find('.speed').val(magnitude.toFixed(2));
            form.find('.direction').val(direction.toFixed(2));
        },

        setupCompass: function() {
            var _this = this;
            var compass = this.$el.find('.compass');

            this.compass = compass.compassUI({
                'arrow-direction': 'in',
                'move': function(magnitude, direction) {
                    _this.compassMoved(magnitude, direction);
                },
                'change': function(magnitude, direction) {
                    _this.compassChanged(magnitude, direction);
                }
            });

            this.compassDialog = this.$el.find('.compass-container').dialog({
                width: 250,
                title: "Compass",
                zIndex: 6000,
                autoOpen: false,
                buttons: {
                    Close: function () {
                        $(this).dialog("close");
                    }
                }
            });
        },

        getTimesTable: function() {
            return this.$el.find('.time-list');
        },

        getWindIdsWithErrors: function() {
            var valuesWithErrors = [];

            if (!this.model.errors) {
               return valuesWithErrors;
            }

            _.each(this.model.errors, function(error) {
                var parts = error.name.split('.');

                if (parts.length > 1 && parts[1] === 'timeseries') {
                    valuesWithErrors.push(parts[2]);
                }
            });

            return valuesWithErrors;
        },

        renderTimeTable: function() {
            var _this = this;
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');
            var units = wind.get('units');
            var rows = [];
            var IdsWithErrors = this.getWindIdsWithErrors();

            // Clear out any existing rows.
            this.getTimesTable().find('tr').not('.table-header').remove();

            _.each(timeseries, function(windValue, index) {
                var tmpl = _.template($("#time-series-row").html());
                var speed = windValue[1];
                var direction = windValue[2];

                if (typeof(direction) === 'number') {
                    direction = direction.toFixed(1);
                }

                if (typeof(speed) === 'number') {
                    speed = speed.toFixed(1);
                }

                var datetime = moment(windValue[0]);
                // TODO: Error handling
                var error = null;
                var row = $(tmpl({
                    error: error ? 'error' : '',
                    date: datetime.format('MM/DD/YYYY'),
                    time: datetime.format('HH:mm'),
                    direction: direction + ' &deg;',
                    speed: speed + ' ' + units
                }));

                row.attr('data-wind-id', index);

                if (_.contains(IdsWithErrors, index)) {
                    row.addClass('error');
                }

                rows.push(row);
            });

            _.each(rows, function(row) {
                row.appendTo(_this.getTimesTable());
            });
        },

        /*
         Get wind timeseries values needed to save a constant wind mover.
         */
        getConstantWindData: function() {
            var form = this.getAddForm();
            var speed = this.getByName('speed', form);
            var direction = this.getByName('direction', form);

            // A datetime is required, but it will be ignored for a constant
            // wind mover during the model run, so we just use the current
            // time.
            return [moment().format(), speed.val(), direction.val()];
        },

        submit: function() {
            // Clear the add time form in the variable wind div as those
            // values must be "saved" in order to mean anything.
            var variable = this.$el.find('.variable-wind');
            variable.find('input').val('');
            variable.find('input:checkbox').prop('checked', false);

            var wind = this.model.get('wind');
            var windUpdatedAt = this.getFormDate(
                this.$el.find('.updated_at_container'));

            if(windUpdatedAt) {
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
                var windData = this.getConstantWindData();

                if (timeseries.length === 1) {
                    // Update an existing time series value.
                    timeseries[0] = windData
                } else {
                    // Add the first (and only) time series value.
                    timeseries = [windData];
                }

                wind.set('timeseries', timeseries);
            }

            WindMoverFormView.__super__.submit.apply(this, arguments);
        },

        getMoverType: function() {
            return this.$el.find('.type').val();
        },

        getMoverTypeDiv: function(type) {
            type = type || this.getMoverType();
            return this.$el.find('.' + type);
        },

        getAddForm: function(type) {
            return this.getMoverTypeDiv(type).find('.add-time-form');
        },

        trashButtonClicked: function(event) {
            event.preventDefault();
            var windId = $(event.target).closest('tr').attr('data-wind-id');
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');
            var windValue = timeseries[windId];
            var addForm = this.getAddForm();

            if (addForm.attr('data-wind-id') === windId) {
                this.setFormDefaults();
                addForm.find('.add-time-buttons').removeClass('hidden');
                addForm.find('.edit-time-buttons').addClass('hidden');
            }

            if (windValue) {
                // Remove the wind value from the timeseries array.
                timeseries.splice(windId, 1);
            }

            this.renderTimeTable();
        },

        getCardinalAngle: function(value) {
            if (value && isNaN(value)) {
                value = util.cardinalAngle(value);
            }

            return value;
        },

        /*
         Search `timeseries` array for a value whose datetime matches `datetime`.

         Skip any duplicate found at index `ignoreId`.

         Return an array containing the index of each duplicate found.
         */
        findDuplicates: function(timeseries, datetime, ignoreIndex) {
            var duplicateIndexes = [];

            _.each(timeseries, function(value, index) {
                if (moment(value[0]).format() == datetime.format()
                        && index !== parseInt(ignoreIndex)) {
                    duplicateIndexes.push(index);
                }
            });

            return duplicateIndexes;
        },

        saveButtonClicked: function(event) {
            event.preventDefault();
            var wind = this.model.get('wind');
            var timeseries = _.clone(wind.get('timeseries'));
            var addForm = this.getAddForm();
            var datetime = this.getFormDate(addForm);
            var windId = addForm.attr('data-wind-id');
            var direction = addForm.find('#direction').val();
            var duplicates = this.findDuplicates(timeseries, datetime, windId);
            var message = 'Wind data for that date and time exists. Replace it?';

            if (duplicates.length) {
                if (window.confirm(message)) {
                    timeseries.remove(duplicates[0]);
                    this.renderTimeTable();
                } else {
                    return;
                }
            }

            timeseries[windId] = [
                datetime.format(),
                this.getByName('speed', addForm).val(),
                this.getCardinalAngle(direction)
            ];

            wind.set('timeseries', timeseries);

            this.setFormDefaults();
            addForm.find('.add-time-buttons').removeClass('hidden');
            addForm.find('.edit-time-buttons').addClass('hidden');
            this.renderTimeTable();
            this.compass.compassUI('reset');
        },

        cancelButtonClicked: function(event) {
            event.preventDefault();
            var addForm = this.getAddForm();
            this.setFormDefaults();
            addForm.find('.add-time-buttons').removeClass('hidden');
            addForm.find('.edit-time-buttons').addClass('hidden');
            var row = $(event.target).closest('tr.info');
            row.removeClass('info');
            this.renderTimeTable();
        },

        getRowForWindId: function(windId) {
            return this.$el.find('tr[data-wind-id="' + windId + '"]')
        },

        setWindValueForm: function(form, data) {
            var datetimeFields = form.find('.datetime_container');

            if (datetimeFields.length) {
                this.setDateFields(datetimeFields, moment(data[0]));
            }

            this.getByName('speed', form).val(data[1]);
            this.getByName('direction', form).val(data[2]);
        },

        showEditFormForWind: function(windId) {
            var row = this.getRowForWindId(windId);
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');
            var windValue = timeseries[windId];
            var addForm = this.getAddForm();

            addForm.attr('data-wind-id', windId);
            addForm.find('.add-time-buttons').addClass('hidden');
            addForm.find('.edit-time-buttons').removeClass('hidden');
            this.setWindValueForm(addForm, windValue);
            this.getTimesTable().find('tr').removeClass('info');
            row.removeClass('error').removeClass('warning').addClass('info');
        },

        editButtonClicked: function(event) {
            event.preventDefault();
            var row = $(event.target).closest('tr');
            var windId = row.attr('data-wind-id');
            if(windId !== null) {
                this.showEditFormForWind(windId);
            }
        },
        
        addButtonClicked: function(event) {
            event.preventDefault();
            var wind = this.model.get('wind');
            var timeseries = _.clone(wind.get('timeseries'));
            var addForm = this.getAddForm();
            var datetime = this.getFormDate(addForm);
            var direction = addForm.find('#direction').val();

            // TODO: Handle this with a field validation plugin.
            if (datetime === undefined) {
                window.alert('Please enter a date and time.');
                return;
            }

            var duplicates = this.findDuplicates(timeseries, datetime);
            var windValue = [
                datetime.format(),
                this.getByName('speed', addForm).val(),
                this.getCardinalAngle(direction)
            ];
            var warning = 'Wind data for that date and time exists. Replace it?';

            if (duplicates.length) {
                if (window.confirm(warning)) {
                    timeseries[duplicates[0]] = windValue;
                    this.renderTimeTable();
                } else {
                    return;
                }
            } else {
                timeseries.push(windValue);
            }

            wind.set('timeseries', timeseries);
            this.renderTimeTable();
            this.compass.compassUI('reset');

            var autoIncrementBy = addForm.find('.auto_increment_by').val();

            // Increase the date and time on the Add form if 'auto increase by'
            // value was provided.
            if (autoIncrementBy) {
                var nextDatetime = datetime.clone().add('hours', autoIncrementBy);
                this.setDateFields(addForm, nextDatetime);
            }
        },

        /*
         Set all fields for which a default value exists.
         */
        setFormDefaults: function() {
            this.setDateFields('.datetime_container',
                moment(this.defaultWindTimeseriesValue[0]));
            this.getByName('speed').val(this.defaultWindTimeseriesValue[1]);
            this.getByName('direction').val(this.defaultWindTimeseriesValue[2]);
            this.clearErrors();
        },

        /*
         Set all fields with the current values of `self.model`.
         */
        setInputsFromModel: function() {
            var wind = this.model.get('wind');
            this.setDateFields('.active_start_container', this.model.get('active_start'));
            this.setDateFields('.active_stop_container', this.model.get('active_stop'));
            this.setDateFields('.updated_at_container', wind.get('updated_at'));

            var moverType = this.$el.find('.type');
            var timeseries = wind.get('timeseries');
            var firstTimeValue = timeseries[0];

            if (timeseries.length > 1) {
                moverType.val('variable-wind');
            } else {
                moverType.val('constant-wind');
            }

            this.typeChanged();

            if (firstTimeValue) {
                this.setWindValueForm(
                    this.getAddForm('constant-wind'), firstTimeValue);
            }
        },

        /*
         Prepare this form for display.
         */
        prepareForm: function() {
            this.renderTimeTable();
            this.setFormDefaults();

            if (this.model && this.model.id) {
                this.setInputsFromModel();
            } else {
                this.typeChanged();
            }
        },

        handleFieldError: function(error) {
            if (error.name.indexOf('wind.') === 0) {
                var parts = error.name.split('.');
                var fieldName = parts[1];
                var field = this.$el.find('*[name="' + fieldName + '"]').not('.hidden');

                this.showErrorForField(field, error);
                return;
            }

            WindMoverFormView.__super__.handleFieldError.apply(this, arguments);
        },

        /*
         Restore the model's wind value and its timeseries values to their
         previous state if there was a server-side error, and render the wind
         values table, in case one of the wind values is erroneous.
         */
        handleServerError: function() {
            var _this = this;
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');

            this.renderTimeTable();

            var windIdsWithErrors = this.getWindIdsWithErrors();

            if (windIdsWithErrors.length) {
                window.alert('Your wind data has errors. The errors have been' +
                    ' highlighted. Please resolve them and save again.');

                this.$el.find('.wind-data-link').find('a').tab('show');

                if (timeseries.length > 1) {
                    this.showEditFormForWind(windIdsWithErrors[0]);
                }

                // Always mark up the table because a user with a constant
                // wind mover could switch to variable wind and edit the value.
                _.each(windIdsWithErrors, function(id) {
                    var row = _this.getRowForWindId(id);

                    if (row.length) {
                        row.addClass('error');
                    }
                });
            }

            this.$el.dialog('option', 'height', 600);

            // After this is called, model.errors will be null.
            WindMoverFormView.__super__.handleServerError.apply(this, arguments);
        }
    });


    var AddWindMoverFormView = WindMoverFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    width: 750,
                    height: 550,
                    title: "Add Wind Mover"
                }
            }, options);

            AddWindMoverFormView.__super__.initialize.apply(this, [opts]);
        },

        /*
         Use a new WindMover every time the form is opened.

         This breaks any event handlers called in superclasses before this
         method is called, so we need to reapply them.
         */
        show: function() {
            this.model = new models.WindMover(this.defaults);
            this.setupModelEvents();
            this.model.on('sync', this.closeDialog);
            AddWindMoverFormView.__super__.show.apply(this);
        }
    });


    var SurfaceReleaseSpillFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    width: 400,
                    height: 550,
                    title: "Edit Surface Release Spill"
                }
            }, options);

            SurfaceReleaseSpillFormView.__super__.initialize.apply(this, [opts]);

            // Extend prototype's events with ours.
            this.events = _.extend({}, FormView.prototype.events, this.events);
        },

        getDataBindings: function() {
            return {spill: this.model};
        },

        prepareForm: function(form, data) {
            this.setDateFields('.release_time_container', this.model.get('release_time'));
            SurfaceReleaseSpillFormView.__super__.prepareForm.apply(this, arguments);
        },

        show: function(startCoords, endCoords) {
            if (startCoords) {
                this.model.set('start_position', [startCoords[0], startCoords[1], 0]);
            }
            if (endCoords) {
                this.model.set('end_position', [endCoords[0], endCoords[1], 0]);
            }
            SurfaceReleaseSpillFormView.__super__.show.apply(this, arguments);
        },

        submit: function() {
            this.model.set('release_time', this.getFormDate(this.getForm()));
            SurfaceReleaseSpillFormView.__super__.submit.apply(this, arguments);
        },

        cancel: function() {
            this.trigger(SurfaceReleaseSpillFormView.CANCELED, this);
            SurfaceReleaseSpillFormView.__super__.cancel.apply(this, arguments);
        },

        handleFieldError: function(error) {
            var field;
            var fieldName = error.name.split('.')[1];
            var positions = {
                0: 'x',
                1: 'y',
                2: 'z'
            };
            var windage = {
                0: 'min',
                1: 'max'
            };

            if (error.name.indexOf('start_position.') === 0) {
                fieldName = 'start_position_' + positions[fieldName];
            } else if(error.name.indexOf('end_position.') === 0) {
                fieldName = 'end_position_' + positions[fieldName];
            } else if (error.name.indexOf('windage_range.') === 0) {
                fieldName = 'windage_' + windage[fieldName];
            }

            if (fieldName) {
                field = this.$el.find('*[name="' + fieldName + '"]');
            }

            if (field) {
                this.showErrorForField(field, error);
                return;
            }

            SurfaceReleaseSpillFormView.__super__.handleFieldError.apply(this, arguments);
        }
    }, {
        CANCELED: 'surfaceReleaseSpillForm:canceled'
    });


    var AddSurfaceReleaseSpillFormView = SurfaceReleaseSpillFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    width: 400,
                    height: 550,
                    title: "Add Surface Release Spill"
                }
            }, options);

            AddSurfaceReleaseSpillFormView.__super__.initialize.apply(this, [opts]);
        },

        show: function(coords) {
            this.model = new models.SurfaceReleaseSpill(this.defaults);
            this.setupModelEvents();
            AddSurfaceReleaseSpillFormView.__super__.show.apply(this, arguments);
        }
    });


    var GnomeSettingsFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    height: 300,
                    width: 470
                }
            }, options);

            GnomeSettingsFormView.__super__.initialize.apply(this, [opts]);
        },

        // Always use the same model when reloading.
        reload: function() {
            GnomeSettingsFormView.__super__.reload.apply(this, [this.model.id]);
        },

        // Always return the same model.
        getModel: function(id) {
            return this.model;
        },

        // Never reset the model.
        resetModel: function() {
            return;
        },

        submit: function() {
            this.model.set('start_time', this.getFormDate(this.getForm()));
            GnomeSettingsFormView.__super__.submit.apply(this, arguments);
        },

        show: function() {
            GnomeSettingsFormView.__super__.show.apply(this, arguments);
            this.setDateFields('.start_time_container', moment(this.model.get('start_time')));
        }
    });


    var RandomMoverFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    height: 350,
                    width: 380,
                    title: "Edit Random Mover"
                }
            }, options);

            RandomMoverFormView.__super__.initialize.apply(this, [opts]);
        },

        show: function() {
            RandomMoverFormView.__super__.show.apply(this, arguments);
            this.setDateFields('.active_start_container', this.model.get('active_start'));
            this.setDateFields('.active_stop_container', this.model.get('active_stop'));
        },

        submit: function() {
            this.model.set('active_start',  this.getFormDate('.active_start_container'));
            this.model.set('active_stop',  this.getFormDate('.active_stop_container'));
            RandomMoverFormView.__super__.submit.apply(this, arguments);
        },

        getDataBindings: function() {
            return {mover: this.model}
        }
    });


    var AddRandomMoverFormView = RandomMoverFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    height: 350,
                    width: 380,
                    title: "Add Random Mover"
                }
            }, options);

            AddRandomMoverFormView.__super__.initialize.apply(this, [opts]);
        },

        show: function() {
            this.model = new models.RandomMover(this.defaults);
            this.setupModelEvents();
            AddRandomMoverFormView.__super__.show.apply(this, arguments);
        },

        submit: function() {
            this.collection.add(this.model);
            AddRandomMoverFormView.__super__.submit.apply(this, arguments);
        }
    });


    return {
        AddMapFormView: AddMapFormView,
        AddMoverFormView: AddMoverFormView,
        AddSpillFormView: AddSpillFormView,
        AddWindMoverFormView: AddWindMoverFormView,
        AddRandomMoverFormView: AddRandomMoverFormView,
        AddSurfaceReleaseSpillFormView: AddSurfaceReleaseSpillFormView,
        MapFormView: MapFormView,
        AddCustomMapFormView: AddCustomMapFormView,
        AddMapFromUploadFormView: AddMapFromUploadFormView,
        WindMoverFormView: WindMoverFormView,
        RandomMoverFormView: RandomMoverFormView,
        SurfaceReleaseSpillFormView: SurfaceReleaseSpillFormView,
        FormView: FormView,
        FormViewContainer: FormViewContainer,
        GnomeSettingsFormView: GnomeSettingsFormView,
        MultiStepFormView: MultiStepFormView,
        LocationFileWizardFormView: LocationFileWizardFormView,
        ModelNotFoundException: ModelNotFoundException
    };

});