
define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'util',
    'lib/geo',
    'lib/moment',
    'lib/compass-ui'
], function($, _, Backbone, models, util, geo) {

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
        },

        get: function(formId) {
            return this.formViews[formId];
        },

        hideAll: function() {
            _.each(this.formViews, function(formView, key) {
                formView.hide();
            });
        }
    });

    /*
     `FormView` is the base class for forms intended to wrap Backbone models.

     Submitting a form from `FormView` sends the value of its inputs to the
     model object passed into the form's constructor, which syncs it with the
     server.
     */
    var FormView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.wasCancelled = false;
            this.id = this.options.id;
            this.model = this.options.model;
            this.collection = this.options.collection;

            if (this.options.id) {
                this.$el = $('#' + this.options.id);
            }

            this.defaults = this.getFormData();
            this.setupDatePickers();
        },

        showErrorForField: function(field, error) {
            if (!field.length) {
                return;
            }

            var errorDiv = field.next('.help-inline.error');

            // If there is no error div, then report the error to the user.
            if (!errorDiv.length) {
                alert(error.description);
                return;
            }

            errorDiv.text(error.description);

            var group = field.closest('.control-group');

            if (!group.length) {
                group.addClass('error');
            }

            group.addClass('error');
        },

        handleFieldError: function(error) {
            var fieldId = this.getFieldIdForError(error);
            var field = this.$el.find(fieldId);

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

        prepareForm: function() {
            var form = this.getForm();

            if (this.model && this.model.id) {
                this.setForm(form, this.model.toJSON());
            } else {
                this.setForm(form, this.defaults);
            }
        },

        clearErrors: function() {
            var groups = this.$el.find('.control-group');
            var errors = this.$el.find('.help-inline.error');

            if (groups.length) {
                groups.removeClass('error');
            }

            if (errors.length) {
                errors.text('');
            }
        },

        show: function() {
            this.prepareForm();
            this.clearErrors();
            $('#main-content').addClass('hidden');
            this.$el.removeClass('hidden');
        },

        hide: function() {
            this.$el.addClass('hidden');
            $('#main-content').removeClass('hidden');
        },

        submit: function() {
            var data = this.getFormData();
            this.model.save(data);
        },

        cancel: function() {
            // Override this in your subclass.
        },

        hasDateFields: function(target) {
            var fields = this.getDateFields(target);
            return fields.date && fields.hour && fields.minute;
        },

        setForm: function(form, data) {
            _.each(data, function(dataVal, fieldName) {
                var input = form.find('#' + fieldName);

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

            if (!datetime || target.length === 0) {
                return;
            }

            var fields = this.getDateFields(target);
            fields.date.val(datetime.format("MM/DD/YYYY"));
            fields.hour.val(datetime.format('HH'));
            fields.minute.val(datetime.format('mm'));
        },

         getFormData: function() {
            var data = {};
            var inputs = this.$el.find('input');
            var checkboxes = this.$el.find('input:checkbox');

            _.each(inputs, function(field) {
                field = $(field);
                data[field.attr('name')] = field.val();
            });

            _.each(checkboxes, function(field) {
                field = $(field);
                data[field.attr('name')] = field.prop('checked');
            });

            return data;
        },

        clearForm: function() {
            var inputs = this.$el.find('select,input');

            if (!inputs.length) {
                return;
            }

            _.each(inputs, function(field) {
                $(field).val('');
            });
        },

        setupModelEvents: function() {
            var _this = this;
            this.model.on('error', this.handleServerError);
            this.model.on('sync', function() {
                _this.$el.dialog('close');
            });
        },

        reset: function() {
            this.model = null;
            this.clearForm();
        },

        reload: function(id) {
            var model;

            if (this.collection) {
                model = this.collection.get(id);
            }

            if (model) {
                this.model = model;
                this.setupModelEvents();
            }
        }
    }, {
        CANCELED: 'formView:canceled',
        REFRESHED: 'formView:refreshed'
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
                close: this.close
            }, options.dialog || this.dialog || {});

            this.$el.dialog(opts);

            // Workaround the default JQuery UI Dialog behavior of auto-
            // focusing on the first input element by adding an invisible one.
            $('<span class="ui-helper-hidden-accessible">' +
                '<input type="text"/></span>').prependTo(this.$el);
        },

        cancel: function() {
            this.$el.dialog("close");
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

        reload: function() {
            var _this = this;
            JQueryUIModalFormView.__super__.reload.apply(this, arguments);

            if (this.model) {
                this.model.on('sync', function() {
                    _this.$el.dialog('close');
                });
            }
        },

        /*
         Hide any other visible modals and show this one.
         */
        show: function() {
            this.prepareForm();
            this.clearErrors();
            this.$el.dialog('open');
            this.$el.removeClass('hide');
        },

        hide: function() {
            this.$el.dialog('close');
        },

        close: function() {
            // Override for custom behavior after the dialog has closed.
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
            var form = this.getForm();
            var moverType = form.find('select[name="mover_type"]').val();

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
        show: function(coords) {
            if (coords) {
                this.coords = coords;
            }

            AddSpillFormView.__super__.show.apply(this);
        },

        submit: function() {
            var form = this.getForm();
            var spillType = form.find('select[name="spill_type"]').val();

            if (spillType) {
                this.trigger(AddSpillFormView.SPILL_CHOSEN, spillType, this.coords);
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
            var option = this.$el.find('#map_file');
            var file = option.find('option:selected');
            if (file) {
                var data = {
                    name: file.text(),
                    filename: file.val(),
                    refloat_halflife: 6 * 3600 // TODO: Allow user to set?
                };
                this.model.save(data);
            }
        }
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
        },

        reload: function(id) {
            // Do nothing - we always use the same map object.
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

            this.defaults = this.getFormDefaults();
            this.windMovers = options.windMovers;
            this.setupCompass();

            // Extend prototype's events with ours.
            this.events = _.extend({}, FormView.prototype.events, this.events);
        },

        events: {
            'click .edit-mover-name': 'editMoverNameClicked',
            'click .save-mover-name': 'saveMoverNameButtonClicked',
            'click input[name="name"]': 'moverNameChanged',
            'click .add-time': 'addButtonClicked',
            'click .edit-time': 'editButtonClicked',
            'click .show-compass': 'showCompass',
            'click .cancel': 'cancelButtonClicked',
            'click .save': 'saveButtonClicked',
            'click .delete-time': 'trashButtonClicked',
            'change .units': 'renderTimeTable',
            'change .type': 'typeChanged'
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
                    OK: function () {
                        $(this).dialog("close");
                    }
                }
            });
        },

        setForm: function(form, data) {
            SurfaceReleaseSpillFormView.__super__.setForm.apply(this, arguments);

            var timeContainer = form.find('.datetime_container');
            var datetime = this.getFormDate(timeContainer);
            this.setDateFields(timeContainer, datetime);
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

                if (parts[1] === 'timeseries') {
                    valuesWithErrors.push(parts[2]);
                }
            });

            return valuesWithErrors;
        },

        clearInputs: function(form) {
            $(form).find(':input').each(function() {
                $(this).val('').prop('checked', false);
            });
        },

        renderTimeTable: function() {
            var _this = this;
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');
            var units = this.$el.find('.units').find('option:selected').val();
            var rows = [];
            var IdsWithErrors = this.getWindIdsWithErrors();

            // Clear out any existing rows.
            this.getTimesTable().find('tr').not('.table-header').remove();

            _.each(timeseries, function(windValue, index) {
                var tmpl = _.template($("#time-series-row").html());
                var direction = windValue.direction;
                var speed = windValue.speed;

                if (typeof(direction) === 'number') {
                    direction = direction.toFixed(1);
                }

                if (typeof(speed) === 'number') {
                    speed = speed.toFixed(1);
                }

                var datetime = moment(windValue.datetime);
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
         Prepare `data` with the wind timeseries values needed to save a
         constant wind mover.
         */
        prepareConstantWindData: function(data) {
            var wind = this.model.get('wind');
            var timeseries = _.clone(wind.get('timeseries'));
            var values = {
                // A 'datetime' field is required, but it will be ignored for a
                // constant wind mover during the model run, so we just use the
                // current time.
                datetime: moment().format(),
                direction: data.direction,
                speed: data.speed
            };

            if (timeseries.length === 1) {
                // Update an existing time series value.
                timeseries[0] = values
            } else {
                // Add the first (and only) time series value.
                timeseries = [values];
            }

            wind.set('timeseries', timeseries);

            delete(data.speed);
            delete(data.direction);

            return data
        },

        submit: function() {
            // Clear the add time form in the variable wind div as those
            // values must be "saved" in order to mean anything.
            var variable = $('.variable-wind');
            variable.find('input').val('');
            variable.find('input:checkbox').prop('checked', false);

            var data = this.getFormData();
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');

            wind.set('units', data.units);
            delete(data.units);

            if (this.$el.find('.type').val() === 'constant-wind'
                    && timeseries.length > 1) {

                var message = 'Changing this mover to use constant wind will ' +
                    'delete variable wind data. Go ahead?';

                if (!window.confirm(message)) {
                   return;
                }
            }

            // A constant wind mover has these values.
            if (data.direction !== undefined && data.speed !== undefined) {
                data = this.prepareConstantWindData(data);
            }

            this.collection.add(this.model);
            this.model.save(data);
        },

        editMoverNameClicked: function(event) {
            event.preventDefault();
            var link = $(event.target);
            var form = link.closest('.form');
            form.find('.top-form').removeClass('hidden');
            form.find('.page-header h3').addClass('hidden');
        },

        saveMoverNameButtonClicked: function(event) {
            event.preventDefault();
            var link = $(event.target);
            var form = link.closest('.form');
            form.find('.top-form').addClass('hidden');
            form.find('.page-header h3').removeClass('hidden');
        },

        moverNameChanged: function(event) {
            var input = $(event.target);
            var form = input.closest('.form');
            var header = form.find('.page-header').find('a');
            header.text(input.val());
        },

        modelChanged: function() {
            WindMoverFormView.__super__.modelChanged.apply(this, arguments);
            this.renderTimeTable();
            this.setupCompass();

            var hasErrors = this.$el.find(
                '.time-list').find('tr.error').length > 0;

            if (hasErrors) {
                alert('At least one of your time series values had errors in ' +
                     'it. The rows with errors will be marked in red.');
            }

            this.setupCompass();
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

        findDuplicate: function(timeseries, datetime, existingWindId) {
            var duplicate = timeseries.filter(function(time) {
                return time.datetime == datetime.format();
            });

            if (existingWindId) {
                duplicate = _.reject(duplicate, function(item) {
                    return item.cid === existingWindId;
                });
            }

            return duplicate;
        },

        saveButtonClicked: function(event) {
            event.preventDefault();
            var wind = this.model.get('wind');
            var timeseries = _.clone(wind.get('timeseries'));
            var addForm = this.getAddForm();
            var datetime = this.getFormDate(addForm);
            var windId = addForm.attr('data-wind-id');
            var direction = addForm.find('#direction').val();
            var duplicate = this.findDuplicate(timeseries, datetime, windId);
            var message = 'Wind data for that date and time exists. Replace it?';

            if (duplicate.length) {
                if (window.confirm(message)) {
                    timeseries.remove(duplicate);
                    this.renderTimeTable();
                } else {
                    return;
                }
            }

            timeseries[windId] = {
                datetime: datetime.format(),
                direction: this.getCardinalAngle(direction),
                speed: addForm.find('#speed').val()
            };

            wind.set('timeseries', timeseries);
            this.model.save();

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

        showEditFormForWind: function(windId) {
            var row = this.getRowForWindId(windId);
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');
            var windValue = timeseries[windId];
            var addForm = this.getAddForm();

            addForm.attr('data-wind-id', windId);
            addForm.find('.add-time-buttons').addClass('hidden');
            addForm.find('.edit-time-buttons').removeClass('hidden');
            this.setForm(addForm, windValue);
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
            var duplicate = this.findDuplicate(timeseries, datetime);
            var message = 'Wind data for that date and time exists. Replace it?';

            if (duplicate.length) {
                if (window.confirm(message)) {
                    timeseries.remove(duplicate);
                    this.renderTimeTable();
                } else {
                    return;
                }
            }

            timeseries.push({
                datetime: datetime.format(),
                direction: this.getCardinalAngle(direction),
                speed: addForm.find('#speed').val()
            });

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

        getBaseFormData: function() {
            return {
                name: this.$el.find('#name').val(),
                units: this.$el.find('#units').val(),
                on: this.$el.find('#on').prop('checked')
            };
        },

        getFormDataForDiv: function(div) {
            var data = {};
            var inputs = div.find('input,select');

            _.each(inputs, function(input) {
                input = $(input);
                var name = input.attr('name');
                var val;

                if (input.is(':checkbox')) {
                    val = input.prop('checked');
                } else {
                    val = input.val();
                }

                if (name && val !== undefined && val !== '') {
                    data[name] = val;
                }
            });

            return data;
        },

        /*
         Save the values in all form fields in an object that can be used later
         to look up field defaults.
         */
        getFormDefaults: function() {
            var data = this.getBaseFormData();
            var constant = this.$el.find('.constant-wind');
            var variable = this.$el.find('.variable-wind');
            var uncertainty = this.$el.find('.uncertainty');

            data['moverTypes'] = {
                'constant-wind': this.getFormDataForDiv(constant),
                'variable-wind': this.getFormDataForDiv(variable)
            };

            data['uncertainty'] = this.getFormDataForDiv(uncertainty);
            data['defaultWindDate'] = moment(this.$el.find('.datetime').val());

            return data;
        },

        /*
         Get the values of all form fields in an object that will be passed
         directly to a model object to be saved to the server.
         */
        getFormData: function() {
            var data = this.getBaseFormData();
            var moverTypeDiv = this.getMoverTypeDiv();
            var divData = this.getFormDataForDiv(moverTypeDiv);

            var uncertainty = this.$el.find('.uncertainty');
            var uncertaintyData = this.getFormDataForDiv(uncertainty);
            data = $.extend(data, divData, uncertaintyData);

            var activeRange = this.$el.find('.active_range');
            var activeRangeData = this.getFormDataForDiv(activeRange);
            data = $.extend(data, divData, activeRangeData);

            return data;
        },

        /*
         Set all fields for which a value exists in the object created by
         `getformDefaults`, in `self.defaults`.
         */
        setFormDefaults: function() {
            var _this = this;
            var data = this.defaults;

            this.$el.find('#name').val(data.name);
            this.$el.find('#on').prop('checked', data.on);
            this.$el.find('#units').val(data.units);
            this.setDateFields(
                this.$el.find('.datetime-container'), data['defaultWindDate']);

            _.each(data['moverTypes'], function(moverData, typeName) {
                var form = _this.getAddForm(typeName);

                if (!form.length) {
                    return;
                }

                _this.setForm(form, moverData);
            });

            var uncertainty = this.$el.find('.uncertainty');
            this.setForm(uncertainty, data['uncertainty']);

            // Reset error states
            this.$el.find('.control-group').removeClass('error');
            this.$el.find('.help-inline.error').text('');
        },

        /*
         Set all fields with the current values of `self.model`.
         */
        setInputsFromModel: function() {
            var wind = this.model.get('wind');

            this.$el.find('#name').val(this.model.get('name'));
            this.$el.find('#active').prop('checked', this.model.get('active'));
            this.$el.find('#units').val(wind.get('units'));
            this.$el.find('#on').prop('checked', this.model.get('on'));

            this.setDateFields(this.$el.find('.active_start_container'),
                               this.model.get('active_start'));
            this.setDateFields(this.$el.find('.active_stop_container'),
                               this.model.get('active_stop'));

            var moverType = this.$el.find('.type');
            var timeseries = wind.get('timeseries');
            var firstTimeValue = timeseries[0];

            if (timeseries.length > 1) {
                moverType.val('variable-wind');
            } else {
                moverType.val('constant-wind');
            }

            this.typeChanged();

            var constantAddForm = this.getAddForm('constant-wind');

            if (firstTimeValue) {
                this.setForm(constantAddForm, firstTimeValue);
            }
        },

        close: function() {
            this.model = null;
            WindMoverFormView.__super__.close.apply(this, arguments);
        },

        /*
         Prepare this form for display. Usually called just before the form is
         */
        prepareForm: function() {
            if (this.model === undefined) {
                window.alert('That mover was not found. Please refresh the page.')
                console.log('Mover undefined.');
                return;
            }

            this.renderTimeTable();
            this.setFormDefaults();

            if (this.model.id) {
                this.setInputsFromModel();
            }
        },

        handleFieldError: function(error) {
            if (error.name.indexOf('wind.') === 0) {
                var parts = error.name.split('.');
                var fieldName = parts[3];
                var form = this.getAddForm();
                var field = form.find('.' + fieldName);

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
            var wind = this.model.get('wind');

            if (wind.previousTimeseries) {
                wind.set('timeseries', wind.previousTimeseries);
            }

            var windIdsWithErrors = this.getWindIdsWithErrors();

            this.renderTimeTable();

            if (windIdsWithErrors.length) {
                this.showEditFormForWind(windIdsWithErrors[0]);
            }

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
         */
        show: function() {
            this.model = new models.WindMover();
            this.setupModelEvents();
            AddWindMoverFormView.__super__.show.apply(this);
        },

        /*
         Reset the form when it closes and clear out the model.
        */
        close: function() {
            this.model = null;
            this.clearForm();
        }
    });


    var SurfaceReleaseSpillFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    width: 400,
                    height: 420,
                    title: "Edit Surface Release Spill"
                }
            }, options);

            SurfaceReleaseSpillFormView.__super__.initialize.apply(this, [opts]);

            // Extend prototype's events with ours.
            this.events = _.extend({}, FormView.prototype.events, this.events);
            this.surfaceReleaseSpills = options.surfaceReleaseSpills;
            this.defaults = this.getFormData();
        },

        events: {
            'click .edit-spill-name': 'editSpillNameClicked',
            'click .save-spill-name': 'saveSpillNameButtonClicked',
            'change input[name="name"]': 'spillNameChanged'
        },

        setForm: function(form, data) {
            if (_.has(data, 'windage_range')) {
                data['windage_min'] = data['windage_range'][0];
                data['windage_max'] = data['windage_range'][1];
            }

            if (_.has(data, 'start_position')) {
                var pos = data['start_position'];
                data['start_position_x'] = pos[0];
                data['start_position_y'] = pos[1];
                data['start_position_z'] = pos[2];
            }

            SurfaceReleaseSpillFormView.__super__.setForm.apply(this, arguments);

            var timeContainer = form.find('.release_time_container');
            var releaseTime = this.getFormDate(timeContainer);
            this.setDateFields(timeContainer, releaseTime);
        },

        show: function(coords) {
            if (coords) {
                var coordInputs = this.$el.find('.coordinate');
                $(coordInputs[0]).val(coords[0]);
                $(coordInputs[1]).val(coords[1]);
            }

            SurfaceReleaseSpillFormView.__super__.show.apply(this, arguments);
        },

        editSpillNameClicked: function(event) {
            event.preventDefault();
            var link = $(event.target);
            var form = link.closest('.form');
            form.find('.top-form').removeClass('hidden');
            form.find('.page-header h3').addClass('hidden');
        },

        saveSpillNameButtonClicked: function(event) {
            event.preventDefault();
            var link = $(event.target);
            var form = link.closest('.form');
            form.find('.top-form').addClass('hidden');
            form.find('.page-header h3').removeClass('hidden');
        },

        spillNameChanged: function(event) {
            var input = $(event.target);
            var form = input.closest('.form');
            var header = form.find('.page-header').find('a');
            header.text(input.val());
        },

        submit: function() {
            var data = this.getFormData();

            data['release_time'] = this.getFormDate(this.getForm());
            data['is_active'] = data['is_active'];
            data['windage_range'] = [
                data['windage_min'], data['windage_max']
            ];
            data['start_position'] = [
                data['start_position_x'], data['start_position_y'],
                data['start_position_z']
            ];

            this.collection.add(this.model);
            this.model.save(data);
        },

        cancel: function() {
            this.trigger(SurfaceReleaseSpillFormView.CANCELED, this);
            SurfaceReleaseSpillFormView.__super__.cancel.apply(this, arguments);
        }
    }, {
        CANCELED: 'surfaceReleaseSpillForm:canceled'
    });


    var AddSurfaceReleaseSpillFormView = SurfaceReleaseSpillFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    width: 400,
                    height: 420,
                    title: "Add Surface Release Spill"
                }
            }, options);

            AddSurfaceReleaseSpillFormView.__super__.initialize.apply(this, [opts]);
        },

        show: function(coords) {
            this.model = new models.SurfaceReleaseSpill();
            this.setupModelEvents();
            AddSurfaceReleaseSpillFormView.__super__.show.apply(this, arguments);

            if (coords) {
                this.$el.find('#start_position_x').val(coords[0]);
                this.$el.find('#start_position_y').val(coords[1]);
            }
        },

        close: function() {
            this.model = null;
            this.clearForm();
        }
    });


    var ModelSettingsFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    height: 300,
                    width: 470
                }
            }, options);

            ModelSettingsFormView.__super__.initialize.apply(this, [opts]);
        },

        submit: function() {
            var data = this.getFormData();
            data['start_time'] = this.getFormDate(this.getForm());
            this.model.save(data);
        },

        show: function() {
            ModelSettingsFormView.__super__.show.apply(this, arguments);
            this.setDateFields(
                this.$el.find('.start_time_container'), this.getFormDate(this.$el));
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
            var isActiveStart = this.$el.find('.active_start_container');
            var isActiveStop = this.$el.find('.active_stop_container');
            this.setDateFields(isActiveStart, this.getFormDate(isActiveStart));
            this.setDateFields(isActiveStop, this.getFormDate(isActiveStop));
        },

        submit: function() {
            var data = this.getFormData();
            var isActiveStart = this.$el.find('.active_start_container');
            var isActiveStop = this.$el.find('.active_stop_container');

            data['active_start'] = this.getFormDate(isActiveStart);
            data['active_stop'] = this.getFormDate(isActiveStop);

            this.model.save(data);
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
            this.model = new models.RandomMover();
            this.setupModelEvents();

            AddRandomMoverFormView.__super__.show.apply(this, arguments);
        },

        close: function() {
            this.model = null;
            this.clearForm();
        },

        submit: function() {
            var data = this.getFormData();
            this.collection.add(this.model);
            this.model.save(data);
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
        WindMoverFormView: WindMoverFormView,
        RandomMoverFormView: RandomMoverFormView,
        SurfaceReleaseSpillFormView: SurfaceReleaseSpillFormView,
        FormView: FormView,
        FormViewContainer: FormViewContainer,
        ModelSettingsFormView: ModelSettingsFormView
    };

});