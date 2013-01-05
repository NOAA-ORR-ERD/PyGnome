
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

        add: function(id, view) {
            var _this = this;
            this.formViews[id] = view;
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
     `FormView` is responsible for displaying HTML forms rendered on the server.
     It listens to 'change' events on a bound model and refreshes itself when
     that event fires.

     The view is designed to handle multi-step forms implemented purely in
     JavaScript (and HTML) using data- properties on DOM elements. The server
     returns one rendered form, but may split its content into several <div>s, each
     with a `data-step` property. If a form is structured this way, the user of the
     JavaScript application will see it as a multi-step form with "next," "back"
     and (at the end) a "save" or "create" button (the label is given by the server,
     but whatever it is, this is the button that signals final submission of the
     form).

     Submitting a form from `FormView` sends the value of its inputs to the
     model object passed into the form's constructor, which syncs it with the
     server.
     */
    var FormView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.wasCancelled = false;
            this.container = $(this.options.formContainerEl);
            this.id = this.options.id;
            this.model = this.options.model;
            if(this.model) {
                this.model.on("update", this.modelChanged);
            }
            this.setupDatePickers();
        },

        events: {
            'click .btn-next': 'goToNextStep',
            'click .btn-prev': 'goToPreviousStep'
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

        modelChanged: function(model) {
            this.refresh();
            this.trigger(FormView.REFRESHED, this);

            if (this.wasCancelled) {
                this.wasCancelled = false;
                return;
            }

            this.show();
        },

        show: function() {
            $('#main-content').addClass('hidden');
            this.$el.removeClass('hidden');
        },

        hide: function() {
            this.$el.addClass('hidden');
            $('#main-content').removeClass('hidden');
        },

        getForm: function() {
            return this.$el.find('form');
        },

        getFirstTabWithError: function() {
            if (this.getForm().find('.nav-tabs').length === 0) {
                return null;
            }

            var errorDiv = $('div.control-group.error').first();
            var tabDiv = errorDiv.closest('.tab-pane');

            if (tabDiv.length) {
                return tabDiv.attr('id');
            }
        },

        getFirstStepWithError: function() {
            var step = 1;

            if (!this.getForm().hasClass('multistep')) {
                return null;
            }

            var errorDiv = $('div.control-group.error').first();
            var stepDiv = errorDiv.closest('div.step');

            if (stepDiv === false) {
                step = stepDiv.attr('data-step');
            }

            return step;
        },

        getStep: function(stepNum) {
            return this.getForm().find('div[data-step="' + stepNum  + '"]').length > 0;
        },

        previousStepExists: function(stepNum) {
           return this.getStep(stepNum - 1);
        },

        nextStepExists: function(stepNum) {
            stepNum = parseInt(stepNum, 10);
            return this.getStep(stepNum + 1);
        },

        goToStep: function(stepNum) {
            var form = this.getForm();

            if (!form.hasClass('multistep')) {
                return;
            }

            var stepDiv = form.find('div.step[data-step="' + stepNum + '"]');

            if (stepDiv.length === 0) {
                return;
            }

            var otherStepDivs = form.find('div.step');
            otherStepDivs.addClass('hidden');
            otherStepDivs.removeClass('active');
            stepDiv.removeClass('hidden');
            stepDiv.addClass('active');

            var prevButton = this.container.find('.btn-prev');
            var nextButton = this.container.find('.btn-next');
            var saveButton = this.container.find('.btn-primary');

            if (this.previousStepExists(stepNum)) {
                prevButton.removeClass('hidden');
            } else {
                prevButton.addClass('hidden');
            }

            if (this.nextStepExists(stepNum)) {
                nextButton.removeClass('hidden');
                saveButton.addClass('hidden');
                return;
            }

            nextButton.addClass('hidden');
            saveButton.removeClass('hidden');
        },

        goToNextStep: function() {
            var form = this.getForm();

            if (!form.hasClass('multistep')) {
                return;
            }

            var activeStepDiv = form.find('div.step.active');
            var currentStep = parseInt(activeStepDiv.attr('data-step'), 10);
            this.goToStep(currentStep + 1);
        },

        goToPreviousStep: function(event) {
            var form = this.getForm();

            if (!form.hasClass('multistep')) {
                return;
            }

            var activeStep = form.find('div.step.active');
            var currentStep = parseInt(activeStep.attr('data-step'), 10);
            this.goToStep(currentStep - 1);
        },

        submit: function(event) {
            event.preventDefault();
            var form = this.getForm();
            console.log('submit', form.serialize());
            this.hide();
            return false;
        },

        cancel: function(event) {
            event.preventDefault();
            this.hide();
            this.wasCancelled = true;
            this.reload();
            this.trigger(FormView.CANCELED, this);
        },

        /*
         Refresh this form from the latest values of `this.model`.
         */
        refresh: function() {
            console.log('refresh', this.model);

            var stepWithError = this.getFirstStepWithError();
            if (stepWithError) {
                this.goToStep(stepWithError);
            }

            var tabWithError = this.getFirstTabWithError();
            if (tabWithError) {
                $('a[href="#' + tabWithError + '"]').tab('show');
            }
        },
    }, {
        CANCELED: 'ajaxFormView:canceled',
        REFRESHED: 'ajaxFormView:refreshed'
    });


    /*
     An `FormView` subclass that displays in a modal window.
     */
    var ModalFormView = FormView.extend({
        /*
         Hide any other visible modals and show this one.
         */
        show: function() {
            $('div.modal').modal('hide');
            this.$el.modal();
        },

        hide: function() {
            this.$el.modal('hide');
        }
    });


     /*
     An `FormView` subclass that displays in a JQuery UI modal window.
     */
    var JQueryUIModalFormView = FormView.extend({
        initialize: function(options) {
            JQueryUIModalFormView.__super__.initialize.apply(this, [options]);
            var _this = this;

            var opts = $.extend({
                autoOpen: false,
                buttons: {
                    Cancel: function() {
                        $(this).dialog("close");
                    },

                    "Save": function() {
                        _this.submit();
                        $(this).dialog("close");
                    }
                },
                close: this.close
            }, options.dialog || {});

            this.$el.dialog(opts);

            // A workaround for the default JQuery UI Dialog behavior of auto-
            // focusing on the first input element.
            $('<span class="ui-helper-hidden-accessible">' +
                '<input type="text"/></span>').prependTo(this.$el);
        },

        /*
         Hide any other visible modals and show this one.
         */
        show: function() {
            $('div.modal').modal('hide');
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
     This is a non-AJAX-enabled modal form object to support the "add mover" form,
     which asks the user to choose a type of mover to add. We then use the selection
     to display another, mover-specific form.
     */
    var AddMoverFormView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.container = $(this.options.formContainerEl);
        },

        events: {
            'click .btn-primary': 'submit'
        },

        getForm: function() {
            return this.$el.find('form');
        },

        show: function() {
            this.$el.modal();
        },

        hide: function() {
            this.$el.modal('hide');
        },

        submit: function(event) {
            event.preventDefault();
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
    var AddSpillFormView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.container = $(this.options.formContainerEl);
        },

        events: {
            'click .btn-primary': 'submit',
            'click .cancel': 'cancel'
        },

        getForm: function() {
            return this.$el.find('form');
        },

        show: function(coords) {
            if (coords) {
                this.coords = coords;
            }

            this.$el.modal();
        },

        hide: function() {
            this.$el.modal('hide');
        },

        submit: function(event) {
            event.preventDefault();
            var form = this.getForm();
            var spillType = form.find('select[name="spill_type"]').val();

            if (spillType) {
                this.trigger(AddSpillFormView.SPILL_CHOSEN, spillType, this.coords);
                this.coords = null;
                this.hide();
            }

            return false;
        },

        cancel: function(event) {
            this.trigger(AddSpillFormView.CANCELED, this);
        }
    }, {
        // Event constants
        SPILL_CHOSEN: 'addSpillFormView:spillChosen',
        CANCELED: 'addSpillFormView:canceled'
    });


    /*
     `WindMoverFormView` handles the WindMover form.
     */
    var WindMoverFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    width: 900,
                    height: 710,
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
            'click .cancel': 'cancelButtonClicked',
            'click .save': 'saveButtonClicked',
            'click .delete-time': 'trashButtonClicked',
            'change .units': 'renderTimeTable',
            'change .type': 'typeChanged'
        },

        typeChanged: function() {
            var type = this.$el.find('.type').val();

            if (type === 'constant') {
                this.$el.find('.constant-wind').removeClass('hidden');
                this.$el.find('.variable-wind').addClass('hidden');
            } else {
                this.$el.find('.constant-wind').addClass('hidden');
                this.$el.find('.variable-wind').removeClass('hidden');
            }
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
        },

        getTimesTable: function() {
            return this.$el.find('.time-list');
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

            // Clear out any existing rows.
            this.getTimesTable().find('tr').not('.table-header').remove();

            timeseries.forEach(function(windValue) {
                var tmpl = _.template($("#time-series-row").html());
                var direction = windValue.get('direction');
                var speed = windValue.get('speed');

                if (typeof(direction) === 'number') {
                    direction = direction.toFixed(1);
                }

                if (typeof(speed) === 'number') {
                    speed = speed.toFixed(1);
                }

                var dateTime = moment(windValue.get('datetime'));
                // TODO: Error handling
                var error = null;

                rows.push($(tmpl({
                    error: error ? 'error' : '',
                    date: dateTime.format('MM/DD/YYYY'),
                    time: dateTime.format('HH:mm'),
                    direction: direction + ' &deg;',
                    speed: speed + ' ' + units
                })).data('data-wind-id', windValue.cid));
            });

            // Sort table by date and time of each item.
            rows = _.sortBy(rows, function(tr) {
                var date = tr.find('.time-series-date').text();
                var time = tr.find(
                    '.time-series-time').text().replace(' ', '', 'g');
                return Date.parse(date + ' ' + time)
            });

            _.each(rows, function(row) {
                row.appendTo(_this.getTimesTable());
            });
        },

        submit: function() {
            var data = this.getFormData();
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');

            if (timeseries.length < 2) {
                if (!data.direction || !data.speed) {
                    alert('You must enter either a direction and speed for ' +
                        'a constant wind mover, or at least one wind value ' +
                        'for a variable wind mover before saving.')
                }

                var values = {
                    // The server will ignore a datetime on a single timeseries.
                    datetime: moment().format(),
                    direction: data.direction,
                    speed: data.speed
                };

                if (timeseries.length === 1) {
                    // Update an existing time series value.
                    var time = timeseries.at(0);
                    time.set(values);
                } else {
                    // Add the first (and only) time series value.
                    timeseries.add(values);
                }

                delete(data.speed);
                delete(data.direction);
            }

            wind.set('units', data.units);
            delete(data.units);
            this.model.set(data);
            this.model.save();
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
            type = type || this.getMoverType() + '-wind';
            return this.$el.find('.' + type);
        },

        getAddForm: function(type) {
            return this.getMoverTypeDiv(type).find('.add-time-form');
        },

        trashButtonClicked: function(event) {
            event.preventDefault();
            var windId = $(event.target).closest('tr').data('data-wind-id');
            var winds = this.model.get('winds');
            var wind = winds.getByCid(windId);
            var addForm = this.getAddForm();

            if (addForm.data('data-wind-id') === wind.cid) {
                console.log('Edit form is visible for this wind.');
                this.setFormDefaults();
                addForm.find('.add-time-buttons').removeClass('hidden');
                addForm.find('.edit-time-buttons').addClass('hidden');
            }

            winds.remove(wind);
            this.renderTimeTable();
        },

        getCardinalAngle: function(value) {
            if (value && isNaN(value)) {
                value = util.cardinalAngle(value);
            }

            return value;
        },

        saveButtonClicked: function(event) {
            event.preventDefault();
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');
            var addForm = this.getAddForm();
            var dateTime = this.getFormDate(addForm);
            var windId = addForm.data('data-wind-id');
            var windValue = timeseries.getByCid(windId);
            var direction = addForm.find('#direction').val();

            windValue.set({
                datetime: dateTime.format(),
                direction: this.getCardinalAngle(direction),
                speed: addForm.find('#speed').val()
            });

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

        editButtonClicked: function(event) {
            event.preventDefault();
            var row = $(event.target).closest('tr');
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');
            var windId = row.data('data-wind-id');
            var windValue = timeseries.getByCid(windId);
            var addForm = this.getAddForm();

            this.clearInputs(addForm);
            addForm.data('data-wind-id', windValue.cid);
            addForm.find('.add-time-buttons').addClass('hidden');
            addForm.find('.edit-time-buttons').removeClass('hidden');
            this.setForm(addForm, windValue.toJSON());
            this.getTimesTable().find('tr').removeClass('info');
            row.removeClass('error').removeClass('warning').addClass('info');
        },
        
        /*
         Clone the add time form and add an item to the table of time series.
         */
        addButtonClicked: function(event) {
            event.preventDefault();
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');
            var addForm = this.getAddForm();
            var dateTime = this.getFormDate(addForm);
            var direction = addForm.find('#direction').val();

            timeseries.add({
                datetime: dateTime.format(),
                direction: this.getCardinalAngle(direction),
                speed: addForm.find('#speed').val()
            });

            wind.set('timeseries', timeseries);
            this.renderTimeTable();
            this.compass.compassUI('reset');

            var autoIncrementBy = addForm.find('.auto_increment_by').val();

            // Increase the date and time on the Add form if 'auto increase by'
            // value was provided.
            if (autoIncrementBy) {
                dateTime.add('hours', autoIncrementBy);
                this.setFormDate(addForm, dateTime);
            }
        },

        getBaseFormData: function() {
            return {
                name: this.$el.find('#name').val(),
                units: this.$el.find('#units').val(),
                is_active: this.$el.find('#is_active').prop('checked')
            };
        },
        
        getFormDateFields: function(form) {
            return {
                date: form.find('.date'),
                hour: form.find('.hour'),
                minute: form.find('.minute')
            }
        },
        
        getFormDate: function(form) {
            var fields = this.getFormDateFields(form);
            var date = fields.date.val();
            var hour = fields.hour.val();
            var minute = fields.minute.val();

            if (hour && minute) {
                date = date + ' ' + hour + ':' + minute;
            }

            // TODO: Handle a date-parsing error here.
            if (date) {
                return moment(date).local();
            }
        },
        
        setFormDate: function(form, dateTime) {
            if (!dateTime) {
                return;
            }
            var fields = this.getFormDateFields(form);
            fields.date.val(dateTime.format("MM/DD/YYYY"));
            fields.hour.val(dateTime.format('HH'));
            fields.minute.val(dateTime.format('mm'));
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

            return data;
        },

        getFormData: function() {
            // Clear the add time form in the variable wind div as those
            // values must be "saved" in order to mean anything.
            var variable = $('.variable-wind');
            variable.find('input').val('');
            variable.find('input:checkbox').prop('checked', false);

            var data = this.getBaseFormData();
            var moverTypeDiv = this.getMoverTypeDiv();
            var divData = this.getFormDataForDiv(moverTypeDiv);
            var uncertainty = this.$el.find('.uncertainty');
            var uncertaintyData = this.getFormDataForDiv(uncertainty);
            data = $.extend(data, divData, uncertaintyData);
            return data;
        },

        formHasDateFields: function(form) {
            var fields = this.getFormDateFields(form);
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

            // Include only the date in the date field, reformat it, and move
            // time values into the hour and minute fields.
            if (this.formHasDateFields(form)) {
                var dateTime = this.getFormDate(form);
                this.setFormDate(form, dateTime);
            }
        },

        setFormDefaults: function() {
            var _this = this;
            var data = this.defaults;

            this.$el.find('#name').val(data.name);
            this.$el.find('#is_active').prop('checked', data.is_active);
            this.$el.find('#units').val(data.units);

            _.each(data['moverTypes'], function(moverData, typeName) {
                var form = _this.getAddForm(typeName);

                if (!form.length) {
                    return;
                }

                _this.setForm(form, moverData);
            });

            var uncertainty = this.$el.find('.uncertainty');
            this.setForm(uncertainty, data['uncertainty']);

            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');

            // A constant wind mover
            if (timeseries && timeseries.length === 1) {
                var addForm = this.getAddForm();
                var formData = timeseries.at(0).toJSON();
                this.setForm(addForm, formData);
            }
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

        reset: function() {
            this.model = null;
            this.clearForm();
        },

        reload: function(mover_id) {
            var mover = this.windMovers.get(mover_id);
            if (mover) {
                this.model = mover;
            }
        },

        show: function() {
            this.renderTimeTable();
            this.setFormDefaults();
            var timeseries = this.model.get('wind').get('timeseries');
            if (timeseries.length > 1) {
                this.$el.find('.type').val('variable');
            }
            this.typeChanged();
            WindMoverFormView.__super__.show.apply(this, arguments);
        }
    });


    var AddWindMoverFormView = WindMoverFormView.extend({
        initialize: function(options) {
            this.windMovers = options.windMovers;

            var opts = _.extend({
                dialog: {
                    width: 900,
                    height: 710,
                    title: "Add Wind Mover"
                },
                model: this.newMover()
            }, options);

            AddWindMoverFormView.__super__.initialize.apply(this, [opts]);
        },

        deleteMover: function() {
            if (this.mover) {
                this.windMovers.remove(this.mover);
            }
        },

        newMover: function() {
            this.deleteMover();
            var mover = new models.WindMover();
            this.windMovers.add(mover);
            return mover;
        },

        /*
         Use a new WindMover every time the form is opened.
         */
        show: function() {
            this.model = this.newMover();
            AddWindMoverFormView.__super__.show.apply(this);
        },

        reset: function() {
            this.deleteMover();
            this.clearForm();
        },

        /*
         Reset the form when it closes and clear out the model.
        */
        close: function() {
            this.reset();
        }
    });


    var PointReleaseSpillFormView = JQueryUIModalFormView.extend({
         initialize: function(options) {
             var opts = _.extend({
                 dialog: {
                     width: 400,
                     height: 420,
                     title: "Edit Point Release Spill"
                 }
            }, options);

            PointReleaseSpillFormView.__super__.initialize.apply(this, [opts]);

            // Extend prototype's events with ours.
            this.events = _.extend({}, FormView.prototype.events, this.events);
        },

        show: function(coords) {
            PointReleaseSpillFormView.__super__.show.apply(this);

            if (coords) {
                var coordInputs = this.$el.find('.coordinate');
                $(coordInputs[0]).val(coords[0]);
                $(coordInputs[1]).val(coords[1]);
            }
        },

        events: {
            'click .edit-spill-name': 'editSpillNameClicked',
            'click .save-spill-name': 'saveSpillNameButtonClicked',
            'change input[name="name"]': 'spillNameChanged'
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
        }
    });


    var AddPointReleaseSpillFormView = PointReleaseSpillFormView.extend({
        submit: function() {
            console.log('add point release form submit');
        }
    });


    return {
        AddMoverFormView: AddMoverFormView,
        AddSpillFormView: AddSpillFormView,
        AddWindMoverFormView: AddWindMoverFormView,
        AddPointReleaseSpillFormView: AddPointReleaseSpillFormView,
        WindMoverFormView: WindMoverFormView,
        PointReleaseSpillFormView: PointReleaseSpillFormView,
        FormView: FormView,
        ModalFormView: ModalFormView,
        FormViewContainer: FormViewContainer,
        JQueryUIModalFormView: JQueryUIModalFormView
    };

});