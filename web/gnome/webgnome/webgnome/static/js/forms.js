
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
            this.setupDatePickers();

            if (this.options.id) {
                this.$el = $('#' + this.options.id);
            }
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

        show: function() {
            $('#main-content').addClass('hidden');
            this.$el.removeClass('hidden');
        },

        hide: function() {
            this.$el.addClass('hidden');
            $('#main-content').removeClass('hidden');
        },

        submit: function() {
            // Override this in your subclass.
        },

        cancel: function() {
            // Override this in your subclass.
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
                var datetime = this.getFormDate(form);
                this.setFormDate(form, datetime);
            }
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
                return moment(date);
            }
        },

        setFormDate: function(form, datetime) {
            if (!datetime) {
                return;
            }
            var fields = this.getFormDateFields(form);
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

        reset: function() {
            this.model = null;
            this.clearForm();
        },

        reload: function(id) {
            var model = this.collection.get(id);
            if (model) {
                this.model = model;
            }
        }
    }, {
        CANCELED: 'ajaxFormView:canceled',
        REFRESHED: 'ajaxFormView:refreshed'
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
                        $(this).dialog("close");
                    },

                    Save: function() {
                        _this.submit();
                        $(this).dialog("close");
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

        /*
         Hide any other visible modals and show this one.
         */
        show: function() {
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


    var ChooseObjectTypeFormView = JQueryUIModalFormView.extend({
        initialize: function(options) {
            var _this = this;
            var opts = _.extend({
                dialog: {
                    height: 175,
                    width: 400,
                    buttons: {
                        Cancel: function() {
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
                this.model.set('name', file.text());
                this.model.set('filename', file.val());
                this.model.set('refloat_halflife', 6 * 3600);
                this.model.save();
                console.log('saved')
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
                    width: 850,
                    height: 705,
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
            var typeDiv = this.$el.find('.' + type);
            var otherDivs = this.$el.find(
                '.tab-pane.wind > div').not(typeDiv);

            console.log(otherDivs.length)
            typeDiv.removeClass('hidden');
            otherDivs.addClass('hidden');
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

                var datetime = moment(windValue.get('datetime'));
                // TODO: Error handling
                var error = null;

                rows.push($(tmpl({
                    error: error ? 'error' : '',
                    date: datetime.format('MM/DD/YYYY'),
                    time: datetime.format('HH:mm'),
                    direction: direction + ' &deg;',
                    speed: speed + ' ' + units
                })).data('data-wind-id', windValue.cid));
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
            var timeseries = this.model.get('wind').get('timeseries');
            var values = {
                // A 'datetime' field is required, but it will be ignored for a
                // constant wind mover during the model run, so we just use the
                // current time.
                datetime: moment(),
                direction: data.direction,
                speed: data.speed
            };

            if (timeseries.length === 1) {
                // Update an existing time series value.
                var time = timeseries.at(0);
                time.set(values);
            } else {
                // Add the first (and only) time series value.
                timeseries.reset([]);
                timeseries.add(values);
            }

            delete(data.speed);
            delete(data.direction);

            return data
        },

        submit: function() {
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

            this.model.set(data);
            this.collection.add(this.model);
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
            type = type || this.getMoverType();
            return this.$el.find('.' + type);
        },

        getAddForm: function(type) {
            return this.getMoverTypeDiv(type).find('.add-time-form');
        },

        trashButtonClicked: function(event) {
            event.preventDefault();
            var windId = $(event.target).closest('tr').data('data-wind-id');
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');
            var windValue = timeseries.getByCid(windId);
            var addForm = this.getAddForm();

            if (addForm.data('data-wind-id') === windValue.cid) {
                this.setFormDefaults();
                addForm.find('.add-time-buttons').removeClass('hidden');
                addForm.find('.edit-time-buttons').addClass('hidden');
            }

            timeseries.remove(windValue);
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
                return time.get('datetime').format() == datetime.format();
            });

            window.timeseries = timeseries;

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
            var timeseries = wind.get('timeseries');
            var addForm = this.getAddForm();
            var datetime = this.getFormDate(addForm);
            var windId = addForm.data('data-wind-id');
            var windValue = timeseries.getByCid(windId);
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

            windValue.set({
                datetime: datetime,
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

            addForm.data('data-wind-id', windValue.cid);
            addForm.find('.add-time-buttons').addClass('hidden');
            addForm.find('.edit-time-buttons').removeClass('hidden');
            this.setForm(addForm, windValue.toJSON());
            this.getTimesTable().find('tr').removeClass('info');
            row.removeClass('error').removeClass('warning').addClass('info');
        },
        
        addButtonClicked: function(event) {
            event.preventDefault();
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');
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

            timeseries.add({
                datetime: datetime,
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
                this.setFormDate(addForm, nextDatetime);
            }
        },

        getBaseFormData: function() {
            return {
                name: this.$el.find('#name').val(),
                units: this.$el.find('#units').val(),
                is_active: this.$el.find('#is_active').prop('checked')
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
        },

        setInputsFromModel: function() {
            var wind = this.model.get('wind');
            var timeseries = wind.get('timeseries');

            this.$el.find('#name').val(this.model.get('name'));
            this.$el.find('#is_active').prop('checked', this.model.get('active'));
            this.$el.find('#units').val(wind.get('units'));

            var constantAddForm = this.getAddForm('constant-wind');
            var firstTimeValue = timeseries.at(0);

            var moverType = this.$el.find('.type');

            if (timeseries.length > 1) {
                moverType.val('variable-wind');
            } else {
                moverType.val('constant-wind');
            }

            this.typeChanged();

            if (firstTimeValue) {
                var formData = firstTimeValue.toJSON();
                this.setForm(constantAddForm, formData);
            }
        },

        close: function() {
            this.model = null;
            WindMoverFormView.__super__.close.apply(this, arguments);
        },

        show: function() {
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

            WindMoverFormView.__super__.show.apply(this, arguments);
        }
    });


    var AddWindMoverFormView = WindMoverFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    width: 850,
                    height: 705,
                    title: "Add Wind Mover"
                }
            }, options);

            AddWindMoverFormView.__super__.initialize.apply(this, [opts]);
        },

        /*
         Use a new WindMover every time the form is opened.
         */
        show: function() {
            this.model = null;
            this.model = new models.WindMover();
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
            this.pointReleaseSpills = options.pointReleaseSpills;
            this.defaults = this.getFormData();
        },

        events: {
            'click .edit-spill-name': 'editSpillNameClicked',
            'click .save-spill-name': 'saveSpillNameButtonClicked',
            'change input[name="name"]': 'spillNameChanged'
        },

        setForm: function(form, data) {
            if (_.has(data, 'windage')) {
                data['windage_min'] = data['windage'][0];
                data['windage_max'] = data['windage'][1];
            }

            if (_.has(data, 'start_position')) {
                var pos = data['start_position'];
                data['start_position_x'] = pos[0];
                data['start_position_y'] = pos[1];
                data['start_position_z'] = pos[2];
            }

            PointReleaseSpillFormView.__super__.setForm.apply(this, arguments);
        },

        show: function(coords) {
            var form = this.getForm();

            if (this.model && this.model.id) {
                this.setForm(form, this.model.toJSON());
            } else {
                this.setForm(form, this.defaults);
            }

            if (coords) {
                var coordInputs = this.$el.find('.coordinate');
                $(coordInputs[0]).val(coords[0]);
                $(coordInputs[1]).val(coords[1]);
            }

            PointReleaseSpillFormView.__super__.show.apply(this, arguments);
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
            data['is_active'] = data['active'];
            data['windage'] = [
                data['windage_min'], data['windage_max']
            ];
            data['start_position'] = [
                data['start_position_x'], data['start_position_y'],
                data['start_position_z']
            ];

            this.model.set(data);
            this.collection.add(this.model);
            this.model.save();
        }
    });


    var AddPointReleaseSpillFormView = PointReleaseSpillFormView.extend({
        initialize: function(options) {
            var opts = _.extend({
                dialog: {
                    width: 400,
                    height: 420,
                    title: "Add Point Release Spill"
                }
            }, options);

            AddPointReleaseSpillFormView.__super__.initialize.apply(this, [opts]);
        },

        show: function() {
            this.model = new models.PointReleaseSpill();
            AddPointReleaseSpillFormView.__super__.show.apply(this, arguments);
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
            this.model.set(data);
            this.model.save();
        },

        show: function(coords) {
            var form = this.getForm();

            if (this.model && this.model.id) {
                this.setForm(form, this.model.toJSON());
            } else {
                this.setForm(form, this.defaults);
            }

            ModelSettingsFormView.__super__.show.apply(this, arguments);
        }
    });


    return {
        AddMapFormView: AddMapFormView,
        AddMoverFormView: AddMoverFormView,
        AddSpillFormView: AddSpillFormView,
        AddWindMoverFormView: AddWindMoverFormView,
        AddPointReleaseSpillFormView: AddPointReleaseSpillFormView,
        MapFormView: MapFormView,
        WindMoverFormView: WindMoverFormView,
        PointReleaseSpillFormView: PointReleaseSpillFormView,
        FormView: FormView,
        FormViewContainer: FormViewContainer,
        ModelSettingsFormView: ModelSettingsFormView
    };

});