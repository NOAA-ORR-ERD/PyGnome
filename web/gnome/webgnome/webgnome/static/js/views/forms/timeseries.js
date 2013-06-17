define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'util',
    'views/base',
    'lib/moment',
    'lib/compass-ui',
    'lib/gmaps'
], function($, _, Backbone, models, util, base) {
     var BaseTimeseriesView = base.BaseView.extend({
        initialize: function() {
            _.bindAll(this);
            var compassEl = $('#' + this.options.compassId);

            this.compass = compassEl.compassUI({
                'arrow-direction': 'in',
                'move': this.compassMoved,
                'change': this.compassChanged
            });

            BaseTimeseriesView.__super__.initialize.apply(this, arguments);
        },

        events: {
            'change .direction': 'updateCompass',
            'change .speed': 'updateCompass'
        },

        updateCompass: function() {
            var direction = this.$el.find('.direction').val();
            var speed = this.$el.find('.speed').val();

            if (!speed) {
                return;
            }

            this.compass.compassUI('update', {direction: direction, speed: speed});
            console.log('updateCompass', this, direction, speed);
        },

        getAddForm: function() {
            return this.$el.find('.add-time-form');
        },

        clearAddForm: function() {
            var form = this.getAddForm();
            form.find('input').val('');
            form.find('input:checkbox').prop('checked', false);
        },

        setModel: function(model) {
            this.model = model;
            this.setAddFormDefaults();
        },

        hide: function() {
            this.$el.addClass('hidden');
        },

        show: function() {
            this.$el.removeClass('hidden');
        },

        close: function() {
            this.stopListening(this.model);
            this.model = null;
        },

        compassChanged: function(magnitude, direction) {
            this.compassMoved(magnitude, direction);
        },

        compassMoved: function(magnitude, direction) {
            var form = this.getAddForm();
            form.find('.speed').val(magnitude.toFixed(2));
            form.find('.direction').val(direction.toFixed(2));
        },

        /*
         Set all fields for which a default value exists.
         */
        setAddFormDefaults: function() {
            var timeseries = this.model.get('timeseries');

            if (timeseries.length) {
                this.setDateFields('.datetime_container', moment(timeseries[0][0]));
                this.getElementByName('speed').val(timeseries[0][1]);
                this.getElementByName('direction').val(timeseries[0][2]);
            }

            this.clearErrors();
        },

        setWindValueForm: function(form, data) {
            var datetimeFields = form.find('.datetime_container');

            if (datetimeFields.length) {
                this.setDateFields(datetimeFields, moment(data[0]));
            }

            this.getElementByName('speed', form).val(data[1]);
            this.getElementByName('direction', form).val(data[2]);
        },
    });


    var ConstantWindTimeseriesView = BaseTimeseriesView.extend({
        initialize: function() {
            this.options.compassId = this.id + '_compass';
            ConstantWindTimeseriesView.__super__.initialize.apply(this, arguments);
        },

        /*
         Get wind timeseries values needed to save a constant wind mover.
         */
        getData: function() {
            var form = this.getAddForm();
            var speed = this.getElementByName('speed', form);
            var direction = this.getElementByName('direction', form);

            // A datetime is required, but it will be ignored for a constant
            // wind mover during the model run, so we just use the current
            // time.
            return [moment().format(), speed.val(), direction.val()];
        },

        render: function() {
            var timeseries = this.model.get('timeseries');
            var firstTimeValue = timeseries[0];
            if (firstTimeValue) {
                this.setWindValueForm(this.getAddForm(), firstTimeValue);
            }
        }
    });


    var VariableWindTimeseriesView = BaseTimeseriesView.extend({
        initialize: function() {
            this.options.compassId = this.id + '_compass';
            VariableWindTimeseriesView.__super__.initialize.apply(this, arguments);
            this.events = _.extend({}, BaseTimeseriesView.prototype.events, this.events);
            this.setupCompassDialog();
        },

        events: {
            'click .add-time': 'addButtonClicked',
            'click .edit-time': 'editButtonClicked',
            'click .show-compass': 'showCompass',
            'click .cancel': 'cancelButtonClicked',
            'click .save': 'saveButtonClicked',
            'click .delete-time': 'trashButtonClicked'
        },

        setupCompassDialog: function() {
            this.compassDialog = this.$el.find('.compass-container').dialog({
                width: 250,
                title: "Compass",
                zIndex: 6000,
                autoOpen: false,
                buttons: {
                    Close: function() {
                        $(this).dialog("close");
                    }
                }
            });
        },

        showCompass: function() {
            this.compass.compassUI('reset');
            this.compassDialog.removeClass('hidden');
            this.compassDialog.dialog('open');
        },

        hideCompass: function() {
            this.compassDialog.dialog('close');
            this.compassDialog.addClass('hidden');
        },

        close: function() {
            this.hideCompass();
            VariableWindTimeseriesView.__super__.close.apply(this, arguments);
        },

        getTimeseriesTable: function() {
            return this.$el.find('.time-list');
        },

        render: function() {
            var wind = this.model;
            var timeseries = wind.get('timeseries');
            var units = wind.get('units');
            var rows = [];

            // Clear out any existing rows.
            this.getTimeseriesTable().find('tr').not('.table-header').remove();

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
                var error = null;
                var row = $(tmpl({
                    error: error ? 'error' : '',
                    date: datetime.format('MM/DD/YYYY'),
                    time: datetime.format('HH:mm'),
                    direction: direction,
                    speed: speed + ' ' + units
                }));

                row.attr('data-wind-id', index);

                if (wind.timeseriesErrors && wind.timeseriesErrors[index]) {
                    row.addClass('error');
                    row.data('error', wind.timeseriesErrors[index]);
                }

                rows.push(row);
            });

            var table = this.getTimeseriesTable();

            _.each(rows, function(row) {
                row.appendTo(table);
            });
        },

        saveButtonClicked: function(event) {
            event.preventDefault();
            var wind = this.model;
            var timeseries = _.clone(wind.get('timeseries'));
            var addForm = this.getAddForm();
            var datetime = this.getDate(addForm);
            var windId = addForm.attr('data-wind-id');
            var direction = addForm.find('#direction').val();
            var duplicates = this.findDuplicates(timeseries, datetime, windId);
            var message = 'Wind data for that date and time exists. Replace it?';

            if (duplicates.length) {
                if (window.confirm(message)) {
                    timeseries.remove(duplicates[0]);
                    this.render();
                } else {
                    return;
                }
            }

            timeseries[windId] = [
                datetime.format(),
                this.getElementByName('speed', addForm).val(),
                this.getCardinalAngle(direction)
            ];

            wind.set('timeseries', timeseries);

            this.setAddFormDefaults();
            addForm.find('.add-time-buttons').removeClass('hidden');
            addForm.find('.edit-time-buttons').addClass('hidden');
            this.render();
            this.compass.compassUI('reset');
        },

        trashButtonClicked: function(event) {
            event.preventDefault();
            var windId = $(event.target).closest('tr').attr('data-wind-id');
            var wind = this.model;
            var timeseries = wind.get('timeseries');
            var windValue = timeseries[windId];
            var addForm = this.getAddForm();

            if (addForm.attr('data-wind-id') === windId) {
                this.setAddFormDefaults();
                addForm.find('.add-time-buttons').removeClass('hidden');
                addForm.find('.edit-time-buttons').addClass('hidden');
            }

            if (windValue) {
                // Remove the wind value from the timeseries array.
                timeseries.splice(windId, 1);
            }

            this.render();
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

        cancelButtonClicked: function(event) {
            event.preventDefault();
            var addForm = this.getAddForm();
            this.setAddFormDefaults();
            addForm.find('.add-time-buttons').removeClass('hidden');
            addForm.find('.edit-time-buttons').addClass('hidden');
            var row = $(event.target).closest('tr.info');
            row.removeClass('info');
            this.render();
        },

        getRowForWindId: function(windId) {
            return this.$el.find('tr[data-wind-id="' + windId + '"]')
        },

        showEditFormForWind: function(windId) {
            var row = this.getRowForWindId(windId);
            var wind = this.model;
            var timeseries = wind.get('timeseries');
            var windValue = timeseries[windId];
            var addForm = this.getAddForm();

            addForm.attr('data-wind-id', windId);
            addForm.find('.add-time-buttons').addClass('hidden');
            addForm.find('.edit-time-buttons').removeClass('hidden');
            this.setWindValueForm(addForm, windValue);
            this.getTimeseriesTable().find('tr').removeClass('info');
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
            var wind = this.model;
            var timeseries = _.clone(wind.get('timeseries'));
            var addForm = this.getAddForm();
            var datetime = this.getDate(addForm);
            var direction = addForm.find('#direction').val();

            // TODO: Handle this with a field validation plugin.
            if (datetime === undefined) {
                window.alert('Please enter a date and time.');
                return;
            }

            var duplicates = this.findDuplicates(timeseries, datetime);
            var windValue = [
                datetime.format(),
                this.getElementByName('speed', addForm).val(),
                this.getCardinalAngle(direction)
            ];
            var warning = 'Wind data for that date and time exists. Replace it?';

            if (duplicates.length) {
                if (window.confirm(warning)) {
                    timeseries[duplicates[0]] = windValue;
                    this.render();
                } else {
                    return;
                }
            } else {
                timeseries.push(windValue);
            }

            wind.set('timeseries', timeseries);
            this.render();
            this.compass.compassUI('reset');

            var autoIncrementBy = addForm.find('.auto_increment_by').val();

            // Increase the date and time on the Add form if 'auto increase by'
            // value was provided.
            if (autoIncrementBy) {
                var nextDatetime = datetime.clone().add('hours', autoIncrementBy);
                this.setDateFields(addForm, nextDatetime);
            }
        }
    });

    return {
        BaseTimeseriesView: BaseTimeseriesView,
        ConstantWindTimeseriesView: ConstantWindTimeseriesView,
        VariableWindTimeseriesView: VariableWindTimeseriesView
    }
});
