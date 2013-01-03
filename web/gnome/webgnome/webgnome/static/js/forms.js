
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
            this.options.pointReleaseSpills.on("create", this.refresh);
            this.options.windMovers.on("create", this.refresh);

            // TODO: Remove this when we remove the Long Island code.
            this.options.model.on(models.ModelRun.RUN_BEGAN, this.refresh);
        },

        /*
         Refresh all forms from the server.

         Called when any new spills or movers are added, so their forms are
         added to the page.
         */
        refresh: function() {
            var _this = this;

            $.ajax({
                type: 'GET',
                url: this.options.url,
                tryCount: 0,
                retryLimit: 3,
                success: function(data) {
                    if (_.has(data, 'html')) {
                        _this.$el.html(data.html);
                        _this.trigger(FormViewContainer.REFRESHED);
                    }
                },
                error: util.handleAjaxError
            });
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

        deleteAll: function() {
            var _this = this;
             _.each(this.formViews, function(formView, key) {
                formView.remove();
                delete _this.formViews[key];
            });
        },

        hideAll: function() {
            _.each(this.formViews, function(formView, key) {
                formView.hide();
            });
        }
    }, {
        REFRESHED: 'modalFormViewContainer:refreshed'
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
            'click .btn-primary': 'submit',
            'click .form-buttons .cancel': 'cancel',
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
//            this.remove();
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

        remove: function() {
            this.hide();
            this.$el.empty();
            this.$el.remove();
            this.container.off('click', this.id + ' .btn-primary', this.submit);
            this.container.off('click', this.id + ' .btn-next', this.goToNextStep);
            this.container.off('click', this.id + ' .btn-prev', this.goToPreviousStep);
        }
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
            JQueryUIModalFormView.__super__.initialize.apply(this, options);

            var opts = $.extend({}, {
                autoOpen: false,
                // Disable the default behavior of auto-focus on first element.
                open: function(event, ui) {
                    $("input").blur();
                },
                buttons: {
                    Cancel: function() {
                        $(this).dialog("close");
                    },

                    "Save": function() {
                        $(this).dialog("close");
                    }
                }
            }, options);

            this.$el.dialog(opts);
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
                width: 900,
                height: 710,
                title: "Edit Wind Mover"
            }, options);

            WindMoverFormView.__super__.initialize.apply(this, [opts]);
            this.renderTimeTable();
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
            'change .units': 'renderTimeTable'
        },

        compassChanged: function(magnitude, direction) {
            this.compassMoved(magnitude, direction);
        },

        compassMoved: function(magnitude, direction) {
            var form = this.$el.find('.time-form').not('.hidden');
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
                $(this).val('').removeAttr('checked');
            });
        },

        renderTimeTable: function() {
            var _this = this;
            var forms = this.$el.find('.edit-time-forms').find('.time-form');
            var units = this.$el.find('.units').find('option:selected').val();
            var rows = [];

            // Clear out any existing rows.
            this.getTimesTable().find('tr').not('.table-header').remove();

            _.each(forms, function(form) {
                form = $(form);
                var tmpl = _.template($("#time-series-row").html());
                var direction = form.find('.direction').val();

                if (isNaN(direction)) {
                    direction = direction.toUpperCase();
                } else {
                    direction = direction + ' &deg;';
                }

                var error = form.find('.error').length > 0;

                var dateTime = moment(
                    form.find('.date').val() + ' ' +
                    form.find('.hour').val() + ':' +
                    form.find('.minute').val());

                rows.push($(tmpl({
                    error: error ? 'error' : '',
                    date: dateTime.format('MM/DD/YYYY'),
                    time: dateTime.format('HH:mm'),
                    direction: direction,
                    speed: form.find('.speed').val() + ' ' + units
                })).data('data-form', form));
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

        /*
         Remove the add form inputs from the tab the user is currently on.
         Remove all form inputs from the tab the user is *not* currently on.
         */
        submit: function(event) {
            var constantWind = this.$el.find('.tab-pane.constant-wind');
            var variableWind = this.$el.find('.tab-pane.variable-wind');

            if (variableWind.hasClass('active')) {
                variableWind.find('.add-time-forms .time-form').empty().remove();
                constantWind.find('.time-form').empty().remove();
            } else {
                variableWind.find('.time-form').empty().remove();
            }

            WindMoverFormView.__super__.submit.apply(this, arguments);
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
        
        formDatesMatch: function(form1, form2) {
            return (form1.find('.date').val() == form2.find('.date').val()
                && form1.find('.hour').val() == form2.find('.hour').val()
                && form1.find('.minute').val() == form2.find('.minute').val())
        },

        trashButtonClicked: function(event) {
            event.preventDefault();
            var form = $(event.target).closest('tr').data('data-form');
            var editForm = this.$el.find('.add-time-forms').find('.edit-time-form');

            // There is an edit form visible
            if (editForm.length && this.formDatesMatch(editForm, form)) {
                // The edit form is for this wind value, so delete it.
                editForm.detach().empty().remove();
                this.$el.find('.add-time-forms').find(
                    '.add-time-form').removeClass('hidden');
            }
            form.detach().empty().remove();
            this.renderTimeTable();
        },

        saveButtonClicked: function(event) {
            event.preventDefault();
            var form = $(event.target).closest('.time-form');
            form.addClass('hidden');

            // Delete the "original" form that we're replacing.
            form.data('form-original').detach().empty().remove();
            form.detach().appendTo('.edit-time-forms');

            // Show the add form
            this.$el.find('.add-time-forms').find(
                '.add-time-form').removeClass('hidden');

            this.renderTimeTable();
            this.compass.compassUI('reset');
        },

        cancelButtonClicked: function(event) {
            event.preventDefault();
            var form = $(event.target).closest('.time-form');

            form.addClass('hidden');
            this.clearInputs(form);
            form.clone().appendTo('.times-list');
            form.empty().remove();

            $('.add-time-forms').find('.add-time-form').removeClass('hidden');

            var row = $(event.target).closest('tr');
            row.removeClass('info');
            this.renderTimeTable();
        },

        editButtonClicked: function(event) {
            event.preventDefault();
            var row = $(event.target).closest('tr');
            var form = row.data('data-form');
            var addFormContainer = form.closest('.tab-pane').find('.add-time-forms');
            var addTimeForm = addFormContainer.find('.add-time-form');

            addTimeForm.addClass('hidden');

            // Delete any edit forms currently in view.
            addFormContainer.find('.edit-time-form').remove();

            var formCopy = form.clone().prependTo(addFormContainer);
            formCopy.data('form-original', form);
            formCopy.removeClass('hidden');

            this.getTimesTable().find('tr').removeClass('info');
            row.removeClass('error').removeClass('warning').addClass('info');
        },

        /*
         Clone the add time form and add an item to the table of time series.
         */
        addButtonClicked: function(event) {
            event.preventDefault();
            var tabPane = $(event.target).closest('.tab-pane');
            var addForm = tabPane.find('.add-time-form');
            var newForm = addForm.clone(true).addClass('hidden').removeClass(
                'add-time-form').addClass('edit-time-form');

            // Grab the first timeseries-specific input field to check its
            // numeric position. This is the second input in the form.
            var formId = addForm.find(':input')[1].name;
            var formNum = parseInt(formId.replace(/.*-(\d{1,4})-.*/m, '$1')) + 1;

            // There are no edit forms, so this is the first time series.
            if (!formNum) {
                formNum = 0;
            }

            // Select all of the options selected on the original form.
            _.each(addForm.find('select option:selected'), function(opt) {
                opt = $(opt);
                var name = opt.closest('select').attr('name');
                var newOpt = newForm.find(
                    'select[name="' + name + '"] option[value="' + opt.val() + '"]');
                newOpt.attr('selected', true);
            });

            function incrementAttr(el, attrName) {
                el = $(el);
                var attr = el.attr(attrName);

                if (attr) {
                    attr = attr.replace('-' + (formNum - 1) + '-', '-' + formNum + '-');
                    var opts = {};
                    opts[attrName] = attr;
                    el.attr(opts);
                }
            }

            // Increment the IDs and names of the add form elements -- it
            // should always be the last (highest #) form of the edit forms.
            addForm.find(':input').each(function() {
                incrementAttr(this, 'name');
                incrementAttr(this, 'id');
            });

            newForm.find('.add-time-buttons').addClass('hidden');
            newForm.find('.edit-time-buttons').removeClass('hidden');

            tabPane.find('.edit-time-forms').append(newForm);
            this.renderTimeTable();
            this.compass.compassUI('reset');

            var autoIncrementBy = addForm.find('.auto_increment_by').val();

            // Increase the date and time on the Add form if 'auto increase by'
            // value was provided.
            if (autoIncrementBy) {
                var date = addForm.find('.date');
                var hour = addForm.find('.hour');
                var $minute = addForm.find('.minute');
                var time = hour.val()  + ':' + $minute.val();

                // TODO: Handle a date-parsing error here.
                var dateTime = moment(date.val() + ' ' + time);
                dateTime.add('hours', autoIncrementBy);

                date.val(dateTime.format("MM/DD/YYYY"));
                hour.val(dateTime.hours());
                $minute.val(dateTime.minutes());
            }
        }
    });


    var AddWindMoverFormView = WindMoverFormView.extend({
         initialize: function(options) {
             var opts = _.extend({
                 title: "Add Wind Mover"
             }, options);

             AddWindMoverFormView.__super__.initialize.apply(this, [opts]);
         },

        submit: function() {
            console.log('add wind mover submit');
        },

        show: function() {
            AddWindMoverFormView.__super__.show.apply(this);
            console.log('add wind mover show');
        },

        hide: function() {
            AddWindMoverFormView.__super__.hide.apply(this);
            console.log('add wind mover hide');
        }
    });


    var PointReleaseSpillFormView = JQueryUIModalFormView.extend({
         initialize: function(options) {
             var opts = _.extend({
                width: 400,
                height: 420,
                title: "Edit Point Release Spill"
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