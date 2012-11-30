
define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'util',
    'lib/moment',
    'lib/compass-ui'
], function($, _, Backbone, models, util) {

    var ModalFormViewContainer = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.options.ajaxForms.on(models.AjaxForm.SUCCESS, this.refresh);
            this.formViews = {};
        },

        /*
         Refresh all forms from the server.

         Called when any `AjaxForm` on the page has a successful submit, in case
         additional forms should appear for new items.
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
                        _this.trigger(ModalFormViewContainer.REFRESHED);
                    }
                },
                error: util.handleAjaxError
            });
        },

        formIdChanged: function(newId, oldId) {
            this.formViews[newId] = this.formViews[oldId];
            delete this.formViews[oldId];
        },

        add: function(opts, obj) {
            if (typeof opts === "number" || typeof opts === "string") {
                this.formViews[opts] = obj;
                return;
            }

            if (typeof opts === "object" &&
                    (_.has(opts, 'id') && opts.id)) {
                var view = new ModalFormView(opts);
                this.formViews[opts.id] = view;
                view.on(ModalFormView.ID_CHANGED, this.formIdChanged);
                return;
            }

            throw "Must pass ID and object or an options object.";
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
        }
    }, {
        REFRESHED: 'modalFormViewContainer:refreshed'
    });


    /*
     `ModalFormView` is responsible for displaying HTML forms retrieved
     from and submitted to the server using an `AjaxForm object. `ModalFormView`
     displays an HTML form in a modal "window" over the page using the rendered HTML
     returned by the server. It listens to 'change' events on a bound `AjaxForm` and
     refreshes itself when that event fires.

     The view is designed to handle multi-step forms implemented purely in
     JavaScript (and HTML) using data- properties on DOM elements. The server
     returns one rendered form, but may split its content into several <div>s, each
     with a `data-step` property. If a form is structured this way, the user of the
     JavaScript application will see it as a multi-step form with "next," "back"
     and (at the end) a "save" or "create" button (the label is given by the server,
     but whatever it is, this is the button that signals final submission of the
     form).

     Submitting a form from `ModalFormView` serializes the form HTML and sends it to
     a bound `AjaxForm` model object, which then handles settings up the AJAX
     request for a POST.
     */
    var ModalFormView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.$container = $(this.options.formContainerEl);
            this.ajaxForm = this.options.ajaxForm;
            this.ajaxForm.on(models.AjaxForm.CHANGED, this.ajaxFormChanged);
            this.setupEventHandlers();
        },

        /*
         Bind listeners to the form container using `on()`, so they persist if
         the underlying form elements are replaced.
         */
        setupEventHandlers: function() {
            this.id = '#' + this.$el.attr('id');
            this.$container.on('click', this.id + ' .btn-primary', this.submit);
            this.$container.on('click', this.id + ' .btn-next', this.goToNextStep);
            this.$container.on('click', this.id + ' .btn-prev', this.goToPreviousStep);
        },

        ajaxFormChanged: function(ajaxForm) {
            var formHtml = ajaxForm.form_html;
            if (formHtml) {
                this.refresh(formHtml);
                this.show();
            }
        },

        /*
         Hide any other visible modals and show this one.
         */
        show: function() {
            $('div.modal').modal('hide');
            this.$el.modal();
        },

        /*
         Reload this form's HTML by initiating an AJAX request via this view's
         bound `AjaxForm`. If the request is successful, this `ModelFormView` will
         fire its `ajaxFormChanged` event handler.
         */
        reload: function(id) {
            this.ajaxForm.get({id: id});
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
            var $form = this.getForm();

            if (!$form.hasClass('multistep')) {
                return;
            }

            var stepDiv = $form.find('div.step[data-step="' + stepNum + '"]');

            if (stepDiv.length === 0) {
                return;
            }

            var otherStepDivs = $form.find('div.step');
            otherStepDivs.addClass('hidden');
            otherStepDivs.removeClass('active');
            stepDiv.removeClass('hidden');
            stepDiv.addClass('active');

            var prevButton = this.$container.find('.btn-prev');
            var nextButton = this.$container.find('.btn-next');
            var saveButton = this.$container.find('.btn-primary');

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
            var $form = this.getForm();

            if (!$form.hasClass('multistep')) {
                return;
            }

            var activeStepDiv = $form.find('div.step.active');
            var currentStep = parseInt(activeStepDiv.attr('data-step'), 10);
            this.goToStep(currentStep + 1);
        },

        goToPreviousStep: function(event) {
            var $form = this.getForm();

            if (!$form.hasClass('multistep')) {
                return;
            }

            var activeStep = $form.find('div.step.active');
            var currentStep = parseInt(activeStep.attr('data-step'), 10);
            this.goToStep(currentStep - 1);
        },

        submit: function(event) {
            event.preventDefault();
            var $form = this.getForm();
            this.ajaxForm.submit({
                data: $form.serialize(),
                url: $form.attr('action')
            });
            this.hide();
            return false;
        },

        /*
         Replace this form with the form in `html`, an HTML string rendered by the
         server. Recreate any jQuery UI datepickers on the form if necessary.
         If there is an error in the form, load the step with errors.
         */
        refresh: function(html) {
            var oldId = this.$el.attr('id');

            this.remove();

            var $html = $(html);
            $html.appendTo(this.$container);

            this.$el = $('#' + $html.attr('id'));

             // Setup datepickers
            _.each(this.$el.find('.date'), function(field) {
                $(field).datepicker({
                    changeMonth: true,
                    changeYear: true
                });
            });

            var stepWithError = this.getFirstStepWithError();
            if (stepWithError) {
                this.goToStep(stepWithError);
            }

            var tabWithError = this.getFirstTabWithError();
            if (tabWithError) {
                $('a[href="#' + tabWithError + '"]').tab('show');
            }

            this.setupEventHandlers();
            util.fixModals();

            var newId = this.$el.attr('id');
            if (oldId !== newId) {
                this.trigger(ModalFormView.ID_CHANGED, newId, oldId);
            }
        },

        hide: function() {
            this.$el.modal('hide');
        },

        remove: function() {
            this.hide();
            this.$el.empty();
            this.$el.remove();
            this.$container.off('click', this.id + ' .btn-primary', this.submit);
            this.$container.off('click', this.id + ' .btn-next', this.goToNextStep);
            this.$container.off('click', this.id + ' .btn-prev', this.goToPreviousStep);
        }
    }, {
        ID_CHANGED: 'modalFormView:idChanged'
    });


    /*
     This is a non-AJAX-enabled modal form object to support the "add mover" form,
     which asks the user to choose a type of mover to add. We then use the selection
     to disply another, more-specific form.
     */
    var AddMoverFormView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.$container = $(this.options.formContainerEl);

            // Bind listeners to the container, using `on()`, so they persist.
            this.id = '#' + this.$el.attr('id');
            this.$container.on('click', this.id + ' .btn-primary', this.submit);
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
            var $form = this.getForm();
            var moverType = $form.find('select[name="mover_type"]').val();

            if (moverType) {
                this.trigger(AddMoverFormView.MOVER_CHOSEN, moverType);
            }

            return false;
        }
    }, {
        // Events
        MOVER_CHOSEN: 'addMoverFormView:moverChosen'
    });


    /*
     `WindMoverFormView` handles the WindMover form.
     */
    var WindMoverFormView = ModalFormView.extend({
        initialize: function(options) {
            this.constructor.__super__.initialize.apply(this, arguments);
            this.renderTimeTable();
            this.setupCompass();

            this.$container.on('click', this.id + ' .show-compass', this.showCompassClicked);
            this.$container.on('click', this.id + ' .edit-mover-name i', this.editMoverNameButtonClicked);
            this.$container.on('click', this.id + ' .save-mover-name i', this.saveMoverNameButtonClicked);
            this.$container.on('change', this.id + ' .edit-mover-name-field', this.moverNameChanged);
            this.$container.on('click', this.id + ' .add-time', this.addButtonClicked);
            this.$container.on('click', this.id + ' .edit-time', this.editButtonClicked);
            this.$container.on('click', this.id + ' .cancel', this.cancelButtonClicked);
            this.$container.on('click', this.id + ' .save', this.saveButtonClicked);
            this.$container.on('click', this.id + ' .delete-time', this.trashButtonClicked);
        },

        toggleCompass: function(action) {
            var $compass = this.$el.find('.compass-container,.compass');
            var $editForms = this.$el.find('.edit-time-forms');

            if (action === undefined) {
                 action = $compass.hasClass('hidden') ? 'show' : 'hide';
            }

            // XXX: Adding and removing the 'span6' class, which is on the
            // .edit-time-forms div by default, is a hack to remove unwanted
            // visual space caused by adding the 'hidden' class to a div that
            // has a 'span6' class. Temporarily removing the 'span6' class
            // while the div is hidden removes the space.
            if (action === 'show') {
                $editForms.addClass('hidden');
                $editForms.removeClass('span6');
                $compass.removeClass('hidden');
                this.compass.compassUI('reset');
            } else {
                $editForms.removeClass('hidden');
                $editForms.addClass('span6');
                $compass.addClass('hidden');
            }
        },

        showCompassClicked: function(event) {
            event.preventDefault();
            this.toggleCompass();
        },

        compassChanged: function(magnitude, direction) {
            this.compassMoved(magnitude, direction);
            this.toggleCompass();
        },

        compassMoved: function(magnitude, direction) {
            var $form = this.$el.find('.time-form').not('.hidden');
            $form.find('.speed').val(magnitude.toFixed(2));
            $form.find('.direction').val(direction.toFixed(2));
        },

        setupCompass: function() {
            var _this = this;
            var $compass = this.$el.find('.compass');

            this.compass = $compass.compassUI({
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
            var $forms = this.$el.find('.edit-time-forms').find('.time-form');
            var rows = [];

            // Clear out any existing rows.
            this.getTimesTable().find('tr').not('.table-header').remove();

            _.each($forms, function(form) {
                var $form = $(form);
                var tmpl = _.template($("#time-series-row").html());
                var speedType = $form.find('.speed_type option:selected').val();
                var direction = $form.find('.direction').val();

                if (isNaN(direction)) {
                    direction = direction.toUpperCase();
                } else {
                    direction = direction + ' &deg;';
                }

                var error = $form.find('.error').length > 0;

                var dateTime = moment(
                    $form.find('.date').val() + ' ' +
                    $form.find('.hour').val() + ':' +
                    $form.find('.minute').val());

                rows.push($(tmpl({
                    error: error ? 'error' : '',
                    date: dateTime.format('MM/DD/YYYY'),
                    time: dateTime.format('HH:mm'),
                    direction: direction,
                    speed: $form.find('.speed').val() + ' ' + speedType
                })).data('data-form', $form));
            });

            // Sort table by date and time of each item.
            rows = _.sortBy(rows, function($tr) {
                var date = $tr.find('.time-series-date').text();
                var time = $tr.find(
                    '.time-series-time').text().replace(' ', '', 'g');
                return Date.parse(date + ' ' + time)
            });

            _.each(rows, function($row) {
                $row.appendTo(_this.getTimesTable());
            });
        },

        /*
         Remove the "Add" form inputs and submit the form.
         */
        submit: function() {
            this.$el.find('.add-time-forms .time-form').empty().remove();
            WindMoverFormView.__super__.submit.apply(this, arguments);
        },

        editMoverNameButtonClicked: function(event) {
            var $button = $(event.target);
            var $modal = $button.closest('.modal');
            var $editField = $modal.find('.edit-mover-name-field');
            $editField.removeClass('hidden');
            var $header = $modal.find('.modal-label');
            $header.addClass('hidden');
            $button.parent().addClass('hidden');
            $modal.find('.save-mover-name').removeClass('hidden');
        },

        saveMoverNameButtonClicked: function(event) {
            var $button = $(event.target);
            var $modal = $button.closest('.modal');
            var $editField = $modal.find('.edit-mover-name-field');
            $editField.addClass('hidden');
            var $header = $modal.find('.modal-label');
            $header.removeClass('hidden');
            $button.parent().addClass('hidden');
            $modal.find('.edit-mover-name').removeClass('hidden');
        },

        moverNameChanged: function(event) {
            var $input = $(event.target);
            var $modal = $input.closest('.modal');
            var $header = $modal.find('.modal-label').find('h3');
            $header.text($input.val());
        },

        ajaxFormChanged: function() {
            WindMoverFormView.__super__.ajaxFormChanged.apply(this, arguments);
            this.renderTimeTable();

            var hasErrors = this.$el.find(
                '.time-list').find('tr.error').length > 0;

            if (hasErrors) {
                alert('At least one of your time series values had errors in ' +
                     'it. The rows with errors will be marked in red.');
            }

            this.setupCompass();
        },

        trashButtonClicked: function(event) {
            event.preventDefault();
            var $form = $(event.target).closest('tr').data('data-form');
            $form.detach().empty().remove();
            this.renderTimeTable();
        },

        saveButtonClicked: function(event) {
            event.preventDefault();
            var $form = $(event.target).closest('.time-form');
            $form.addClass('hidden');

            // Delete the "original" form that we're replacing.
            $form.data('form-original').detach().empty().remove();
            $form.detach().appendTo('.times-list');

            // Show the add form
            $('.add-time-forms').find('.add-time-form').removeClass('hidden');

            this.getTimesTable().append($form);
            this.renderTimeTable();
            this.toggleCompass('hide');
        },

        cancelButtonClicked: function(event) {
            event.preventDefault();
            var $form = $(event.target).closest('.time-form');

            $form.addClass('hidden');
            this.clearInputs($form);
            $form.clone().appendTo('.times-list');
            $form.empty().remove();

            $('.add-time-forms').find('.add-time-form').removeClass('hidden');

            var $row = $(event.target).closest('tr');
            $row.removeClass('info');
            this.renderTimeTable();
            this.toggleCompass('hide');
        },

        editButtonClicked: function(event) {
            event.preventDefault();
            var $row = $(event.target).closest('tr');
            var $form = $row.data('data-form');
            var $addFormContainer = $('.add-time-forms');
            var $addTimeForm = $addFormContainer.find('.add-time-form');

            $addTimeForm.addClass('hidden');

            // Delete any edit forms currently in view.
            $addFormContainer.find('.edit-time-form').remove();

            var $formCopy = $form.clone().appendTo($addFormContainer);
            $formCopy.data('form-original', $form);
            $formCopy.removeClass('hidden');

            this.getTimesTable().find('tr').removeClass('info');
            $row.removeClass('error').removeClass('warning').addClass('info');
        },

        /*
         Clone the add time form and add an item to the table of time series.
         */
        addButtonClicked: function(event) {
            event.preventDefault();
            var $addForm = this.$el.find('.add-time-form');
            var $newForm = $addForm.clone(true).addClass('hidden').removeClass(
                'add-time-form').addClass('edit-time-form');
            var formId = $addForm.find(':input')[0].id;
            var formNum = parseInt(formId.replace(/.*-(\d{1,4})-.*/m, '$1')) + 1;

            // There are no edit forms, so this is the first time series.
            if (!formNum) {
                formNum = 0;
            }

            // Select all of the options selected on the original form.
            _.each($addForm.find('select option:selected'), function(opt) {
                var $opt = $(opt);
                var name = $opt.closest('select').attr('name');
                var $newOpt = $newForm.find(
                    'select[name="' + name + '"] option[value="' + $opt.val() + '"]');
                $newOpt.attr('selected', true);
            });

            function incrementAttr(el, attrName) {
                var $el = $(el);
                var attr = $el.attr(attrName);

                if (attr) {
                    attr = attr.replace('-' + (formNum - 1) + '-', '-' + formNum + '-');
                    var opts = {};
                    opts[attrName] = attr;
                    $el.attr(opts);
                }
            }

            // Increment the IDs and names of the add form elements -- it
            // should always be the last (highest #) form of the edit forms.
            $addForm.find(':input').each(function() {
                incrementAttr(this, 'name');
                incrementAttr(this, 'id');
            });

            $newForm.find('.add-time-buttons').addClass('hidden');
            $newForm.find('.edit-time-buttons').removeClass('hidden');

            this.getTimesTable().after($newForm);
            this.renderTimeTable();

            var autoIncrementBy = $addForm.find('.auto_increment_by').val();

            // Increase the date and time on the Add form if 'auto increase by'
            // value was provided.
            if (autoIncrementBy) {
                var $date = $addForm.find('.date');
                var $hour = $addForm.find('.hour');
                var $minute = $addForm.find('.minute');
                var time = $hour.val()  + ':' + $minute.val();

                // TODO: Handle a date-parsing error here.
                var dateTime = moment($date.val() + ' ' + time);
                dateTime.add('hours', autoIncrementBy);

                $date.val(dateTime.format("MM/DD/YYYY"));
                $hour.val(dateTime.hours());
                $minute.val(dateTime.minutes());
            }
        }
    });

    return {
        AddMoverFormView: AddMoverFormView,
        WindMoverFormView: WindMoverFormView,
        ModalFormView: ModalFormView,
        ModalFormViewContainer: ModalFormViewContainer
    };

});