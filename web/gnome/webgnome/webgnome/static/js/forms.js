
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
    'lib/bootstrap.file-input',
    'lib/jquery.imagesloaded.min'
], function($, _, Backbone, models, util, geo, rivets) {


    var DeferredManager = function() {
        this.deferreds = [];
        this.namedDeferreds = {};
    };

    DeferredManager.prototype = {
        /*
         Add a deferred method call.

         Calling this method multiple times with the same `fn` value will add
         the method call multiple times.
         */
        add: function(fn) {
            this.deferreds.push(fn);
        },

        /*
         Add a deferred method call by name.

         Multiple calls to this method using the same value for `name` will
         overwrite the value, resulting in only one deferred method call for
         each `name` value.
         */
        addNamed: function(name, fn) {
            this.namedDeferreds[name] = fn;
        },

        /*
         Loop through the closures saved in `this.deferreds` and
         `this.namedDeferreds` and call them. Keep track of any result that is
         a jQuery Deferred object in an `actualDeferreds` array.

         Calling this method returns a jQuery Deferred object that is only
         resolved when all Deferred objects returned by closures are resolved.

         We attach `done` and `fail` handlers to any deferreds in `actualDeferreds`
         so that when all of these Deferred objects are resolved, we resolve
         the call to `run`. If *any* of Deferred objects fail, we fail the call
         to `run`.
         */
        run: function() {
            var dfd = $.Deferred();
            var potentialDeferreds = this.deferreds.concat(
                _.values(this.namedDeferreds));
            var actualDeferreds = [];

            _.each(potentialDeferreds, function(fn) {
                var result = fn();

                if (result && typeof result.done === 'function') {
                    actualDeferreds.push(result);
                }
            });

            $.when.apply(null, actualDeferreds).done(function() {
                dfd.resolve();
            }).fail(function() {
                // XXX: If any deferred method fails, the run operation fails.
                dfd.fail();
            });

            return dfd;
        }
    };


    var deferreds = new DeferredManager();


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

        /*
         Show the form with `formId`.
         */
        show: function(formId, success, cancel, customButtons, defaults) {
            var formView = this.get(formId);

            if (formView) {
                if (success) {
                    formView.once(FormView.SUBMITTED, success);
                }
                if (cancel) {
                    formView.once(FormView.CANCELED, cancel);
                }
                if (customButtons && customButtons.length) {
                    formView.setCustomButtons(customButtons);
                }
                if (defaults) {
                    formView.defaults = defaults;
                }

                formView.reload();
                formView.show(true);
            }
        }
    });


    var BaseView = Backbone.View.extend({
        initialize: function() {
            this.model = this.options.model;
            this.id = this.options.id;

            if (this.options.id) {
                this.$el = $('#' + this.options.id);
            }

            BaseView.__super__.initialize.apply(this, arguments);
        },

        /*
         Get an element by searching for its name within `container`.

         If `container` is undefined, default to the FormView's element.
         */
        getElementByName: function(name, container, type) {
            container = container && container.length ? container : this.$el;
            type = type ? type : '*';
            return container.find(type + '[name="' + name + '"]');
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

        hasDateFields: function(target) {
            var fields = this.getDateFields(target);
            return fields.date && fields.hour && fields.minute;
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

        /*
         TODO: Rename this method. It's not specific to forms.
         */
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

        sendMessage: function(message) {
            this.trigger(BaseView.MESSAGE_READY, message);
        },
    }, {
        MESSAGE_READY: 'baseView:messageReady',
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
    var FormView = BaseView.extend({
        initialize: function() {
            var _this = this;
            _.bindAll(this);

            FormView.__super__.initialize.apply(this, arguments);

            this.wasCancelled = false;
            this.gnomeModel = this.options.gnomeModel;
            this.collection = this.options.collection;
            this.defaults = this.options.defaults;

            // A `FormView` may have an optional validator web service that
            // validates data without saving it. Here we allow the caller to
            //
            // Allow the creator to override a prototype default.
            if (this.options.validator) {
                this.validator = this.options.validator;
            }

            // Validators require a `gnomeModel` to reference the current
            // Gnome model ID.
            if (this.validator) {
                this.validator = new this.validator({}, {gnomeModel: this.gnomeModel});
            }

            if (this.model) {
                this.listenTo(this.model, 'destroy', function() {
                    _this.model.clear();
                });
            }

            this.setupDatePickers();
            $('.error').tooltip({selector: "a"});
        },

        /*
         Validate the form's model without submitting -- only check that the
         data conforms to a schema.

         This is used when deferring a submit, when we want to verify that the
         data on the form is correct without actually saving it.
         */
        validate: function() {
            if (!this.validator || !this.model) {
                return;
            }

            this.prepareSubmitData();
            return this.validator.save(this.model.toJSON());
        },

        /*
         When we validate a form separately from submitting the model, we
         send the model's current attributes to a /validate web service that
         only checks that they conform to a schema.

         If the object has errors, we'll show them as if they were errors on
         the model itself -- which they are.
         */
        handleValidatorError: function() {
            if (!this.model) {
                console.log("Error: FormView's model is null. Cannot handle " +
                            "validator errors.", this, this.validator.errors);
                return;
            }

            this.model.errors = this.validator.errors;
            this.handleServerError();
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

            var tabs = this.$el.find('.tab-pane');

            if (tabs.length) {
                for (var i = 0; i < tabs.length; i += 1) {
                    var tabErrors = $(tabs[i]).find('.error');

                    if (tabErrors.length) {
                        // TODO: Open tab.
                        // http://stackoverflow.com/questions/11742130/how-do-i-hide-and-show-twitter-bootstrap-tabs
                        console.log('Tab has an error: ' + tabs[i].id);
                        break;
                    }
                }
            }

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

        prepareSubmitData: function() {
            // Override in subclasses if you want to separate preparing the
            // model for submission from submitting -- e.g. for extra steps
            // needed before validation (see WindMoverFormView).
            return;
        },

        submit: function() {
            var _this = this;

            this.prepareSubmitData();

            if (this.collection) {
                this.collection.add(this.model);
            }

            // Return a jQuery Promise object.
            return this.model.save(null, {
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
                this.stopListening(this.model);
                this.model = null;
            }
            // TODO: Should we stop listening to the collection here too?
        },

        setForm: function(form, data) {
            var _this = this;
            _.each(data, function(dataVal, fieldName) {
                var input = _this.getElementByName(fieldName, form);

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

        clearForm: function() {
            var inputs = this.$el.find('select,input,textarea');

            if (!inputs.length) {
                return;
            }

            _.each(inputs, function(field) {
                $(field).val('');
            });
        },

        /*
         This can probably be done better -- e.g. with an object like
         'modelEvents' on the object.
         */
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
        SHOW_FORM: 'formView:showForm'
    });


     /*
     An `FormView` subclass that displays in a JQuery UI modal window.
     */
    var JQueryUIModalFormView = FormView.extend({
        initialize: function() {
            JQueryUIModalFormView.__super__.initialize.apply(this, arguments);
            this.setupDialog();
        },

        setupDialog: function() {
            var _this = this;
            var height = this.$el.attr('data-height');
            var width = this.$el.attr('data-width');

            // The default set of UI Dialog widget options. Pass in a 'dialog'
            // object in the view's options to add additional settings.
            var dialogOptions = $.extend({
                zIndex: 5000,
                autoOpen: false,
                height: height,
                width: width,
                title: this.$el.attr('title') || '',
                buttons: {
                    Save: function() {
                        _this.submit();
                    },

                    Cancel: function() {
                        _this.cancel();
                    }
                },
                close: this.close,
                beforeClose: this.beforeClose
            }, this.options.dialog || {});

            this.title = dialogOptions.title;
            this.$el.dialog(dialogOptions);

            // Workaround the default JQuery UI Dialog behavior of auto-
            // fAddocusing on the first input element by adding an invisible one.
            $('<span class="ui-helper-hidden-accessible">' +
                '<input type="text"/></span>').prependTo(this.$el);
        },

        beforeClose: function() {
            if (!this.wasCancelled && this.model && this.model.dirty
                    && !this.isDeferred) {
                if (!window.confirm('You have unsaved changes. Really close?')) {
                    return false;
                }
            }

            return true;
        },

        cancel: function() {
            this.wasCancelled = true;
            this.closeDialog();
            this.trigger(FormView.CANCELED);
            // Don't call FormView's cancel because we don't want to reset the
            // model here. That's already been done in close().
        },

        submit: function() {
            // Close if the form isn't using a model. If the form IS using a
            // model, it will close when the model saves without error.
            if (!this.model) {
                this.closeDialog();
            }
            return JQueryUIModalFormView.__super__.submit.apply(this, arguments);
        },

        setupModelEvents: function() {
            if (this.model) {
                this.listenTo(this.model, 'sync', this.closeDialog);
            }
            if (this.collection) {
                this.listenTo(this.collection, 'sync', this.closeDialog);
            }
            JQueryUIModalFormView.__super__.setupModelEvents.apply(this, arguments);
        },

        /*
         Hide any other visible modals and show this one.
         */
        show: function() {
            this.wasCancelled = false;

            // Load defaults if the form is being used to show a new model.
            // The lack of ID indicates that the model has not yet been saved.
            if (this.defaults && !this.model.id) {
                this.model.set(this.defaults);
            }

            this.clearErrors();
            this.prepareForm();
            this.openDialog();
            this.$el.removeClass('hide');
            this.bindData();
        },

        hide: function() {
            this.$el.dialog('close');
        },

        close: function() {
            this.closeDialog();
            if (this.originalButtons) {
                this.$el.dialog('option', 'buttons', this.originalButtons);
            }
            this.resetModel();
        },

        closeDialog: function() {
            this.$el.dialog('close');
        },

        openDialog: function() {
            this.$el.dialog('open');
        },

        addDeferredButton: function(button) {
            var _this = this;
            var name = this.$el.attr('id') + '-' + button.fnName;

            deferreds.addNamed(name, function() {
                // The result of the function may be a jQuery
                // Deferred object, e.g. a 'submit' function,
                // so return the result (a Deferred or Promise).
                var val = _this[button.fnName]();
                _this.isDeferred = false;
                return val;
            });

            this.close();

            // XXX: Do we really want to trigger these events
            // for deferred methods? Right now we have to because
            // FormViewContainer is listening to them so we can
            // continue in a Wizard-style form; however, is there
            // a better way of handling that? And if we do want
            // to trigger these events, should they really be
            // hard-coded like this?
            if (button.fnName === 'submit') {
                this.trigger(FormView.SUBMITTED);
            } else if (button.fnName === 'cancel') {
                this.trigger(FormView.CANCELED);
            }
        },

        setCustomButtons: function(buttonDefinitions) {
            var buttons = [];
            var _this = this;

            if (!buttonDefinitions || !buttonDefinitions.length) {
                return;
            }

            _.each(buttonDefinitions, function(button, index, list) {
                buttons.push({
                    text: button.text,
                    click: function() {
                        if (!button.deferred) {
                            _this[button.fnName]();
                            return;
                        }

                        _this.isDeferred = true;

                        _this.validate()
                            .done(function() {
                                _this.addDeferredButton(button);
                            })
                            .fail(_this.handleValidatorError);
                    }
                });
            });

            this.originalButtons = this.$el.dialog('option', 'buttons');
            this.$el.dialog('option', 'buttons', buttons);
        }
    });


    var MultiStepFormView = JQueryUIModalFormView.extend({
        events: {
            'click .ui-button.next': 'next',
            'click .ui-button.back': 'back'
        },

        initialize: function() {
            // Extend prototype's events with ours.
            this.events = _.extend({}, FormView.prototype.events, this.events);

            // Have to initialize super super before showing current step.
            MultiStepFormView.__super__.initialize.apply(this, arguments);

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
        initialize: function() {
            LocationFileWizardFormView.__super__.initialize.apply(this, arguments);

            _.bindAll(this);

            this.widget = this.$el.dialog('widget');
            this.widget.on('click', '.ui-button.next', this.next);
            this.widget.on('click', '.ui-button.back', this.back);
            this.widget.on('click', '.ui-button.cancel', this.cancel);
            this.widget.on('click', '.ui-button.finish', this.finish);
            this.widget.on('click', '.ui-button.references', this.showReferences);

            this.locationFileMeta = this.options.locationFileMeta;

            this.references = this.$el.find('div.references').dialog({
                autoOpen: false,
                buttons: {
                    Ok: function() {
                        $(this).dialog("close");
                    }
                }
            });

            this.setButtons();
        },

        setStepSize: function() {
            var step = this.getCurrentStep();
            var height = step.data('height');
            var width = step.data('width');

            if (height) {
                this.$el.dialog('option', 'height', parseInt(height));
            }
            if (width) {
                this.$el.dialog('option', 'width', parseInt(width));
            }
        },

        getDataBindings: function() {
            return {wizard: this.model};
        },

        resetModel: function() {
            // Don't reset the model for this form
        },

        /*
         Finish the wizard. In order:

            - Load the location file parameters into the user's current model
            - Run 'deferred' functions -- model-related form submits
            - Save the location-file-specific parameters the user selected
            - Reload the page
         */
        finish: function() {
            var _this = this;
             deferreds.run().done(function() {
                 _this.model.save().done(function() {
                     // TODO: Trigger event, let AppView handle this.
                     util.refresh();
                 }).fail(function() {
                     console.log('Error submitting the location file wizard.');
                     alert('Error setting up your model. Please try again.');
                 });
             }).fail(function() {
                 console.log('Error running deferred methods.');
                 alert('Error setting up your model. Please try again.');
             });

        },

        loadLocationFile: function() {
            var model = new models.GnomeModelFromLocationFile({
                location_name: this.locationFileMeta.get('filename')
            }, {
                gnomeModel: this.gnomeModel
            });

            return model.save();
        },

        showReferences: function() {
            this.references.dialog('open');
        },

        setButtons: function() {
            var step = this.getCurrentStep();
            var buttons = step.find('.custom-dialog-buttons');
            if (buttons.length) {
                var buttonPane = this.widget.find('.ui-dialog-buttonpane');
                buttonPane.empty();
                buttons.clone().appendTo(buttonPane).removeClass('hidden');
            }
        },

        getCustomStepButtons: function(step) {
            var buttons = [];
            var customFormButtons = step.find('.custom-dialog-buttons').find(
                '.ui-button');

            if (customFormButtons.length) {
                _.each(customFormButtons, function(el, index, list) {
                    var fnName = $(el).data('function-name');
                    if (!fnName) {
                        return;
                    }
                    buttons.push({
                        deferred: $(el).data('deferred'),
                        text: $(el).text(),
                        fnName: fnName
                    });
                })
            }

            return buttons;
        },

        showStep: function(step) {
            LocationFileWizardFormView.__super__.showStep.apply(this, [step]);
            this.setButtons();

            var form = step.data('show-form');
            if (form) {
                this.widget.addClass('hidden');
                this.showForm(form, this.getCustomStepButtons(step));
            } else {
                this.widget.removeClass('hidden');
                this.setStepSize();
            }
        },

        showForm: function(form, customButtons) {
            var _this = this;
            var nextStep = this.getNextStep();
            var previousStep = this.getPreviousStep();

            // Use closures to keep a reference to the actual next and previous
            // steps for this form.
            function showNextStep() {
                _this.showStep(nextStep)
            }

            function showPreviousStep() {
                _this.showStep(previousStep);
            }

            function trigger(defaults) {
                _this.trigger(FormView.SHOW_FORM, form, showNextStep,
                    showPreviousStep, customButtons, defaults);
            }

            if (form === 'model-settings') {
                var locationFile = new models.LocationFile({
                    filename: this.locationFileMeta.get('filename')
                }, {
                    gnomeModel: this.gnomeModel
                });

                locationFile.once('sync', function(model) {
                    trigger(model.attributes);
                });

                locationFile.fetch();
            } else {
                trigger();
            }
        }
    }, {
        LOCATION_CHOSEN: 'locationFileWizardFormView:locationChosen'
    });


    /*
     A base class for modal forms that ask the user to choose from a list of
     object types that are themselves represented by a `FormView` instance.
     */
    var ChooseObjectTypeFormView = JQueryUIModalFormView.extend({
        initialize: function() {
            var _this = this;

            this.options.dialog = _.extend({
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
            }, this.options.dialog);

            ChooseObjectTypeFormView.__super__.initialize.apply(this, arguments);
        }
    });


    /*
     This is a non-AJAX-enabled modal form object to support the "add mover" form,
     which asks the user to choose a type of mover to add. We then use the selection
     to display another, mover-specific form.
     */
    var AddMoverFormView = ChooseObjectTypeFormView.extend({
        submit: function() {
            var moverType = this.getElementByName('mover-type').val();

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


    var AddEnvironmentFormView = ChooseObjectTypeFormView.extend({
        submit: function() {
            var environmentType = this.getElementByName('environment-type').val();

            if (environmentType) {
                this.trigger(AddEnvironmentFormView.ENVIRONMENT_CHOSEN, environmentType);
                this.hide();
            }

            return false;
        }
    }, {
        // Events
        ENVIRONMENT_CHOSEN: 'addEnvironmentFormView:environmentChosen'
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
            var spillType = this.getElementByName('spill-type').val();

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
            var source = this.getElementByName('map-source').val();
            if (source) {
                this.trigger(AddMapFormView.SOURCE_CHOSEN, source);
            }
        }
    }, {
        // Event constants
        SOURCE_CHOSEN: 'addMapFormView:sourceChosen'
    });


    var MapFormView = JQueryUIModalFormView.extend({
        initialize: function() {
            this.options.dialog = _.extend({
                height: 200,
                width: 425
            }, this.options.dialog);

            MapFormView.__super__.initialize.apply(this, arguments);

            this.setupModelEvents();
        },

        // Always use the same model
        getModel: function(id) {
            return this.model;
        },

        validator: models.MapValidator,

        // Never reset the model
        resetModel: function() {
        },

        getDataBindings: function() {
            return {map: this.model};
        }
    });


    var AddCustomMapFormView = JQueryUIModalFormView.extend({
        initialize: function() {
            this.options.dialog = _.extend({
                height: 450,
                width: 700
            }, this.options.dialog);

            AddCustomMapFormView.__super__.initialize.apply(this, arguments);

            this.map = this.$el.find('#custom-map').mapGenerator({
                change: this.updateSelection
            });
        },

        validator: models.CustomMapValidator,

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
        },

        getDataBindings: function() {
            return {map: this.model};
        },

        show: function() {
            this.model.clear();
            this.map.clearSelection();
            // Have to set these manually since reload() doesn't get called
            this.setupModelEvents();
            AddCustomMapFormView.__super__.show.apply(this);
            this.map.resize();
        }
    });


    var AddMapFromUploadFormView = JQueryUIModalFormView.extend({
        initialize: function() {
            var _this = this;
            this.options.dialog = _.extend({
                height: 250,
                width: 500
            }, this.options.dialog);

            AddMapFromUploadFormView.__super__.initialize.apply(this, arguments);

            this.uploadUrl = this.options.uploadUrl;
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

        validator: models.MapValidator,

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
        initialize: function() {
            this.winds = this.options.winds;
            this.router = this.options.router;

            this.options.dialog = _.extend({
                width: 750,
                height: 550,
                title: "Edit Wind Mover"
            }, this.options.dialog);

            WindMoverFormView.__super__.initialize.apply(this, arguments);

            // Extend prototype's events with ours.
            this.events = _.extend({}, FormView.prototype.events, this.events);
        },

        showWind: function() {
            var windForm;
            var windId = this.model.get('wind_id');
            var windFormId = this.id + '_wind';

            if (windId === 'new') {
                windId = null;
            }

            if (windId) {
                 var wind = this.winds.get(windId);

                if (wind === undefined) {
                    alert('Wind does not exist!');
                    console.log('Invalid wind ID: ', windId);
                    return;
                }

                windForm = new EmbeddedWindFormView({
                    id: windFormId,
                    collection: this.winds,
                    defaults: this.options.defaultWind,
                    gnomeModel: this.gnomeModel
                });

                this.model.set('wind_id', windId);
            } else {
                windForm = new EmbeddedAddWindFormView({
                    id: windFormId,
                    collection: this.winds,
                    defaults: this.options.defaultWind,
                    gnomeModel: this.gnomeModel
                });
            }

            if (this.windForm) {
                this.windForm.resetModel();
                this.windForm.undelegateEvents();
                this.windForm = null;
            }

            this.windForm = windForm;
            this.windForm.reload(windId);
            this.windForm.show();
        },

        prepareForm: function() {
            var tmpl = _.template($("#wind-select").html());
            var windSelect = this.getElementByName('wind_id');

            windSelect.find('option').not('option[value="new"]').remove();

            for (var i = 0; i < this.winds.length; i++) {
                var wind = this.winds.at(i);
                var windOption = $(tmpl({
                    id: wind.id,
                    name: wind.get('name')
                }));
                windSelect.append(windOption);
            }

            // We changed the wind_id select box HTML out from under Rivets.js,
            // so trigger a 'change' event on the model that will reselect the
            // correct option in the wind_id select.
            this.model.trigger('change:wind_id');

            this.setDateFields('.active_start_container', this.model.get('active_start'));
            this.setDateFields('.active_stop_container', this.model.get('active_stop'));
        },

        validator: models.WindMoverValidator,

        getDataBindings: function() {
            return {
                mover: this.model
            };
        },

        submit: function() {
            var _this = this;
            var windId = this.model.get('wind_id');
            this.windForm.submit().then(function() {
                if (windId === 'new') {
                    _this.model.set('wind_id', _this.windForm.model.id);
                }
                WindFormView.__super__.submit.apply(_this, arguments);
            });
        },

        setupModelEvents: function() {
            this.listenTo(this.model, 'change:wind_id', this.showWind);
            WindMoverFormView.__super__.setupModelEvents.apply(this, arguments);
        },

        close: function() {
            WindMoverFormView.__super__.close.apply(this, arguments);
            this.windForm.close();
        }
    });


    var AddWindMoverFormView = WindMoverFormView.extend({
        initialize: function() {
            this.options.dialog = _.extend({
                width: 750,
                height: 550,
                title: "Add Wind Mover"
            }, this.options.dialog);

            AddWindMoverFormView.__super__.initialize.apply(this, arguments);
        },

        /*
         Use a new WindMover every time the form is opened.

         This breaks any event handlers called in superclasses before this
         method is called, so we need to reapply them.
         */
        show: function() {
            // Use "new" as the default so we get the "New Wind" option.
            this.defaults.wind_id = 'new';

            this.model = new models.WindMover(this.defaults, {
                gnomeModel: this.gnomeModel
            });
            this.setupModelEvents();
            this.listenTo(this.model, 'sync', this.closeDialog);
            AddWindMoverFormView.__super__.show.apply(this);
        }
    });


    var ExternalWindDataView = BaseView.extend({
        initialize: function() {
            _.bindAll(this);

            // Setup this.$el, which we need for `setupWindMap`.
            ExternalWindDataView.__super__.initialize.apply(this, arguments);
            this.events = _.extend({}, Backbone.View.prototype.events, this.events);

            this.setupWindMap();
        },

        events: {
            'click .query-source': 'querySource',
        },

        resizeWindMap: function() {
            google.maps.event.trigger(this.windMap, 'resize');
            this.windMap.setCenter(this.windMapCenter.getCenter());
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
            var desc = this.getElementByName('description');

            desc.val(data.description);
            desc.change();

            this.setDateFields('.updated_at_container', moment());

            var wind = this.model;
            var timeseries = [];

            _.each(data.results, function(windData) {
                timeseries.push([windData[0], windData[1], windData[2]]);
            });

            wind.set('timeseries', timeseries);

            // NWS data is in knots, so the entire wind mover data set will have
            // to use knots, since we don't have a way of setting it per value.
            wind.set('units', 'knots');

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
                    _this.resizeWindMap();
                });
            }
        }
    });


    var BaseTimeseriesView = BaseView.extend({
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
            this.events = _.extend({}, Backbone.View.prototype.events, this.events);
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
                    direction: direction + ' &deg;',
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
            var datetime = this.getFormDate(addForm);
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


    var WindFormView = JQueryUIModalFormView.extend({
        initialize: function() {
            this.options.dialog = _.extend({
                width: 750,
                height: 550,
                title: "Edit Wind"
            }, this.options.dialog);

            WindFormView.__super__.initialize.apply(this, arguments);

            // Extend prototype's events with ours.
            this.events = _.extend({}, FormView.prototype.events, this.events);

            this.externalDataView = new ExternalWindDataView({
                id: this.id + '_data_source'
            });

            this.externalDataView.on('dataReceived', this.externalDataReceived);

            this.variableTimeseriesView = new VariableWindTimeseriesView({
                id: this.id + '_variable'
            });

            this.constantTimeseriesView = new ConstantWindTimeseriesView({
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
            var windUpdatedAt = this.getFormDate(
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


    var SurfaceReleaseSpillFormView = JQueryUIModalFormView.extend({
        initialize: function() {
            this.options.dialog = _.extend({
                width: 400,
                height: 550,
                title: "Edit Surface Release Spill"
            }, this.options.dialog);

            SurfaceReleaseSpillFormView.__super__.initialize.apply(this, arguments);

            // Extend prototype's events with ours.
            this.events = _.extend({}, FormView.prototype.events, this.events);
        },

        validator: models.SurfaceReleaseSpillValidator,

        getDataBindings: function() {
            return {spill: this.model};
        },

        prepareForm: function() {
            this.setDateFields('.release_time_container', this.model.get('release_time'));
            SurfaceReleaseSpillFormView.__super__.prepareForm.apply(this, arguments);
        },

        show: function(startCoords, endCoords) {
            SurfaceReleaseSpillFormView.__super__.show.apply(this, arguments);

            if (startCoords) {
                this.model.set('start_position', [startCoords[0], startCoords[1], 0]);
            }
            if (endCoords) {
                this.model.set('end_position', [endCoords[0], endCoords[1], 0]);
            }
        },

        prepareSubmitData: function() {
            this.model.set('release_time', this.getFormDate(this.getForm()));
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
        initialize: function() {
            this.options.dialog = _.extend({
                width: 400,
                height: 550,
                title: "Add Surface Release Spill"
            }, this.options.dialog);

            AddSurfaceReleaseSpillFormView.__super__.initialize.apply(this, arguments);
        },

        show: function(coords) {
            this.model = new models.SurfaceReleaseSpill(this.defaults, {
                gnomeModel: this.gnomeModel
            });
            this.setupModelEvents();
            AddSurfaceReleaseSpillFormView.__super__.show.apply(this, arguments);
        }
    });


    var GnomeSettingsFormView = JQueryUIModalFormView.extend({
        initialize: function() {
            this.options.dialog = _.extend({
                height: 300,
                width: 470
            }, this.options.dialog);

            GnomeSettingsFormView.__super__.initialize.apply(this, arguments);
        },

        validator: models.GnomeModelValidator,

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

        prepareSubmitData: function() {
            this.model.set('start_time', this.getFormDate(this.getForm()));
        },

        show: function(withDefaults) {
            GnomeSettingsFormView.__super__.show.apply(this, arguments);

            // XXX: Do this here instead of in prototype because we always
            // have an ID. Should come before we set the start_time_container
            // because it will set the model's start_time.
            if (withDefaults) {
                this.model.set(this.defaults);
            }

            this.setDateFields('.start_time_container', moment(this.model.get('start_time')));
        }
    });


    var RandomMoverFormView = JQueryUIModalFormView.extend({
        initialize: function() {
            this.options.dialog = _.extend({
                height: 350,
                width: 380,
                title: "Edit Random Mover"
            }, this.options.dialog);

            RandomMoverFormView.__super__.initialize.apply(this, arguments);
        },

        validator: models.RandomMoverValidator,

        show: function() {
            RandomMoverFormView.__super__.show.apply(this, arguments);
            this.setDateFields('.active_start_container', this.model.get('active_start'));
            this.setDateFields('.active_stop_container', this.model.get('active_stop'));
        },

        prepareSubmitData: function() {
            this.model.set('active_start', this.getFormDate(
                this.$el.find('.active_start_container')));
            this.model.set('active_stop', this.getFormDate(
                this.$el.find('.active_stop_container')));
        },

        getDataBindings: function() {
            return {mover: this.model}
        }
    });


    var AddRandomMoverFormView = RandomMoverFormView.extend({
        initialize: function(options) {
            this.options.dialog = _.extend({
                height: 350,
                width: 380,
                title: "Add Random Mover"
            }, this.options.dialog);

            AddRandomMoverFormView.__super__.initialize.apply(this, arguments);
        },

        show: function() {
            this.model = new models.RandomMover(this.defaults, {
                gnomeModel: this.gnomeModel
            });
            this.setupModelEvents();
            AddRandomMoverFormView.__super__.show.apply(this, arguments);
        }
    });


    return {
        AddMapFormView: AddMapFormView,
        AddMoverFormView: AddMoverFormView,
        AddEnvironmentFormView: AddEnvironmentFormView,
        AddSpillFormView: AddSpillFormView,
        AddWindFormView: AddWindFormView,
        WindFormView: WindFormView,
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
        BaseView: BaseView,
        FormViewContainer: FormViewContainer,
        GnomeSettingsFormView: GnomeSettingsFormView,
        MultiStepFormView: MultiStepFormView,
        LocationFileWizardFormView: LocationFileWizardFormView,
        ModelNotFoundException: ModelNotFoundException,
        deferreds: deferreds
    };

});
