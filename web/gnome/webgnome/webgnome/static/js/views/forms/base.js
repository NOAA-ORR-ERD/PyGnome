define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'util',
    'lib/rivets',
    'views/base',
    'lib/moment',
], function($, _, Backbone, models, util, rivets, base) {
    /*
     `FormView` is the base class for forms intended to wrap Backbone models.

     Submitting a form from `FormView` sends the value of its inputs to the
     model object passed into the form's constructor, which syncs it with the
     server.
     */
    var FormView = base.BaseView.extend({
        initialize: function() {
            var _this = this;

            // XXX: Using _.bindAll without specifying a function (bind ALL
            // values on the object) breaks IE.
            // _.bindAll(this);

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

            if (!this.model) {
                return;
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

        submit: function(opts) {
            var _this = this;

            var options = $.extend({}, {
                success: function() {
                    _this.trigger(FormView.SUBMITTED);
                }
            }, opts);

            this.prepareSubmitData();
            this.clearErrors();

            var errors = this.model.validate();

            if (errors && errors.length) {
                for (var i = 0; i < errors.length; i++) {
                    var obj = errors[i];
                    this.handleFieldError(obj);
                }
                return;
            }

            if (this.collection) {
                this.collection.add(this.model);
            }

            // Return a jQuery Promise object.
            return this.model.save(null, options);
        },

        cancel: function() {
            this.resetModel();
            this.trigger(FormView.CANCELED);
        },

        resetModel: function() {
            if (this.model && this.model.id) {
                this.model.fetch({reloadTree: false});
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
                throw new base.ModelNotFoundException('Model missing on reload.');
            }

            this.setupModelEvents();
        }
    }, {
        SUBMITTED: 'formView.submitted',
        CANCELED: 'formView:canceled',
        REFRESHED: 'formView:refreshed',
        SHOW_FORM: 'formView:showForm'
    });

    return {
        FormView: FormView
    }
});