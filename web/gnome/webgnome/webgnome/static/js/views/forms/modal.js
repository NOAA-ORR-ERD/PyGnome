define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'views/forms/deferreds',
    'views/forms/base',
    'lib/jquery.ui',
], function($, _, Backbone, deferreds, base) {
     /*
     An `FormView` subclass that displays in a JQuery UI modal window.
     */
    var JQueryUIModalFormView = base.FormView.extend({
        initialize: function() {
            JQueryUIModalFormView.__super__.initialize.apply(this, arguments);
            this.setupDialog();
            _.bindAll(this, 'closeDialog', 'close');
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
                close: function() {
                    _this.close();
                },
                beforeClose: this.beforeClose
            }, this.options.dialog || {});

            this.title = dialogOptions.title;
            this.$el.dialog(dialogOptions);

            // Workaround the default JQuery UI Dialog behavior of auto-
            // focusing on the first input element by adding an invisible one.
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
            this.trigger(base.FormView.CANCELED);
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
                this.trigger(base.FormView.SUBMITTED);
            } else if (button.fnName === 'cancel') {
                this.trigger(base.FormView.CANCELED);
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
                        var promise = _this.validate();

                        if (!promise) {
                            return;
                        }

                        promise.done(function() {
                            _this.addDeferredButton(button);
                        });
                        promise.fail(_this.handleValidatorError);
                    }
                });
            });

            this.originalButtons = this.$el.dialog('option', 'buttons');
            this.$el.dialog('option', 'buttons', buttons);
        }
    });

    return {
        JQueryUIModalFormView: JQueryUIModalFormView
    }
});