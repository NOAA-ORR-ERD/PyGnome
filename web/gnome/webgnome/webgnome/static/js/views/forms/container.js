define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'views/forms/base'
], function($, _, Backbone, base) {
    var FormViewContainer = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.formViews = {};
        },

        add: function(view) {
            var _this = this;
            this.formViews[view.id] = view;

            view.on(base.FormView.CANCELED, function(form) {
                _this.trigger(base.FormView.CANCELED, form);
            });
            view.on(base.FormView.REFRESHED, function(form) {
                _this.trigger(base.FormView.REFRESHED, form);
            });
            view.on(base.FormView.MESSAGE_READY, function(message) {
                _this.trigger(base.FormView.MESSAGE_READY, message);
            });

            view.on(base.FormView.SHOW_FORM, this.show);

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
                    formView.once(base.FormView.SUBMITTED, success);
                }
                if (cancel) {
                    formView.once(base.FormView.CANCELED, cancel);
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

    return {
        FormViewContainer: FormViewContainer
    }
});