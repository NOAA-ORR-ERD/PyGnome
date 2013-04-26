define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'views/forms/modal',
], function($, _, Backbone, models, modal) {
    var GnomeSettingsFormView = modal.JQueryUIModalFormView.extend({
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
            this.model.set('start_time', this.getDate(this.getForm()));
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

    return {
        GnomeSettingsFormView: GnomeSettingsFormView
    }
});