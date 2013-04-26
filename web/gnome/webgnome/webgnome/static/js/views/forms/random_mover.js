define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'views/forms/modal',
], function($, _, Backbone, models, modal) {
    var RandomMoverFormView = modal.JQueryUIModalFormView.extend({
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
            this.model.set('active_start', this.getDate(
                this.$el.find('.active_start_container')));
            this.model.set('active_stop', this.getDate(
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
        RandomMoverFormView: RandomMoverFormView,
        AddRandomMoverFormView: AddRandomMoverFormView
    }
});