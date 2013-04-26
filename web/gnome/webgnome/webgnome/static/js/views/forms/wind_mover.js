define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'views/forms/modal',
    'views/forms/base',
    'views/forms/wind',
    'lib/bootstrap-tab',
], function($, _, Backbone, models, modal, base, wind) {
    /*
     `WindMoverFormView` handles the WindMover form.
     */
    var WindMoverFormView = modal.JQueryUIModalFormView.extend({
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
            this.events = _.extend({}, base.FormView.prototype.events, this.events);
        },

        showWind: function() {
            var windForm;
            var windId = this.model.get('wind_id');
            var windFormId = this.id + '_wind';

            if (windId === 'new') {
                windId = null;
            }

            if (windId) {
                 var windObj = this.winds.get(windId);

                if (windObj === undefined) {
                    alert('Wind does not exist!');
                    console.log('Invalid wind ID: ', windId);
                    return;
                }

                windForm = new wind.EmbeddedWindFormView({
                    id: windFormId,
                    collection: this.winds,
                    defaults: this.options.defaultWind,
                    gnomeModel: this.gnomeModel
                });

                this.model.set('wind_id', windId);
            } else {
                windForm = new wind.EmbeddedAddWindFormView({
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

            var activeStart = this.model.get('active_start');
            var activeStop = this.model.get('active_stop');

            // TODO: Is this really how we want to handle this, or should the
            // model return a special datetime for -inf and inf?
            if (activeStart != '-inf') {
                this.setDateFields('.active_start_container', activeStart);
            }
            if (activeStop != 'inf') {
                this.setDateFields('.active_stop_container', activeStop);
            }
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
                wind.WindFormView.__super__.submit.apply(_this, arguments);
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


    return {
        AddWindMoverFormView: AddWindMoverFormView,
        WindMoverFormView: WindMoverFormView
    }
});