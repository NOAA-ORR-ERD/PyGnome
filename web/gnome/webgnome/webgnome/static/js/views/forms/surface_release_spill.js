define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'views/forms/modal',
    'views/forms/base',
], function($, _, Backbone, models, modal, base) {
    var SurfaceReleaseSpillFormView = modal.JQueryUIModalFormView.extend({
        initialize: function() {
            this.options.dialog = _.extend({
                width: 400,
                height: 550,
                title: "Edit Surface Release Spill"
            }, this.options.dialog);

            SurfaceReleaseSpillFormView.__super__.initialize.apply(this, arguments);

            // Extend prototype's events with ours.
            this.events = _.extend({}, base.FormView.prototype.events, this.events);
        },

        validator: models.SurfaceReleaseSpillValidator,

        getDataBindings: function() {
            return {spill: this.model};
        },

        prepareForm: function() {
            this.setDateFields('.release_time_container', this.model.get('release_time'));
            this.setDateFields('.end_release_time_container', this.model.get('end_release_time'));
            SurfaceReleaseSpillFormView.__super__.prepareForm.apply(this, arguments);
        },

        show: function(startCoords, endCoords) {
            SurfaceReleaseSpillFormView.__super__.show.apply(this, arguments);

            if (this.gnomeModel) {
                this.setDateFields('.release_time_container', this.gnomeModel.get('start_time'));
            }

            if (startCoords) {
                this.model.set('start_position', [startCoords[0], startCoords[1], 0]);
            }
            if (endCoords) {
                this.model.set('end_position', [endCoords[0], endCoords[1], 0]);
            }
        },

        prepareSubmitData: function() {
            var form = this.getForm();
            var releaseTime = this.getDate(form.find('.release_time_container'));
            var endReleaseTime = this.getDate(form.find('.end_release_time_container'));
            this.model.set('release_time', releaseTime);
            this.model.set('end_release_time', endReleaseTime);
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


    return {
        SurfaceReleaseSpillFormView: SurfaceReleaseSpillFormView,
        AddSurfaceReleaseSpillFormView: AddSurfaceReleaseSpillFormView
    }
});