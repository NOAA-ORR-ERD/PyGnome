define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models',
    'views/forms/modal',
    'map_generator',
    'lib/jquery.ui',
    'lib/jquery.fileupload',
    'lib/jquery.iframe-transport',
    '//netdna.bootstrapcdn.com/bootstrap/3.0.0/js/bootstrap.min.js',
], function($, _, Backbone, models, modal) {

    /*
     A map form that allows the user to edit details of an existing map.
     */
    var MapFormView = modal.JQueryUIModalFormView.extend({
        initialize: function() {
            this.options.dialog = _.extend({
                height: 220,
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


    /*
     A map form that reuses jQuery components from GOODs and posts to that
     service to obtain a BNA file for the coastline selected by the user.
     */
    var AddCustomMapFormView = modal.JQueryUIModalFormView.extend({
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
        },

        submit: function() {
            AddCustomMapFormView.__super__.submit.apply(this, [{
                added: true
            }]);
        }
    });


    /*
     A map form the allows the user to upload a custom BNA file.
     */
    var AddMapFromUploadFormView = modal.JQueryUIModalFormView.extend({
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

                // TODO: This should set a spinner.
                submit: function(e, data) {
                    saveButton.button('disable');
                },

                // TODO: This should unset a spinner.
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
        },

        submit: function() {
            AddMapFromUploadFormView.__super__.submit.apply(this, [{
                added: true
            }]);
        }
    });

    return {
        MapFormView: MapFormView,
        AddCustomMapFormView: AddCustomMapFormView,
        AddMapFromUploadFormView: AddMapFromUploadFormView
    }
});