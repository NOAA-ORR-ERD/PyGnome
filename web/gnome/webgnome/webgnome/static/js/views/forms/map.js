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
    'lib/bootstrap.file-input',
], function($, _, Backbone, models, modal) {
    var MapFormView = modal.JQueryUIModalFormView.extend({
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
        }
    });


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

    return {
        MapFormView: MapFormView,
        AddCustomMapFormView: AddCustomMapFormView,
        AddMapFromUploadFormView: AddMapFromUploadFormView
    }
});