define([
    'lib/underscore',
    'lib/backbone',
    'app',
], function(_, Backbone, app) {

    var Router = Backbone.Router.extend({
        routes: {
            '': 'index',
            'model': 'model',
            'splash': 'splash',
            'location_map': 'locationMap',
            'wind/:id': 'wind',
            'spill/:id': 'spill'
        },

        initialize: function(options) {
            this.newModel = options.newModel;
            options.appOptions.router = this;
            this.appView = new app.AppView(options.appOptions);
        },

        index: function() {
            if (this.newModel) {
                this.navigate('splash', true);
            } else {
                this.navigate('model', true);
            }
        },
        
        model: function() {
        	if (!this.appView.sbUncollapsedWidth) {
        		this.appView.disableFullscreen();
        	}
            this.newModel = false;
            this.appView.showSection('model');
        },

        locationMap: function() {
            this.appView.showSection('location-file-map')
        },

        splash: function() {
            if (!this.newModel) {
                return this.navigate('model', true);
            }

            return this.appView.showSection('splash-page')
        },

        showForm: function(formId, objectId) {
            this.appView.disableFullscreen();
            this.appView.showSection('model');
            this.appView.formViews.hideAll();
            var formView = this.appView.formViews.get(formId);
            formView.reload(objectId);
            formView.show();
        },

        wind: function(id) {
            var formId;

            if (id === 'new') {
                id = null;
                formId = 'add-wind';
            } else {
                formId = 'edit-wind';
            }
            this.showForm(formId, id);
        },

        spill: function(id) {
            var formId;

            if (id === 'new') {
                id = null;
                formId = 'add-surface-release-spill';
            } else {
                formId = 'edit-surface-release-spill';
            }
            this.showForm(formId, id);
        }
    });


    return {
        Router: Router
    }
});
