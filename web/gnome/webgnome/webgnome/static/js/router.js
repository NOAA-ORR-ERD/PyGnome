define([
    'lib/underscore',
    'lib/backbone',
    'views',
    'app_view'
], function(_, Backbone, views, app_view) {

    var Router = Backbone.Router.extend({
        routes: {
            '': 'index',
            'model': 'model',
            'splash': 'splash',
            'location_map': 'locationMap',
            'wind/:id': 'wind'
        },

        initialize: function(options) {
            this.newModel = options.newModel;
            options.appOptions.router = this;
            this.appView = new app_view.AppView(options.appOptions);
        },

        index: function() {
            if (this.newModel) {
                this.navigate('splash', true);
            } else {
                this.navigate('model', true);
            }
        },
        
        model: function() {
            this.appView.disableFullscreen();
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

        wind: function(id) {
            var formId;

            if (id === 'new') {
                id = null;
                formId = 'add-wind';
            } else {
                formId = 'edit-wind';
            }

            this.appView.disableFullscreen();
            this.appView.showSection('model');
            this.appView.formViews.hideAll();
            var formView = this.appView.formViews.get(formId);
            formView.reload(id);
            formView.show();
        }
    });


    return {
        Router: Router
    }
});
