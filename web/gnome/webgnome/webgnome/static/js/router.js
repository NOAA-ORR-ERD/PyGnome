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
            'location_map': 'locationMap'
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
        }
    });


    return {
        Router: Router
    }
});
