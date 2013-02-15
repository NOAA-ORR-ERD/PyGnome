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
            'location/:location': 'setLocation'
        },

        initialize: function(options) {
            this.newModel = options.newModel;

            options.appOptions.router = this;
            options.splashOptions.router = this;
            options.locationMapOptions.router = this;

            this.splashView = new views.SplashView(options.splashOptions);
            this.appView = new app_view.AppView(options.appOptions);
            this.locationFileMapView = new views.LocationFileMapView(
                options.locationMapOptions
            );
        },

        index: function() {
            if (this.newModel) {
                this.navigate('splash', true);
            } else {
                this.navigate('model', true);
            }
        },
        
        model: function() {
            this.newModel = false;
            this.splashView.hide();
            this.locationFileMapView.hide();
            this.appView.mapView.hidePlaceholder();
            this.appView.show();
        },

        locationMap: function() {
            this.splashView.hide();
            this.appView.hide();
            this.locationFileMapView.show();
        },

        setLocation: function(location) {
            this.locationFileMapView.loadLocationFile(location);
        },
        
        splash: function() {
            this.appView.hide();
            this.locationFileMapView.hide();
            this.splashView.show();
        }
    });


    return {
        Router: Router
    }
});
