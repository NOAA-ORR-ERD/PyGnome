define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
], function($, _, Backbone) {
    var SplashView = Backbone.View.extend({
        initialize: function() {
            this.router = this.options.router;
        },

        events: {
            'click .choose-location': 'chooseLocation',
            'click .build-model': 'buildModel'
        },

        chooseLocation: function(event) {
            event.preventDefault();
            this.router.navigate('location_map', true);
        },

        buildModel: function(event) {
            event.preventDefault();
            this.router.navigate('model', true);
        }
    });

    return {
        SplashView: SplashView
    }
});