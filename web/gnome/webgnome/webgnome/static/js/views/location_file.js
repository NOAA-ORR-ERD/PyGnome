define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'lib/jquery.imagesloaded.min',
    'lib/gmaps'
], function($, _, Backbone) {
    var LocationFileMapView = Backbone.View.extend({
        events: {
            'click .load-location-file': 'locationChosen'
        },

        initialize: function() {
            _.bindAll(this);
            this.mapCanvas = $(this.options.mapCanvas);
            this.locationFilesMeta = this.options.locationFilesMeta;

            this.setupLocationFileMap();
        },

        locationChosen: function(event) {
            event.preventDefault();
            var location = $(event.target).data('location');
            if (location) {
                this.trigger(LocationFileMapView.LOCATION_CHOSEN, location);
            }
            this.infoWindow.close();
        },

        setupLocationFileMap: function() {
            var _this = this;
            this.center = new google.maps.LatLng(-34.397, 150.644);
            this.infoWindow = new google.maps.InfoWindow();
            var gmapOptions = {
                center: this.center,
                backgroundColor: '#212E68',
                zoom: 1,
                scrollwheel: true,
                scaleControl: true,
                mapTypeId: google.maps.MapTypeId.HYBRID,
            };

            this.locationFileMap = new google.maps.Map(
                this.mapCanvas[0], gmapOptions);

            this.locationFilesMeta.each(function(location) {
                var latLng = new google.maps.LatLng(
                    location.get('latitude'), location.get('longitude'));

                var marker = new google.maps.Marker({
                    position: latLng,
                    map: _this.locationFileMap
                });

                google.maps.event.addListener(marker, 'click', function() {
                    var template = _.template(
                        $('#location-file-template').text());
                    _this.infoWindow.setContent(template(location.toJSON()));
                    _this.infoWindow.open(_this.locationFileMap, marker);
                });
            });
        },

        centerMap: function() {
            google.maps.event.trigger(this.locationFileMap, 'resize');
            this.locationFileMap.setCenter(this.center);
        },

        show: function() {
            this.$el.imagesLoaded(this.centerMap);
        }
    }, {
        LOCATION_CHOSEN: 'locationFileMapView:locationChosen'
    });

    return {
        LocationFileMapView: LocationFileMapView
    }
});
