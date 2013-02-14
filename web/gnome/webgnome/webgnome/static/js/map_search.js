(function($) {

    $.fn.mapGenerator = function(options, value) {

        if (!this.length) {
            return;
        }

        if (this.length > 1) {
            this.each(function() {
                $(this).mapGenerator(options, value)
            });
            return this;
        }

        var settings = $.extend({
            change: null,
            rect: {northlat: 0, southlat: 0, westlon: 0, eastlon: 0}
        }, options);

        // Private state variables
        var state = {};
        var mapCanvas = this;
        var rect = settings.rect;

        return this;
    };
})(jQuery);

