define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'models'
], function($, _, Backbone, models) {
     /*
     `MessageView` is responsible for displaying messages sent back from the server
     during AJAX form submissions. These are non-form error conditions, usually,
     but can also be success messages.
     */
    var MessageView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);

            this.options.gnomeRun.on(
                models.GnomeRun.MESSAGE_RECEIVED, this.displayMessage);
            this.options.gnomeModel.on(
                models.GnomeModel.MESSAGE_RECEIVED, this.displayMessage);
            this.options.surfaceReleaseSpills.on(
                models.SurfaceReleaseSpill.MESSAGE_RECEIVED, this.displayMessage);
            this.options.windMovers.on(
                models.WindMover.MESSAGE_RECEIVED, this.displayMessage);

            this.hideAll();
        },

        displayMessage: function(message) {
            if (!_.has(message, 'type') || !_.has(message, 'text')) {
                return false;
            }

            var alertDiv = $('div .alert-' + message.type);

            if (message.text && alertDiv) {
                alertDiv.find('span.message').text(message.text);
                alertDiv.fadeIn(10);
                // The hidden class makes the div hidden on page load. After
                // we fade out the first time, jQuery sets the display: none;
                // value on the element directly.
                alertDiv.removeClass('hidden');
            }

            this.hideAll();
            return true;
        },

        // Hide alerts automatically after a timeout.
        hideAll: function() {
            setTimeout(function() {
                $('.alert').fadeOut();
            }, 2000);
        }
    });

    return {
        MessageView: MessageView
    }
});