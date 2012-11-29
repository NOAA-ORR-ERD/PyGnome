// util.js: Utility functions for the WebGNOME JavaScript application.

define([
    'jquery',
    'lib/underscore'
], function($, _) {

    /*
     Safely wrap `window.console.log`. Sends all arguments to that function
     if it exists.
     */
    function log(var_args) {
        var args = Array.prototype.slice.call(arguments);
        if (window.console && window.console.log) {
            window.console.log.apply(window.console, args);
        }
    }

    /*
     Generic AJAX error handler.
     Retry on error if the request specified tryCount.
     */
    function handleAjaxError(xhr, textStatus, errorThrown) {
        if (textStatus === 'timeout') {
            this.tryCount++;
            if (this.tryCount <= this.retryLimit) {
                // Retry count is below the limit, so try the request again.
                $.ajax(this);
                return;
            }
            return;
        }

        window.alert('Could not connect to server.');
        window.noaa.erd.util.log(xhr, textStatus, errorThrown);
    }


    /*
     Fix Bootstrap modal windows to work with responsive design and adjust width
     based on data-width properties.
     */
    function fixModals() {
        var $modals = $('.modal');
        $modals.modalResponsiveFix({debug: true});
        $modals.touchScroll();
    }


    /*
      Retrieve a message object from the object `data` if the `message` key
      exists, annotate the message object ith an `error` value set to true
      if the message is an error type, and return the message object.
     */
    function parseMessage(data) {
        var message;

        if (data === null || data === undefined) {
            return false;
        }

        if (_.has(data, 'message')) {
            message = data.message;

            if (data.message.type === 'error') {
                message.error = true;
            }

            return message;
        }

        return false;
    }

    /*
     Return a UTC date string for `timestamp`, which should be in a format
     acceptable to `Date.parse`.
     */
    function getUTCStringForTimestamp(timestamp) {
        var date = new Date(Date.parse(timestamp));
        if (date) {
            timestamp = date.toUTCString();
        }
        return timestamp;
    }
    
    return {
        log: log,
        handleAjaxError: handleAjaxError,
        fixModals: fixModals,
        parseMessage: parseMessage,
        getUTCStringForTimestamp: getUTCStringForTimestamp
    };

});
