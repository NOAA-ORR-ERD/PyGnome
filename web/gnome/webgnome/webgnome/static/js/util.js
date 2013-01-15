// util.js: Utility functions for the WebGNOME JavaScript application.

define([
    'jquery',
    'lib/underscore',
    'lib/moment'
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
        log(xhr, textStatus, errorThrown);
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
     Return a date string for `timestamp`.

     `timestamp` which should be in a format acceptable to `Date.parse`.
     */
    function formatTimestamp(timestamp) {
        var date = moment(timestamp);
        if (date.isValid()) {
            timestamp = date.format('MM/DD/YYYY HH:mm')
        }
        return timestamp;
    }


    var dirNames = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S',
                    'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];

    function cardinalName(angle) {
        return dirNames[Math.floor((+(angle) + 360 / 32) / (360 / 16.0) % 16)];
    }

    function cardinalAngle(name) {
        var idx = dirNames.indexOf(name.toUpperCase());
        if (idx === -1) {
            return null;
        }
        else {
            return (360.0 / 16) * idx
        }
    }

    return {
        log: log,
        handleAjaxError: handleAjaxError,
        parseMessage: parseMessage,
        formatTimestamp: formatTimestamp,
        cardinalName: cardinalName,
        cardinalAngle: cardinalAngle
    };

});
