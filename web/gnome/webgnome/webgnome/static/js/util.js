// util.js: Utility functions for the WebGNOME JavaScript application.
"use strict";

window.noaa.erd.util = window.noaa.erd.util || {};

/*
 Safely wrap `window.console.log`. Sends all arguments to that function
 if it exists.
 */
window.noaa.erd.util.log = function(var_args) {
    var args = Array.prototype.slice.call(arguments);
    if (window.console && window.console.log) {
        window.console.log.apply(window.console, args);
    }
};

/*
 Generic AJAX error handler.
 Retry on error if the request specified tryCount.
 */
window.noaa.erd.util.handleAjaxError = function(xhr, textStatus, errorThrown) {
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
};
