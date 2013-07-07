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
        if (console && console.log) {
            Function.prototype.apply.apply(console.log, [console, arguments]);
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

     /*\
     |*|
     |*|  :: cookies.js ::
     |*|
     |*|  A complete cookies reader/writer framework with full unicode support.
     |*|
     |*|  https://developer.mozilla.org/en-US/docs/DOM/document.cookie
     |*|
     |*|  Syntaxes:
     |*|
     |*|  * docCookies.setItem(name, value[, end[, path[, domain[, secure]]]])
     |*|  * docCookies.getItem(name)
     |*|  * docCookies.removeItem(name[, path])
     |*|  * docCookies.hasItem(name)
     |*|  * docCookies.keys()
     |*|
     \*/
    var Cookies = {
        getItem: function(sKey) {
            if (!sKey || !this.hasItem(sKey)) {
                return null;
            }
            return unescape(document.cookie.replace(new RegExp("(?:^|.*;\\s*)" + escape(sKey).replace(/[\-\.\+\*]/g, "\\$&") + "\\s*\\=\\s*((?:[^;](?!;))*[^;]?).*"), "$1"));
        },
        setItem: function(sKey, sValue, vEnd, sPath, sDomain, bSecure) {
            if (!sKey || /^(?:expires|max\-age|path|domain|secure)$/i.test(sKey)) {
                return;
            }
            var sExpires = "";
            if (vEnd) {
                switch (vEnd.constructor) {
                    case Number:
                        sExpires = vEnd === Infinity ? "; expires=Tue, 19 Jan 2038 03:14:07 GMT" : "; max-age=" + vEnd;
                        break;
                    case String:
                        sExpires = "; expires=" + vEnd;
                        break;
                    case Date:
                        sExpires = "; expires=" + vEnd.toGMTString();
                        break;
                }
            }
            document.cookie = escape(sKey) + "=" + escape(sValue) + sExpires + (sDomain ? "; domain=" + sDomain : "") + (sPath ? "; path=" + sPath : "") + (bSecure ? "; secure" : "");
        },
        removeItem: function(sKey, sPath) {
            if (!sKey || !this.hasItem(sKey)) {
                return;
            }
            document.cookie = escape(sKey) + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT" + (sPath ? "; path=" + sPath : "");
        },
        hasItem: function(sKey) {
            return (new RegExp("(?:^|;\\s*)" + escape(sKey).replace(/[\-\.\+\*]/g, "\\$&") + "\\s*\\=")).test(document.cookie);
        },
        keys: /* optional method: you can safely remove it! */ function() {
            var aKeys = document.cookie.replace(/((?:^|\s*;)[^\=]+)(?=;|$)|^\s*|\s*(?:\=[^;]*)?(?:\1|$)/g, "").split(/\s*(?:\=[^;]*)?;\s*/);
            for (var nIdx = 0; nIdx < aKeys.length; nIdx++) {
                aKeys[nIdx] = unescape(aKeys[nIdx]);
            }
            return aKeys;
        }
    };

    function refresh() {
        window.location = window.location.protocol + "//" + window.location.host;
    }

    // https://github.com/redpie/backbone-schema/pull/3
    function formatError(str, values) {
        return str.replace(/%((\w+))/g, function(match, name) {
            return values[name] || match;
        });
    }

    return {
        log: log,
        handleAjaxError: handleAjaxError,
        parseMessage: parseMessage,
        formatTimestamp: formatTimestamp,
        cardinalName: cardinalName,
        cardinalAngle: cardinalAngle,
        Cookies: Cookies,
        refresh: refresh,
        parseError: formatError
    };

});
