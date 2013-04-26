define([
    'jquery',
    'lib/underscore'
], function($, _) {
    var DeferredManager = function() {
        this.deferreds = [];
        this.namedDeferreds = {};
    };

    DeferredManager.prototype = {
        /*
         Add a deferred method call.

         Calling this method multiple times with the same `fn` value will add
         the method call multiple times.
         */
        add: function(fn) {
            this.deferreds.push(fn);
        },

        /*
         Add a deferred method call by name.

         Multiple calls to this method using the same value for `name` will
         overwrite the value, resulting in only one deferred method call for
         each `name` value.
         */
        addNamed: function(name, fn) {
            this.namedDeferreds[name] = fn;
        },

        /*
         Loop through the closures saved in `this.deferreds` and
         `this.namedDeferreds` and call them. Keep track of any result that is
         a jQuery Deferred object in an `actualDeferreds` array.

         Calling this method returns a jQuery Deferred object that is only
         resolved when all Deferred objects returned by closures are resolved.

         We attach `done` and `fail` handlers to any deferreds in `actualDeferreds`
         so that when all of these Deferred objects are resolved, we resolve
         the call to `run`. If *any* of Deferred objects fail, we fail the call
         to `run`.
         */
        run: function() {
            var dfd = $.Deferred();
            var potentialDeferreds = this.deferreds.concat(
                _.values(this.namedDeferreds));
            var actualDeferreds = [];

            _.each(potentialDeferreds, function(fn) {
                var result = fn();

                if (result && typeof result.done === 'function') {
                    actualDeferreds.push(result);
                }
            });

            $.when.apply(null, actualDeferreds).done(function() {
                dfd.resolve();
            }).fail(function() {
                // XXX: If any deferred method fails, the run operation fails.
                dfd.fail();
            });

            return dfd;
        }
    };

    return new DeferredManager();
});
