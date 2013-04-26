define([
    'lib/underscore',
    'views/forms/all',
    'views/menu',
    'views/message',
    'views/map',
    'views/map_control',
    'views/tree',
    'views/tree_control',
    'views/location_file',
    'views/splash',
], function(_, forms) {
    // Ignore _, forms
    var modules = Array.prototype.slice.call(arguments, 2);

    /*
      A helper module that makes it easy to require all views in app.js.
    */
    var views = {};

    for (var i = 0; i < modules.length; i++) {
        _.extend(views, modules[i]);
    }

    // Views object will have a 'forms' object that includes all the forms.
    views.forms = forms;

    return views;
});
