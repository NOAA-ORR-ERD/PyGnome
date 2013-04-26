define([
    'lib/underscore',
    'views/base',
    'views/forms/choosers',
    'views/forms/container',
    'views/forms/deferreds',
    'views/forms/gnome_settings',
    'views/forms/location_file_wizard',
    'views/forms/map',
    'views/forms/modal',
    'views/forms/multi_step',
    'views/forms/random_mover',
    'views/forms/surface_release_spill',
    'views/forms/timeseries',
    'views/forms/wind',
    'views/forms/wind_mover'
], function() {
    var _ = arguments[0];
    // Ignore _
    var modules = Array.prototype.slice.call(arguments, 1);
    
    /*
      A helper module that makes it easy to require all forms in app.js.
    */
    var forms = {};

    for (var i = 0; i < modules.length; i++) {
        _.extend(forms, modules[i]);
    }

    return forms;
});
