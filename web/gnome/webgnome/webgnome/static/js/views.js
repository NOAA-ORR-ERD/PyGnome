define([
    'views/menu',
    'views/message',
    'views/map',
    'views/map_control',
    'views/tree',
    'views/tree_control',
    'views/location_file',
    'views/splash'
], function(menu, message, map, map_control, tree, tree_control, location_file, splash) {
    /*
      A helper module that makes it easy to require all views in app.js.
    */
    var views = {};

    for (var i = 0; i < arguments.length; i++) {
	_.extend(views, arguments[i]);
    }

    return views;
});
