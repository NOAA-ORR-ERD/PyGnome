
// Configure RequireJS
requirejs.config({
    baseUrl: "/static/js",
    priority: ['lib/underscore', 'lib/jquery.ui'],
    paths: {
        async: 'lib/async',
        'lib/jquery.ui': 'lib/jquery-ui-1.9.2.custom.min'
    },
    shim: {
        'map_generator': ['jquery'],
        'lib/gmaps-amd': {
            exports: "google"
        },
        'lib/jquery.dynatree': ['lib/jquery.ui', 'lib/jquery.cookie'],
        'lib/underscore': {
            exports: "_"
        },
        'lib/rivets': {
            exports: "rivets"
        },
        'lib/backbone': {
            deps: ["lib/underscore", "jquery"],
            exports: "Backbone"
        },
        'lib/backbone-nested': {
            deps: ['lib/backbone']
        },
        'lib/mousetrap': {
            exports: "Mousetrap"
        },
        'lib/geo': {
            exports: "Geo"
        },
        'lib/jquery.fileupload': ['lib/jquery.ui']
    }
});