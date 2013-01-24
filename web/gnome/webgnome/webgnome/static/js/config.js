
// Configure RequireJS
requirejs.config({
    baseUrl: "/static/js",
    priority: ['lib/underscore'],
    paths: {
        async: 'lib/async'
    },
    shim: {
        'lib/gmaps-amd': {
            exports: "google"
        },
        'lib/jquery.dynatree.min': ['lib/jquery-ui-1.9.2.custom.min', 'lib/jquery.cookie'],
        'lib/underscore': {
            exports: "_"
        },
        'lib/backbone': {
            deps: ["lib/underscore", "jquery"],
            exports: "Backbone"
        },
        'lib/mousetrap': {
            exports: "Mousetrap"
        },
        'lib/geo': {
            exports: "Geo"
        }
    }
});