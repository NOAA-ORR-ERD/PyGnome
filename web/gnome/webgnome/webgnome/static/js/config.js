
// Configure RequireJS
requirejs.config({
    baseUrl: "/static/js",
    priority: ['lib/underscore'],
    shim: {
        'lib/jquery.dynatree.min': ['lib/jquery-ui-1.8.24.custom.min', 'lib/jquery.cookie'],
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