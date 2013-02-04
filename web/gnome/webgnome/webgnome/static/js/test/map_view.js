define([
    'views',
    'models'
], function(views, models) {

    module('MapView');


    // Test helpers

    var testUrl = "/test";
    var testUrl2 = "/test2";

    var modelRun = new models.ModelRun([], {
        url: testUrl,
        expectedTimeSteps: [],
        currentTimeStep: 0
    });

    var map = new models.Map({
        map_bounds: makeBounds()
    }, {
        url: testUrl2
    });

    function makeBounds() {
        // Use bounds from the Long Island Sound script.
        return [
            [-73.083328, 40.922832], // Bottom left
            [-73.083328, 41.330833], // Top left
            [-72.336334, 41.330833], // Top right
            [-72.336334, 40.922832]  // Bottom right
        ];
    }

    function makeMap(el, placeholderEl) {
        return new views.MapView({
            mapEl: el || '#map-empty',
            placeholderEl: placeholderEl || '#placeholder-empty',
            backgroundImageUrl: '/static/js/test/fixtures/background_map.png',
            frameClass: 'frame',
            activeFrameClass: 'active',
            modelRun: modelRun,
            model: map
        });

    }

    function makeRect(x1, y1, x2, y2) {
        return {start: {x: x1, y: y1}, end: {x: x2, y: y2}};
    }

    function assertRectIsEqual(rect, expected) {
        ok(rect.start.x === expected.start.x);
        ok(rect.start.y === expected.start.y);
        ok(rect.end.x === expected.end.x);
        ok(rect.end.y === expected.end.y);
    }


    // Tests

    test('getRect returns the same rect if user clicked and dragged from left to right', function() {
        var map = makeMap();
        var rect = makeRect(100, 200, 200, 200);
        var expected = {start: rect.start, end: rect.end};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });

    test('getRect returns the same rect if user clicked and dragged from top to bottom', function() {
        var map = makeMap();
        var rect = makeRect(100, 200, 100, 300);
        var expected = {start: rect.start, end: rect.end};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });

    test('getRect returns the same rect if user dragged from top-left to bottom-right', function() {
        var map = makeMap();
        var rect = makeRect(100, 200, 200, 300);
        var expected = {start: rect.start, end: rect.end};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });

    test('getRect flips the rect if the user selected from left to right', function() {
        var map = makeMap();
        var rect = makeRect(200, 200, 100, 200);
        var expected = {start: rect.end, end: rect.start};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });

    test('getRect flips the rect if the user selected from bottom to top', function() {
        var map = makeMap();
        var rect = makeRect(100, 300, 100, 200);
        var expected = {start: rect.end, end: rect.start};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });

    test('getRect flips the rect if the user selected from bottom-right to top-left', function() {
        var map = makeMap();
        var rect = makeRect(200, 300, 100, 200);
        var expected = {start: rect.end, end: rect.start};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });


    test('pixelsFromCoordinates should transform a lat/long coordinate into pixels', function() {
        var el = '#map-loaded';
        var map = makeMap(el);
        var frame = $(el).find('.frame').first();
        // Set an "active" image
        frame.addClass('active');

        // Use coordinates for the spill in the Long Island Sound script.
        var orig = {lat: 41.202120, long: -72.419992};
        var point = map.pixelsFromCoordinates(orig);

        ok(point.x === 710);
        ok(point.y === 411);
    });

    test('coordinatesFromPixels should transform a pixel coordinate into lat/long',function() {
        var el = '#map-loaded';
        var map = makeMap(el);
        var frame = $(el).find('.frame').first();
        // Set an "active" image
        frame.addClass('active');

        // Approximate spill coordinates from the Long Island Sound script.
        var orig = {x: 710, y: 411};
        var point = map.coordinatesFromPixels(orig);

        ok(point.lat === 41.202312684999995);
        ok(point.long === -72.42037082499999);
    });

});