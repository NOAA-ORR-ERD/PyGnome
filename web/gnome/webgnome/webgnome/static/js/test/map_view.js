define([
    'views',
    'models'
], function(views, models) {

    module('MapView');


    // Test helpers

    var testUrl = "/test";

    var model = new models.Model([], {
        url: testUrl,
        expectedTimeSteps: [],
        currentTimeStep: 0
    });


    function makeMap(el, placeholderEl) {
        return new views.MapView({
            mapEl: el || '#map-empty',
            placeholderEl: placeholderEl || '#placeholder-empty',
            backgroundImageUrl: '/static/js/test/fixtures/background_map.png',
            frameClass: 'frame',
            activeFrameClass: 'active',
            model: model
        });

    }

    function createRect(x1, y1, x2, y2) {
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
        var rect = createRect(100, 200, 200, 200);
        var expected = {start: rect.start, end: rect.end};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });

    test('getRect returns the same rect if user clicked and dragged from top to bottom', function() {
        var map = makeMap();
        var rect = createRect(100, 200, 100, 300);
        var expected = {start: rect.start, end: rect.end};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });

    test('getRect returns the same rect if user dragged from top-left to bottom-right', function() {
        var map = makeMap();
        var rect = createRect(100, 200, 200, 300);
        var expected = {start: rect.start, end: rect.end};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });

    test('getRect flips the rect if the user selected from left to right', function() {
        var map = makeMap();
        var rect = createRect(200, 200, 100, 200);
        var expected = {start: rect.end, end: rect.start};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });

    test('getRect flips the rect if the user selected from bottom to top', function() {
        var map = makeMap();
        var rect = createRect(100, 300, 100, 200);
        var expected = {start: rect.end, end: rect.start};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });

    test('getRect flips the rect if the user selected from bottom-right to top-left', function() {
        var map = makeMap();
        var rect = createRect(200, 300, 100, 200);
        var expected = {start: rect.end, end: rect.start};
        var newRect = map.getRect(rect);

        assertRectIsEqual(newRect, expected);
    });

});