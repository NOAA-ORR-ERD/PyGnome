// JavaScript tests for WebGNOME

// Namespace
gnome = window.noaa.erd.gnome;

test('The test framework works', function() {
    ok(1 == 1);
});


module('MapModel');

function assertRectIsEqual(rect, expected) {
    ok(rect.start.x === expected.start.x);
    ok(rect.start.y === expected.start.y);
    ok(rect.end.x === expected.end.x);
    ok(rect.end.y === expected.end.y);
}

test('getRect returns the same rect if user clicked and dragged from left to right', function() {
    var model = new gnome.MapModel({
        id: 1,
        bbox: [{x: 0, y: 0}, {x: 400, y: 400}]
    });

    var rect = [{x: 100, y: 200}, {x: 200, y: 200}];
    var expected = {start: rect[0], end: rect[1]};
    var newRect = model.getRect(rect[0], rect[1]);
    
    assertRectIsEqual(newRect, expected);
});

test('getRect returns the same rect if user clicked and dragged from top to bottom', function() {
    var model = new gnome.MapModel({
        id: 1,
        bbox: [{x: 0, y: 0}, {x: 400, y: 400}]
    });

    var rect = [{x: 100, y: 200}, {x: 100, y: 300}];
    var expected = {start: rect[0], end: rect[1]};
    var newRect = model.getRect(rect[0], rect[1]);

    assertRectIsEqual(newRect, expected);
});

test('getRect returns the same rect if user dragged from top-left to bottom-right', function() {
    var model = new gnome.MapModel({
        id: 1,
        bbox: [{x: 0, y: 0}, {x: 400, y: 400}]
    });

    var rect = [{x: 100, y: 200}, {x: 200, y: 300}];
    var expected = {start: rect[0], end: rect[1]};
    var newRect = model.getRect(rect[0], rect[1]);

    assertRectIsEqual(newRect, expected);
});

test('getRect flips the rect if the user selected from left to right', function() {
    var model = new gnome.MapModel({
        id: 1,
        bbox: [{x: 0, y: 0}, {x: 400, y: 400}]
    });

    var rect = [{x: 200, y: 200}, {x: 100, y: 200}];
    var expected = {start: rect[1], end: rect[0]};
    var newRect = model.getRect(rect[0], rect[1]);

    assertRectIsEqual(newRect, expected);
});

test('getRect flips the rect if the user selected from bottom to top', function() {
    var model = new gnome.MapModel({
        id: 1,
        bbox: [{x: 0, y: 0}, {x: 400, y: 400}]
    });

    var rect = [{x: 100, y: 300}, {x: 100, y: 200}];
    var expected = {start: rect[1], end: rect[0]};
    var newRect = model.getRect(rect[0], rect[1]);

    assertRectIsEqual(newRect, expected);
});

test('getRect flips the rect if the user selected from bottom-right to top-left', function() {
    var model = new gnome.MapModel({
        id: 1,
        bbox: [{x: 0, y: 0}, {x: 400, y: 400}]
    });

    var rect = [{x: 200, y: 300}, {x: 100, y: 200}];
    var expected = {start: rect[1], end: rect[0]};
    var newRect = model.getRect(rect[0], rect[1]);

    assertRectIsEqual(newRect, expected);
});
