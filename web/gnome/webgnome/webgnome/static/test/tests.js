// JavaScript tests for WebGNOME

// Namespace
gnome = window.noaa.erd.gnome;


// Test helpers
function assertRectIsEqual(rect, expected) {
    ok(rect.start.x === expected.start.x);
    ok(rect.start.y === expected.start.y);
    ok(rect.end.x === expected.end.x);
    ok(rect.end.y === expected.end.y);
}

function createModel(bbox) {
    if (bbox === undefined) {
        bbox = [{x: 0, y: 0}, {x: 400, y: 400}];
    }

    return new gnome.MapModel({id:1, bbox:bbox});
}

function createRect(x1, y1, x2, y2) {
    return {start:{x: x1, y: y1}, end: {x: x2, y: y2}};
}


// Tests
test('The test framework works', function() {
    ok(1 == 1);
});

module('MapModel');

test('getRect returns the same rect if user clicked and dragged from left to right', function() {
    var model = createModel();
    var rect = createRect(100, 200, 200, 200);
    var expected = {start: rect.start, end: rect.end};
    var newRect = model.getRect(rect);
    
    assertRectIsEqual(newRect, expected);
});

test('getRect returns the same rect if user clicked and dragged from top to bottom', function() {
    var model = createModel();
    var rect = createRect(100, 200, 100, 300);
    var expected = {start: rect.start, end: rect.end};
    var newRect = model.getRect(rect);

    assertRectIsEqual(newRect, expected);
});

test('getRect returns the same rect if user dragged from top-left to bottom-right', function() {
    var model = createModel();
    var rect = createRect(100, 200, 200, 300);
    var expected = {start: rect.start, end: rect.end};
    var newRect = model.getRect(rect);

    assertRectIsEqual(newRect, expected);
});

test('getRect flips the rect if the user selected from left to right', function() {
    var model = createModel();
    var rect = createRect(200, 200, 100, 200);
    var expected = {start: rect.end, end: rect.start};
    var newRect = model.getRect(rect);

    assertRectIsEqual(newRect, expected);
});

test('getRect flips the rect if the user selected from bottom to top', function() {
    var model = createModel();
    var rect = createRect(100, 300, 100, 200);
    var expected = {start: rect.end, end: rect.start};
    var newRect = model.getRect(rect);

    assertRectIsEqual(newRect, expected);
});

test('getRect flips the rect if the user selected from bottom-right to top-left', function() {
    var model = createModel();
    var rect = createRect(200, 300, 100, 200);
    var expected = {start: rect.end, end: rect.start};
    var newRect = model.getRect(rect);

    assertRectIsEqual(newRect, expected);
});

