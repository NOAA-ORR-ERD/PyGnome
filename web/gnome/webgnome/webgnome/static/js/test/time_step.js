define([
    'models'
], function(models) {

    module('TimeStep');

    test('get should return timestamp as a UTC string', function() {
        var timeStep = new models.TimeStep({
            "url": "/static/js/test/fixtures/foreground_00000.png",
            "timestamp": "2012-12-07T12:00:00",
            "id": 0
        });

        ok(timeStep.get('timestamp') === 'Fri, 07 Dec 2012 12:00:00 GMT');
    });
});